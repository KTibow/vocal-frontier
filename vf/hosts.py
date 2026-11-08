"""Host side of each layout: the STFT, masking and iSTFT around an ONNX core. A core maps one chunk of 44.1 kHz stereo
[2, chunk] to its vocals [2, chunk]. site/src/layouts.js is the same math for the browser; `vf transfer` checks they agree on real clips.

A new architecture usually fits one of these. If it doesn't, add a layout here and in layouts.js in your PR."""
import numpy as np

from .audio import istft, stft


def _session(path, threads, runtime, arena=True):
    if runtime == "openvino":
        return _OvSession(path, threads)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = arena      # big graphs: with the arena, HTDemucs RSS passes 3.8 GB by the second chunk and MelBand Kim reaches 6.9 GB
    return _OrtSession(ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]))


class _OrtSession:
    def __init__(self, s):
        self.s = s
        self.inputs = {i.name: i.shape for i in s.get_inputs()}
        self.outputs = [o.name for o in s.get_outputs()]

    def run(self, feeds):
        return dict(zip(self.outputs, self.s.run(None, feeds)))


class _OvSession:
    def __init__(self, path, threads):
        import openvino as ov
        core = ov.Core()
        m = core.read_model(path)
        if len(m.inputs) == 1 and m.inputs[0].get_partial_shape().is_dynamic:     # dynamic batch: static compiles faster
            m.reshape({m.inputs[0].get_any_name(): [1] + [d.get_length() for d in list(m.inputs[0].get_partial_shape())[1:]]})
        self.cm = core.compile_model(m, "CPU", {"INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY",
                                                "INFERENCE_PRECISION_HINT": "f32", "ENABLE_CPU_PINNING": False})
        self.req = self.cm.create_infer_request()
        self.inputs = {i.get_any_name(): [d.get_length() for d in i.get_partial_shape()] for i in m.inputs}
        self.outputs = [o.get_any_name() for o in m.outputs]

    def run(self, feeds):
        r = self.req.infer(feeds)
        return {n: r[self.cm.output(n)] for n in self.outputs}


class Mdx:
    """TFC-TDF family (UVR MDX-Net): [1, 4 (L re, L im, R re, R im), dim_f, dim_t] in, the vocal spectrum out."""

    def __init__(self, sess, n_fft=7680, hop=1024, dim_f=3072, dim_t=256, compensate=1.021):
        self.s, self.n_fft, self.hop, self.dim_f, self.dim_t, self.comp = sess, n_fft, hop, dim_f, dim_t, compensate
        self.chunk = hop * (dim_t - 1)
        self.inp = next(iter(sess.inputs))

    def song(self, x):
        pass

    def __call__(self, x):
        z = stft(x, self.n_fft, self.hop)[:, :self.dim_f]
        spec = np.stack([z.real, z.imag], 1).reshape(1, 4, self.dim_f, -1).astype(np.float32)
        y = next(iter(self.s.run({self.inp: spec}).values())).reshape(2, 2, self.dim_f, -1)
        full = np.zeros((2, self.n_fft // 2 + 1, y.shape[-1]), np.complex64)
        full[:, :self.dim_f] = y[:, 0] + 1j * y[:, 1]
        return istft(full, self.n_fft, self.hop, x.shape[-1]) * self.comp


class HtDemucs:
    """Hybrid Demucs: waveform "input" [1, 2, chunk] and CaC spectrogram "x" [1, 4, 2048, frames] in; "output"
    [1, stems, 4, 2048, frames] and "add_67" [1, stems, 2, chunk] out. Vocals = time branch + iSTFT of the frequency
    branch. The whole song is normalized by its mono mean and std first."""
    n_fft, hop = 4096, 1024

    def __init__(self, sess, stem=3, freq_output="output", time_output="add_67"):
        self.s, self.stem, self.fo, self.to = sess, stem, freq_output, time_output
        self.chunk = sess.inputs["input"][-1]
        self.frames = -(-self.chunk // self.hop)
        self.pad = self.hop // 2 * 3
        self.stats = (0.0, 1.0)

    def song(self, x):
        mono = x.astype(np.float64).mean(0)
        std = mono.std(ddof=1) if mono.size > 1 else 0.0
        self.stats = (0.0, 1.0) if not np.isfinite(std) or std < 1e-8 else (float(mono.mean()), float(std))

    def __call__(self, x):
        mean, std = self.stats
        xn = ((x - mean) / std).astype(np.float32)
        p, F, T = self.pad, self.n_fft // 2, self.frames
        xp = np.pad(xn, ((0, 0), (p, p + T * self.hop - x.shape[-1])), mode="reflect")
        z = stft(xp, self.n_fft, self.hop)[:, :F, 2:2 + T] / np.sqrt(self.n_fft)
        spec = np.stack([z.real, z.imag], 1).reshape(1, 4, F, T).astype(np.float32)
        o = self.s.run({"input": xn[None], "x": spec})
        y = o[self.fo][0, self.stem].reshape(2, 2, F, T)
        full = np.zeros((2, F + 1, T + 4), np.complex64)
        full[:, :F, 2:2 + T] = y[:, 0] + 1j * y[:, 1]
        fb = istft(full * np.sqrt(self.n_fft), self.n_fft, self.hop, T * self.hop + 2 * p)[:, p:p + x.shape[-1]]
        return (o[self.to][0, self.stem] + fb) * std + mean


class Mask:
    """RoFormer and other maskers: real STFT [1, (F ch), frames, 2] in, complex masks of the same shape out."""

    def __init__(self, sess, n_fft=2048, hop=441, frames=None):
        self.s, self.n_fft, self.hop = sess, n_fft, hop
        self.inp = next(iter(sess.inputs))
        self.frames = frames or sess.inputs[self.inp][2]
        self.chunk = hop * (self.frames - 1)

    def song(self, x):
        pass

    def __call__(self, x):
        z = stft(x, self.n_fft, self.hop)
        rep = np.stack([z.real, z.imag], -1).transpose(1, 0, 2, 3).reshape(1, -1, z.shape[-1], 2)
        m = next(iter(self.s.run({self.inp: np.ascontiguousarray(rep, np.float32)}).values()))
        m = m.reshape(-1, 2, z.shape[-1], 2).transpose(1, 0, 2, 3)
        return istft(z * (m[..., 0] + 1j * m[..., 1]), self.n_fft, self.hop, x.shape[-1])


LAYOUTS = {"mdx": Mdx, "htdemucs": HtDemucs, "mask": Mask}


def make_core(path, layout, threads, runtime):
    kw = {k: v for k, v in layout.items() if k != "type"}
    return LAYOUTS[layout["type"]](_session(path, threads, runtime, arena=layout["type"] == "mdx"), **kw)
