// Host side of each layout, in the browser: the STFT, masking and iSTFT around an ONNX core. Same math as vf/hosts.py;
// Official runs check each output against the native one. A layout maps a chunk [L, R] to the core's feeds, and the core's outputs to vocals [L, R].
// Stereo runs as one complex FFT per frame (L + iR) and one inverse per frame (two real outputs packed).

function factorize(n) {
  const f = [];
  while (n % 4 === 0) { f.push(4); n /= 4; }
  for (const p of [2, 3, 5, 7]) while (n % p === 0) { f.push(p); n /= p; }
  if (n !== 1) throw new Error("FFT size has a prime factor > 7");
  return f;
}

// Stockham DIF, natural order output. Plan holds the radices and twiddles; fft(re, im) in place (uses two work buffers).
export function makeFFT(n) {
  const radices = factorize(n), wr = new Float64Array(n), wi = new Float64Array(n);
  for (let k = 0; k < n; k++) { wr[k] = Math.cos(-2 * Math.PI * k / n); wi[k] = Math.sin(-2 * Math.PI * k / n); }
  const br = [new Float64Array(n), new Float64Array(n)], bi = [new Float64Array(n), new Float64Array(n)];
  const ar = new Float64Array(8), ai = new Float64Array(8);
  return function fft(re, im, inverse = false) {
    let xr = br[0], xi = bi[0], yr = br[1], yi = bi[1];
    const sg = inverse ? -1 : 1;
    for (let i = 0; i < n; i++) { xr[i] = re[i]; xi[i] = sg * im[i]; }
    let len = n, s = 1;
    for (const p of radices) {
      const m = len / p, tw = n / len;            // omega_len^j = w[j * tw]
      if (p === 4) {                               // radix 4: hand-unrolled (-i for the forward transform)
        for (let q = 0; q < m; q++) {
          const e1 = q * tw, w1r = wr[e1], w1i = wi[e1], w2r = wr[2 * e1], w2i = wi[2 * e1], w3r = wr[3 * e1], w3i = wi[3 * e1];
          for (let t = 0; t < s; t++) {
            const i0 = s * q + t, i1 = i0 + s * m, i2 = i1 + s * m, i3 = i2 + s * m;
            const a0r = xr[i0], a0i = xi[i0], a1r = xr[i1], a1i = xi[i1], a2r = xr[i2], a2i = xi[i2], a3r = xr[i3], a3i = xi[i3];
            const b0r = a0r + a2r, b0i = a0i + a2i, b1r = a0r - a2r, b1i = a0i - a2i;
            const b2r = a1r + a3r, b2i = a1i + a3i, b3r = a1i - a3i, b3i = a3r - a1r;    // (a1 - a3) * -i
            const o = s * 4 * q + t;
            yr[o] = b0r + b2r; yi[o] = b0i + b2i;
            let cr = b1r + b3r, ci = b1i + b3i; yr[o + s] = cr * w1r - ci * w1i; yi[o + s] = cr * w1i + ci * w1r;
            cr = b0r - b2r; ci = b0i - b2i; yr[o + 2 * s] = cr * w2r - ci * w2i; yi[o + 2 * s] = cr * w2i + ci * w2r;
            cr = b1r - b3r; ci = b1i - b3i; yr[o + 3 * s] = cr * w3r - ci * w3i; yi[o + 3 * s] = cr * w3i + ci * w3r;
          }
        }
      } else if (p === 2) {
        for (let q = 0; q < m; q++) {
          const c = wr[q * tw], d = wi[q * tw];
          for (let t = 0; t < s; t++) {
            const i0 = s * q + t, i1 = i0 + s * m, o = s * 2 * q + t;
            const ar0 = xr[i0], ai0 = xi[i0], ar1 = xr[i1], ai1 = xi[i1];
            yr[o] = ar0 + ar1; yi[o] = ai0 + ai1;
            const cr = ar0 - ar1, ci = ai0 - ai1;
            yr[o + s] = cr * c - ci * d; yi[o + s] = cr * d + ci * c;
          }
        }
      } else {
        const ep = new Int32Array(p * p);
        for (let j = 0; j < p; j++) for (let k = 0; k < p; k++) ep[j * p + k] = ((j * k) % p) * (n / p);
        for (let q = 0; q < m; q++) {
          for (let t = 0; t < s; t++) {
            for (let k = 0; k < p; k++) { const idx = s * (q + k * m) + t; ar[k] = xr[idx]; ai[k] = xi[idx]; }
            for (let j = 0; j < p; j++) {
              let sr = 0, si = 0;
              for (let k = 0; k < p; k++) { const e = ep[j * p + k], c = wr[e], d = wi[e]; sr += ar[k] * c - ai[k] * d; si += ar[k] * d + ai[k] * c; }
              const e = j * q * tw, c = wr[e], d = wi[e];
              const o = s * (p * q + j) + t;
              yr[o] = sr * c - si * d; yi[o] = sr * d + si * c;
            }
          }
        }
      }
      [xr, yr] = [yr, xr]; [xi, yi] = [yi, xi];
      len = m; s *= p;
    }
    const sc = inverse ? 1 / n : 1;
    for (let i = 0; i < n; i++) { re[i] = xr[i] * sc; im[i] = sg * xi[i] * sc; }
  };
}

const reflect = (x, i, L) => { while (i < 0 || i >= L) i = i < 0 ? -i : 2 * L - 2 - i; return x[i]; };
export function makeMdx({ nFft = 7680, hop = 1024, dimF = 3072, dimT = 256, compensate = 1.021 } = {}) {
  const fft = makeFFT(nFft), win = new Float64Array(nFft), chunk = hop * (dimT - 1), half = nFft / 2;
  for (let i = 0; i < nFft; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / nFft);
  const re = new Float64Array(nFft), im = new Float64Array(nFft);
  // [2, chunk] -> Float32Array [1, 4, dimF, dimT]
  function spec(L, R) {
    const out = new Float32Array(4 * dimF * dimT), FT = dimF * dimT;
    for (let t = 0; t < dimT; t++) {
      const s0 = t * hop - half;
      for (let i = 0; i < nFft; i++) { re[i] = reflect(L, s0 + i, chunk) * win[i]; im[i] = reflect(R, s0 + i, chunk) * win[i]; }
      fft(re, im);
      for (let f = 0; f < dimF; f++) {        // unpack the two real spectra: XL = (Z[f] + conj Z[-f]) / 2, XR = (Z[f] - conj Z[-f]) / 2i
        const g = f === 0 ? 0 : nFft - f, zr = re[f], zi = im[f], cr = re[g], ci = -im[g];
        const o = f * dimT + t;
        out[o] = 0.5 * (zr + cr); out[FT + o] = 0.5 * (zi + ci);                 // L re, L im
        out[2 * FT + o] = 0.5 * (zi - ci); out[3 * FT + o] = -0.5 * (zr - cr);   // R re, R im
      }
    }
    return out;
  }
  // core output [1, 4, dimF, dimT] -> [L, R] Float32Array(chunk), compensated
  function wave(y) {
    const FT = dimF * dimT, total = nFft + hop * (dimT - 1);
    const oL = new Float64Array(total), oR = new Float64Array(total), env = new Float64Array(total);
    for (let t = 0; t < dimT; t++) {
      re.fill(0); im.fill(0);
      for (let f = 0; f < dimF; f++) {          // Z = XL + i XR, Hermitian-extended
        const o = f * dimT + t, lr = y[o], li = y[FT + o], rr = y[2 * FT + o], ri = y[3 * FT + o];
        if (f === 0) { re[0] = lr; im[0] = rr; continue; }      // irfft ignores the DC imaginary part
        re[f] = lr - ri; im[f] = li + rr;
        re[nFft - f] = lr + ri; im[nFft - f] = -li + rr;
      }
      fft(re, im, true);
      const s0 = t * hop;
      for (let i = 0; i < nFft; i++) { oL[s0 + i] += re[i] * win[i]; oR[s0 + i] += im[i] * win[i]; env[s0 + i] += win[i] * win[i]; }
    }
    const L = new Float32Array(chunk), R = new Float32Array(chunk);
    for (let i = 0; i < chunk; i++) { const k = i + half, e = env[k] > 1e-8 ? env[k] : 1; L[i] = oL[k] / e * compensate; R[i] = oR[k] / e * compensate; }
    return [L, R];
  }

  return { feeds: (L, R, inp) => ({ [inp[0]]: { data: spec(L, R), dims: [1, 4, dimF, dimT] } }), wave: (o, names) => wave(o[names[0]].data), chunk };
}

export function makeMask({ frames = 1100, nFft = 2048, hop = 441 } = {}) {
  const fft = makeFFT(nFft), win = new Float64Array(nFft), chunk = hop * (frames - 1), half = nFft / 2, F = half + 1;
  for (let i = 0; i < nFft; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / nFft);
  const re = new Float64Array(nFft), im = new Float64Array(nFft);
  let Z = null;      // [2 ch][F * frames] complex STFT of the current chunk, kept for the mask product
  function spec(L, R) {
    Z = [[new Float32Array(F * frames), new Float32Array(F * frames)], [new Float32Array(F * frames), new Float32Array(F * frames)]];
    const feed = new Float32Array(F * 2 * frames * 2);
    for (let t = 0; t < frames; t++) {
      const s0 = t * hop - half;
      for (let i = 0; i < nFft; i++) { re[i] = reflect(L, s0 + i, chunk) * win[i]; im[i] = reflect(R, s0 + i, chunk) * win[i]; }
      fft(re, im);
      for (let f = 0; f < F; f++) {
        const g = f === 0 ? 0 : nFft - f, zr = re[f], zi = im[f], cr = re[g], ci = -im[g];
        const v = [[0.5 * (zr + cr), 0.5 * (zi + ci)], [0.5 * (zi - ci), -0.5 * (zr - cr)]];
        for (let ch = 0; ch < 2; ch++) {
          Z[ch][0][f * frames + t] = v[ch][0]; Z[ch][1][f * frames + t] = v[ch][1];
          const o = (((f * 2 + ch) * frames) + t) * 2;
          feed[o] = v[ch][0]; feed[o + 1] = v[ch][1];
        }
      }
    }
    return feed;
  }
  function wave(m) {
    const total = nFft + hop * (frames - 1), oL = new Float64Array(total), oR = new Float64Array(total), env = new Float64Array(total);
    for (let t = 0; t < frames; t++) {
      re.fill(0); im.fill(0);
      for (let f = 0; f < F; f++) {
        const y = [0, 1].map((ch) => {
          const o = (((f * 2 + ch) * frames) + t) * 2, mr = m[o], mi = m[o + 1], zr = Z[ch][0][f * frames + t], zi = Z[ch][1][f * frames + t];
          return [zr * mr - zi * mi, zr * mi + zi * mr];
        });
        const [[lr, li], [rr, ri]] = y;
        if (f === 0 || f === half) { re[f] = lr; im[f] = rr; continue; }     // irfft ignores DC / Nyquist imaginary parts
        re[f] = lr - ri; im[f] = li + rr;
        re[nFft - f] = lr + ri; im[nFft - f] = -li + rr;
      }
      fft(re, im, true);
      const s0 = t * hop;
      for (let i = 0; i < nFft; i++) { oL[s0 + i] += re[i] * win[i]; oR[s0 + i] += im[i] * win[i]; env[s0 + i] += win[i] * win[i]; }
    }
    const L = new Float32Array(chunk), R = new Float32Array(chunk);
    for (let i = 0; i < chunk; i++) { const k = i + half, e = env[k] > 1e-8 ? env[k] : 1; L[i] = oL[k] / e; R[i] = oR[k] / e; }
    return [L, R];
  }

  return { feeds: (L, R, inp) => ({ [inp[0]]: { data: spec(L, R), dims: [1, F * 2, frames, 2] } }), wave: (o, names) => wave(o[names[0]].data), chunk };
}

// Hybrid Demucs: waveform "input" and CaC spectrogram "x" in; "output" (frequency branch) and "add_67" (time branch) out.
// Vocals = time branch + iSTFT of the frequency branch. The song is normalized as a whole by song(): see vf/hosts.py HtDemucs.
export function makeHtDemucs({ nFft = 4096, hop = 1024, chunk = 343980, stem = 3, freqOutput = "output", timeOutput = "add_67" } = {}) {
  const fft = makeFFT(nFft), half = nFft / 2, F = half, T = Math.ceil(chunk / hop), pad = (hop / 2) * 3, scale = Math.sqrt(nFft);
  const win = new Float64Array(nFft);
  for (let i = 0; i < nFft; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / nFft);
  const re = new Float64Array(nFft), im = new Float64Array(nFft), xpLen = T * hop + 2 * pad;
  const xpL = new Float32Array(xpLen), xpR = new Float32Array(xpLen);
  let mean = 0, std = 1;
  function song(L, R) {
    const n = L.length;
    let m = 0, s = 0;
    for (let i = 0; i < n; i++) m += 0.5 * (L[i] + R[i]);
    m /= n;
    for (let i = 0; i < n; i++) { const d = 0.5 * (L[i] + R[i]) - m; s += d * d; }
    s = n > 1 ? Math.sqrt(s / (n - 1)) : 0;
    if (!Number.isFinite(s) || s < 1e-8) { mean = 0; std = 1; } else { mean = m; std = s; }
  }
  function feeds(L, R) {
    const wave = new Float32Array(2 * chunk);
    for (let i = 0; i < chunk; i++) { wave[i] = (L[i] - mean) / std; wave[chunk + i] = (R[i] - mean) / std; }
    const nL = wave.subarray(0, chunk), nR = wave.subarray(chunk);
    for (let i = 0; i < xpLen; i++) { xpL[i] = reflect(nL, i - pad, chunk); xpR[i] = reflect(nR, i - pad, chunk); }
    const FT = F * T, x = new Float32Array(4 * FT);
    for (let t = 0; t < T; t++) {
      const s0 = (t + 2) * hop - half;                    // frames 2..T+1 of the centered STFT: they never reach the center padding
      for (let i = 0; i < nFft; i++) { re[i] = xpL[s0 + i] * win[i]; im[i] = xpR[s0 + i] * win[i]; }
      fft(re, im);
      for (let f = 0; f < F; f++) {
        const g = f === 0 ? 0 : nFft - f, zr = re[f], zi = im[f], cr = re[g], ci = -im[g], o = f * T + t;
        x[o] = 0.5 * (zr + cr) / scale; x[FT + o] = 0.5 * (zi + ci) / scale;
        x[2 * FT + o] = 0.5 * (zi - ci) / scale; x[3 * FT + o] = -0.5 * (zr - cr) / scale;
      }
    }
    return { input: { data: wave, dims: [1, 2, chunk] }, x: { data: x, dims: [1, 4, F, T] } };
  }
  function wave(o) {
    const y = o[freqOutput].data, w = o[timeOutput].data, FT = F * T, base = stem * 4 * FT;
    const frames = T + 4, total = nFft + hop * (frames - 1), oL = new Float64Array(total), oR = new Float64Array(total), env = new Float64Array(total);
    for (let t = 0; t < frames; t++) {
      const s0 = t * hop;
      for (let i = 0; i < nFft; i++) env[s0 + i] += win[i] * win[i];   // zero frames count in the window sum too
      if (t < 2 || t >= T + 2) continue;
      re.fill(0); im.fill(0);
      for (let f = 0; f < F; f++) {
        const k = base + f * T + (t - 2), lr = y[k] * scale, li = y[FT + k] * scale, rr = y[2 * FT + k] * scale, ri = y[3 * FT + k] * scale;
        if (f === 0) { re[0] = lr; im[0] = rr; continue; }
        re[f] = lr - ri; im[f] = li + rr;
        re[nFft - f] = lr + ri; im[nFft - f] = -li + rr;
      }
      fft(re, im, true);
      for (let i = 0; i < nFft; i++) { oL[s0 + i] += re[i] * win[i]; oR[s0 + i] += im[i] * win[i]; }
    }
    const L = new Float32Array(chunk), R = new Float32Array(chunk), tb = stem * 2 * chunk;
    for (let i = 0; i < chunk; i++) {
      const k = half + pad + i, e = env[k] > 1e-8 ? env[k] : 1;
      L[i] = (w[tb + i] + oL[k] / e) * std + mean;
      R[i] = (w[tb + chunk + i] + oR[k] / e) * std + mean;
    }
    return [L, R];
  }
  return { feeds, wave, song, chunk };
}

export function makeLayout(l) {
  if (l.type === "mdx") return makeMdx({ nFft: l.n_fft, hop: l.hop, dimF: l.dim_f, dimT: l.dim_t, compensate: l.compensate });
  if (l.type === "mask") return makeMask({ frames: l.frames, nFft: l.n_fft, hop: l.hop });
  if (l.type === "htdemucs") return makeHtDemucs({ chunk: l.chunk, stem: l.stem, freqOutput: l.freq_output, timeOutput: l.time_output });
  throw new Error("unknown layout " + l.type);
}
