"""The transfer rule's limits, and that the board won't take a borrowed score without a recorded pass."""
import json
import os

import pytest

from vf import cli, transfer
from vf import manifest as M

P = {f"{s}/{c}": 10.0 for s, cl in transfer.CLIPS.items() for c in cl}
PUBLIC = {k: v for k, v in P.items() if not k.startswith("private30")}


def test_limits():
    assert transfer.judge(P, {k: v + .01 for k, v in P.items()})[0]["verdict"] == "pass"
    one = {**P, "cc30/c40": 10.051}
    assert transfer.judge(P, one)[0]["verdict"] == "fail"                 # one clip past 0.05
    assert transfer.judge(P, {k: v + .03 for k, v in P.items()})[0]["verdict"] == "fail"      # the mean past 0.02
    assert transfer.judge(PUBLIC, dict(PUBLIC))[0]["verdict"] == "public-pass"                # no private clips: not a pass


MANIFEST = ("title: {n}\nkind: tuned\nlayout: {{type: mask, frames: 801}}\noverlap: 1\nfiles: {{fp32: {{path: x/core.onnx}}, fp16: {{path: x_f16/core.onnx}}}}\n"
            "native: {{file: fp32, runtime: onnxruntime}}\n")
GPU = "webgpu: {file: fp16, runtime: webgpu@1.23.0}\n"         # only the child runs a second file


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", str(tmp_path))
    monkeypatch.setattr(M, "MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(M, "CACHE", str(tmp_path / "cache.json"))
    for f, b in (("x/core.onnx", b"fp32"), ("x_f16/core.onnx", b"fp16")):
        (tmp_path / "models" / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "models" / f).write_bytes(b)
    for n in (M.REFERENCE, "parent", "child"):
        d = tmp_path / "baselines" / n
        d.mkdir(parents=True)
        (d / "manifest.yaml").write_text(MANIFEST.format(n=n) + (GPU if n == "child" else ""))
    q = tmp_path / "results" / "quality"
    q.mkdir(parents=True)
    sets = {s: {"mean": 12.0, "clips": 50} for s in cli.SETS}
    for n in (M.REFERENCE, "parent"):
        (q / f"{n}.json").write_text(json.dumps({"sets": sets}))
    (q / "child.json").write_text(json.dumps({"sets": sets, "inherits": "parent"}))
    dev = tmp_path / "results" / "devices" / "native"
    dev.mkdir(parents=True)
    (dev / "a.json").write_text(json.dumps({"date": "d", "models": {n: {"xrt": 1.0} for n in (M.REFERENCE, "parent", "child")}}))
    return tmp_path / "results" / "transfer"


def test_board_refuses_a_borrowed_score_without_a_pass(tmp_path, monkeypatch):
    t = _setup(tmp_path, monkeypatch)
    assert "child" not in [p["name"] for p in cli.board_data()["points"]]
    t.mkdir()
    files = transfer.identity(M.load("child"), "webgpu", M.load("parent"))
    rec = {"variant": "child", "parent": "parent", "verdict": "public-pass", "files": files}
    (t / "child.webgpu.json").write_text(json.dumps(rec))
    assert "child" not in [p["name"] for p in cli.board_data()["points"]]
    (t / "child.webgpu.json").write_text(json.dumps({**rec, "verdict": "pass", "files": None}))
    assert "child" not in [p["name"] for p in cli.board_data()["points"]]        # a pass that doesn't say what it checked
    (t / "child.webgpu.json").write_text(json.dumps({**rec, "verdict": "pass"}))
    row = next(p for p in cli.board_data()["points"] if p["name"] == "child")
    assert row["inherits"] == "parent"


def test_a_pass_stops_counting_when_a_file_changes(tmp_path, monkeypatch):
    t = _setup(tmp_path, monkeypatch)
    t.mkdir()
    rec = {"variant": "child", "parent": "parent", "verdict": "pass", "files": transfer.identity(M.load("child"), "webgpu", M.load("parent"))}
    (t / "child.webgpu.json").write_text(json.dumps(rec))
    assert "child" in [p["name"] for p in cli.board_data()["points"]]
    os.utime(tmp_path / "models" / "x_f16" / "core.onnx", (1, 1))
    (tmp_path / "models" / "x_f16" / "core.onnx").write_bytes(b"something faster")
    assert "child" not in [p["name"] for p in cli.board_data()["points"]]


def test_inheriting_needs_the_parents_own_native_file(tmp_path, monkeypatch):
    t = _setup(tmp_path, monkeypatch)
    t.mkdir()
    d = tmp_path / "baselines" / "child" / "manifest.yaml"
    d.write_text(d.read_text().replace("x/core.onnx", "y/core.onnx"))
    (tmp_path / "models" / "y").mkdir()
    (tmp_path / "models" / "y" / "core.onnx").write_bytes(b"different weights")
    rec = {"variant": "child", "parent": "parent", "verdict": "pass", "files": transfer.identity(M.load("child"), "webgpu", M.load("parent"))}
    (t / "child.webgpu.json").write_text(json.dumps(rec))
    assert "child" not in [p["name"] for p in cli.board_data()["points"]]


def test_a_failed_transfer_counts_once_scored_in_full_and_the_lower_score_wins(tmp_path, monkeypatch):
    t = _setup(tmp_path, monkeypatch)
    t.mkdir()
    rec = {"variant": "child", "parent": "parent", "verdict": "fail"}
    (t / "child.webgpu.json").write_text(json.dumps(rec))
    assert "child" not in [p["name"] for p in cli.board_data()["points"]]
    native = next(p for p in cli.board_data()["points"] if p["name"] == "parent")["q"]
    full = {"sets": {s: {"mean": native - 0.5, "clips": 50} for s in cli.SETS}, "backend": "webgpu"}
    (t.parent / "quality" / "child.webgpu.json").write_text(json.dumps(full))
    row = next(p for p in cli.board_data()["points"] if p["name"] == "child")
    assert row["q"] == pytest.approx(native - 0.5)
    full["sets"] = {s: {"mean": native + 0.5, "clips": 50} for s in cli.SETS}
    (t.parent / "quality" / "child.webgpu.json").write_text(json.dumps(full))
    row = next(p for p in cli.board_data()["points"] if p["name"] == "child")
    assert row["q"] == pytest.approx(native)


def test_a_different_file_needs_a_check_even_with_its_own_native_score(tmp_path, monkeypatch):
    m = {"native": {"file": "fp32"}, "webgpu": {"file": "fp16"}, "browser": {"file": "fp32"}}
    monkeypatch.setattr(cli.M, "ROOT", str(tmp_path))
    m["files"] = {"fp32": {"path": "x/core.onnx", "sha256": "a" * 64}, "fp16": {"path": "x_f16/core.onnx", "sha256": "b" * 64}}
    m["dir"] = str(tmp_path)
    monkeypatch.setattr(cli.M, "load", lambda n: m)
    monkeypatch.setattr(cli.M, "names", lambda: ["a", "b"])
    (tmp_path / "results" / "transfer").mkdir(parents=True)
    (tmp_path / "results" / "quality").mkdir()
    assert cli._unchecked("a", "a") == ["webgpu"]       # the browser runs the native file, so only the GPU file is in question
    rec = tmp_path / "results" / "transfer" / "a.webgpu.json"
    rec.write_text(json.dumps({"verdict": "fail", "parent": "a"}))
    assert cli._unchecked("a", "a") == ["webgpu"]
    rec.write_text(json.dumps({"verdict": "pass", "parent": "a"}))
    assert cli._unchecked("a", "a") == ["webgpu"]       # says nothing about which files it checked
    rec.write_text(json.dumps({"verdict": "pass", "parent": "a", "files": transfer.identity(m, "webgpu", m)}))
    assert cli._unchecked("a", "a") == []
    rec.unlink()
    assert cli._unchecked("a", "a") == ["webgpu"]
    q = tmp_path / "results" / "quality"
    (q / "b.json").write_text(json.dumps({"sets": {"cc30": {"mean": 15.0}}}))
    (q / "b.webgpu.json").write_text(json.dumps({"sets": {"cc30": {"mean": 14.5}}}))     # another entry scored the same file in full, and it came out low
    assert cli._unchecked("a", "a") == ["webgpu"]
    (q / "b.webgpu.json").write_text(json.dumps({"sets": {"cc30": {"mean": 15.0}}}))     # and it held up
    assert cli._unchecked("a", "a") == []


def test_one_failed_session_after_a_success_waits_for_a_second(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", str(tmp_path))
    d = tmp_path / "results" / "devices" / "vega"
    d.mkdir(parents=True)
    run = lambda date, xrt: json.dumps({"date": date, "models": {M.REFERENCE: {"xrt": 1.0}, "m": {"xrt": xrt}}})
    (d / "1.json").write_text(run("1", 2.0))
    (d / "2.json").write_text(run("2", None))
    r = cli._latest_runs()["vega"]["m"]
    assert r["xrt"] == 2.0 and r["failed_since"] == "2"
    (d / "3.json").write_text(run("3", None))
    assert cli._latest_runs()["vega"]["m"]["xrt"] is None
    (d / "4.json").write_text(run("4", 3.0))
    assert cli._latest_runs()["vega"]["m"]["xrt"] == 3.0


def test_a_maintainer_entry_added_after_the_start_queues(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", str(tmp_path))
    (tmp_path / "results").mkdir()
    m = {"name": "late", "group": "baselines"}
    assert M.submitted_at(m) == 0                                    # before the start, every baseline is the starting board
    (tmp_path / "results" / "starting.json").write_text(json.dumps(["htdemucs"]))
    assert M.submitted_at(m) == M.UNQUEUED
    (tmp_path / "results" / "queue.json").write_text(json.dumps({"late": 1700000000}))
    assert M.submitted_at(m) == 1700000000


def test_a_submission_keeps_its_files_in_its_own_folder(tmp_path, monkeypatch, capsys):
    from vf import validate
    monkeypatch.setattr(M, "MODELS", str(tmp_path))
    m = {"name": "mine", "group": "submissions", "dir": str(tmp_path), "kind": "contestant", "layout": {"type": "mdx"}, "overlap": 1,
         "files": {"fp32": {"path": "voc-ft-p55/core.onnx", "url": "https://x", "sha256": "a" * 64}}}
    validate.validate(m)
    assert "has to start with mine/" in capsys.readouterr().out


def test_a_file_rejoined_from_the_split_copy_matches_the_original(tmp_path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import helper, numpy_helper
    monkeypatch.setattr(M, "CACHE", str(tmp_path / "cache.json"))
    w = numpy_helper.from_array(np.arange(4096, dtype=np.float32).reshape(64, 64), "w")
    g = helper.make_graph([helper.make_node("MatMul", ["x", "w"], ["y"])], "g",
                          [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 64])],
                          [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 64])], [w])
    model = helper.make_model(g)
    for d in ("one", "split", "rejoined", "other"):
        (tmp_path / d).mkdir()
    onnx.save(model, str(tmp_path / "one" / "core.onnx"))
    onnx.save(model, str(tmp_path / "split" / "core.onnx"), save_as_external_data=True, location="core.onnx.data", size_threshold=0)
    onnx.save(onnx.load(str(tmp_path / "split" / "core.onnx")), str(tmp_path / "rejoined" / "core.onnx"))
    w2 = numpy_helper.from_array(np.arange(4096, dtype=np.float32).reshape(64, 64) + 1, "w")
    del g.initializer[:]
    g.initializer.append(w2)
    onnx.save(helper.make_model(g), str(tmp_path / "other" / "core.onnx"))
    ids = {d: M.model_id(str(tmp_path / d / "core.onnx")) for d in ("one", "split", "rejoined", "other")}
    assert ids["one"] == ids["split"] == ids["rejoined"]
    assert ids["other"] != ids["one"]
