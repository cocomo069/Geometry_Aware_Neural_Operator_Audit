"""Kaggle cycle + lean-export tests (PLAN_PHASE3 sections 1-2). CPU, no network.

Covers the autonomous loop's state machine (advances only on COMPLETE, is a
no-op while RUNNING, the queue advances), the driver's per-run export selection
(``ckpt_<run_id>.zip`` names and the best/last file choice), the puller's
``needed`` policy, and sweep.py's multi-spec ordering. Every kaggle/git call is
monkeypatched, so nothing here touches the network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "kaggle")):
    if p not in sys.path:
        sys.path.insert(0, p)

import cycle  # noqa: E402
import pull_results  # noqa: E402
import session_driver  # noqa: E402
from scripts import sweep as sweep_mod  # noqa: E402


# =========================================================================== #
# 1. driver lean export: select_export_set
# =========================================================================== #
def _make_tree(tmp_path, runs):
    """runs = {rid: {"metrics": bool, "pts": [names]}} -> build results/ + checkpoints/."""
    results = tmp_path / "results"
    ckpt = tmp_path / "checkpoints"
    for rid, spec in runs.items():
        (ckpt / rid).mkdir(parents=True, exist_ok=True)
        for pt in spec["pts"]:
            (ckpt / rid / pt).write_bytes(b"x")
        if spec["metrics"]:
            (results / rid).mkdir(parents=True, exist_ok=True)
            (results / rid / "metrics.json").write_text("{}")
    return results, ckpt


def test_select_export_set_names_and_files(tmp_path):
    results, ckpt = _make_tree(tmp_path, {
        "gnn_full_s2": {"metrics": True, "pts": ["best.pt", "last.pt"]},      # finished
        "gnn_aoa_s1": {"metrics": False, "pts": ["last.pt", "best.pt"]},      # unfinished
        "gnn_reynolds_s1": {"metrics": False, "pts": ["last.pt"]},            # unfinished, last only
        "sdf_fno_full_s0": {"metrics": True, "pts": ["last.pt"]},             # finished, no best
        "transolver_full_s0": {"metrics": True, "pts": ["best.pt"]},         # pre-done (restored)
    })
    pre_done = {"transolver_full_s0"}
    sel = session_driver.select_export_set(results, ckpt, pre_done)

    # pre-done run is excluded entirely -- this is what stops cumulative growth
    assert "transolver_full_s0" not in sel
    # finished run with both -> best.pt only (last.pt is dead weight once done)
    assert sel["gnn_full_s2"] == {"done": True, "files": ["best.pt"]}
    # unfinished -> every .pt, last.pt first (it is what resumes next session)
    assert sel["gnn_aoa_s1"] == {"done": False, "files": ["last.pt", "best.pt"]}
    assert sel["gnn_reynolds_s1"] == {"done": False, "files": ["last.pt"]}
    # finished but best missing -> fall back to last.pt
    assert sel["sdf_fno_full_s0"] == {"done": True, "files": ["last.pt"]}

    # the export filenames the driver writes are ckpt_<rid>.zip
    names = {f"ckpt_{rid}.zip" for rid in sel}
    assert names == {"ckpt_gnn_full_s2.zip", "ckpt_gnn_aoa_s1.zip",
                     "ckpt_gnn_reynolds_s1.zip", "ckpt_sdf_fno_full_s0.zip"}


def test_select_export_set_empty_when_no_checkpoints(tmp_path):
    assert session_driver.select_export_set(tmp_path / "r", tmp_path / "c", set()) == {}


# =========================================================================== #
# 2. puller: needed_checkpoints policy
# =========================================================================== #
def test_needed_checkpoints_policy(tmp_path):
    manifest = {"runs": {
        "gnn_full_s2": {"done": True, "files": ["best.pt"]},          # core/ens -> needed
        "gnn_full_n100_s0": {"done": True, "files": ["best.pt"]},     # data-eff finished -> skip
        "sdf_fno_full_s0_cond_mask": {"done": True, "files": ["best.pt"]},  # tagged finished -> skip
        "gnn_reynolds_s1": {"done": False, "files": ["last.pt"]},     # unfinished -> needed
    }}
    needed = set(pull_results.needed_checkpoints(manifest, tmp_path, "needed"))
    assert needed == {"gnn_full_s2", "gnn_reynolds_s1"}

    # policy=all takes everything, policy=none nothing
    assert set(pull_results.needed_checkpoints(manifest, tmp_path, "all")) == set(manifest["runs"])
    assert pull_results.needed_checkpoints(manifest, tmp_path, "none") == []


def test_needed_checkpoints_skips_already_local(tmp_path):
    manifest = {"runs": {"gnn_full_s2": {"done": True, "files": ["best.pt"]}}}
    (tmp_path / "checkpoints" / "gnn_full_s2").mkdir(parents=True)
    (tmp_path / "checkpoints" / "gnn_full_s2" / "best.pt").write_bytes(b"x")
    assert pull_results.needed_checkpoints(manifest, tmp_path, "needed") == []


# =========================================================================== #
# 3. cycle state machine (pure)
# =========================================================================== #
def test_classify_status_buckets():
    assert cycle.classify_status(0, 'x has status "running"') == cycle.RUNNING
    assert cycle.classify_status(0, 'x has status "queued"') == cycle.RUNNING
    assert cycle.classify_status(0, 'x has status "complete"') == cycle.DONE
    assert cycle.classify_status(0, 'x has status "error"') == cycle.DONE
    assert cycle.classify_status(1, "404 not found") == cycle.ABSENT


def test_next_action_advances_only_on_complete():
    assert cycle.next_action(cycle.RUNNING, None) == "wait"
    assert cycle.next_action(cycle.ABSENT, None) == "launch"
    assert cycle.next_action(cycle.DONE, True) == "advance"
    assert cycle.next_action(cycle.DONE, False) == "relaunch"


def test_current_index_finds_first_not_done():
    q = {"queue": [{"slug": "a", "done": True}, {"slug": "b", "done": False},
                   {"slug": "c", "done": False}]}
    assert cycle.current_index(q) == 1
    assert cycle.current_index({"queue": [{"slug": "a", "done": True}]}) is None


# =========================================================================== #
# 4. cycle.step end to end (kaggle/git monkeypatched)
# =========================================================================== #
def _queue_file(tmp_path):
    q = {"owner": "tester", "runs_dataset": "geo-op-runs", "queue": [
        {"slug": "geo-op-core", "specs": ["configs/sweeps/kaggle_core.yaml"], "done": True},
        {"slug": "geo-op-dataeff", "specs": ["configs/sweeps/kaggle_dataeff.yaml"], "done": False},
        {"slug": "geo-op-ablations", "specs": ["configs/sweeps/kaggle_ablations.yaml"], "done": False},
    ]}
    path = tmp_path / "q.json"
    path.write_text(json.dumps(q))
    return path


def _boom(*a, **k):
    raise AssertionError("must not be called in this branch")


def test_step_is_noop_while_running(tmp_path, monkeypatch):
    path = _queue_file(tmp_path)
    monkeypatch.setattr(cycle, "kernel_status", lambda slug: cycle.RUNNING)
    monkeypatch.setattr(cycle, "launch_item", _boom)
    monkeypatch.setattr(cycle, "pull_item", _boom)
    rc = cycle.step(queue_path=path)
    assert rc == 0
    # queue untouched: the current item is still not done
    q = json.loads(path.read_text())
    assert q["queue"][1]["done"] is False


def test_step_launches_when_absent(tmp_path, monkeypatch):
    path = _queue_file(tmp_path)
    launched = []
    monkeypatch.setattr(cycle, "kernel_status", lambda slug: cycle.ABSENT)
    monkeypatch.setattr(cycle, "launch_item", lambda item, owner, rd: launched.append(item["slug"]))
    monkeypatch.setattr(cycle, "pull_item", _boom)
    rc = cycle.step(queue_path=path)
    assert rc == 0
    assert launched == ["geo-op-dataeff"]
    assert json.loads(path.read_text())["queue"][1]["done"] is False  # not advanced


def test_step_advances_on_complete_when_sweep_done(tmp_path, monkeypatch):
    path = _queue_file(tmp_path)
    pulled, launched = [], []
    monkeypatch.setattr(cycle, "kernel_status", lambda slug: cycle.DONE)
    monkeypatch.setattr(cycle, "pull_item", lambda item, owner, rd: pulled.append(item["slug"]))
    monkeypatch.setattr(cycle, "sweep_is_done", lambda specs: True)
    monkeypatch.setattr(cycle, "launch_item", lambda *a, **k: launched.append(a))
    rc = cycle.step(queue_path=path)
    assert rc == 0
    assert pulled == ["geo-op-dataeff"]          # pulled before deciding
    assert launched == []                         # advance does not relaunch
    q = json.loads(path.read_text())
    assert q["queue"][1]["done"] is True          # advanced
    assert cycle.current_index(q) == 2            # now points at ablations


def test_step_relaunches_on_complete_when_truncated(tmp_path, monkeypatch):
    path = _queue_file(tmp_path)
    pulled, launched = [], []
    monkeypatch.setattr(cycle, "kernel_status", lambda slug: cycle.DONE)
    monkeypatch.setattr(cycle, "pull_item", lambda item, owner, rd: pulled.append(item["slug"]))
    monkeypatch.setattr(cycle, "sweep_is_done", lambda specs: False)
    monkeypatch.setattr(cycle, "launch_item", lambda item, owner, rd: launched.append(item["slug"]))
    rc = cycle.step(queue_path=path)
    assert rc == 0
    assert pulled == ["geo-op-dataeff"]
    assert launched == ["geo-op-dataeff"]         # same item relaunched
    assert json.loads(path.read_text())["queue"][1]["done"] is False


def test_step_reports_drained_queue(tmp_path, monkeypatch, capsys):
    q = {"owner": "tester", "queue": [{"slug": "a", "specs": [], "done": True}]}
    path = tmp_path / "q.json"
    path.write_text(json.dumps(q))
    monkeypatch.setattr(cycle, "kernel_status", _boom)
    rc = cycle.step(queue_path=path)
    assert rc == 0
    assert "drained" in capsys.readouterr().out


def test_step_dry_run_makes_no_calls(tmp_path, monkeypatch, capsys):
    path = _queue_file(tmp_path)
    monkeypatch.setattr(cycle, "kernel_status", _boom)
    monkeypatch.setattr(cycle, "launch_item", _boom)
    monkeypatch.setattr(cycle, "pull_item", _boom)
    rc = cycle.step(queue_path=path, dry_run=True)
    assert rc == 0
    assert "geo-op-dataeff" in capsys.readouterr().out


# =========================================================================== #
# 5. sweep multi-spec ordering
# =========================================================================== #
def test_sweep_multi_spec_preserves_order(tmp_path, capsys):
    s1 = tmp_path / "s1.yaml"
    s1.write_text("name: s1\nruns:\n  - config: configs/gnn.yaml\n    overrides: [split=full]\n")
    s2 = tmp_path / "s2.yaml"
    s2.write_text("name: s2\nruns:\n  - config: configs/sdf_fno.yaml\n    overrides: [split=aoa]\n")
    rc = sweep_mod.main(["--spec", str(s1), "--spec", str(s2), "--dry-run",
                         "--results-root", str(tmp_path / "r")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2 runs" in out
    # spec order preserved: gnn_full before sdf_fno_aoa
    assert out.index("gnn_full_s0") < out.index("sdf_fno_aoa_s0")
