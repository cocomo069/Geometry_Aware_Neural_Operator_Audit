"""CPU tests for scripts/compare_fluent (no dataset, no real checkpoints).

Pins the three things the plan (PLAN_FLUENT_POST 4.3) calls out: the
surrogate_vs_fluent CSV schema, the freestream-speed convention fed to the
surrogate, and the offset arithmetic + offset correction / coverage logic.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.compare_fluent as cf


# --------------------------------------------------------------------------- #
# offset arithmetic
# --------------------------------------------------------------------------- #
def _fake_npz(dirpath: Path, sim: str, cd_true: float, cl_true: float):
    np.savez(dirpath / f"{sim}.npz", cd_true=np.float32(cd_true),
             cl_true=np.float32(cl_true))


def test_offset_arithmetic(tmp_path, monkeypatch):
    monkeypatch.setattr(cf, "PROCESSED", tmp_path)
    _fake_npz(tmp_path, "simA", 0.010, 0.50)
    _fake_npz(tmp_path, "simB", 0.020, 1.00)
    summary = [
        dict(case_id="offset_1", arm="offset", model="sa", status="converged",
             airfrans_sim="simA", cd="0.011", cl="0.55"),
        dict(case_id="offset_2", arm="offset", model="sa", status="converged",
             airfrans_sim="simB", cd="0.023", cl="1.06"),
        # a diverged replica must not enter the offset
        dict(case_id="offset_3", arm="offset", model="sa", status="diverged",
             airfrans_sim="simA", cd="nan", cl="nan"),
    ]
    off = cf.compute_offset(summary)
    s = off["summary"]["sa"]
    assert s["n"] == 2
    # deltas: 0.001 and 0.003 -> mean 0.002
    assert s["dbar_cd"] == pytest.approx(0.002, abs=1e-9)
    # cl deltas: 0.05 and 0.06 -> mean 0.055
    assert s["dbar_cl"] == pytest.approx(0.055, abs=1e-9)
    assert {r["case_id"] for r in off["rows"]} == {"offset_1", "offset_2"}


# --------------------------------------------------------------------------- #
# conformal half-widths
# --------------------------------------------------------------------------- #
def test_conformal_halfwidths(tmp_path, monkeypatch):
    kjson = {"coef": {t: {"0.9": {"width_mean": 0.02, "width_median": 0.02}}
                      for t in ("cd_int", "cd_head", "cl_int", "cl_head")}}
    (tmp_path / "uq").mkdir()
    (tmp_path / "uq" / "fake_k.json").write_text(json.dumps(kjson), encoding="utf-8")
    monkeypatch.setattr(cf, "RESULTS", tmp_path)
    monkeypatch.setattr(cf, "MODEL_K_JSON", {"transolver": "fake_k.json"})
    half = cf.conformal_halfwidths("transolver")
    assert half["cd_int"] == pytest.approx(0.01)   # width_mean / 2


# --------------------------------------------------------------------------- #
# row building: schema, offset correction, coverage, diverged handling
# --------------------------------------------------------------------------- #
def test_build_rows_schema_and_arithmetic(monkeypatch):
    summary = [
        dict(case_id="al_rand_naca0010_re7e6_a6", arm="random", model="sa",
             status="converged", naca="0010", re="7e6", aoa_deg="6",
             u_inf="109.0", cd="0.0100", cl="0.60", airfrans_sim=""),
        dict(case_id="offset_1_naca0016_a2p4", arm="offset", model="sa",
             status="quasi_steady", naca="0016", re="3.9e6", aoa_deg="2.4",
             u_inf="60.54", cd="0.0110", cl="0.26",
             airfrans_sim="airFoil2D_SST_x"),
        dict(case_id="al_acq_naca0010_re7e6_a18", arm="acquisition", model="sa",
             status="diverged", naca="0010", re="7e6", aoa_deg="18",
             u_inf="109.0", cd="nan", cl="nan", airfrans_sim=""),
    ]
    offset = {"summary": {"sa": dict(n=1, dbar_cd=0.001, s_cd=0.0,
                                     dbar_cl=0.01, s_cl=0.0)}}

    monkeypatch.setattr(cf, "conformal_halfwidths",
                        lambda model, level="0.9": {"cd_int": 0.02, "cd_head": 0.02,
                                                    "cl_int": 0.1, "cl_head": 0.1})
    # canned predictions: non-replica via score_pool, replica via per_sim
    def fake_score(model, cases):
        return {c["case_id"]: dict(
            cd_int_mean=0.0105, cd_int_std=0.001, cd_head_mean=0.0102, cd_head_std=0.0005,
            cl_int_mean=0.61, cl_int_std=0.02, cl_head_mean=0.60, cl_head_std=0.01)
            for c in cases}
    monkeypatch.setattr(cf, "_score_pool_cases", fake_score)
    monkeypatch.setattr(cf, "_replica_prediction",
                        lambda model, sim: dict(
                            cd_int_mean=0.0108, cd_int_std=0.0008, cd_head_mean=0.0106, cd_head_std=0.0004,
                            cl_int_mean=0.27, cl_int_std=0.01, cl_head_mean=0.265, cl_head_std=0.008))

    rows = cf.build_rows("transolver", summary, offset)
    by = {r["case_id"]: r for r in rows}
    assert set(by) == {r["case_id"] for r in summary}

    # schema: every declared field present
    for r in rows:
        for k in cf.CSV_FIELDS:
            assert k in r, k

    # non-replica converged: err = pred - (cd_fluent - dbar) = 0.0105 - (0.0100-0.001)
    r0 = by["al_rand_naca0010_re7e6_a6"]
    assert r0["pred_source"] == "score_pool"
    assert r0["cd_fluent_corrected"] == pytest.approx(0.009, abs=1e-9)
    assert r0["err_cd_int"] == pytest.approx(0.0105 - 0.009, abs=1e-9)
    assert r0["covered90_cd_int"] is True  # |0.0015| <= 0.02

    # replica quasi_steady enters (b) and uses per_sim
    r1 = by["offset_1_naca0016_a2p4"]
    assert r1["pred_source"] == "per_sim"
    assert np.isfinite(r1["err_cd_int"])

    # diverged: surrogate columns filled, err NaN, coverage None
    r2 = by["al_acq_naca0010_re7e6_a18"]
    assert np.isfinite(r2["cd_int_mean"]) and not np.isfinite(r2["err_cd_int"])
    assert r2["covered90_cd_int"] is None


# --------------------------------------------------------------------------- #
# freestream-speed convention (the surrogate must see the Fluent U)
# --------------------------------------------------------------------------- #
def test_score_pool_uses_per_entry_speed():
    torch = pytest.importorskip("torch")
    from scripts.run_active import score_pool
    from src.active.pool import NACA4, PoolEntry
    from src.physics.force_integration import integrate_forces
    from tests.test_run_active import _predictor

    ent = PoolEntry(NACA4(0.0, 0.0, 0.12), 3.0e6, 0.0)
    ident = lambda name, x: x  # noqa: E731
    kw = dict(integrate=integrate_forces, denorm=ident, normalize_cond=lambda x: x,
              device=torch.device("cpu"), batch_size=4, compute_sym=False,
              n_points=64, return_members=True, log=lambda *_: None)
    slow = score_pool(_predictor(torch), [ent], speeds=[10.0], **kw)
    fast = score_pool(_predictor(torch), [ent], speeds=[100.0], **kw)
    # the fake member's cd depends on |cond|, so a different speed => different cd
    assert not np.isclose(slow["cd_int"][0], fast["cd_int"][0])
    # per-member array shape (K, N)
    assert slow["cd_int_members"].shape == (3, 1)
