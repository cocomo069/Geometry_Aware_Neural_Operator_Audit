"""End-to-end CPU tests for ``scripts/run_uq`` (no dataset, no real checkpoints).

Two layers:

* the pure record-assembly path (``build_coefficient_records`` /
  ``build_field_records``) driven by a controlled synthetic K-member ensemble,
  where matched conformal must recover nominal coverage; and
* the full inference path (``infer_dataset``) driven by a tiny synthetic
  ensemble of callables + the *real* frozen force integrator, proving the
  checkpoints -> inference -> conformal -> schema-valid JSON pipeline runs on CPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.run_uq import (
    build_coefficient_records,
    build_field_records,
    ensemble_mean_std_axis0,
    find_seed_checkpoints,
    infer_dataset,
    k_subsets,
)
from src.uq.coverage import make_uq_report, validate_uq_report, write_uq_report

LEVELS = (0.8, 0.9, 0.95)


# ==========================================================================
# small pure helpers
# ==========================================================================
def test_k_subsets():
    assert k_subsets(1, None) == [1]
    assert k_subsets(5, None) == [1, 3, 5]
    assert k_subsets(4, None) == [1, 3, 4]
    assert k_subsets(5, [1, 5]) == [1, 5]
    assert k_subsets(2, [5]) == [2]  # requested>K dropped, K always present


def test_ensemble_mean_std_axis0_k1_is_zero_spread():
    members = np.arange(6, dtype=float).reshape(1, 6)
    mean, std = ensemble_mean_std_axis0(members)
    assert np.allclose(mean, members[0])
    assert np.allclose(std, 0.0) and np.all(np.isfinite(std))
    m2, s2 = ensemble_mean_std_axis0(np.stack([np.ones(4), 3 * np.ones(4)]))
    assert np.allclose(m2, 2.0)
    assert np.allclose(s2, np.sqrt(2.0))  # ddof=1 std of {1,3}


def test_find_seed_checkpoints(tmp_path):
    root = tmp_path / "checkpoints"
    (root / "gnn_full_s0").mkdir(parents=True)
    (root / "gnn_full_s0" / "best.pt").write_text("x")
    (root / "gnn_full_s2").mkdir(parents=True)
    (root / "gnn_full_s2" / "last.pt").write_text("x")  # last.pt fallback
    found = find_seed_checkpoints("gnn", "full", [0, 1, 2, 3], root)
    seeds = [s for s, _ in found]
    assert seeds == [0, 2]
    assert found[0][1].name == "best.pt" and found[1][1].name == "last.pt"


# ==========================================================================
# controlled synthetic ensemble -> record assembly
# ==========================================================================
def _store(n_sim, n_pts, rng, k=5, *, sigma=0.4, npts_jitter=False):
    """A store dict shaped like ``infer_dataset``'s output.

    Ground truth is a per-sim (CL, CD) and a per-sim pressure curve; each of the
    K members is truth + i.i.d. Gaussian noise, so the ensemble mean tracks
    truth and its spread ~ sigma -- exactly the exchangeable setting where split
    conformal must hit nominal coverage.
    """
    cl_true = rng.normal(0.4, 0.3, size=n_sim)
    cd_true = rng.normal(0.05, 0.02, size=n_sim)

    def members_for(truth):
        return truth[None, :] + rng.normal(0.0, sigma * 0.05, size=(k, n_sim))

    coef_int = np.stack([members_for(cl_true), members_for(cd_true)], axis=-1)  # (K,N,2)
    coef_head = np.stack([members_for(cl_true), members_for(cd_true)], axis=-1)

    p_members, p_true = [], []
    for i in range(n_sim):
        m = (n_pts + i) if npts_jitter else n_pts
        s = np.linspace(0.0, 1.0, m)
        base = np.cos(4.0 * s) + 0.1 * i
        p_members.append(base[None, :] + rng.normal(0.0, sigma, size=(k, m)))
        p_true.append(base + rng.normal(0.0, sigma, size=m))
    return {
        "coef_int": coef_int,
        "coef_head": coef_head,
        "p_members": p_members,
        "p_true": p_true,
        "cl_true": cl_true,
        "cd_true": cd_true,
        "sim_names": [f"sim{i}" for i in range(n_sim)],
    }


def test_coefficient_records_matched_hit_nominal_coverage():
    rng = np.random.default_rng(20260825)
    cal = _store(300, 32, rng)
    test = _store(1500, 32, rng)
    recs = build_coefficient_records(cal, test, 5, levels=LEVELS)
    # 4 targets x 2 scores
    assert len(recs) == 8
    by = {(r["target"], r["score"]): r for r in recs}
    for target in ("cd_int", "cl_int", "cd_head", "cl_head"):
        for score in ("normalized", "absolute"):
            rec = by[(target, score)]
            covs = [lv["coverage"] for lv in rec["levels"]]
            widths = [lv["mean_width"] for lv in rec["levels"]]
            assert covs == sorted(covs), (target, score, "coverage monotone")
            assert widths == sorted(widths), (target, score, "width monotone")
            cov90 = rec["levels"][1]["coverage"]
            lo = rec["levels"][1]["coverage_ci_lo"]
            hi = rec["levels"][1]["coverage_ci_hi"]
            # nominal inside the Wilson CI (with a little calibration slack)
            assert lo - 0.04 <= 0.9 <= hi + 0.04, (target, score, cov90, lo, hi)


def test_field_records_have_both_granularities_and_scores():
    rng = np.random.default_rng(7)
    cal = _store(200, 40, rng)
    test = _store(600, 40, rng)
    recs = build_field_records(cal, test, 5, levels=LEVELS, agg_q=0.9)
    grans = {(r["granularity"], r["score"]) for r in recs}
    assert grans == {
        ("field_max", "normalized"), ("field_max", "absolute"),
        ("field_quantile", "normalized"), ("field_quantile", "absolute"),
    }
    for r in recs:
        assert all(0.0 <= lv["coverage"] <= 1.0 for lv in r["levels"])
        if r["granularity"] == "field_quantile":
            assert r["agg_q"] == 0.9


def test_k1_uses_absolute_score_only():
    rng = np.random.default_rng(1)
    cal = _store(120, 24, rng)
    test = _store(300, 24, rng)
    coef = build_coefficient_records(cal, test, 1, levels=LEVELS)
    field = build_field_records(cal, test, 1, levels=LEVELS)
    assert {r["score"] for r in coef} == {"absolute"}
    assert {r["score"] for r in field} == {"absolute"}
    assert len(coef) == 4 and len(field) == 2


def test_full_report_validates_and_roundtrips(tmp_path):
    rng = np.random.default_rng(99)
    cal = _store(200, 30, rng)
    test = _store(800, 30, rng)
    records = build_coefficient_records(cal, test, 5, levels=LEVELS)
    records += build_field_records(cal, test, 5, levels=LEVELS)
    report = make_uq_report(
        "gnn_reynolds_k5", "reynolds", records,
        model="gnn", seeds=[0, 1, 2, 3, 4], k=5,
        extra={"mode": "matched", "cal_split": "reynolds"},
    )
    path = write_uq_report(report, tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    validate_uq_report(loaded)  # raises on any schema violation
    assert loaded["k"] == 5 and loaded["split"] == "reynolds"
    assert set(loaded["coef"]) == {"cd_int", "cl_int", "cd_head", "cl_head"}
    assert set(loaded["field"]) == {"p"}

    # also satisfies A4's flat UQ_SCHEMA
    schema = pytest.importorskip("src.eval.schema")
    assert schema.validate_uq(loaded) == []


# ==========================================================================
# full inference path: fake ensemble + real integrator -> schema-valid JSON
# ==========================================================================
class _FakeMember:
    """A callable 'model' returning p/tau/coef_head as torch tensors.

    p = true pressure + member bias + noise; coef_head = a per-sim coefficient
    with the same structure.  Deterministic given the batch, so the ensemble
    spread is controlled entirely by ``bias``/``noise``.
    """

    def __init__(self, torch, bias: float, noise: float, seed: int):
        self.torch = torch
        self.bias = bias
        self.noise = noise
        self.g = torch.Generator().manual_seed(seed)

    def eval(self):
        return self

    def to(self, *a, **k):
        return self

    def __call__(self, batch):
        torch = self.torch
        n = batch["surf_p_phys"].shape[0]
        b = int(batch["num_graphs"])
        p = batch["surf_p_phys"] + self.bias + self.noise * torch.randn(
            n, generator=self.g
        )
        tau = torch.zeros(n, 2)
        coef = torch.stack(
            [batch["cl_true_phys"], batch["cd_true_phys"]], dim=1
        ) + self.bias + self.noise * torch.randn(b, 2, generator=self.g)
        return {"p": p, "tau": tau, "coef_head": coef}


def _make_synth_dataset(torch, n_sim, n_pts, seed):
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n_sim):
        theta = np.linspace(0.0, 2 * np.pi, n_pts, endpoint=False)
        normal = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        ds = np.ones(n_pts) * (2 * np.pi / n_pts)
        p = np.sin(theta) + rng.normal(0, 0.1)
        items.append({
            "surf_normal": torch.tensor(normal, dtype=torch.float32),
            "surf_ds": torch.tensor(ds, dtype=torch.float32),
            "surf_p_phys": torch.tensor(p, dtype=torch.float32),
            "cond_phys": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "cl_true_phys": torch.tensor(float(rng.normal(0.3, 0.2)), dtype=torch.float32),
            "cd_true_phys": torch.tensor(float(rng.normal(0.05, 0.02)), dtype=torch.float32),
            "sim_name": f"sim{i}",
        })
    return items


def _collate(items):
    import torch

    b = len(items)
    counts = [it["surf_p_phys"].shape[0] for it in items]
    batch_idx = torch.cat([torch.full((c,), i, dtype=torch.long) for i, c in enumerate(counts)])
    return {
        "surf_normal": torch.cat([it["surf_normal"] for it in items], dim=0),
        "surf_ds": torch.cat([it["surf_ds"] for it in items], dim=0),
        "surf_p_phys": torch.cat([it["surf_p_phys"] for it in items], dim=0),
        "cond_phys": torch.stack([it["cond_phys"] for it in items], dim=0),
        "cl_true_phys": torch.stack([it["cl_true_phys"] for it in items], dim=0),
        "cd_true_phys": torch.stack([it["cd_true_phys"] for it in items], dim=0),
        "sim_name": [it["sim_name"] for it in items],
        "batch_idx": batch_idx,
        "num_graphs": b,
    }


def test_infer_dataset_end_to_end_cpu(tmp_path):
    torch = pytest.importorskip("torch")
    from src.physics.force_integration import integrate_forces
    from src.uq.ensembles import EnsemblePredictor

    device = torch.device("cpu")
    members = [_FakeMember(torch, bias=b, noise=0.05, seed=10 + j)
               for j, b in enumerate((-0.02, 0.0, 0.02))]
    predictor = EnsemblePredictor(members)

    def denorm(name, x):  # identity: fields already physical
        return x

    cal_ds = _make_synth_dataset(torch, 120, 48, seed=1)
    test_ds = _make_synth_dataset(torch, 200, 48, seed=2)
    kw = dict(collate=_collate, denorm=denorm, integrate=integrate_forces,
              device=device, batch_size=8)

    cal = infer_dataset(predictor, cal_ds, **kw)
    test = infer_dataset(predictor, test_ds, **kw)

    assert cal["coef_int"].shape == (3, 120, 2)
    assert test["coef_head"].shape == (3, 200, 2)
    assert len(test["p_members"]) == 200
    assert test["p_members"][0].shape[0] == 3  # member axis
    assert np.all(np.isfinite(test["cd_true"]))

    records = build_coefficient_records(cal, test, 3, levels=LEVELS)
    records += build_field_records(cal, test, 3, levels=LEVELS)
    report = make_uq_report(
        "gnn_full_k3", "full", records, model="gnn", seeds=[0, 1, 2], k=3,
        extra={"mode": "matched"},
    )
    path = write_uq_report(report, tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    validate_uq_report(loaded)

    # coverage was actually computed for every record/level and is a probability
    for rec in loaded["records"]:
        covs = [lv["coverage"] for lv in rec["levels"]]
        assert all(0.0 <= c <= 1.0 for c in covs)
        assert covs == sorted(covs), rec["target"]  # monotone in nominal level
