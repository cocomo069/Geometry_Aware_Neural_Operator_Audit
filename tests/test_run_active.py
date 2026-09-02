"""CPU tests for ``scripts/run_active`` (no dataset, no real checkpoints).

A tiny fake K-member ensemble scores a small NACA pool with the *real* frozen
force integrator and the *real* symmetry residual, then the three selection arms
and the Fluent-manifest emission are exercised end to end.  The generated
``cases_to_run.json`` is validated against ``fluent/make_cases.py``'s own loader
and validator, so a schema drift there breaks this test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_active import (
    ARM_NAMES,
    arm_to_fluent_cases,
    score_pool,
    select_arms,
    write_cases_to_run,
    write_ranking_csv,
    write_selection_summary,
)
from src.active.acquisition import acquisition_score
from src.active.pool import NACA4, NACA5, build_pool

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# tiny fake ensemble
# --------------------------------------------------------------------------- #
class _FakeMember:
    """Callable 'model' returning p/tau/coef_head; per-member bias gives spread."""

    def __init__(self, torch, bias: float, seed: int):
        self.torch = torch
        self.bias = float(bias)
        self.g = torch.Generator().manual_seed(int(seed))

    def eval(self):
        return self

    def to(self, *a, **k):
        return self

    def __call__(self, batch):
        torch = self.torch
        pos = batch["surf_pos"]
        cond = batch["cond"]
        n = pos.shape[0]
        b = cond.shape[0]
        # pressure depends on geometry (x, y) + a member bias + a little noise so
        # the integrated drag differs member-to-member (=> sigma_CD > 0) and the
        # y-dependence makes the model non-equivariant (=> L_sym > 0).
        p = pos[:, 0] * 0.7 + pos[:, 1] * 0.3 + self.bias + 0.01 * torch.randn(
            n, generator=self.g
        )
        tau = 0.001 * torch.ones(n, 2)
        mag = torch.linalg.norm(cond, dim=1)
        cd = 0.02 + 1e-4 * mag + self.bias
        cl = 0.10 + 1e-3 * mag + self.bias
        coef = torch.stack([cl, cd], dim=1)
        return {"p": p, "tau": tau, "coef_head": coef}


def _predictor(torch, k=3):
    from src.uq.ensembles import EnsemblePredictor

    members = [_FakeMember(torch, bias=b, seed=100 + j)
               for j, b in enumerate(np.linspace(-0.02, 0.02, k))]
    return EnsemblePredictor(members)


def _small_pool():
    shapes = [NACA4(0.0, 0.0, 0.12), NACA4(0.02, 0.4, 0.12), NACA5(2, 3, 0, 0.12)]
    return build_pool(shapes, re_grid=(3.0e6, 5.0e6), aoa_grid=(0.0, 8.0), exclude=None)


def _score(torch, entries, compute_sym=True):
    device = torch.device("cpu")
    from src.physics.force_integration import integrate_forces

    ident = lambda name, x: x  # noqa: E731
    return score_pool(
        _predictor(torch), entries,
        integrate=integrate_forces, denorm=ident, normalize_cond=lambda x: x,
        device=device, batch_size=4, compute_sym=compute_sym,
        n_points=64, log=lambda *_: None,
    )


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def test_score_pool_components_are_finite_and_shaped():
    torch = pytest.importorskip("torch")
    entries = _small_pool()
    comp = _score(torch, entries)
    n = len(entries)
    for key in ("sigma_cd", "fsc", "l_sym", "cd_int", "cd_head"):
        assert comp[key].shape == (n,), key
        assert np.all(np.isfinite(comp[key])), key
    assert np.all(comp["sigma_cd"] >= 0.0)
    assert np.any(comp["sigma_cd"] > 0.0)   # a real ensemble has spread
    assert np.any(comp["l_sym"] > 0.0)      # the fake is not equivariant


def test_skip_sym_zeroes_the_symmetry_term():
    torch = pytest.importorskip("torch")
    entries = _small_pool()
    comp = _score(torch, entries, compute_sym=False)
    assert np.allclose(comp["l_sym"], 0.0)


# --------------------------------------------------------------------------- #
# z-normalization scale invariance
# --------------------------------------------------------------------------- #
def test_acquisition_scale_invariant():
    rng = np.random.default_rng(0)
    n = 40
    sigma = rng.random(n) + 0.1
    fsc = rng.random(n) + 0.1
    lsym = rng.random(n) + 0.1
    base = acquisition_score(sigma, fsc, lsym, beta=1.0, gamma=1.0)
    # each component under an independent positive affine map a*x + c
    scaled = acquisition_score(
        3.0 * sigma + 5.0, 0.5 * fsc - 2.0, 10.0 * lsym + 1.0, beta=1.0, gamma=1.0
    )
    assert np.allclose(base, scaled, atol=1e-9)
    # ranking (hence the selection) is therefore identical
    assert np.array_equal(np.argsort(-base), np.argsort(-scaled))


# --------------------------------------------------------------------------- #
# three arms: distinct, diverse, k each
# --------------------------------------------------------------------------- #
def test_three_arms_return_k_distinct_cases():
    torch = pytest.importorskip("torch")
    entries = _small_pool()
    comp = _score(torch, entries)
    k = 4
    arms = select_arms(entries, comp, k=k, beta=1.0, gamma=1.0, random_seed=0)
    assert set(arms) == set(ARM_NAMES)
    for arm in ARM_NAMES:
        sel = arms[arm]["selected"]
        assert len(sel) == k
        idx = [c["pool_index"] for c in sel]
        assert len(set(idx)) == k, f"{arm} picked duplicates"
        # farthest-point picks span more than one design point
        keys = {c["key"] for c in sel}
        assert len(keys) == k


def test_acquisition_arm_keeps_the_argmax():
    torch = pytest.importorskip("torch")
    entries = _small_pool()
    comp = _score(torch, entries)
    arms = select_arms(entries, comp, k=3, beta=1.0, gamma=1.0)
    scores = arms["acquisition"]["scores"]
    top = int(np.argmax(scores))
    picked = {c["pool_index"] for c in arms["acquisition"]["selected"]}
    assert top in picked  # pure exploitation is never thrown away


# --------------------------------------------------------------------------- #
# Fluent manifest emission validates against make_cases.py
# --------------------------------------------------------------------------- #
def test_cases_to_run_matches_make_cases_schema(tmp_path):
    torch = pytest.importorskip("torch")
    entries = _small_pool()
    comp = _score(torch, entries)
    k = 4
    arms = select_arms(entries, comp, k=k, beta=1.0, gamma=1.0)

    arm_cases = []
    for arm in ARM_NAMES:
        arm_cases += arm_to_fluent_cases(arm, arms[arm]["selected"], mesh_level=2)
    assert len(arm_cases) == 3 * k

    out = tmp_path / "cases_to_run.json"
    # pull the gridstudy trio + defaults from the real committed manifest
    write_cases_to_run(arm_cases, out, existing_manifest=REPO / "fluent" / "cases_to_run.json",
                       meta={"ensemble_id": "fake_full_k3"})

    # validate with make_cases.py's own loader + validator (imports the module)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_cases_under_test", REPO / "fluent" / "make_cases.py"
    )
    make_cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_cases)

    defaults, cases = make_cases.load_manifest(out)
    make_cases.validate(cases, defaults)  # raises SystemExit on any schema error

    # every required key present, arm-tagged, gridstudy preserved, ids unique
    required = ("case_id", "naca_digits", "re", "aoa_deg", "mesh_level")
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case_id"
    al = [c for c in cases if c["case_id"].startswith("al_")]
    grid = [c for c in cases if c["case_id"].startswith("gridstudy")]
    assert len(al) == 3 * k
    assert len(grid) >= 1, "gridstudy trio was not preserved"
    for c in al:
        for key in required:
            assert key in c, key
        assert c["arm"] in ARM_NAMES
        assert str(c["naca_digits"]).isdigit() and len(str(c["naca_digits"])) in (4, 5)


def test_ranking_csv_and_summary_written(tmp_path):
    torch = pytest.importorskip("torch")
    entries = _small_pool()
    comp = _score(torch, entries)
    arms = select_arms(entries, comp, k=4, beta=1.0, gamma=1.0)

    csv_path = write_ranking_csv(entries, comp, arms, tmp_path / "acquisition_ranking.csv")
    summ_path = write_selection_summary(arms, {"ensemble_id": "fake"}, tmp_path / "selection_summary.json")

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(entries) + 1  # header + one row per pool entry
    assert "acq_score" in lines[0] and "selected_acquisition" in lines[0]

    summary = json.loads(summ_path.read_text(encoding="utf-8"))
    assert set(summary["arms"]) == set(ARM_NAMES)
    assert all(len(summary["arms"][a]) == 4 for a in ARM_NAMES)
