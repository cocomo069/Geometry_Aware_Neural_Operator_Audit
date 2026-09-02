"""Active-learning acquisition: score an unseen NACA pool -> Fluent case list (C4).

    .venv/Scripts/python.exe -m scripts.run_active --help
    .venv/Scripts/python.exe -m scripts.run_active --model transolver --split full --k 8

This is the *now-runnable* part of PLAN_PHASE2 Phase D (design doc section 5.8):
inference only, GPU-light, no training, no Fluent execution, no Kaggle.  It

1. builds a parametric pool of NACA 4- and 5-digit airfoils crossed with a grid
   of (Re, AoA) that reaches OUTSIDE the AirfRANS envelope, excluding every
   training combination of the scorer's split (``src.active.pool``);
2. loads a K-member deep ensemble (default the Transolver K=5 ``full`` ensemble,
   the best model at full data) via ``src.uq.ensembles`` and runs single-process
   inference over the pool, computing the three label-free acquisition
   components of section 5.8

       a(g, c) = z(sigma_CD) + beta * z(FSC) + gamma * z(L_sym)

   with sigma_CD the ensemble std of the *integrated* drag coefficient
   (integrate-then-average through the frozen ``src.physics.force_integration``),
   FSC = |mean C_D^int - mean C_D^head| the force self-consistency gap, and
   L_sym the mirror-symmetry residual (``src.geometry.symmetry``), each
   z-normalized over the pool (beta = gamma = 1 by default, CLI-overridable);
3. produces THREE selections of k (default 8) via farthest-point diversity in
   normalized design space (``src.active.diversity``) -- the three arms the paper
   compares: (a) top-acquisition, (b) top-ensemble-variance-only, (c) random;
4. writes ``fluent/cases_to_run.json`` in the schema ``fluent/make_cases.py``
   consumes (case_id / naca_digits / re / aoa_deg / mesh_level plus an ``arm``
   tag), PRESERVING the ``gridstudy_*`` grid-independence trio and replacing only
   the placeholder cases, plus a human-readable
   ``results/active/acquisition_ranking.csv`` and
   ``results/active/selection_summary.json``.

CPU-light by mandate (a user Fluent job may be running): ``num_workers=0``, small
batches, ``torch.no_grad``; the P2000 GPU carries the forward passes when
available.  Nothing here trains, runs Fluent, or touches the Ansys MCP tools --
it only reads checkpoints, calls the frozen integrator, and emits text.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.active.acquisition import (  # noqa: E402
    acquisition_score,
    random_scores,
    select_cases,
    variance_only_scores,
    zscore,
)
from src.active.pool import (  # noqa: E402
    ExclusionSet,
    NU_AIR,
    PoolEntry,
    build_pool,
    collate_batches,
    entry_to_batch,
)
from src.data.splits import NU_DATASET, parse_sim_name  # noqa: E402
from src.uq.ensembles import load_ensemble_checkpoints  # noqa: E402

__all__ = [
    "DEFAULT_RE_GRID",
    "DEFAULT_AOA_GRID",
    "ARM_NAMES",
    "ARM_SHORT",
    "training_exclusions",
    "build_active_pool",
    "score_pool",
    "select_arms",
    "arm_to_fluent_cases",
    "write_cases_to_run",
    "write_ranking_csv",
    "write_selection_summary",
    "run_active",
    "main",
]

#: (Re, AoA) grid.  Deliberately overshoots the AirfRANS envelope (Re ~2-6e6,
#: AoA ~-5..15 deg) with a 7e6 Reynolds row and +-18/-6 deg angles so the
#: acquisition score can pick genuinely out-of-distribution cases.
DEFAULT_RE_GRID: tuple[float, ...] = (2.0e6, 3.0e6, 4.0e6, 5.0e6, 6.0e6, 7.0e6)
DEFAULT_AOA_GRID: tuple[float, ...] = (-6.0, 0.0, 6.0, 12.0, 18.0)

#: The three selection arms the paper compares, and their case_id prefixes.
ARM_NAMES: tuple[str, ...] = ("acquisition", "variance", "random")
ARM_SHORT: dict[str, str] = {"acquisition": "acq", "variance": "var", "random": "rand"}


# --------------------------------------------------------------------------- #
# pool construction + training exclusion
# --------------------------------------------------------------------------- #
def _sim_to_exclusion(name: str) -> tuple[str, float, float]:
    """Best-effort ``(naca-code, Re, AoA)`` for one AirfRANS training sim name.

    AirfRANS samples the NACA design space *continuously*, so we round each
    training sim's parameters to the nearest standard integer code; combined with
    the Re/AoA tolerances of :class:`ExclusionSet`, this drops any pool member
    that lands on top of a training case.  The pool codes are exact integers, so
    only near-integer training shapes at a matching (Re, AoA) can collide -- which
    is exactly the "not in the training set" guarantee we want.
    """
    s = parse_sim_name(name)
    if s.n_digits == 4:
        m, p, t = s.naca_params
        m_d, p_d, t_d = int(round(m)), int(round(p)), int(round(t))
        if m_d == 0:
            p_d = 0
        code = f"{m_d:d}{p_d:d}{t_d:02d}"
    else:  # 5-digit: (l, p, reflex, thickness) -> 5-char code, best effort
        vals = s.naca_params
        t_d = int(round(vals[-1]))
        lead = "".join(str(int(round(v)) % 10) for v in vals[:-1])
        code = f"{lead}{t_d:02d}"
    return code, s.re, s.aoa_deg


def training_exclusions(
    split: str,
    splits_dir: Path,
    *,
    re_tol: float = 2.5e4,
    aoa_tol: float = 0.5,
) -> ExclusionSet:
    """Build an :class:`ExclusionSet` from a split's ``train`` sim list."""
    payload = json.loads((Path(splits_dir) / f"{split}.json").read_text(encoding="utf-8"))
    entries = []
    for name in payload.get("train", []):
        try:
            entries.append(_sim_to_exclusion(name))
        except ValueError:
            continue  # unparseable name: cannot exclude, skip
    return ExclusionSet(entries, re_tol=re_tol, aoa_tol=aoa_tol)


def build_active_pool(
    split: str,
    splits_dir: Path,
    *,
    shapes: Sequence[Any] | None = None,
    re_grid: Sequence[float] = DEFAULT_RE_GRID,
    aoa_grid: Sequence[float] = DEFAULT_AOA_GRID,
    re_tol: float = 2.5e4,
    aoa_tol: float = 0.5,
) -> tuple[list[PoolEntry], ExclusionSet, int]:
    """``(pool, exclusion_set, n_dropped)`` for the scorer's split.

    Every returned entry is guaranteed absent from the split's training set.
    """
    excl = training_exclusions(split, splits_dir, re_tol=re_tol, aoa_tol=aoa_tol)
    full = build_pool(shapes, re_grid=re_grid, aoa_grid=aoa_grid, exclude=None)
    kept = [e for e in full if not excl.excludes(e)]
    n_dropped = len(full) - len(kept)
    # Hard guarantee (design doc 5.8: "none in the training set").
    assert not any(excl.excludes(e) for e in kept), "pool still contains a training combination"
    return kept, excl, n_dropped


# --------------------------------------------------------------------------- #
# inference / scoring
# --------------------------------------------------------------------------- #
def _to_device(batch: Mapping[str, Any], torch, device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _chunks(seq: Sequence[Any], n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def score_pool(
    predictor,
    entries: Sequence[PoolEntry],
    *,
    integrate: Callable[..., Mapping[str, Any]],
    denorm: Callable[[str, Any], Any],
    normalize_cond: Callable[[Any], Any],
    device: Any,
    batch_size: int = 16,
    compute_sym: bool = True,
    n_points: int = 200,
    nu: float = NU_DATASET,
    chord: float = 1.0,
    rho: float = 1.0,
    a_ref: float = 1.0,
    ddof: int = 1,
    log: Callable[[str], None] = print,
) -> dict[str, np.ndarray]:
    """Run the ensemble over the pool; return the acquisition components.

    Returns a dict of ``(N,)`` arrays: ``sigma_cd`` (ensemble std of the
    integrated C_D), ``fsc`` (|mean C_D^int - mean C_D^head|), ``l_sym`` (mean
    mirror-symmetry residual over members, or all-zeros when ``compute_sym`` is
    False), plus the per-entry means ``cd_int``/``cd_head``/``cl_int``/``cl_head``.

    Inputs are fed to the model with **normalized** ``cond`` (as trained) while
    the force integral uses the **physical** freestream; model outputs ``p``/
    ``tau`` are denormalized before integration (integrate-then-average, spec 5.6).
    """
    import torch

    if compute_sym:
        from src.geometry.symmetry import symmetry_residual

    sigma_cd: list[np.ndarray] = []
    cd_int_mean: list[np.ndarray] = []
    cd_head_mean: list[np.ndarray] = []
    cl_int_mean: list[np.ndarray] = []
    cl_head_mean: list[np.ndarray] = []
    l_sym: list[np.ndarray] = []

    n_done = 0
    with torch.no_grad():
        for chunk in _chunks(list(entries), int(batch_size)):
            batches = [
                entry_to_batch(e, n_points=n_points, nu=nu, chord=chord, as_torch=False)
                for e in chunk
            ]
            batch = collate_batches(batches, as_torch=True)
            batch["num_graphs"] = len(chunk)
            batch = _to_device(batch, torch, device)

            cond_phys = batch["cond"]
            model_batch = dict(batch)
            model_batch["cond"] = normalize_cond(cond_phys)

            bidx = batch["batch_idx"]
            normal = batch["surf_normal"]
            ds = batch["surf_ds"]
            b = len(chunk)

            outs = predictor.member_outputs(model_batch, raw=True)
            kk = len(outs)
            cd_i = np.empty((kk, b), dtype=np.float64)
            cl_i = np.empty((kk, b), dtype=np.float64)
            cd_h = np.empty((kk, b), dtype=np.float64)
            cl_h = np.empty((kk, b), dtype=np.float64)
            for m, out in enumerate(outs):
                p_phys = denorm("surf_p", out["p"].reshape(-1))
                tau_phys = denorm("surf_tau", out["tau"])
                forces = integrate(
                    p_phys, tau_phys, normal, ds, cond_phys,
                    batch_idx=bidx, rho=rho, a_ref=a_ref,
                )
                cl_i[m] = _np(forces["cl"]).reshape(-1)
                cd_i[m] = _np(forces["cd"]).reshape(-1)
                coef = out["coef_head"].reshape(b, -1)
                cl_h[m] = _np(denorm("cl_true", coef[:, 0])).reshape(-1)
                cd_h[m] = _np(denorm("cd_true", coef[:, 1])).reshape(-1)

            sig = np.zeros(b) if kk < 2 else cd_i.std(axis=0, ddof=ddof)
            sigma_cd.append(sig)
            cd_int_mean.append(cd_i.mean(axis=0))
            cd_head_mean.append(cd_h.mean(axis=0))
            cl_int_mean.append(cl_i.mean(axis=0))
            cl_head_mean.append(cl_h.mean(axis=0))

            if compute_sym:
                res = np.zeros((kk, b), dtype=np.float64)
                for m, member in enumerate(predictor.models):
                    r = symmetry_residual(member, model_batch)
                    res[m] = _np(r).reshape(-1)
                l_sym.append(res.mean(axis=0))
            else:
                l_sym.append(np.zeros(b))

            n_done += b
            log(f"[active]   scored {n_done}/{len(entries)} pool entries")

    out = {
        "sigma_cd": np.concatenate(sigma_cd) if sigma_cd else np.zeros(0),
        "fsc": np.abs(
            np.concatenate(cd_int_mean) - np.concatenate(cd_head_mean)
        ) if cd_int_mean else np.zeros(0),
        "l_sym": np.concatenate(l_sym) if l_sym else np.zeros(0),
        "cd_int": np.concatenate(cd_int_mean) if cd_int_mean else np.zeros(0),
        "cd_head": np.concatenate(cd_head_mean) if cd_head_mean else np.zeros(0),
        "cl_int": np.concatenate(cl_int_mean) if cl_int_mean else np.zeros(0),
        "cl_head": np.concatenate(cl_head_mean) if cl_head_mean else np.zeros(0),
    }
    for key in ("sigma_cd", "fsc", "l_sym"):
        if out[key].size and not np.all(np.isfinite(out[key])):
            raise ValueError(f"acquisition component {key!r} has non-finite values")
    return out


# --------------------------------------------------------------------------- #
# selection (the three arms)
# --------------------------------------------------------------------------- #
def select_arms(
    entries: Sequence[PoolEntry],
    components: Mapping[str, np.ndarray],
    *,
    k: int,
    beta: float = 1.0,
    gamma: float = 1.0,
    use_sym: bool = True,
    random_seed: int = 0,
    oversample: int = 4,
) -> dict[str, dict[str, Any]]:
    """Compute the three arms' scores and diverse k-selections.

    Returns ``{arm: {"scores": (N,), "selected": [case dict, ...]}}`` for
    ``acquisition``, ``variance`` and ``random``.
    """
    sigma = np.asarray(components["sigma_cd"], dtype=np.float64)
    fsc = np.asarray(components["fsc"], dtype=np.float64)
    lsym = np.asarray(components["l_sym"], dtype=np.float64) if use_sym else None
    n = sigma.size

    scores = {
        "acquisition": acquisition_score(sigma, fsc, lsym, beta=beta, gamma=gamma),
        "variance": variance_only_scores(sigma),
        "random": random_scores(n, seed=random_seed),
    }
    arms: dict[str, dict[str, Any]] = {}
    for arm, s in scores.items():
        selected = select_cases(entries, s, k, diverse=True, oversample=oversample)
        arms[arm] = {"scores": np.asarray(s, dtype=np.float64), "selected": selected}
    return arms


# --------------------------------------------------------------------------- #
# Fluent manifest emission
# --------------------------------------------------------------------------- #
def _tag_re(re: float) -> str:
    return f"{re / 1e6:g}e6"


def _tag_aoa(aoa: float) -> str:
    return f"{aoa:g}".replace("-", "m").replace(".", "p")


def arm_to_fluent_cases(
    arm: str,
    selected: Sequence[Mapping[str, Any]],
    *,
    mesh_level: int = 2,
) -> list[dict[str, Any]]:
    """Turn one arm's selected pool cases into ``make_cases.py`` case dicts."""
    short = ARM_SHORT.get(arm, arm[:4])
    cases: list[dict[str, Any]] = []
    for sel in selected:
        code = str(sel["code"])
        re = float(sel["re"])
        aoa = float(sel["aoa_deg"])
        case_id = f"al_{short}_naca{code}_re{_tag_re(re)}_a{_tag_aoa(aoa)}"
        cases.append({
            "case_id": case_id,
            "naca_digits": code,
            "re": re,
            "aoa_deg": aoa,
            "mesh_level": int(mesh_level),
            "arm": arm,
            "score": float(sel.get("score", float("nan"))),
            "rank": int(sel.get("rank", -1)),
            "note": (
                f"Active-learning '{arm}' pick (rank {sel.get('rank', '?')}, "
                f"score {sel.get('score', float('nan')):.4g}). {sel.get('family', 'naca')} "
                f"section NACA {code} at Re {re:.3g}, AoA {aoa:g} deg. "
                "Selected by scripts/run_active.py; NOT hand-tuned."
            ),
        })
    return cases


def write_cases_to_run(
    arm_cases: Sequence[Mapping[str, Any]],
    out_path: Path,
    *,
    existing_manifest: Path | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``fluent/cases_to_run.json`` = gridstudy trio + active-learning arms.

    The ``gridstudy_*`` grid-independence family and the ``defaults`` block are
    read from ``existing_manifest`` (when present) and preserved verbatim; only
    the placeholder cases are replaced by the active-learning selection.
    """
    out_path = Path(out_path)
    defaults: dict[str, Any] = {}
    readme: Any = None
    gridstudy: list[dict[str, Any]] = []
    if existing_manifest and Path(existing_manifest).is_file():
        raw = json.loads(Path(existing_manifest).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            defaults = dict(raw.get("defaults", {}))
            readme = raw.get("_readme")
            for c in raw.get("cases", []):
                if str(c.get("case_id", "")).startswith("gridstudy"):
                    gridstudy.append(dict(c))

    payload: dict[str, Any] = {}
    if readme is not None:
        payload["_readme"] = readme
    if meta:
        payload["active_learning"] = dict(meta)
    if defaults:
        payload["defaults"] = defaults
    payload["cases"] = gridstudy + [dict(c) for c in arm_cases]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def write_ranking_csv(
    entries: Sequence[PoolEntry],
    components: Mapping[str, np.ndarray],
    arms: Mapping[str, Mapping[str, Any]],
    out_path: Path,
) -> Path:
    """Write the full pool score table (also serves as label-free FSC evidence)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sigma = np.asarray(components["sigma_cd"], dtype=np.float64)
    fsc = np.asarray(components["fsc"], dtype=np.float64)
    lsym = np.asarray(components["l_sym"], dtype=np.float64)
    z_sigma = zscore(sigma)
    z_fsc = zscore(fsc)
    z_lsym = zscore(lsym)

    acq_scores = np.asarray(arms["acquisition"]["scores"], dtype=np.float64)
    acq_order = np.argsort(-acq_scores, kind="stable")
    acq_rank = np.empty(acq_scores.size, dtype=int)
    acq_rank[acq_order] = np.arange(acq_scores.size)

    selected_idx = {arm: {int(c["pool_index"]) for c in arms[arm]["selected"]} for arm in arms}

    fields = [
        "pool_index", "key", "family", "code", "re", "aoa_deg",
        "max_camber", "x_max_camber", "max_thickness", "x_max_thickness",
        "sigma_cd", "fsc", "l_sym", "z_sigma_cd", "z_fsc", "z_l_sym",
        "acq_score", "var_score", "acq_rank",
        "selected_acquisition", "selected_variance", "selected_random",
    ]
    var_scores = np.asarray(arms["variance"]["scores"], dtype=np.float64)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, e in enumerate(entries):
            d = e.descriptor()
            w.writerow({
                "pool_index": i,
                "key": e.key,
                "family": e.family,
                "code": e.code,
                "re": f"{e.re:.6g}",
                "aoa_deg": f"{e.aoa_deg:g}",
                "max_camber": f"{d['max_camber']:.6g}",
                "x_max_camber": f"{d['x_max_camber']:.6g}",
                "max_thickness": f"{d['max_thickness']:.6g}",
                "x_max_thickness": f"{d['x_max_thickness']:.6g}",
                "sigma_cd": f"{sigma[i]:.6g}",
                "fsc": f"{fsc[i]:.6g}",
                "l_sym": f"{lsym[i]:.6g}",
                "z_sigma_cd": f"{z_sigma[i]:.6g}",
                "z_fsc": f"{z_fsc[i]:.6g}",
                "z_l_sym": f"{z_lsym[i]:.6g}",
                "acq_score": f"{acq_scores[i]:.6g}",
                "var_score": f"{var_scores[i]:.6g}",
                "acq_rank": int(acq_rank[i]),
                "selected_acquisition": int(i in selected_idx["acquisition"]),
                "selected_variance": int(i in selected_idx["variance"]),
                "selected_random": int(i in selected_idx["random"]),
            })
    return out_path


def write_selection_summary(
    arms: Mapping[str, Mapping[str, Any]],
    meta: Mapping[str, Any],
    out_path: Path,
) -> Path:
    """Write ``results/active/selection_summary.json`` (meta + the three arms)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def trim(sel: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "rank": int(sel["rank"]),
            "pool_index": int(sel["pool_index"]),
            "key": sel["key"],
            "family": sel["family"],
            "naca_digits": sel["code"],
            "re": float(sel["re"]),
            "aoa_deg": float(sel["aoa_deg"]),
            "score": float(sel["score"]),
        }

    payload = {
        **dict(meta),
        "arms": {arm: [trim(s) for s in arms[arm]["selected"]] for arm in arms},
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _np(x) -> np.ndarray:
    from src.uq.ensembles import as_numpy

    return as_numpy(x)


def _resolve_device(spec: str):
    import torch

    if spec in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _find_checkpoints(model: str, split: str, seeds: Sequence[int], ckpt_root: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for seed in seeds:
        run_id = f"{model}_{split}_s{seed}"
        best = ckpt_root / run_id / "best.pt"
        last = ckpt_root / run_id / "last.pt"
        if best.is_file():
            found.append((int(seed), best))
        elif last.is_file():
            found.append((int(seed), last))
    return found


def _load_config(model: str, split: str, seeds: Sequence[int], results_root: Path) -> Mapping[str, Any]:
    from src.utils.config import load_config

    for seed in list(seeds) + [0]:
        cfg_path = results_root / f"{model}_{split}_s{seed}" / "config.yaml"
        if cfg_path.is_file():
            return load_config(cfg_path, [])
    raise FileNotFoundError(
        f"no config.yaml for {model}_{split} under {results_root} (seeds {list(seeds)} + 0)"
    )


def _norm_stats(split: str, processed_dir: Path, splits_dir: Path):
    """Per-split train-only normalization stats (D-021), as a ``NormStats``."""
    from src.data.airfrans_loader import NormStats
    from scripts.build_cache import compute_norm_stats

    train = json.loads((splits_dir / f"{split}.json").read_text(encoding="utf-8"))["train"]
    payload = compute_norm_stats(train, processed_dir, split_name=split)
    return NormStats(payload)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_active(
    *,
    model: str = "transolver",
    split: str = "full",
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    k: int = 8,
    beta: float = 1.0,
    gamma: float = 1.0,
    compute_sym: bool = True,
    n_points: int = 200,
    re_grid: Sequence[float] = DEFAULT_RE_GRID,
    aoa_grid: Sequence[float] = DEFAULT_AOA_GRID,
    mesh_level: int = 2,
    random_seed: int = 0,
    oversample: int = 4,
    batch_size: int = 16,
    device_spec: str = "auto",
    processed_dir: Path = Path("data/processed/airfrans"),
    splits_dir: Path = Path("data/splits"),
    ckpt_root: Path = Path("checkpoints"),
    results_root: Path = Path("results"),
    cases_out: Path = Path("fluent/cases_to_run.json"),
    active_out_dir: Path = Path("results/active"),
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """End-to-end: pool -> ensemble scoring -> 3 arms -> Fluent case list."""
    import torch

    from src.physics.force_integration import integrate_forces

    found = _find_checkpoints(model, split, seeds, ckpt_root)
    if not found:
        raise FileNotFoundError(
            f"no checkpoints for {model}_{split} seeds {list(seeds)} under {ckpt_root}"
        )
    used_seeds = [s for s, _ in found]
    paths = [p for _, p in found]
    log(f"[active] {model}_{split}: K={len(paths)} checkpoints (seeds {used_seeds})")

    cfg = _load_config(model, split, seeds, results_root)
    device = _resolve_device(device_spec)
    log(f"[active] device = {device}")

    predictor = load_ensemble_checkpoints(paths, cfg, map_location="cpu")
    for m in predictor.models:
        try:
            m.to(device)
        except Exception:  # pragma: no cover
            pass

    stats = _norm_stats(split, processed_dir, splits_dir)

    def denorm(name: str, x):
        return stats.denormalize(name, x)

    def normalize_cond(x):
        return stats.normalize("cond", x)

    entries, _excl, n_dropped = build_active_pool(
        split, splits_dir, re_grid=re_grid, aoa_grid=aoa_grid
    )
    log(f"[active] pool: {len(entries)} entries ({n_dropped} dropped as training combos)")

    components = score_pool(
        predictor, entries,
        integrate=integrate_forces, denorm=denorm, normalize_cond=normalize_cond,
        device=device, batch_size=batch_size, compute_sym=compute_sym,
        n_points=n_points, log=log,
    )

    arms = select_arms(
        entries, components, k=k, beta=beta, gamma=gamma,
        use_sym=compute_sym, random_seed=random_seed, oversample=oversample,
    )

    ensemble_id = f"{model}_{split}_k{len(paths)}"
    meta = {
        "generated": str(date.today()),
        "ensemble_id": ensemble_id,
        "model": model,
        "split": split,
        "seeds": used_seeds,
        "k_members": len(paths),
        "k_select": int(k),
        "beta": float(beta),
        "gamma": float(gamma) if compute_sym else 0.0,
        "compute_sym": bool(compute_sym),
        "pool_size": len(entries),
        "n_dropped_training": int(n_dropped),
        "re_grid": list(re_grid),
        "aoa_grid": list(aoa_grid),
        "mesh_level": int(mesh_level),
        "random_seed": int(random_seed),
        "device": str(device),
        "source": "scripts/run_active.py",
    }

    arm_cases: list[dict[str, Any]] = []
    for arm in ARM_NAMES:
        arm_cases += arm_to_fluent_cases(arm, arms[arm]["selected"], mesh_level=mesh_level)

    cases_path = write_cases_to_run(
        arm_cases, cases_out, existing_manifest=cases_out, meta=meta
    )
    csv_path = write_ranking_csv(entries, components, arms, active_out_dir / "acquisition_ranking.csv")
    summary_path = write_selection_summary(arms, meta, active_out_dir / "selection_summary.json")

    log(f"[active] wrote {cases_path} ({len(arm_cases)} AL cases + gridstudy trio)")
    log(f"[active] wrote {csv_path}")
    log(f"[active] wrote {summary_path}")

    return {
        "entries": entries,
        "components": components,
        "arms": arms,
        "meta": meta,
        "cases_path": cases_path,
        "csv_path": csv_path,
        "summary_path": summary_path,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in str(text).replace(",", " ").split()]


def _parse_float_list(text: str) -> list[float]:
    return [float(x) for x in str(text).replace(",", " ").split()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts/run_active.py",
        description="Score an unseen NACA pool with an ensemble's uncertainty and "
                    "emit the Fluent case list (active learning, design doc 5.8).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="transolver", help="scorer architecture (default transolver)")
    p.add_argument("--split", default="full", help="ensemble split to load (default full)")
    p.add_argument("--seeds", default="0,1,2,3,4", help="ensemble seeds (default 0,1,2,3,4)")
    p.add_argument("--k", type=int, default=8, help="cases to select per arm (default 8)")
    p.add_argument("--beta", type=float, default=1.0, help="FSC weight (default 1.0)")
    p.add_argument("--gamma", type=float, default=1.0, help="L_sym weight (default 1.0)")
    p.add_argument("--skip-sym", action="store_true",
                   help="drop the symmetry-residual term (gamma=0); faster, no reflected passes")
    p.add_argument("--n-points", type=int, default=200, help="contour points per airfoil (default 200)")
    p.add_argument("--re-grid", default=None, help="comma list of Reynolds numbers (default overshoots the envelope)")
    p.add_argument("--aoa-grid", default=None, help="comma list of AoA in deg (default overshoots the envelope)")
    p.add_argument("--mesh-level", type=int, default=2, help="Fluent mesh level 1/2/3 (default 2)")
    p.add_argument("--random-seed", type=int, default=0, help="seed for the random arm (default 0)")
    p.add_argument("--oversample", type=int, default=4, help="diversity shortlist multiplier (default 4)")
    p.add_argument("--batch-size", type=int, default=16, help="pool inference chunk size (default 16)")
    p.add_argument("--device", default="auto", help="cpu | cuda | auto")
    p.add_argument("--processed-dir", default="data/processed/airfrans")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--checkpoints-root", default="checkpoints")
    p.add_argument("--results-root", default="results")
    p.add_argument("--cases-out", default="fluent/cases_to_run.json")
    p.add_argument("--active-out-dir", default="results/active")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    re_grid = _parse_float_list(args.re_grid) if args.re_grid else DEFAULT_RE_GRID
    aoa_grid = _parse_float_list(args.aoa_grid) if args.aoa_grid else DEFAULT_AOA_GRID

    run_active(
        model=args.model,
        split=args.split,
        seeds=_parse_int_list(args.seeds),
        k=args.k,
        beta=args.beta,
        gamma=args.gamma,
        compute_sym=not args.skip_sym,
        n_points=args.n_points,
        re_grid=re_grid,
        aoa_grid=aoa_grid,
        mesh_level=args.mesh_level,
        random_seed=args.random_seed,
        oversample=args.oversample,
        batch_size=args.batch_size,
        device_spec=args.device,
        processed_dir=Path(args.processed_dir),
        splits_dir=Path(args.splits_dir),
        ckpt_root=Path(args.checkpoints_root),
        results_root=Path(args.results_root),
        cases_out=Path(args.cases_out),
        active_out_dir=Path(args.active_out_dir),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
