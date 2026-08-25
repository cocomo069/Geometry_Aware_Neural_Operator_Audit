"""Turn trained ensemble checkpoints into the paper's calibration results (C3).

    .venv/Scripts/python.exe -m scripts.run_uq --model gnn --split reynolds --seeds 0,1,2,3,4
    .venv/Scripts/python.exe -m scripts.run_uq --all --seeds 0,1,2,3,4

For each ``(model, split)`` this script

1. reads ``results/{model}_{split}_s0/config.yaml`` for the model architecture;
2. builds the split's ``cal``/``test`` datasets (and ``full``'s ``cal`` for the
   transfer mode) with **per-split train-only** normalization (D-021), so no
   test/cal statistic ever enters the transform and the numbers match the
   committed core-grid metrics;
3. loads the K available ``best.pt`` checkpoints (``src.uq.ensembles``), runs
   inference over cal + test, and reduces each ensemble to per-simulation
   coefficient members (``cd_int``/``cl_int`` via the frozen
   ``src.physics.force_integration``; ``cd_head``/``cl_head`` from ``coef_head``)
   and per-point field members (denormalized ``p``);
4. fits split conformal (``src.uq.conformal``) on the cal set and evaluates the
   swept levels on the test set, in **two calibration modes** --

   * *matched* (``{model}_{split}_k{K}.json``): calibrate on the split's own
     ``cal`` -> "conformal delivers nominal coverage when exchangeable";
   * *transfer* (``{model}_{split}_k{K}_calfull.json``): calibrate on ``full``'s
     ``cal``, evaluate on the OOD split's ``test`` -> "the guarantee breaks
     under shift, by this much";

5. writes schema-valid ``results/uq/*.json`` (``src.uq.coverage.write_uq_report``,
   which satisfies both ``uq-1`` and ``src/eval/schema.py::UQ_SCHEMA``).

Zero-GPU ablation knobs, all folded in for free (no extra training):

* **ensemble size** K in {1, 3, 5} -- member subsets, one file each
  (``k{K}``); K=1 uses the ``absolute`` score only (sigma is identically zero).
* **conformal score** ``normalized`` vs ``absolute`` -- both fitted per file
  (separate records, distinguished by the record's ``score`` field).
* **field aggregation** ``max`` vs ``quantile(agg_q)`` -- both fitted per file
  (granularities ``field_max`` / ``field_quantile``).

CPU-light by mandate (a user Fluent job may be running): ``num_workers=0``, small
batches, ``torch.no_grad``; the P2000 GPU is used for the forward passes when
available.  Nothing here trains, and it never touches training code, models or
physics -- it only *reads* checkpoints and *calls* the frozen integrator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.uq.conformal import DEFAULT_EPS  # noqa: E402
from src.uq.coverage import (  # noqa: E402
    DEFAULT_LEVELS,
    make_uq_record,
    make_uq_report,
    sweep_levels_coefficients,
    sweep_levels_field,
    write_uq_report,
)
from src.uq.ensembles import load_ensemble_checkpoints  # noqa: E402

__all__ = [
    "DEFAULT_MODELS",
    "DEFAULT_SPLITS",
    "find_seed_checkpoints",
    "k_subsets",
    "ensemble_mean_std_axis0",
    "infer_dataset",
    "build_coefficient_records",
    "build_field_records",
    "run_one",
    "main",
]

#: The three surrogates and the four C3 splits (PLAN_PHASE2 Phase A).
DEFAULT_MODELS: tuple[str, ...] = ("gnn", "sdf_fno", "transolver")
DEFAULT_SPLITS: tuple[str, ...] = ("full", "reynolds", "aoa", "shape5")

#: Coefficient targets, each a ``(name, member-array accessor)`` pair.  The
#: member arrays are ``(K, N, 2)`` with column 0 = CL, column 1 = CD.
_COEF_TARGETS = (
    ("cd_int", "coef_int", 1),
    ("cl_int", "coef_int", 0),
    ("cd_head", "coef_head", 1),
    ("cl_head", "coef_head", 0),
)


# --------------------------------------------------------------------------- #
# checkpoint / config discovery
# --------------------------------------------------------------------------- #
def find_seed_checkpoints(
    model: str,
    split: str,
    seeds: Sequence[int],
    ckpt_root: str | Path = "checkpoints",
) -> list[tuple[int, Path]]:
    """``[(seed, best.pt), ...]`` for every requested seed that has a checkpoint.

    Falls back to ``last.pt`` when ``best.pt`` is absent (a run interrupted
    before its first improvement still carries a usable ``last.pt``).
    """
    root = Path(ckpt_root)
    found: list[tuple[int, Path]] = []
    for seed in seeds:
        run_id = f"{model}_{split}_s{seed}"
        best = root / run_id / "best.pt"
        if best.is_file():
            found.append((int(seed), best))
            continue
        last = root / run_id / "last.pt"
        if last.is_file():
            found.append((int(seed), last))
    return found


def load_run_config(
    model: str,
    split: str,
    seeds: Sequence[int],
    results_root: str | Path = "results",
) -> Mapping[str, Any]:
    """Read a run's ``config.yaml`` (any available seed) for the architecture."""
    from src.utils.config import load_config

    root = Path(results_root)
    for seed in list(seeds) + [0]:
        cfg_path = root / f"{model}_{split}_s{seed}" / "config.yaml"
        if cfg_path.is_file():
            return load_config(cfg_path, [])
    raise FileNotFoundError(
        f"no config.yaml for {model}_{split} under {root} (looked at seeds "
        f"{list(seeds)} + 0)"
    )


def k_subsets(k_available: int, requested: Sequence[int] | None) -> list[int]:
    """K subset sizes to emit: ``requested`` capped at K, always including K.

    Default (``requested is None``) is ``{1, 3, 5} intersect [1, K]`` plus K.
    """
    if requested is None:
        base = {1, 3, 5}
    else:
        base = {int(x) for x in requested}
    ks = {k for k in base if 1 <= k <= k_available}
    ks.add(int(k_available))
    return sorted(ks)


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def _norm_stats_for_split(split: str, processed_dir: Path, splits_dir: Path) -> dict:
    """Per-split **train-only** normalization stats (D-021), computed locally."""
    from scripts.build_cache import compute_norm_stats

    train_sims = json.loads((splits_dir / f"{split}.json").read_text())["train"]
    return compute_norm_stats(train_sims, processed_dir, split_name=split)


def build_dataset(
    split: str,
    subset: str,
    processed_dir: Path,
    splits_dir: Path,
    stats: Any,
):
    """One ``AirfransSurfaceDataset`` for ``(split, subset)`` with fixed stats.

    ``stats`` are the *model's own* split-train stats, passed explicitly so the
    transfer mode (``full`` cal served through a split-X model) still sees
    inputs normalized exactly as that model was trained.  ``missing='skip'`` so
    a partial local cache still yields whatever sims are present.
    """
    from src.data.airfrans_loader import AirfransSurfaceDataset

    return AirfransSurfaceDataset(
        split_file=splits_dir / f"{split}.json",
        processed_dir=processed_dir,
        normalize_stats=stats,
        subset=subset,
        load_volume=False,
        cache_in_ram=True,
        missing="skip",
    )


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def ensemble_mean_std_axis0(members: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and (ddof=1) std over the leading member axis; std=0 when K==1."""
    arr = np.asarray(members, dtype=np.float64)
    k = arr.shape[0]
    mean = arr.mean(axis=0)
    std = np.zeros_like(mean) if k < 2 else arr.std(axis=0, ddof=1)
    return mean, std


def infer_dataset(
    predictor,
    dataset,
    *,
    collate: Callable | None,
    denorm: Callable[[str, Any], Any],
    integrate: Callable[..., Mapping[str, Any]],
    device: Any,
    batch_size: int = 8,
    rho: float = 1.0,
    a_ref: float = 1.0,
) -> dict[str, Any]:
    """Run every ensemble member over ``dataset``; return per-sim member arrays.

    Returns a dict with

    * ``coef_int``  : ``(K, N, 2)`` integrated (CL, CD) per member per sim;
    * ``coef_head`` : ``(K, N, 2)`` direct-head (CL, CD) per member per sim;
    * ``p_members`` : list of N arrays ``(K, n_pts_i)`` denormalized pressure;
    * ``p_true``    : list of N arrays ``(n_pts_i,)`` physical truth pressure;
    * ``cl_true`` / ``cd_true`` : ``(N,)`` physical truth coefficients;
    * ``sim_names`` : list of N names.

    Each model is run exactly once per batch (``member_outputs(raw=True)``); the
    integrator is the frozen ``integrate_forces`` called per member so the
    ensemble spread is *integrate-then-average* (spec 5.6), not the other way.
    """
    import torch
    from torch.utils.data import DataLoader

    from src.eval.harness import batch_get, batch_to

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    coef_int_batches: list[np.ndarray] = []
    coef_head_batches: list[np.ndarray] = []
    p_members: list[np.ndarray] = []
    p_true: list[np.ndarray] = []
    cl_true: list[float] = []
    cd_true: list[float] = []
    sim_names: list[str] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch_to(batch, device)
            bidx = batch_get(batch, "batch_idx")
            normal = batch_get(batch, "surf_normal")
            ds = batch_get(batch, "surf_ds")
            cond = batch_get(batch, "cond_phys")
            if cond is None:
                cond = denorm("cond", batch_get(batch, "cond"))

            n = _n_graphs(batch, bidx)
            if bidx is None:
                bidx = torch.zeros(
                    normal.shape[0], dtype=torch.long, device=normal.device
                )

            outs = predictor.member_outputs(batch, raw=True)
            kk = len(outs)

            ci = np.empty((kk, n, 2), dtype=np.float64)   # (CL, CD)
            ch = np.empty((kk, n, 2), dtype=np.float64)
            p_stack = np.empty((kk, int(normal.shape[0])), dtype=np.float64)
            for m, out in enumerate(outs):
                p = out["p"].reshape(-1)
                tau = out["tau"]
                p_phys = denorm("surf_p", p)
                tau_phys = denorm("surf_tau", tau)
                forces = integrate(
                    p_phys, tau_phys, normal, ds, cond,
                    batch_idx=bidx, rho=rho, a_ref=a_ref,
                )
                ci[m, :, 0] = _np(forces["cl"]).reshape(-1)
                ci[m, :, 1] = _np(forces["cd"]).reshape(-1)
                coef = out["coef_head"].reshape(n, -1)
                ch[m, :, 0] = _np(denorm("cl_true", coef[:, 0])).reshape(-1)
                ch[m, :, 1] = _np(denorm("cd_true", coef[:, 1])).reshape(-1)
                p_stack[m, :] = _np(p_phys).reshape(-1)

            coef_int_batches.append(ci)
            coef_head_batches.append(ch)

            bidx_np = _np(bidx).astype(np.int64).reshape(-1)
            p_true_all = _np(_phys(batch, "surf_p", denorm)).reshape(-1)
            names = _sim_names(batch, n)
            cl_t = _as_vec(_np(_phys(batch, "cl_true", denorm)), n)
            cd_t = _as_vec(_np(_phys(batch, "cd_true", denorm)), n)
            for i in range(n):
                mask = bidx_np == i
                p_members.append(p_stack[:, mask])
                p_true.append(p_true_all[mask])
                cl_true.append(float(cl_t[i]))
                cd_true.append(float(cd_t[i]))
                sim_names.append(names[i])

    return {
        "coef_int": (
            np.concatenate(coef_int_batches, axis=1)
            if coef_int_batches else np.zeros((0, 0, 2))
        ),
        "coef_head": (
            np.concatenate(coef_head_batches, axis=1)
            if coef_head_batches else np.zeros((0, 0, 2))
        ),
        "p_members": p_members,
        "p_true": p_true,
        "cl_true": np.asarray(cl_true, dtype=np.float64),
        "cd_true": np.asarray(cd_true, dtype=np.float64),
        "sim_names": sim_names,
    }


# --------------------------------------------------------------------------- #
# record assembly
# --------------------------------------------------------------------------- #
def _scores_for_k(kk: int) -> tuple[str, ...]:
    """K=1 -> absolute only (sigma is identically zero); K>1 -> both."""
    return ("absolute",) if kk < 2 else ("normalized", "absolute")


def build_coefficient_records(
    cal: Mapping[str, Any],
    test: Mapping[str, Any],
    kk: int,
    *,
    levels: Sequence[float],
) -> list[dict[str, Any]]:
    """Coefficient records for one K subset: 4 targets x {scores}."""
    records: list[dict[str, Any]] = []
    for name, key, col in _COEF_TARGETS:
        cal_arr = np.asarray(cal[key])[:kk, :, col]     # (kk, N_cal)
        test_arr = np.asarray(test[key])[:kk, :, col]   # (kk, N_test)
        cal_mu, cal_sig = ensemble_mean_std_axis0(cal_arr)
        test_mu, test_sig = ensemble_mean_std_axis0(test_arr)
        cal_y = cal["cd_true"] if name.startswith("cd") else cal["cl_true"]
        test_y = test["cd_true"] if name.startswith("cd") else test["cl_true"]
        for score in _scores_for_k(kk):
            cs = None if score == "absolute" else cal_sig
            ts = None if score == "absolute" else test_sig
            rows = sweep_levels_coefficients(
                cal_mu, cs, cal_y, test_mu, ts, test_y,
                levels=levels, score=score, per_target=True,
                target_names=(name,),
            )
            records.append(
                make_uq_record(name, "coefficient", rows, score=score,
                               n_cal=int(cal_mu.shape[0]))
            )
    return records


def build_field_records(
    cal: Mapping[str, Any],
    test: Mapping[str, Any],
    kk: int,
    *,
    levels: Sequence[float],
    agg_q: float = 0.9,
) -> list[dict[str, Any]]:
    """Field ``p`` records for one K subset: {max, quantile} x {scores}."""
    def reduce_sims(store: Mapping[str, Any]):
        mus, sigs = [], []
        for m in store["p_members"]:
            mu, sig = ensemble_mean_std_axis0(np.asarray(m)[:kk, :])
            mus.append(mu)
            sigs.append(sig)
        return mus, sigs, list(store["p_true"])

    cal_mu, cal_sig, cal_y = reduce_sims(cal)
    test_mu, test_sig, test_y = reduce_sims(test)

    records: list[dict[str, Any]] = []
    for aggregation in ("max", "quantile"):
        for score in _scores_for_k(kk):
            cs = None if score == "absolute" else cal_sig
            ts = None if score == "absolute" else test_sig
            rows = sweep_levels_field(
                cal_mu, cs, cal_y, test_mu, ts, test_y,
                levels=levels, score=score, aggregation=aggregation, agg_q=agg_q,
            )
            records.append(
                make_uq_record(
                    "p", f"field_{aggregation}", rows, score=score,
                    n_cal=len(cal_mu),
                    agg_q=agg_q if aggregation == "quantile" else None,
                )
            )
    return records


# --------------------------------------------------------------------------- #
# driver for one (model, split)
# --------------------------------------------------------------------------- #
def run_one(
    model: str,
    split: str,
    seeds: Sequence[int],
    *,
    processed_dir: Path,
    splits_dir: Path,
    ckpt_root: Path,
    results_root: Path,
    out_dir: Path,
    modes: Sequence[str] = ("matched", "transfer"),
    levels: Sequence[float] = DEFAULT_LEVELS,
    requested_k: Sequence[int] | None = None,
    agg_q: float = 0.9,
    batch_size: int = 8,
    device_spec: str = "auto",
    log: Callable[[str], None] = print,
) -> list[Path]:
    """Produce every ``results/uq/*.json`` for one ``(model, split)``.

    Returns the list of written report paths (possibly empty if no checkpoints).
    """
    import torch

    from src.physics.force_integration import integrate_forces

    found = find_seed_checkpoints(model, split, seeds, ckpt_root)
    if not found:
        log(f"[uq] {model}_{split}: no checkpoints for seeds {list(seeds)} -- skipped")
        return []
    used_seeds = [s for s, _ in found]
    paths = [p for _, p in found]
    k_available = len(paths)
    log(f"[uq] {model}_{split}: K={k_available} checkpoints (seeds {used_seeds})")

    cfg = load_run_config(model, split, seeds, results_root)
    device = _resolve_device(device_spec)

    predictor = load_ensemble_checkpoints(paths, cfg, map_location="cpu")
    for m in predictor.models:
        try:
            m.to(device)
        except Exception:  # pragma: no cover - non-module model
            pass

    stats = _norm_stats_for_split(split, processed_dir, splits_dir)
    from src.data.airfrans_loader import collate as _collate

    test_ds = build_dataset(split, "test", processed_dir, splits_dir, stats)
    collate = getattr(test_ds, "collate", None) or _collate

    def denorm(name: str, tensor):
        # Always the split-X dataset's transform: in transfer mode the full-cal
        # sims are served *through* a split-X model, so their predictions must
        # be denormalized with split-X stats (matching how the model trained).
        return test_ds.denormalize(name, tensor)

    infer_kw = dict(
        collate=collate, denorm=denorm, integrate=integrate_forces,
        device=device, batch_size=batch_size,
    )

    if len(test_ds) == 0:
        log(f"[uq] {model}_{split}: empty test set -- skipped")
        return []
    log(f"[uq] {model}_{split}: inference over test ({len(test_ds)} sims)")
    test = infer_dataset(predictor, test_ds, **infer_kw)

    # cal sources: matched = split's own cal; transfer = full's cal (X stats).
    cal_cache: dict[str, dict[str, Any]] = {}

    def get_cal(cal_split: str) -> dict[str, Any] | None:
        if cal_split in cal_cache:
            return cal_cache[cal_split]
        cal_ds = build_dataset(cal_split, "cal", processed_dir, splits_dir, stats)
        if len(cal_ds) == 0:
            cal_cache[cal_split] = None  # type: ignore[assignment]
            return None
        log(f"[uq] {model}_{split}: inference over {cal_split} cal ({len(cal_ds)} sims)")
        out = infer_dataset(predictor, cal_ds, **infer_kw)
        cal_cache[cal_split] = out
        return out

    ksets = k_subsets(k_available, requested_k)
    written: list[Path] = []
    mode_calsplit = {"matched": split, "transfer": "full"}

    for mode in modes:
        cal_split = mode_calsplit.get(mode)
        if cal_split is None:
            continue
        cal = get_cal(cal_split)
        if cal is None:
            log(f"[uq] {model}_{split}: no {cal_split} cal -- {mode} skipped")
            continue
        for kk in ksets:
            records = build_coefficient_records(cal, test, kk, levels=levels)
            records += build_field_records(cal, test, kk, levels=levels, agg_q=agg_q)
            suffix = "" if mode == "matched" else "_calfull"
            ensemble_id = f"{model}_{split}_k{kk}{suffix}"
            report = make_uq_report(
                ensemble_id, split, records,
                model=model, seeds=used_seeds[:kk], k=kk, method="conformal",
                extra={
                    "mode": mode,
                    "cal_split": cal_split,
                    "n_test": int(len(test["sim_names"])),
                    "agg_q": float(agg_q),
                },
            )
            path = write_uq_report(report, out_dir)
            written.append(path)
            log(f"[uq]   wrote {path.name}  ({len(records)} records)")

    return written


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


def _n_graphs(batch, bidx) -> int:
    from src.eval.harness import batch_get

    ng = batch_get(batch, "num_graphs")
    if ng is not None:
        try:
            return int(ng)
        except (TypeError, ValueError):
            pass
    if bidx is not None and getattr(bidx, "numel", lambda: 0)():
        return int(bidx.max().item()) + 1
    names = batch_get(batch, "sim_name")
    if isinstance(names, (list, tuple)):
        return len(names)
    return 1


def _sim_names(batch, n: int) -> list[str]:
    from src.eval.harness import batch_get

    names = batch_get(batch, "sim_name")
    if isinstance(names, str):
        return [names]
    if isinstance(names, (list, tuple)) and len(names) == n:
        return [str(x) for x in names]
    return [f"sim{i}" for i in range(n)]


def _phys(batch, key: str, denorm):
    from src.eval.harness import batch_get

    val = batch_get(batch, f"{key}_phys")
    if val is not None:
        return val
    val = batch_get(batch, key)
    return None if val is None else denorm(key, val)


def _as_vec(value, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == n:
        return arr
    if arr.size == 1:
        return np.repeat(arr, n)
    return arr


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in str(text).replace(",", " ").split()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts/run_uq.py",
        description="Fit split conformal on ensemble checkpoints -> results/uq/*.json (C3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default=None,
                   help="gnn | sdf_fno | transolver (omit with --all)")
    p.add_argument("--split", default=None,
                   help="full | reynolds | aoa | shape5 | ... (omit with --all)")
    p.add_argument("--all", action="store_true",
                   help=f"run every model x split in {DEFAULT_MODELS} x {DEFAULT_SPLITS}")
    p.add_argument("--models", default=None,
                   help="comma list overriding --all's model set")
    p.add_argument("--splits", default=None,
                   help="comma list overriding --all's split set")
    p.add_argument("--seeds", default="0,1,2,3,4",
                   help="ensemble seeds to look for (default 0,1,2,3,4)")
    p.add_argument("--modes", default="matched,transfer",
                   help="matched,transfer (default both)")
    p.add_argument("--levels", nargs="+", type=float, default=list(DEFAULT_LEVELS),
                   help="nominal coverage levels (default 0.8 0.9 0.95)")
    p.add_argument("--k-subsets", default=None,
                   help="ensemble-size ablation sizes, e.g. 1,3,5 "
                        "(default: {1,3,5} capped at K, plus K)")
    p.add_argument("--agg-q", type=float, default=0.9,
                   help="field quantile-aggregation fraction (default 0.9)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="auto", help="cpu | cuda | auto")
    p.add_argument("--processed-dir", default="data/processed/airfrans")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--checkpoints-root", default="checkpoints")
    p.add_argument("--results-root", default="results")
    p.add_argument("--out", default="results/uq")
    return p


def _targets(args) -> list[tuple[str, str]]:
    if args.all or args.models or args.splits:
        models = (args.models.split(",") if args.models else list(DEFAULT_MODELS))
        splits = (args.splits.split(",") if args.splits else list(DEFAULT_SPLITS))
        return [(m.strip(), s.strip()) for m in models for s in splits]
    if not (args.model and args.split):
        raise SystemExit("give --model and --split, or --all (optionally --models/--splits)")
    return [(args.model, args.split)]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = _parse_int_list(args.seeds)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    requested_k = _parse_int_list(args.k_subsets) if args.k_subsets else None

    common = dict(
        seeds=seeds,
        processed_dir=Path(args.processed_dir),
        splits_dir=Path(args.splits_dir),
        ckpt_root=Path(args.checkpoints_root),
        results_root=Path(args.results_root),
        out_dir=Path(args.out),
        modes=modes,
        levels=args.levels,
        requested_k=requested_k,
        agg_q=args.agg_q,
        batch_size=args.batch_size,
        device_spec=args.device,
    )

    written: list[Path] = []
    for model, split in _targets(args):
        try:
            written += run_one(model, split, **common)
        except FileNotFoundError as exc:
            print(f"[uq] {model}_{split}: {exc}")
        except Exception as exc:  # keep the batch going; report and move on
            print(f"[uq] {model}_{split}: FAILED ({type(exc).__name__}: {exc})")

    print(f"\n{len(written)} UQ report(s) written to {args.out}:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
