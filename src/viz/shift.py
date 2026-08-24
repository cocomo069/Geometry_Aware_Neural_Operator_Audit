"""Shift magnitude ``Delta`` -- the x-axis of the paper's central figure.

Spec section 5.7 asks for, per split, "a shift magnitude ``Delta``, the
normalized distance of the test condition from training support", and then
error and coverage plotted against it.  This module is the one definition of
that quantity, so Figure 4, Figure 7 and Table 2 cannot drift apart.

Definition
----------
Every AirfRANS simulation name encodes its own design point (see
``src.data.splits.parse_sim_name``), so ``Delta`` needs no cached fields:

``d_axis(x)  = max(0, (x - hi_axis) / s_axis, (lo_axis - x) / s_axis)``

over the numeric design axes ``re``, ``aoa_deg`` and ``thickness``, where
``[lo, hi]`` is the **train** support (min/max over the split's train list) and
``s`` is the standard deviation of that axis over the reference population
(train + test by default).  A test point inside the training box contributes
zero on that axis; outside, it contributes its distance in units of the design
variable's own spread, which is what makes the axes commensurable.

The shape-family axis is categorical, not numeric: NACA 4-digit and 5-digit
series are different parameterisations (3 vs 4 camber parameters), so no
Euclidean distance between them is meaningful.  ``delta_shape`` is therefore
``1.0`` when the test simulation's series is absent from the training series
set and ``0.0`` otherwise -- one "unit of shift", on the same scale as one
standard deviation of a numeric axis.

The aggregate is the Euclidean norm over the per-axis terms::

    Delta = sqrt(delta_re^2 + delta_aoa^2 + delta_thickness^2 + delta_shape^2)

so a point in-support on every axis has ``Delta = 0`` and the ``combined``
split -- shifted on two axes at once -- lands further out than either of its
parents, which is the ordering the paper claims.

Ordinal fallback
----------------
When the split manifests are unavailable (a fresh checkout, or a results tree
copied without ``data/splits/``), :data:`SHIFT_RANK` gives the ordinal
``full < scarce < reynolds = aoa < shape5 < combined`` used by CONTEXT.md
section 5, and figures fall back to plotting against it with the axis relabelled
so no reader mistakes an ordinal for a metric distance.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "SHIFT_RANK",
    "SHIFT_RANK_LABELS",
    "NUMERIC_AXES",
    "AXIS_LABELS",
    "DELTA_COLUMNS",
    "design_frame",
    "train_support",
    "axis_scales",
    "shift_magnitude",
    "load_split_manifest",
    "split_shift_frame",
    "all_shift_frames",
    "attach_shift",
    "bin_by_shift",
]

#: Ordinal shift used when a numeric ``Delta`` is not derivable.
SHIFT_RANK: dict[str, float] = {
    "full": 0.0,
    "scarce": 1.0,
    "reynolds": 2.0,
    "aoa": 2.0,
    "shape5": 3.0,
    "combined": 4.0,
}

SHIFT_RANK_LABELS: dict[float, str] = {
    0.0: "full",
    1.0: "scarce",
    2.0: "Re / AoA",
    3.0: "shape5",
    4.0: "combined",
}

#: Numeric design axes over which out-of-support distance is measured.
NUMERIC_AXES: tuple[str, ...] = ("re", "aoa_deg", "thickness")

AXIS_LABELS: dict[str, str] = {
    "re": r"$\mathrm{Re}$",
    "aoa_deg": r"$\alpha$ [deg]",
    "thickness": "thickness [% chord]",
    "shape": "NACA series",
}

DELTA_COLUMNS: tuple[str, ...] = (
    "delta_re",
    "delta_aoa_deg",
    "delta_thickness",
    "delta_shape",
    "delta",
)

_DEFAULT_SPLITS_DIR = Path("data/splits")


def _parse(name: str):
    """``parse_sim_name`` with a lazy import so ``src.viz`` stays standalone."""
    from src.data.splits import parse_sim_name

    return parse_sim_name(name)


def design_frame(sim_names: Iterable[str], *, strict: bool = False) -> pd.DataFrame:
    """Design point of each simulation, parsed from its name.

    Returns columns ``sim_name, u_inf, aoa_deg, re, thickness, n_digits``.
    Names that do not parse are dropped (with a warning) unless ``strict``.
    """
    rows = []
    for name in sim_names:
        try:
            sim = _parse(str(name))
        except Exception as exc:  # malformed / non-AirfRANS name
            if strict:
                raise
            warnings.warn(f"unparseable sim name {name!r}: {exc}", RuntimeWarning, stacklevel=2)
            continue
        rows.append(
            {
                "sim_name": str(name),
                "u_inf": sim.u_inf,
                "aoa_deg": sim.aoa_deg,
                "re": sim.re,
                "thickness": sim.thickness,
                "n_digits": int(sim.n_digits),
            }
        )
    cols = ["sim_name", "u_inf", "aoa_deg", "re", "thickness", "n_digits"]
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols}).astype(
            {"sim_name": "object"}
        )
    return pd.DataFrame(rows)[cols]


def train_support(train: pd.DataFrame) -> dict[str, object]:
    """``{axis: (lo, hi)}`` over the numeric axes, plus the NACA series set."""
    support: dict[str, object] = {}
    for axis in NUMERIC_AXES:
        if train.empty or axis not in train.columns:
            support[axis] = (float("nan"), float("nan"))
            continue
        col = pd.to_numeric(train[axis], errors="coerce").dropna()
        support[axis] = (
            (float(col.min()), float(col.max())) if len(col) else (float("nan"), float("nan"))
        )
    digits = (
        set(int(d) for d in train["n_digits"].dropna().unique())
        if not train.empty and "n_digits" in train.columns
        else set()
    )
    support["n_digits"] = digits
    return support


def axis_scales(reference: pd.DataFrame) -> dict[str, float]:
    """Per-axis normaliser: the population standard deviation of that axis.

    A degenerate (zero or NaN) spread falls back to ``1.0`` so ``Delta`` stays
    finite; that only happens for a reference set with a single design point.
    """
    scales: dict[str, float] = {}
    for axis in NUMERIC_AXES:
        if reference.empty or axis not in reference.columns:
            scales[axis] = 1.0
            continue
        col = pd.to_numeric(reference[axis], errors="coerce").dropna()
        s = float(col.std(ddof=0)) if len(col) > 1 else 0.0
        scales[axis] = s if math.isfinite(s) and s > 0 else 1.0
    return scales


def _outside(value: float, lo: float, hi: float, scale: float) -> float:
    if not math.isfinite(value) or not math.isfinite(lo) or not math.isfinite(hi):
        return float("nan")
    return max(0.0, (value - hi) / scale, (lo - value) / scale)


def shift_magnitude(
    test_names: Sequence[str],
    train_names: Sequence[str],
    *,
    scales: Mapping[str, float] | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """Per-test-simulation shift magnitude relative to a training set.

    Parameters
    ----------
    test_names, train_names:
        Simulation names, e.g. the ``test`` and ``train`` lists of a split
        manifest.
    scales:
        Optional per-axis normalisers; defaults to the standard deviation over
        ``train + test`` so the quantity is reproducible from the split alone.

    Returns
    -------
    pandas.DataFrame
        ``sim_name`` + the design columns + :data:`DELTA_COLUMNS`.
    """
    test = design_frame(test_names, strict=strict)
    train = design_frame(train_names, strict=strict)
    reference = pd.concat([train, test], ignore_index=True) if len(train) else test
    scl = dict(axis_scales(reference)) if scales is None else dict(scales)
    support = train_support(train)

    out = test.copy()
    terms = []
    for axis in NUMERIC_AXES:
        lo, hi = support[axis]  # type: ignore[misc]
        s = float(scl.get(axis, 1.0)) or 1.0
        col = (
            pd.to_numeric(out[axis], errors="coerce").map(lambda v: _outside(v, lo, hi, s))
            if axis in out.columns and len(out)
            else pd.Series(dtype=float)
        )
        name = f"delta_{axis}"
        out[name] = col
        terms.append(name)

    train_digits = support["n_digits"]  # type: ignore[assignment]
    if len(out):
        out["delta_shape"] = out["n_digits"].map(
            lambda d: 0.0 if (not train_digits or int(d) in train_digits) else 1.0
        )
    else:
        out["delta_shape"] = pd.Series(dtype=float)
    terms.append("delta_shape")

    if len(out):
        sq = sum(pd.to_numeric(out[t], errors="coerce").fillna(0.0) ** 2 for t in terms)
        out["delta"] = np.sqrt(sq)
    else:
        out["delta"] = pd.Series(dtype=float)
    return out


def load_split_manifest(
    name: str, splits_dir: str | Path = _DEFAULT_SPLITS_DIR, *, strict: bool = False
) -> dict | None:
    """Read ``<splits_dir>/<name>.json``; ``None`` when absent or unreadable."""
    path = Path(splits_dir) / f"{name}.json"
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        if strict:
            raise ValueError(f"{path}: unreadable split manifest ({exc})") from exc
        warnings.warn(f"{path}: split manifest unavailable ({exc})", RuntimeWarning, stacklevel=2)
        return None
    if not isinstance(payload, Mapping):
        if strict:
            raise ValueError(f"{path}: split manifest is not a JSON object")
        return None
    return dict(payload)


def split_shift_frame(
    name: str,
    splits_dir: str | Path = _DEFAULT_SPLITS_DIR,
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """``shift_magnitude`` for one named split, with a ``split`` column added."""
    manifest = load_split_manifest(name, splits_dir, strict=strict)
    if manifest is None:
        return pd.DataFrame()
    frame = shift_magnitude(
        list(manifest.get("test", [])), list(manifest.get("train", [])), strict=strict
    )
    frame.insert(0, "split", name)
    return frame


def all_shift_frames(
    splits: Iterable[str] | None = None,
    splits_dir: str | Path = _DEFAULT_SPLITS_DIR,
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """Stack :func:`split_shift_frame` over every split that has a manifest."""
    names = list(splits) if splits is not None else list(SHIFT_RANK)
    frames = [f for f in (split_shift_frame(n, splits_dir, strict=strict) for n in names) if len(f)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def attach_shift(
    per_sim: pd.DataFrame,
    splits_dir: str | Path = _DEFAULT_SPLITS_DIR,
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """Left-join per-simulation ``Delta`` onto a per-sim frame.

    Joins on ``(split, sim_name)``.  Rows whose split has no manifest keep
    ``NaN`` deltas, which the figures treat as "numeric shift not derivable"
    and fall back to :data:`SHIFT_RANK`.
    """
    if per_sim.empty or "split" not in per_sim.columns or "sim_name" not in per_sim.columns:
        out = per_sim.copy()
        for col in DELTA_COLUMNS:
            if col not in out.columns:
                out[col] = float("nan")
        out["shift_rank"] = (
            out["split"].map(SHIFT_RANK) if "split" in out.columns else float("nan")
        )
        return out

    shifts = all_shift_frames(
        sorted(set(per_sim["split"].dropna().astype(str))), splits_dir, strict=strict
    )
    out = per_sim.copy()
    if shifts.empty:
        for col in DELTA_COLUMNS:
            out[col] = float("nan")
    else:
        keep = ["split", "sim_name", *DELTA_COLUMNS, "re", "aoa_deg", "thickness", "n_digits"]
        keep = [c for c in keep if c in shifts.columns]
        out = out.merge(
            shifts[keep].drop_duplicates(subset=["split", "sim_name"]),
            on=["split", "sim_name"],
            how="left",
            suffixes=("", "_design"),
        )
        for col in DELTA_COLUMNS:
            if col not in out.columns:
                out[col] = float("nan")
    out["shift_rank"] = out["split"].map(SHIFT_RANK)
    return out


def bin_by_shift(
    df: pd.DataFrame,
    value: str,
    *,
    delta_col: str = "delta",
    n_bins: int = 5,
    min_count: int = 3,
    agg: str = "median",
) -> pd.DataFrame:
    """Aggregate ``value`` into equal-count bins of ``delta``.

    Returns ``delta`` (bin centre, the median ``Delta`` inside the bin),
    ``value`` (the aggregate), ``lo``/``hi`` (inter-quartile spread inside the
    bin, for the shaded band) and ``n``.  Bins with fewer than ``min_count``
    members are dropped -- a one-simulation "trend" is noise, not a curve.

    Binning is on ``delta`` **values**, not on their ranks, with duplicate bin
    edges dropped.  That matters: an in-distribution reference set has
    ``Delta = 0`` for every simulation, and rank-binning would spread those
    identical points across several bins, drawing a spurious vertical "trend"
    at the origin.
    """
    cols = ["delta", "value", "lo", "hi", "n"]
    if df.empty or value not in df.columns or delta_col not in df.columns:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in cols})
    work = df[[delta_col, value]].apply(pd.to_numeric, errors="coerce").dropna()
    if work.empty:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in cols})

    n_bins = max(1, min(int(n_bins), max(1, len(work) // max(1, min_count))))
    try:
        codes = pd.qcut(work[delta_col], n_bins, labels=False, duplicates="drop")
    except (ValueError, IndexError):
        codes = pd.Series(np.zeros(len(work), dtype=int), index=work.index)
    codes = pd.Series(codes, index=work.index).fillna(-1)

    rows = []
    for _, grp in work.groupby(codes):
        if len(grp) < min_count:
            continue
        vals = grp[value]
        rows.append(
            {
                "delta": float(grp[delta_col].median()),
                "value": float(getattr(vals, agg)()),
                "lo": float(vals.quantile(0.25)),
                "hi": float(vals.quantile(0.75)),
                "n": int(len(grp)),
            }
        )
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in cols})
    return pd.DataFrame(rows)[cols].sort_values("delta").reset_index(drop=True)
