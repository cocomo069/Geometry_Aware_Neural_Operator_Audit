"""Evaluation metrics -- all consumed in **denormalized physical units**.

Definitions follow the spec (5.4, 8 "Metrics") and populate the frozen
``metrics.json`` blocks of CONTEXT.md 9.

Aggregation convention (important, and deliberately explicit):

* ``*_rel_l2`` / ``*_mae``  -- computed **per simulation**, then averaged over
  simulations with equal weight.  A simulation with more surface points must
  not dominate.
* ``cl_head_mae`` etc.      -- mean absolute error over simulations.
* ``cl_rel`` / ``cd_rel``   -- *aggregate* relative error
  ``sum|C_pred - C_true| / sum|C_true|``.  Per-sim ratios are unusable for
  :math:`C_L`, which crosses zero on symmetric airfoils near
  :math:`\\alpha = 0`; the aggregate form is stable and is what the CFD
  literature normally reports.  Computed on the **integrated** coefficient
  (the physics-grounded one); see :func:`coef_metrics`.
* ``fsc_rel_*``             -- per-sim ``|C_int - C_head| / (|C_true| + eps)``
  aggregated by **median**, again because of the zero crossing.

Inputs may be numpy arrays or torch tensors; everything is converted to
float64 numpy internally.  No torch import is required at module import time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _dc_field
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "EPS",
    "as_numpy",
    "rel_l2",
    "mae",
    "rmse",
    "spearman",
    "fsc",
    "fsc_rel",
    "nanmean",
    "nanmedian",
    "coef_metrics",
    "field_metrics",
    "SimRecord",
    "MetricAccumulator",
]

#: Guard for relative denominators (spec 5.4 writes this as epsilon).
EPS = 1.0e-8


def as_numpy(x: Any) -> np.ndarray:
    """Convert torch tensors / lists / scalars to a float64 numpy array."""
    if x is None:
        return np.asarray(np.nan, dtype=np.float64)
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Field metrics
# --------------------------------------------------------------------------- #
def rel_l2(pred: Any, true: Any, eps: float = EPS) -> float:
    """Relative :math:`L_2` error ``||pred - true||_2 / (||true||_2 + eps)``.

    Shapes must match exactly (no broadcasting: a silent shape coercion here
    would quietly change the metric).  Vector fields such as ``tau`` (Ns, 2)
    are flattened, i.e. the Frobenius norm is used.
    """
    p, t = as_numpy(pred), as_numpy(true)
    if p.shape != t.shape:
        raise ValueError(f"rel_l2: shape mismatch pred{p.shape} vs true{t.shape}")
    num = float(np.linalg.norm(p.ravel() - t.ravel()))
    den = float(np.linalg.norm(t.ravel()))
    return num / (den + eps)


def mae(pred: Any, true: Any) -> float:
    """Mean absolute error over all entries (vector fields flattened)."""
    p, t = as_numpy(pred), as_numpy(true)
    if p.shape != t.shape:
        raise ValueError(f"mae: shape mismatch pred{p.shape} vs true{t.shape}")
    return float(np.mean(np.abs(p - t)))


def rmse(pred: Any, true: Any) -> float:
    """Root mean squared error over all entries."""
    p, t = as_numpy(pred), as_numpy(true)
    if p.shape != t.shape:
        raise ValueError(f"rmse: shape mismatch pred{p.shape} vs true{t.shape}")
    return float(np.sqrt(np.mean((p - t) ** 2)))


def field_metrics(pred: Any, true: Any) -> dict[str, float]:
    """``{'rel_l2': ..., 'mae': ..., 'rmse': ...}`` for one field of one sim."""
    return {"rel_l2": rel_l2(pred, true), "mae": mae(pred, true), "rmse": rmse(pred, true)}


# --------------------------------------------------------------------------- #
# Rank correlation
# --------------------------------------------------------------------------- #
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks (a tiny stand-in for ``scipy.stats.rankdata``)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # average ties
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def spearman(x: Any, y: Any) -> float:
    """Spearman rank correlation.  Uses scipy when available, else ranks+Pearson.

    Returns ``nan`` for fewer than 3 finite pairs or a constant input (an
    undefined correlation must not be reported as 0.0).
    """
    a, b = as_numpy(x).ravel(), as_numpy(y).ravel()
    if a.shape != b.shape:
        raise ValueError(f"spearman: shape mismatch {a.shape} vs {b.shape}")
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float("nan")
    if a.min() == a.max() or b.min() == b.max():
        return float("nan")  # constant input: undefined, and scipy only warns
    try:
        from scipy.stats import spearmanr  # noqa: PLC0415

        rho = float(spearmanr(a, b).statistic)
        return rho
    except Exception:
        ra, rb = _rankdata(a), _rankdata(b)
        sa, sb = ra.std(), rb.std()
        if sa == 0.0 or sb == 0.0:
            return float("nan")
        return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


# --------------------------------------------------------------------------- #
# Force self-consistency (spec 5.4)
# --------------------------------------------------------------------------- #
def fsc(c_int: Any, c_head: Any) -> np.ndarray:
    """``|C_int - C_head|`` -- label-free self-inconsistency, per sim."""
    return np.abs(as_numpy(c_int) - as_numpy(c_head))


def fsc_rel(c_int: Any, c_head: Any, c_true: Any, eps: float = EPS) -> np.ndarray:
    """``|C_int - C_head| / (|C_true| + eps)`` per sim (spec 5.4)."""
    return fsc(c_int, c_head) / (np.abs(as_numpy(c_true)) + eps)


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def nanmean(values: Iterable[float]) -> float | None:
    """Mean of the finite entries; ``None`` if there are none (JSON-safe)."""
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)] if arr.size else arr
    return float(arr.mean()) if arr.size else None


def nanmedian(values: Iterable[float]) -> float | None:
    """Median of the finite entries; ``None`` if there are none."""
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)] if arr.size else arr
    return float(np.median(arr)) if arr.size else None


def _agg_rel(err: np.ndarray, true: np.ndarray, eps: float = EPS) -> float | None:
    ok = np.isfinite(err) & np.isfinite(true)
    if not ok.any():
        return None
    den = float(np.abs(true[ok]).sum())
    return float(np.abs(err[ok]).sum() / (den + eps))


def coef_metrics(
    cl_head: Sequence[float],
    cd_head: Sequence[float],
    cl_int: Sequence[float],
    cd_int: Sequence[float],
    cl_true: Sequence[float],
    cd_true: Sequence[float],
    *,
    spearman_source: str = "int",
) -> dict[str, float | None]:
    """Build the ``coef`` block of ``metrics.json`` from per-sim arrays.

    ``spearman_source`` selects which C_D prediction is ranked against truth:
    ``'int'`` (default, physics-grounded; falls back to the head when the
    integrated values are all non-finite) or ``'head'``.
    """
    clh, cdh = as_numpy(cl_head), as_numpy(cd_head)
    cli, cdi = as_numpy(cl_int), as_numpy(cd_int)
    clt, cdt = as_numpy(cl_true), as_numpy(cd_true)

    cd_rank = cdi if spearman_source == "int" else cdh
    if not np.isfinite(cd_rank).any():
        cd_rank = cdh

    return {
        "cl_head_mae": nanmean(np.abs(clh - clt)),
        "cd_head_mae": nanmean(np.abs(cdh - cdt)),
        "cl_int_mae": nanmean(np.abs(cli - clt)),
        "cd_int_mae": nanmean(np.abs(cdi - cdt)),
        "cd_spearman": spearman(cd_rank, cdt),
        "cd_head_spearman": spearman(cdh, cdt),
        "cd_int_spearman": spearman(cdi, cdt) if np.isfinite(cdi).any() else None,
        "cl_rel": _agg_rel(cli - clt, clt) if np.isfinite(cli).any() else _agg_rel(clh - clt, clt),
        "cd_rel": _agg_rel(cdi - cdt, cdt) if np.isfinite(cdi).any() else _agg_rel(cdh - cdt, cdt),
    }


# --------------------------------------------------------------------------- #
# Per-sim accumulator
# --------------------------------------------------------------------------- #
@dataclass
class SimRecord:
    """One simulation's denormalized predictions and truth (physical units)."""

    sim_name: str = ""
    p_rel_l2: float = math.nan
    tau_rel_l2: float = math.nan
    p_mae: float = math.nan
    tau_mae: float = math.nan
    cl_head: float = math.nan
    cd_head: float = math.nan
    cl_int: float = math.nan
    cd_int: float = math.nan
    cl_true: float = math.nan
    cd_true: float = math.nan
    sym_residual: float = math.nan
    antisym_cl_gap: float = math.nan


@dataclass
class MetricAccumulator:
    """Collect :class:`SimRecord` s and emit the frozen metric blocks.

    Every quantity is expected in denormalized physical units -- the harness
    denormalizes before calling :meth:`add`.
    """

    records: list[SimRecord] = _dc_field(default_factory=list)

    # ---- ingestion ------------------------------------------------------- #
    def add_record(self, rec: SimRecord) -> None:
        """Append a pre-built record."""
        self.records.append(rec)

    def add(
        self,
        *,
        sim_name: str = "",
        p_pred=None,
        p_true=None,
        tau_pred=None,
        tau_true=None,
        cl_head: float = math.nan,
        cd_head: float = math.nan,
        cl_int: float = math.nan,
        cd_int: float = math.nan,
        cl_true: float = math.nan,
        cd_true: float = math.nan,
        sym_residual: float = math.nan,
        antisym_cl_gap: float = math.nan,
    ) -> SimRecord:
        """Compute and store one simulation's metrics.

        Field arguments may be ``None`` when a model predicts no fields
        (e.g. the ridge baseline); the corresponding entries stay ``nan`` and
        are excluded from the averages.
        """
        rec = SimRecord(
            sim_name=sim_name,
            cl_head=float(cl_head),
            cd_head=float(cd_head),
            cl_int=float(cl_int),
            cd_int=float(cd_int),
            cl_true=float(cl_true),
            cd_true=float(cd_true),
            sym_residual=float(sym_residual),
            antisym_cl_gap=float(antisym_cl_gap),
        )
        if p_pred is not None and p_true is not None:
            rec.p_rel_l2 = rel_l2(p_pred, p_true)
            rec.p_mae = mae(p_pred, p_true)
        if tau_pred is not None and tau_true is not None:
            rec.tau_rel_l2 = rel_l2(tau_pred, tau_true)
            rec.tau_mae = mae(tau_pred, tau_true)
        self.records.append(rec)
        return rec

    # ---- reduction ------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.records)

    def _col(self, name: str) -> np.ndarray:
        return np.asarray([getattr(r, name) for r in self.records], dtype=np.float64)

    def field_block(self) -> dict[str, float | None]:
        """``metrics.json['field']`` -- per-sim metrics averaged over sims."""
        return {
            "p_rel_l2": nanmean(self._col("p_rel_l2")),
            "tau_rel_l2": nanmean(self._col("tau_rel_l2")),
            "p_mae": nanmean(self._col("p_mae")),
            "tau_mae": nanmean(self._col("tau_mae")),
        }

    def coef_block(self, *, spearman_source: str = "int") -> dict[str, float | None]:
        """``metrics.json['coef']``."""
        return coef_metrics(
            self._col("cl_head"),
            self._col("cd_head"),
            self._col("cl_int"),
            self._col("cd_int"),
            self._col("cl_true"),
            self._col("cd_true"),
            spearman_source=spearman_source,
        )

    def consistency_block(self) -> dict[str, float | None]:
        """``metrics.json['consistency']`` -- FSC (5.4) + symmetry checks (5.5)."""
        cli, cdi = self._col("cl_int"), self._col("cd_int")
        clh, cdh = self._col("cl_head"), self._col("cd_head")
        clt, cdt = self._col("cl_true"), self._col("cd_true")
        return {
            "fsc_cl": nanmean(fsc(cli, clh)),
            "fsc_cd": nanmean(fsc(cdi, cdh)),
            "fsc_rel_cl": nanmedian(fsc_rel(cli, clh, clt)),
            "fsc_rel_cd": nanmedian(fsc_rel(cdi, cdh, cdt)),
            "sym_residual": nanmean(self._col("sym_residual")),
            "antisym_cl_gap": nanmean(self._col("antisym_cl_gap")),
        }

    def per_sim_rows(self) -> list[dict[str, Any]]:
        """Per-simulation rows (for ``per_sim.csv`` and downstream figures)."""
        return [
            {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
             for k, v in vars(r).items()}
            for r in self.records
        ]
