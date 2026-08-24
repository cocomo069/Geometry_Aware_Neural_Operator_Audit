"""Greedy farthest-point diversity selection in normalized design space.

Spec section 5.8: "Select top-k subject to greedy farthest-point diversity
selection in normalized design space so we do not pick k near-duplicates."

The design space mixes quantities with wildly different units -- camber ~1e-2,
Reynolds number ~1e6, angle of attack ~1e1 -- so every distance computation is
preceded by an explicit normalization (min-max to the unit cube by default,
which is bounded and therefore robust to the heavy tails a z-score would let
through).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

__all__ = [
    "NormalizationStats",
    "normalize_design_space",
    "apply_normalization",
    "farthest_point_selection",
    "select_top_k_diverse",
    "greedy_score_diversity",
    "min_pairwise_distance",
    "coverage_radius",
]

Method = Literal["minmax", "zscore", "none"]


@dataclass(frozen=True)
class NormalizationStats:
    """Per-column shift/scale so new points map into the same space."""

    method: str
    shift: np.ndarray
    scale: np.ndarray
    columns: tuple[str, ...] | None = None

    def transform(self, x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != self.shift.shape[0]:
            raise ValueError(f"expected {self.shift.shape[0]} columns, got {arr.shape[1]}")
        return (arr - self.shift) / self.scale


def normalize_design_space(
    x: Any,
    method: Method = "minmax",
    *,
    columns: Sequence[str] | None = None,
    weights: Sequence[float] | None = None,
) -> tuple[np.ndarray, NormalizationStats]:
    """Map an ``(n, d)`` design matrix into a comparable space.

    ``minmax`` -> each column to ``[0, 1]``; ``zscore`` -> zero mean, unit std;
    ``none`` -> passthrough.  Constant columns (zero range/std) are mapped to
    zero rather than producing NaNs -- a dimension the pool does not vary in
    must not dominate or corrupt the distance.

    ``weights`` (optional, length ``d``) rescales columns *after* normalization,
    e.g. to make shape parameters count more than flow conditions.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"design matrix must be 2-D, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("empty design matrix")
    if not np.all(np.isfinite(arr)):
        raise ValueError("design matrix contains non-finite values")
    d = arr.shape[1]

    if method == "minmax":
        shift = arr.min(axis=0)
        scale = arr.max(axis=0) - shift
    elif method == "zscore":
        shift = arr.mean(axis=0)
        scale = arr.std(axis=0)
    elif method == "none":
        shift = np.zeros(d)
        scale = np.ones(d)
    else:
        raise ValueError(f"unknown normalization method {method!r}")

    degenerate = scale <= 0
    scale = np.where(degenerate, 1.0, scale)
    out = (arr - shift) / scale
    out[:, degenerate] = 0.0

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape[0] != d:
            raise ValueError(f"weights must have length {d}, got {w.shape[0]}")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        out = out * w
        scale = scale / np.where(w > 0, w, 1.0)

    stats = NormalizationStats(
        method=method,
        shift=shift,
        scale=scale,
        columns=tuple(columns) if columns else None,
    )
    return out, stats


def apply_normalization(x: Any, stats: NormalizationStats) -> np.ndarray:
    """Transform new points with previously fitted stats."""
    return stats.transform(x)


def farthest_point_selection(
    x: Any,
    k: int,
    *,
    start: int | Literal["first", "medoid"] = "medoid",
    already_selected: Sequence[int] | None = None,
) -> np.ndarray:
    """Greedy farthest-point (max-min) selection, returning ``k`` row indices.

    At each step the point maximizing the distance to the *nearest* already
    selected point is added.  This is the classic 2-approximation to the k-center
    problem and is what "greedy farthest-point diversity" means in the spec.

    ``start``:
      * an ``int``    -- seed with that row (use it to seed with the top-scoring
        candidate, which is how :func:`select_top_k_diverse` keeps the highest
        acquisition score in the selection);
      * ``"first"``   -- seed with row 0;
      * ``"medoid"``  -- seed with the row closest to the centroid (default:
        deterministic and independent of row order).

    ``already_selected`` seeds the min-distance field with points chosen
    earlier (e.g. the existing training set) without returning them.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"design matrix must be 2-D, got shape {arr.shape}")
    n = arr.shape[0]
    k = int(k)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k > n:
        raise ValueError(f"cannot select k={k} points from a pool of {n}")

    min_dist = np.full(n, np.inf)
    if already_selected:
        prior = arr[np.asarray(already_selected, dtype=np.int64)]
        for p in prior:
            min_dist = np.minimum(min_dist, np.linalg.norm(arr - p, axis=1))

    if isinstance(start, (int, np.integer)):
        first = int(start)
        if not (0 <= first < n):
            raise ValueError(f"start index {first} out of range for pool of {n}")
    elif start == "first":
        first = 0
    elif start == "medoid":
        centroid = arr.mean(axis=0)
        first = int(np.argmin(np.linalg.norm(arr - centroid, axis=1)))
    else:
        raise ValueError(f"unknown start {start!r}")

    selected = np.empty(k, dtype=np.int64)
    selected[0] = first
    min_dist = np.minimum(min_dist, np.linalg.norm(arr - arr[first], axis=1))
    min_dist[first] = -np.inf  # never re-select

    for i in range(1, k):
        nxt = int(np.argmax(min_dist))
        selected[i] = nxt
        min_dist = np.minimum(min_dist, np.linalg.norm(arr - arr[nxt], axis=1))
        min_dist[nxt] = -np.inf
    return selected


def select_top_k_diverse(
    scores: Any,
    x: Any,
    k: int,
    *,
    oversample: int = 4,
    method: Method = "minmax",
    weights: Sequence[float] | None = None,
    already_selected: Sequence[int] | None = None,
) -> np.ndarray:
    """Top-k by acquisition score, de-duplicated by farthest-point selection.

    Take the ``oversample * k`` highest-scoring candidates, then run
    farthest-point selection *within* that shortlist seeded by the single
    best-scoring candidate.  The result always contains the argmax of the score
    (so pure exploitation is never thrown away) while the remaining k-1 picks
    are spread across the shortlist.

    Returns indices into the **original** pool, ordered by selection step.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != s.shape[0]:
        raise ValueError(f"scores ({s.shape}) and design matrix ({arr.shape}) disagree")
    k = int(k)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k > s.shape[0]:
        raise ValueError(f"cannot select k={k} from a pool of {s.shape[0]}")
    if oversample < 1:
        raise ValueError("oversample must be >= 1")

    m = min(s.shape[0], max(k, int(oversample) * k))
    # Descending order, ties broken by index for determinism.
    order = np.lexsort((np.arange(s.shape[0]), -s))
    shortlist = order[:m]

    xn, _ = normalize_design_space(arr, method, weights=weights)
    sub = xn[shortlist]
    prior_local = None
    if already_selected:
        # keep only prior points that are in the shortlist; the rest are handled
        # by passing their coordinates through the same normalization below
        prior_local = [int(np.where(shortlist == i)[0][0]) for i in already_selected if i in set(shortlist.tolist())]

    picked_local = farthest_point_selection(sub, k, start=0, already_selected=prior_local)
    return shortlist[picked_local]


def greedy_score_diversity(
    scores: Any,
    x: Any,
    k: int,
    *,
    lam: float = 1.0,
    method: Method = "minmax",
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Alternative: explicit score/diversity trade-off, greedily maximized.

    At each step pick ``argmax_i [ z(score_i) + lam * z(min-dist to selected) ]``
    with both terms z-scored over the remaining candidates.  ``lam = 0`` reduces
    to pure top-k; large ``lam`` approaches pure farthest-point.  Provided for
    the sensitivity check the spec asks for; the default selector is
    :func:`select_top_k_diverse`.
    """
    from .acquisition import zscore  # local import: avoid a module cycle

    s = np.asarray(scores, dtype=np.float64).ravel()
    arr = np.asarray(x, dtype=np.float64)
    n = s.shape[0]
    if arr.shape[0] != n:
        raise ValueError("scores and design matrix disagree in length")
    k = int(k)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k > n:
        raise ValueError(f"cannot select k={k} from a pool of {n}")

    xn, _ = normalize_design_space(arr, method, weights=weights)
    zs = zscore(s)
    selected: list[int] = [int(np.argmax(zs))]
    min_dist = np.linalg.norm(xn - xn[selected[0]], axis=1)
    mask = np.ones(n, dtype=bool)
    mask[selected[0]] = False

    while len(selected) < k:
        cand = np.where(mask)[0]
        zd = zscore(min_dist[cand])
        obj = zs[cand] + float(lam) * zd
        nxt = int(cand[int(np.argmax(obj))])
        selected.append(nxt)
        mask[nxt] = False
        min_dist = np.minimum(min_dist, np.linalg.norm(xn - xn[nxt], axis=1))
    return np.asarray(selected, dtype=np.int64)


def min_pairwise_distance(x: Any, indices: Sequence[int] | None = None) -> float:
    """Smallest distance between any two of the given rows (diversity metric)."""
    arr = np.asarray(x, dtype=np.float64)
    if indices is not None:
        arr = arr[np.asarray(indices, dtype=np.int64)]
    n = arr.shape[0]
    if n < 2:
        return float("inf")
    d = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
    iu = np.triu_indices(n, k=1)
    return float(np.min(d[iu]))


def coverage_radius(x: Any, indices: Sequence[int]) -> float:
    """Max distance from any pool point to its nearest selected point.

    The k-center objective farthest-point selection approximates; lower is
    better coverage of the design space.
    """
    arr = np.asarray(x, dtype=np.float64)
    sel = arr[np.asarray(indices, dtype=np.int64)]
    if sel.shape[0] == 0:
        return float("inf")
    d = np.linalg.norm(arr[:, None, :] - sel[None, :, :], axis=-1)
    return float(np.max(np.min(d, axis=1)))
