"""Active-learning acquisition score (spec section 5.8).

    a(g, c) = z(sigma_CD) + beta * z(FSC) + gamma * z(L_sym)

with all three components "normalized to unit variance over the pool" and
``beta = gamma = 1`` by default plus a sensitivity check.  ``z`` here is the
standard shift-and-scale ``(x - mean) / std`` over the pool, which makes the
score **invariant to any positive affine rescaling of a raw component** -- the
property that lets us combine a drag-coefficient spread (order 1e-3), a force
self-consistency gap (order 1e-2) and a symmetry residual (dimensionless) in one
sum without one of them silently dominating.  That invariance is asserted in
``tests/test_active.py``.

All three components are computable **without ground truth** (ensemble spread,
force self-consistency, symmetry violation), which is what makes the score usable
on an unlabelled pool of proposed geometries.

Baselines for the falsifiable claim ("high-acquisition cases really do have
higher surrogate error than random ones"): :func:`random_scores` and
:func:`variance_only_scores`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "zscore",
    "AcquisitionResult",
    "acquisition_score",
    "rank_pool",
    "select_top_k",
    "random_scores",
    "variance_only_scores",
    "select_cases",
    "write_case_list",
]


def zscore(x: Any, *, ddof: int = 0, eps: float = 1e-12) -> np.ndarray:
    """Unit-variance normalization over the pool: ``(x - mean) / std``.

    A constant component (``std == 0``) maps to all zeros: it carries no
    information about which case to pick, so it must contribute nothing rather
    than blow up to NaN/inf.

    Invariance: for ``a > 0``, ``zscore(a*x + b) == zscore(x)`` up to floating
    point.  For ``a < 0`` the sign flips, as it must.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("empty score array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("acquisition component contains non-finite values")
    std = arr.std(ddof=ddof)
    if std <= eps * max(1.0, float(np.abs(arr).max())):
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


@dataclass
class AcquisitionResult:
    """Scores plus the normalized components, for the sensitivity study."""

    scores: np.ndarray
    components: dict[str, np.ndarray]
    normalized: dict[str, np.ndarray]
    beta: float
    gamma: float

    @property
    def n(self) -> int:
        return int(self.scores.size)

    def order(self) -> np.ndarray:
        """Pool indices sorted by descending score (ties broken by index)."""
        return np.lexsort((np.arange(self.scores.size), -self.scores))

    def top_k(self, k: int) -> np.ndarray:
        return self.order()[: int(k)]

    def as_table(self) -> list[dict[str, float]]:
        rows = []
        for i in range(self.n):
            row: dict[str, float] = {"index": i, "score": float(self.scores[i])}
            for name, vals in self.components.items():
                row[name] = float(vals[i])
                row[f"z_{name}"] = float(self.normalized[name][i])
            rows.append(row)
        return rows


def acquisition_score(
    sigma_cd: Any,
    fsc: Any | None = None,
    l_sym: Any | None = None,
    *,
    beta: float = 1.0,
    gamma: float = 1.0,
    ddof: int = 0,
    return_result: bool = False,
) -> np.ndarray | AcquisitionResult:
    """``a = z(sigma_CD) + beta*z(FSC) + gamma*z(L_sym)`` over a pool.

    Parameters
    ----------
    sigma_cd : ``(n,)`` ensemble standard deviation of the integrated drag
        coefficient over the pool (from
        ``EnsemblePredictor.propagate_forces(...)["cd"].std``).
    fsc : ``(n,)`` force self-consistency gap ``|C_int - C_head|`` (spec 5.4);
        omit (``None``) to drop the term.
    l_sym : ``(n,)`` symmetry residual (spec 5.5); omit to drop the term.
    beta, gamma : component weights; both 1.0 by default per the spec.

    Returns the ``(n,)`` score array, or the full :class:`AcquisitionResult`
    when ``return_result=True``.
    """
    sig = np.asarray(sigma_cd, dtype=np.float64).ravel()
    n = sig.size
    if n == 0:
        raise ValueError("empty pool")
    components: dict[str, np.ndarray] = {"sigma_cd": sig}
    weights: dict[str, float] = {"sigma_cd": 1.0}
    if fsc is not None:
        arr = np.asarray(fsc, dtype=np.float64).ravel()
        if arr.size != n:
            raise ValueError(f"fsc has {arr.size} entries, expected {n}")
        components["fsc"] = arr
        weights["fsc"] = float(beta)
    if l_sym is not None:
        arr = np.asarray(l_sym, dtype=np.float64).ravel()
        if arr.size != n:
            raise ValueError(f"l_sym has {arr.size} entries, expected {n}")
        components["l_sym"] = arr
        weights["l_sym"] = float(gamma)

    normalized = {name: zscore(vals, ddof=ddof) for name, vals in components.items()}
    scores = np.zeros(n, dtype=np.float64)
    for name, z in normalized.items():
        scores = scores + weights[name] * z

    if return_result:
        return AcquisitionResult(
            scores=scores,
            components=components,
            normalized=normalized,
            beta=float(beta),
            gamma=float(gamma),
        )
    return scores


def rank_pool(scores: Any) -> np.ndarray:
    """Pool indices sorted by descending score, ties broken by index."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    return np.lexsort((np.arange(s.size), -s))


def select_top_k(scores: Any, k: int) -> np.ndarray:
    """Pure exploitation baseline: the k highest-scoring pool indices."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    k = int(k)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k > s.size:
        raise ValueError(f"cannot select k={k} from a pool of {s.size}")
    return rank_pool(s)[:k]


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def random_scores(n: int, seed: int | np.random.Generator | None = 0) -> np.ndarray:
    """Random-selection baseline: i.i.d. uniform scores (seeded, reproducible)."""
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    if n <= 0:
        raise ValueError("n must be positive")
    return rng.random(int(n))


def variance_only_scores(sigma_cd: Any, *, ddof: int = 0) -> np.ndarray:
    """Pure ensemble-variance baseline: ``z(sigma_CD)`` alone (beta = gamma = 0)."""
    return zscore(sigma_cd, ddof=ddof)


# --------------------------------------------------------------------------
# end-to-end selection
# --------------------------------------------------------------------------
def select_cases(
    entries: Sequence[Any],
    scores: Any,
    k: int,
    *,
    design: Any | None = None,
    diverse: bool = True,
    oversample: int = 4,
    method: str = "minmax",
    weights: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Pick ``k`` pool cases and return them as serializable dicts.

    ``entries`` are :class:`src.active.pool.PoolEntry` objects (or anything with
    ``as_dict``).  When ``diverse`` is True the picks go through farthest-point
    selection in normalized design space; ``design`` defaults to
    ``pool.design_matrix(entries)``.
    """
    from .diversity import select_top_k_diverse
    from .pool import design_matrix

    s = np.asarray(scores, dtype=np.float64).ravel()
    if len(entries) != s.size:
        raise ValueError(f"{len(entries)} entries vs {s.size} scores")
    if diverse:
        x = design_matrix(list(entries)) if design is None else np.asarray(design, dtype=np.float64)
        idx = select_top_k_diverse(s, x, k, oversample=oversample, method=method, weights=weights)  # type: ignore[arg-type]
    else:
        idx = select_top_k(s, k)

    out: list[dict[str, Any]] = []
    for rank, i in enumerate(idx):
        e = entries[int(i)]
        d = e.as_dict() if hasattr(e, "as_dict") else dict(e)
        d["rank"] = rank
        d["pool_index"] = int(i)
        d["score"] = float(s[int(i)])
        out.append(d)
    return out


def write_case_list(
    cases: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    strategy: str = "acquisition",
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write a selected case list (e.g. ``fluent/cases_to_run.json``)."""
    payload: dict[str, Any] = {
        "strategy": str(strategy),
        "n_cases": len(cases),
        "cases": [dict(c) for c in cases],
    }
    if meta:
        payload.update(dict(meta))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p
