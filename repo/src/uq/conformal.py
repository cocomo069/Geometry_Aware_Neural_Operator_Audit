"""Split conformal prediction (spec section 5.6).

Two granularities are calibrated **separately**, exactly as the spec demands:

1. ``SplitConformalCoefficients`` -- per-simulation coefficient intervals.
   The exchangeable unit is a whole simulation, so the marginal guarantee
   ``P(y in C_alpha(x)) >= 1 - alpha`` is clean.
2. ``SplitConformalField`` -- pointwise field intervals.  Points *within* one
   simulation are not exchangeable, so we conformalize at the **simulation**
   level by reducing each calibration simulation's per-point scores to a single
   number, either ``max`` over points or a ``quantile`` over points (q = 0.9 by
   default).  The resulting interval is a simultaneous band over the whole
   surface (``max``) or a "at least a q-fraction of points are covered" band
   (``quantile``); it is **not** a pointwise guarantee, and the returned object
   records which weaker interpretation applies.

Nonconformity scores (both granularities):

* ``"normalized"`` (default, spec):  s = |y - mu| / (sigma + eps)
* ``"absolute"``  (ablation, sigma == 1):  s = |y - mu|

Calibrated quantile: with n calibration units, ``q_hat`` is the
``ceil((n+1)(1-alpha)) / n`` empirical quantile of the scores, i.e. the
``k``-th smallest score with ``k = ceil((n+1)(1-alpha))``.  If ``k > n`` (too
few calibration points for the requested level) the quantile is ``+inf`` and
the intervals are infinite -- the honest answer, not a silent clamp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .ensembles import as_numpy

__all__ = [
    "DEFAULT_EPS",
    "conformal_quantile",
    "conformal_quantile_index",
    "nonconformity_scores",
    "aggregate_point_scores",
    "CalibratedIntervals",
    "SplitConformalCoefficients",
    "SplitConformalField",
    "split_conformal",
]

DEFAULT_EPS = 1e-8

ScoreName = Literal["normalized", "absolute"]
AggName = Literal["max", "quantile"]


# --------------------------------------------------------------------------
# quantile
# --------------------------------------------------------------------------
def conformal_quantile_index(n: int, alpha: float) -> int:
    """``k = ceil((n + 1) * (1 - alpha))`` -- the 1-based rank to take.

    Examples (the edge cases the tests pin):
        n=9,  alpha=0.10 -> ceil(10 * 0.90) = 9   (the max of 9 scores)
        n=10, alpha=0.10 -> ceil(11 * 0.90) = 10  (the max of 10 scores)
        n=8,  alpha=0.10 -> ceil(9  * 0.90) = 9   > n -> quantile is +inf
        n=19, alpha=0.10 -> ceil(20 * 0.90) = 18
        n=200,alpha=0.10 -> ceil(201 * 0.90) = 181
    """
    if n < 1:
        raise ValueError(f"need at least one calibration unit, got n={n}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    # Guard against binary-float noise in products like 201 * 0.9 = 180.90000000000003
    prod = (n + 1) * (1.0 - alpha)
    k = int(math.ceil(round(prod, 10)))
    return k


def conformal_quantile(scores: Any, alpha: float) -> float:
    """Empirical ``ceil((n+1)(1-alpha))/n`` quantile of a 1-D score array.

    Returns ``+inf`` when the calibration set is too small to certify ``alpha``.
    """
    s = np.asarray(as_numpy(scores), dtype=np.float64).ravel()
    if s.size == 0:
        raise ValueError("empty calibration score array")
    if not np.all(np.isfinite(s)):
        # Infinite scores are legal (sigma == 0 with a nonzero residual); NaN is not.
        if np.any(np.isnan(s)):
            raise ValueError("calibration scores contain NaN")
    n = s.size
    k = conformal_quantile_index(n, alpha)
    if k > n:
        return float("inf")
    part = np.partition(s, k - 1)
    return float(part[k - 1])


# --------------------------------------------------------------------------
# scores
# --------------------------------------------------------------------------
def nonconformity_scores(
    mu: Any,
    sigma: Any | None,
    y: Any,
    *,
    score: ScoreName = "normalized",
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """Elementwise nonconformity score.

    ``normalized``: ``|y - mu| / (sigma + eps)``  (spec section 5.6)
    ``absolute``  : ``|y - mu|``                  (sigma ignored; ablation)
    """
    mu_a = np.asarray(as_numpy(mu), dtype=np.float64)
    y_a = np.asarray(as_numpy(y), dtype=np.float64)
    if mu_a.shape != y_a.shape:
        raise ValueError(f"mu shape {mu_a.shape} != y shape {y_a.shape}")
    resid = np.abs(y_a - mu_a)
    if score == "absolute":
        return resid
    if score != "normalized":
        raise ValueError(f"unknown score {score!r}; use 'normalized' or 'absolute'")
    if sigma is None:
        raise ValueError("normalized score needs sigma (use score='absolute' for the sigma==1 ablation)")
    sig = np.asarray(as_numpy(sigma), dtype=np.float64)
    if sig.shape != mu_a.shape:
        raise ValueError(f"sigma shape {sig.shape} != mu shape {mu_a.shape}")
    if np.any(sig < 0):
        raise ValueError("sigma must be non-negative")
    return resid / (sig + eps)


def aggregate_point_scores(
    scores: np.ndarray,
    *,
    aggregation: AggName = "max",
    agg_q: float = 0.9,
) -> float:
    """Reduce one simulation's per-point scores to a single sim-level score.

    ``max``      -> simultaneous band over all surface points.
    ``quantile`` -> the ``agg_q`` quantile (``method='higher'`` so agg_q == 1.0
                    reduces exactly to ``max``); the band then covers at least
                    an ``agg_q`` fraction of the points of a new simulation.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size == 0:
        raise ValueError("simulation has zero surface points")
    if aggregation == "max":
        return float(np.max(s))
    if aggregation == "quantile":
        if not (0.0 < agg_q <= 1.0):
            raise ValueError(f"agg_q must lie in (0, 1], got {agg_q}")
        return float(np.quantile(s, agg_q, method="higher"))
    raise ValueError(f"unknown aggregation {aggregation!r}; use 'max' or 'quantile'")


def _as_sim_list(
    values: Any,
    sim_idx: Any | None,
    name: str,
) -> list[np.ndarray]:
    """Normalize per-simulation point data into a list of 1-D arrays.

    Accepts: list/tuple of per-sim arrays (ragged OK), a 2-D ``(n_sim, n_pts)``
    array, or a flat concatenated array together with ``sim_idx`` (the
    ``batch_idx`` convention of CONTEXT.md section 4).
    """
    if isinstance(values, (list, tuple)):
        out = [np.asarray(as_numpy(v), dtype=np.float64).ravel() for v in values]
        if not out:
            raise ValueError(f"{name} is empty")
        return out
    arr = np.asarray(as_numpy(values), dtype=np.float64)
    if sim_idx is not None:
        flat = arr.ravel()
        idx = np.asarray(as_numpy(sim_idx)).ravel().astype(np.int64)
        if idx.shape[0] != flat.shape[0]:
            raise ValueError(f"{name} has {flat.shape[0]} entries but sim_idx has {idx.shape[0]}")
        order = np.unique(idx)
        return [flat[idx == u] for u in order]
    if arr.ndim == 2:
        return [row.ravel() for row in arr]
    raise ValueError(
        f"{name} must be a list of per-sim arrays, a 2-D (n_sim, n_pts) array, "
        "or a flat array plus sim_idx"
    )


# --------------------------------------------------------------------------
# calibrated object
# --------------------------------------------------------------------------
@dataclass
class CalibratedIntervals:
    """Result of a split-conformal fit; call :meth:`predict` to get (lo, hi).

    Attributes
    ----------
    q_hat : scalar, or ``(d,)`` for per-target coefficient calibration.
    level : nominal coverage ``1 - alpha``.
    alpha : miscoverage rate.
    score : ``"normalized"`` or ``"absolute"``.
    granularity : ``"coefficient"``, ``"field_max"`` or ``"field_quantile"``.
    guarantee : plain-English statement of what the interval actually certifies.
    """

    q_hat: np.ndarray
    level: float
    alpha: float
    score: ScoreName = "normalized"
    eps: float = DEFAULT_EPS
    granularity: str = "coefficient"
    aggregation: str | None = None
    agg_q: float | None = None
    n_cal: int = 0
    target_names: tuple[str, ...] | None = None
    guarantee: str = ""
    cal_scores: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.q_hat = np.asarray(self.q_hat, dtype=np.float64)
        if not self.guarantee:
            self.guarantee = _guarantee_text(self.granularity, self.level, self.aggregation, self.agg_q)

    # -- use --------------------------------------------------------------
    def half_width(self, sigma: Any | None = None, shape: tuple[int, ...] | None = None) -> np.ndarray:
        """``q_hat * (sigma + eps)``, or just ``q_hat`` for the absolute score."""
        q = self.q_hat
        if self.score == "absolute":
            if sigma is not None:
                base = np.ones_like(np.asarray(as_numpy(sigma), dtype=np.float64))
            elif shape is not None:
                base = np.ones(shape, dtype=np.float64)
            else:
                base = np.asarray(1.0)
            return q * base
        if sigma is None:
            raise ValueError("normalized intervals need sigma at predict time")
        sig = np.asarray(as_numpy(sigma), dtype=np.float64)
        return q * (sig + self.eps)

    def predict(self, mu: Any, sigma: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lo, hi)`` arrays with the same shape as ``mu``."""
        mu_a = np.asarray(as_numpy(mu), dtype=np.float64)
        if sigma is not None:
            sig = np.asarray(as_numpy(sigma), dtype=np.float64)
            if sig.shape != mu_a.shape:
                raise ValueError(f"sigma shape {sig.shape} != mu shape {mu_a.shape}")
        half = self.half_width(sigma, shape=mu_a.shape)
        half = np.broadcast_to(np.asarray(half, dtype=np.float64), mu_a.shape)
        return mu_a - half, mu_a + half

    def __call__(self, mu: Any, sigma: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
        return self.predict(mu, sigma)

    def contains(self, y: Any, mu: Any, sigma: Any | None = None) -> np.ndarray:
        lo, hi = self.predict(mu, sigma)
        y_a = np.asarray(as_numpy(y), dtype=np.float64)
        return (y_a >= lo) & (y_a <= hi)

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        q = self.q_hat
        return {
            "q_hat": float(q) if q.ndim == 0 else [float(v) for v in q.ravel()],
            "level": float(self.level),
            "alpha": float(self.alpha),
            "score": self.score,
            "eps": float(self.eps),
            "granularity": self.granularity,
            "aggregation": self.aggregation,
            "agg_q": None if self.agg_q is None else float(self.agg_q),
            "n_cal": int(self.n_cal),
            "target_names": list(self.target_names) if self.target_names else None,
            "guarantee": self.guarantee,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CalibratedIntervals":
        names = d.get("target_names")
        return cls(
            q_hat=np.asarray(d["q_hat"], dtype=np.float64),
            level=float(d["level"]),
            alpha=float(d["alpha"]),
            score=d.get("score", "normalized"),
            eps=float(d.get("eps", DEFAULT_EPS)),
            granularity=d.get("granularity", "coefficient"),
            aggregation=d.get("aggregation"),
            agg_q=d.get("agg_q"),
            n_cal=int(d.get("n_cal", 0)),
            target_names=tuple(names) if names else None,
            guarantee=d.get("guarantee", ""),
        )


def _guarantee_text(granularity: str, level: float, aggregation: str | None, agg_q: float | None) -> str:
    lv = f"{level:.3g}"
    if granularity == "coefficient":
        return (
            f"Marginal coverage >= {lv} over simulations: the exchangeable unit is a whole "
            "simulation and the guarantee is exact (split conformal)."
        )
    if aggregation == "max":
        return (
            f"Simultaneous band: with probability >= {lv} over simulations, ALL surface points "
            "of a new simulation are covered. Not a pointwise guarantee."
        )
    if aggregation == "quantile":
        q = "?" if agg_q is None else f"{agg_q:.3g}"
        return (
            f"With probability >= {lv} over simulations, at least a {q} fraction of the surface "
            "points of a new simulation are covered. Not a pointwise guarantee."
        )
    return f"Coverage >= {lv} at the simulation level."


# --------------------------------------------------------------------------
# fitters
# --------------------------------------------------------------------------
class _BaseSplitConformal:
    score: ScoreName
    eps: float

    def __init__(self, *, score: ScoreName = "normalized", eps: float = DEFAULT_EPS) -> None:
        if score not in ("normalized", "absolute"):
            raise ValueError(f"unknown score {score!r}")
        self.score = score
        self.eps = float(eps)

    @staticmethod
    def _alpha_from_level(level: float) -> float:
        if not (0.0 < level < 1.0):
            raise ValueError(f"level (nominal coverage 1-alpha) must lie in (0,1), got {level}")
        return 1.0 - float(level)


class SplitConformalCoefficients(_BaseSplitConformal):
    """Per-simulation coefficient intervals -- the clean-guarantee granularity.

    One calibration *unit* = one simulation, so with ``n`` calibration sims and
    a ``(n, d)`` array of predicted coefficient means the guarantee is the
    standard split-conformal marginal one.

    ``per_target=True`` (default) calibrates each coefficient (CL, CD, ...)
    independently -- d separate conformal problems, each marginally valid.
    ``per_target=False`` pools all targets into one score set, giving a single
    shared ``q_hat`` (valid only if the targets are on a comparable scale, which
    is why normalized scores are the default).
    """

    def __init__(
        self,
        *,
        score: ScoreName = "normalized",
        eps: float = DEFAULT_EPS,
        per_target: bool = True,
    ) -> None:
        super().__init__(score=score, eps=eps)
        self.per_target = bool(per_target)

    def fit(
        self,
        cal_mu: Any,
        cal_sigma: Any | None,
        cal_y: Any,
        level: float = 0.9,
        *,
        target_names: Sequence[str] | None = None,
    ) -> CalibratedIntervals:
        """Calibrate on ``n`` simulations.

        Parameters
        ----------
        cal_mu, cal_sigma, cal_y : ``(n,)`` or ``(n, d)`` arrays.  ``cal_sigma``
            may be ``None`` when ``score='absolute'``.
        level : nominal coverage ``1 - alpha`` (e.g. 0.8, 0.9, 0.95).
        """
        alpha = self._alpha_from_level(level)
        mu = np.asarray(as_numpy(cal_mu), dtype=np.float64)
        y = np.asarray(as_numpy(cal_y), dtype=np.float64)
        sig = None if cal_sigma is None else np.asarray(as_numpy(cal_sigma), dtype=np.float64)
        if mu.ndim == 1:
            mu = mu[:, None]
            y = y.reshape(mu.shape)
            if sig is not None:
                sig = sig.reshape(mu.shape)
            squeeze = True
        elif mu.ndim == 2:
            squeeze = False
        else:
            raise ValueError(f"cal_mu must be 1-D or 2-D, got shape {mu.shape}")
        if y.shape != mu.shape:
            raise ValueError(f"cal_y shape {y.shape} != cal_mu shape {mu.shape}")
        n, d = mu.shape
        scores = nonconformity_scores(mu, sig, y, score=self.score, eps=self.eps)

        if self.per_target:
            q = np.array([conformal_quantile(scores[:, j], alpha) for j in range(d)], dtype=np.float64)
            if squeeze:
                q = q.reshape(())  # 1-D input -> scalar q_hat
            cal_scores = scores[:, 0] if squeeze else scores
        else:
            # Pooled: one score per simulation = max over targets, so the
            # exchangeable unit stays the simulation and the band is
            # simultaneous over the coefficients.
            pooled = scores.max(axis=1)
            q = np.asarray(conformal_quantile(pooled, alpha))
            cal_scores = pooled
        return CalibratedIntervals(
            q_hat=q,
            level=float(level),
            alpha=alpha,
            score=self.score,
            eps=self.eps,
            granularity="coefficient",
            n_cal=int(n),
            target_names=tuple(target_names) if target_names else None,
            cal_scores=cal_scores,
        )


class SplitConformalField(_BaseSplitConformal):
    """Pointwise field bands conformalized at the **simulation** level.

    Each calibration simulation contributes exactly one score, obtained by
    reducing its per-point scores with ``aggregation`` (``"max"`` or
    ``"quantile"`` with ``agg_q``).  Both variants are implemented and selected
    by parameter, per the spec.
    """

    def __init__(
        self,
        *,
        score: ScoreName = "normalized",
        eps: float = DEFAULT_EPS,
        aggregation: AggName = "max",
        agg_q: float = 0.9,
    ) -> None:
        super().__init__(score=score, eps=eps)
        if aggregation not in ("max", "quantile"):
            raise ValueError(f"unknown aggregation {aggregation!r}")
        self.aggregation = aggregation
        self.agg_q = float(agg_q)

    def fit(
        self,
        cal_mu: Any,
        cal_sigma: Any | None,
        cal_y: Any,
        level: float = 0.9,
        *,
        sim_idx: Any | None = None,
    ) -> CalibratedIntervals:
        """Calibrate on ``n`` simulations' worth of surface points.

        ``cal_mu``/``cal_sigma``/``cal_y`` may be

        * a list of per-simulation 1-D arrays (ragged point counts fine),
        * a 2-D ``(n_sim, n_pts)`` array, or
        * flat concatenated arrays plus ``sim_idx`` (the ``batch_idx``
          convention of CONTEXT.md section 4).
        """
        alpha = self._alpha_from_level(level)
        mus = _as_sim_list(cal_mu, sim_idx, "cal_mu")
        ys = _as_sim_list(cal_y, sim_idx, "cal_y")
        if len(mus) != len(ys):
            raise ValueError(f"{len(mus)} mu simulations vs {len(ys)} y simulations")
        if cal_sigma is None:
            if self.score == "normalized":
                raise ValueError("normalized score needs cal_sigma")
            sigs: list[np.ndarray | None] = [None] * len(mus)
        else:
            sigs = _as_sim_list(cal_sigma, sim_idx, "cal_sigma")  # type: ignore[assignment]
            if len(sigs) != len(mus):
                raise ValueError(f"{len(mus)} mu simulations vs {len(sigs)} sigma simulations")

        sim_scores = np.empty(len(mus), dtype=np.float64)
        for i, (m, s, yy) in enumerate(zip(mus, sigs, ys)):
            if m.shape != yy.shape:
                raise ValueError(f"simulation {i}: mu shape {m.shape} != y shape {yy.shape}")
            pts = nonconformity_scores(m, s, yy, score=self.score, eps=self.eps)
            sim_scores[i] = aggregate_point_scores(pts, aggregation=self.aggregation, agg_q=self.agg_q)

        q = np.asarray(conformal_quantile(sim_scores, alpha))
        return CalibratedIntervals(
            q_hat=q,
            level=float(level),
            alpha=alpha,
            score=self.score,
            eps=self.eps,
            granularity=f"field_{self.aggregation}",
            aggregation=self.aggregation,
            agg_q=self.agg_q if self.aggregation == "quantile" else None,
            n_cal=int(sim_scores.size),
            cal_scores=sim_scores,
        )


def split_conformal(
    cal_mu: Any,
    cal_sigma: Any | None,
    cal_y: Any,
    level: float = 0.9,
    *,
    granularity: Literal["coefficient", "field"] = "coefficient",
    score: ScoreName = "normalized",
    aggregation: AggName = "max",
    agg_q: float = 0.9,
    eps: float = DEFAULT_EPS,
    per_target: bool = True,
    sim_idx: Any | None = None,
    target_names: Sequence[str] | None = None,
) -> CalibratedIntervals:
    """One-call front end over both fitters (convenience for sweeps)."""
    if granularity == "coefficient":
        return SplitConformalCoefficients(score=score, eps=eps, per_target=per_target).fit(
            cal_mu, cal_sigma, cal_y, level, target_names=target_names
        )
    if granularity == "field":
        return SplitConformalField(score=score, eps=eps, aggregation=aggregation, agg_q=agg_q).fit(
            cal_mu, cal_sigma, cal_y, level, sim_idx=sim_idx
        )
    raise ValueError(f"unknown granularity {granularity!r}")
