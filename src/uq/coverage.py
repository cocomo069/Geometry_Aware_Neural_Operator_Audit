"""Coverage / width / reliability diagnostics and the ``results/uq`` schema.

Reported quantities (spec section 5.6): empirical coverage at nominal levels
``1 - alpha`` in {0.8, 0.9, 0.95}, interval width, reliability curves (nominal
vs empirical over a grid of levels) and expected calibration error.

Schema ownership note
---------------------
CONTEXT.md section 9 says the UQ record lives at ``results/uq/<ensemble_id>.json``
with "schema in ``src/eval/schema.py``".  ``src/eval/`` is A4's directory and did
not exist when this module was written, so the UQ schema is **defined here**
(``UQ_SCHEMA_VERSION``, :func:`make_uq_record`, :func:`make_uq_report`,
:func:`validate_uq_report`).  :func:`validate_uq_report` late-imports
``src.eval.schema`` and defers to a ``validate_uq_report``/``validate_uq`` found
there if A4 later provides one, so the two can converge without a breaking
change.  Flagged in docs/PROGRESS.md.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .conformal import (  # noqa: F401  (aggregate_point_scores re-exported)
    CalibratedIntervals,
    SplitConformalCoefficients,
    SplitConformalField,
    aggregate_point_scores,
)
from .ensembles import as_numpy

__all__ = [
    "UQ_SCHEMA_VERSION",
    "DEFAULT_LEVELS",
    "empirical_coverage",
    "width_stats",
    "wilson_interval",
    "binomial_ci",
    "coverage_with_ci",
    "expected_calibration_error",
    "reliability_curve",
    "evaluate_intervals",
    "sweep_levels_coefficients",
    "sweep_levels_field",
    "field_coverage",
    "make_uq_record",
    "make_uq_report",
    "validate_uq_report",
    "write_uq_report",
]

UQ_SCHEMA_VERSION = "uq-1"

#: The three nominal levels the spec reports.
DEFAULT_LEVELS: tuple[float, ...] = (0.8, 0.9, 0.95)

#: Denser grid for reliability curves.
RELIABILITY_LEVELS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def empirical_coverage(y: Any, lo: Any, hi: Any) -> float:
    """Fraction of targets inside ``[lo, hi]`` (closed interval)."""
    y_a = np.asarray(as_numpy(y), dtype=np.float64)
    lo_a = np.broadcast_to(np.asarray(as_numpy(lo), dtype=np.float64), y_a.shape)
    hi_a = np.broadcast_to(np.asarray(as_numpy(hi), dtype=np.float64), y_a.shape)
    if y_a.size == 0:
        raise ValueError("empty target array")
    inside = (y_a >= lo_a) & (y_a <= hi_a)
    return float(np.mean(inside))


def width_stats(lo: Any, hi: Any) -> dict[str, float]:
    """Mean / median / max interval width."""
    lo_a = np.asarray(as_numpy(lo), dtype=np.float64)
    hi_a = np.asarray(as_numpy(hi), dtype=np.float64)
    w = np.broadcast_arrays(hi_a, lo_a)
    width = (w[0] - w[1]).ravel()
    if width.size == 0:
        raise ValueError("empty interval array")
    finite = width[np.isfinite(width)]
    return {
        "mean_width": float(np.mean(width)) if finite.size == width.size else float("inf"),
        "median_width": float(np.median(width)),
        "max_width": float(np.max(width)),
    }


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation near 0/1, which is exactly where
    coverage at the 0.95 level lives.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie in (0,1)")
    z = _normal_quantile(0.5 * (1.0 + confidence))
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def binomial_ci(p_hat: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson interval from an observed proportion (rounds to the nearest count)."""
    return wilson_interval(int(round(p_hat * n)), n, confidence)


def _normal_quantile(p: float) -> float:
    """Inverse standard-normal CDF via ``erfinv`` (no scipy dependency)."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must lie in (0,1)")
    # math.erf is stdlib; invert by bisection on the CDF (cheap, exact enough).
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def coverage_with_ci(y: Any, lo: Any, hi: Any, confidence: float = 0.95) -> dict[str, float]:
    """Coverage plus its Wilson CI and the effective sample size."""
    y_a = np.asarray(as_numpy(y), dtype=np.float64)
    cov = empirical_coverage(y_a, lo, hi)
    n = int(y_a.size)
    ci_lo, ci_hi = wilson_interval(int(round(cov * n)), n, confidence)
    return {"coverage": cov, "coverage_ci_lo": ci_lo, "coverage_ci_hi": ci_hi, "n": n}


def expected_calibration_error(
    nominal: Sequence[float] | np.ndarray,
    empirical: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Mean absolute gap between nominal and empirical coverage over the grid.

    This is the calibration-curve ECE used for interval predictors: the
    (optionally weighted) L1 distance between the reliability curve and the
    diagonal.
    """
    nom = np.asarray(nominal, dtype=np.float64).ravel()
    emp = np.asarray(empirical, dtype=np.float64).ravel()
    if nom.shape != emp.shape:
        raise ValueError(f"nominal shape {nom.shape} != empirical shape {emp.shape}")
    if nom.size == 0:
        raise ValueError("empty reliability grid")
    gaps = np.abs(nom - emp)
    if weights is None:
        return float(np.mean(gaps))
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != nom.shape:
        raise ValueError("weights must match the grid")
    if np.sum(w) <= 0:
        raise ValueError("weights must sum to a positive number")
    return float(np.sum(w * gaps) / np.sum(w))


# --------------------------------------------------------------------------
# level sweeps
# --------------------------------------------------------------------------
def evaluate_intervals(
    cal: CalibratedIntervals,
    mu: Any,
    sigma: Any | None,
    y: Any,
    *,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Coverage + width for one calibrated level on a test set."""
    lo, hi = cal.predict(mu, sigma)
    out = coverage_with_ci(y, lo, hi, confidence)
    out.update(width_stats(lo, hi))
    q = np.asarray(cal.q_hat, dtype=np.float64)
    out["q_hat"] = float(q) if q.ndim == 0 else float(np.mean(q))
    out["nominal"] = float(cal.level)
    out["alpha"] = float(cal.alpha)
    return out


def sweep_levels_coefficients(
    cal_mu: Any,
    cal_sigma: Any | None,
    cal_y: Any,
    test_mu: Any,
    test_sigma: Any | None,
    test_y: Any,
    *,
    levels: Sequence[float] = DEFAULT_LEVELS,
    score: str = "normalized",
    per_target: bool = True,
    target_names: Sequence[str] | None = None,
    confidence: float = 0.95,
) -> list[dict[str, float]]:
    """Fit + evaluate the coefficient granularity at every nominal level."""
    fitter = SplitConformalCoefficients(score=score, per_target=per_target)  # type: ignore[arg-type]
    rows: list[dict[str, float]] = []
    for level in levels:
        ci = fitter.fit(cal_mu, cal_sigma, cal_y, level, target_names=target_names)
        rows.append(evaluate_intervals(ci, test_mu, test_sigma, test_y, confidence=confidence))
    return rows


def field_coverage(
    cal: CalibratedIntervals,
    mu_sims: Sequence[Any],
    sigma_sims: Sequence[Any] | None,
    y_sims: Sequence[Any],
    *,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Sim-level and pointwise coverage for a field band.

    ``sim_coverage`` is the quantity the band actually certifies:

    * ``field_max``      -> fraction of test sims with *all* points covered;
    * ``field_quantile`` -> fraction of test sims with at least ``agg_q`` of
      their points covered.

    ``point_coverage`` (fraction of all points covered, pooled) is reported as a
    diagnostic only -- it carries no guarantee.
    """
    n_sim = len(mu_sims)
    if n_sim == 0:
        raise ValueError("no test simulations")
    if sigma_sims is None:
        sigma_sims = [None] * n_sim  # type: ignore[list-item]
    hits = np.zeros(n_sim, dtype=bool)
    pt_cov = np.zeros(n_sim, dtype=np.float64)
    n_pts = np.zeros(n_sim, dtype=np.int64)
    widths: list[np.ndarray] = []
    thresh = 1.0 if cal.aggregation != "quantile" else float(cal.agg_q or 0.9)
    for i in range(n_sim):
        m = np.asarray(as_numpy(mu_sims[i]), dtype=np.float64).ravel()
        yy = np.asarray(as_numpy(y_sims[i]), dtype=np.float64).ravel()
        s = None if sigma_sims[i] is None else np.asarray(as_numpy(sigma_sims[i]), dtype=np.float64).ravel()
        lo, hi = cal.predict(m, s)
        inside = (yy >= lo) & (yy <= hi)
        frac = float(np.mean(inside))
        pt_cov[i] = frac
        n_pts[i] = inside.size
        widths.append(hi - lo)
        hits[i] = frac >= thresh - 1e-12
    sim_cov = float(np.mean(hits))
    ci_lo, ci_hi = wilson_interval(int(hits.sum()), n_sim, confidence)
    all_w = np.concatenate([w.ravel() for w in widths])
    return {
        "nominal": float(cal.level),
        "alpha": float(cal.alpha),
        "q_hat": float(np.asarray(cal.q_hat)),
        "coverage": sim_cov,
        "coverage_ci_lo": ci_lo,
        "coverage_ci_hi": ci_hi,
        "n": n_sim,
        "point_coverage": float(np.sum(pt_cov * n_pts) / np.sum(n_pts)),
        "mean_width": float(np.mean(all_w)),
        "median_width": float(np.median(all_w)),
        "max_width": float(np.max(all_w)),
    }


def sweep_levels_field(
    cal_mu: Sequence[Any],
    cal_sigma: Sequence[Any] | None,
    cal_y: Sequence[Any],
    test_mu: Sequence[Any],
    test_sigma: Sequence[Any] | None,
    test_y: Sequence[Any],
    *,
    levels: Sequence[float] = DEFAULT_LEVELS,
    score: str = "normalized",
    aggregation: str = "max",
    agg_q: float = 0.9,
    confidence: float = 0.95,
) -> list[dict[str, float]]:
    """Fit + evaluate a field granularity at every nominal level."""
    fitter = SplitConformalField(score=score, aggregation=aggregation, agg_q=agg_q)  # type: ignore[arg-type]
    rows: list[dict[str, float]] = []
    for level in levels:
        ci = fitter.fit(cal_mu, cal_sigma, cal_y, level)
        rows.append(field_coverage(ci, test_mu, test_sigma, test_y, confidence=confidence))
    return rows


def reliability_curve(rows: Sequence[Mapping[str, float]]) -> dict[str, list[float]]:
    """Extract ``{"nominal": [...], "empirical": [...]}`` from level-sweep rows."""
    nominal = [float(r["nominal"]) for r in rows]
    empirical = [float(r["coverage"]) for r in rows]
    order = np.argsort(nominal)
    return {
        "nominal": [nominal[i] for i in order],
        "empirical": [empirical[i] for i in order],
    }


# --------------------------------------------------------------------------
# results/uq schema
# --------------------------------------------------------------------------
_LEVEL_KEYS = ("nominal", "alpha", "q_hat", "coverage", "coverage_ci_lo", "coverage_ci_hi", "mean_width", "median_width", "n")
_RECORD_KEYS = ("target", "granularity", "score", "n_cal", "n_test", "levels", "ece", "reliability")
_REPORT_KEYS = ("schema_version", "ensemble_id", "split", "k", "records")


def make_uq_record(
    target: str,
    granularity: str,
    rows: Sequence[Mapping[str, float]],
    *,
    score: str = "normalized",
    n_cal: int = 0,
    agg_q: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One record = one (target, granularity) pair swept over nominal levels."""
    if not rows:
        raise ValueError("no level rows")
    levels = []
    for r in rows:
        entry = {k: (float(r[k]) if k != "n" else int(r["n"])) for k in _LEVEL_KEYS if k in r}
        for k in _LEVEL_KEYS:
            entry.setdefault(k, float("nan") if k != "n" else 0)
        if "point_coverage" in r:
            entry["point_coverage"] = float(r["point_coverage"])
        levels.append(entry)
    rel = reliability_curve(rows)
    rec: dict[str, Any] = {
        "target": str(target),
        "granularity": str(granularity),
        "score": str(score),
        "agg_q": None if agg_q is None else float(agg_q),
        "n_cal": int(n_cal),
        "n_test": int(levels[0].get("n", 0)),
        "levels": levels,
        "ece": expected_calibration_error(rel["nominal"], rel["empirical"]),
        "reliability": rel,
    }
    if extra:
        rec.update(dict(extra))
    return rec


def _a4_compat_views(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[float], dict[str, Any], dict[str, Any]]:
    """Flat ``levels`` / ``coef`` / ``field`` views required by A4's UQ_SCHEMA.

    ``src/eval/schema.py`` (A4) expects ``coef``/``field`` to be
    ``quantity -> level -> {coverage, width_mean, width_median}``.  Our richer
    ``records`` list carries strictly more information, so we project it down
    into those two dicts and ship both -- A4's validator allows extra keys, so
    one payload satisfies both schemas.  See docs/PROGRESS.md.
    """
    levels: set[float] = set()
    coef: dict[str, Any] = {}
    field_: dict[str, Any] = {}
    for rec in records:
        bucket = coef if rec.get("granularity") == "coefficient" else field_
        target = str(rec.get("target", "unknown"))
        per_level = bucket.setdefault(target, {})
        for lv in rec.get("levels", []):
            nominal = float(lv["nominal"])
            levels.add(nominal)
            per_level[f"{nominal:g}"] = {
                "coverage": float(lv.get("coverage", float("nan"))),
                "width_mean": float(lv.get("mean_width", float("nan"))),
                "width_median": float(lv.get("median_width", float("nan"))),
            }
    return sorted(levels), coef, field_


def make_uq_report(
    ensemble_id: str,
    split: str,
    records: Sequence[Mapping[str, Any]],
    *,
    model: str | None = None,
    seeds: Sequence[int] | None = None,
    k: int | None = None,
    method: str = "conformal",
    created: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ``results/uq/<ensemble_id>.json`` payload.

    The payload satisfies **both** this module's ``uq-1`` schema (the
    ``records`` list) and A4's ``src/eval/schema.py::UQ_SCHEMA`` (the flat
    ``method``/``levels``/``coef``/``field`` view derived from it).
    """
    recs = [dict(r) for r in records]
    flat_levels, coef_view, field_view = _a4_compat_views(recs)
    report: dict[str, Any] = {
        "schema_version": UQ_SCHEMA_VERSION,
        "ensemble_id": str(ensemble_id),
        "model": "" if model is None else str(model),
        "split": str(split),
        "seeds": [int(s) for s in seeds] if seeds is not None else [],
        "k": int(k) if k is not None else (len(seeds) if seeds is not None else 0),
        "method": str(method),
        "created": created or date.today().isoformat(),
        "records": recs,
        # --- flat view for src/eval/schema.py::UQ_SCHEMA ---
        "levels": flat_levels,
        "coef": coef_view,
        "field": field_view,
    }
    if extra:
        report.update(dict(extra))
    validate_uq_report(report)
    return report


def validate_uq_report(report: Mapping[str, Any]) -> None:
    """Raise ``ValueError``/``SchemaError`` if the payload is malformed.

    Runs this module's structural checks first, then -- if A4's
    ``src.eval.schema`` is importable -- its ``assert_valid_uq`` as well, so a
    payload has to satisfy both.  Note that A4's ``validate_uq`` *returns* a
    list of errors rather than raising; only ``assert_valid_uq`` raises.
    """
    _validate_uq_report_local(report)
    try:  # late import; src/eval is A4's directory
        import importlib

        schema_mod = importlib.import_module("src.eval.schema")
    except Exception:
        return
    asserter = getattr(schema_mod, "assert_valid_uq", None)
    if callable(asserter):
        asserter(report)
        return
    checker = getattr(schema_mod, "validate_uq", None)
    if callable(checker):
        errors = checker(report)
        if errors:
            raise ValueError("uq report violates src/eval/schema.py: " + "; ".join(map(str, errors)))


def _validate_uq_report_local(report: Mapping[str, Any]) -> None:
    if not isinstance(report, Mapping):
        raise ValueError("uq report must be a mapping")
    for key in _REPORT_KEYS:
        if key not in report:
            raise ValueError(f"uq report missing key {key!r}")
    if report["schema_version"] != UQ_SCHEMA_VERSION:
        raise ValueError(f"unexpected schema_version {report['schema_version']!r}")
    records = report["records"]
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("uq report needs a non-empty 'records' list")
    for i, rec in enumerate(records):
        for key in _RECORD_KEYS:
            if key not in rec:
                raise ValueError(f"uq record {i} missing key {key!r}")
        if rec["granularity"] not in ("coefficient", "field_max", "field_quantile"):
            raise ValueError(f"uq record {i}: bad granularity {rec['granularity']!r}")
        if rec["score"] not in ("normalized", "absolute"):
            raise ValueError(f"uq record {i}: bad score {rec['score']!r}")
        levels = rec["levels"]
        if not isinstance(levels, (list, tuple)) or not levels:
            raise ValueError(f"uq record {i}: needs a non-empty 'levels' list")
        for j, lv in enumerate(levels):
            for key in ("nominal", "coverage", "mean_width", "q_hat"):
                if key not in lv:
                    raise ValueError(f"uq record {i} level {j} missing key {key!r}")
            if not (0.0 < float(lv["nominal"]) < 1.0):
                raise ValueError(f"uq record {i} level {j}: nominal out of (0,1)")
            if not (0.0 <= float(lv["coverage"]) <= 1.0):
                raise ValueError(f"uq record {i} level {j}: coverage out of [0,1]")
        rel = rec["reliability"]
        if set(rel) != {"nominal", "empirical"} or len(rel["nominal"]) != len(rel["empirical"]):
            raise ValueError(f"uq record {i}: malformed reliability curve")


def write_uq_report(report: Mapping[str, Any], out_dir: str | Path = "results/uq") -> Path:
    """Validate and write ``<out_dir>/<ensemble_id>.json``; returns the path."""
    validate_uq_report(report)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{report['ensemble_id']}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    return path
