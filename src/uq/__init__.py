"""Uncertainty quantification: deep ensembles + split conformal (spec section 5.6).

    from src.uq import EnsemblePredictor, SplitConformalCoefficients, coverage_with_ci
"""

from .conformal import (
    DEFAULT_EPS,
    CalibratedIntervals,
    SplitConformalCoefficients,
    SplitConformalField,
    aggregate_point_scores,
    conformal_quantile,
    conformal_quantile_index,
    nonconformity_scores,
    split_conformal,
)
from .coverage import (
    DEFAULT_LEVELS,
    RELIABILITY_LEVELS,
    UQ_SCHEMA_VERSION,
    binomial_ci,
    coverage_with_ci,
    empirical_coverage,
    evaluate_intervals,
    expected_calibration_error,
    field_coverage,
    make_uq_record,
    make_uq_report,
    reliability_curve,
    sweep_levels_coefficients,
    sweep_levels_field,
    validate_uq_report,
    width_stats,
    wilson_interval,
    write_uq_report,
)
from .ensembles import (
    EnsemblePredictor,
    EnsembleStats,
    build_model_from_config,
    ensemble_mean_std,
    load_ensemble_checkpoints,
    propagate,
    stack_members,
)

__all__ = [
    # ensembles
    "EnsemblePredictor",
    "EnsembleStats",
    "ensemble_mean_std",
    "stack_members",
    "propagate",
    "load_ensemble_checkpoints",
    "build_model_from_config",
    # conformal
    "CalibratedIntervals",
    "SplitConformalCoefficients",
    "SplitConformalField",
    "split_conformal",
    "conformal_quantile",
    "conformal_quantile_index",
    "nonconformity_scores",
    "aggregate_point_scores",
    "DEFAULT_EPS",
    # coverage
    "empirical_coverage",
    "width_stats",
    "wilson_interval",
    "binomial_ci",
    "coverage_with_ci",
    "expected_calibration_error",
    "reliability_curve",
    "evaluate_intervals",
    "field_coverage",
    "sweep_levels_coefficients",
    "sweep_levels_field",
    "make_uq_record",
    "make_uq_report",
    "validate_uq_report",
    "write_uq_report",
    "UQ_SCHEMA_VERSION",
    "DEFAULT_LEVELS",
    "RELIABILITY_LEVELS",
]
