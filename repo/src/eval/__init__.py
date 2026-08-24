"""Evaluation: metrics, the metrics.json harness, schema validation, baselines.

``src.eval.harness.evaluate`` is the **only** sanctioned producer of
``results/<run_id>/metrics.json`` (CONTEXT.md 9).
"""

from src.eval.metrics import (
    MetricAccumulator,
    SimRecord,
    coef_metrics,
    fsc,
    fsc_rel,
    mae,
    rel_l2,
    spearman,
)
from src.eval.schema import (
    METRICS_SCHEMA,
    SchemaError,
    assert_valid_metrics,
    empty_metrics,
    validate_metrics,
)

__all__ = [
    "METRICS_SCHEMA",
    "MetricAccumulator",
    "SchemaError",
    "SimRecord",
    "assert_valid_metrics",
    "coef_metrics",
    "empty_metrics",
    "evaluate",
    "fsc",
    "fsc_rel",
    "mae",
    "rel_l2",
    "spearman",
    "validate_metrics",
]


def __getattr__(name: str):
    """Lazily expose ``evaluate`` so importing metrics/schema never needs torch."""
    if name == "evaluate":
        from src.eval.harness import evaluate

        return evaluate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
