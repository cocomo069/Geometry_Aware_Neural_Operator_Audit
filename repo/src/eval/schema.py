"""Validation of the FROZEN results schema (CONTEXT.md 9).

Dependency choice: **plain-dict validation, no ``jsonschema``**.  The schema is
small, frozen, and fully described by CONTEXT.md 9; adding a runtime dependency
(and a second source of truth in JSON-Schema syntax) buys nothing here, and the
error messages below are more useful than jsonschema's.  Logged in PROGRESS.md.

Numeric leaves accept ``None`` (serialised as JSON ``null``) to mean "not
computed on this run" -- e.g. ``peak_mem_mb`` on a CPU-only evaluation, or the
symmetry residual when ``src/geometry/symmetry.py`` is unavailable.  A *missing*
key is always an error; that distinction is the point of the validator.

API convention (fail loud by default)
-------------------------------------
``validate_metrics`` / ``validate_uq`` **raise** :class:`SchemaError`; that is
the documented entry point everywhere, including the harness.  Ignoring their
return value cannot silently pass an invalid payload: on success they hand back
the (necessarily empty) error list, so ``assert validate_uq(x) == []`` still
reads correctly, and on failure they raise instead of returning.
``check_metrics`` / ``check_uq`` are the non-raising variants: they return a
list of human-readable violations (empty == valid), for callers that want to
report several problems at once.  ``assert_valid_*`` remain as aliases of the
raising form.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "METRICS_SCHEMA",
    "UQ_SCHEMA",
    "FIELD_KEYS",
    "COEF_KEYS",
    "CONSISTENCY_KEYS",
    "COST_KEYS",
    "validate_metrics",
    "validate_uq",
    "assert_valid_metrics",
    "assert_valid_uq",
    "check_metrics",
    "check_uq",
    "empty_metrics",
    "SchemaError",
]

NUM = (int, float)


class SchemaError(ValueError):
    """Raised by ``validate_*`` / ``assert_valid_*`` when validation fails."""


FIELD_KEYS = ("p_rel_l2", "tau_rel_l2", "p_mae", "tau_mae")
COEF_KEYS = (
    "cl_head_mae",
    "cd_head_mae",
    "cl_int_mae",
    "cd_int_mae",
    "cd_spearman",
    "cd_head_spearman",
    "cd_int_spearman",
    "cl_rel",
    "cd_rel",
)
CONSISTENCY_KEYS = (
    "fsc_cl",
    "fsc_cd",
    "fsc_rel_cl",
    "fsc_rel_cd",
    "sym_residual",
    "antisym_cl_gap",
)
COST_KEYS = ("infer_ms_per_sim", "peak_mem_mb")

#: ``key -> spec``.  A spec is either a tuple of accepted python types
#: (``None`` allowed if ``type(None)`` is in the tuple) or a nested dict.
METRICS_SCHEMA: dict[str, Any] = {
    "run_id": (str,),
    "model": (str,),
    "split": (str,),
    "seed": (int,),
    "tag": (str, type(None)),
    "params": (int, type(None)),
    "train_time_s": (*NUM, type(None)),
    "epochs": (int, type(None)),
    "device": (str,),
    "field": {k: (*NUM, type(None)) for k in FIELD_KEYS},
    "coef": {k: (*NUM, type(None)) for k in COEF_KEYS},
    "consistency": {k: (*NUM, type(None)) for k in CONSISTENCY_KEYS},
    "cost": {k: (*NUM, type(None)) for k in COST_KEYS},
}

#: UQ results (``results/uq/<ensemble_id>.json``), CONTEXT.md 9 + spec 5.6.
UQ_SCHEMA: dict[str, Any] = {
    "ensemble_id": (str,),
    "model": (str,),
    "split": (str,),
    "seeds": (list,),
    "method": (str,),  # 'ensemble' | 'conformal'
    "levels": (list,),  # nominal 1-alpha, e.g. [0.8, 0.9, 0.95]
    "coef": (dict,),  # per-quantity -> per-level {coverage, width_mean, width_median}
    "field": (dict,),
}


def _walk(node: Any, spec: Mapping[str, Any], path: str, errors: list[str],
          allow_extra: bool) -> None:
    if not isinstance(node, Mapping):
        errors.append(f"{path or '<root>'}: expected a mapping, got {type(node).__name__}")
        return
    for key, sub in spec.items():
        full = f"{path}.{key}" if path else key
        if key not in node:
            errors.append(f"missing key: {full}")
            continue
        value = node[key]
        if isinstance(sub, dict):
            _walk(value, sub, full, errors, allow_extra)
            continue
        # bool is an int subclass but is never a valid metric value
        if isinstance(value, bool) and bool not in sub:
            errors.append(f"{full}: expected {_names(sub)}, got bool")
            continue
        if not isinstance(value, sub):
            errors.append(f"{full}: expected {_names(sub)}, got {type(value).__name__}")
    if not allow_extra:
        for key in node:
            if key not in spec:
                full = f"{path}.{key}" if path else key
                errors.append(f"unexpected key: {full}")


def _names(types: tuple[type, ...]) -> str:
    return "/".join("null" if t is type(None) else t.__name__ for t in types)


def check_metrics(metrics: Any, *, allow_extra: bool = True) -> list[str]:
    """Return a list of human-readable schema violations (empty == valid).

    Non-raising counterpart of :func:`validate_metrics`.

    Parameters
    ----------
    metrics:
        Candidate ``metrics.json`` content as a dict.
    allow_extra:
        Extra top-level/nested keys are tolerated by default (runs may carry
        provenance such as ``n_sims`` or ``timestamp``).  Set ``False`` for a
        strict check.
    """
    errors: list[str] = []
    _walk(metrics, METRICS_SCHEMA, "", errors, allow_extra)
    return errors


def check_uq(payload: Any, *, allow_extra: bool = True) -> list[str]:
    """Return schema violations for a ``results/uq/<ensemble_id>.json`` payload."""
    errors: list[str] = []
    _walk(payload, UQ_SCHEMA, "", errors, allow_extra)
    return errors


def validate_metrics(metrics: Any, *, allow_extra: bool = True) -> list[str]:
    """Raise :class:`SchemaError` if ``metrics`` violates CONTEXT.md 9.

    Raising is deliberate: a validator whose only signal is a return value is
    one bare ``validate(x)`` statement away from passing an invalid payload
    through silently.  On success the (necessarily empty) error list is
    returned, so callers written as ``assert validate_metrics(x) == []`` keep
    working; use :func:`check_metrics` to get errors *without* an exception.
    """
    errors = check_metrics(metrics, allow_extra=allow_extra)
    if errors:
        raise SchemaError(
            "metrics.json does not match the frozen schema (CONTEXT.md 9):\n  - "
            + "\n  - ".join(errors)
        )
    return errors


def validate_uq(payload: Any, *, allow_extra: bool = True) -> list[str]:
    """Raise :class:`SchemaError` if a UQ payload violates the schema.

    Returns the empty error list on success; see :func:`validate_metrics`.
    """
    errors = check_uq(payload, allow_extra=allow_extra)
    if errors:
        raise SchemaError(
            "uq json does not match the schema:\n  - " + "\n  - ".join(errors)
        )
    return errors


#: Aliases; ``assert_valid_*`` and ``validate_*`` are the same raising check.
assert_valid_metrics = validate_metrics
assert_valid_uq = validate_uq


def empty_metrics(**top: Any) -> dict[str, Any]:
    """A schema-complete metrics dict with every numeric leaf set to ``None``.

    Useful as a starting point (harness, baselines) so no key can be forgotten.
    """
    out: dict[str, Any] = {
        "run_id": "",
        "model": "",
        "split": "",
        "seed": 0,
        "tag": None,
        "params": None,
        "train_time_s": None,
        "epochs": None,
        "device": "cpu",
        "field": {k: None for k in FIELD_KEYS},
        "coef": {k: None for k in COEF_KEYS},
        "consistency": {k: None for k in CONSISTENCY_KEYS},
        "cost": {k: None for k in COST_KEYS},
    }
    for k, v in top.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out
