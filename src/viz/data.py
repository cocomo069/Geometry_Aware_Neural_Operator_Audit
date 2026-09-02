"""Read ``results/`` into tidy pandas frames.

Three loaders, one per artefact defined by CONTEXT.md section 9:

* :func:`load_runs`     -- every ``results/<run_id>/metrics.json``, one row per
  run, nested groups flattened to ``field_p_rel_l2``, ``coef_cd_int_mae``, ...
* :func:`load_per_sim`  -- ``results/<run_id>/per_sim.csv``, one row per test
  simulation, with the run's identity columns joined on.
* :func:`load_uq`       -- ``results/uq/<ensemble_id>.json``, one row per
  (record, nominal level).

**Robustness contract.**  A missing directory, a run that has only written
``config.yaml`` so far (training still in flight), a truncated JSON file, or a
``metrics.json`` that fails :mod:`src.eval.schema` are all *skipped* with a
warning rather than raised: the figure pipeline must run against a partially
populated results tree, which is its normal state.  Set ``strict=True`` to turn
those skips into errors.

Every loader returns a frame with a stable column set even when it finds
nothing, so downstream ``df[df.split == "full"]`` never raises ``KeyError`` on
an empty tree.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

__all__ = [
    "RUN_ID_COLUMNS",
    "RUN_COLUMNS",
    "PER_SIM_COLUMNS",
    "UQ_COLUMNS",
    "load_metrics_file",
    "load_runs",
    "load_per_sim",
    "load_all_per_sim",
    "load_uq_reports",
    "load_uq",
    "reliability_frame",
    "add_derived_columns",
    "pivot_best",
]

#: Identity of a run; present on every frame this module returns.
RUN_ID_COLUMNS: tuple[str, ...] = ("run_id", "model", "split", "seed", "tag")

_GROUPS = ("field", "coef", "consistency", "cost")

_SCALAR_KEYS = (
    "run_id",
    "model",
    "split",
    "seed",
    "tag",
    "params",
    "train_time_s",
    "epochs",
    "device",
    "n_sims",
)

_FIELD_KEYS = ("p_rel_l2", "tau_rel_l2", "p_mae", "tau_mae")
_COEF_KEYS = (
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
_CONSISTENCY_KEYS = (
    "fsc_cl",
    "fsc_cd",
    "fsc_rel_cl",
    "fsc_rel_cd",
    "sym_residual",
    "antisym_cl_gap",
)
_COST_KEYS = ("infer_ms_per_sim", "peak_mem_mb")

#: Full flattened column set of :func:`load_runs`.
RUN_COLUMNS: tuple[str, ...] = (
    _SCALAR_KEYS
    + tuple(f"field_{k}" for k in _FIELD_KEYS)
    + tuple(f"coef_{k}" for k in _COEF_KEYS)
    + tuple(f"consistency_{k}" for k in _CONSISTENCY_KEYS)
    + tuple(f"cost_{k}" for k in _COST_KEYS)
    + ("gpu_hours", "train_size", "results_dir")
)

PER_SIM_COLUMNS: tuple[str, ...] = RUN_ID_COLUMNS + (
    "sim_name",
    "p_rel_l2",
    "tau_rel_l2",
    "p_mae",
    "tau_mae",
    "cl_head",
    "cd_head",
    "cl_int",
    "cd_int",
    "cl_true",
    "cd_true",
    "sym_residual",
    "antisym_cl_gap",
)

UQ_COLUMNS: tuple[str, ...] = (
    "ensemble_id",
    "model",
    "split",
    "method",
    "k",
    "target",
    "granularity",
    "score",
    "nominal",
    "coverage",
    "coverage_ci_lo",
    "coverage_ci_hi",
    "mean_width",
    "median_width",
    "ece",
    "n",
)

#: ``..._n400_...`` / ``n400`` in a run tag == training-set size (Figure 9).
_TRAIN_SIZE_RE = re.compile(r"(?:^|[_-])n(\d+)(?:[_-]|$)")


def _warn(message: str, strict: bool) -> None:
    if strict:
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _num(value: Any) -> float | None:
    """Coerce to float; ``None``/non-numeric become ``NaN`` (never an error)."""
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_metrics_file(path: str | Path, *, strict: bool = False) -> dict[str, Any] | None:
    """Parse one ``metrics.json``; ``None`` if unreadable or not a mapping."""
    p = Path(path)
    try:
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        _warn(f"{p}: no metrics.json (run incomplete?)", strict)
        return None
    except (json.JSONDecodeError, OSError) as exc:
        _warn(f"{p}: unreadable metrics.json ({exc})", strict)
        return None
    if not isinstance(payload, Mapping):
        _warn(f"{p}: metrics.json is not a JSON object", strict)
        return None
    return dict(payload)


def _flatten_metrics(payload: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key in _SCALAR_KEYS:
        row[key] = payload.get(key)
    if not row.get("run_id"):
        row["run_id"] = run_dir.name
    row["model"] = str(row.get("model") or "unknown")
    row["split"] = str(row.get("split") or "unknown")
    try:
        row["seed"] = int(row["seed"]) if row.get("seed") is not None else -1
    except (TypeError, ValueError):
        row["seed"] = -1
    tag = row.get("tag")
    row["tag"] = None if tag is None else str(tag)
    for key in ("params", "epochs", "n_sims"):
        val = row.get(key)
        row[key] = None if val is None else _num(val)
    row["train_time_s"] = _num(row.get("train_time_s"))
    row["device"] = str(row.get("device") or "")

    keysets = {
        "field": _FIELD_KEYS,
        "coef": _COEF_KEYS,
        "consistency": _CONSISTENCY_KEYS,
        "cost": _COST_KEYS,
    }
    for group in _GROUPS:
        sub = payload.get(group)
        sub = sub if isinstance(sub, Mapping) else {}
        for key in keysets[group]:
            row[f"{group}_{key}"] = _num(sub.get(key))
    row["results_dir"] = str(run_dir)
    return row


def _train_size_from(row: Mapping[str, Any], config: Mapping[str, Any] | None) -> float:
    """Training-set size for the data-efficiency figure, or NaN if unknown.

    Resolution order (first hit wins):

    1. an ``n<digits>`` token in the run ``tag`` (e.g. ``n400``);
    2. the same token in the ``run_id`` (data-efficiency runs are named
       ``{model}_full_n{size}_s{seed}``, so this covers them directly);
    3. an explicit size key under ``config["data"]``
       (``n_train``/``train_size``/``subset_n``/``max_train``);
    4. the ``n_train`` recorded in the run's own split manifest -- this is what
       lets the *free* ``full`` runs (700-sim train, no ``n`` token anywhere)
       land at the right x on Figure 9.
    """
    tag = row.get("tag")
    if tag:
        match = _TRAIN_SIZE_RE.search(str(tag))
        if match:
            return float(match.group(1))
    run_id = str(row.get("run_id") or "")
    match = _TRAIN_SIZE_RE.search(run_id)
    if match:
        return float(match.group(1))
    if config:
        data_cfg = config.get("data")
        if isinstance(data_cfg, Mapping):
            for key in ("n_train", "train_size", "subset_n", "max_train"):
                if data_cfg.get(key) is not None:
                    return _num(data_cfg[key])
            n_split = _n_train_from_split(data_cfg)
            if n_split is not None:
                return n_split
    return float("nan")


def _n_train_from_split(data_cfg: Mapping[str, Any]) -> float | None:
    """``n_train`` read from the run's split manifest, or ``None`` if unavailable.

    Prefers an explicit ``data.split_file`` (what the data-efficiency runs set),
    else ``data.splits_dir / <data.split>.json``.  All I/O errors are swallowed
    into ``None`` per this module's robustness contract -- a run whose manifest
    cannot be found simply has an unknown train size.
    """
    split_file = data_cfg.get("split_file")
    if not split_file:
        split = data_cfg.get("split")
        if not split:
            return None
        splits_dir = data_cfg.get("splits_dir") or "data/splits"
        split_file = Path(splits_dir) / f"{split}.json"
    try:
        with Path(split_file).open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    n = payload.get("n_train")
    if n is None:
        train = payload.get("train")
        n = len(train) if isinstance(train, (list, tuple)) else None
    if n is None:
        return None
    value = _num(n)
    return value if value == value else None  # drop NaN


def _read_config(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "config.yaml"
    if not path.exists():
        return None
    try:
        import yaml  # optional; only needed for config-derived train sizes
    except Exception:  # pragma: no cover - yaml is a hard dep of the project
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except Exception:
        return None
    return cfg if isinstance(cfg, Mapping) else None


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``gpu_hours`` and the shift ordinal columns used by several figures."""
    from .shift import SHIFT_RANK  # local import: shift imports nothing from here

    out = df.copy()
    if "train_time_s" in out.columns:
        out["gpu_hours"] = out["train_time_s"].astype(float) / 3600.0
    else:
        out["gpu_hours"] = float("nan")
    out["shift_rank"] = out["split"].map(SHIFT_RANK).astype(float)
    out["is_baseline"] = out["model"].isin(["constant", "ridge"])
    return out


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


def load_runs(
    results_dir: str | Path = "results",
    *,
    strict: bool = False,
    validate: bool = True,
    include: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Tidy frame of every run under ``results_dir`` that has a ``metrics.json``.

    Parameters
    ----------
    results_dir:
        Usually ``results/``.  Non-existent directory -> empty frame + warning.
    strict:
        Turn skips (missing/invalid metrics) into ``ValueError``.
    validate:
        Run ``src.eval.schema.check_metrics`` on each payload and *warn* about
        violations.  The row is still kept: a run written before a metric was
        added to the schema (e.g. the pre-``cd_head_spearman`` smoke run) is
        useful data with a hole in it, and :func:`_flatten_metrics` already
        renders a missing leaf as ``NaN``.  ``strict=True`` promotes the warning
        to an error.
    include:
        Optional allow-list of ``run_id`` values.

    Returns
    -------
    pandas.DataFrame
        One row per run, columns :data:`RUN_COLUMNS`, sorted by
        ``(model, split, seed, tag)``.  Numeric metric columns are float with
        ``NaN`` for "not computed"; ``tag`` is ``None`` when absent.
    """
    root = Path(results_dir)
    if not root.is_dir():
        _warn(f"{root}: results directory does not exist", strict)
        return add_derived_columns(_empty(RUN_COLUMNS))

    checker = None
    if validate:
        try:
            from src.eval.schema import check_metrics as checker  # type: ignore
        except Exception:  # pragma: no cover - schema module is present in-repo
            checker = None

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if run_dir.name == "uq":
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            # Normal for a run still training -- config.yaml exists, metrics do not.
            continue
        payload = load_metrics_file(metrics_path, strict=strict)
        if payload is None:
            continue
        if checker is not None:
            errors = checker(payload)
            if errors:
                _warn(
                    f"{metrics_path}: schema violations (row kept, missing "
                    "leaves become NaN): " + "; ".join(errors[:4]),
                    strict,
                )
        row = _flatten_metrics(payload, run_dir)
        if include is not None and row["run_id"] not in set(include):
            continue
        row["train_size"] = _train_size_from(row, _read_config(run_dir))
        rows.append(row)

    if not rows:
        return add_derived_columns(_empty(RUN_COLUMNS))

    df = pd.DataFrame(rows)
    for col in RUN_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    df = df[list(RUN_COLUMNS)]
    df = df.sort_values(
        ["model", "split", "seed", "tag"], na_position="first", kind="stable"
    ).reset_index(drop=True)
    return add_derived_columns(df)


def load_per_sim(
    run_dir: str | Path, *, strict: bool = False
) -> pd.DataFrame | None:
    """Read one run's ``per_sim.csv``; ``None`` when the run has not written one."""
    d = Path(run_dir)
    path = d / "per_sim.csv"
    if not path.exists():
        _warn(f"{path}: no per_sim.csv", strict)
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        _warn(f"{path}: unreadable per_sim.csv ({exc})", strict)
        return None
    if df.empty:
        return None
    return df


def load_all_per_sim(
    results_dir: str | Path = "results",
    *,
    runs: pd.DataFrame | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """Long frame of every run's per-simulation rows, keyed by run identity.

    ``runs`` (the output of :func:`load_runs`) supplies the identity columns; if
    omitted it is loaded.  Adds derived per-simulation error columns:
    ``cd_int_err``/``cd_head_err``/``cl_int_err``/``cl_head_err`` (absolute),
    their ``*_rel`` relative counterparts, and ``fsc_cd``/``fsc_cl`` --- the
    label-free head-vs-integration disagreement of spec section 5.4.
    """
    runs_df = load_runs(results_dir, strict=strict) if runs is None else runs
    frames: list[pd.DataFrame] = []
    for _, meta in runs_df.iterrows():
        run_dir = Path(meta.get("results_dir") or Path(results_dir) / str(meta["run_id"]))
        per_sim = load_per_sim(run_dir, strict=False)
        if per_sim is None:
            continue
        for col in RUN_ID_COLUMNS:
            per_sim[col] = meta.get(col)
        frames.append(per_sim)

    if not frames:
        return _add_per_sim_derived(_empty(PER_SIM_COLUMNS))

    df = pd.concat(frames, ignore_index=True, sort=False)
    lead = [c for c in RUN_ID_COLUMNS if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]
    return _add_per_sim_derived(df)


def _add_per_sim_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def col(name: str) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce")
        return pd.Series([float("nan")] * len(out), index=out.index, dtype=float)

    for coef in ("cd", "cl"):
        truth = col(f"{coef}_true")
        denom = truth.abs().replace(0.0, float("nan"))
        for route in ("int", "head"):
            err = (col(f"{coef}_{route}") - truth).abs()
            out[f"{coef}_{route}_err"] = err
            out[f"{coef}_{route}_rel"] = err / denom
        fsc = (col(f"{coef}_int") - col(f"{coef}_head")).abs()
        out[f"fsc_{coef}"] = fsc
        out[f"fsc_rel_{coef}"] = fsc / denom
    return out


# --------------------------------------------------------------------------
# UQ (results/uq/*.json)
# --------------------------------------------------------------------------


def load_uq_reports(
    uq_dir: str | Path = "results/uq", *, strict: bool = False
) -> list[dict[str, Any]]:
    """Raw UQ payloads.  Empty list (with a warning) when none exist yet."""
    root = Path(uq_dir)
    if not root.is_dir():
        _warn(f"{root}: no UQ results yet", strict)
        return []
    reports: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            _warn(f"{path}: unreadable UQ report ({exc})", strict)
            continue
        if not isinstance(payload, Mapping):
            _warn(f"{path}: UQ report is not a JSON object", strict)
            continue
        payload = dict(payload)
        payload.setdefault("ensemble_id", path.stem)
        payload.setdefault("path", str(path))
        reports.append(payload)
    return reports


def load_uq(
    uq_dir: str | Path = "results/uq",
    *,
    strict: bool = False,
    reports: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Tidy frame: one row per (UQ record, nominal level).

    Understands the ``uq-1`` payload written by ``src/uq/coverage.py``
    (``records`` list of per-(target, granularity) level sweeps) and falls back
    to the flat ``coef``/``field`` view required by ``src/eval/schema.py`` when
    ``records`` is absent -- both live in the same file, so either is enough.
    """
    payloads = list(load_uq_reports(uq_dir, strict=strict)) if reports is None else list(reports)
    rows: list[dict[str, Any]] = []
    for rep in payloads:
        base = {
            "ensemble_id": str(rep.get("ensemble_id", "")),
            "model": str(rep.get("model", "") or "unknown"),
            "split": str(rep.get("split", "") or "unknown"),
            "method": str(rep.get("method", "") or "unknown"),
            "k": _num(rep.get("k")),
        }
        records = rep.get("records")
        if isinstance(records, (list, tuple)) and records:
            for rec in records:
                if not isinstance(rec, Mapping):
                    continue
                rec_base = dict(base)
                rec_base.update(
                    {
                        "target": str(rec.get("target", "")),
                        "granularity": str(rec.get("granularity", "")),
                        "score": str(rec.get("score", "")),
                        "ece": _num(rec.get("ece")),
                    }
                )
                for level in rec.get("levels", []) or []:
                    if not isinstance(level, Mapping):
                        continue
                    row = dict(rec_base)
                    row.update(
                        {
                            "nominal": _num(level.get("nominal")),
                            "coverage": _num(level.get("coverage")),
                            "coverage_ci_lo": _num(level.get("coverage_ci_lo")),
                            "coverage_ci_hi": _num(level.get("coverage_ci_hi")),
                            "mean_width": _num(level.get("mean_width")),
                            "median_width": _num(level.get("median_width")),
                            "n": _num(level.get("n")),
                        }
                    )
                    rows.append(row)
            continue

        # Flat A4-schema view: {"coef": {target: {"0.9": {...}}}, "field": {...}}
        for group, granularity in (("coef", "coefficient"), ("field", "field_quantile")):
            bucket = rep.get(group)
            if not isinstance(bucket, Mapping):
                continue
            for target, per_level in bucket.items():
                if not isinstance(per_level, Mapping):
                    continue
                for nominal, stats in per_level.items():
                    if not isinstance(stats, Mapping):
                        continue
                    row = dict(base)
                    row.update(
                        {
                            "target": str(target),
                            "granularity": granularity,
                            "score": str(rep.get("score", "")),
                            "ece": float("nan"),
                            "nominal": _num(nominal),
                            "coverage": _num(stats.get("coverage")),
                            "coverage_ci_lo": float("nan"),
                            "coverage_ci_hi": float("nan"),
                            "mean_width": _num(stats.get("width_mean")),
                            "median_width": _num(stats.get("width_median")),
                            "n": float("nan"),
                        }
                    )
                    rows.append(row)

    if not rows:
        return _empty(UQ_COLUMNS)
    df = pd.DataFrame(rows)
    for col in UQ_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    return df[list(UQ_COLUMNS)].sort_values(
        ["model", "split", "target", "granularity", "nominal"], kind="stable"
    ).reset_index(drop=True)


def reliability_frame(
    reports: Sequence[Mapping[str, Any]],
    *,
    target: str | None = None,
    granularity: str = "coefficient",
) -> pd.DataFrame:
    """Nominal-vs-empirical curves for the reliability diagram (Figure 6).

    Prefers each record's dense ``reliability`` curve; falls back to the three
    reported nominal levels when the dense curve is absent.
    """
    rows: list[dict[str, Any]] = []
    for rep in reports:
        for rec in rep.get("records", []) or []:
            if not isinstance(rec, Mapping):
                continue
            if granularity and str(rec.get("granularity")) != granularity:
                continue
            if target and str(rec.get("target")) != target:
                continue
            curve = rec.get("reliability")
            if isinstance(curve, Mapping) and curve.get("nominal"):
                pairs = zip(curve["nominal"], curve.get("empirical", []))
            else:
                levels = rec.get("levels", []) or []
                pairs = [
                    (lv.get("nominal"), lv.get("coverage"))
                    for lv in levels
                    if isinstance(lv, Mapping)
                ]
            for nominal, empirical in pairs:
                rows.append(
                    {
                        "ensemble_id": str(rep.get("ensemble_id", "")),
                        "model": str(rep.get("model", "") or "unknown"),
                        "split": str(rep.get("split", "") or "unknown"),
                        "method": str(rep.get("method", "") or "unknown"),
                        "target": str(rec.get("target", "")),
                        "granularity": str(rec.get("granularity", "")),
                        "nominal": _num(nominal),
                        "empirical": _num(empirical),
                    }
                )
    cols = (
        "ensemble_id",
        "model",
        "split",
        "method",
        "target",
        "granularity",
        "nominal",
        "empirical",
    )
    if not rows:
        return _empty(cols)
    return pd.DataFrame(rows)[list(cols)].sort_values(
        ["model", "split", "method", "nominal"], kind="stable"
    ).reset_index(drop=True)


def pivot_best(
    df: pd.DataFrame, value: str, *, index: str = "model", columns: str = "split"
) -> pd.DataFrame:
    """Mean of ``value`` over seeds, as an ``index x columns`` matrix.

    Convenience shared by Figure 4's fallback path and Table 2.
    """
    if df.empty or value not in df.columns:
        return pd.DataFrame()
    return df.pivot_table(index=index, columns=columns, values=value, aggfunc="mean")
