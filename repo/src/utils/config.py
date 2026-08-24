"""YAML configuration loading with dotted CLI overrides.

Frozen interface (CONTEXT.md 10):

    load_config(path, overrides: list[str]) -> dict

Plain YAML, no Hydra.  Overrides are ``key.path=value`` strings with automatic
type inference, e.g. ``train.lr=3e-4``, ``train.epochs=50``, ``split=full``,
``model.use_curvature=false``, ``train.weights.force=0.1``.

Bare (dot-free) keys listed in :data:`ALIASES` expand to their canonical dotted
path so the documented CLI form ``scripts/train.py --config configs/gnn.yaml
split=full seed=0`` works.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

__all__ = [
    "ALIASES",
    "DEFAULTS",
    "load_config",
    "save_config",
    "apply_overrides",
    "parse_override",
    "infer_type",
    "get_in",
    "set_in",
    "flatten",
    "deep_merge",
    "ConfigError",
]


class ConfigError(ValueError):
    """Raised for malformed config files or override strings."""


# --------------------------------------------------------------------------- #
# Aliases: convenience bare keys -> canonical dotted path
# --------------------------------------------------------------------------- #
ALIASES: dict[str, str] = {
    "split": "data.split",
    "seed": "seed",
    "tag": "tag",
    "epochs": "train.epochs",
    "lr": "train.lr",
    "batch_size": "train.batch_size",
    "device": "device",
    "model_name": "model.name",
}

#: Keys every config is guaranteed to have after :func:`load_config`.
DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "tag": None,
    "device": "auto",
    "model": {"name": None},
    "train": {
        "epochs": 400,
        "lr": 1.0e-3,
        "batch_size": 16,
        "weight_decay": 0.0,
        "optimizer": "adam",
        "scheduler": "cosine",
        "min_lr": 1.0e-6,
        "grad_clip": 1.0,
        "num_workers": 0,
        "val_every": 1,
        "val_metric": "val_loss",
        "val_metric_mode": "min",
        "weights": {"head": 0.1,  # lambda_H (D-016): always-on coef-head term
                    "force": 0.0, "sym": 0.0, "bc": 0.0, "div": 0.0},
    },
    "data": {
        "split": "full",
        "splits_dir": "data/splits",
        "processed_dir": "data/processed/airfrans",
        "norm_stats": "data/processed/airfrans/norm_stats.json",
    },
    "paths": {
        "results_root": "results",
        "checkpoints_root": "checkpoints",
    },
}

_INT_RE = re.compile(r"^[-+]?\d+$")
_FLOAT_RE = re.compile(r"^[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?$")
_TRUE = {"true", "yes", "on"}
_FALSE = {"false", "no", "off"}
_NONE = {"null", "none", "~", ""}


def infer_type(value: str) -> Any:
    """Infer a Python value from an override string.

    Handles ``3e-4`` (which PyYAML's 1.1 resolver leaves as a *string*), ints,
    floats, booleans, ``null``, inline JSON/YAML collections (``[1,2]``,
    ``{a: 1}``) and bare comma-separated lists (``a,b,c``).  Quoted values are
    always returned as strings with the quotes stripped.
    """
    if not isinstance(value, str):
        return value
    v = value.strip()

    # Explicitly quoted -> string, no inference.
    if len(v) >= 2 and v[0] == v[-1] and v[0] in {'"', "'"}:
        return v[1:-1]

    low = v.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    if low in _NONE:
        return None
    if _INT_RE.match(v):
        return int(v)
    if _FLOAT_RE.match(v):
        return float(v)
    if low in {"inf", "+inf", "-inf", "nan"}:
        return float(low)
    if v[0] in "[{":
        try:
            return yaml.safe_load(v)
        except yaml.YAMLError as exc:  # pragma: no cover - defensive
            raise ConfigError(
                f"cannot parse collection override {value!r}: {exc}"
            ) from exc
    if "," in v:
        return [infer_type(part) for part in v.split(",")]
    return v


def parse_override(item: str) -> tuple[str, Any]:
    """Split ``'train.lr=3e-4'`` into ``('train.lr', 0.0003)``.

    Bare keys present in :data:`ALIASES` are expanded to their dotted path.
    """
    if not isinstance(item, str) or "=" not in item:
        raise ConfigError(
            f"override {item!r} is not of the form key=value (e.g. train.lr=3e-4)"
        )
    key, _, raw = item.partition("=")
    key = key.strip()
    if not key:
        raise ConfigError(f"override {item!r} has an empty key")
    if "." not in key and key in ALIASES:
        key = ALIASES[key]
    return key, infer_type(raw)


def get_in(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Fetch ``cfg['a']['b']`` for ``dotted='a.b'``; ``default`` if absent."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def set_in(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``cfg['a']['b'] = value`` for ``dotted='a.b'``, creating dicts."""
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def deep_merge(base: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``other`` onto a deep copy of ``base``."""
    out = copy.deepcopy(dict(base))
    for k, v in other.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(
    cfg: dict[str, Any], overrides: Iterable[str] | None
) -> dict[str, Any]:
    """Apply a sequence of ``key=value`` override strings in order (in place)."""
    for item in overrides or []:
        key, value = parse_override(item)
        set_in(cfg, key, value)
    return cfg


def flatten(cfg: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config to ``{'train.lr': 0.001, ...}``."""
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def load_config(
    path: str | Path,
    overrides: Sequence[str] | None = None,
    *,
    defaults: Mapping[str, Any] | None = DEFAULTS,
) -> dict[str, Any]:
    """Load ``path`` (YAML), merge onto :data:`DEFAULTS`, apply ``overrides``.

    Parameters
    ----------
    path:
        Path to a YAML file, e.g. ``configs/gnn.yaml``.
    overrides:
        List of ``key=value`` strings; dotted keys index nested sections.
    defaults:
        Mapping merged *under* the file contents.  Pass ``None`` to disable.

    Returns
    -------
    dict
        The fully resolved config.  ``_meta.config_path`` records the source.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config {p} must contain a YAML mapping, got {type(raw).__name__}"
        )

    cfg = deep_merge(defaults, raw) if defaults else copy.deepcopy(raw)
    apply_overrides(cfg, overrides)

    meta = cfg.setdefault("_meta", {})
    meta["config_path"] = p.as_posix()
    meta["overrides"] = list(overrides or [])
    return cfg


def save_config(cfg: Mapping[str, Any], path: str | Path) -> Path:
    """Write the *resolved* config to ``path`` as YAML (atomic, parents made)."""
    from src.utils.io import atomic_write_text  # local import: avoids cycles

    text = yaml.safe_dump(_plain(cfg), sort_keys=False, default_flow_style=False)
    return atomic_write_text(path, text)


def _plain(obj: Any) -> Any:
    """Convert to plain YAML-serialisable containers (drops numpy/torch types)."""
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    if hasattr(obj, "item") and getattr(obj, "ndim", None) == 0:
        return obj.item()
    return obj
