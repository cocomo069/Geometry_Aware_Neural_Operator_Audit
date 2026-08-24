"""Atomic JSON/CSV writers and run-directory helpers.

Every artefact this project commits (``metrics.json``, ``history.csv``,
``config.yaml``) is written through an atomic temp-file + ``os.replace`` so a
killed training run never leaves a half-written file behind (CONTEXT.md 9/10;
runs on this machine are expected to be interrupted).

Run identity (FROZEN, CONTEXT.md 9)::

    run_id = {model}_{split}_s{seed}[_{tag}]
    results/<run_id>/{config.yaml,metrics.json,history.csv}
    checkpoints/<run_id>/{last,best}.pt
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ensure_dir",
    "atomic_write_text",
    "atomic_write_bytes",
    "write_json",
    "read_json",
    "write_csv",
    "append_csv_row",
    "read_csv",
    "build_run_id",
    "results_dir",
    "checkpoint_dir",
    "to_jsonable",
]

_SAFE = re.compile(r"[^0-9A-Za-z._-]+")


def ensure_dir(path: str | Path) -> Path:
    """``mkdir -p`` for a directory path; returns it as a :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Atomic primitives
# --------------------------------------------------------------------------- #
def _atomic_write(path: str | Path, data: str | bytes, mode: str) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    binary = "b" in mode
    fd, tmp = tempfile.mkstemp(
        prefix=p.name + ".", suffix=".tmp", dir=str(p.parent)
    )
    try:
        with os.fdopen(fd, mode, **({} if binary else {"encoding": "utf-8", "newline": ""})) as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)  # atomic on Windows and POSIX
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:  # pragma: no cover
            pass
        raise
    return p


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write ``text`` to ``path`` atomically (UTF-8, no newline translation)."""
    return _atomic_write(path, text, "w")


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` atomically."""
    return _atomic_write(path, data, "wb")


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/torch scalars and arrays to plain Python.

    Non-finite floats become ``None`` so the emitted file is *strict* JSON
    (``NaN``/``Infinity`` are not valid JSON and break other tooling).
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return obj.as_posix()
    # numpy / torch scalars and arrays, without importing either eagerly
    if hasattr(obj, "detach"):  # torch.Tensor
        obj = obj.detach().cpu()
    item = getattr(obj, "item", None)
    if item is not None and getattr(obj, "ndim", None) == 0:
        return to_jsonable(item())
    tolist = getattr(obj, "tolist", None)
    if tolist is not None and not isinstance(obj, (list, tuple)):
        return to_jsonable(tolist())
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return obj


def write_json(path: str | Path, obj: Any, *, indent: int = 2, sort_keys: bool = False) -> Path:
    """Atomically write ``obj`` as JSON (numpy/torch aware, strict output)."""
    text = json.dumps(to_jsonable(obj), indent=indent, sort_keys=sort_keys, allow_nan=False)
    return atomic_write_text(path, text + "\n")


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]],
              fieldnames: Sequence[str] | None = None) -> Path:
    """Atomically write a list of dict rows as CSV (header from first row)."""
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    import io as _io

    buf = _io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(fieldnames), extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _csv_val(row.get(k)) for k in fieldnames})
    return atomic_write_text(path, buf.getvalue())


def append_csv_row(path: str | Path, row: Mapping[str, Any],
                   fieldnames: Sequence[str] | None = None) -> Path:
    """Append one row, writing the header if the file does not exist yet.

    Appending is a single ``write``+``flush``+``fsync`` on a line-sized buffer,
    which is atomic enough in practice; the whole-file rewrite of
    :func:`write_csv` would be O(epochs^2) over a 400-epoch run.
    """
    p = Path(path)
    ensure_dir(p.parent)
    names = list(fieldnames) if fieldnames is not None else list(row.keys())
    exists = p.is_file() and p.stat().st_size > 0
    import io as _io

    buf = _io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=names, extrasaction="ignore",
                            lineterminator="\n")
    if not exists:
        writer.writeheader()
    writer.writerow({k: _csv_val(row.get(k)) for k in names})
    with p.open("a", encoding="utf-8", newline="") as fh:
        fh.write(buf.getvalue())
        fh.flush()
        os.fsync(fh.fileno())
    return p


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV written by :func:`write_csv` / :func:`append_csv_row`."""
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _csv_val(v: Any) -> Any:
    v = to_jsonable(v)
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.10g}"
    return v


# --------------------------------------------------------------------------- #
# Run identity / directories  (FROZEN, CONTEXT.md 9)
# --------------------------------------------------------------------------- #
def _slug(value: Any) -> str:
    return _SAFE.sub("-", str(value)).strip("-")


def build_run_id(model: str, split: str, seed: int, tag: str | None = None) -> str:
    """``{model}_{split}_s{seed}[_{tag}]`` with unsafe characters slugged out."""
    if model is None or str(model) == "":
        raise ValueError("build_run_id: model name is required")
    if split is None or str(split) == "":
        raise ValueError("build_run_id: split name is required")
    rid = f"{_slug(model)}_{_slug(split)}_s{int(seed)}"
    if tag not in (None, "", "none"):
        rid = f"{rid}_{_slug(tag)}"
    return rid


def run_id_from_config(cfg: Mapping[str, Any]) -> str:
    """Build the run id from a resolved config (model.name, data.split, seed, tag)."""
    from src.utils.config import get_in  # local import: avoids cycles

    return build_run_id(
        get_in(cfg, "model.name"),
        get_in(cfg, "data.split"),
        get_in(cfg, "seed", 0),
        get_in(cfg, "tag"),
    )


def results_dir(run_id: str, root: str | Path = "results", *, create: bool = True) -> Path:
    """``results/<run_id>/`` (created by default)."""
    p = Path(root) / run_id
    return ensure_dir(p) if create else p


def checkpoint_dir(run_id: str, root: str | Path = "checkpoints", *, create: bool = True) -> Path:
    """``checkpoints/<run_id>/`` (created by default; gitignored)."""
    p = Path(root) / run_id
    return ensure_dir(p) if create else p


def iter_metrics(results_root: str | Path = "results") -> Iterable[tuple[str, dict]]:
    """Yield ``(run_id, metrics_dict)`` for every ``results/*/metrics.json``."""
    root = Path(results_root)
    if not root.is_dir():
        return
    for mp in sorted(root.glob("*/metrics.json")):
        try:
            yield mp.parent.name, read_json(mp)
        except (OSError, json.JSONDecodeError):  # pragma: no cover
            continue
