"""Sequential sweep runner (PLAN.md Phase 3).

    .venv/Scripts/python.exe -m scripts.sweep --spec configs/sweeps/core.yaml
    .venv/Scripts/python.exe -m scripts.sweep --spec configs/sweeps/core.yaml --max-seconds 39600

Runs are executed **strictly one at a time** -- the 4 GB P2000 cannot hold two
(CONTEXT.md 1, PLAN.md "Runs are sequenced, never parallel").  A run whose
``results/<run_id>/metrics.json`` already exists is skipped, so an interrupted
sweep is resumed simply by re-running the same command.

``--max-seconds`` is a wall-clock budget for hosted sessions (Kaggle's 12 h
limit; ``kaggle/session_driver.py`` passes it).  The remaining budget is checked
*before* each run starts, and is also enforced as a timeout on the child
process; per-epoch checkpointing (CONTEXT.md 10) makes a killed run free to
resume in the next session.

Spec format (either style, or both)::

    name: core
    defaults:
      overrides: [seed=0]
    runs:
      - config: configs/gnn.yaml
        overrides: [split=full]
    matrix:
      config: [configs/gnn.yaml, configs/sdf_fno.yaml]
      overrides:
        split: [full, scarce]
        seed: [0]
"""

from __future__ import annotations

import argparse
import itertools
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config import get_in, load_config  # noqa: E402
from src.utils.io import build_run_id  # noqa: E402

__all__ = ["load_spec", "expand_spec", "run_id_for", "main"]


# --------------------------------------------------------------------------- #
# Spec handling
# --------------------------------------------------------------------------- #
def load_spec(path: str | Path) -> dict[str, Any]:
    """Read a sweep spec YAML file."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"sweep spec not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh) or {}
    if not isinstance(spec, Mapping):
        raise SystemExit(f"sweep spec {p} must be a YAML mapping")
    return dict(spec)


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v]


def expand_spec(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand a spec into an ordered list of ``{'config', 'overrides', 'name'}``.

    ``matrix`` entries are expanded as a cartesian product in declaration order
    (configs outermost), then ``runs`` entries are appended verbatim.  Defaults
    from ``defaults.overrides`` are prepended to every run so an explicit
    per-run override always wins (later keys overwrite earlier ones).
    """
    base = _as_list(get_in(spec, "defaults.overrides"))
    out: list[dict[str, Any]] = []

    matrix = spec.get("matrix")
    if matrix:
        configs = _as_list(matrix.get("config") or matrix.get("configs"))
        axes: dict[str, list[Any]] = dict(matrix.get("overrides") or {})
        keys = list(axes)
        combos = list(itertools.product(*[axes[k] for k in keys])) if keys else [()]
        for cfg_path in configs:
            for combo in combos:
                ov = base + [f"{k}={v}" for k, v in zip(keys, combo)]
                out.append({"config": cfg_path, "overrides": ov})

    for entry in spec.get("runs") or []:
        if isinstance(entry, str):
            entry = {"config": entry}
        cfg_path = entry.get("config")
        if not cfg_path:
            raise SystemExit(f"sweep run entry has no 'config': {entry!r}")
        out.append({
            "config": str(cfg_path),
            "overrides": base + _as_list(entry.get("overrides")),
            "name": entry.get("name"),
        })
    return out


def run_id_for(config_path: str, overrides: Sequence[str]) -> str:
    """Resolve the frozen ``{model}_{split}_s{seed}[_{tag}]`` id for a run."""
    cfg = load_config(config_path, list(overrides))
    return build_run_id(
        get_in(cfg, "model.name"),
        get_in(cfg, "data.split"),
        get_in(cfg, "seed", 0),
        cfg.get("tag"),
    )


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def _fmt(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="scripts/sweep.py",
        description="Run a list of training runs sequentially, skipping finished ones.",
    )
    p.add_argument("--spec", required=True, action="append",
                   help="sweep spec YAML, e.g. configs/sweeps/core.yaml. Repeatable: "
                        "pass --spec twice to drain several specs in order in one run "
                        "(PLAN_PHASE3 1.3, e.g. data-eff then ablations).")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="wall-clock budget; stops cleanly before starting a run "
                        "that cannot fit, and times out a running child")
    p.add_argument("--reserve-seconds", type=float, default=0.0,
                   help="keep this much of the budget in hand (e.g. for exporting results)")
    p.add_argument("--min-run-seconds", type=float, default=60.0,
                   help="do not start a new run with less than this left")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    p.add_argument("--force", action="store_true", help="re-run even if metrics.json exists")
    p.add_argument("--filter", default=None, help="substring filter on run_id")
    p.add_argument("--limit", type=int, default=None, help="run at most N runs")
    p.add_argument("--results-root", default="results")
    p.add_argument("--extra", nargs="*", default=[],
                   help="extra overrides appended to every run")
    p.add_argument("--train-args", default="",
                   help="one quoted string of flags forwarded verbatim to "
                        "scripts/train.py, e.g. --train-args=\"--device cpu --no-eval\" "
                        "(a single string, because argparse will not absorb "
                        "dash-prefixed values into a list)")
    args = p.parse_args(argv)

    train_flags = shlex.split(args.train_args) if args.train_args else []
    specs = list(args.spec)
    runs: list[dict[str, Any]] = []
    for sp in specs:
        runs.extend(expand_spec(load_spec(sp)))
    if len(specs) == 1:
        name = load_spec(specs[0]).get("name", Path(specs[0]).stem)
    else:
        name = "+".join(Path(sp).stem for sp in specs)
    t_start = time.perf_counter()
    budget = None if args.max_seconds is None else float(args.max_seconds) - float(args.reserve_seconds)

    # Keep the child's output root identical to the one we test for skipping,
    # otherwise a sweep re-runs everything forever.
    extra = list(args.extra)
    if not any(o.startswith("paths.results_root=") for o in extra):
        extra.append(f"paths.results_root={args.results_root}")

    plan: list[dict[str, Any]] = []
    for entry in runs:
        overrides = list(entry["overrides"]) + extra
        try:
            rid = run_id_for(entry["config"], overrides)
        except Exception as exc:
            print(f"[sweep] SKIP unresolvable run {entry['config']} {overrides}: {exc}")
            continue
        if args.filter and args.filter not in rid:
            continue
        done = (Path(args.results_root) / rid / "metrics.json").is_file()
        plan.append({"run_id": rid, "config": entry["config"],
                     "overrides": overrides, "done": done})

    todo = [r for r in plan if args.force or not r["done"]]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"[sweep] {name}: {len(plan)} runs, {len(plan) - len(todo)} already complete, "
          f"{len(todo)} to run"
          + (f", budget {_fmt(budget)}" if budget is not None else ""))
    for r in plan:
        mark = "done" if r["done"] else "todo"
        print(f"  [{mark}] {r['run_id']}  ({r['config']} {' '.join(r['overrides'])})")
    if args.dry_run:
        return 0

    n_ok = n_fail = n_skipped = 0
    for i, r in enumerate(todo, 1):
        elapsed = time.perf_counter() - t_start
        remaining = None if budget is None else budget - elapsed
        if remaining is not None and remaining <= args.min_run_seconds:
            n_skipped = len(todo) - i + 1
            print(f"[sweep] budget exhausted ({_fmt(elapsed)} used); "
                  f"stopping cleanly with {n_skipped} run(s) not started")
            break

        cmd = [sys.executable, "-m", "scripts.train", "--config", r["config"],
               *train_flags, *r["overrides"]]
        print(f"\n[sweep] ({i}/{len(todo)}) {r['run_id']}"
              + (f"  [{_fmt(remaining)} left]" if remaining is not None else ""))
        print(f"[sweep] $ {' '.join(cmd)}", flush=True)

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, cwd=str(_ROOT), timeout=remaining, check=False)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            print(f"[sweep] {r['run_id']} hit the wall-clock budget and was stopped; "
                  "its last.pt will resume in the next session")
            n_skipped = len(todo) - i + 1
            break
        except KeyboardInterrupt:  # pragma: no cover
            print("[sweep] interrupted by user")
            return 130
        dt = time.perf_counter() - t0

        if rc == 0:
            n_ok += 1
            print(f"[sweep] OK {r['run_id']} in {_fmt(dt)}")
        else:
            n_fail += 1
            print(f"[sweep] FAILED {r['run_id']} (exit {rc}) after {_fmt(dt)}")

    print(f"\n[sweep] {name} finished: {n_ok} ok, {n_fail} failed, "
          f"{n_skipped} not started, {_fmt(time.perf_counter() - t_start)} elapsed")
    return 1 if n_fail else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
