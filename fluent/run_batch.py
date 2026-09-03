"""fluent/run_batch.py -- standalone, resumable batch driver for the Fluent runs.

Runs the 2D RANS verification campaign UNATTENDED as an ordinary background
process. It drives `fluent.exe` directly (NOT the Ansys MCP), one solve at a
time (a single solver licence means concurrent jobs collide), and is safe to
re-launch: a case whose results already exist on disk is skipped.

What it does, per (case, turbulence-model):
  1. generate the mesh if `<case>.msh` is missing (runs the case's own
     `<case>_mesh.cmd`, which is what fluent/make_cases.py wrote);
  2. skip if the case is already complete (its `<case>_<model>_coeffs.out` has a
     final data row AND its `<case>_<model>.cas.h5` exists);
  3. otherwise launch, BLOCKING:
        <fluent.exe> 2ddp -g -t<procs> -i <case>_<model>.jou -wait
  4. append a timestamped line to logs/fluent_batch.log and continue to the next
     case even if this one errored or timed out.

Run set:
  * default  = every case in fluent/cases_to_run.json, Spalart-Allmaras only
               (the 3-level grid study runs first, then the 24 active-learning
               cases; SA is the like-for-like model with AirfRANS).
  * --models = comma list applied to every case (e.g. "sa,sst").
  * --sst-gridstudy = additionally run k-omega SST for the three gridstudy_*
    cases (the SA<->SST spread and the two-model GCI band the plan asks for),
    without running SST on all 24 AL cases.

The meshes, journals and mesh.cmds are produced by fluent/make_cases.py; run
that (with --write-mesh) before, or let this script build any missing mesh.

Usage
-----
    # see the plan without running anything
    .venv/Scripts/python.exe fluent/run_batch.py --dry-run

    # the full unattended campaign (SA on all 27 + SST on the 3 gridstudy)
    .venv/Scripts/python.exe fluent/run_batch.py --sst-gridstudy

    # one case only
    .venv/Scripts/python.exe fluent/run_batch.py --only gridstudy_naca0012_re3e6_a5_L2

    # quick smoke test of the machinery (caps every /solve/iterate to N)
    .venv/Scripts/python.exe fluent/run_batch.py --only gridstudy_naca0012_re3e6_a5_L2 --smoke-iters 120
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CASES_DIR = HERE / "cases"
MANIFEST = CASES_DIR / "MANIFEST.json"
CASES_JSON = HERE / "cases_to_run.json"
LOG_DIR = REPO / "logs"
LOG_FILE = LOG_DIR / "fluent_batch.log"

# Default v211 solver. Override with --fluent or the FLUENT_EXE env var.
DEFAULT_FLUENT = (r"D:\softwares\ANSYS 2021R1\ANSYS Inc\v211"
                  r"\fluent\ntbin\win64\fluent.exe")
# Where Fluent drops its own scratch (cleanup .bat, root transcript). The
# journals write every real output to the case dir via absolute paths, so this
# only keeps clutter out of the repo root.
WORKDIR = HERE / "runs"

GRIDSTUDY_PREFIX = "gridstudy_"


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    line = "%s  %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# run-plan construction
# ---------------------------------------------------------------------------

def load_cases():
    """Return the ordered list of case dicts (case_id, journals, mesh_cmd, ...).

    Prefers cases/MANIFEST.json (written by make_cases.py, has journals + the
    exact mesh command); falls back to cases_to_run.json if the manifest is
    missing, deriving the paths by convention.
    """
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        return m["cases"]
    raw = json.loads(CASES_JSON.read_text(encoding="utf-8"))
    cases = raw["cases"] if isinstance(raw, dict) else raw
    out = []
    for c in cases:
        cid = c["case_id"]
        out.append(dict(case_id=cid, models=c.get("models", ["sa", "sst"]),
                        journals=[], mesh_cmd=None))
    return out


def build_plan(cases, models, sst_gridstudy, only):
    """Return an ordered list of (case_dict, model) pairs to run."""
    plan = []
    for c in cases:
        cid = c["case_id"]
        if only and cid not in only:
            continue
        mods = list(models)
        if sst_gridstudy and cid.startswith(GRIDSTUDY_PREFIX) and "sst" not in mods:
            mods.append("sst")
        for mdl in mods:
            plan.append((c, mdl))
    return plan


# ---------------------------------------------------------------------------
# per-case file helpers
# ---------------------------------------------------------------------------

def case_paths(case_id, model, variant=""):
    cdir = CASES_DIR / case_id
    label = model if not variant else "%s_%s" % (model, variant)
    return dict(
        cdir=cdir,
        mesh=cdir / (case_id + ".msh"),
        journal=cdir / ("%s_%s.jou" % (case_id, label)),
        coeffs=cdir / ("%s_%s_coeffs.out" % (case_id, label)),
        cas=cdir / ("%s_%s.cas.h5" % (case_id, label)),
        mesh_cmd=cdir / (case_id + "_mesh.cmd"),
    )


def final_coeffs_row(coeffs_path: Path):
    """Return (iteration, cd, cl) of the last data row in a Fluent report file,
    or None if the file has no numeric data row yet."""
    if not coeffs_path.exists():
        return None
    last = None
    for line in coeffs_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                it = int(parts[0])
            except ValueError:
                continue
            vals = []
            for p in parts[1:]:
                try:
                    vals.append(float(p))
                except ValueError:
                    vals = None
                    break
            if vals:
                last = (it, vals)
    return last


def is_complete(p) -> bool:
    """A case+model counts as done (skip it) when its coefficient history has a
    final data row AND the case file was written -- i.e. the solve reached the
    save at the end of the journal. Both are required: a killed run can leave a
    coeffs.out with rows but no .cas.h5."""
    return final_coeffs_row(p["coeffs"]) is not None and p["cas"].exists()


# ---------------------------------------------------------------------------
# mesh generation (runs the case's own mesh.cmd)
# ---------------------------------------------------------------------------

def ensure_mesh(case, p) -> bool:
    if p["mesh"].exists():
        return True
    cmd_str = None
    if case.get("mesh_cmd"):
        cmd_str = case["mesh_cmd"]
    elif p["mesh_cmd"].exists():
        cmd_str = p["mesh_cmd"].read_text(encoding="utf-8").strip()
    if not cmd_str:
        log("  MESH ERROR: no mesh.cmd for %s and %s is missing"
            % (case["case_id"], p["mesh"].name))
        return False
    tokens = shlex.split(cmd_str, posix=True)
    # Run with the interpreter that is running this script (the .venv python),
    # regardless of how the stored command spelled the python path.
    if tokens and tokens[0].lower().endswith("python.exe"):
        tokens[0] = sys.executable
    log("  MESH  %s" % cmd_str)
    r = subprocess.run(tokens, cwd=str(REPO))
    if r.returncode != 0 or not p["mesh"].exists():
        log("  MESH ERROR: mesh.cmd returned %d for %s" % (r.returncode, case["case_id"]))
        return False
    return True


# ---------------------------------------------------------------------------
# smoke journal (validation only): cap every /solve/iterate to N
# ---------------------------------------------------------------------------

_ITER_RE = re.compile(r"^(/solve/iterate\s+)\d+\s*$")


def make_smoke_journal(journal: Path, n: int) -> Path:
    text = journal.read_text(encoding="utf-8").splitlines()
    out = []
    for ln in text:
        m = _ITER_RE.match(ln.strip())
        out.append("/solve/iterate %d" % n if m else ln)
    smoke = journal.with_name(journal.stem + "_smoke.jou")
    smoke.write_text("\n".join(out) + "\n", encoding="utf-8")
    return smoke


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------

def run_one(case, model, fluent_exe, procs, dim, timeout_min, smoke_iters, variant=""):
    cid = case["case_id"]
    p = case_paths(cid, model, variant)
    mlabel = model if not variant else "%s_%s" % (model, variant)

    if not p["journal"].exists():
        log("  SKIP %s [%s]: journal %s missing (run make_cases.py)"
            % (cid, mlabel, p["journal"].name))
        return "missing-journal"

    if smoke_iters is None and is_complete(p):
        row = final_coeffs_row(p["coeffs"])
        log("  SKIP %s [%s]: already complete (iter %d, cd=%.6g, cl=%.6g)"
            % (cid, model, row[0], row[1][0], row[1][1] if len(row[1]) > 1 else float("nan")))
        return "skip"

    if not ensure_mesh(case, p):
        return "mesh-error"

    journal = p["journal"]
    if smoke_iters is not None:
        journal = make_smoke_journal(journal, smoke_iters)
        log("  SMOKE %s [%s]: capped iterations to %d (%s)"
            % (cid, model, smoke_iters, journal.name))

    cmd = [fluent_exe, dim, "-g", "-t%d" % procs, "-i", str(journal), "-wait"]
    log("  RUN  %s [%s]  %s" % (cid, model, " ".join(cmd)))
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(WORKDIR),
                           timeout=timeout_min * 60 if timeout_min else None)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        log("  ERROR %s [%s]: TIMEOUT after %d min -- killed, continuing"
            % (cid, model, timeout_min))
        return "timeout"
    except Exception as exc:                          # noqa: BLE001 - unattended
        log("  ERROR %s [%s]: launch failed: %r -- continuing" % (cid, model, exc))
        return "launch-error"
    dt = (time.time() - t0) / 60.0

    row = final_coeffs_row(p["coeffs"])
    if rc == 0 and row is not None:
        cd = row[1][0]
        cl = row[1][1] if len(row[1]) > 1 else float("nan")
        log("  DONE %s [%s]  rc=0  %.1f min  iter=%d  cd=%.6g  cl=%.6g"
            % (cid, model, dt, row[0], cd, cl))
        return "done"
    log("  ERROR %s [%s]: rc=%d after %.1f min, coeffs row=%s -- continuing"
        % (cid, model, rc, dt, "yes" if row else "none"))
    return "error"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="sa",
                    help="comma list of turbulence models for every case (default: sa)")
    ap.add_argument("--sst-gridstudy", action="store_true",
                    help="also run k-omega SST for the three gridstudy_* cases")
    ap.add_argument("--only", action="append", default=None,
                    help="restrict to these case_ids (repeatable, or comma-separated)")
    ap.add_argument("--only-file", default=None,
                    help="restrict to the case_ids listed one-per-line in this file")
    ap.add_argument("--variant", default="",
                    help="solver variant suffix (e.g. r1): uses journals / outputs "
                         "named <case>_<model>_<variant> (see make_cases.py --variant)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (with skip/run status) and exit")
    ap.add_argument("--smoke-iters", type=int, default=None,
                    help="VALIDATION ONLY: cap every /solve/iterate to N and never "
                         "skip -- proves the fluent.exe launch + logging path quickly")
    ap.add_argument("--fluent", default=os.environ.get("FLUENT_EXE", DEFAULT_FLUENT),
                    help="path to fluent.exe")
    ap.add_argument("--processors", type=int, default=6, help="parallel cores (default 6)")
    ap.add_argument("--dim", default="2ddp", help="Fluent dimension flag (default 2ddp)")
    ap.add_argument("--timeout-min", type=int, default=180,
                    help="per-case wall-clock cap in minutes; 0 = no limit (default 180)")
    ap.add_argument("--log-file", default=None,
                    help="append the batch log here instead of logs/fluent_batch.log")
    args = ap.parse_args(argv)

    if args.log_file:
        global LOG_FILE
        LOG_FILE = Path(args.log_file)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    only = None
    if args.only:
        only = set()
        for tok in args.only:
            only.update(t for t in tok.split(",") if t)
    if args.only_file:
        only = only or set()
        for line in Path(args.only_file).read_text(encoding="utf-8").splitlines():
            cid = line.strip()
            if cid and not cid.startswith("#"):
                only.add(cid)

    cases = load_cases()
    plan = build_plan(cases, models, args.sst_gridstudy, only)
    if not plan:
        log("nothing to run (check --only / --models)")
        return 1

    log("=" * 70)
    log("BATCH START  %d run(s)  models=%s  sst_gridstudy=%s  procs=%d  smoke=%s"
        % (len(plan), models, args.sst_gridstudy, args.processors, args.smoke_iters))
    log("fluent: %s" % args.fluent)

    # Show the plan (and, for a real run, the skip/run status of each).
    for case, model in plan:
        p = case_paths(case["case_id"], model, args.variant)
        if args.smoke_iters is not None:
            status = "SMOKE-RUN"
        elif is_complete(p):
            status = "skip (complete)"
        elif not p["mesh"].exists():
            status = "RUN (mesh will be built)"
        else:
            status = "RUN"
        log("  PLAN  %-34s [%s]  -> %s" % (case["case_id"], model, status))

    if args.dry_run:
        log("dry-run: nothing executed")
        return 0

    if not Path(args.fluent).exists():
        log("FATAL: fluent.exe not found at %s (use --fluent or FLUENT_EXE)" % args.fluent)
        return 2

    tally = {}
    for case, model in plan:
        res = run_one(case, model, args.fluent, args.processors, args.dim,
                      args.timeout_min, args.smoke_iters, args.variant)
        tally[res] = tally.get(res, 0) + 1

    log("BATCH DONE  " + "  ".join("%s=%d" % (k, v) for k, v in sorted(tally.items())))
    log("=" * 70)
    # Non-zero exit if anything errored, so a wrapping scheduler can notice.
    bad = sum(v for k, v in tally.items() if k in ("error", "timeout",
                                                   "mesh-error", "launch-error"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
