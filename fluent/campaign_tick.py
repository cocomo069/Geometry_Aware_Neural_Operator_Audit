"""fluent/campaign_tick.py -- one-solve-per-tick Fluent campaign driver.

Session-independent completion of the Fluent post-processing campaign, driven by
Windows Task Scheduler (the D-024 GeoOpCycle pattern) because Claude-session
background tasks are killed after ~1 h in this environment. Each tick:

  1. if fluent.exe is already running -> exit (a previous tick is still solving;
     the scheduled task uses IgnoreNew so ticks never overlap);
  2. otherwise find the FIRST not-yet-complete solve in priority order and run
     exactly ONE solve (~10-21 min), then exit. The next tick runs the next.

Priority order (single licence, serial):
  A. offset study      offset_1..6, SA then SST  (variant "")
  B. r1 control gate   al_rand_naca22012_re3e6_a12 SA  (variant r1)
  C. GATE: once the control is complete, classify it. If it did NOT reach a
     (quasi-)steady solution, STOP -- do not spend licence time on the 18-deg
     set (PLAN_FLUENT_POST 2.4/2.5); write STOP_R1 marker.
  D. r1 retry          the remaining 19 diverged cases, SA  (variant r1)

When nothing is left (or the gate stops the retry) it writes a DONE marker and
disables its own scheduled task. Idempotent + resumable: safe to run any number
of times; run_batch's is_complete skips finished solves.

    .venv/Scripts/python.exe fluent/campaign_tick.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import run_batch as rb  # noqa: E402

LOG = REPO / "logs" / "fluent_campaign.log"
STATE_DIR = REPO / "logs"
DONE_MARKER = STATE_DIR / "fluent_campaign.DONE"
FINALIZED_MARKER = STATE_DIR / "fluent_campaign.FINALIZED"
STOP_R1_MARKER = STATE_DIR / "fluent_campaign.STOP_R1"
TASK_NAME = "FluentCampaign"
CONTROL = "al_rand_naca22012_re3e6_a12"
FLUENT_EXE = rb.DEFAULT_FLUENT
PROCS = 6


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  [tick] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def fluent_running() -> bool:
    try:
        out = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=30).stdout
        return "fluent.exe" in out.lower()
    except Exception:  # noqa: BLE001
        return False


def _case(cid: str, cases):
    for c in cases:
        if c["case_id"] == cid:
            return c
    return None


def _complete(cid: str, model: str, variant: str) -> bool:
    return rb.is_complete(rb.case_paths(cid, model, variant))


def _diverged_list() -> list[str]:
    txt = (HERE / "diverged_cases.txt").read_text(encoding="utf-8")
    return [ln.strip() for ln in txt.splitlines() if ln.strip() and not ln.startswith("#")]


def control_status() -> str:
    """Classify the r1 control's coefficient history (converged/quasi/diverged/not_run)."""
    from scripts.collect_fluent import _read_coeffs, classify
    coeffs = rb.CASES_DIR / CONTROL / f"{CONTROL}_sa_r1_coeffs.out"
    parsed = _read_coeffs(coeffs)
    if parsed is None:
        return "not_run"
    _its, cd, _cl = parsed
    return classify(cd)["status"]


def next_solve(cases):
    """Return (case_dict, model, variant) for the first incomplete solve, or None."""
    # A. offset study, SA then SST
    for i in range(1, 7):
        prefix = f"offset_{i}_"
        oc = next((c for c in cases if c["case_id"].startswith(prefix)), None)
        if oc is None:
            continue
        for model in ("sa", "sst"):
            if not _complete(oc["case_id"], model, ""):
                return oc, model, ""
    # B. r1 control gate
    ctrl = _case(CONTROL, cases)
    if ctrl is not None and not _complete(CONTROL, "sa", "r1"):
        return ctrl, "sa", "r1"
    # C. gate decision
    if STOP_R1_MARKER.exists():
        return None
    st = control_status()
    if st in ("not_run",):
        return None  # control still needs to finish (should have been caught above)
    if st not in ("converged", "quasi_steady"):
        STOP_R1_MARKER.write_text(
            f"r1 control {CONTROL} outcome={st} at {datetime.now():%Y-%m-%d %H:%M:%S}: "
            "gate FAILED, 18-degree r1 set not attempted (PLAN 2.4/2.5).\n",
            encoding="utf-8")
        log(f"GATE FAILED: control r1 outcome={st}; stopping the r1 retry (marker written)")
        return None
    # D. remaining 19 diverged cases, SA r1
    for cid in _diverged_list():
        if cid == CONTROL:
            continue
        c = _case(cid, cases)
        if c is not None and not _complete(cid, "sa", "r1"):
            return c, "sa", "r1"
    return None


def disable_task() -> None:
    for args in (["schtasks", "/Change", "/TN", TASK_NAME, "/DISABLE"],
                 ["schtasks", "/End", "/TN", TASK_NAME]):
        try:
            subprocess.run(args, capture_output=True, text=True, timeout=30)
        except Exception:  # noqa: BLE001
            pass


def finalize() -> None:
    """Run the numerical post-processing once every solve is done (session-independent).

    collect_fluent (+GCI) -> compare_fluent (offset + 3 models) -> tables -> figures.
    Writes results/fluent/FINAL_SUMMARY.md. Guarded by FINALIZED_MARKER so it runs once.
    RESULTS.md/paper prose stays a human step (needs voice); the numbers land here.
    """
    if FINALIZED_MARKER.exists():
        return
    py = sys.executable
    steps = [
        [py, "-m", "scripts.collect_fluent", "--gci"],
        [py, "-m", "scripts.compare_fluent"],
        [py, "-m", "scripts.make_tables"],
        [py, "-m", "scripts.make_figures"],
    ]
    for cmd in steps:
        log("finalize: " + " ".join(cmd[2:] if cmd[1] == "-m" else cmd))
        try:
            r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=3600)
            if r.returncode != 0:
                log(f"finalize step FAILED rc={r.returncode}: {r.stderr[-500:]}")
        except Exception as exc:  # noqa: BLE001
            log(f"finalize step EXCEPTION: {exc!r}")
    FINALIZED_MARKER.write_text(f"finalized at {datetime.now():%Y-%m-%d %H:%M:%S}\n", encoding="utf-8")
    log("finalize complete -- results/fluent CSVs, Table 5, Fig 11 regenerated")


def main() -> int:
    if DONE_MARKER.exists():
        finalize()
        log("campaign already DONE; disabling task")
        disable_task()
        return 0
    if fluent_running():
        log("fluent.exe already running -- a previous tick is still solving; skip")
        return 0

    cases = rb.load_cases()
    todo = next_solve(cases)
    if todo is None:
        DONE_MARKER.write_text(f"campaign complete at {datetime.now():%Y-%m-%d %H:%M:%S}\n",
                               encoding="utf-8")
        log("no solves left -- campaign COMPLETE; running finalize then disabling task")
        finalize()
        disable_task()
        return 0

    case, model, variant = todo
    mlabel = model if not variant else f"{model}_{variant}"
    log(f"running one solve: {case['case_id']} [{mlabel}]")
    res = rb.run_one(case, model, FLUENT_EXE, PROCS, "2ddp",
                     timeout_min=90, smoke_iters=None, variant=variant)
    log(f"solve result: {case['case_id']} [{mlabel}] -> {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
