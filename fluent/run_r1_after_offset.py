"""fluent/run_r1_after_offset.py -- chained r1 retry with the control gate.

Runs UNATTENDED after the offset batch frees the single Fluent licence:

  1. wait until the offset batch has finished (its log shows "BATCH DONE", and no
     fluent.exe is still running);
  2. run the r1 CONTROL case first (al_rand_naca22012_re3e6_a12, alpha=12, the
     inside-envelope case) via run_batch --variant r1;
  3. classify the control's r1 coefficient history with the SAME rules as
     scripts.collect_fluent;
  4. GATE: if the control is converged or quasi_steady, run the remaining 19
     diverged cases with r1 (SA); if it still diverged, STOP and do NOT spend
     licence time on the 18-degree set (PLAN_FLUENT_POST section 2, step 4/5).

Everything is logged to logs/fluent_retry.log. Launch this DETACHED right after
the offset batch; it self-serialises on the licence.

    .venv/Scripts/python.exe fluent/run_r1_after_offset.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PY = sys.executable
RUN_BATCH = HERE / "run_batch.py"
OFFSET_LOG = REPO / "logs" / "fluent_offset.log"
RETRY_LOG = REPO / "logs" / "fluent_retry.log"
DIVERGED = HERE / "diverged_cases.txt"
CONTROL = "al_rand_naca22012_re3e6_a12"


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    RETRY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RETRY_LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _fluent_running() -> bool:
    try:
        out = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=30).stdout
        return "fluent.exe" in out.lower()
    except Exception:  # noqa: BLE001
        return False


def wait_for_offset(poll_s: int = 120, max_h: float = 6.0) -> bool:
    """Block until the offset batch has finished. Returns True when clear."""
    deadline = time.time() + max_h * 3600
    log("waiting for the offset batch to finish (BATCH DONE + no fluent.exe)...")
    while time.time() < deadline:
        done = OFFSET_LOG.exists() and "BATCH DONE" in OFFSET_LOG.read_text(
            encoding="utf-8", errors="replace")
        if done and not _fluent_running():
            log("offset batch finished and no fluent.exe running -- proceeding")
            return True
        time.sleep(poll_s)
    log("TIMEOUT waiting for the offset batch -- aborting r1 to avoid a licence collision")
    return False


def classify_control() -> str:
    sys.path.insert(0, str(REPO))
    from scripts.collect_fluent import _read_coeffs, classify
    coeffs = REPO / "fluent" / "cases" / CONTROL / f"{CONTROL}_sa_r1_coeffs.out"
    parsed = _read_coeffs(coeffs)
    if parsed is None:
        return "not_run"
    _its, cd, _cl = parsed
    return classify(cd)["status"]


def run_batch(args: list[str]) -> int:
    cmd = [PY, str(RUN_BATCH)] + args + ["--log-file", str(RETRY_LOG)]
    log("LAUNCH " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def main() -> int:
    log("=" * 70)
    log("r1 retry driver start")
    if not wait_for_offset():
        return 2

    # 1) control case first
    log(f"running the r1 CONTROL: {CONTROL} (alpha=12, inside-envelope gate)")
    run_batch(["--variant", "r1", "--models", "sa", "--only", CONTROL, "--timeout-min", "90"])
    status = classify_control()
    log(f"CONTROL r1 outcome: {status}")

    if status not in ("converged", "quasi_steady"):
        log("GATE FAILED: the alpha=12 control did not reach a steady/quasi-steady "
            "solution under r1. STOPPING -- not running the 18-degree set (would "
            "burn licence time for no steady answer). r2 (pseudo-transient) is the "
            "documented next step on this single case only, per PLAN section 2.5.")
        log("r1 retry driver done (gate failed)")
        return 0

    # 2) gate passed -> run the remaining 19 (control skips as complete)
    log("GATE PASSED: running the remaining 19 diverged cases with r1 (SA).")
    rc = run_batch(["--variant", "r1", "--models", "sa",
                    "--only-file", str(DIVERGED), "--timeout-min", "120"])
    log(f"r1 batch finished rc={rc}")
    log("r1 retry driver done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
