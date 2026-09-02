"""Self-sustaining Kaggle sweep-chaining loop (PLAN_PHASE3 section 2).

One invocation = ONE step of the state machine, no internal sleep loop, so a
scheduler (a cron job, the /loop skill, or an orchestrator session start) can
call it repeatedly and it is always safe:

  * a session is RUNNING/QUEUED  -> report and exit 0, touch nothing;
  * a session is COMPLETE/ERROR  -> pull it (lean), merge, republish, then either
        mark the queue item done (its sweep dry-run says "0 to run") and advance,
        or relaunch the SAME item to finish the remaining runs;
  * a queue item was never launched -> launch it and exit 0.

The queue lives in kaggle/cycle_queue.json (ordered). cycle.py acts only on the
first item with ``done == false`` -- the "current" sweep. ``sweep.py --dry-run``
is the single source of truth for "done": it is exactly the skip logic the cloud
session uses, so local and cloud never disagree.

Guardrails (PLAN_PHASE3 2.1): push_code.py runs before every launch (the driver
reads code from geo-op-code, so any repo change must be pushed first);
pull_results refuses duplicate session ids; any non-zero subprocess stops the
cycle rather than continuing.

Usage:
  python kaggle/cycle.py                 # run one step
  python kaggle/cycle.py --dry-run       # print the current item + queue, no calls
  python kaggle/cycle.py --queue PATH    # use a different queue file
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

QUEUE_PATH = HERE / "cycle_queue.json"

# Normalized status buckets.
RUNNING = "running"      # in flight; do nothing
DONE = "done"           # COMPLETE/ERROR; output is ready to pull
ABSENT = "absent"       # never launched / unreachable


# --------------------------------------------------------------------------- #
# queue io (pure)
# --------------------------------------------------------------------------- #
def load_queue(path=QUEUE_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_queue(queue: dict, path=QUEUE_PATH) -> None:
    Path(path).write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")


def current_index(queue: dict):
    """Index of the first not-done item, or None when the queue is drained."""
    for i, item in enumerate(queue.get("queue", [])):
        if not item.get("done"):
            return i
    return None


# --------------------------------------------------------------------------- #
# status parsing (pure)
# --------------------------------------------------------------------------- #
def classify_status(returncode: int, text: str) -> str:
    """Map a `kaggle kernels status` result to a bucket.

    A non-zero return code means the slug is unreachable (never pushed), so it is
    ABSENT and should be launched. Otherwise the printed status word decides:
    running/queued -> RUNNING; complete/error/cancel* -> DONE (output ready).
    """
    if returncode != 0:
        return ABSENT
    t = (text or "").lower()
    if "running" in t or "queued" in t:
        return RUNNING
    if "complete" in t or "error" in t or "cancel" in t:
        return DONE
    return ABSENT


def next_action(status: str, sweep_done: bool | None) -> str:
    """State machine (pure). Returns one of: wait, advance, relaunch, launch.

    ``sweep_done`` is only consulted for a DONE session (after the pull): True
    means every run in the item's specs has metrics.json, so advance; False means
    the session was truncated, so relaunch the same item.
    """
    if status == RUNNING:
        return "wait"
    if status == ABSENT:
        return "launch"
    # DONE
    return "advance" if sweep_done else "relaunch"


# --------------------------------------------------------------------------- #
# side-effecting helpers
# --------------------------------------------------------------------------- #
def _run(cmd, check=True, capture=False):
    print(f"[cycle] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT),
                          capture_output=capture, text=True)
    if capture:
        print((proc.stdout or "") + (proc.stderr or ""), end="")
    if check and proc.returncode != 0:
        raise SystemExit(f"[cycle] STOP: command failed ({proc.returncode}): "
                         f"{' '.join(cmd)}")
    return proc


def kernel_status(owner_slug: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "status", owner_slug],
        capture_output=True, text=True)
    return classify_status(proc.returncode, (proc.stdout or "") + (proc.stderr or ""))


def sweep_is_done(specs: list[str]) -> bool:
    """True iff `sweep.py --dry-run` over the specs reports 0 runs to run."""
    cmd = [sys.executable, "-m", "scripts.sweep", "--dry-run"]
    for sp in specs:
        cmd += ["--spec", sp]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print((proc.stdout or "") + (proc.stderr or ""), end="")
    if proc.returncode != 0:
        raise SystemExit(f"[cycle] STOP: sweep dry-run failed for {specs}")
    import re
    m = re.search(r"(\d+)\s+to run", proc.stdout or "")
    if not m:
        raise SystemExit("[cycle] STOP: could not parse sweep dry-run output")
    return int(m.group(1)) == 0


def launch_item(item: dict, owner: str, runs_dataset: str) -> None:
    """push_code then launch this item's specs under its own kernel slug."""
    specs = item["specs"]
    missing = [s for s in specs if not (ROOT / s).is_file()]
    if missing:
        raise SystemExit(f"[cycle] STOP: spec file(s) missing, create them first: {missing}")
    _run([sys.executable, str(HERE / "push_code.py")])
    cmd = [sys.executable, str(HERE / "launch.py"),
           "--kernel-slug", f"{owner}/{item['slug']}",
           "--runs-dataset", f"{owner}/{runs_dataset}"]
    for sp in specs:
        cmd += ["--sweep", sp]
    _run(cmd)


def pull_item(item: dict, owner: str, runs_dataset: str) -> None:
    _run([sys.executable, str(HERE / "pull_results.py"),
          "--kernel", f"{owner}/{item['slug']}",
          "--runs-dataset", f"{owner}/{runs_dataset}"])


def regenerate_and_commit(item: dict) -> None:
    """Regenerate paper outputs from the freshly-merged results and commit+push.

    Keeps the repo (and the pushed private mirror) continuously up to date with no
    human in the loop: figures + tables always; run_uq only for ensemble items
    (it needs the just-fetched member checkpoints). Failures here are non-fatal to
    the chaining loop -- a bad regen must not strand the sweep queue -- so each
    step is guarded and merely logged.
    """
    def soft(cmd):
        try:
            _run(cmd, check=True)
        except SystemExit as exc:
            print(f"[cycle] regen step non-fatal failure: {exc}")

    if "ens" in item.get("slug", ""):  # ensemble sessions changed the members
        soft([sys.executable, "-m", "scripts.run_uq", "--all", "--seeds", "0,1,2,3,4"])
    soft([sys.executable, "-m", "scripts.make_figures", "--outdir", "paper/figures"])
    soft([sys.executable, "-m", "scripts.make_tables", "--outdir", "paper/tables"])
    # Commit + push the merged results and regenerated artifacts (own repo, authorized).
    try:
        subprocess.run(["git", "add", "results", "paper/figures", "paper/tables",
                        "docs/PROGRESS.md"], cwd=str(ROOT), check=False)
        msg = f"[results] cycle: merge {item.get('slug','?')} + regen figures/tables"
        subprocess.run(["git", "-c", "user.name=Taimoor Amin",
                        "-c", "user.email=taimooramin419@gmail.com",
                        "commit", "-q", "-m", msg], cwd=str(ROOT), check=False)
        # push under cocomo069, then restore prior active account if any
        subprocess.run(["gh", "auth", "switch", "-u", "cocomo069"], cwd=str(ROOT), check=False,
                       capture_output=True)
        subprocess.run(["git", "push", "-q", "origin", "main"], cwd=str(ROOT), check=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[cycle] commit/push non-fatal failure: {exc}")


# --------------------------------------------------------------------------- #
def step(queue_path=QUEUE_PATH, dry_run=False) -> int:
    queue = load_queue(queue_path)
    owner = queue.get("owner")
    runs_dataset = queue.get("runs_dataset", "geo-op-runs")
    if not owner:
        from _common import kaggle_username
        owner = kaggle_username()

    idx = current_index(queue)
    if idx is None:
        print("[cycle] queue drained: every item is done. Nothing to launch.")
        return 0
    item = queue["queue"][idx]
    slug = f"{owner}/{item['slug']}"
    print(f"[cycle] current item [{idx}] {slug}  specs={item['specs']}")

    if dry_run:
        print("[cycle] --dry-run: no kaggle/git calls. Queue state:")
        for i, it in enumerate(queue["queue"]):
            mark = "done" if it.get("done") else ("HERE" if i == idx else "todo")
            print(f"  [{mark}] {it['slug']}  {it['specs']}")
        return 0

    status = kernel_status(slug)
    print(f"[cycle] {slug} status = {status}")

    if status == RUNNING:
        print(f"[cycle] {slug} is in flight; exiting without changes (idempotent).")
        return 0

    if status == ABSENT:
        print(f"[cycle] {slug} not launched yet; launching.")
        launch_item(item, owner, runs_dataset)
        return 0

    # status == DONE: pull first (refuses duplicate session ids internally), then
    # decide advance vs relaunch from the local metrics via the sweep dry-run.
    pull_item(item, owner, runs_dataset)
    regenerate_and_commit(item)
    done = sweep_is_done(item["specs"])
    action = next_action(status, done)
    if action == "advance":
        item["done"] = True
        save_queue(queue, queue_path)
        print(f"[cycle] {slug} complete and all runs present -> marked done, "
              "advanced. Run cycle again to launch the next item.")
    else:  # relaunch
        print(f"[cycle] {slug} was truncated ({item['specs']} still has runs to "
              "run) -> relaunching the same item.")
        launch_item(item, owner, runs_dataset)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--queue", default=str(QUEUE_PATH))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the current item and queue without any kaggle/git call")
    args = ap.parse_args(argv)
    return step(queue_path=Path(args.queue), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
