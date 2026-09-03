"""Pull a finished Kaggle session's output, merge into the repo, republish LEAN.

PLAN_PHASE3 1.5. The old puller ran ``kaggle kernels output`` once (which grabs
EVERY file in one un-chunked GET) and then re-uploaded the entire local
``checkpoints/`` (1.6 GB) to ``geo-op-runs`` every cycle -- both grew every
session and the 2 GB GET stalled and never finished. This version:

1. fetches the SMALL always-present files first (results.zip, session.log,
   STATUS.txt, EXPORT_MANIFEST.json) and merges results immediately, so metrics
   land even if a checkpoint later stalls;
2. fetches per-run checkpoint zips lazily, ONE at a time, with retry, and skips
   any whose files are already local (idempotent re-runs);
3. republishes ``geo-op-runs`` LEAN -- results/** plus only the ``last.pt`` of
   still-unfinished runs (what the next session needs to resume). The finished
   ``best.pt`` trove stays local, so the runs dataset is ~10-50 MB, not GBs.

Legacy sessions (old driver, single cumulative checkpoints.zip, no manifest) are
handled per PLAN_PHASE3 1.7: merge results.zip only, never touch checkpoints.zip.
For a legacy fat zip you actually need members from, use
``kaggle/fetch_zip_members.py`` (HTTP-Range member extraction).

Usage:
  python kaggle/pull_results.py --kernel user/geo-op-dataeff \
      [--runs-dataset user/geo-op-runs] [--checkpoints needed|all|none] \
      [--no-republish] [--dest DIR] [--force]
"""
from __future__ import annotations

import os
import argparse
import datetime
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).parent

# Core / ensemble runs whose best.pt feeds run_uq.py and run_active.py downstream.
# Data-eff (``*_n<size>_*``) and tagged ablation runs are metrics-only downstream,
# so ``needed`` skips their (finished) checkpoints.
NEEDED_RE = re.compile(
    r"^(gnn|sdf_fno|transolver)_(full|scarce|reynolds|aoa|shape5|combined)_s\d+$")

SMALL_PATTERN = r"^(results\.zip|session\.log|STATUS\.txt|EXPORT_MANIFEST\.json|MOUNT_TREE\.txt)$"


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _kaggle_output(kernel: str, dest: Path, pattern: str, timeout=None) -> subprocess.CompletedProcess:
    """One ``kaggle kernels output`` call filtered to a filename regex.

    Force UTF-8 for the child: on a Windows cp1252 console the kaggle CLI's own
    print of unicode (dataset/kernel names) raises 'charmap codec can't encode'
    and exits non-zero even though the files downloaded fine -- which would abort
    the whole autonomous cycle. PYTHONUTF8/IOENCODING make that print harmless.
    """
    dest.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-m", "kaggle", "kernels", "output", kernel,
           "-p", str(dest), "--file-pattern", pattern]
    print(f"[pull] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, timeout=timeout, env=env)


# --------------------------------------------------------------------------- #
# 1. small files first
# --------------------------------------------------------------------------- #
def fetch_small(kernel: str, dest: Path) -> dict | None:
    """Download the small always-present files; return the parsed manifest or None."""
    _kaggle_output(kernel, dest, SMALL_PATTERN)
    mpath = dest / "EXPORT_MANIFEST.json"
    if mpath.exists():
        return json.loads(mpath.read_text(encoding="utf-8"))
    print("[pull] no EXPORT_MANIFEST.json -- legacy session (old driver); "
          "results only, checkpoints skipped (PLAN_PHASE3 1.7)")
    return None


# --------------------------------------------------------------------------- #
# 2. merge results
# --------------------------------------------------------------------------- #
def merge_results(dest: Path, root: Path, force=False) -> list[str]:
    """Extract results/** from the pulled results.zip into the repo.

    Refuses to overwrite a local ``results/<rid>/metrics.json`` whose bytes
    differ from the incoming one unless ``force`` (D-019 clobber guard).
    Returns the list of run ids whose metrics were written or refreshed.
    """
    zp = dest / "results.zip"
    if not zp.exists():
        print("[pull] no results.zip in output")
        return []
    merged: list[str] = []
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        for name in names:
            if name.endswith("/"):
                continue
            target = root / name
            if target.name == "metrics.json" and target.exists() and not force:
                if target.read_bytes() != z.read(name):
                    print(f"[pull] SKIP clobber: {name} differs locally "
                          f"(use --force to overwrite)")
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            if target.name == "metrics.json":
                merged.append(target.parent.name)
    print(f"[pull] merged results: {len(merged)} metrics.json")
    return merged


# --------------------------------------------------------------------------- #
# 3. which checkpoints to fetch
# --------------------------------------------------------------------------- #
def needed_checkpoints(manifest: dict, root: Path, policy="needed") -> list[str]:
    """Run ids whose checkpoint zip should be fetched, per policy.

    ``needed`` = the run is a core/ensemble run (best.pt feeds UQ/AL) OR it is
    unfinished (its last.pt must be republished to resume), AND its member files
    are not already present locally. ``all`` = every run in the manifest. ``none``
    = nothing.
    """
    runs = manifest.get("runs", {})
    if policy == "none":
        return []
    out: list[str] = []
    for rid, info in runs.items():
        if policy == "all":
            want = True
        else:  # needed
            want = bool(NEEDED_RE.match(rid)) or not info.get("done", True)
        if not want:
            continue
        files = info.get("files", [])
        if files and all((root / "checkpoints" / rid / f).is_file() for f in files):
            continue  # skip-if-already-local
        out.append(rid)
    return out


# --------------------------------------------------------------------------- #
# 4. fetch one checkpoint zip, verify, extract
# --------------------------------------------------------------------------- #
def fetch_zip(kernel: str, name: str, dest: Path, info: dict | None,
              attempts=4, timeout=900) -> Path | None:
    """Fetch a single per-run zip with retry; verify bytes+sha256 vs the manifest.

    One stalled 16-33 MB file costs one retry, never the whole session.
    """
    zp = dest / name
    for attempt in range(1, attempts + 1):
        try:
            _kaggle_output(kernel, dest, f"^{re.escape(name)}$", timeout=timeout)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            print(f"[pull] {name} attempt {attempt}/{attempts} failed: {exc}")
            continue
        if not zp.exists():
            print(f"[pull] {name} not produced on attempt {attempt}/{attempts}")
            continue
        if info:
            if info.get("bytes") and zp.stat().st_size != info["bytes"]:
                print(f"[pull] {name} size {zp.stat().st_size} != manifest "
                      f"{info['bytes']}; retrying")
                continue
            if info.get("sha256") and _sha256(zp) != info["sha256"]:
                print(f"[pull] {name} sha256 mismatch; retrying")
                continue
        return zp
    print(f"[pull] GAVE UP on {name} after {attempts} attempts")
    return None


def extract_ckpt_zip(zp: Path, root: Path) -> None:
    with zipfile.ZipFile(zp) as z:
        z.extractall(root)  # members stored as checkpoints/<rid>/<file>


# --------------------------------------------------------------------------- #
# 5. republish LEAN runs dataset
# --------------------------------------------------------------------------- #
def _lean_stage_list(root: Path) -> tuple[list[tuple[str, int]], list[str]]:
    """Files to stage into geo-op-runs: results/** + last.pt of unfinished runs."""
    staged: list[tuple[str, int]] = []
    rdir = root / "results"
    if rdir.exists():
        for f in sorted(rdir.rglob("*")):
            if f.is_file():
                staged.append((str(f.relative_to(root)).replace("\\", "/"), f.stat().st_size))
    unfinished: list[str] = []
    cdir = root / "checkpoints"
    if cdir.exists():
        for d in sorted(cdir.iterdir()):
            if not d.is_dir():
                continue
            if (root / "results" / d.name / "metrics.json").is_file():
                continue
            last = d / "last.pt"
            if last.is_file():
                unfinished.append(d.name)
                staged.append((f"checkpoints/{d.name}/last.pt", last.stat().st_size))
    return staged, unfinished


def republish(root: Path, runs_slug: str, session_id="", force=False) -> bool:
    """Republish geo-op-runs LEAN. Skips if the staged file set is unchanged."""
    staged, unfinished = _lean_stage_list(root)
    state_hash = hashlib.sha256(repr(sorted(staged)).encode()).hexdigest()
    state_path = HERE / "runs_dataset_state.json"
    if not force and state_path.exists():
        try:
            prev = json.loads(state_path.read_text(encoding="utf-8")).get("hash")
        except Exception:
            prev = None
        if prev == state_hash:
            print("[pull] runs dataset unchanged since last republish; skipping")
            return False

    total_mb = sum(s for _, s in staged) / 1e6
    print(f"[pull] republishing {runs_slug}: {len(staged)} files, {total_mb:.1f} MB, "
          f"{len(unfinished)} unfinished last.pt")
    (root / "data").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root / "data") as td:
        tdp = Path(td)
        if (root / "results").exists():
            shutil.copytree(root / "results", tdp / "results")
        for rid in unfinished:
            dst = tdp / "checkpoints" / rid
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / "checkpoints" / rid / "last.pt", dst / "last.pt")
        (tdp / "dataset-metadata.json").write_text(json.dumps({
            "title": "geo-op-runs", "id": runs_slug,
            "licenses": [{"name": "other"}],
        }, indent=2))
        exists = subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "status", runs_slug],
            capture_output=True).returncode == 0
        msg = f"{session_id} lean merge" if session_id else "lean merge"
        cmd = ([sys.executable, "-m", "kaggle", "datasets", "version", "-p", str(tdp),
                "-m", msg, "--dir-mode", "zip"]
               if exists else
               [sys.executable, "-m", "kaggle", "datasets", "create", "-p", str(tdp),
                "--dir-mode", "zip"])
        subprocess.run(cmd, check=True)
    state_path.write_text(json.dumps(
        {"hash": state_hash, "utc": _utcnow(), "slug": runs_slug,
         "files": len(staged), "unfinished": unfinished}, indent=2), encoding="utf-8")
    print(f"[pull] republished {runs_slug}")
    return True


# --------------------------------------------------------------------------- #
# session dedupe
# --------------------------------------------------------------------------- #
def _already_pulled(session_id: str) -> bool:
    log = HERE / "pulled_sessions.jsonl"
    if not session_id or not log.exists():
        return False
    for line in log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            if json.loads(line).get("session_id") == session_id:
                return True
        except Exception:
            continue
    return False


def _record_pull(session_id: str, kernel: str, merged: list[str], fetched: list[str]) -> None:
    rec = {"session_id": session_id, "kernel": kernel, "utc": _utcnow(),
           "merged_runs": len(merged), "fetched_zips": fetched}
    with (HERE / "pulled_sessions.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--runs-dataset", default=None)
    ap.add_argument("--checkpoints", choices=["needed", "all", "none"], default="needed")
    ap.add_argument("--no-republish", action="store_true")
    ap.add_argument("--dest", default=None,
                    help="download dir; defaults kaggle/pulls/<session_id or kernel>")
    ap.add_argument("--force", action="store_true",
                    help="re-merge a session already in pulled_sessions.jsonl and "
                         "overwrite differing local metrics.json / force republish")
    args = ap.parse_args(argv)

    root = HERE.parent
    from _common import kaggle_username
    user = kaggle_username()
    kernel = args.kernel or f"{user}/geo-op-session"
    runs_slug = args.runs_dataset or f"{user}/geo-op-runs"

    # small files -> a scratch dir keyed by kernel until the session id is known
    scratch = Path(args.dest) if args.dest else HERE / "pulls" / kernel.split("/")[-1]
    manifest = fetch_small(kernel, scratch)
    session_id = manifest.get("session_id", "") if manifest else ""

    if session_id and not args.dest:
        dest = HERE / "pulls" / session_id
        dest.mkdir(parents=True, exist_ok=True)
        for f in scratch.glob("*"):
            shutil.copy2(f, dest / f.name)
    else:
        dest = scratch

    if session_id and _already_pulled(session_id) and not args.force:
        print(f"[pull] session {session_id} already merged; nothing to do "
              "(use --force to re-merge)")
        return 0

    merged = merge_results(dest, root, force=args.force)

    fetched: list[str] = []
    if manifest and args.checkpoints != "none":
        wanted = needed_checkpoints(manifest, root, args.checkpoints)
        print(f"[pull] {len(wanted)} checkpoint zip(s) to fetch "
              f"(policy={args.checkpoints})")
        for rid in wanted:
            info = manifest["runs"].get(rid, {})
            name = info.get("zip", f"ckpt_{rid}.zip")
            zp = fetch_zip(kernel, name, dest, info)
            if zp is not None:
                extract_ckpt_zip(zp, root)
                fetched.append(rid)
                print(f"[pull] extracted {name}")

    if not args.no_republish:
        republish(root, runs_slug, session_id=session_id, force=args.force)

    _record_pull(session_id, kernel, merged, fetched)
    print(f"[pull] done: merged {len(merged)} results, fetched {len(fetched)} checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
