"""Kaggle kernel driver: clone repo, wire cache, run a sweep partition, export results.

Pushed by kaggle/launch.py which rewrites the CONFIG block below. Runs inside a Kaggle
GPU kernel. Chaining: pass a runs-dataset to resume checkpoints from a prior session.
"""

# ==== CONFIG (rewritten by launch.py) ====
REPO_URL = "{{REPO_URL}}"            # e.g. https://github.com/<user>/geo-operator-audit.git
COMMIT = "{{COMMIT}}"                # pinned commit sha
SWEEP = "{{SWEEP}}"                  # e.g. configs/sweeps/core.yaml
CACHE_DATASET_DIR = "/kaggle/input/airfrans-cache"   # attached via kernel-metadata
CODE_DATASET_DIR = "/kaggle/input/geo-op-code"      # repo.tar.gz snapshot
RUNS_SLUG = "{{RUNS_SLUG}}"           # "" or the runs dataset slug name (e.g. geo-op-runs)
MAX_SECONDS = 11 * 3600
# =========================================

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path

T0 = time.time()
WORK = Path("/kaggle/working")
# Repo lives OFF the output path (/kaggle/working) so kernel-output pulls stay
# small -- only the zips + session.log below are exported.
REPO = Path("/kaggle/tmp/repo")

# One id per kernel run, printed and written into STATUS.txt and the export
# manifest so pull_results.py can dedupe already-merged sessions (D-023 cycle).
SESSION_ID = uuid.uuid4().hex[:12]
STARTED_UTC = datetime.datetime.now(datetime.timezone.utc).isoformat()


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_export_set(results_dir, ckpt_dir, pre_done) -> dict:
    """Pick which per-run checkpoints to export this session.

    Returns ``{run_id: {"done": bool, "files": [...]}}`` for every
    ``checkpoints/<run_id>/`` that was NOT already finished when the session
    started (i.e. ``run_id`` not in ``pre_done``). This is what stops the
    cumulative-growth failure: runs restored-and-already-finished are skipped.

    ``done`` is True when ``results/<run_id>/metrics.json`` exists. A finished
    run exports ``best.pt`` only (fallback ``last.pt`` if best is missing) --
    a run with metrics.json never resumes, so its optimizer/RNG state is dead
    weight. An unfinished run exports every ``*.pt`` present (``last.pt`` is what
    resumes it next session; ``best.pt`` first if it exists too).
    """
    results_dir = Path(results_dir)
    ckpt_dir = Path(ckpt_dir)
    pre_done = set(pre_done)
    out: dict = {}
    if not ckpt_dir.exists():
        return out
    for run_dir in sorted(ckpt_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        rid = run_dir.name
        if rid in pre_done:
            continue
        pts = {p.name for p in run_dir.glob("*.pt")}
        if not pts:
            continue
        done = (results_dir / rid / "metrics.json").is_file()
        if done:
            if "best.pt" in pts:
                files = ["best.pt"]
            elif "last.pt" in pts:
                files = ["last.pt"]
            else:
                files = sorted(pts)
        else:
            files = [f for f in ("last.pt", "best.pt") if f in pts]
            files += sorted(pts - set(files))
        out[rid] = {"done": done, "files": files}
    return out


def sh(cmd, **kw):
    print(f"+ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def main():
    # 1. Obtain the code from the geo-op-code dataset, which launch.py attaches
    #    and push_code.py refreshes each launch. Needs no GitHub token. Kaggle
    #    auto-extracts the uploaded repo.tar.gz into a read-only ``repo/`` dir,
    #    so copy it into the writable working tree; handle a raw tarball too.
    REPO.mkdir(parents=True, exist_ok=True)
    extracted = Path(CODE_DATASET_DIR) / "repo"
    snapshot = Path(CODE_DATASET_DIR) / "repo.tar.gz"
    if extracted.is_dir():
        shutil.copytree(extracted, REPO, dirs_exist_ok=True)
    elif snapshot.exists():
        sh(f"tar -xzf {snapshot} -C {REPO}")
    else:
        token = ""
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("GH_TOKEN")
        except Exception:
            pass
        if not token:
            raise SystemExit(f"no repo.tar.gz at {CODE_DATASET_DIR} and no GH_TOKEN secret")
        url = REPO_URL.replace("https://", f"https://{token}@")
        sh(f"git clone --quiet {url} {REPO}")
        sh(f"git -C {REPO} checkout --quiet {COMMIT}")

    os.chdir(REPO)
    # Kaggle preinstalls a CUDA-enabled torch; installing our pinned torch would
    # replace it with an incompatible wheel. Install everything else.
    reqs = [ln for ln in Path("requirements.txt").read_text().splitlines()
            if ln.strip() and not ln.strip().startswith(("#", "torch"))]
    Path("requirements.kaggle.txt").write_text("\n".join(reqs))
    sh(f"{sys.executable} -m pip install -q -r requirements.kaggle.txt")

    # 2. Locate the cache by finding the per-sim manifest.json ANYWHERE under the
    #    input mounts -- Kaggle's zip extraction can nest it under an extra dir,
    #    so we search rather than assume a layout. Passed to runs via --extra.
    candidates = sorted(
        m.parent for m in Path("/kaggle/input").glob("**/manifest.json")
        if list(m.parent.glob("*.npz")))
    # Always dump the mount tree (dirs) so layout is diagnosable from the export.
    tree = "\n".join(sorted(str(p) for p in Path("/kaggle/input").glob("**/*")
                            if p.is_dir()))
    (WORK / "MOUNT_TREE.txt").write_text(tree)
    if len(candidates) > 1:
        print(f"[diag] WARNING multiple cache candidates, using first: {candidates}")
    cache_processed = candidates[0] if candidates else Path(CACHE_DATASET_DIR)
    print(f"[diag] located cache at {cache_processed} "
          f"({len(list(cache_processed.glob('*.npz')))} npz)")

    # 3. Resume state from a previous session's runs-dataset, if one was passed.
    #    Datasets mount at /kaggle/input/datasets/<owner>/<slug>/ (D-020), so we
    #    LOCATE the runs dataset by its slug name rather than a hardcoded path,
    #    the same content-search fix used for the cache. Copying prior results/ +
    #    checkpoints/ into the repo is what lets sweep.py skip finished runs and
    #    resume partial ones -- a silent miss here retrains everything (QA #1).
    if RUNS_SLUG:
        runs_roots = sorted(
            p for p in Path("/kaggle/input").glob(f"**/{RUNS_SLUG}")
            if p.is_dir() and ((p / "results").exists() or (p / "checkpoints").exists()))
        if not runs_roots:
            print(f"[diag] RUNS resume: no dataset '{RUNS_SLUG}' with results/ or "
                  f"checkpoints/ found under /kaggle/input -- starting fresh")
        else:
            root = runs_roots[0]
            for sub in ("results", "checkpoints"):
                src = root / sub
                if src.exists():
                    shutil.copytree(src, REPO / sub, dirs_exist_ok=True)
                    n = len(list(src.rglob("metrics.json"))) if sub == "results" else \
                        len(list(src.glob("*/last.pt")))
                    print(f"[diag] RUNS resume: restored {sub} from {root} ({n} items)")

    # Record the finished set BEFORE the sweep runs. Any run already carrying a
    # metrics.json at this point was restored from the runs-dataset, not trained
    # this session, so its checkpoints must NOT be re-exported (lean export).
    pre_done = {p.parent.name for p in (REPO / "results").glob("*/metrics.json")}
    print(f"[driver] session_id={SESSION_ID} pre_done={len(pre_done)} finished runs restored")

    # 3b. Environment diagnostics (Kaggle logs are unreliable on failure, so we
    #     print these and also tee the sweep output into an always-exported file).
    proc = cache_processed
    n_npz = len(list(proc.glob("*.npz"))) if proc.exists() else -1
    print(f"[diag] python={sys.version.split()[0]} cwd={os.getcwd()}")
    print(f"[diag] cache_processed={proc} exists={proc.exists()} n_npz={n_npz}")
    print(f"[diag] norm_stats={(proc / 'norm_stats.json').exists()} "
          f"manifest={(proc / 'manifest.json').exists()}")
    try:
        import torch
        print(f"[diag] torch={torch.__version__} cuda={torch.cuda.is_available()}")
    except Exception as exc:
        print(f"[diag] torch import failed: {exc}")

    # 4. Run sweep with wall-clock guard, teeing output into WORK/session.log so
    #    the real error survives even if Kaggle returns an empty kernel log.
    session_log = WORK / "session.log"
    budget = int(MAX_SECONDS - (time.time() - T0))
    sweep_rc = 0
    # SWEEP may be a comma-separated list of specs; sweep.py drains them in
    # order within the one session (--spec is repeatable). One session can then
    # cover e.g. data-eff then ablations without a relaunch (PLAN_PHASE3 1.3).
    spec_args = []
    for spec in [s.strip() for s in SWEEP.split(",") if s.strip()]:
        spec_args += ["--spec", spec]
    try:
        with open(session_log, "w", encoding="utf-8") as lf:
            p = subprocess.Popen(
                [sys.executable, "-u", "-m", "scripts.sweep", *spec_args,
                 "--max-seconds", str(budget),
                 "--extra", f"data.processed_dir={cache_processed}",
                 f"data.norm_stats={cache_processed / 'norm_stats.json'}"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                print(line, end="")
                lf.write(line)
            sweep_rc = p.wait()
    except Exception as exc:  # noqa: BLE001
        sweep_rc = 1
        session_log.write_text(f"{session_log.read_text() if session_log.exists() else ''}\n"
                               f"DRIVER EXCEPTION: {exc}\n")
        print(f"[driver] sweep raised: {exc}")

    # 5. Export LEAN, pass or fail (PLAN_PHASE3 1.2). results.zip is small and
    #    always exported. Checkpoints are exported per-run, only for runs trained
    #    or advanced THIS session, as ckpt_<run_id>.zip (ZIP_STORED: torch files
    #    do not compress, and stored members make HTTP-Range member reads trivial
    #    for pull_results.py). EXPORT_MANIFEST.json indexes them with size+sha256.
    results_zip = WORK / "results.zip"
    with zipfile.ZipFile(results_zip, "w", zipfile.ZIP_DEFLATED) as z:
        rdir = REPO / "results"
        if rdir.exists():
            for f in rdir.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(REPO))
    print(f"exported {results_zip} ({results_zip.stat().st_size/1e6:.1f} MB)")

    export_set = select_export_set(REPO / "results", REPO / "checkpoints", pre_done)
    manifest_runs: dict = {}
    total_bytes = 0
    for rid, spec in export_set.items():
        zpath = WORK / f"ckpt_{rid}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for fname in spec["files"]:
                src = REPO / "checkpoints" / rid / fname
                if src.is_file():
                    z.write(src, f"checkpoints/{rid}/{fname}")
        size = zpath.stat().st_size
        total_bytes += size
        manifest_runs[rid] = {
            "done": spec["done"], "zip": zpath.name, "files": spec["files"],
            "bytes": size, "sha256": _sha256(zpath),
        }
        print(f"[driver] exported {zpath.name} ({size/1e6:.1f} MB, {spec['files']})")

    manifest = {
        "session_id": SESSION_ID,
        "sweep": SWEEP,
        "commit": COMMIT,
        "started_utc": STARTED_UTC,
        "finished_utc": _utcnow(),
        "sweep_rc": sweep_rc,
        "pre_done_count": len(pre_done),
        "runs": manifest_runs,
    }
    (WORK / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"[driver] exported {len(manifest_runs)} run zips, {total_bytes/1e6:.1f} MB total")

    # Record the outcome in the exported log, but let the kernel finish
    # "complete" (not raise) so Kaggle preserves its output and log for pulling.
    with open(session_log, "a", encoding="utf-8") as lf:
        lf.write(f"\n[driver] sweep rc={sweep_rc}\n")
    (WORK / "STATUS.txt").write_text(f"session_id={SESSION_ID}\nsweep_rc={sweep_rc}\n")
    print(f"[driver] sweep rc={sweep_rc}; session.log tail:")
    if session_log.exists():
        print("\n".join(session_log.read_text().splitlines()[-40:]))


if __name__ == "__main__":
    main()
