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

import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

T0 = time.time()
WORK = Path("/kaggle/working")
# Repo lives OFF the output path (/kaggle/working) so kernel-output pulls stay
# small -- only the zips + session.log below are exported.
REPO = Path("/kaggle/tmp/repo")


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
    try:
        with open(session_log, "w", encoding="utf-8") as lf:
            p = subprocess.Popen(
                [sys.executable, "-u", "-m", "scripts.sweep", "--spec", SWEEP,
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

    # 5. Always export state (results/checkpoints + the session log), pass or fail.
    for name, folder in (("results", REPO / "results"), ("checkpoints", REPO / "checkpoints")):
        zpath = WORK / f"{name}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            if folder.exists():
                for f in folder.rglob("*"):
                    if f.is_file():
                        z.write(f, f.relative_to(REPO))
        print(f"exported {zpath} ({zpath.stat().st_size/1e6:.1f} MB)")
    # Record the outcome in the exported log, but let the kernel finish
    # "complete" (not raise) so Kaggle preserves its output and log for pulling.
    with open(session_log, "a", encoding="utf-8") as lf:
        lf.write(f"\n[driver] sweep rc={sweep_rc}\n")
    (WORK / "STATUS.txt").write_text(f"sweep_rc={sweep_rc}\n")
    print(f"[driver] sweep rc={sweep_rc}; session.log tail:")
    if session_log.exists():
        print("\n".join(session_log.read_text().splitlines()[-40:]))


if __name__ == "__main__":
    main()
