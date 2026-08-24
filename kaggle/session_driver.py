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
RUNS_DATASET_DIR = "{{RUNS_DIR}}"    # "" or /kaggle/input/<runs-dataset>
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
REPO = WORK / "repo"


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

    # 2. Wire data cache (read-only input -> expected paths)
    (REPO / "data").mkdir(exist_ok=True)
    for sub in ("processed", "splits"):
        src = Path(CACHE_DATASET_DIR) / sub
        dst = REPO / "data" / sub
        if not dst.exists():
            os.symlink(src, dst, target_is_directory=True)

    # 3. Resume state from previous session, if any
    if RUNS_DATASET_DIR and Path(RUNS_DATASET_DIR).exists():
        for sub in ("results", "checkpoints"):
            src = Path(RUNS_DATASET_DIR) / sub
            if src.exists():
                shutil.copytree(src, REPO / sub, dirs_exist_ok=True)

    # 4. Run sweep with wall-clock guard
    budget = int(MAX_SECONDS - (time.time() - T0))
    sh(f"{sys.executable} -m scripts.sweep --spec {SWEEP} --max-seconds {budget}")

    # 5. Export state as kernel output
    for name, folder in (("results", REPO / "results"), ("checkpoints", REPO / "checkpoints")):
        zpath = WORK / f"{name}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            if folder.exists():
                for f in folder.rglob("*"):
                    if f.is_file():
                        z.write(f, f.relative_to(REPO))
        print(f"exported {zpath} ({zpath.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
