"""Download a finished Kaggle session's output, merge into local repo, republish runs dataset.

Usage: python kaggle/pull_results.py [--kernel user/geo-op-session] [--runs-dataset user/geo-op-runs]
"""
import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).parent


def unzip_into(zpath: Path, root: Path):
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)  # archives store paths relative to repo root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--runs-dataset", default=None)
    args = ap.parse_args()
    root = HERE.parent
    from _common import kaggle_username
    user = kaggle_username()
    kernel = args.kernel or f"{user}/geo-op-session"
    runs_slug = args.runs_dataset or f"{user}/geo-op-runs"

    with tempfile.TemporaryDirectory() as td:
        subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", kernel, "-p", td], check=True)
        for name in ("results.zip", "checkpoints.zip"):
            zp = Path(td) / name
            if zp.exists():
                unzip_into(zp, root)
                print(f"merged {name}")

    # republish merged state so the next session can resume
    with tempfile.TemporaryDirectory(dir=root / "data") as td:
        tdp = Path(td)
        import shutil
        for sub in ("results", "checkpoints"):
            if (root / sub).exists():
                shutil.copytree(root / sub, tdp / sub)
        (tdp / "dataset-metadata.json").write_text(json.dumps({
            "title": "geo-op-runs", "id": runs_slug,
            "licenses": [{"name": "other"}],
        }, indent=2))
        exists = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "status", runs_slug],
                                capture_output=True).returncode == 0
        cmd = ([sys.executable, "-m", "kaggle", "datasets", "version", "-p", str(tdp), "-m", "session merge", "--dir-mode", "zip"]
               if exists else
               [sys.executable, "-m", "kaggle", "datasets", "create", "-p", str(tdp), "--dir-mode", "zip"])
        subprocess.run(cmd, check=True)
    print(f"republished {runs_slug}")


if __name__ == "__main__":
    main()
