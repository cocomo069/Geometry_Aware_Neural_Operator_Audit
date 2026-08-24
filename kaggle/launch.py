"""Push a sweep session to Kaggle. Requires kaggle CLI + token; see kaggle/README.md.

Usage:
  python kaggle/launch.py --sweep configs/sweeps/core.yaml \
      [--repo-url URL] [--runs-dataset user/geo-op-runs] [--kernel-slug user/geo-op-session]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def git_head(repo_root: Path) -> str:
    return subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


from _common import kaggle_username  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--repo-url", default=None, help="defaults to `git remote get-url origin`")
    ap.add_argument("--runs-dataset", default="", help="dataset slug with prior results/checkpoints")
    ap.add_argument("--cache-dataset", default=None, help="defaults <user>/airfrans-cache")
    ap.add_argument("--kernel-slug", default=None, help="defaults <user>/geo-op-session")
    args = ap.parse_args()

    root = HERE.parent
    user = kaggle_username()
    repo_url = args.repo_url or subprocess.run(
        ["git", "-C", str(root), "remote", "get-url", "origin"],
        capture_output=True, text=True, check=True).stdout.strip()
    cache_ds = args.cache_dataset or f"{user}/airfrans-cache"
    code_ds = f"{user}/geo-op-code"
    slug = args.kernel_slug or f"{user}/geo-op-session"

    driver = (HERE / "session_driver.py").read_text()
    driver = (driver
              .replace("{{REPO_URL}}", repo_url)
              .replace("{{COMMIT}}", git_head(root))
              .replace("{{SWEEP}}", args.sweep)
              .replace("{{RUNS_SLUG}}", args.runs_dataset.split('/')[-1] if args.runs_dataset else ""))

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "session_driver.py").write_text(driver)
        meta = {
            "id": slug,
            "title": slug.split("/")[-1],
            "code_file": "session_driver.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "machine_shape": "NvidiaTeslaT4",
            "enable_internet": True,
            "dataset_sources": [cache_ds, code_ds] + ([args.runs_dataset] if args.runs_dataset else []),
            "kernel_sources": [],
            "competition_sources": [],
        }
        (tdp / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", td], check=True)
    print(f"pushed {slug}; monitor: kaggle kernels status {slug}")


if __name__ == "__main__":
    main()
