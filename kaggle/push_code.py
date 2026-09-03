"""Push the repo (git archive of HEAD) as a small Kaggle dataset so kernels get
fresh code without a GitHub token. Re-run on every code change before launch.

Usage: python kaggle/push_code.py [--slug user/geo-op-code]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=None)
    args = ap.parse_args()
    root = HERE.parent
    from _common import kaggle_username
    user = kaggle_username()
    slug = args.slug or f"{user}/geo-op-code"

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        subprocess.run(["git", "-C", str(root), "archive", "--format=tar.gz",
                        "-o", str(tdp / "repo.tar.gz"), "HEAD"], check=True)
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        (tdp / "COMMIT.txt").write_text(head)
        import json
        (tdp / "dataset-metadata.json").write_text(json.dumps(
            {"title": "geo-op-code", "id": slug, "licenses": [{"name": "MIT"}]}, indent=2))
        exists = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "status", slug],
                                capture_output=True).returncode == 0
        cmd = ([sys.executable, "-m", "kaggle", "datasets", "version", "-p", str(tdp),
                "-m", f"code {head[:8]}", "--dir-mode", "zip"]
               if exists else
               [sys.executable, "-m", "kaggle", "datasets", "create", "-p", str(tdp), "--dir-mode", "zip"])
        subprocess.run(cmd, check=True)
    # Wait for Kaggle to finish PROCESSING the new version before returning: a
    # kernel launched against a still-processing dataset mounts no repo.tar.gz and
    # dies with "no repo.tar.gz ... and no GH_TOKEN" (observed on the ablations
    # launch). Poll status until "ready" so callers (cycle.py) can launch safely.
    import time
    for _ in range(40):  # ~10 min max
        r = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "status", slug],
                           capture_output=True, text=True)
        if "ready" in ((r.stdout or "") + (r.stderr or "")).lower():
            break
        time.sleep(15)
    else:
        print(f"WARNING: {slug} not confirmed ready after ~10 min; launching anyway")
    print(f"pushed code dataset {slug} @ {head[:8]} (ready)")


if __name__ == "__main__":
    main()
