"""Package data/processed/airfrans + data/splits as a versioned Kaggle Dataset.

Usage: python kaggle/push_cache.py [--slug user/airfrans-cache]
First run creates the dataset; later runs push new versions.
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=None)
    args = ap.parse_args()
    root = HERE.parent
    user = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
    slug = args.slug or f"{user}/airfrans-cache"

    with tempfile.TemporaryDirectory(dir=root / "data") as td:
        tdp = Path(td)
        shutil.copytree(root / "data" / "processed", tdp / "processed")
        shutil.copytree(root / "data" / "splits", tdp / "splits")
        (tdp / "dataset-metadata.json").write_text(json.dumps({
            "title": "airfrans-cache",
            "id": slug,
            "licenses": [{"name": "ODbL-1.0"}],
        }, indent=2))
        exists = subprocess.run(["kaggle", "datasets", "status", slug],
                                capture_output=True).returncode == 0
        cmd = (["kaggle", "datasets", "version", "-p", str(tdp), "-m", "cache update", "--dir-mode", "zip"]
               if exists else
               ["kaggle", "datasets", "create", "-p", str(tdp), "--dir-mode", "zip"])
        subprocess.run(cmd, check=True)
    print(f"pushed cache dataset {slug}")


if __name__ == "__main__":
    main()
