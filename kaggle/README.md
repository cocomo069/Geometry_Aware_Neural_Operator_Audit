# Kaggle execution tier (D-013)

Research-grade sweeps run on Kaggle free tier (P100 or 2×T4, ~30 GPU-h/week, 12 h max
per session). The local P2000 only does smoke tests and gating. Sessions are chainable:
every run checkpoints per epoch and `scripts/sweep.py` skips completed runs, so a killed
session costs at most one epoch.

## One-time setup (user)

1. Create a Kaggle API token (kaggle.com → Settings → API → Create New Token) and save
   it at `C:\Users\LAPTOP\.kaggle\kaggle.json`.
2. Approve creation of the GitHub repo (needed so the Kaggle kernel can clone the code).
   Private repo works: add a fine-grained read-only PAT as a Kaggle **secret** named
   `GH_TOKEN` (Kaggle → Add-ons → Secrets) — do not paste tokens anywhere else.
3. `uv pip install --python .venv/Scripts/python.exe kaggle`

## Workflow (orchestrator, repeatable)

1. `python kaggle/push_cache.py` — packages `data/processed/airfrans` + `data/splits`
   into a versioned Kaggle Dataset (`<user>/airfrans-cache`). Re-run only when the cache
   changes.
2. `python kaggle/launch.py --sweep configs/sweeps/core.yaml [--runs-dataset <user>/geo-op-runs]`
   — renders `kernel-metadata.json`, sets the sweep name as an env line in the driver,
   pushes the kernel (`kaggle kernels push`), which Kaggle executes with GPU.
3. The kernel (`session_driver.py`): clones the repo at a pinned commit, installs
   requirements, symlinks the cache dataset to `data/processed`, downloads previous
   results/checkpoints dataset if given (chaining), runs the sweep with a wall-clock
   guard (~11 h), then leaves `results/` + `checkpoints/` zipped in `/kaggle/working`
   as kernel output.
4. `python kaggle/pull_results.py --kernel <slug>` — downloads kernel output, unpacks
   `results/` into the repo (metrics.json are committed), keeps checkpoints under
   `checkpoints/` (gitignored), and uploads the merged state as the runs dataset for
   the next session to resume from.

## Session sizing

One P100 session ≈ 11 usable hours. Partition sweeps in `configs/sweeps/` so a
partition fits one session; the skip-existing logic makes over-provisioning harmless.
GPU quota resets weekly — queue partitions in priority order from PLAN.md Phase 3.
