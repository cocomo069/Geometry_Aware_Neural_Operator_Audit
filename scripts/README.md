# scripts/ — entry points

Run everything from the repo root as modules: `.venv/Scripts/python.exe -m scripts.<name>`.

## Data

| Script | Purpose |
|---|---|
| `download_data.py` | Fetch AirfRANS (~10 GB) into `data/raw/` |
| `build_cache.py` | Raw → per-sim `.npz` cache under `data/processed/airfrans/` (schema: `docs/CONTEXT.md` §4) |
| `subset_drivaernet.py` | DrivAerNet++ subset prep (unused; 3D leg is future work) |

## Training and evaluation

| Script | Purpose |
|---|---|
| `train.py` | Train one run (`--config configs/<model>.yaml split=<split> seed=<n> [tag=..]`); checkpoints every epoch, resumes from last |
| `evaluate.py` | Evaluate a run dir into `metrics.json` + `per_sim.csv` |
| `sweep.py` | Run a sweep spec (`configs/sweeps/*.yaml`), skipping finished runs |
| `run_uq.py` | Deep-ensemble + split-conformal calibration reports into `results/uq/` |
| `run_active.py` | Score the NACA pool, select the three k=8 arms into `results/active/` |

## Fluent leg

| Script | Purpose |
|---|---|
| `collect_fluent.py` | Collect per-case Fluent outputs into `results/fluent/` (`--gci` adds the grid study) |
| `compare_fluent.py` | Surrogate-vs-Fluent comparison on the accepted set |

## Paper artifacts

| Script | Purpose |
|---|---|
| `make_tables.py` | Tables 1–5 as LaTeX float bodies + Markdown mirrors into `paper/tables/` |
| `make_figures.py` | Figures 1, 4–12 from committed metrics into `paper/figures/` |
| `render_examples.py` | Figures 2–3 (checkpoint inference on CPU) |
| `make_model_cards.py` | Model cards from committed metrics into `model_cards/` |
| `make_results_digest.py` | `docs/RESULTS_AUTO.md` |
| `make_arxiv.py` | Self-contained arXiv source zip from `paper/` |
