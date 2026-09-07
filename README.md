# geo-operator-audit

**Geometry-aware neural operator surrogates for external aerodynamics: physics-consistency,
OOD generalization and calibrated uncertainty on AirfRANS.**

Evaluation/methodology study: three geometry-conditioning families (kNN GNN, SDF-conditioned
FNO in the GINO style, Transolver-style geometry transformer) audited with a shared protocol —
force self-consistency, symmetry residuals, multi-axis OOD splits, deep ensembles + split
conformal coverage, data-efficiency curves, and an uncertainty-driven active-learning loop
verified against independent Ansys Fluent RANS cases.

Design document: [`docs/05_geometry_aware_neural_operator_surrogate_uq.md`](docs/05_geometry_aware_neural_operator_surrogate_uq.md)

## Status

**Complete.** All planned experiments have run: the 3-model × 6-split core grid (400 epochs each),
deep ensembles (K=5 for Transolver/SDF-FNO, K=3 for the GNN) with split-conformal calibration,
data-efficiency curves (25–700 training sims × 3 seeds), ablations, and the full Ansys Fluent
verification leg (grid study, solver-offset replicas, and the active-learning case set). The paper
sources are in [`paper/`](paper/) and every figure and table regenerates from the committed
`results/` via `scripts/make_figures.py`, `scripts/render_examples.py` and `scripts/make_tables.py`.

**New here? Read [docs/OVERVIEW.md](docs/OVERVIEW.md) — the living
master document** that explains the whole project in plain language: what it is, every concept
behind the paper, everything built, every problem faced, and where we are. For the numbers and
what they mean, read **[docs/RESULTS.md](docs/RESULTS.md)** (living results doc).
The full documentation index is [docs/README.md](docs/README.md).

## Repository map

| Path | What lives there |
|---|---|
| [`paper/`](paper/) | The manuscript: single- and two-column builds, generated figures/tables, arXiv packages ([paper/README.md](paper/README.md)) |
| [`docs/`](docs/) | Living documentation + frozen plans ([docs/README.md](docs/README.md)) |
| [`src/`](src/) | The library: data, geometry, physics, models, eval, UQ, active learning, viz ([src/README.md](src/README.md)) |
| [`scripts/`](scripts/) | Entry points: data prep, training, evaluation, UQ, Fluent collection, paper artifacts ([scripts/README.md](scripts/README.md)) |
| [`configs/`](configs/) | Per-model YAML + sweep specs ([configs/README.md](configs/README.md)) |
| [`results/`](results/) | Committed run metrics, UQ reports, active-learning outputs, Fluent campaign data ([results/README.md](results/README.md)) |
| [`data/splits/`](data/splits/) | The exact train/cal/test manifests behind every result (raw/processed data gitignored) |
| [`fluent/`](fluent/) | The Fluent verification leg: mesh generator, journals, rendered cases, runbook ([fluent/README.md](fluent/README.md)) |
| [`model_cards/`](model_cards/) | One card per architecture, generated from committed metrics |
| [`kaggle/`](kaggle/) | Free-tier cloud training drivers ([kaggle/README.md](kaggle/README.md)) |
| [`tests/`](tests/) | 449 CPU-only pytest tests |

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt \
  --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
.venv/Scripts/python.exe -m scripts.download_data      # AirfRANS (~10 GB)
.venv/Scripts/python.exe -m scripts.build_cache        # per-sim .npz cache
.venv/Scripts/python.exe -m pytest tests/              # all CPU
```

Train / evaluate:

```bash
.venv/Scripts/python.exe -m scripts.train --config configs/gnn.yaml split=full seed=0
.venv/Scripts/python.exe -m scripts.evaluate --run gnn_full_s0
```

## Licences

Code: MIT. AirfRANS data: ODbL-1.0 (not redistributed here; split manifests only).
DrivAerNet++ (optional extension): CC BY-NC 4.0. Fluent verification cases (when released):
CC BY 4.0.

## Author

Taimoor Amin (BSc Mechanical Engineering, NUST; ML minor).
