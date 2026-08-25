# geo-operator-audit

**Geometry-aware neural operator surrogates for external aerodynamics: physics-consistency,
OOD generalization and calibrated uncertainty on AirfRANS.**

Evaluation/methodology study: three geometry-conditioning families (kNN GNN, SDF-conditioned
FNO in the GINO style, Transolver-style geometry transformer) audited with a shared protocol —
force self-consistency, symmetry residuals, multi-axis OOD splits, deep ensembles + split
conformal coverage, data-efficiency curves, and an uncertainty-driven active-learning loop
verified against independent Ansys Fluent RANS cases.

Design document: [`05_geometry_aware_neural_operator_surrogate_uq.md`](05_geometry_aware_neural_operator_surrogate_uq.md)

## Status

Under active development. **New here? Read [docs/OVERVIEW.md](docs/OVERVIEW.md) — the living
master document** that explains the whole project in plain language: what it is, every concept
behind the paper, everything built, every problem faced, the results so far, and where we are.

Deeper detail: [docs/PLAN.md](docs/PLAN.md) (execution plan),
[docs/PROGRESS.md](docs/PROGRESS.md) (dated log), [docs/DECISIONS.md](docs/DECISIONS.md)
(decision log), [docs/CONTEXT.md](docs/CONTEXT.md) (frozen engineering interfaces),
[docs/HANDOFF.md](docs/HANDOFF.md) (cold-resume state).

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
