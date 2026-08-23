# CONTEXT.md — Engineering context & frozen interfaces

> Read this before touching any code. It is the single source of truth for
> environment constraints, conventions, and cross-module interfaces.
> Changes to a **FROZEN** section require an entry in `DECISIONS.md` and a
> migration of all dependents in the same commit.

Project spec: [`05_geometry_aware_neural_operator_surrogate_uq.md`](../05_geometry_aware_neural_operator_surrogate_uq.md) (the research design doc).
Plan: [`docs/PLAN.md`](PLAN.md). Decision log: [`docs/DECISIONS.md`](DECISIONS.md). Status: [`docs/PROGRESS.md`](PROGRESS.md).

---

## 1. Hardware & environment (measured 2026-08-24)

| Item | Value | Consequence |
|---|---|---|
| GPU | NVIDIA Quadro P2000 (mobile), **4 GB VRAM**, Pascal **sm_61**, no tensor cores | fp32 only (Pascal fp16 is crippled); small batches; subsample volume points; models ≈1–2 M params |
| CUDA driver | 573.71, CUDA 12.8 driver | must use **cu126** PyTorch wheels — cu128+ dropped Pascal |
| CPU | i7-8850H, 6C/12T | avoid CPU-heavy work per user instruction; single-process dataloading, `num_workers<=2` |
| RAM | 32 GB | whole cached AirfRANS surface set fits in RAM |
| Disk | D: 120 GB free | AirfRANS preprocessed ≈ 10 GB zip + unzipped; cache adds a few GB. Fine. |
| OS | Windows 11, Git Bash + PowerShell | pure-Python deps only, **no compiled extensions** (no torch_scatter/torch_cluster/PyG) |
| Python | 3.11.15 via `uv` at `.venv/` | run with `.venv/Scripts/python.exe` |
| Torch | 2.8.0+cu126 | verified against sm_61 |

Env recreate: `uv venv --python 3.11 .venv && uv pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple`

**Never**: AMP/fp16 training, torch.compile (Windows+Pascal flaky), multi-GPU code paths, dependencies with C extensions beyond the standard scientific stack (numpy/scipy/pyvista are fine — they ship wheels).

## 2. Dataset facts (AirfRANS)

- 1000 steady incompressible RANS (OpenFOAM, SA turbulence model) sims over NACA 4/5-digit airfoils. Re ∈ [2M, 6M], AoA ∈ [−5°, +15°]. Chord = 1 m, freestream given as inlet velocity vector.
- Access via `airfrans` pip package: `af.dataset.download(root, file_name='Dataset', unzip=True, OpenFOAM=False)` → preprocessed, cropped version (use this, NOT the raw OpenFOAM one). Data lives in `data/raw/Dataset/` (gitignored).
- Official tasks/splits: `full`, `scarce`, `reynolds` (Re-extrapolation), `aoa` (AoA-extrapolation) via `af.dataset.load(root, task, train)`.
- Licence **ODbL-1.0**: attribution + share-alike on derived *databases*. We commit split manifests and code, not processed data.
- Per-sim: volume nodes (~180k pre-crop; verify actual) with position, velocity (2), pressure, nu_t, distance function, normals (nonzero only on surface nodes); surface identified where distance function == 0 / normals nonzero. `airfrans.Simulation` exposes positions, fields, airfoil surface arrays and force coefficient computation — **A1 must introspect the installed package and record exact attribute names, shapes, units and the coefficient convention in `docs/DATA_NOTES.md`**. Do not trust this paragraph's details; verify.
- Pressure in the dataset is kinematic-like / needs a normalization convention — pin it in DATA_NOTES.md by reproducing the dataset's own C_L, C_D within tolerance (the gating test, see §6).

## 3. Repository layout (FROZEN)

```
src/
  data/       airfrans_loader.py  splits.py         # A1
  geometry/   sdf.py normals.py quadrature.py symmetry.py   # A2
  physics/    force_integration.py residuals.py     # A2
  models/     common.py gnn.py sdf_fno.py geo_transformer.py heads.py  # A3
  eval/       metrics.py harness.py schema.py       # A4
  uq/         ensembles.py conformal.py coverage.py # A5
  active/     pool.py acquisition.py diversity.py   # A5
  utils/      config.py seed.py io.py               # A4
scripts/      build_cache.py train.py evaluate.py make_figures.py sweep.py subset_drivaernet.py
configs/      *.yaml (one per model; split & seed via CLI overrides)
tests/        pytest, no GPU required to pass
fluent/       TUI journals + README (runs deferred)
data/         raw/ processed/ splits/ (raw+processed gitignored)
results/      <run_id>/  (metrics.json + config.yaml committed; checkpoints gitignored)
docs/         this file, PLAN, DECISIONS, PROGRESS, DATA_NOTES
paper/        LaTeX skeleton
```

`src` is a package: absolute imports `from src.physics.force_integration import ...`. Run everything from repo root. Every script accepts `--help`.

## 4. Data cache interface (FROZEN)

`scripts/build_cache.py` converts raw AirfRANS → one `.npz` per sim at `data/processed/airfrans/<sim_name>.npz` + `data/processed/airfrans/manifest.json` (list of sims with per-sim scalars). All arrays float32, C-order. Keys:

| Key | Shape | Meaning |
|---|---|---|
| `surf_pos` | (Ns, 2) | surface point coords, ordered along the airfoil contour (TE→ around →TE) |
| `surf_normal` | (Ns, 2) | **outward** unit normals (pointing into the fluid) |
| `surf_ds` | (Ns,) | quadrature weight = facet length share per point, Σ ≈ perimeter |
| `surf_p` | (Ns,) | surface pressure in dataset native units |
| `surf_tau` | (Ns, 2) | wall shear stress vector, dataset native units |
| `vol_pos` | (Nv, 2) | volume node coords (subsampled to ≤ 32768 if larger) |
| `vol_u` | (Nv, 2) | velocity |
| `vol_p` | (Nv,) | pressure |
| `vol_nut` | (Nv,) | turbulent viscosity |
| `vol_sdf` | (Nv,) | distance function to airfoil (≥0 in fluid) |
| `cond` | (2,) | freestream velocity vector (u∞x, u∞y) — encodes speed + AoA |
| scalars | — | `re`, `aoa_deg`, `u_inf_mag`, `cl_true`, `cd_true`, `n_surf`, `n_vol` |

`src/data/airfrans_loader.py` provides
`AirfransSurfaceDataset(split_file, processed_dir, normalize_stats)` returning per-item dict of torch tensors with the keys above (surface subset + `cond`, `cl_true`, `cd_true`, `sim_name`), and `collate(items) -> Batch` where **Batch concatenates points and carries `batch_idx` (long, (ΣNs,))** — PyG-style, no padding. Normalization stats (mean/std per field, computed on train split of `full` only) stored at `data/processed/airfrans/norm_stats.json`; inputs normalized, model outputs are in normalized space, loader exposes `denormalize(field_name, tensor)`.

## 5. Splits (FROZEN)

`src/data/splits.py` writes `data/splits/<name>.json`:
`{"name", "train": [sim...], "cal": [sim...], "test": [sim...], "seed", "description"}`.

| name | source |
|---|---|
| `full` | official `full` task; last 100 of shuffled(seed=0) train → `cal` |
| `scarce` | official `scarce` task; cal = 20% of train (seed=0) |
| `reynolds` | official `reynolds` task; cal carved from train |
| `aoa` | official `aoa` task; cal carved from train |
| `shape5` | NEW: train = all NACA-4-digit sims (from sim-name parsing), test = all 5-digit; cal from train |
| `combined` | NEW: train = 4-digit ∩ mid-Re band; test = 5-digit ∩ outer-Re. Definition detail → DATA_NOTES.md |

Rule: **cal never overlaps train or test; test is never touched by normalization or calibration.** `tests/test_splits_no_leakage.py` asserts pairwise disjointness for every manifest.

## 6. Physics interface (FROZEN)

`src/physics/force_integration.py`:
```python
def integrate_forces(p, tau, normal, ds, cond, batch_idx=None, rho=1.0, a_ref=1.0)
    -> dict(cl=..., cd=..., fx=..., fy=...)   # torch, differentiable, per-sample
```
Sign/convention: F = Σ (−p·n + τ) ds with n **outward** (into fluid); drag = F·ê∞, lift = F·ê⊥ where ê∞ = cond/|cond|, ê⊥ = rot90(ê∞). Coefficients divide by ½ρ|u∞|²·A_ref. Whatever unit convention the dataset uses (kinematic pressure etc.) is absorbed here so that **on ground-truth fields this reproduces `cl_true`/`cd_true` to ≤1% median relative error across sims — this is the gating test `tests/test_force_integration.py` and nothing downstream proceeds until it passes.**

`src/physics/residuals.py`: `divergence_residual(u, pos, batch_idx)` (least-squares gradient on kNN neighborhoods), `noslip_residual(u, sdf, delta)`, and `symmetry_residual(model, batch)` lives in `src/geometry/symmetry.py`.

## 7. Model interface (FROZEN)

All models subclass `src/models/common.py::SurrogateBase(nn.Module)`:

```python
forward(batch: dict) -> dict
# batch: surf_pos, surf_normal, surf_ds, cond, batch_idx (+ vol_* for volume variants)
# returns: {"p": (ΣNs,), "tau": (ΣNs,2), "coef_head": (B,2)}  # normalized space; coef_head = (CL, CD)
```

- `M1` `gnn.py`: kNN message-passing GNN (k=16, native scatter via `torch.index_add_`), node feats (pos, normal, curvature, cond broadcast).
- `M2` `sdf_fno.py`: GINO-lite — kernel encoder scattering surface feats to a regular latent grid (64×64 over a fixed bbox) conditioned on SDF, FNO2d stack (modes 16, width 32), kernel decoder querying surface points. Self-implemented; no `neuraloperator` dependency.
- `M3` `geo_transformer.py`: Transolver-style physics attention, M=32 slices, 4 layers, d=128.
- `heads.py`: shared pooled MLP head for (CL, CD) — same head module for all three.
- Param budgets within ±20% of 1.5 M. `common.py` provides `count_params`, positional (Fourier feature) encodings, MLP builder.
- Constructors take a plain config dict; no global state; deterministic given `src/utils/seed.py::seed_all(seed)`.

## 8. Losses (FROZEN)

`src/models/losses.py` (A4 owns): `total_loss(pred, batch, weights, norm_ref)` = rel-L2 on p and tau + λ_F·force + λ_S·sym (train-time terms per spec §5.3–5.5). Each auxiliary term normalized by its value at init (`norm_ref` captured on first batch). Default λ's = 0 (pure data loss) — physics terms are ablation flags in config.

## 9. Results schema (FROZEN)

Each run: `results/<run_id>/` with `config.yaml`, `metrics.json`, `history.csv`; checkpoint at `checkpoints/<run_id>/{last,best}.pt` (gitignored). `run_id = {model}_{split}_s{seed}[_{tag}]`.

`metrics.json` (produced only by `src/eval/harness.py`, schema-validated by `src/eval/schema.py`):
```json
{"run_id","model","split","seed","tag","params","train_time_s","epochs","device",
 "field":{"p_rel_l2","tau_rel_l2","p_mae","tau_mae"},
 "coef":{"cl_head_mae","cd_head_mae","cl_int_mae","cd_int_mae","cd_spearman","cl_rel","cd_rel"},
 "consistency":{"fsc_cl","fsc_cd","fsc_rel_cl","fsc_rel_cd","sym_residual","antisym_cl_gap"},
 "cost":{"infer_ms_per_sim","peak_mem_mb"}}
```
UQ adds `results/uq/<ensemble_id>.json` with coverage/width per nominal level per split (schema in `src/eval/schema.py`). All evaluation in **denormalized physical units**.

## 10. Config & training (FROZEN)

Plain YAML (no Hydra), loaded by `src/utils/config.py::load_config(path, overrides: list[str])` supporting dotted CLI overrides (`train.lr=3e-4`). One YAML per model in `configs/{gnn,sdf_fno,transolver}.yaml` containing `model:`, `train:` (lr, epochs, batch_size, weights λ), `data:` (split name, paths). `scripts/train.py --config configs/gnn.yaml split=full seed=0`:
- checkpoints **every epoch** (`last.pt`: model+optim+epoch+RNG) and auto-resumes if `last.pt` exists;
- logs epoch losses to `history.csv`; no wandb (offline machine-friendly);
- fp32, Adam + cosine schedule, gradient clip 1.0;
- ends by invoking the eval harness → `metrics.json`.

## 11. Testing & quality bar

- `pytest tests/` must pass **on CPU** (models tested with tiny widths + synthetic data).
- Synthetic geometry tests: force integration on a circle with analytic pressure field (zero net force for constant p; known force for p = n·ê); normals/quadrature on a unit circle sum to perimeter 2π within 1e-3.
- Every module: docstrings stating shapes; no silent shape coercion — assert early.
- Commit style: small, message prefix `[data]`, `[models]`, `[physics]`, `[uq]`, `[eval]`, `[infra]`, `[docs]`, `[paper]`.

## 12. Ground rules for agents

1. Do not modify files outside your assigned directories (see PLAN.md task table) except appending to `docs/PROGRESS.md`.
2. Interface changes → propose in PROGRESS.md, do not unilaterally change FROZEN sections.
3. GPU is available but keep it free for training runs; unit tests run on CPU. Nothing CPU-parallel-heavy (user constraint): `num_workers<=2`, no multiprocessing pools.
4. Record any dataset/API surprise in `docs/DATA_NOTES.md` immediately.
5. Windows paths: always use `pathlib`, never hard-code separators; scripts must run via `.venv/Scripts/python.exe` from repo root.
