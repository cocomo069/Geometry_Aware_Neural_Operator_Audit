<!--
model_cards/TEMPLATE.md

One card per (architecture x dataset x split), per spec section 13. Copy to
model_cards/<run_id>.md where run_id = {model}_{split}_s{seed}[_{tag}], matching
results/<run_id>/. Ensemble cards use the ensemble_id instead.

Rules for filling this in:
  * Every number comes from a committed results/<run_id>/metrics.json or
    results/uq/<ensemble_id>.json. If a number is not in a committed file, it
    does not go on the card.
  * Leave a field as "not measured" rather than estimating it. A card whose
    numbers are partly guessed is worse than no card, because a reader cannot
    tell which half to trust.
  * The "Do not use for" section is the point of the exercise. It is not
    boilerplate and it must not be softened.
  * Cards are generated/refreshed by scripts/make_figures.py; hand edits to the
    metric tables will be overwritten. Hand-write the prose sections.
-->

# Model card — `<run_id>`

*Card version: 1.0 · Last updated: YYYY-MM-DD · Generated from `results/<run_id>/metrics.json`*

---

## 1. Model details

| Field | Value |
|---|---|
| Run ID | `<run_id>` |
| Architecture | `M1 GNN` / `M2 SDF-FNO (GINO-lite)` / `M3 Transolver-style` |
| Reference paper the design follows | *(citation)* |
| **Reimplementation?** | **Yes — reimplemented in the style of the cited paper, not run from the authors' code.** Absolute numbers are therefore not comparable with published ones. |
| Parameters | *(count)* |
| Task | surface fields (`p`, `tau_w`) + coefficient head (`CL`, `CD`) |
| Input | surface point cloud (position, outward normal, curvature) + freestream condition |
| Output | per-point `p`, `tau_w`; pooled `(CL, CD)`; `C^int` by quadrature of the predicted fields |
| Precision | fp32 (no AMP — the target GPU has no usable fp16) |
| Framework | PyTorch *(version)* |
| Config | `configs/<model>.yaml` at commit `<sha>` |
| Seed | *(int)* — `src/utils/seed.py::seed_all` |

## 2. Intended use

**Intended.** Research use as a *surrogate under audit*: rapid ranking of airfoil
sections within the training envelope, and as a subject of the consistency and
calibration protocol this repository implements.

**Out of scope.** Anything load-bearing. See §7.

**Intended users.** Researchers reproducing or extending this study. Not a
design tool.

## 3. Training data

| Field | Value |
|---|---|
| Dataset | AirfRANS *(or)* DrivAerNet++ subset |
| **Licence** | **ODbL-1.0** (AirfRANS: attribution + share-alike on derived *databases*) *(or)* **CC BY-NC 4.0** (DrivAerNet++: non-commercial only, attribution) |
| Split | `full` / `scarce` / `reynolds` / `aoa` / `shape5` / `combined` |
| Split manifest | `data/splits/<name>.json` (committed) |
| Train / cal / test sizes | *(n / n / n)* |
| Leakage check | `tests/test_splits_no_leakage.py` asserts pairwise disjointness — **passing** |
| Preprocessing | cached to `data/processed/`; inputs normalised with train-split statistics only |
| Known composition | NACA 4-digit: *n*; 5-digit: *n*; Re range: *(a, b)*; AoA range: *(a, b)* |

**Redistribution.** This repository ships split manifests and code, never a
processed copy of either dataset.

## 4. Training procedure

| Field | Value |
|---|---|
| Hardware | *(GPU, VRAM)* |
| Training wall-clock | *(h)* — measured, not estimated |
| Epochs (planned / completed) | *(n / n)* |
| Optimiser / schedule | Adam, cosine, grad clip 1.0 |
| Learning rate / batch size | *(lr / bs)* |
| Loss weights | `lambda_F`, `lambda_B`, `lambda_S`, `lambda_D` = *(...)* |
| Checkpointing | every epoch; resumes from `last.pt` |
| Energy / cost | *(kWh or GPU-hours; state if not measured)* |

## 5. Evaluation

All metrics in **denormalised physical units**, produced only by
`src/eval/harness.py` and schema-validated. Test data was never seen during
normalisation, training or calibration.

### 5.1 In-distribution

| Metric | Value |
|---|---|
| `p` relative L2 | |
| `tau_w` relative L2 | |
| `CD` MAE (head / integrated) | |
| `CL` MAE (head / integrated) | |
| `CD` Spearman rank correlation | |
| FSC (`CD`) / FSC-relative | |
| Symmetry residual | |
| Antisymmetry gap, `CL(-a) + CL(a)` | |
| Inference latency per simulation | |
| Peak memory | |

**Baselines on the same split** (required — a model that does not beat these has
not learned anything worth reporting):

| Baseline | `CD` MAE | `CD` Spearman |
|---|---|---|
| Constant predictor (dataset mean) | | |
| Ridge (geometry params + condition → `CD`) | | |
| This model | | |

### 5.2 Out-of-distribution

| Split | Shift Δ | `p` rel-L2 | `CD` MAE | FSC-rel | Coverage @90% |
|---|---|---|---|---|---|
| `reynolds` | | | | | |
| `aoa` (pre-stall) | | | | | |
| `aoa` (post-stall) | | | | | |
| `shape5` | | | | | |
| `combined` | | | | | |

### 5.3 Calibration

Deep ensemble (K = *n*) + split conformal, calibrated on the `cal` split.

| Split | Nominal 80% | Nominal 90% | Nominal 95% | Median width @90% |
|---|---|---|---|---|
| in-distribution | | | | |
| *(each OOD split)* | | | | |

**OOD/ID coverage ratio:** *(value)* — the single number summarising calibration
robustness.

**Guarantee scope.** The conformal guarantee holds under exchangeability between
calibration and test. It **does not hold** on the shape-family splits, where
exchangeability fails by construction. Coefficient intervals are calibrated at
the simulation level, where the exchangeable unit is well defined; pointwise
field intervals are conformalised at the simulation level with a
quantile-over-points score and carry only that weaker interpretation.

## 6. Known failure modes

Fill from measurement, not expectation. Delete any that were not observed and add
any that were.

- **Post-stall angle of attack.** Error rises sharply once test cases cross into
  separation. Degradation tracks separation onset rather than numerical distance
  from the training range: the model interpolates within a flow regime, it does
  not extrapolate across one.
- **Shape-family shift (4-digit → 5-digit).** *(quantify)*. Conformal coverage
  falls below nominal here; intervals do not widen enough to compensate.
- **Combined shift.** The hardest cell; error and undercoverage compound.
- **Force self-consistency degrades faster than field error under shift.** A
  large FSC is a warning that the model is unreliable on that input *even when no
  ground truth exists* — this is the intended use of the diagnostic.
- **Symmetry.** *(residual value)*. Not an equivariant architecture; a nonzero
  residual on symmetric sections at zero incidence quantifies reliance on
  coordinate accidents.
- **Extrapolation beyond the trained Reynolds band.** Untested outside
  *(range)*. Absence of a measurement is not evidence of good behaviour.

## 7. Do not use for

Explicit and non-negotiable:

- **Do not use for certification, safety-critical, or load-bearing engineering
  decisions.** This is a research surrogate audited for its failure modes, not a
  validated analysis tool.
- **Do not use outside the trained envelope** — Re outside *(range)*, AoA outside
  *(range)*, or section families other than *(families)* — without re-running the
  OOD protocol on the new envelope.
- **Do not treat the conformal interval as a guarantee under shape or regime
  shift.** Exchangeability fails there and the measured coverage shows it.
- **Do not substitute it for a solver in the post-stall regime.** Prediction and
  uncertainty are both unreliable there.
- **Do not use commercially if trained on DrivAerNet++** — CC BY-NC 4.0 forbids it.
- **Do not compare these numbers with published results for the cited
  architectures.** These are reimplementations under a different budget and a
  different data pipeline; only within-study comparisons are meaningful.

## 8. Ethical and licensing considerations

- Data licences: ODbL-1.0 (AirfRANS) and/or CC BY-NC 4.0 (DrivAerNet++). Both
  carry attribution obligations; the second forbids commercial use.
- Code released under MIT. The Fluent verification set generated for this work is
  MIT and deliberately unencumbered by either dataset licence.
- No personal data. Environmental cost is reported in §4 rather than omitted.

## 9. Reproduction

```bash
.venv/Scripts/python.exe scripts/train.py --config configs/<model>.yaml \
    split=<split> seed=<seed>
.venv/Scripts/python.exe scripts/evaluate.py --run-id <run_id>
```

| Field | Value |
|---|---|
| Commit | `<sha>` |
| Environment | `requirements.txt` (locked) |
| Determinism | `seed_all(seed)`; caveats *(cuDNN nondeterminism, etc.)* |
| Artifacts | `results/<run_id>/{config.yaml,metrics.json,history.csv}` (committed); checkpoints *(location; gitignored if large)* |

## 10. Citation

```bibtex
@misc{amin2026geoaudit,
  title  = {Geometry-aware neural operator surrogates for external aerodynamics:
            physics-consistency, out-of-distribution generalization and
            calibrated uncertainty},
  author = {Amin, T.},
  year   = {2026},
  note   = {arXiv preprint}
}
```

Also cite the dataset(s) and the architecture paper this model reimplements.
