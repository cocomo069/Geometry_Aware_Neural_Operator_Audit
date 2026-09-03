# RESULTS — the living results document

**Every result the project has produced, with plain-language interpretation.** Read
[OVERVIEW.md](OVERVIEW.md) first for the concepts; this file is the numbers and what they mean.
Updated as runs land. Last updated: **2026-08-25**.

Status legend: ✅ complete · 🔄 partial (some cells still training) · ⏳ not started.

Quick map of results → paper figures/tables:
- Core grid accuracy → **Table 1, Table 2, Fig 4, Fig 12**
- Force self-consistency → **Fig 5**
- Calibration/uncertainty → **Fig 6, Fig 7, Fig 8, Table 3**
- Data efficiency → **Fig 9** (⏳)
- Active learning + Fluent → **Fig 11, Table 5** (⏳, CPU-gated)

---

## 0. Gate G1 — is our "physics ruler" correct? ✅

Before trusting any force number, we checked that our force-integration code reproduces the
dataset's *own* published drag/lift when fed the true pressure fields.

- **C_D: median relative error 3.3e-4 (0.03%), max 1.6e-3, across all 1000 airfoils.**
- **C_L: median relative error 4.8e-6.**

**Meaning:** our integration is essentially exact — the numerical error is ~100× smaller than
any model error we measure, so every force-self-consistency (FSC) number downstream is
trustworthy. This was the single most important check in the project.

---

## 1. Baselines — the bar to beat ✅

Two non-learned baselines on all six splits:
- **Constant** (predict the training-mean field): field rel-L2 ≈ 1.4 (i.e. useless), C_D
  ranking Spearman ρ ≈ 0.27–0.42.
- **Ridge regression** (from the NACA shape numbers + flow condition → C_D directly):
  **C_D ranking ρ = 0.83–0.90 on every split, including out-of-distribution.**

**Meaning:** ridge is a "shockingly strong" baseline the field under-reports — a simple linear
model ranks designs by drag surprisingly well. Any neural model must clearly beat ρ ≈ 0.85 to
justify itself. (Our models reach 0.94–0.999, so they do — but the gap is smaller than you'd
expect, which is itself a finding.)

---

## 2. Core grid — 3 models × 6 splits, 400 epochs each ✅

The heart of the paper. **M1 = GNN, M2 = SDF-FNO (GINO-style), M3 = Transolver.** Splits ordered
easy→hard: `full` (in-distribution), `scarce`, `reynolds`, `aoa`, `shape5`, `combined`.

### 2.1 Field accuracy — surface pressure relative L2 error (lower = better)

| model | full | scarce | reynolds | aoa | shape5 | combined |
|---|---|---|---|---|---|---|
| **Transolver (M3)** | **0.017** | **0.067** | **0.077** | **0.109** | **0.055** | **0.139** |
| **SDF-FNO (M2)** | 0.026 | 0.083 | 0.098 | 0.111 | 0.064 | 0.153 |
| **GNN (M1)** | 0.074 | 0.154 | 0.223 | 0.188 | 0.130 | 0.270 |

Wall shear stress (τ_w) rel-L2 follows the same ordering (Transolver best, ~0.018 ID → 0.108
combined; GNN worst, ~0.068 → 0.236).

**Meaning:** on raw field accuracy the architecture matters and the ordering is clean —
**Transformer > Operator > Graph net**, at every split. Transolver is ~4× more accurate than the
GNN in-distribution. This matches the operator-learning literature.

### 2.2 Drag ranking — Spearman ρ (higher = better; ridge ≈ 0.83–0.90)

| model | full | scarce | reynolds | aoa | shape5 | combined |
|---|---|---|---|---|---|---|
| Transolver | 0.999 | 0.996 | 0.976 | 0.947 | 0.983 | 0.972 |
| SDF-FNO | 0.999 | 0.992 | 0.982 | 0.955 | 0.984 | 0.968 |
| GNN | 0.999 | 0.996 | 0.977 | 0.945 | 0.983 | 0.970 |

**Meaning — this is the paper's headline.** On the metric engineers actually use (ranking designs
by drag), **all three architectures are nearly identical** (ρ within ~0.01 of each other), and the
spread *between splits* (0.999 → 0.945) is larger than the spread *between models*. So the field's
usual "which architecture wins" comparison is measuring noise relative to the axis that matters
for deployment (distribution shift). All three still beat ridge — but not by as much as their
4× field-accuracy gap would suggest.

### 2.3 Force self-consistency (FSC) — |C_D from field − C_D from head|, lower = more consistent

| model | full | scarce | reynolds | aoa | shape5 | combined |
|---|---|---|---|---|---|---|
| Transolver | 0.0011 | 0.0039 | 0.0061 | 0.0043 | 0.0019 | 0.0086 |
| SDF-FNO | 0.0017 | 0.0052 | 0.0052 | 0.0034 | 0.0022 | 0.0086 |
| GNN | 0.0053 | 0.0086 | 0.0164 | 0.0104 | 0.0089 | 0.0176 |

**Meaning (contribution C1):** FSC is smallest in-distribution and largest on the hardest shift
(`combined`) for every model — so the gap between the model's predicted field and its predicted
drag **grows as the model is pushed out of distribution**. Because FSC needs no ground truth, this
makes it a usable **label-free "don't trust this prediction" detector**. It is not perfectly
monotonic across all six splits (e.g. `reynolds` FSC can exceed `aoa`), and the exact ordering is
mildly architecture-dependent — reported honestly in the paper. The better-field-accuracy models
(Transolver, SDF-FNO) are also more self-consistent (lower FSC) than the GNN.

### 2.4 Symmetry residual — equivariance violation (lower = more symmetric)

Range ~4.4 (in-distribution) to ~8.2 (aoa) across all models. **Meaning:** none of the three
architectures is built to be symmetric, so all violate the +α/−α mirror symmetry substantially,
and the violation grows with shift. This is the expected "plain models rely on coordinate
accidents, not equivariance" finding — it motivates the augmentation ablation.

### 2.5 The OOD hierarchy (a cross-cutting finding)

For every model, error grows in roughly this order: `full` < `shape5` < `scarce` < `aoa` ≈
`reynolds` < `combined`. The flow-regime shifts (`reynolds`, `aoa` — new Reynolds numbers,
crossing toward stall) hurt more than the shape-family shift (`shape5` — new airfoil shapes, same
flow regime). **Meaning:** *these models interpolate within a flow regime; they do not extrapolate
across one.* `combined` (shape + Reynolds shifted together) is hardest, as designed.

---

## 3. Calibration / uncertainty — deep ensembles + conformal 🔄

**K=5 deep ensembles complete for Transolver and SDF-FNO on {full, reynolds, aoa, shape5}; GNN
ensembles training (currently K=1–2, upgrading to K=5).** Then split conformal prediction on each
split's calibration set. Two modes: *matched* (calibrate and test on the same split) and
*transfer* (calibrate on `full`, deploy on the shifted split).

### 3.1 Coverage at nominal 90% — does the interval contain the truth 90% of the time?

Coefficient-C_D coverage, matched calibration, largest available K:

| model | full | reynolds | aoa | shape5 |
|---|---|---|---|---|
| SDF-FNO (K=5) | 0.95 | 0.90 | 0.94 | 0.76 |
| Transolver (K=5) | 0.91 | 0.94 | ~0.95 | (varies) |
| GNN (K=1) | 0.93 | 0.70 | 0.79 | 0.95 |

(Field-level and 80%/95% nominal levels are in Table 3; GNN row upgrades to K=5 when its
ensembles finish.)

**Meaning (contribution C3):**
- **In-distribution, the conformal guarantee holds** — coverage ≈ the nominal 90% it promises.
- **Under shift it degrades**, and the calibration error (ECE) jumps 6–10× on the flow-regime
  shifts (reynolds, aoa). The proper K=5 ensembles hold up better under mild shift than a single
  model (K=1) does, but shape-family and combined shifts still break the guarantee.
- **The takeaway sentence:** *conformal prediction repairs in-distribution calibration but its
  coverage guarantee is not robust to distribution shift — a conformal interval must not be read
  as protection against design-space extrapolation.* This is exactly the honest, useful message
  the paper is built to deliver.

### 3.2 Interval width (Fig 8)

Intervals widen from in-distribution to OOD (the model "knows" it's less sure), but not enough to
restore coverage — i.e. the widening under-compensates for the accuracy loss. This is why coverage
still drops despite wider intervals.

---

## 4. Data efficiency — Fig 9 ⏳

Not yet run. Plan: retrain each model at training-set sizes {25, 50, 100, 200, 400} × 3 seeds and
plot error vs training size (log-log). Hypothesis (from the literature): the GNN degrades more
gracefully with scarce data than the transformer, so the model ranking may **flip** at small data
— a crossing-curves result would be a clean finding.

---

## 5. Ablations — Table 4 ⏳

Planned (lean set for v1): SDF-FNO conditioning (SDF vs SDF+normals), physics-loss on/off for the
GNN. These isolate *why* the models behave as they do.

---

## 6. Active learning + Fluent verification — Fig 11, Table 5 🔄 (running)

The scoring half (rank a pool of unseen NACA shapes by the ensemble's uncertainty + FSC, then pick
a diverse set) runs on the existing models and emits `fluent/cases_to_run.json`. The verification
half is now **executing** in Ansys Fluent v211 (6 cores; y+-resolved C-grid; SA turbulence). The
falsifiable claim it tests: *do the "uncertain" cases really have higher surrogate error than
randomly chosen ones?*

### 6.1 Grid-independence study ✅ (NACA0012, Re 3e6, α5°, SA)

| level | cells | y+max | C_D | C_L |
|---|---|---|---|---|
| L1 (coarse) | 18,796 | 0.87 | 0.011122 | 0.5538 |
| L2 (medium) | 43,008 | 0.59 | 0.010755 | 0.5530 |
| L3 (fine) | 96,768 | 0.39 | 0.010755 | 0.5521 |

**C_D changes 3.3% from L1→L2 but only 0.003% from L2→L3 — the solution is grid-independent at
L2.** y+ < 1 at every level (wall-resolved). C_L varies < 0.3% across all levels. This is the
numerical-uncertainty band a CFD reviewer requires; the AL-verification and offset cases use the
L2 resolution. (Sanity: C_L = 0.553 vs thin-airfoil theory 2πα = 0.548, within 1%; C_D in the
expected 0.008–0.012 range for a smooth NACA0012 at this Re.)

### 6.2 Active-learning verification — running

The 24 AL-selected cases (acquisition / variance / random arms × 8) are solving now. Note ~20 are
at α=18° (post-stall) where steady RANS may not converge to a flat C_D — that is expected and is
itself the AL signal (acquisition concentrated on the hardest envelope corners). Surrogate-vs-Fluent
comparison + ParaView field contours (EnSight export bridge) land when the batch completes.

---

## 7. Cost accounting (honest small-compute reporting)

- Local gate/dev: Quadro P2000 (4 GB). Cloud training: Kaggle **T4** free tier, sessions chained.
- Core grid throughput: GNN ~0.7–3 h/run (kNN is heavy), Transolver/SDF-FNO ~6–25 min/run.
- Core grid (18 runs) ≈ two ~11 h T4 sessions. Ensembles (48 runs) span more.
- All training on free-tier cloud; the local machine's CPU was never used for training.

---

## 8. One-paragraph summary for a conversation

*"We trained three geometry-conditioned neural surrogates on AirfRANS at matched budget and put
them through a trust protocol. On raw field accuracy the transformer beats the operator beats the
graph net by up to 4×. But on drag ranking — what design actually uses — they're
indistinguishable, and all only modestly beat a linear baseline, so architecture choice matters
far less than the field assumes. Their force self-consistency degrades out of distribution, which
makes it a free uncertainty flag. And split-conformal uncertainty intervals are calibrated
in-distribution but lose their guarantee under shift, worst when the flow regime changes. The
message is: these models are usable inside their training regime and should be trusted with
explicit uncertainty there, but neither accuracy nor calibration survives extrapolation across
flow regimes."*
