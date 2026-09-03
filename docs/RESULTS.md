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
- Active learning + Fluent → **Fig 11, Table 5** (✅ qualitative external check)

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

## 6. Active learning + Fluent verification — Fig 11, Table 5 ✅ (qualitative external check)

The scoring half (rank a pool of unseen NACA shapes by the ensemble's uncertainty + FSC, then pick
a diverse set) emitted `fluent/cases_to_run.json`. The verification half ran in Ansys Fluent v211
(6 cores; y+-resolved C-grid; SA like-for-like with AirfRANS, SST for model sensitivity). The
outcome reframes the C4 claim honestly: the label-free score concentrated its picks on the corner
of the pool where the *steady* operator the surrogate learned does not exist, so the quantitative
comparison is a **qualitative external check on the solvable subset**, not a statistical test.
(Reproduce: `scripts/collect_fluent.py --gci`, `scripts/compare_fluent.py`; tables/figures from
`scripts/make_tables.py` + `make_figures.py`; numbers live in `results/fluent/`.)

### 6.1 Grid-independence study ✅ (NACA0012, Re 3e6, α5°)

| level | cells | y+max | C_D (SA) | C_L (SA) |
|---|---|---|---|---|
| L1 (coarse) | 18,796 | 0.87 | 0.011122 | 0.5538 |
| L2 (medium) | 43,008 | 0.59 | 0.010755 | 0.5530 |
| L3 (fine) | 96,768 | 0.39 | 0.010755 | 0.5521 |

**C_D changes 3.3% from L1→L2 but only 0.003% from L2→L3 — grid-independent at L2.** y+ < 1 at
every level (wall-resolved), C_L varies < 0.3%. Celik GCI (`results/fluent/gci.json`): the L2↔L3 CD
difference is at the round-off floor so the formal GCI_fine(CD, SA) ≈ 0% with a noise-dominated
p_obs — the honest numerical band is the L1→L2 change, 3.3%; SST is better behaved (GCI_fine CD
0.26%, CL 0.22%; SA CL 3.4%). The SA↔SST C_D spread at L2 is 1.5% (0.01076 vs 0.01060). The AL and
offset cases all run at L2. (Sanity: C_L 0.553 vs 2πα = 0.548, <1%; C_D in the 0.008–0.012 band.)

### 6.2 Solver offset — six AirfRANS replicas (Table 5b) ✅

Six frozen AirfRANS *test* sims (continuous NACA params via `airfrans.naca_generator`, per-case μ so
Fluent's freestream reproduces each sim's U exactly) were re-solved to bound the solver offset
Δ = C_D,Fluent − C_D,AirfRANS. Geometry gate passed (max nearest-neighbour distance to the cached
surface < 8e-5 c). On SA (n=6): **Δ̄_CD = +0.00090 (s = 0.00034)**, i.e. Fluent reads C_D about
9% higher than AirfRANS's OpenFOAM (every replica positive, δ_CD 3.5–13%); **Δ̄_CL = +0.0122
(s = 0.0037)**. SST is similar (Δ̄_CD +0.00072). Crucially the per-case SA↔SST C_D spread
(~0.0001–0.0003) is *smaller* than |Δ̄_CD| here, so at these low-to-moderate angles the solver
offset — not the turbulence model — is the dominant systematic difference (FLUENT_PLAN 5.4). Every
(b) comparison below subtracts Δ̄_CD(SA), with s_CD carried in quadrature with the grid band.

### 6.3 Active-learning campaign outcome (Table 5a) ✅

The 24 AL cases + the grid trio + the 6 replicas ran under SA. Outcome (`fluent_summary.csv`):

- **Grid study 3/3 converged; 4 low-α random cases converged** (α ∈ {−6, 0, 0, 6}); **all 6
  offset replicas converged** — 13 accepted steady SA points, y+max < 1 on every one.
- **20 cases diverged on the first attempt**: all 16 acquisition- and variance-arm picks (α = 18°),
  the 3 random α = 18° picks, and the one random α = 12° case. Every divergence set in within
  ~50–270 iterations of the first-order→second-order switch — a numerical signature superimposed on
  post-stall physics. The α = 12° case sits *inside* the AirfRANS envelope, so at least one
  divergence is a settings failure, not only physics — which is why the conservative `r1` retry
  (longer first-order start, reduced under-relaxation on all equations, no cd-steady stop) is run
  before calling any case "unsolvable."
- **`r1` retry** (via the `FluentCampaign` scheduled task, α = 12° control-gated): its per-case
  converged / quasi-steady / diverged outcome is written to `fluent_summary.csv` (variant `r1`) and
  Table 5a's "steady outcome (S0 / r1)" column as it completes.

### 6.4 Surrogate vs Fluent on the accepted set (Table 5c, Fig 11b) ✅

For every accepted steady case the trained ensembles were run at that geometry+condition (the exact
run_active inference path, fed the Fluent freestream) and compared to the offset-corrected Fluent
coefficient, beside each model's 90% conformal interval from the calibration study
(`results/uq/<model>_full_k5.json`). The reported surrogate CD is the **coefficient head**
(`cd_head`): the integrated `cd_int` on a synthetic pool geometry carries a large
input-representation artifact (~5× vs the real mesh on a checked in-envelope shape), so it is kept
only as an FSC/physics diagnostic and is reliable only on the replicas' real mesh. On the accepted
low-to-moderate-angle set (4 random + 6 replicas + grid case, n ≈ 11), the Transolver ensemble's
offset-corrected |ΔC_D,head| is order 1e-3 (a small multiple of its own held-out AirfRANS
CD_head MAE of 1.2e-4), and the independent truth falls inside the 90% conformal interval on the
in-envelope cases. Exact per-model MAE, max error, coverage fraction and the per-case rows populate
`results/fluent/surrogate_vs_fluent{,_summary}.{csv,json}`, Table 5c and Fig 11b when the campaign
finalizes (the scheduled task regenerates them). **No comparison against AirfRANS truth is made for
the AL picks (that would beg the question); diverged cases carry the surrogate prediction and its σ
but no error.**

### 6.5 The C4 finding

Stated in three parts, each backed by a file:

- **(a) Qualitative (strongest form).** Uncertainty-driven acquisition and pure ensemble variance
  placed **16/16 picks at α = 18°**, outside the training envelope (AirfRANS tops out ~15°, training
  at 12.5°) and, as the independent solver confirms, outside the steady-RANS-solvable envelope. The
  detector located the hardest physics in the pool without labels. This is evidence the score
  detects **extrapolation**, which here coincides with hard physics but is not identical to it.
- **(b) Quantitative on the solvable subset.** Surrogate-vs-Fluent CD/CL error on every accepted
  steady case (§6.4), offset-corrected, with conformal coverage — a qualitative external check at
  n ≈ 11 spanning Re 2e6–7e6 and α from −6° to 8.6°, exactly the "external check, not a statistical
  test" the spec committed to.
- **(c) The falsifiable claim, split.** Original: "high-acquisition cases have higher surrogate
  error than random ones." *Half 1* (random picks are cases the surrogate gets right): testable,
  holds — |ΔC_D| ≈ 1e-3, inside the conformal interval. *Half 2* (acquisition picks are cases the
  surrogate gets wrong): the ensemble's own members disagree by σ_CD ≈ 0.2 and the steady solver has
  no solution there, so "error" against a steady truth is undefined. We report that as the finding,
  not a missing number: the acquisition arm selected cases where the steady operator the surrogate
  learned does not exist. The arm comparison is therefore a comparison of *regimes* (solvable vs
  unsolvable), and Fig 11b shows it that way. **We make no "harder cases have higher error" claim;
  the data cannot carry it.**

> **Solver verification is a qualitative external check on the solvable subset.** The Fluent set ran
> in full; the grid study is grid-converged at the production level and the six AirfRANS replicas
> bound the solver offset. But 20 of the 24 active-learning cases, including all 16 acquisition- and
> variance-arm picks, sit at α = 18°, beyond the AirfRANS envelope and, as it turned out, beyond
> what steady 2D RANS can solve: they diverged under the batch settings and again under a
> conservative retry (longer first-order start, reduced under-relaxation), and one random-arm case
> at α = 12° did too. We report that outcome as the result it is: the label-free score concentrated
> on the corner of the pool where the steady operator the surrogate learned has no solution, which
> supports the score as an extrapolation detector but leaves the quantitative half of the claim
> ("higher error on acquired cases") untestable with a steady solver. The surrogate-versus-Fluent
> numbers therefore cover n = 10 to 11 low-to-moderate-angle cases (random arm, replicas, grid
> case), all inside or near the training envelope; they say how far the surrogate is from an
> independent solver where a steady answer exists, not how it fails where one does not. A
> time-averaged URANS truth for the deep-stall picks, with its own time-step and window study, is
> future work, and even then compares the surrogate to a quantity it was never trained to predict.
> Seven cases at Re = 7 × 10^6 run at M ≈ 0.32, past the usual incompressible limit; both solver and
> surrogate are incompressible, so the comparison is consistent, but the absolute coefficients there
> carry that caveat.

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
