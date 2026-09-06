# RESULTS — the living results document

**Every result the project has produced, with plain-language interpretation.** Read
[OVERVIEW.md](OVERVIEW.md) first for the concepts; this file is the numbers and what they mean.
Updated as runs land.

Status legend: ✅ complete · 🔄 partial · ⏳ not started. Last updated: **2026-09-06** (GNN K=3 ensemble complete; tables/§6 corrected and re-derived from files after the round-1 final review).

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

**Coefficient head (`cd_head`, the directly-supervised drag output):**

| model | full | scarce | reynolds | aoa | shape5 | combined |
|---|---|---|---|---|---|---|
| Transolver | 0.999 | 0.996 | 0.976 | 0.947 | 0.983 | 0.972 |
| SDF-FNO | 0.999 | 0.992 | 0.982 | 0.955 | 0.984 | 0.968 |
| GNN | 0.999 | 0.996 | 0.977 | 0.945 | 0.983 | 0.970 |

**Meaning — this is the paper's headline.** On the metric engineers actually use (ranking designs
by drag), **all three architectures are nearly identical** (ρ within ~0.01 of each other), and the
spread *between splits* (0.999 → 0.945) is larger than the spread *between models*. So the field's
usual "which architecture wins" comparison is measuring noise relative to the axis that matters
for deployment (distribution shift). On the head, all three still beat ridge — but not by as much as
their 4× field-accuracy gap would suggest.

**Physics-integrated route (`cd_int`, drag from integrating the predicted field) — the honest
counterpoint:** ranking drag by integrating the *predicted surface field* is a completely different,
and much weaker, story:

| model | full ρ (int) | combined ρ (int) |
|---|---|---|
| Transolver | 0.89 | 0.28 |
| SDF-FNO | 0.83 | −0.02 |
| GNN | 0.36 | −0.28 |

So integrating the fields ranks drag **no better than the 29-parameter ridge baseline** for the best
architecture in-distribution (0.89 vs ridge ~0.90) and **far worse for the GNN (0.36)**; under the
`combined` shift the integrated ranking collapses to ≈ 0 or negative for all three. The head and the
field **disagree on drag by tens of percent** (relative FSC below), which is the spec's Gap 1 landing
hard: the models learn a good drag *regressor* but not a field whose integral *is* that drag. Any
"beats ridge" statement in the paper must be qualified as **head-route only**; the field-consistent
drag does not beat ridge. `cd_int` also carries a synthetic-pool representation artifact off the real
mesh (D-026), but the numbers above are on the real AirfRANS test mesh and are trustworthy.

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

**In relative terms this gap is large, not cosmetic.** The absolute FSC above is small only because
C_D itself is ~0.01; as a *fraction* of the true drag (median `fsc_rel_cd`) the head and the
integrated field disagree by **≈ 5 % (Transolver), 7 % (SDF-FNO) and 24 % (GNN) in-distribution**,
rising to **≈ 37 %, 51 % and 115 % on `combined`**. That is why the integrated drag ranking in §2.2
is so much weaker than the head ranking: the two "drags" a single model reports are tens of percent
apart, and further apart the more it extrapolates.

### 2.4 Symmetry residual — equivariance violation (lower = more symmetric)

Range ~4.4 (in-distribution) to ~8.2 (aoa) across all models. **Meaning:** none of the three
architectures is built to be symmetric, so all violate the +α/−α mirror symmetry substantially,
and the violation grows with shift. This is the expected "plain models rely on coordinate
accidents, not equivariance" finding — it motivates the augmentation ablation.

**Read this as a response-to-reflection indicator, not a pure equivariance number.** The harness
reflects *every* test geometry about the chord line and keeps per-point index correspondence. For a
*cambered* airfoil the reflection is an inverted-camber shape that lies outside the NACA training
family, so the residual for those samples mixes true equivariance violation with ordinary
out-of-family extrapolation, and the large absolute magnitudes (440–820 %) partly reflect the latter.
The spec's clean form of the test is Rg = g — symmetric (zero-camber) sections at α = 0, where the
reflected input is the *same* input; a zero-camber-at-α0 subset is the honest equivariance number and
is logged as a quick follow-up. (We did verify one mechanical worry raised in review: the cached
signed Menger curvature passed through `reflect_x_batch` is numerically identical to recomputing it on
the mirrored contour — `orient` from the signed area flips with the cross product and cancels — so the
reflected batch is fed the correct curvature; that part is not a bug.)

### 2.5 The OOD hierarchy (a cross-cutting finding)

For every model, error grows in roughly this order: `full` < `shape5` < `scarce` < `aoa` ≈
`reynolds` < `combined`. The flow-regime shifts (`reynolds`, `aoa` — new Reynolds numbers,
crossing toward stall) hurt more than the shape-family shift (`shape5` — new airfoil shapes, same
flow regime). **Meaning:** *these models interpolate within a flow regime; they do not extrapolate
across one.* `combined` (shape + Reynolds shifted together) is hardest, as designed.

---

## 3. Calibration / uncertainty — deep ensembles + conformal ✅

**Deep ensembles complete for all three models on {full, reynolds, aoa, shape5}: K=5 for
Transolver and SDF-FNO, K=3 for the GNN (the kNN GNN is ~0.7–3 h/run, so it runs a smaller
ensemble by design, D-019).** Then split conformal prediction on each split's calibration set.
Two modes: *matched* (calibrate on the split's own held-out cal set, test on that split) and
*transfer* (calibrate on `full`'s cal set, deploy on the shifted split). **Important wording caveat:**
the cal set is carved from the *training* pool of each split (`splits.py`), which is drawn from the
same distribution as that split's test set only in the sense that both are in-distribution to the
split. So "matched" is *not* calibration on a handful of labelled shifted deployment cases; it is the
optimistic case where a like-distributed cal set is available. The practically relevant experiment
(recalibrate on a few labelled cases from the shifted deployment) is *not* run here and is flagged as
follow-up.

### 3.1 Coverage at nominal 90% — does the interval contain the truth 90% of the time?

Coefficient-C_D coverage, matched calibration, largest available K. **Table 3 reports the
*normalized*-score band** (variance-scaled), so read coverage together with the band width:

| model | full | reynolds | aoa | shape5 |
|---|---|---|---|---|
| SDF-FNO (K=5) | 0.95 | 0.90 | 0.94 | 0.76 |
| Transolver (K=5) | 0.91 | 0.94 | ~0.95 | 0.87 |
| GNN (K=3) | 0.82 | 0.97 | 0.89 | 0.86 |

(Field-level and 80%/95% nominal levels are in Table 3, now fully populated for all three models.)

**Meaning (contribution C3):**
- **In-distribution, the conformal guarantee is close but not automatic** — SDF-FNO/Transolver sit at
  0.91–0.95, while the **GNN under-covers in-distribution (0.82, Wilson CI [0.76, 0.87], excludes
  0.90)**. So "the guarantee holds in-distribution" is true for the two stronger models, not the GNN.
- **The GNN's high shifted coverage is a band-width artifact, not a real calibration win.** Its
  Reynolds "0.97" is the *normalized* band with **mean width 0.228 for a C_D of ~0.01** (≈ 23× the
  quantity) — a near-vacuous interval; the tight *absolute*-score band on the same split covers only
  **0.702**, essentially unchanged from K=1. So going K=1→K=3 did not genuinely fix the GNN under
  shift; it inflated the variance-scaled interval. We report both scores rather than quote the
  flattering one.
- **ECE still jumps 6–10×** on the flow-regime shifts (reynolds, aoa) for all models, and
  shape-family and combined shifts break the guarantee outright.
- **The takeaway sentence:** *conformal prediction repairs in-distribution calibration (for a
  well-behaved model, with a like-distributed cal set) but its coverage guarantee is not robust to
  distribution shift — a conformal interval must not be read as protection against design-space
  extrapolation, and a high coverage number must be read next to its interval width.* This is exactly
  the honest, useful message the paper is built to deliver.

### 3.2 Interval width (Fig 8)

Intervals widen from in-distribution to OOD (the model "knows" it's less sure), but not enough to
restore coverage — i.e. the widening under-compensates for the accuracy loss. This is why coverage
still drops despite wider intervals.

---

## 4. Data efficiency — Fig 9 ✅

Complete: all three models retrained at training-set sizes {25, 50, 100, 200, 400} × 3 seeds
(45 runs), plus the full-700 point reused from the core grid — a proper log-log error-vs-size curve
with seed bands (Fig 9, `paper/figures/fig09_data_efficiency.png`). The field-accuracy ordering
(Transolver > SDF-FNO > GNN) **holds at every training size** down to 25 airfoils — the transformer
does not lose its edge in the scarce regime here (no curve crossing at these sizes), and all three
improve smoothly with more data. Per-size numbers are in `docs/RESULTS_AUTO.md` and the run metrics.

---

## 5. Ablations — Table 4 ✅ (2 of 3 axes; see note)

**M2 geometry conditioning (SDF vs mask vs SDF+normals), field p rel-L2:**

| conditioning | full | shape5 (OOD) |
|---|---|---|
| SDF (baseline) | 0.0257 | 0.0641 |
| binary mask | 0.0294 | 0.0728 |
| SDF + normals | 0.0309 | **0.0613** |

**Meaning:** the binary occupancy mask is the worst encoding on both splits; SDF is best
in-distribution; **SDF+normals gives the best out-of-distribution (shape5) accuracy**. This
reproduces the Communications Engineering benchmark's finding that SDF beats a mask, and adds that
surface normals help specifically under shape-family shift.

**M1 physics/force-consistency loss (λ_F on vs off):**

| GNN variant | full p rel-L2 | full FSC | combined p rel-L2 | combined FSC |
|---|---|---|---|---|
| baseline (λ_F=0) | 0.074 | 0.0053 | 0.270 | 0.0176 |
| + force loss (λ_F>0) | 0.104 | **0.0013** | **0.259** | **0.0059** |

**Meaning (a clean predicted negative result):** adding the force-consistency loss **worsens field
accuracy in-distribution** (0.074 → 0.104) while **sharply cutting FSC** (0.0053 → 0.0013, and 3×
lower on `combined`). So the physics term does exactly what the spec anticipated — it shrinks the
residual it directly penalizes without improving (indeed slightly hurting) the field prediction.
Useful to report: a force-consistency penalty buys self-consistency, not accuracy.

**Note:** the third planned axis (augmentation on/off for the symmetry residual) and larger λ sweeps
were cut for v1 per the lean-scope plan (PLAN_PHASE2). The ensemble-size ablation is implicit in the
K∈{1,3,5} conformal files.

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
(c) comparison below subtracts the mean offset Δ̄_CD(SA) = +0.00090 from the Fluent C_D
(`compare_fluent.py`). Note the replica scatter s_CD = 0.00034 is itself as large as the surrogate's
in-distribution C_D MAE (~0.00012–0.00014): it is reported as a solver-side systematic uncertainty,
but it is *not* folded into the coverage test below (that test uses each surrogate's own conformal
half-width). So a fraction of the surrogate-vs-Fluent gap in §6.4 is solver scatter, not surrogate
error — see the caveat there.

### 6.3 Active-learning campaign outcome (Table 5a) ✅

The 24 AL cases + the grid trio + the 6 replicas ran under SA. Outcome (`fluent_summary.csv`,
regenerated by `collect_fluent.py`; 86 rows = 66 first-attempt s0 + 20 r1):

- **Grid study 3/3 converged; 4 low-α random cases converged** (α ∈ {−6, 0, 0, 6}); **all 6
  offset replicas converged (5) or quasi-steady (1)**; the α = 12° random case reached quasi-steady
  on retry. y+max < 1 on every accepted point. Twelve of these enter the surrogate comparison (§6.4).
- **20 cases diverged on the first attempt**: all 16 acquisition- and variance-arm picks (α = 18°),
  the 3 random α = 18° picks, and the one random α = 12° case. Every divergence set in within
  ~50–270 iterations of the first-order→second-order switch — a numerical signature superimposed on
  post-stall physics. The α = 12° case sits *inside* the AirfRANS envelope, so at least one
  divergence is a settings failure, not only physics — which is why the conservative `r1` retry
  (longer first-order start, reduced under-relaxation on all equations, no cd-steady stop) is run
  before calling any case "unsolvable."
- **`r1` retry COMPLETE, and it splits the diverged set cleanly** (α = 12° control-gated; detached
  batch). The one *in-envelope* case — the α = 12° random pick — recovered to a clean **quasi-steady**
  point (C_D = 0.019), confirming its first-attempt divergence was a settings failure. The **19
  out-of-envelope α = 18° picks did not settle**: they now run the full 7000 iterations at bounded,
  physical post-stall *magnitude* (final C_D ≈ 0.09–0.12, C_L ≈ 0.3–2.3), but every one passes
  through a transient excursion above the |C_D| > 10 divergence threshold (peak |C_D| ≈ 10–12) during
  the first-order→second-order transition and keeps **drifting 3–7 % over the last 500 iterations**,
  so `collect_fluent` classifies all 19 `diverged` (one α = 18° case, naca4415_re7e6, errored at
  iteration 725). Net: r1 fixed the settings-limited α = 12° case but **confirmed the α = 18° picks
  sit beyond the steady-RANS-solvable envelope** — bounded but non-settling, not clean steady points.
  (This corrects an earlier draft that called the α = 18° r1 results "quasi-steady"; the histories and
  the collector's drift/excursion rules say otherwise.)

### 6.4 Surrogate vs Fluent on the accepted set (Table 5c, Fig 11b) ✅

For every accepted steady case the trained ensembles were run at that geometry+condition (the exact
run_active inference path, fed the Fluent freestream) and compared to the offset-corrected Fluent
coefficient, beside each model's 90% conformal interval from the calibration study
(`results/uq/<model>_full_k5.json` for Transolver/SDF-FNO, `gnn_full_k3.json` for the GNN). The
reported surrogate CD is the **coefficient head** (`cd_head`): the integrated `cd_int` on a synthetic
pool geometry carries a large input-representation artifact (~5× vs the real mesh on a checked
in-envelope shape, D-026), so it is kept only as an FSC/physics diagnostic and is reliable only on
the replicas' real mesh.

**Final numbers (n = 12 accepted cases per model, offset-corrected `cd_head`;
`surrogate_vs_fluent_summary.json`):**

| model | MAE C_D (Fluent) | in-dist MAE | ratio | conformal cov@90% |
|---|---|---|---|---|
| GNN | 0.00037 | 0.00012 | 3.0× | 0.83 |
| Transolver | 0.00120 | 0.00012 | 10× | 0.42 |
| SDF-FNO | 0.00126 | 0.00014 | 9× | 0.33 |

**Meaning (C4 verdict):** on the independently-simulated Fluent cases the surrogate C_D error is
**3–10× its in-distribution error**, and the 90% conformal intervals **under-cover** for two of the
three models (Transolver 0.42, SDF-FNO 0.33; the GNN's 0.83 comes with its wider band, cf. §3.1). So
the accepted cases are genuinely harder for the surrogate, verified against a solver that did not
generate the training data, and the calibration degradation seen in C3 reproduces against external
truth. **Caveat (solver scatter):** the replica offset scatter s_CD = 0.00034 (§6.2) is as large as
the in-distribution MAE, and it is not folded into the coverage test, so part of the 3–10× gap is
solver-side systematic rather than surrogate error; the ratio is an upper bound on the true surrogate
degradation. Per-case rows: `results/fluent/surrogate_vs_fluent{,_summary}.{csv,json}`, Table 5c,
Fig 11b. **No comparison against AirfRANS truth is made for the AL picks (that would beg the
question).**

### 6.5 The C4 finding

Stated in three parts, each backed by a file:

- **(a) Qualitative (strongest form).** Uncertainty-driven acquisition and pure ensemble variance
  placed **16/16 picks at α = 18°**, outside the training envelope (AirfRANS tops out ~15°, training
  at 12.5°) and, as the independent solver confirms, outside the steady-RANS-solvable envelope. The
  detector located the hardest physics in the pool without labels. This is evidence the score
  detects **extrapolation**, which here coincides with hard physics but is not identical to it.
  **Disclosure (score construction).** The acquisition score is σ and FSC of the *integrated* C_D on
  the synthetic pool contour, and that contour carries the ~5× `cd_int` representation artifact of
  D-026 (pool σ_CD ≈ 0.15–0.24, FSC ≈ 0.08–0.31 for C_D ≈ 0.03, versus ~0.001 on the real mesh in
  distribution). Independently, the pool grid overshoots the training envelope on *both* axes
  (Re up to 7e6, α up to 18°), so the extreme α = 18° corner is nearly guaranteed to top the ranking.
  The "16/16 at α = 18°" outcome is therefore confounded by how the score is built and how the pool
  is gridded; it should be read as "the score flags the out-of-envelope corner," not as proof that a
  well-calibrated field-level uncertainty did so. The root cause of the `cd_int` pool artifact was not
  isolated (point count is not it, PROGRESS) and is logged as open (see Suggested-work note).
- **(b) Quantitative on the solvable subset.** Surrogate-vs-Fluent CD/CL error on the **n = 12**
  accepted cases (§6.4), offset-corrected, with conformal coverage — a qualitative external check
  spanning Re 2e6–7e6 and α from −6° to 8.6°, exactly the "external check, not a statistical test"
  the spec committed to.
- **(c) The falsifiable claim, split.** Original: "high-acquisition cases have higher surrogate
  error than random ones." *Half 1* (random picks are cases the surrogate gets right): testable,
  holds — |ΔC_D| ≈ 1e-3, inside the conformal interval. *Half 2* (acquisition picks are cases the
  surrogate gets wrong): no *clean* steady truth exists there (the α = 18° picks are bounded but
  non-settling under r1, §6.3), so a rigorous error is undefined. But the retry does give an
  *indicative* comparison the earlier draft discarded: the r1 histories settle to bounded final
  C_D ≈ 0.09–0.12 on the α = 18° picks, while the surrogate `cd_head` there is 0.031–0.041 (Table 5a)
  — a **~2.5–3.5× underprediction**, in the direction the claim predicts. We report it as indicative,
  not a coverage number, because the r1 values are drifting non-converged estimates, not fixed points.
  So Half 2 has *supporting evidence* (surrogate underpredicts where the steady operator breaks down),
  short of a formal test. Fig 11b shows the regime split (solvable vs non-settling).

> **Solver verification is a qualitative external check on the solvable subset.** The Fluent set ran
> in full; the grid study is grid-converged at the production level and the six AirfRANS replicas
> bound the solver offset. But 20 of the 24 active-learning cases, including all 16 acquisition- and
> variance-arm picks, sit at α = 18°, beyond the AirfRANS envelope and, as it turned out, beyond what
> steady 2D RANS can settle: they diverged under the batch settings, and under a conservative retry
> (longer first-order start, reduced under-relaxation) they stayed **bounded and physical in
> magnitude (final C_D ≈ 0.09–0.12) but did not settle** — each passed through a |C_D| > 10 transient
> excursion and kept drifting 3–7 % over the last 500 iterations, so the collector classifies them
> diverged. The one in-envelope α = 12° random case, by contrast, recovered to a clean quasi-steady
> point on retry, showing its first-attempt divergence was a settings failure. We report that outcome
> as the result it is: the label-free score concentrated on the corner of the pool where the steady
> operator the surrogate learned has no settled solution, which supports the score as an
> extrapolation detector (with the score-construction caveat above) but leaves the quantitative half
> of the claim testable only indirectly. The formal surrogate-versus-Fluent numbers therefore cover
> the **n = 12** accepted low-to-moderate-angle cases (random arm, replicas, grid case), all inside
> or near the training envelope; they say how far the surrogate is from an independent solver where a
> settled answer exists. A time-averaged URANS truth for the deep-stall picks, with its own time-step
> and window study, is future work, and even then compares the surrogate to a quantity it was never
> trained to predict. Seven cases at Re = 7 × 10^6 run at M ≈ 0.32, past the usual incompressible
> limit; both solver and surrogate are incompressible, so the comparison is consistent, but the
> absolute coefficients there carry that caveat.

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
