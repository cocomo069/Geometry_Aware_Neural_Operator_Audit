# PROJECT OVERVIEW — the living master document

**This is the one file to read to understand the whole project.** It explains, in plain
language, what the project is, every concept behind it (so you can discuss the paper with
anyone), everything we have built and why, every problem we hit and how we solved it, the
results so far and what they mean, and exactly where we are right now.

It is maintained **alongside** the work — updated as things happen. Last updated: **2026-09-06**
(all research legs complete; results corrected after the round-1 final review).

> Companion docs (deeper detail lives here):
> - [CONTEXT.md](CONTEXT.md) — the frozen engineering interfaces (data formats, function signatures)
> - [DECISIONS.md](DECISIONS.md) — every engineering decision with rationale (D-001 … D-022)
> - [PLAN.md](PLAN.md) — the execution plan and phases
> - [PROGRESS.md](PROGRESS.md) — dated log of what each agent/session did
> - [DATA_NOTES.md](DATA_NOTES.md) — how the AirfRANS dataset actually works
> - [HANDOFF.md](HANDOFF.md) — quick "resume from cold" state
> - `05_geometry_aware_neural_operator_surrogate_uq.md` — the original research design doc

---

# PART 1 — WHAT THIS PROJECT IS (the one-paragraph version)

We are writing a **research paper** that asks a simple, practical question about AI models
that predict airflow around shapes (airfoils, cars): **can you actually trust them?** Modern
"neural surrogate" models predict the pressure on a wing surface in milliseconds instead of
hours of simulation. They score well on the usual benchmark (average error). But an engineer
can't use "average error" — they need to know: (1) do the model's own predictions even agree
with each other (does the predicted pressure field integrate to the drag the model reports)?
(2) does accuracy survive when you change the flow conditions or the shape family (does it
*generalize out of distribution*)? (3) when the model says "I'm uncertain," does that mean
anything (is the uncertainty *calibrated*)? Our paper builds a **reproducible protocol** to
measure all three, runs it on three different model families on a real aerodynamics dataset,
and reports where these models break. **It is a measurement/methodology paper, not a
"we-built-a-better-model" paper** — which is exactly why it's publishable even if the results
are "these models have problems."

**Author:** Taimoor Amin. **Target:** arXiv preprint, then a workshop/conference/journal.
**Why it matters for you:** it signals you can read ML papers critically, implement three
non-trivial architectures, reason about the physics (forces, quadrature, boundary
conditions), and run a rigorous evaluation — the exact profile computational-mechanics / SciML
grad programs and CFD-ML groups look for.

---

# PART 2 — THE CONCEPTS (so you can discuss the paper)

Read this part slowly — it's the vocabulary you need for a conversation about the paper.

## 2.1 The physical problem: external aerodynamics & RANS

Air flowing over a wing is governed by the **Navier–Stokes equations** (fluid motion). Solving
them exactly for turbulent flow is impossible at engineering scale, so CFD (Computational
Fluid Dynamics) uses **RANS** — Reynolds-Averaged Navier–Stokes — which solves for the *average*
flow and models the turbulence with a "turbulence model" (here, Spalart–Allmaras). A RANS
solution over one airfoil gives you the velocity, pressure, etc. at every point in the flow.

- **Airfoil**: the cross-section shape of a wing. **NACA airfoils** are a classic family
  described by a few digits (e.g. NACA 2412). **4-digit** and **5-digit** are two sub-families
  with different math for their shape — we use this split to test "shape-family generalization."
- **Reynolds number (Re)**: a dimensionless number = (speed × size) / (viscosity). It tells you
  how turbulent the flow is. Higher Re = more turbulent. Our data spans Re ≈ 2–6 million.
- **Angle of attack (AoA, α)**: the tilt of the airfoil into the flow. Too high → the flow
  **separates** (detaches from the surface) → **stall** (sudden loss of lift). This is a
  regime change, and models trained below stall fail above it — a key OOD finding.
- **Lift and drag**: the two force components on the airfoil. **Drag (C_D)** is along the
  flow (resistance); **Lift (C_L)** is perpendicular (what holds the plane up). These are the
  numbers engineers actually design around.

## 2.2 What a "neural surrogate" / "operator learning" is

A normal neural network maps a fixed vector to a fixed vector. But a flow field is a
*function* (pressure defined everywhere on the surface), and the geometry is also a function
(the shape). A **neural operator** is a network designed to map **functions to functions** —
so it can, in principle, work at any resolution (this property is called **discretization
invariance**). Formally it learns an operator 𝒮: (geometry, flow condition) → (surface fields).

- **FNO (Fourier Neural Operator)**: the workhorse operator. It does its "mixing" in **Fourier
  space** — it FFTs the field, keeps only the low-frequency modes, multiplies by learned
  weights, and inverse-FFTs. Efficient, but needs a **regular grid** (like an image). Airfoils
  are not on a grid, which is the problem the next model solves.
- **GINO (Geometry-Informed Neural Operator)**: FNO adapted to arbitrary geometry. It uses a
  small **graph network** to carry the surface points onto a regular background grid, runs an
  FNO on that grid, then carries the result back to the surface. It conditions on a **signed
  distance function (SDF)** — a field that says "how far is this grid point from the airfoil
  surface, and inside or outside." This is our **M2**.
- **Transolver**: a **transformer** for physics on meshes. Transformers normally cost O(N²)
  (every point attends to every other) — too expensive for 30,000 mesh points. Transolver's
  trick ("physics attention") softly groups the N points into a small number M of **slices**
  (learned clusters that share a physical state), does the expensive attention among just the
  M slices, then scatters back. Cost drops to O(N·M). This is our **M3**.
- **GNN (Graph Neural Network)**: the classic baseline. It builds a graph connecting each
  surface point to its k nearest neighbors and passes "messages" along edges a few times so
  each point aggregates information from its neighborhood. This is our **M1**.

**Why three?** The paper compares three *geometry-conditioning families* — graph (M1), grid+SDF
(M2), and transformer/slice (M3) — at matched size and training budget, so any difference is
about the *approach*, not one model being bigger.

## 2.3 The three things we measure (the paper's contributions)

### C1 — Force self-consistency (FSC)
The model predicts a surface pressure field. Physics says: if you **integrate** that pressure
(sum it up around the airfoil, weighted by area and direction), you get the drag and lift.
The model *also* has a separate output "head" that predicts drag/lift directly. **FSC = the gap
between these two.** If the field integrates to a different drag than the model reports, the
model is internally inconsistent — a red flag no standard benchmark catches. FSC needs **no
ground truth** (it's model-vs-itself), so you can compute it on brand-new shapes → it doubles as
a **label-free "this prediction is untrustworthy" detector**. *Our finding so far: FSC grows as
you push the model out of distribution — it works as a detector.*

### C2 — Out-of-distribution (OOD) generalization
We test each model on shifts away from the training distribution, one axis at a time:
- **in-distribution** (`full`): train and test from the same pool. The easy case.
- **scarce**: very little training data.
- **Reynolds shift** (`reynolds`): test on Re values outside the training band.
- **AoA shift** (`aoa`): test on angles outside training — crosses into stall/separation.
- **shape-family shift** (`shape5`): train on NACA 4-digit, test on 5-digit (new shapes).
- **combined**: shape *and* Reynolds shifted together — the hardest cell.

We plot **error vs shift** (Figure 4, the centerpiece). *Our finding: the OOD ordering is
consistent across all models, and combined is hardest — the models interpolate within a
regime but don't extrapolate across one.*

### C3 — Calibrated uncertainty
Two tools:
- **Deep ensembles**: train the *same* model K=5 times with different random seeds; where the 5
  disagree, the model is uncertain. The spread is a cheap uncertainty estimate — but it's a
  heuristic, not a guarantee.
- **Split conformal prediction**: a beautiful, distribution-free statistical method. Given a
  held-out **calibration set**, it converts any uncertainty score into an interval with a
  *mathematical coverage guarantee*: "the true value falls in this interval 90% of the time" —
  provably, as long as calibration and test data are **exchangeable** (drawn from the same
  distribution). The punchline for our paper: **that guarantee holds in-distribution but breaks
  under shift**, and we measure exactly how much. Conformal fixes calibration, not robustness.

Plus two supporting pieces:
- **Data-efficiency**: how does accuracy scale as you shrink the training set (25 → 800 cases)?
  Does the ranking of models flip when data is scarce? (Fig 9)
- **Active learning + CFD verification**: use the model's own uncertainty to *pick* new shapes
  to simulate, run them in **Ansys Fluent** (a real CFD solver, independent of the training
  data's solver), and check whether the "uncertain" cases really were harder. This is the piece
  that needs your CPU/Fluent and is deferred.

## 2.4 Key technical terms you'll be asked about

- **Signed Distance Function (SDF)**: field = distance to the surface, negative inside the body.
  How M2 "sees" the geometry.
- **Quadrature**: numerically integrating a quantity over a surface by summing (value × little
  area element). Our force integration is quadrature on the airfoil contour.
- **Relative L2 error**: the standard field-accuracy metric = ‖prediction − truth‖ / ‖truth‖.
  Lower is better; 0.02 means "2% off on average."
- **Spearman rank correlation (ρ)**: measures whether the model ranks designs in the right
  *order* (even if absolute values are off). ρ=1 is perfect ordering. For design, ranking
  matters more than absolute error — this is why we report it.
- **Normalization**: rescaling inputs/outputs to mean 0, std 1 so training is stable. Subtlety
  we had to get right (see Problem 8): the rescaling constants must come **only** from training
  data, never test data.
- **Equivariance / symmetry residual**: a symmetric airfoil at +α and −α should give mirror-image
  flows. We measure how badly a model violates this (it's a free correctness check). Plain GNNs
  violate it a lot (they're not built to be symmetric) — an expected finding.

---

# PART 3 — THE ARCHITECTURE OF WHAT WE BUILT

A clean, tested Python research codebase. Everything is reproducible from a seed.

```
src/
  data/       — load the AirfRANS dataset, build the train/test splits, normalize
  geometry/   — SDF, surface normals, quadrature weights, curvature, symmetry ops
  physics/    — FORCE INTEGRATION (the paper-critical bit), flow residuals
  models/     — M1 GNN, M2 SDF-FNO (GINO-style), M3 Transolver, shared coef head, losses
  eval/       — metrics, the evaluation "harness", results schema, baselines (ridge/constant)
  uq/         — deep ensembles, split conformal, coverage/reliability
  active/     — NACA shape generator, uncertainty acquisition, diversity selection
  viz/        — plotting style + data loading for figures
scripts/      — build_cache, train, evaluate, sweep, make_figures, make_tables
configs/      — one YAML per model + sweep specs
kaggle/       — the cloud-training system (see Part 5)
fluent/       — Ansys Fluent CFD verification pipeline (ready, deferred)
paper/        — LaTeX skeleton + generated figures/tables
tests/        — 449 tests, all pass on CPU
docs/         — this file + the companions listed at the top
```

**How the pieces connect:** `build_cache.py` turns the raw 10 GB dataset into fast per-airfoil
`.npz` files. `train.py` trains one model on one split (checkpoints every epoch, resumes
cleanly). `sweep.py` runs many train jobs in sequence, skipping finished ones. The `eval`
harness scores every run into a fixed-format `metrics.json`. `make_figures.py`/`make_tables.py`
turn those JSONs into the paper's figures and tables. On the cloud, the `kaggle/` system does
all this on a free GPU and chains sessions together.

---

# PART 4 — THE STORY, START TO FINISH (what we did, in order)

**Phase 0 — Foundation.** Set up the repo, picked the software stack, wrote the frozen
interface contracts so parallel work wouldn't collide. Downloaded the AirfRANS dataset
(10 GB) and built the per-airfoil cache (1000 simulations).

**Phase 1 — Parallel implementation.** Six AI sub-agents built the six subsystems in parallel
(data, physics/geometry, models, training infra, uncertainty, and the Fluent/paper scaffolds),
each against the frozen interfaces. Then a seventh built the figures/tables pipeline. This is
why the codebase came together fast.

**Gate G1 — the most important check.** Before trusting anything, we verified our force
integration reproduces the dataset's *own* published drag/lift when fed the true pressure
fields. Result: **median error 0.03%** across all 1000 airfoils. This means our "physics ruler"
is correct — every FSC number downstream is trustworthy. (If this had failed, nothing else
would matter.)

**Baselines.** We ran two dumb baselines — "always predict the average" and "ridge regression
from the shape numbers." **Surprise finding:** ridge ranks drag at ρ=0.83–0.90 even out of
distribution. This is the "shockingly strong baseline" the field under-reports — and the bar our
real models must clear (they do: 0.945–0.999).

**The compute pivot.** The local laptop GPU (see Problem 1) is too small/slow for the full
study, so we built a system to run everything on **Kaggle's free T4 GPUs**, chaining 12-hour
sessions together. This took nine debugging iterations (see Problems 2–7) but now works.

**Double-QA pass.** Before spending GPU-hours on results people will trust, three independent
reviewer agents audited the code for correctness. **The most important result: zero data
leakage** — train/test are provably disjoint on all six splits (verified with numbers). They
also caught two real bugs we fixed (Problems 8 & 9).

**The core grid.** All 18 runs (3 models × 6 splits) trained to completion on Kaggle. **These
are the paper's Table 1, Table 2, and Figure 4.** Results in Part 6.

**Status (2026-09-06): all complete.** Deep ensembles (K=5 Transolver/SDF-FNO, K=3 GNN), conformal
calibration, data-efficiency, ablations, and the full Fluent C4 verification are all done. See Part 6
/ RESULTS.md for the numbers.

---

# PART 5 — HOW THE CLOUD TRAINING WORKS (Kaggle system)

Because the laptop GPU can't do the heavy lifting, we run training on Kaggle's free tier:

1. **Two Kaggle datasets** hold the inputs: `airfrans-cache` (the 867 MB processed data) and
   `geo-op-code` (a snapshot of the code). A third, `geo-op-runs`, accumulates results so
   sessions can resume each other.
2. `kaggle/launch.py` pushes a "kernel" (a script job) to Kaggle that requests a **T4 GPU**,
   attaches the datasets, and runs our sweep.
3. Each session runs for up to ~11 hours, training as many models as fit, **checkpointing every
   epoch**. When it stops, it exports results + checkpoints.
4. `kaggle/pull_results.py` downloads them, merges into the repo, and re-publishes `geo-op-runs`.
5. Re-launch with `--runs-dataset geo-op-runs` and it **resumes** — finished runs are skipped,
   half-done ones continue from their last checkpoint.

This is what lets a 50-GPU-hour study run on a free tier that only gives ~30 hours/week: it just
spans multiple sessions automatically. **Your CPU is never used for training** — it stays free
for your Fluent work.

---

# PART 6 — RESULTS SO FAR (and what they mean)

**The complete core grid (3 models × 6 splits, 400 epochs each).**

**Field accuracy** — relative L2 error on surface pressure (lower = better):

| model | full | scarce | reynolds | aoa | shape5 | combined |
|---|---|---|---|---|---|---|
| **Transolver (M3)** | **0.017** | **0.067** | **0.077** | **0.109** | **0.055** | **0.139** |
| **SDF-FNO (M2)** | 0.026 | 0.083 | 0.098 | 0.111 | 0.064 | 0.153 |
| **GNN (M1)** | 0.074 | 0.154 | 0.223 | 0.188 | 0.130 | 0.270 |

**Drag ranking** — Spearman ρ (higher = better; ridge baseline ≈ 0.83–0.90):

| model | full | scarce | reynolds | aoa | shape5 | combined |
|---|---|---|---|---|---|---|
| Transolver | 0.999 | 0.996 | 0.976 | 0.947 | 0.983 | 0.972 |
| SDF-FNO | 0.999 | 0.992 | 0.982 | 0.955 | 0.984 | 0.968 |
| GNN | 0.999 | 0.996 | 0.977 | 0.945 | 0.983 | 0.969 |

**Force self-consistency (FSC)** — drag gap, lower = more self-consistent:

| model | full | scarce | reynolds | aoa | shape5 | combined |
|---|---|---|---|---|---|---|
| Transolver | 0.001 | 0.004 | 0.006 | 0.004 | 0.002 | 0.009 |
| SDF-FNO | 0.002 | 0.005 | 0.005 | 0.003 | 0.002 | 0.009 |
| GNN | 0.005 | 0.009 | 0.016 | 0.010 | 0.009 | 0.018 |

**What these mean — the three things to say in a discussion:**

1. **On field accuracy, the architecture matters and the ordering is clean:** Transformer
   (Transolver) beats the operator (SDF-FNO) beats the graph net (GNN), everywhere. This matches
   the operator-learning literature.

2. **On drag ranking — the metric engineers actually care about — the architecture barely
   matters.** All three sit at ρ ≈ 0.95–0.999, and the spread *between splits* is bigger than the
   spread *between models*. This is the paper's headline: **the field's model-comparison
   exercises are measuring noise relative to the deployment-relevant axis (distribution shift).**

3. **Force self-consistency grows with shift for every model** (smallest in-distribution, largest
   on `combined`). So FSC is a working **label-free OOD detector** — you can flag an untrustworthy
   prediction without knowing the true answer. That's contribution C1, and it's visible across all
   three architectures, not a fluke of one.

Also: the OOD hierarchy (`full` easiest → `combined` hardest) is consistent and physical —
`reynolds` and `aoa` (crossing into new flow regimes) hurt more than `shape5` (new shapes, same
regime). This supports the story *"these models interpolate within a flow regime; they don't
extrapolate across one."*

---

# PART 7 — PROBLEMS WE FACED AND HOW WE SOLVED THEM

The honest engineering story (useful when someone asks "what was hard?").

1. **The laptop GPU is tiny and old.** A Quadro P2000 (4 GB, Pascal architecture). Modern PyTorch
   dropped support for Pascal, so we had to pin an older CUDA build (cu126). It's fine for quick
   checks but too small/slow for the real study → forced the Kaggle pivot. (Decisions D-001, D-004)

2. **No compiled dependencies allowed.** The usual geometry-ML libraries (PyTorch Geometric,
   torch-scatter) are a nightmare to install on Windows + old GPU. We **re-implemented** the graph
   ops, the FNO, and the Transolver attention ourselves in ~pure PyTorch. Upside: full control and
   we understand every line. (D-002)

3. **Kaggle gives P100 GPUs by default — which don't work.** The P100 is *also* Pascal, and
   Kaggle's stock PyTorch dropped Pascal, so it failed the same way as the laptop. Fix: explicitly
   request the **T4** GPU (newer architecture). (D-020)

4. **Getting code onto Kaggle without leaking a token.** First we tried a private-repo clone
   (needs a GitHub token in the cloud), then a repo snapshot inside the data. Landed on shipping the
   code as its own small Kaggle dataset — Kaggle auto-unpacks it, no token needed. (D-018)

5. **Kaggle mounts datasets at an unpredictable path.** The cache didn't appear where we expected.
   Fix: the cloud job **searches** for the data by looking for the manifest file, instead of assuming
   a path.

6. **Kaggle hides the error log when a job fails.** We were flying blind on failures. Fix: the job
   now **tees its own log** into an always-exported file, so we can always see what went wrong. This
   is how we debugged the remaining issues.

7. **Cross-session resume was silently broken (a BLOCKER, caught in QA).** The "resume from last
   session" path used the same wrong mount-path assumption as #5. If we hadn't caught it, every new
   session would have **retrained everything from scratch**, burning the entire weekly GPU quota. Fix:
   locate the previous results by name+content search, with diagnostics. Verified working when
   session 2 resumed a half-trained model from epoch 303 in 10 minutes.

8. **Test statistics were leaking into normalization (MAJOR, caught in QA).** We normalized every
   split using constants computed from one global training set — but for the OOD splits, most of those
   splits' *test* airfoils were inside that global set, so their statistics (especially Reynolds/AoA)
   tainted the normalization. No labels leaked and model *comparisons* were unaffected, but for an
   OOD paper this is an easy reviewer criticism. Fix: compute normalization **per split, from that
   split's own training data only**. (D-021)

9. **A reflection bug in the symmetry check (minor, caught in QA).** When mirroring an airfoil to
   test symmetry, we forgot to also mirror the cached background-grid SDF, so M2's symmetry number
   used stale geometry. Impact was tiny (~1e-5) but it was a real logic error in a paper-critical
   diagnostic. Fixed by recomputing the grid from the mirrored shape. (D-022)

**The reassuring result of all this scrutiny:** the physics is correct (G1: 0.03% force error),
there is **no data leakage** on any split (verified with numbers), and the uncertainty math
(conformal coverage) is empirically calibrated. The results we're reporting are trustworthy.

---

# PART 8 — WHERE WE ARE RIGHT NOW & WHAT'S LEFT

**Done (all research legs complete, 2026-09-06):**
- ✅ Full pipeline built and tested (449 tests pass).
- ✅ Force integration validated (Gate G1: 0.03% error).
- ✅ Baselines on all 6 splits.
- ✅ **Core grid complete: 18 runs (3 models × 6 splits).** Table 1, Table 2, Figure 4 done.
- ✅ Three-way QA audit passed; all findings fixed.
- ✅ Cloud training system working and self-chaining.
- ✅ **Deep ensembles** — K=5 Transolver/SDF-FNO, K=3 GNN (the kNN GNN is expensive, D-019).
- ✅ **Conformal calibration** — Figures 6, 7, 8 + Table 3, all three models.
- ✅ **Data-efficiency curves** — 25/50/100/200/400 training airfoils → Figure 9.
- ✅ **Ablations** — SDF vs mask conditioning + physics-loss on/off (Table 4; 2 of 3 axes, see note).
- ✅ **Active learning + Fluent verification** — grid study, offset replicas, AL campaign, r1 retry,
  surrogate-vs-Fluent comparison → Table 5, Figure 11.

**Not done (out of scope / deprioritized, stated honestly):**
- ⏳ **The paper write-up** — deliberately left to coco; `paper/main.tex` is a pre-final 2026-09-03
  snapshot and does NOT reflect the final results (it still says GNN K=1, Fluent not run, etc.). The
  numbers of record live in RESULTS.md and `paper/tables|figures/`, not the .tex.
- ⏳ **DrivAerNet++ (3D cars)** — optional stretch, not attempted; needs a Globus login.
- ⏳ **Model cards (spec C5)** — only the template exists; flagged as an archiving to-do.

**What needs you:**
- (Optional) Globus login if you want the 3D car dataset leg.

---

# PART 9 — HOW TO CHECK ON THINGS YOURSELF

```bash
# See the latest results as a table
.venv/Scripts/python.exe -m scripts.make_tables --outdir paper/tables && cat paper/tables/tab2_ood.md

# Regenerate all figures from current results
.venv/Scripts/python.exe -m scripts.make_figures --outdir paper/figures

# Run the test suite (should say 449 passed)
.venv/Scripts/python.exe -m pytest tests/ -q

# Check the cloud training status
.venv/Scripts/python.exe -m kaggle kernels status cocomo069/geo-op-session
```

The figures live in `paper/figures/*.png`. The decision history is in `docs/DECISIONS.md`. The
blow-by-blow log is in `docs/PROGRESS.md`. **This file (OVERVIEW.md) is the summary of all of it.**

---

*Maintained by the project's AI pair (Fable planning + Opus execution). If you're reading this
cold, start at Part 1 and read straight through — it's written to be read start to finish.*
