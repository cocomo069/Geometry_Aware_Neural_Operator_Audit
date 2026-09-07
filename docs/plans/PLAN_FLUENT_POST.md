# PLAN_FLUENT_POST.md: post-processing the Fluent campaign and salvaging C4

Planning document (Fable, 2026-09-03, per D-023). Executors: Opus agents for
code, batch driving and docs; the ParaView MCP (`mcp__paraview__*`) for field
figures; `fluent/run_batch.py` (direct `fluent.exe`) for every solve. Nothing in
this file has been executed.

Read first: `docs/plans/FLUENT_PLAN.md` (design + acceptance criteria), `fluent/README.md`
(operating manual), D-025 in `docs/DECISIONS.md`, `docs/RESULTS.md` section 6.

---

## 0. Verified state of the campaign (what the plan is built on)

Batch `fluent/run_batch.py --sst-gridstudy` finished 2026-09-03 15:57:
30 runs, 10 done, 20 diverged (`logs/fluent_batch.log`). Per-case tabulation from
`fluent/cases/*/*_coeffs.out`, `*_yplus_max.txt`, `*_mesh.json`:

| group | cases | outcome |
|---|---|---|
| gridstudy L1/L2/L3, SA + SST | 6 | all converged, 4300 iters, y+max 0.38 to 0.87, CD 0.01112 / 0.010755 / 0.010755 (SA), 0.01085 / 0.01060 / 0.01066 (SST). Deliverable #2 done. |
| `al_rand_*` low-AoA (a0, a0, a6, am6) | 4 | converged, 4300 iters, y+max 0.38 to 0.77, physical CD/CL |
| `al_rand_naca22012_re3e6_a12` | 1 | **diverged** at iter 667 (min orthogonality 0.201) |
| `al_rand_*_a18` | 3 | diverged at iters 377 to 411 |
| `al_acq_*_a18` | 8 | diverged at iters 350 to 568 (one at 2300) |
| `al_var_*_a18` | 8 | diverged at iters 388 to 473 |

Facts the decisions below rest on:

1. **Every divergence starts within ~50 to 270 iterations of the first-order to
   second-order switch at iteration 300** (stage 1 = 300 first-order iterations,
   stage 2 = second-order at URF p 0.3 / mom 0.7). One case survived to 2300. That
   timing signature is a numerical-robustness failure of the stage-2 settings on
   sheared cells, superimposed on the physics. It does not by itself prove the
   flow has no steady solution; it proves the *current* settings cannot find one.
2. The α = 12° random case (inside the AirfRANS envelope, where AirfRANS itself has
   converged simpleFoam solutions) also diverged. So at least one divergence is a
   setup failure, not physics. A reviewer will ask "did you lower the URFs / try
   pseudo-transient?" and the honest answer today is no.
3. Mesh quality at α = 18°: min orthogonality 0.134 to 0.193, i.e. below the 0.15
   acceptance floor of FLUENT_PLAN section 6 for 13 of the 19 cases (rotating the
   section 18° shears the wake-cut fold). All Jacobians positive.
4. Converged L2 cases cost 10.6 to 16.8 min each (6 cores). Diverging cases die
   in ~1.5 min. CPU is cheap here; the licence is the only serial resource.
5. The `cd-steady` stop condition never triggered (all converged cases ran the
   full 4000 stage-2 iterations); final CD is flat to 1e-6 relative anyway, so
   acceptance is unaffected, but the post-processor must apply the flatness
   criterion itself from `*_coeffs.out` rather than trusting a stop event.
6. Seven Re = 7e6 cases sit at M ≈ 0.32 (U = 109 m/s), past the incompressible
   comfort line; the one converged member (`al_rand_naca0010_re7e6_a6`) carries
   that caveat. The surrogate is incompressible-trained too, so the comparison is
   like-for-like; the caveat belongs in the limitation text, not a correction.
7. Ensemble spread on the acquisition picks is σ_CD = 0.16 to 0.24 (Table 5), i.e.
   the members disagree by more than the entire plausible post-stall CD. On the
   converged random cases σ_CD is ~0.004 to 0.01 (`acquisition_ranking.csv`).
   In-distribution yardstick for "surrogate error": Transolver `full` test
   CD_int MAE = 0.0011, CD_head MAE = 0.00012 (`results/transolver_full_s0/metrics.json`).
8. Pool freestream convention vs Fluent: pool `cond` uses ν = 1.56e-5 (NU_DATASET),
   Fluent uses μ/ρ = 1.5625e-5. At fixed Re the velocities differ by 0.2 %: negligible,
   but section 4 fixes the convention so nobody re-derives it.

---

## 1. Decision 1: how to reframe C4 (recommendation: (a) + (b) + (c), not thin *if* section 2 is done)

**Recommended framing.** C4 becomes a three-part statement, each part backed by a
file:

- **(a) Qualitative, strongest form.** Uncertainty-driven acquisition (and pure
  ensemble variance) placed 16/16 picks at α = 18°, outside the training envelope
  (AirfRANS tops out at ~15°, training at 12.5°) and, as the independent solver
  confirms, outside the steady-RANS-solvable envelope even after a robustness
  retry (section 2). The detector located the hardest physics in the pool without
  labels. This is evidence *for* the detector, and it is exactly what an
  epistemic-uncertainty score is supposed to do under extrapolation. The paper
  must say plainly that it is evidence the score detects *extrapolation*, which
  coincides with hard physics here but is not the same thing.
- **(b) Quantitative, on the solvable subset.** Surrogate-vs-Fluent CD and CL
  error on every case with an accepted steady solution: the 4 converged random-arm
  cases, the 6 offset-study replicas (section 3, which also carry AirfRANS truth,
  so they give a three-way AirfRANS / Fluent / surrogate comparison), the grid-study
  case (NACA 0012, Re 3e6, α 5°, treated as one more low-AoA point), plus any
  α = 12° / α = 18° case the retry in section 2 rescues. Each surrogate error is
  reported after subtracting the SA solver offset Δ̄ (section 3) with Δ's spread
  propagated, and beside the model's own conformal 90 % interval from
  `results/uq/<model>_full_k5.json` so the reader sees whether the independent
  truth falls inside the interval the calibration study promised.
- **(c) The falsifiable claim, split into its two halves and tested where it can
  be.** Original claim: "high-acquisition cases have higher surrogate error than
  random ones." Half 1 (random picks are cases the surrogate gets right): testable,
  n = 4 (+ rescued cases); expected |ΔCD| ≈ 1e-3, inside the conformal interval.
  Half 2 (acquisition picks are cases the surrogate gets wrong): the surrogate's
  own members disagree by σ_CD ≈ 0.2 and the steady solver has no solution, so
  "error" against a steady truth is undefined there. Report that as the finding,
  not as a missing number: the acquisition arm selected cases where the steady
  operator the surrogate learned does not exist. The comparison of arms is then
  a comparison of *regimes* (solvable vs unsolvable), and Fig 11(b) shows it that
  way (section 6).

**Is it too thin?** Honest weighing: (b) alone with n = 4 would be too thin and a
reviewer would say so. (a) alone is a story, not a test. Together with the offset
replicas (n = 6, every one carrying an independent OpenFOAM number too) the
quantitative part reaches n = 10 to 11 accepted steady points spanning Re 2e6 to
7e6 and α from -6° to 8.6°, which is exactly the "qualitative external check, not a
statistical test" the spec (section 5.8, risk "Overclaiming") already committed
to. What makes it publishable rather than thin is (i) the robustness retry in
section 2 so "unsolvable" is a documented outcome rather than a first-attempt
divergence, and (ii) reporting the acquisition arm's σ_CD next to the solver
outcome so the reader sees the score and the solver agreeing. Do not add a
"harder cases have higher error" sentence anywhere; the data cannot carry it.

Executor steps for Decision 1 are the union of sections 2 to 6; the framing text
itself lands in section 6.

---

## 2. Decision 2: transient URANS salvage (recommendation: NO URANS for v1; DO a cheap steady robustness retry; URANS is future work, optionally one demonstrator)

**Why not URANS now.**
- The quantity it would produce is ill-defined for this test. The surrogate learned
  a *steady* RANS operator from AirfRANS. At α = 18° that operator has no fixed
  point; a URANS time-average is a different object (different closure behaviour,
  2D-shedding artefacts that 3D flow does not have, strong sensitivity to time step
  and averaging window). |CD_surrogate − CD_URANS| would measure "distance to
  something the model was never asked to predict", and the paper could not
  interpret it either way.
- Cost/benefit: 3 to 4 cases × 2 to 6 h = up to a day of the single licence, plus
  a new template, new time-step and averaging-window studies (a reviewer will ask
  for both), plus the fidelity caveat above. The result would be 3 to 4 points that
  cannot change the conclusion of section 1.
- The one thing URANS would add, evidence that the α = 18° flow is physically
  unsteady, is cheaper to obtain as a single demonstrator (below).

**Do instead: steady robustness retry, variant `r1`, on all 20 diverged cases.**
Unattended, ~4 to 6 h licence time, no human babysitting, and it converts "diverged
on first attempt" into one of three documented outcomes per case:

| outcome | rule (from `*_coeffs.out`, stage-2 window = last 1000 iterations) | how it is reported |
|---|---|---|
| converged | finite, relative CD change ≤ 1e-5 over the last 150 iterations | accepted steady point; enters (b) |
| quasi-steady | finite and bounded, peak-to-peak(CD)/mean(CD) < 0.5 over the window, no monotone drift | mean ± half peak-to-peak, hollow marker in Fig 11(b), flagged "limit cycle of the steady iteration, not a time average"; enters (b) with that error bar |
| diverged | non-finite, or |CD| > 10 at any point | "no steady solution found under settings S0 and S1" |

Variant `r1` = `fluent/templates/case_template_r1.jou`, a copy of the current
template with only these edits (keep sections 11 to 16, including the baked
display objects and the EnSight export, unchanged):

```
; stage 1: longer first-order start, conservative URFs
/solve/set/under-relaxation/pressure 0.2
/solve/set/under-relaxation/mom 0.5
{{TURB_URF_BLOCK}}                     ; make_cases: nut/k/omega 0.5 in r1
/solve/iterate 1000                    ; was 300
; stage 2: second order, URFs kept low, longer budget, no stop condition
/solve/set/under-relaxation/pressure 0.2
/solve/set/under-relaxation/mom 0.4
/solve/iterate 6000
```

Output names get the suffix `_sa_r1` (`<case>_sa_r1.jou`, `_sa_r1_coeffs.out`,
`_sa_r1.cas.h5`, `_sa_r1.encas`, ...), so the original diverged transcripts stay
on disk as evidence. Executor steps:

1. `fluent/make_cases.py`: add `--variant r1` (template lookup
   `case_template_{variant}.jou`, journal/output suffix `_{model}_{variant}`, URF
   block per variant in `MODELS[...]`). Add `--only` filtering as today. Write
   `fluent/cases/RUNBOOK_r1.md`.
2. `fluent/run_batch.py`: add `--variant r1` (journal name + completeness check use
   the suffixed files; `--only` list accepted from a file, `--only-file`). Do not
   touch the skip logic for the original runs.
3. Render and launch, offset study first (section 3), then:
   `.venv/Scripts/python.exe fluent/run_batch.py --variant r1 --only-file fluent/diverged_cases.txt`
   where `fluent/diverged_cases.txt` lists the 20 case ids from section 0 (write it
   from `logs/fluent_batch.log` ERROR lines, do not hand-type).
4. Order inside the r1 batch: `al_rand_naca22012_re3e6_a12` first (the inside-envelope
   control: if r1 does not rescue it, stop the batch and escalate to `r2` below on
   that single case before spending licence time on the 18° set), then the 8
   acquisition, then 8 variance, then 3 random 18° cases.
5. Fallback `r2`, only if the α = 12° control still diverges under r1:
   pseudo-transient coupled (`/solve/set/p-v-coupling 24`, then the v211
   pseudo-time menu). The TUI keywords for pseudo-transient in v211 are NOT verified
   in this repo: run a Step-0 style probe on one case first (`fluent/templates/
   probe_bc_keywords.jou` pattern, transcript in `fluent/runs/`), reconcile, then
   run. Do not apply r2 to the 18° set unless it rescues the control.
6. Post: `scripts/collect_fluent.py` (section 4) classifies every run (S0 and r1)
   with the rules in the table above and writes `results/fluent/fluent_summary.csv`.

**Optional Tier 3, one URANS demonstrator (only if the user wants the appendix
figure; ~2 to 3 h licence, zero effect on the conclusions).** Case
`al_acq_naca0010_re7e6_a18` (acquisition rank 0), 2D URANS SA, dual-time, second
order implicit, Δt = c/(100 U) (≈ 9e-5 s), 20 sub-iterations, 40 convective times
after a 10-convective-time transient discarded; CL/CD time history exported. Its
only use is a figure showing a periodic CL/CD signal, i.e. proof that the deep-stall
picks are physically unsteady in 2D RANS, under the appendix heading "why steady
RANS could not be used". State the time-step and window as un-studied. If
skipped, the limitation text says "URANS time-averaged truth for the deep-stall
picks is future work" and nothing else is lost.

---

## 3. Decision 3: solver-offset study (deliverable #1)

**Frozen replica list** (drawn only from `data/splits/full.json` `test`, by the
stratification rules of FLUENT_PLAN section 5.2, with the convergence-friendly
adjustment that "high α" is capped at ~8.6° after the α = 12° divergence).
Commit this list before any run; do not re-draw after seeing results.

| # | rule | AirfRANS sim | Re | α (°) | shape (AirfRANS params) | AirfRANS CD / CL |
|---|---|---|---|---|---|---|
| 1 | near-symmetric 4-digit, low α, mid Re | `airFoil2D_SST_60.541_2.401_0.081_0.0_16.295` | 3.88e6 | 2.40 | M 0.081, P 0.0, T 16.30 | 0.00970 / 0.251 |
| 2 | near-symmetric 4-digit, higher α, mid Re | `airFoil2D_SST_62.607_8.578_0.875_0.0_8.812` | 4.01e6 | 8.58 | M 0.875, P 0.0, T 8.81 | 0.01151 / 0.926 |
| 3 | cambered 4-digit, mid α, mid Re | `airFoil2D_SST_71.226_3.333_2.424_2.411_10.928` | 4.57e6 | 3.33 | M 2.424, P 2.411, T 10.93 | 0.00905 / 0.607 |
| 4 | 4-digit, low-Re edge | `airFoil2D_SST_31.863_-1.58_1.567_6.956_10.028` | 2.04e6 | -1.58 | M 1.567, P 6.956, T 10.03 | 0.00923 / 0.051 |
| 5 | 4-digit, high-Re edge | `airFoil2D_SST_93.213_3.79_0.418_0.0_9.665` | 5.98e6 | 3.79 | M 0.418, P 0.0, T 9.67 | 0.00806 / 0.419 |
| 6 | 5-digit (non-reflex), mid conditions | `airFoil2D_SST_64.144_0.932_1.956_3.115_0.0_17.336` | 4.11e6 | 0.93 | L 1.956, P 3.115, Q 0, T 17.34 | 0.00981 / 0.219 |

Stretch replica (only if the r1 variant rescues the α = 12° control, otherwise
skip): `airFoil2D_SST_59.75_10.766_1.309_0.0_13.674` (Re 3.83e6, α 10.77), the
"where turbulence-model differences bite" cell FLUENT_PLAN wanted.

Case ids: `offset_1_naca0016_a2p4` ... `offset_6_naca5_17_a0p9` (short, filesystem-safe;
the sim name goes in the case `note` and `airfrans_sim` field). Mesh level 2, both
models (SA like-for-like, SST for the model-sensitivity spread): 12 runs, ~2.5 to 3 h.

**Geometry: use AirfRANS's own generator, not a reimplementation.** The replicas
have continuous NACA parameters (M = 0.081 is a legal "digit"); `mesh_gen.py`'s
integer-digit parser cannot represent them and pool.py's `NACA5` only takes
integer digits. `airfrans` 0.1.2 is installed in the venv and exposes
`airfrans.naca_generator.camber_line(params, x)` (dispatches on 2 vs 3 camber
params, handles reflex and off-table P) and `thickness_dist(t, x, CTE=True)`.
Executor steps:

1. `fluent/mesh_gen.py`: add `--naca-params "M,P,T"` (3 values, 4-digit) /
   `"L,P,Q,T"` (4 values, 5-digit), mutually exclusive with `--naca`. In
   `airfoil_loop`, when params are given, compute `yc, dyc` with
   `airfrans.naca_generator.camber_line(params[:-1], xc)` and `yt` with
   `thickness_dist(params[-1]/100, xc, CTE=True)` (check the exact `t` scaling
   against the package source before trusting it), then the same normal-offset,
   TE/LE pinning and rotation as today. Record `naca_params` in the mesh json.
2. **Validation gate before meshing anything:** for each of the 6 sims, generate
   the contour at 400 points and compare against the cached surface nodes
   `data/processed/airfrans/<sim>.npz["surf_pos"]` (994 points, unordered): max
   nearest-neighbour distance < 5e-4 c. If it fails, the parameter convention is
   wrong; fix before running Fluent. Save the check as
   `results/fluent/offset_geometry_check.json`.
3. `fluent/make_cases.py`: accept `naca_params` (list) as an alternative to
   `naca_digits`; per-case `mu` override so the freestream matches the sim
   exactly: set case `"mu": rho * nu_sim` with `nu_sim = npz["nu"]` (≈ 1.5498e-5)
   and `"re": u_inf_sim / nu_sim`, so that `u_inf = re*mu/rho` reproduces the sim's
   U (e.g. 60.541 m/s) to the last digit. ρ stays 1.184 (DATA_NOTES A1.2: ρ cancels
   in AirfRANS's coefficients, so only U and ν matter). Record `airfrans_sim`.
4. Append the 6 (7) cases to `fluent/cases_to_run.json` under a new `"offset"`
   block with `"models": ["sa", "sst"]`; render with `make_cases.py --write-mesh
   --only offset_*`; check `min_orthogonality ≥ 0.2` (all |α| < 9°, expect ~0.25).
5. Run first, before r1: `run_batch.py --models sa,sst --only-file fluent/offset_cases.txt`.
6. While the licence is busy, also queue SST on the 4 converged random cases
   (`run_batch.py --models sst --only al_rand_naca0010_re7e6_a6 ...`, 4 × ~13 min):
   it gives the SA↔SST spread on the very cases used in (b), which FLUENT_PLAN
   section 5.4 needs to decide whether the model spread exceeds |Δ̄|.

**Computing the offset** (`scripts/compare_fluent.py --offset`, section 4 owns the
script):

```
Δ_CD = CD_fluent − CD_airfrans     δ_CD = Δ_CD / CD_airfrans
Δ_CL = CL_fluent − CL_airfrans     δ_CL = Δ_CL / |CL_airfrans|
```
- `CD_airfrans`, `CL_airfrans` = `npz["cd_true"]`, `npz["cl_true"]` (already verified
  to 3e-4 relative by gate G1 against AirfRANS's own `force_coefficient`).
- `CD_fluent`, `CL_fluent` = last row of `<case>_<model>_coeffs.out` (the `cd`/`cl`
  report definitions with force vectors (1,0)/(0,1); valid because AoA is baked
  into the mesh so drag = Fx, lift = Fy). Cross-check against `_force_x.txt` /
  `_force_y.txt` totals (must agree to 1e-6).
- Report per model: the six values, mean ± sample std, and the SA↔SST spread per
  case. Surface distributions: Fluent `_surface.csv` (map columns BY HEADER:
  `x-coordinate`, `pressure-coefficient`, `skin-friction-coef`) vs AirfRANS
  `cp = surf_p / (0.5 U²)` (surf_p is kinematic, DATA_NOTES A1.2) and
  `cf = |surf_tau| / (0.5 U²)` on the same x/c axis. Note the Fluent surface is
  rotated by -α: un-rotate about (0.25, 0) before plotting x/c.
- Outputs: `results/fluent/offset.csv` (one row per sim × model), a figure
  `paper/figures/figB2_offset_cp.pdf` (6 panels cp, ours vs AirfRANS, SA and SST),
  and the numbers Δ̄_CD(SA), s_CD, Δ̄_CL(SA), s_CL for Table 5 and the text.
- Applied correction (FLUENT_PLAN 5.4): on every (b) case the surrogate is compared
  against `CD_fluent − Δ̄_CD(SA)` with uncertainty `s_CD` added in quadrature to the
  GCI band of the grid study (CD: 0.003 % L2→L3, effectively zero; use the L1→L2
  change 3.3 % as the conservative band, computed per Celik in
  `scripts/collect_fluent.py --gci`).

---

## 4. Decision 4: surrogate-vs-Fluent mechanics

Two new scripts, both CPU-light / GPU-light, reusing `scripts/run_active.py`,
`src/active/pool.py`, `src/uq/ensembles.py`, `src/physics/force_integration.py`.

### 4.1 `scripts/collect_fluent.py` (parser + classifier, pure numpy/pandas)

Reads every `fluent/cases/<case>/` and writes `results/fluent/fluent_summary.csv`
with columns: `case_id, arm (gridstudy|acquisition|variance|random|offset),
naca, re, aoa_deg, mesh_level, model, variant (s0|r1|r2), n_iter, status
(converged|quasi_steady|diverged|not_run), cd, cl, cd_window_mean,
cd_window_std, cd_p2p, cl_window_mean, cl_p2p, cd_rel_change_150, yplus_max,
yplus_avg, min_orthogonality, mach, wall_min, has_dat, has_encas`. Rules: section
2 table; `cd_rel_change_150` = |CD_last − CD_last−150| / |CD_last|. Also `--gci`:
Celik GCI on the gridstudy trio per model (FLUENT_PLAN 4.2), written to
`results/fluent/gci.json` (p_obs, φ_ext, GCI_fine for CD and CL, SA and SST,
oscillatory flag). Batch-integrity check from FLUENT_PLAN section 6: cell count
in the saved `.cas.h5` (read with h5py, dataset `meshes/1/cells/...`; if the
layout is awkward, fall back to the `/mesh/check` block in the `.trn`) must equal
`mesh.json["counts"]["n_cells"]`.

### 4.2 `scripts/compare_fluent.py` (surrogate predictions on the Fluent cases)

```
.venv/Scripts/python.exe -m scripts.compare_fluent --model transolver --split full --seeds 0,1,2,3,4
.venv/Scripts/python.exe -m scripts.compare_fluent --model sdf_fno    --split full --seeds 0,1,2,3,4
.venv/Scripts/python.exe -m scripts.compare_fluent --model gnn        --split full --seeds 0,1      # K=2, absolute score
.venv/Scripts/python.exe -m scripts.compare_fluent --offset             # section 3 arithmetic
```

Mechanics, reusing existing functions verbatim so no second inference path exists:

1. Load the ensemble exactly as `run_active.run_active` does: `_find_checkpoints`,
   `_load_config`, `load_ensemble_checkpoints`, `_norm_stats(split)` (per-split
   train-only stats, D-021), `denorm` / `normalize_cond` closures.
2. Build one `PoolEntry` per accepted Fluent case: `PoolEntry(parse_naca(code), re,
   aoa_deg)` from `results/fluent/fluent_summary.csv` rows with status in
   {converged, quasi_steady}. **Freestream convention:** pass
   `speed = params.json["u_inf"]` (the Fluent value) into `entry_to_batch(...,
   speed=...)` so the surrogate sees the exact velocity Fluent ran at; do not let
   the pool's ν default recompute it. For the offset replicas, the surrogate
   prediction on the AirfRANS geometry already exists in
   `results/<model>_full_s*/per_sim.csv` (verify columns: per-sim cd_int / cd_head
   / cl_int / cl_head predictions and truths); use those rows rather than
   regenerating the geometry, and record which path each row came from.
3. Score with `run_active.score_pool(predictor, entries, integrate=integrate_forces,
   denorm=..., normalize_cond=..., device=..., compute_sym=False, n_points=200)`.
   Extend `score_pool` to also return the per-member arrays (`cd_int_members
   (K,N)`, `cl_int_members`, `cd_head_members`, `cl_head_members`), behind a
   keyword `return_members=True` so `run_active` and its tests are untouched.
4. Per case and model, write `results/fluent/surrogate_vs_fluent.csv`:
   `case_id, arm, model, K, cd_fluent, cd_fluent_corrected (−Δ̄_CD), cl_fluent,
   cl_fluent_corrected, cd_int_mean, cd_int_std, cd_head_mean, cd_head_std,
   cl_int_mean, cl_int_std, cl_head_mean, cl_head_std, err_cd_int (= cd_int_mean −
   cd_fluent_corrected), err_cd_head, err_cl_int, err_cl_head, fsc (=|cd_int_mean −
   cd_head_mean|), sigma_cd_pool (from acquisition_ranking.csv, same key),
   q90_halfwidth_cd_int, covered90_cd_int, covered90_cd_head`.
   Interval half-widths come from `results/uq/<model>_full_k5.json["coef"]["cd_int"]["0.9"]`
   (matched calibration on `full`; the normalized score scales the half-width by
   the ensemble std, so rebuild the `CalibratedIntervals` object from that dict via
   `src.uq.conformal.CalibratedIntervals.from_dict` and call `contains(y, mu,
   sigma)`; check the stored dict shape first).
5. Summary block printed and saved to `results/fluent/surrogate_vs_fluent_summary.json`:
   per model, MAE and max |err| of CD and CL over the accepted set, split by arm
   (random / offset / gridstudy / rescued), coverage fraction at 90 %, and the
   in-distribution yardstick from `results/<model>_full_s0/metrics.json`
   (`coef.cd_int_mae`, `coef.cd_head_mae`) for the sentence "Fluent-truth error is
   within a factor F of the model's own held-out AirfRANS error".
6. Tests: `tests/test_compare_fluent.py` with the fake ensemble from
   `tests/test_run_active.py` and a synthetic `fluent_summary.csv` (two rows, one
   quasi_steady) so the CSV schema, the speed convention and the offset arithmetic
   are pinned. No dataset, no checkpoints, CPU-only, < 10 s.

### 4.3 What must NOT be done
- No comparison against AirfRANS truth for the AL picks (begs the question; the
  paper says so explicitly).
- No surrogate "error" for diverged cases; the CSV row exists with status diverged
  and the surrogate columns filled (so Fig 11(b) can show the prediction and σ),
  but `err_*` is NaN.

---

## 5. Decision 5: ParaView deliverables

Per the user's directive (D-025), field figures come from the EnSight exports via
`mcp__paraview__*`; the native Fluent display objects (`ct_pressure`, `ct_velocity`,
`ct_cp`, `vec_velocity`, report definitions `cd`, `cl`) are already baked into every
saved `.cas.h5` by journal section 16, which satisfies the "build it natively in
Fluent too" rule; the r1 template keeps that block, so no extra Fluent work.
Diverged cases have no `.dat.h5` / `.encas` and get no figure.

**Which cases (8 figures, all converged/accepted):**

| figure | case | purpose |
|---|---|---|
| B1 (a to c) | `gridstudy_naca0012_re3e6_a5_L1/L2/L3` SA, cp contour + surface cp overlay | grid-independence visual, appendix B |
| B3 (a to d) | the 4 converged random cases (`al_rand_naca0010_re7e6_a6`, `al_rand_naca2415_re6e6_a0`, `al_rand_naca33012_re6e6_am6`, `al_rand_naca4415_re2e6_a0`) SA | the (b) comparison cases: velocity magnitude with streamlines, cp legend |
| B4 | one offset replica, #2 (`airFoil2D_SST_62.607_...`, α 8.6°) SA vs SST side by side | turbulence-model sensitivity visual |
| B5 (optional) | any α = 12°/18° case rescued by r1 (quasi-steady) | separated-flow field, hollow-marker cases in Fig 11(b) |

**Pipeline per case (MCP calls, in order):**
1. `paraview_open_data({"path": "<abs>/fluent/cases/<case>/<case>_sa.encas", "name": "<case>"})`;
   read back the array names (expect `pressure`, `pressure-coefficient`,
   `velocity-magnitude`, `x-wall-shear`, `y-wall-shear`, `velocity`).
2. `paraview_set_coloring({"name": "<case>", "array": "pressure-coefficient",
   "association": "POINTS", "colormap": "Cool to Warm", "rescale": [-2.0, 1.0]})`
   (same fixed range on every case so panels are comparable; the α 8.6° case may
   need [-4, 1], state it in the caption).
3. `paraview_apply_filter({"input": "<case>", "filter_type": "StreamTracer",
   "name": "<case>_sl", "properties": {"Vectors": ["POINTS", "velocity"],
   "SeedType": "Line", "SeedType.Point1": [-1.0, -1.0, 0], "SeedType.Point2":
   [-1.0, 1.0, 0], "SeedType.Resolution": 60}})` for the velocity panels (2D:
   z = 0 plane; if the reader reports a z-extent, seed at its mid-plane).
4. `paraview_set_camera({"preset": "+Z", "position": [0.5, 0, 6], "focal_point":
   [0.5, 0, 0], "view_up": [0, 1, 0]})` then a second zoomed call for the LE
   (focal 0.05, 0, height ±0.15) where the suction peak lives.
5. `paraview_add_color_legend({"name": "<case>", "array": "pressure-coefficient",
   "title": "C_p"})`.
6. `paraview_export_figure({"path": "<abs>/paper/figures/figB3_<case>_cp.png",
   "width_in": 3.2, "height_in": 2.4, "dpi": 300})`; also export `.pdf` for the
   paper's vector build.
7. `paraview_save_state` to `fluent/paraview/<case>.pvsm` so every figure is
   reproducible without the MCP.
8. Surface cp overlay (B1 only): matplotlib from `_surface.csv` (by header) on
   the three levels plus AirfRANS is not available for NACA 0012 in the test split,
   so overlay the three Fluent levels only.

Before the batch of exports, verify one export renders correctly (open the PNG
with Read) and that the airfoil outline is visible (add a `Contour` of
`velocity-magnitude` at 0 or an `ExtractSurface` of the wall zone if the EnSight
part list includes `airfoil`).

---

## 6. What lands where

### 6.1 `docs/RESULTS.md` section 6 (rewrite; keep 6.1, replace 6.2, add 6.3 to 6.5)

- 6.1 Grid study: as is, plus the GCI numbers from `results/fluent/gci.json`
  (p_obs, GCI_fine CD/CL, SA and SST) and the SA↔SST spread at L2 (CD 0.01076 vs
  0.01060, 1.5 %).
- 6.2 Solver offset (Table 5a): the six replicas, Δ per model, mean ± s, the
  cp-overlay figure, and the one-sentence verdict of FLUENT_PLAN 5.4 (is the
  SA↔SST spread larger than |Δ̄|?).
- 6.3 The active-learning campaign outcome (Table 5b): 24 cases × {S0, r1}
  status. State the section 0 facts 1 to 3 in plain language: divergence pattern,
  the α = 12° control, what r1 changed, what was rescued.
- 6.4 Surrogate vs Fluent on the accepted set (Table 5c, Fig 11b): per model MAE
  of CD/CL after offset correction, coverage at 90 %, the factor-F sentence, the
  quasi-steady rows with their error bars.
- 6.5 The C4 finding, stated exactly as section 1 (a)/(b)/(c), followed by the
  limitation paragraph below verbatim (so the paper and RESULTS.md never drift).
- Status flag for section 6 becomes ✅ with the words "qualitative external check".

### 6.2 Table 5 (`scripts/make_tables.py`, new `tab5_fluent.tex/.md`; the current
`tab5_selected_cases.tex` stays as Table 5a input and gets a "solver outcome"
column)

Three stacked parts in one `table*` environment:
- **5a selection + outcome:** the existing 8 acquisition rows plus "steady
  outcome (S0 / r1)" and the surrogate prediction (`cd_int_mean ± cd_int_std`) so
  the reader sees σ next to the outcome.
- **5b offset:** 6 rows, AirfRANS CD/CL, Fluent SA, Fluent SST, Δ_CD, δ_CD, Δ_CL;
  footer mean ± s per model.
- **5c verified comparison:** one row per accepted case (arm, NACA, Re, α, cells,
  y+max, status, CD_fluent, CD_fluent − Δ̄, Transolver CD_int ± σ, SDF-FNO, GNN,
  90 % covered yes/no). Read from `results/fluent/surrogate_vs_fluent.csv`.
Caption must contain "grid-converged at L2 (GCI_fine CD = ...)" and "n = ...,
qualitative external check".

### 6.3 Fig 11 (`scripts/make_figures.py`)

- (a) unchanged (`fig11_active_pool`).
- (b) new `fig11_active_verify(outdir, results)` reading
  `results/fluent/surrogate_vs_fluent.csv` + `fluent_summary.csv`: x = arm
  (random, offset replicas, grid study, acquisition, variance), y = |CD_surrogate −
  CD_fluent,corrected| for Transolver (markers), error bars = Δ spread ⊕ quasi-steady
  half p2p; filled markers = converged, hollow = quasi-steady; the acquisition and
  variance arms plotted as *vertical bands at the top of the axis* labelled "no
  steady solution (S0, r1)", with the surrogate's own σ_CD written beside each,
  so the plot shows the regime split rather than hiding it. Small-n must be
  visible: annotate n per arm. Also add the 90 % conformal half-width as a shaded
  reference level. Write `fig11_active_verify.pdf/png`; the paper figure keeps
  the two-panel layout, replacing the `\figph` placeholder.

### 6.4 Paper (`paper/main.tex`)

- Section 7 "Active learning with solver verification": replace the "deferred"
  paragraph and the "Solver verification" / "solver offset" paragraphs with the
  results; keep the "falsifiable claim, restated precisely" paragraph and add the
  two-halves resolution from section 1 (c). Remove the `\todo`.
- Appendix B: grid study numbers + GCI, y+ achieved, the r1 settings and the
  outcome table, the offset method and Fig B2, the ParaView field figures.
- Discussion, replace the "Fluent verification set has not been run yet" bullet
  with this limitation text (edit for voice, keep every fact):

> **Solver verification is a qualitative external check on the solvable subset.**
> The 27-case Fluent set ran in full; the grid study is grid-converged at the
> production level and the six AirfRANS replicas bound the solver offset. But 20 of
> the 24 active-learning cases, including all 16 acquisition- and variance-arm
> picks, sit at α = 18°, beyond the AirfRANS envelope and, as it turned out,
> beyond what steady 2D RANS can solve: they diverged under the batch settings and
> again under a conservative retry (longer first-order start, reduced
> under-relaxation), and one random-arm case at α = 12° did too. We report that
> outcome as the result it is: the label-free score concentrated on the corner of
> the pool where the steady operator the surrogate learned has no solution, which
> supports the score as an extrapolation detector but leaves the quantitative
> half of the claim ("higher error on acquired cases") untestable with a steady
> solver. The surrogate-versus-Fluent numbers therefore cover n = 10 to 11
> low-to-moderate-angle cases (random arm, replicas, grid case), all inside or
> near the training envelope; they say how far the surrogate is from an
> independent solver where a steady answer exists, not how it fails where one
> does not. A time-averaged URANS truth for the deep-stall picks, with its own
> time-step and window study, is future work, and even then compares the
> surrogate to a quantity it was never trained to predict. Seven cases at Re =
> 7 × 10^6 run at M ≈ 0.32, past the usual incompressible limit; both solver and
> surrogate are incompressible, so the comparison is consistent, but the absolute
> coefficients there carry that caveat.

### 6.5 Bookkeeping
- `docs/DECISIONS.md` D-026: "Fluent campaign outcome and C4 reframing" recording
  the three decisions of this plan (no URANS in v1, r1 retry, six frozen replicas).
- `docs/PROGRESS.md` entry per executor session.
- `docs/plans/FLUENT_PLAN.md` section 5.2: replace "Exact sim ids are pinned once..."
  with the frozen table of section 3; section 7 cost table replaced by measured
  wall times from `fluent_summary.csv`.
- `fluent/README.md`: drop "Nothing in here has been executed"; add the `r1`
  variant and `--naca-params` to the contents table.
- `.gitignore`: EnSight and `.cas.h5/.dat.h5` already ignored; `results/fluent/*.csv`,
  `*.json`, `fluent/paraview/*.pvsm` are committed.

---

## 7. Execution sequence (single licence, serial solves; everything else overlaps)

| step | who | what | wall | gate |
|---|---|---|---|---|
| 1 | Opus | `collect_fluent.py` + `--gci`; `fluent_summary.csv` for S0; commit | 1 h | classifications match section 0 table |
| 2 | Opus | mesh_gen `--naca-params`; geometry check vs npz (gate 3.2); make_cases `naca_params`/`mu`; append offset block; render `--write-mesh` | 1.5 h | max NN distance < 5e-4 c on all 6; min orth ≥ 0.2 |
| 3 | batch | offset study, 12 runs SA+SST, then SST on the 4 converged random cases | ~3.5 h licence | all 16 converged, y+max < 1 |
| 4 | Opus (parallel to 3) | r1 template + make_cases/run_batch `--variant`; `diverged_cases.txt`; `compare_fluent.py` + test; ParaView figures B1, B3 from the existing exports | 3 h | tests green; 7 PNGs render |
| 5 | batch | r1 on the α = 12° control; decision point (r2 probe only if it fails); then r1 on the 19 α = 18° cases | ~4 to 6 h licence | each case classified |
| 6 | Opus | `compare_fluent.py` on all accepted cases, 3 models; `fig11_active_verify`; `tab5_fluent`; B4/B5 figures | 2 h | CSVs + figures regenerate from `scripts/` only |
| 7 | Opus / Sonnet | RESULTS.md 6.x, paper section 7 + appendix B + limitation bullet, DECISIONS D-026, FLUENT_PLAN/README edits; `make_results_digest.py`; commit + push (cocomo069) | 2 h | `\todo` count 0 in section 7; digest regenerates |
| 8 (opt.) | batch | Tier-3 URANS demonstrator, only on explicit user go | 2 to 3 h licence | CL/CD history figure |

Total: ~9 to 12 h licence time unattended (steps 3 and 5 chained in one
`run_batch.py` invocation each, resumable), ~9 h executor time, mostly overlapping.

**Definition of done for this plan:** `results/fluent/{fluent_summary.csv, gci.json,
offset.csv, surrogate_vs_fluent.csv, surrogate_vs_fluent_summary.json}` exist and
regenerate from scripts; Table 5 (a/b/c) and Fig 11 (a/b) build from them;
RESULTS.md section 6 is ✅ with the limitation text; the paper's section 7 contains no
"pending" sentence; every diverged case has both an S0 and an r1 transcript on disk.
