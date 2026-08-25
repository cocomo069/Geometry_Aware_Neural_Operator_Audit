# PLAN_PHASE2.md — Remaining work to arXiv v1

Planned by Fable 5, 2026-08-25. Executor: Opus/Sonnet agents. This sequences everything
after the core grid (done: 3 models × 6 splits × s0, committed) and the in-flight
K=5 ensembles (`configs/sweeps/kaggle_ensembles.yaml`, 48 runs on Kaggle T4).

**Budget model.** GPU = chained Kaggle T4 sessions, ~30 GPU-h/week, ≤~11 h/session.
Measured per-run (400 epochs, full 700-sim train): **GNN ≈ 1.5 h, transolver ≈ 15 min,
sdf_fno ≈ 15 min**; epoch cost scales ≈ linearly with train-set size. Local machine:
CPU must stay light (user Fluent job running) — local work is inference/analysis only,
`num_workers=0`, small batches; the P2000 GPU is free to use for inference.

**Two lanes run in parallel:**
- **GPU lane (Kaggle):** ensembles (finishing) → data-efficiency → trained ablations.
- **CPU lane (local):** UQ glue code + conformal + Figs 6–8/Tab 3 → AL scoring → paper.

Ablation-run rule (critical): `run_id = {model}_{split}_s{seed}[_{tag}]` does **not**
encode config deltas — every ablation run MUST set `tag=` (e.g. `tag=cond_mask`) or it
will clobber/skip against the core runs.

---

## Phase A — Conformal calibration & UQ figures (C3; highest priority; CPU-only)

**Preconditions.** Ensembles finished on Kaggle and merged locally via
`python kaggle/pull_results.py` (merges `results/` and `checkpoints/`, republishes
`geo-op-runs`). After merge, for each of 3 models × {full, reynolds, aoa, shape5}:
`results/{model}_{split}_s{0..4}/metrics.json` and `checkpoints/{model}_{split}_s{0..4}/best.pt`.
Verify with a reconcile pass (12 × 5 = 60 run dirs incl. the 12 core s0 runs).

**What exists (no design needed, only wiring):**
- `src/uq/ensembles.py` — `load_ensemble_checkpoints(paths, config)` → `EnsemblePredictor`
  with `.predict()` (per-point/per-sim mean+std) and `.propagate_forces(batch, integrate_forces,
  denormalize=...)` (integrate-then-average, per spec).
- `src/uq/conformal.py` — `SplitConformalCoefficients`, `SplitConformalField`
  (sim-level scores, `max`/`quantile` aggregation), scores `normalized`/`absolute`.
- `src/uq/coverage.py` — `sweep_levels_coefficients`, `sweep_levels_field`,
  `make_uq_record`, `make_uq_report`, `write_uq_report` → validated `results/uq/<id>.json`
  (satisfies both `uq-1` and `src/eval/schema.py::UQ_SCHEMA`).
- `scripts/train.py::build_datasets` builds train/cal/test datasets with per-split
  train-only normalization (D-021); datasets expose `denormalize`.
- `scripts/evaluate.py --subset cal` already exists (fallback path only; see below).

**MISSING glue (executor tasks, in order):**

1. **`scripts/run_uq.py` (the main deliverable).** One script, per (model, split):
   - CLI: `--model gnn|sdf_fno|transolver --split full|reynolds|aoa|shape5
     --seeds 0,1,2,3,4 --cal-split <split>|full --levels 0.8 0.9 0.95 --device auto
     --out results/uq`.
   - Read `results/{model}_{split}_s0/config.yaml`; build **cal** and **test** datasets
     via `scripts.train.build_datasets`; load the K `best.pt` checkpoints with
     `load_ensemble_checkpoints`.
   - Run inference over cal and test (batched, `torch.no_grad`, P2000 or CPU):
     per-sim **coef** members (`cd_int`/`cl_int` via `propagate_forces` with
     `src.physics.force_integration.integrate_forces` + dataset `denormalize`;
     `cd_head`/`cl_head` from `coef_head`, denormalized) and per-point **field** members
     (denormalized `p`). Truth: `cl_true`/`cd_true` and true `p` from the batch.
   - Fit + sweep: `sweep_levels_coefficients` (per-target: cd_int, cl_int, cd_head,
     cl_head) and `sweep_levels_field` (aggregation `max` AND `quantile` agg_q=0.9)
     at levels {0.8, 0.9, 0.95}; both scores `normalized` and `absolute`
     (= conformal-score ablation, free); K-subsets {1, 3, 5} of the members
     (= ensemble-size ablation, free — K=1 uses `absolute` score only, σ is zero).
   - **Two calibration modes — this is the C3 experiment:**
     (i) *matched*: calibrate on the split's own `cal` → "conformal delivers nominal
     coverage when exchangeable"; (ii) *transfer*: calibrate on `full`'s cal, evaluate
     on the OOD split's test → "the guarantee breaks under shift, by this much".
   - Output: `results/uq/{model}_{split}_k{K}.json` (matched) and
     `results/uq/{model}_{split}_k{K}_calfull.json` (transfer) via `write_uq_report`
     (`ensemble_id` = filename stem; `seeds`, `k`, `model`, `split` filled).
   - Smoke test now, before ensembles land: run with `--seeds 0` (K=1, absolute score)
     against the existing core checkpoints; add `tests/test_run_uq.py` (tiny synthetic
     run dir, CPU).
2. **`fig7` + `fig8` in `scripts/make_figures.py`** (currently only fig6 exists;
   7/8 are listed in the docstring but unimplemented):
   - Fig 7 *coverage vs shift*: x = split ordered {full, shape5, reynolds, aoa},
     y = empirical coverage at nominal 0.9 (coef `cd_int` record), one line per model,
     solid = transfer (`_calfull`), dashed = matched, horizontal line at 0.9.
     Data via `src.viz.data.load_uq` / `load_uq_reports`.
   - Fig 8 *interval width ID vs OOD*: mean/median width (coef cd + field) per split,
     grouped bars or box per model; log-y if needed.
3. **`table3_calibration` in `scripts/make_tables.py`** (listed in docstring, missing):
   rows = model × split, cols = coverage@{0.8,0.9,0.95} (matched | transfer), mean width
   @0.9, ECE. LaTeX + MD, same `_render` machinery.
4. Run the batch: 3 models × 4 splits × 2 modes (+ K/score sub-records) — a single
   driver loop or `--all` flag in run_uq.py. Minutes per (model, split) on the P2000;
   keep CPU footprint trivial.

**Expected outputs.** ~24 `results/uq/*.json`; `paper/figures/fig0{6,7,8}_*.{pdf,png}`;
`paper/tables/tab3_calibration.{tex,md}`.

**Compute.** 0 GPU-h Kaggle; ≲1 h local inference total. **CPU-light — mandatory.**

**Definition of done.** All 24 reports schema-valid; matched coverage within Wilson CI
of nominal on `full` for all models; figs 6–8 + tab 3 render from committed JSON;
`pytest tests/ -q` green; results + figures committed.

---

## Phase B — Data-efficiency sweep (GPU) → Figure 9

**Design decision — run ALL THREE models.** Cost is dominated by the GNN yet still
small (see below, ~7 GPU-h total); the scientific question ("does the ranking flip when
data is scarce?") is precisely about the cheap-vs-expensive architectures, so cutting
M1 or M2 would gut the figure. No reason to cut.

**Sizes.** {25, 50, 100, 200, 400} × seeds {0, 1, 2} as *new* runs; the largest point
(700 = full train) is **free** — reuse `{model}_full_s{0,1,2}` from the core grid +
ensembles. (Spec said "800"; the full train list after the cal carve is ~700 — cap
there and label the axis with the true n.)

**Preconditions / MISSING glue:**
1. **Split manifests** `data/splits/full_n{25,50,100,200,400}.json`: derived from
   `full.json` — train = deterministic seed-0 shuffled prefix of full-train (nested:
   n25 ⊂ n50 ⊂ … ⊂ n400), **cal and test identical to `full`'s**. Add a small generator
   in `src/data/splits.py` (reuse `_pack`/`_assert_disjoint`) + extend
   `tests/test_splits_no_leakage.py` to the new manifests. D-021 handles normalization
   automatically (per-split train-only stats from the new, smaller train list).
2. **Sweep spec** `configs/sweeps/kaggle_dataeff.yaml`:
   ```yaml
   name: kaggle_dataeff
   defaults: { overrides: [train.batch_size=16] }
   matrix:
     config: [configs/gnn.yaml, configs/transolver.yaml, configs/sdf_fno.yaml]
     overrides:
       split: [full_n25, full_n50, full_n100, full_n200, full_n400]
       seed: [0, 1, 2]
   ```
   Model-major so truncated sessions yield complete model rows. Epochs stay 400
   (matched budget); best.pt by val guards small-n overfitting.
   Also push the new manifests + spec to the `geo-op-code` dataset
   (`kaggle/push_code.py`) before launching.
3. **Fig 9 wiring check**: `src/viz/data.py::_train_size_from` infers `train_size` from
   config keys (`n_train`/`train_size`/…) — verify it resolves for split-file-driven
   runs; if not, teach it to read `n_train` from `data/splits/<split>.json` (it is
   written by `_pack`). `fig9_data_efficiency` already exists and gates on ≥3 sizes.

**Compute estimate** (epoch time ∝ n_train; Σ sizes = 775 ≈ 1.1 × a full run per seed):
| model | per seed | × 3 seeds |
|---|---|---|
| GNN | ~1.7 h | ~5.0 h |
| transolver | ~0.28 h | ~0.9 h |
| sdf_fno | ~0.28 h | ~0.9 h |
| **total** | | **≈ 7 GPU-h** (45 runs) |

**Definition of done.** 45 `results/{model}_full_n*_s{0..2}/metrics.json` committed;
`fig09_data_efficiency` renders 6 size points × 3 models with seed spread; log-log
slope quoted in the caption.

---

## Phase C — Ablations (sequenced by value ÷ cost)

**Tier 0 — FREE (CPU, already inside Phase A's run_uq.py; run all):**
1. Ensemble size K ∈ {1, 3, 5} — member subsets, no new training.
2. Conformal score `normalized` vs `absolute` — both fitted per report.
3. Field aggregation `max` vs `quantile(0.9)` — both fitted per report.

**Tier 1 — cheap GPU, RUN for v1 (~4 GPU-h):**
4. **M2 conditioning** (the model with a conditioning knob; spec §8): variants `mask`
   and `sdf+normals` vs the existing `sdf`, on splits {full, shape5} (the geometry
   axis), seed 0 → 4 runs × ~15 min ≈ **1 GPU-h**.
   Overrides: `model.conditioning=mask tag=cond_mask` / `"model.conditioning=sdf+normals"
   tag=cond_sdfnrm`. NOTE: code accepts `sdf+normals` (`src/models/sdf_fno.py::CONDITIONINGS`);
   the yaml comment saying `sdf_normals` is wrong — quote the `+` on the CLI.
5. **Physics-loss λ_F on/off, M1 only** (spec §5.4 ablation): `train.weights.force=1.0
   tag=lamF` on {full, combined}, seed 0 → 2 runs × 1.5 h ≈ **3 GPU-h**. Question
   answered: does optimizing consistency buy OOD accuracy, or only smaller FSC?
   Single on/off, not a λ sweep (see cuts).
   → New spec `configs/sweeps/kaggle_ablations.yaml` (6 runs, both axes, tags baked
   into overrides).

**Tier 2 — CUT for arXiv v1** (revisit for the journal version): λ_F multi-value sweep;
λ_S symmetry-loss training; `spectral_groups=1` textbook-FNO variant; volume-variant
residual ablation (D-006); K > 5; M2/M3 ensembles on `combined`.
*Optional half-tier if Fig 7 wants a hardest-shift point:* seeds 1–4 for transolver +
sdf_fno on `combined` (8 runs ≈ 2 GPU-h; skip GNN's 6 h) — decide after seeing Fig 7.

**Definition of done.** 6 tagged runs committed; a short ablation table (extend
make_tables or an appendix table) comparing p_rel_l2 / cd_spearman / fsc_cd vs the
core counterparts; one paragraph of findings per axis in the paper appendix.

---

## Phase D — Active learning scoring (+ Fluent, user-gated)

**Runnable NOW (no CPU green-light needed — inference only, GPU-light):** everything up
to and including emitting the case list. Do it right after Phase A (needs the same
merged ensemble checkpoints).

**What exists:** `src/active/pool.py` (NACA 4/5-digit generator emitting frozen batch
dicts, validated vs Abbott & von Doenhoff), `acquisition.py` (`acquisition_score` =
z(σ_CD) + β·z(FSC) + γ·z(L_sym), `select_cases`, `random_scores`, `write_case_list`),
`diversity.py` (`select_top_k_diverse` farthest-point), `fluent/make_cases.py` +
templates (rendered, lint-checked), `fluent/cases_to_run.json` (placeholders to be
replaced).

**MISSING glue:** **`scripts/run_active.py`**:
1. Build the pool: ~200 NACA shapes (4- and 5-digit, parameter ranges excluding
   training shapes) × a small (Re, AoA) grid including values outside the training
   band; assert no pool member matches a training sim (`parse_sim_name` params).
2. Score with the **transolver K=5 `full` ensemble** (best model, cheapest inference):
   σ_CD from `propagate_forces(...)["cd"].std`; FSC = |mean cd_int − mean cd_head|;
   optionally L_sym (skip if slow — β,γ default 1.0, γ=0 acceptable).
3. Three selections of 8, via `select_cases` + `select_top_k_diverse`:
   (a) uncertainty-only (`variance_only_scores`), (b) FSC-augmented
   (`acquisition_score`), (c) `random_scores(seed=0)`.
4. `write_case_list(...)` → `fluent/cases_to_run.json` (24 cases, strategy-labelled,
   meta: ensemble_id, pool size, β/γ), then run `fluent/make_cases.py` to render
   journals into `fluent/cases/` (CPU-trivial).
5. Save the full pool score table (`results/active/pool_scores.csv`) — it also serves
   as extra label-free FSC evidence for the paper.

**WAITS for user CPU green-light (deferred, D-007):** running the Fluent journals
(24 cases + 3-level grid-independence study; ansys MCP tools wired), extracting
Fluent C_D/C_L, the "were uncertain cases actually harder" comparison, Figs 10–11.

**Compute.** Scoring: minutes on the P2000. Fluent: hours of CPU — user-gated.

**Definition of done (now-part).** `fluent/cases_to_run.json` + rendered `fluent/cases/`
committed; pool_scores.csv committed; selection reproducible from seed.

---

## Phase E — Paper writing (parallel, CPU-trivial)

`paper/main.tex` compiles (12 pp skeleton, all figure/table placeholders, refs.bib).

**Draft NOW from existing results (~70% of the paper):**
- Intro + contributions (C1 consistency, C2 OOD, C3 calibration; C4 framed as protocol).
- Related work; Methods (3 architectures @1.5 M matched params, reimplementation note
  per D-002; FSC definition; symmetry residual; split definitions incl. shape5/combined).
- Experimental setup (AirfRANS, cache, per-split normalization D-021, G1 validation
  0.03%, no-leakage audit, batch-16 matched budget D-019).
- Results — core grid: Tables 1–2, Fig 4 (centerpiece), Fig 5, Fig 12; the three
  headline claims (architecture ordering on fields; split-spread ≫ model-spread on
  Spearman vs the ridge baseline 0.83–0.90; FSC grows monotonically with shift =
  label-free OOD detector).
- Reproducibility appendix + limitations.
- **Schematic figures 1–3** (missing, executor task, CPU-light): Fig 1 protocol
  overview (draw), Fig 2 splits diagram (draw), Fig 3 example surface-pressure field
  render from a cached npz + one model prediction.

**After Phase A:** C3 section — matched vs transfer coverage, Figs 6–8, Tab 3, the
"conformal fixes calibration, not robustness" punchline.
**After Phase B:** data-efficiency subsection (Fig 9, slope + any ranking flip).
**After Phase C:** ablations appendix.
**After Phase D (now-part):** AL methodology subsection + selected-case table; Fluent
verification explicitly marked as in-progress (or included if green-lit in time).

**Recommended arXiv v1 scope (minimal complete):** core grid + full C3 + Fig 9 +
Tier-0/1 ablations + AL selection-as-protocol. Do NOT block v1 on Fluent or
DrivAerNet++ — both are framed as ongoing/extension work.

**Definition of done.** main.tex compiles with zero placeholder figures/tables in the
main body; numbers in text regenerated from committed metrics (no hand-typed results);
internal read-through pass; user sign-off before any public push (GITHUB_UPLOAD_RULES
staging + Zenodo happen at release, per D-018).

---

## EXECUTION ORDER (single recommended sequence)

GPU lane = Kaggle sessions; CPU lane = local, light. Steps in the same numbered block
run in parallel across lanes.

| # | Lane | Action | Cost | Gate |
|---|---|---|---|---|
| 1a | CPU | Write Phase-A glue now: `scripts/run_uq.py` (+K=1 smoke on core s0 checkpoints), fig7/fig8, table3, tests | 0 GPU | tests green |
| 1b | CPU | Draft paper sections that need no new results (Phase E list) + schematic Figs 1–3 | 0 | — |
| 1c | GPU | Ensembles finish on Kaggle (in flight, ~24 GPU-h total ≈ this week's quota); chain sessions via `pull_results.py` → relaunch until 48/48 | in flight | 60 run dirs reconciled |
| 2 | CPU | Ensembles merged → run conformal batch (matched + transfer, all K/score variants) → `results/uq/*.json`, Figs 6–8, Tab 3 | ~1 h local | Phase A DoD |
| 3 | CPU | `scripts/run_active.py` → pool scores + 3×8 selection → `fluent/cases_to_run.json` + rendered journals | minutes | Phase D now-DoD |
| 4 | GPU | Next quota window: build data-eff manifests + specs (1 short local task), push code dataset, launch `kaggle_dataeff.yaml` (~7 h) then `kaggle_ablations.yaml` (~4 h) — fits ONE ~11 h session; chain if truncated | ~11 GPU-h | 45+6 runs committed |
| 5 | CPU | Fig 9 + ablation table; C3 + data-eff + ablation text into the paper | 0 | fig09 renders |
| 6 | CPU | Assemble arXiv v1 (scope above), regenerate all figs/tables from committed results, review pass | 0 | Phase E DoD |
| 7 | — | User-gated, anytime after 3: Fluent runs + grid study (CPU green-light) → Figs 10–11 → fold into v1 if it lands before submission, else v2 | user CPU | — |
| 8 | — | Optional post-v1: `combined` ensembles for M2/M3 (2 GPU-h), λ_F sweep, DrivAerNet++ (Globus, user) | — | — |

Total remaining Kaggle GPU: ~24 h (ensembles, in flight) + ~11 h (data-eff + ablations)
→ fits in two weekly quota windows. Everything else is local and CPU-light.
