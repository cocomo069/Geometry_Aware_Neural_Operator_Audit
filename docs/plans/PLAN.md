# PLAN.md — Master execution plan

Planned by Claude Fable 5 (2026-08-24); executed by Opus-class subagents with Fable
handling integration, review, and the correctness-critical validations.
The spec's week-by-week timeline is **discarded** (user instruction: ASAP). Ordering
below is by dependency, not calendar. Scope decisions are logged in DECISIONS.md.

## Strategy in one paragraph

AirfRANS carries the paper end-to-end (spec's own fallback position). We build the whole
2D pipeline — data, three architectures, consistency protocol, OOD splits, ensembles +
conformal, data-efficiency, active-learning scoring — as runnable, tested code, then
execute training sweeps on the local P2000 (4 GB) with per-epoch checkpointing.
DrivAerNet++ (needs Globus auth — **user action**) and Fluent verification runs
(CPU-heavy — deferred per user instruction) get complete, ready-to-run scaffolding now
and execute later. Fable plans and gates; Opus agents implement in parallel against the
frozen interfaces in CONTEXT.md.

## Phase 0 — Foundation (Fable, direct) ✅ when checked

- [x] Probe hardware/software; pick stack (py3.11 + torch 2.8.0+cu126, see D-001..D-004)
- [ ] `uv` venv + installs (background)
- [x] git init, skeleton, LICENSE (MIT), .gitignore
- [x] Docs: CONTEXT.md (frozen interfaces), PLAN.md, DECISIONS.md, PROGRESS.md
- [ ] AirfRANS download (~10 GB, background; verify size via HEAD first)
- [ ] requirements.txt lock; smoke-test torch CUDA on P2000

## Phase 1 — Parallel implementation (Opus agents, disjoint directories)

| Agent | Scope (owns) | Key deliverables | Gate/tests |
|---|---|---|---|
| A1 data | `src/data/`, `scripts/build_cache.py`, `docs/DATA_NOTES.md` | package introspection, cache builder to §4 schema, splits incl. `shape5`+`combined`, norm stats | `test_splits_no_leakage.py`; cache builds for 5 sims |
| A2 physics | `src/geometry/`, `src/physics/` | force integration (§6), quadrature/normals/curvature, SDF-on-grid, symmetry ops, residuals | synthetic circle tests pass on CPU |
| A3 models | `src/models/` | M1 GNN, M2 GINO-lite SDF-FNO, M3 Transolver-style, shared coef head, losses hooks | forward/backward smoke on random batch, CPU; param budgets 1.5M ±20% |
| A4 infra | `src/utils/`, `src/eval/`, `scripts/train.py`, `scripts/evaluate.py`, `configs/` | YAML config+overrides, seed, train loop w/ per-epoch checkpoint+resume, eval harness, metrics schema | 2-epoch train on synthetic data resumes correctly |
| A5 uq+al | `src/uq/`, `src/active/` | ensembles aggregation, split conformal (sim-level + coef-level), coverage/reliability, pool gen (NACA 4/5-digit param family), acquisition score, farthest-point diversity | unit tests: conformal coverage on synthetic gaussian data ≈ nominal |
| A6 aux | `fluent/`, `scripts/subset_drivaernet.py`, `paper/`, `model_cards/` | Fluent TUI journal templates + meshing + grid-study plan + README, DrivAerNet++ subset script (manifest-driven, Globus instructions), LaTeX skeleton w/ all figure/table placeholders, model-card template | journals lint-reviewed; LaTeX compiles if pdflatex present (else content-only) |

All agents: obey CONTEXT.md §12; log to PROGRESS.md.

## Phase 2 — Integration & gating (Fable-led, sequential)

1. Build full cache (1000 sims) — I/O bound, run in background.
2. **GATE G1: force-integration validation** on ground-truth fields vs dataset C_L/C_D
   (median rel err ≤1%). Nothing proceeds until green. (Spec's "most important day".)
3. Wire models ↔ loader ↔ trainer; short GPU smoke run per model (5 epochs, `full`).
4. **GATE G2:** M1 trained on `full` beats constant-predictor and ridge baselines on
   C_D. Establishes the pipeline learns signal.
5. Fit VRAM: measure peak memory per model; fix batch sizes.

## Phase 3 — Experiment execution (scripted sweeps, GPU, background)

Priority order (each produces committed `metrics.json`; sweeps via `scripts/sweep.py`):

1. **Core grid:** 3 models × splits {full, scarce, reynolds, aoa, shape5, combined} × seed 0.
2. **Ensembles:** +4 seeds for 3 models × {full, reynolds, aoa, shape5} (K=5).
3. **Conformal + coverage:** calibrate on `cal`, report per split (CPU-cheap, GPU-free).
4. **Data efficiency:** sizes {25,50,100,200,400,800} × 3 seeds × M1+M3 (M2 if budget allows).
5. **Ablations:** M2 conditioning (SDF/mask/SDF+normals); physics-loss λ sweep on M1 only;
   ensemble size K ∈ {1,3,5}; conformal score choice.
6. **Baselines:** constant predictor, ridge (geometry params + cond → C_D). CPU-trivial.
7. **Active learning scoring:** pool of unseen NACA shapes × (Re, AoA) grid; rank by
   acquisition; select 3×8 case lists (uncertainty / FSC-augmented / random) → emit
   `fluent/cases_to_run.json`. Fluent execution deferred.

Budget rules: any single run projected >2 h on the P2000 gets its epoch budget cut;
per-epoch checkpointing means interruptions are free. Runs are sequenced, never parallel
on the 4 GB card.

## Phase 4 — Deferred / user-blocked

- **Fluent verification runs** (CPU-heavy): journals ready in `fluent/`; run when user
  green-lights CPU use. Includes 3-level grid independence on one case + AirfRANS
  replication offset study. Ansys MCP tools available in-session for this.
- **DrivAerNet++**: requires Globus login + CC BY-NC acceptance → user downloads
  coefficients CSV + decimated meshes for the manifest our script emits; then rerun the
  same protocol via the 3D variants (stretch goal; 2D paper stands alone).

## Phase 5 — Figures, tables, writing

- `scripts/make_figures.py`: the 12 spec figures from committed metrics (error-vs-shift
  is the centerpiece), tables 1–6 auto-generated to LaTeX.
- Populate paper skeleton; model cards from metrics; README results table.

## Risk register (delta from spec §10)

| Risk | Mitigation |
|---|---|
| 4 GB VRAM ceiling (spec assumed 16 GB T4/P100) | surface-only primary task (D-006); volume ablation subsampled to ≤16k pts; batch size autotuned; grad accumulation |
| Pascal fp32-only → slower epochs | small models, surface point clouds are tiny (~1k pts); realistic |
| Parallel agents drift from interfaces | CONTEXT.md frozen sections + Fable integration gate G1/G2 |
| airfrans package API mismatch vs assumptions | A1 introspects first, writes DATA_NOTES.md before cache code |
| Dataset download stalls | size-check via HEAD; resumable; if >15 GB stop and reassess |
