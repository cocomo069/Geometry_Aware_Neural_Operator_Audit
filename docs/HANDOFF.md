# HANDOFF.md — Session handoff (updated 2026-08-24, session 2)

Complete state of the project for whoever (human or Claude) picks this up next.
Read order for a fresh session: this file → [CONTEXT.md](CONTEXT.md) (frozen interfaces)
→ [DECISIONS.md](DECISIONS.md) (20 logged decisions + rationale) → [PLAN.md](PLAN.md)
→ [DATA_NOTES.md](DATA_NOTES.md) (dataset conventions) → [PROGRESS.md](PROGRESS.md) (per-agent logs).

## Session-2 update (what changed since session 1)

- **Kaggle cloud training is LIVE on T4** (D-020: P100 is Pascal sm_60, incompatible with
  Kaggle's torch — must request `machine_shape: NvidiaTeslaT4`). Full path validated end
  to end (smoke v9, sweep_rc=0, GNN 2-epoch in 50s). Setup: private repo
  `cocomo069/Geometry_Aware_Neural_Operator_Audit`; two Kaggle datasets
  (`airfrans-cache` = 867 MB processed cache+splits, `geo-op-code` = repo snapshot);
  driver `kaggle/session_driver.py`, helpers `kaggle/{launch,push_code,push_cache,pull_results}.py`.
- **Core grid launched** (`configs/sweeps/kaggle_core.yaml`: 3 models × 6 splits × seed 0,
  batch 16 per D-019) on `cocomo069/geo-op-session`. ~2.8 h/run on T4 → ~5 sessions.
  **To resume across sessions:** `python kaggle/pull_results.py` (merges results+checkpoints,
  republishes `geo-op-runs`), then `python kaggle/launch.py --sweep configs/sweeps/kaggle_core.yaml
  --runs-dataset cocomo069/geo-op-runs`. Sweep skips finished runs.
- **Figures/tables pipeline done** (`scripts/make_figures.py`, `scripts/make_tables.py`,
  `src/viz/`): Fig 4 (error-vs-shift) and Fig 5 (FSC scatter) render live from baselines;
  rest data-gated. Tables 1–2 as LaTeX+MD. 9 viz tests; full suite **369 passing**.
- **Three QA agents** auditing physics/models, data/eval/UQ, infra/kaggle (session 2, in flight).
- Push mechanics reminder: `gh` default account is **wkabz07** (client — never touch);
  `gh auth switch -u cocomo069` before any push, switch back after. Kaggle token at
  `C:\Users\LAPTOP\.kaggle\access_token`.

## Session-3 update (2026-09-04) — project complete except one quota-blocked item

- **Everything is done and pushed** except the **GNN K=3 deep-ensemble** (upgrades ONE row of
  Table 3: GNN calibration from K=1 → K=3). All other deliverables complete: core grid, K=5
  ensembles (transolver + sdf_fno), data-efficiency, ablations (Table 4), full Fluent C4
  verification (grid study + AL + offset + r1 salvage, Table 5), 22 figures, Tables 1–5,
  RESULTS.md / OVERVIEW.md / RESULTS_AUTO.md. 98+ commits on `cocomo069/Geometry_Aware_Neural_Operator_Audit`.
- **Why GNN K=3 is blocked (both GPU paths dead):**
  1. `cocomo069` weekly GPU quota (30 h) is **exhausted** — a kernel push is rejected with
     "Maximum weekly GPU quota of 30.00 hours reached". Clears on the weekly rolling reset.
  2. Second account `taimooramin0699` (new token) **cannot use GPU**: verified 2026-09-04 via a
     probe kernel — `enable_gpu:true` but `torch.cuda.is_available()==False`, device_count 0.
     Root cause: the account is **not phone-verified** (Kaggle disables GPU + internet until then).
     Only coco can phone-verify it. Its datasets are irrelevant while GPU is off, so the ports
     were stopped (only tiny `geo-op-code` + a `geo-op-gpuprobe` kernel remain there — safe to delete).
  - `cocomo069` token restored as active (`~/.kaggle/access_token.cocomo069.bak` → `access_token`).
- **Autonomous finish (no human needed):** scheduled task **GeoOpCycle** (every 6 h) runs
  `kaggle/cycle.py`. While quota is exhausted each tick is a fast no-op (6 h cooldown). When the
  weekly quota resets, cycle.py launches `geo-op-gnnens` (gnn × {full,reynolds,aoa,shape5} ×
  seed{1,2} = 8 runs), pulls results, reruns `run_uq`, regenerates figures + Table 3, and
  commits+pushes under `cocomo069` (`regenerate_and_commit()`). Queue: `kaggle/cycle_queue.json`,
  only `geo-op-gnnens` pending.
- **To finish TODAY instead of waiting for reset:** coco phone-verifies `taimooramin0699`, then
  re-run the three dataset ports + `kaggle/launch.py --kernel-slug taimooramin0699/geo-op-gnnens
  --runs-dataset taimooramin0699/geo-op-runs --sweep configs/sweeps/kaggle_gnn_ens_k3.yaml`.

## Session-4 update (2026-09-06) — PROJECT COMPLETE ✅

- **The GNN K=3 ensemble finished. Every deliverable is now done; the cycle queue is fully drained.**
- What happened: the `taimooramin0699` GPU path stayed dead (never phone-verified), so the finish
  came via the **autonomous cocomo069 path**. The weekly GPU quota reset on **2026-09-06** (my
  Saturday-00:00-UTC estimate was wrong; Kaggle's reset landed ~a day later), and `GeoOpCycle`
  auto-launched `geo-op-gnnens` on the first post-cooldown tick. The kernel needed **two versions**
  (v1 ran ~8 h and completed 5 of 8 runs; v2 resumed via the runs-dataset skip logic and finished
  the last 3: `gnn_aoa_s2`, `gnn_shape5_s1`, `gnn_shape5_s2`). `cycle.py` then pulled results,
  reran `run_uq`, regenerated figures + Table 3, and committed+pushed (`f06a494`, `6ddee8e`).
- **Result:** all 12 GNN ensemble runs present (4 splits × seeds s0/s1/s2 = **K=3**). Table 3 now
  has real M1 GNN calibration on all four splits. Headline: K=3 lifts GNN matched cov@.9 on the
  Reynolds shift 0.70 → 0.97 and AoA 0.79 → 0.89 on the *normalized* band (a near-vacuous width, see
  RESULTS.md §3.1); the tight absolute-score coverage stays ~0.70, so this is a band-width effect, not
  a genuine calibration gain.
- **Cleanup done:** project scratch temp dirs removed; `cocomo069` kaggle token active. Left as-is
  (safe, need coco to delete via web UI if wanted): tiny `taimooramin0699/geo-op-code` dataset +
  `geo-op-gpuprobe` kernels. `GeoOpCycle` scheduled task disabled (queue drained, nothing to launch).
- **Reusable C: caches NOT deleted** without coco's say-so (he flagged the C: drive): `uv` 5.9 G,
  `.cache` 1.4 G (codex-runtimes, not this project), session `Temp/claude` 1.6 G. Awaiting his choice.

### Round-1 final-review remediation (2026-09-06)

A Fable agent did a final sign-off review and returned DO-NOT-APPROVE with real findings. Fixed:
- **Tables 1 & 2** were averaging tagged smoke/ablation runs and data-eff sub-splits into core cells
  (GNN full printed 0.256 vs true 0.0753). `make_tables.py` now filters `tag.isna()` + canonical
  splits and reports seed mean±std. Regenerated.
- **Fluent leg** was stale/contradictory: `fluent_summary.csv` had no r1 rows and no `has_cas`;
  the surrogate comparison on disk was Transolver-only n=6. Re-ran `collect_fluent --gci` (now 86
  rows incl. 20 r1) and `compare_fluent` (GNN K=2→K=3); all three models now n=12, matching the
  numbers RESULTS §6.4 already quoted (which previously had no committed provenance). Regenerated
  Table 5 + Fig 11. Rewrote RESULTS §6.2–6.5: r1 does NOT rescue the α=18° picks (19/20 stay
  diverged: bounded final C_D≈0.09–0.12 but transient |C_D|>10 excursion + 3–7% drift), it only
  fixed the in-envelope α=12° case; added the indicative surrogate-underprediction (~2.5–3.5×).
- **Disclosures added to RESULTS**: `cd_int` drag ranking is no better than ridge (GNN 0.36 on full)
  and collapses under shift, so "beats ridge" is head-route only (§2.2/§2.3); the AL score rests on
  the D-026 `cd_int` pool artifact and an envelope-overshooting pool grid (§6.5a); the GNN K=3
  "0.97" reynolds coverage is a near-vacuous normalized band, absolute-score coverage stays 0.70,
  and the GNN under-covers in-distribution (§3.1); "matched" calibration is in-distribution (§3).
- **Symmetry**: disclosed that reflecting cambered airfoils makes out-of-family shapes (§2.4). The
  review's "signed-curvature sign bug" was checked and is a FALSE POSITIVE (recompute-on-reflected
  == cache, verified numerically); not changed.
- **Doc hygiene**: test count 369→449, OVERVIEW Part 8 statuses, RESULTS dual date, DECISIONS D-027.
- **`paper/main.tex` is a pre-final 2026-09-03 snapshot** and is deliberately NOT updated (coco
  writes the paper). It still says GNN K=1 / Fluent not run / data-eff pending; the numbers of
  record are RESULTS.md + `paper/tables|figures/`, not the .tex. Model cards (spec C5) remain a
  template only — logged as an archiving to-do.
- Per coco's instruction the review→fix loop repeats until a review returns no issues.

### Round-2 final-review remediation (2026-09-06)

Round-2 confirmed all 7 round-1 fixes HOLD, but caught that the same table fixes had not been applied
to the figures, plus prose the files refuted. Fixed:
- **Figures were still contaminated** (`make_figures.py` had no tag filter): `_neural()` now drops
  tagged runs too, and fig04/fig05 route through it (+ canonical-split restriction), so fig04/05/09/
  10/12 match the corrected tables. Regenerated all PNG/PDF.
- **Fig 11b** plotted `cd_head` error against the `cd_int` half-width; `compare_fluent` now emits
  `q90_halfwidth_cd_head` and fig11b uses it (log-y, regime-marked).
- **Collector startup-guard bug (real):** `collect_fluent` applied the |C_D|>10 blow-up test to the
  iteration-1 impulsive start (peak ~10–14 on every case), wrongly flagging settled α=18° cases as
  diverged. Added `STARTUP_GUARD=50`. This promotes **two α=18° picks** to accepted
  (`al_var_naca32012` converged C_D 0.167, `al_rand_naca4412` quasi-steady 0.113), so the surrogate
  comparison is now **n=14** and Half 2 has real deep-stall data (surrogate off by ~0.05–0.13,
  ~700–900× in-dist).
- **RESULTS §6.4 rewritten as a regime split** (moderate n=12: 3–10× in-dist, cov 0.33–0.83;
  post-stall n=2: ~0.1 error). **§6.3** r1 mechanism corrected (startup spike at iteration 1, not the
  1st→2nd-order switch; 17/19 non-settling with 7–21% drift; 2 settled). **§6.5(c)** Half 1 corrected
  (random-arm coverage 3/6 GNN, 0/6 others — not "all inside the interval"); Half 2 now supported.
- **§4** "no curve crossing" corrected: Transolver best at every size, but GNN beats SDF-FNO at
  n≤50 (they cross ~n≈70–100).
- **Caveats**: §3.1 ECE clarified (Table 3 = normalized-score ECE; 6–10× is absolute-score) and the
  unsupported `combined` calibration claim removed (no combined ensemble exists); §2 grids labeled
  seed-0; Table 5c cov90 labeled Transolver; small-number drift fixed (SST +0.00069, δ_CD 5.3–13%).
- The round-2 review's other claims (curvature sign bug) remained false positives. 449 tests pass.

---

## 1. What this project is

Implementation of the research design in
[`05_geometry_aware_neural_operator_surrogate_uq.md`](../05_geometry_aware_neural_operator_surrogate_uq.md):
an evaluation/methodology paper auditing three geometry-conditioned neural surrogates
(GNN, SDF-FNO "GINO-lite", Transolver-style transformer) on AirfRANS with a consistency
protocol (force self-consistency, symmetry), multi-axis OOD splits, conformal
uncertainty, data-efficiency curves, and a Fluent-verified active-learning loop.
Target: arXiv preprint (spec's timeline was discarded per user — ASAP, dependency-ordered).

**Execution model (user-mandated):** Fable 5 plans/gates/integrates; Opus subagents
implement. Nothing CPU-heavy on this laptop (Fluent runs deferred); GPU free to use;
research-grade sweeps belong on Kaggle free tier (D-013), local P2000 is for smoke/gates.

## 2. Environment (all working, verified)

- Repo: `D:\Personal Projects\geom_aware_neural_operator`, git on `main`, no remote yet.
- venv: `.venv` — **Python 3.11, torch 2.13.0+cu126** (P2000 is Pascal sm_61: cu126 is
  the LAST wheel line that supports it; never upgrade past cu126). Run everything as
  `.venv/Scripts/python.exe` from repo root. fp32 only, no AMP, no torch.compile (D-004).
- GPU: Quadro P2000 4 GB. Measured: M1 GNN needs batch 4 (grad-checkpoint knob exists),
  M2/M3 batch 8. M1 ≈ 0.7 s/sim/epoch → full-split epoch ≈ 8 min locally.
- Data: AirfRANS fully downloaded (`data/raw/Dataset.zip`, 10.03 GB, extracted to
  `data/raw/Dataset/`, 1000 sims) and fully cached (`data/processed/airfrans/`,
  1000 .npz, cache_version=2, ~5 GB, plus `manifest.json` + `norm_stats.json` computed
  from the 700 `full`-train sims). Six split manifests committed in `data/splits/`.
- Tests: **449 passed** as of the 2026-09-06 full run (`.venv/Scripts/python.exe -m
  pytest tests/ -q`, ~35 s, CPU-only). Run this first after any resume.

## 3. What is DONE (with the numbers that matter)

### Infrastructure (all committed, all tested)
| Piece | Where | State |
|---|---|---|
| Frozen interfaces | docs/CONTEXT.md | cache schema §4, splits §5, physics §6, model API §7, results schema §9, config/training §10 |
| Data pipeline (A1) | src/data/, scripts/build_cache.py | cache v2 incl. precomputed grid_sdf (stored TRANSPOSED to match M2 — don't "fix"), edge_index, curvature; bit-exact vs model fallback paths |
| Physics core (A2) | src/geometry/, src/physics/ | force integration validated analytically (Kutta–Joukowski, d'Alembert) and against dataset |
| Models (A3) | src/models/ | gnn 1.49M / sdf_fno 1.49M / transolver 1.45M params (matched ±3%); registry `build_model(name, cfg)`; losses with λ_H head term (D-016) |
| Train/eval infra (A4) | src/utils/, src/eval/, scripts/train.py, evaluate.py, sweep.py, configs/ | per-epoch checkpoint + bit-identical resume (norm_ref in ckpt); `sweep.py --max-seconds`; `data.missing=skip` for partial caches |
| UQ + active (A5) | src/uq/, src/active/ | split conformal (sim-level coef + field via max/quantile scores), NACA 4/5-digit pool generator (validated vs Abbott & von Doenhoff), acquisition + farthest-point diversity |
| Aux scaffolds (A6) | fluent/, paper/, model_cards/, scripts/subset_drivaernet.py | Fluent gmsh+journal pipeline with 2 placeholder cases + 3-level grid study RENDERED in fluent/cases/; paper/main.tex COMPILES (12 pp, all 12 figures + 6 tables as placeholders, refs.bib complete); DrivAerNet access doc |
| Kaggle tier | kaggle/ | session_driver/launch/push_cache/pull_results ready; UNTESTED (blocked on token) |

### Gates & results (the science so far)
- **GATE G1 PASSED (definitive, n=1000):** integrating cached ground-truth surface
  fields reproduces dataset coefficients — CD rel err median **3.31e-4**, p95 1.20e-3,
  max 1.61e-3; CL median **4.84e-6**. Convention set: `pressure_mode='kinematic'`,
  rho=1, a_ref=1 (dataset stores p/ρ; .vtp normals are INWARD, cache flips to outward).
- **Baselines done on all 6 splits** (12 runs in `results/{constant,ridge}_<split>_s0/`):
  headline finding — **ridge (NACA params + condition → CD) ranks CD at Spearman
  0.83–0.90 on every split incl. OOD**; constant-field integration only 0.27–0.42.
  This is the bar the neural models must beat (spec §8.b anticipated exactly this).
- **Pipeline learns on real data:** 2-epoch GNN dry-run trained on GPU end-to-end,
  metrics.json written (`results/gnn_full_s0_smoke_dryrun/`).
- Split sizes: full 700/100/200, scarce 160/40/200, reynolds 404/100/496,
  aoa 704/100/196, shape5 391/98/511, combined 191/48/246 (train/cal/test, disjoint,
  leakage-tested).

### Git state
13 commits on `main`, latest `ce978ad`. **Untracked, intentionally:** `src/viz/`
(A7's half-finished work, see §4), `results/gnn_full_s0_g2/` (dead run, only
config.yaml — safe to delete), `data/cache_build*.log`, `results/g2_train*.log`
(empty). `data/raw`, `data/processed`, `checkpoints/` are gitignored by design.

## 4. What was IN FLIGHT when the session died (needs restart)

1. **GATE G2 run — NOT done.** A 60-epoch GNN training (`gnn_full_s0_g2`) was launched
   detached and died at startup when the machine/session went down (0-byte logs, empty
   checkpoint dir). Relaunch:
   `.venv/Scripts/python.exe scripts/train.py --config configs/gnn.yaml split=full seed=0 tag=g2 train.epochs=60`
   (~6–8 h locally; auto-resumes from `checkpoints/gnn_full_s0_g2/last.pt` if interrupted).
   G2 = trained M1 beats constant + ridge on CD metrics.
2. **A7 (figures/tables agent) — half done.** On disk: `src/viz/{style,data,__init__}.py`,
   `paper/figures/README.md`. Missing: `scripts/make_figures.py`, `scripts/make_tables.py`,
   `tests/test_viz.py`. The original A7 brief is in the session-1 transcript; in a new
   session simply re-launch an agent with the same scope (spec Figures 4,5,6,7,8,9,12 +
   Tables 1–3 from `results/*/metrics.json`; the 12 baseline runs are live fixtures).

## 5. What REMAINS (dependency order, from PLAN.md Phase 3–5)

1. **G2** (above) — then commit its metrics.
2. **Core grid:** 3 models × 6 splits × seed 0 (18 runs, ~400 epochs each at research
   fidelity). Sweep spec exists: `configs/sweeps/core.yaml`; runner:
   `scripts/sweep.py --spec ... --max-seconds N` (skips completed runs; every run
   checkpoints per epoch). **Local is too slow for the full grid (~days) — this is the
   Kaggle workload** (see §6 blockers).
3. **Ensembles:** +4 seeds for 3 models × {full, reynolds, aoa, shape5}
   (`configs/sweeps/ensembles.yaml`) → K=5 ensembles.
4. **Conformal + coverage:** CPU-cheap once ensembles exist — `src/uq/` is ready;
   produce `results/uq/*.json` per split, then reliability/coverage-vs-shift figures.
5. **Data-efficiency:** sizes {25,50,100,200,400,800} × 3 seeds × (M1, M3 at least).
6. **Ablations:** M2 conditioning (sdf/mask/sdf+normals — config switch exists),
   physics-loss λ sweep on M1, ensemble size K∈{1,3,5}, conformal score choice.
7. **Active learning:** score the NACA pool with the trained ensembles
   (`src/active/`), select 3×8 cases (uncertainty / +FSC / random), write
   `fluent/cases_to_run.json` → `fluent/make_cases.py` renders journals.
8. **Fluent verification runs** — DEFERRED by user instruction (CPU-heavy). Everything
   is rendered and runnable via the Ansys MCP tools when green-lit; grid-study cases
   for NACA0012 Re3e6 α5 already in `fluent/cases/`. See docs/FLUENT_PLAN.md.
9. **Figures/tables** (finish A7) → populate `paper/main.tex` → model cards from
   metrics → README results table.
10. **DrivAerNet++ 3D leg** — BLOCKED on user's Globus login (docs/DRIVAERNET_ACCESS.md
    has the step-by-step); `scripts/subset_drivaernet.py` is ready. Optional stretch;
    the 2D paper stands alone (D-005).

## 6. Blockers needing the USER (all non-blocking for local G2/figures work)

1. **Kaggle API token** → save at `C:\Users\LAPTOP\.kaggle\kaggle.json`; account must
   be phone-verified. Unblocks: `kaggle/push_cache.py` (upload the 5 GB processed cache
   as a dataset) → `kaggle/launch.py --sweep configs/sweeps/core.yaml` (P100 sessions).
2. **GitHub repo approval** — `gh` is authenticated as **cocomo069** (NEVER use the
   wkabz07 account — client's). Asked to create private repo `geo-operator-audit`; user
   has not yet said yes. Before ANY push, read `D:\Personal Projects\GITHUB_UPLOAD_RULES.md`
   (mandatory per user's global CLAUDE.md; note its naming/structure rules conflict with
   this repo's layout — resolve with user or via the rules file before pushing).
3. After repo exists: user adds a read-only fine-grained PAT as Kaggle secret `GH_TOKEN`
   (kernel clones the repo). Never handle the raw PAT yourself.
4. (Later) Globus login for DrivAerNet++; green light for CPU-heavy Fluent runs.

## 7. Gotchas the next session must not rediscover the hard way

- **cu126 pin** (D-001): `uv pip install torch --index-url https://download.pytorch.org/whl/cu126`.
  cu128+ has no Pascal kernels — torch.cuda.is_available() would still be True-ish
  trap-free but kernels fail; don't touch the torch install.
- **No compiled-extension deps** (D-002): no PyG/torch_scatter/neuraloperator. Everything
  is self-implemented; kNN via scipy, scatter via index_add_.
- **AirfRANS conventions** (DATA_NOTES.md): .vtp normals INWARD (cache flips); pressure
  is kinematic p/ρ; wall shear uses molecular ν only; `force_coefficient` returns
  **drag first** `((cd,..),(cl,..))`; sim name fields 2,3 = inlet velocity, AoA;
  NACA family distinguished only by param count (3→4-digit, 4→5-digit). Canonical
  parser: `src.data.splits.parse_sim_name` — never re-implement.
- **grid_sdf in the cache is transposed on purpose** (D-017a) to match M2's row-major
  latent grid; a test asserts bit-exact equality. Cache/model comparability is tracked
  by `cache_version` (currently 2); metrics are only comparable within a version.
- **Normalization** (D-014): positions physical (chord units), fields standardized;
  the trainer's force-loss closure denormalizes fields AND re-normalizes coefficients —
  both halves required.
- **λ_H=0.1 head term must stay on** (D-016) or the coef head gets zero gradient and
  FSC is meaningless.
- **Schema validators**: `validate_*` raise, `check_*` return error lists (A4/A5
  harmonized). `cd_spearman` is on integrated CD; `cd_head_spearman` /
  `cd_int_spearman` added for the ridge comparison.
- **Windows**: paths via pathlib; Bash tool heredocs choke on some quoting — agents
  should use the Write tool for files with heavy `$`/backticks; git prints CRLF
  warnings (harmless).
- **Detached long jobs**: Bash background tool calls are killed at 10 min — launch
  multi-hour jobs via PowerShell `Start-Process` with log redirect (pattern used for
  cache build; note a machine sleep/restart kills those too, hence per-epoch resume).
- Session usage limits killed all 6 agents mid-flight once; they were resumed by
  SendMessage with "continue where you left off" and lost nothing. Partial work on
  disk + PROGRESS.md is the recovery mechanism.

## 8. Quick-resume commands

```bash
cd "D:/Personal Projects/geom_aware_neural_operator"
.venv/Scripts/python.exe -m pytest tests/ -q            # expect 449 passed
.venv/Scripts/python.exe -m pytest tests/test_force_integration.py -k gate -q  # G1 re-check
# G2 (relaunch, detached via PowerShell Start-Process or just foreground overnight):
.venv/Scripts/python.exe scripts/train.py --config configs/gnn.yaml split=full seed=0 tag=g2 train.epochs=60
# Baseline table sanity:
.venv/Scripts/python.exe -c "import json,glob; [print(p.split('\\\\')[1], json.load(open(p))['coef']['cd_head_spearman']) for p in sorted(glob.glob('results/*_s0/metrics.json'))]"
```
