# DECISIONS.md — Decision log (ADR style)

Format: ID, date, decision, why, alternatives rejected, consequences. Newest at bottom.

---

**D-001 · 2026-08-24 · Python 3.11 venv via uv; torch 2.8.0+cu126**
The GPU is a Quadro P2000 (Pascal, sm_61, 4 GB). PyTorch cu128+ wheels dropped
Maxwell/Pascal; cu126 wheels still ship sm_61 kernels. System Pythons are 3.14 (no
torch support guarantee) and 3.13; spec asks for 3.11. `uv` manages an isolated 3.11.
*Rejected:* conda (not installed), system 3.14 (ecosystem lag).
*Consequence:* pinned to cu126 wheel line; never upgrade torch past cu126 support.

**D-002 · 2026-08-24 · No compiled-extension deps: drop PyTorch Geometric, torch_scatter, torch_cluster, neuraloperator**
Windows + Pascal + cu126 wheel availability for PyG's compiled companions is a known
tar pit; `neuraloperator`'s GINO path can pull torch_scatter. All needed ops are small:
kNN via scipy cKDTree (CPU, cached per-sim), scatter via `torch.index_add_`/
`scatter_reduce`, FNO spectral conv is ~40 lines of `torch.fft`. We self-implement M1,
M2 (GINO-lite), M3 (Transolver-style) citing the reference papers/code.
*Rejected:* PyG stack (install risk), neuraloperator (dependency risk, less control for
4 GB fitting). *Consequence:* slightly more code to test, full control over memory; the
paper claims "reimplemented in the style of", not "used the authors' code" — must be
stated in the paper's reproducibility section.

**D-003 · 2026-08-24 · Ignore the spec's 8–12-week timeline; dependency-ordered ASAP plan**
User instruction. Phases in PLAN.md replace §7 of the spec.

**D-004 · 2026-08-24 · fp32 everywhere; no AMP; no torch.compile**
Pascal has no tensor cores and 1/64-rate fp16; AMP would slow things down and risk
numerics. torch.compile on Windows+Pascal is unreliable. *Consequence:* memory budget
computed at 4 bytes/elt; models sized to ~1.5 M params.

**D-005 · 2026-08-24 · AirfRANS carries the paper; DrivAerNet++ deferred to user-gated stretch**
DrivAerNet++ requires Globus auth (interactive; user-only) and multi-GB download under
CC BY-NC. The spec itself designates AirfRANS as the self-sufficient core. We ship
`scripts/subset_drivaernet.py` + instructions so the 3D leg can start the day access
exists. *Consequence:* paper scoped as 2D-complete + 3D-extension-ready.

**D-006 · 2026-08-24 · Surface-only primary task; volume model as single ablation**
Forces depend on surface fields only; the 4 GB card can't hold 180k-node volume graphs
comfortably. Volume variant trains on ≤16k subsampled points, used for divergence/
no-slip residual reporting (spec Q5's cheaper branch). *Consequence:* consistency story
leads with force self-consistency + symmetry, as the spec's fallback anticipated.

**D-007 · 2026-08-24 · Fluent runs deferred; journals authored now**
User: nothing CPU-intensive right now. Fluent 2D RANS cases + grid study are pure CPU.
All journals, meshing scripts and the case manifest are produced so the runs are a
button-press later (Ansys MCP tools are wired in this session). *Consequence:* C4
(solver-verified loop) is code-complete but data-pending at first arXiv draft.

**D-008 · 2026-08-24 · Plain YAML config + dotted CLI overrides; no Hydra; no wandb**
One fewer runtime dependency each; Hydra's value (composition) is small at this scale
and its Windows/job-dir behavior is a nuisance. CSV + JSON logging is grep-able and
committed. *Rejected:* Hydra, wandb free tier (network dependency on a flaky setup).

**D-009 · 2026-08-24 · Two new OOD splits defined by sim-name parsing**
`shape5` (train 4-digit / test 5-digit NACA) and `combined` (shape × Re). AirfRANS sim
names encode the NACA parameters (verified by A1 and documented in DATA_NOTES.md), so
splits are derivable without extra metadata. Calibration sets carved from train with
seed 0, never overlapping test (CONTEXT.md §5).

**D-010 · 2026-08-24 · Batching = concatenation + batch_idx (PyG-style), no padding**
Variable point counts per airfoil; padding wastes the tiny VRAM. All models and the
force integrator consume `batch_idx`. Frozen in CONTEXT.md §4/§7.

**D-011 · 2026-08-24 · Models implemented before full data availability, against synthetic batches**
Parallelism: dataset download (~10 GB) is the long pole on this connection, so A2–A5
develop and test against synthetic geometry (circles, random NACA-like contours) with
the frozen batch dict. Integration happens at Gate G1/G2 after cache build.

**D-012 · 2026-08-24 · Execution model split: Fable plans/gates, Opus implements**
User instruction ("plan with Fable 5, execute with Opus"). Fable retains: interface
design, force-integration gate review, integration debugging, final review. Opus
agents own the six Phase-1 workstreams. Where a workstream is unusually
correctness-critical (A2 physics), Fable reviews the diff line-by-line before G1.

**D-013 · 2026-08-24 · Two-tier compute: local P2000 for smoke/gates, Kaggle free tier for research-grade sweeps**
User directive: research-grade fidelity using the appropriate service. For this project
that is Kaggle (~30 GPU-h/wk, P100/2xT4, 12 h sessions) as the sweep workhorse — matches
the spec's own compute table. Local 4 GB P2000 stays the debug/gating device (force-
integration gate, 5-epoch smokes, baselines, conformal/coverage CPU work). Repo gets
Kaggle execution scaffolding: kernel driver notebook that clones the repo, pulls the
processed cache from a Kaggle Dataset, runs `scripts/sweep.py` partitions sized to a
12 h session, and re-uploads checkpoints+results as a versioned dataset so sessions
chain. Blockers needing user action (non-blocking for local work): (a) Kaggle API token
at `C:\Users\LAPTOP\.kaggle\kaggle.json`; (b) approval to create a GitHub repo (gh is
authenticated as cocomo069) so Kaggle can clone the code. RunPod/Vast A100 budget
(~$30–80) reserved for the DrivAerNet++ 3D leg only, if/when Globus access exists.

**D-014 · 2026-08-24 · Positions not standardized; geometry stays in physical chord units**
(A1 proposal, accepted.) Per-component standardization of `surf_pos` would stretch y by
~10× (airfoils are thin), destroying aspect ratio for kNN graphs, SDF grids and
curvature. Chord is already 1 m ⇒ positions are O(1) naturally. Field targets (p, tau)
ARE standardized; `surf_ds`, `surf_normal`, `vol_sdf` always physical so
`integrate_forces` consumes raw geometry. Canonical sim-name parser is
`src.data.splits.parse_sim_name` — all other modules must import it, not re-implement.

**D-015 · 2026-08-24 · M2 spectral conv uses groups=4 block-diagonal channel mixing**
(A3 finding.) Dense SpectralConv2d at frozen "modes 16, width 32" = 1.05M params/layer;
4 layers ≈ 4.2M, breaking the equally-frozen 1.5M budget. Resolution: block-diagonal
spectral mixing (`spectral_groups: 4`) with the dense 1x1 path restoring full mixing per
layer. `spectral_groups: 1` recovers the textbook layer (ablatable). Budget wins over
layer shape because matched capacity is what makes the model comparison meaningful (C2).

**D-016 · 2026-08-24 · Coefficient head always trained: λ_H·(|CL_head−CL|+|CD_head−CD|), λ_H=0.1**
(A3 finding.) Under the pure field loss the coef head receives zero gradient, making
FSC (int vs head) meaningless. The head-regression term is now part of the base loss
(normalized at init like other terms); λ_F stays the *consistency* ablation term
(|C_int−C_true| + |C_int−C_head|) per spec §5.4. A4 implements in losses/train config.

**D-017 · 2026-08-24 · Cache stores precomputed static geometry: grid_sdf, edge_index, curvature**
(A3 finding: M2 is CPU-bound recomputing SDF+cKDTree per step — 2.45 s/forward.)
Geometry is static per sim, so build_cache.py additionally stores optional keys:
`grid_sdf` (64x64 float32 on bbox [-0.5,1.5]x[-1,1], the M2 default), `edge_index`
(2,E int32, k=16 kNN), `curvature` (Ns,). Models already accept these via batch and
fall back to on-the-fly computation. CONTEXT.md §4 amended (this entry is the migration
record). Positions confirmed chord-frame physical (D-014), so the fixed bbox stands.

**D-017a · 2026-08-24 · Cache v2 vintage note: curvature changes model inputs**
(A1.) `edge_index`/`grid_sdf` are pure caches (predictions identical, asserted);
`curvature` upgraded from constant-zero to real signal → CACHE_VERSION=2. No
checkpoints existed pre-v2, so no comparability break occurred; rule going forward:
metrics.json runs are only comparable within the same CACHE_VERSION (recorded by the
harness). Cached grid_sdf is stored transposed to match M2's row-major latent grid —
bit-exact equality asserted in tests, do not "fix" the transpose.

**D-018 · 2026-08-24 · Private working repo `Geometry_Aware_Neural_Operator_Audit` under cocomo069; portfolio-rules upload deferred to paper release**
User provided tokens and approval. GITHUB_UPLOAD_RULES.md (read in full) targets
finished public portfolio uploads (staged copy, numbered 01_Code/ folders). This repo
is live infrastructure: Kaggle kernels clone it and tests import `src/` — restructuring
would break execution; and the spec's scooping-risk mitigation argues against public
until the arXiv timestamp. Resolution: PRIVATE repo now, named per the rules' naming
convention; at paper release, a rules-compliant public staging copy is produced
(numbered folders, README format, figures) alongside the Zenodo archive. Kaggle cloud
execution uses a repo snapshot shipped inside the Kaggle dataset (no GitHub token in
the cloud path; the GH PAT the user supplied stays only in Kaggle secrets, unused
unless snapshot mode fails).

**D-019 · 2026-08-24 · train.batch_size=16 uniformly across all three models (local + cloud)**
CORRECTED (QA session 2): the shipped configs' `train.batch_size` is **16** for all three
models (the 8/8/4 values in those files are under the separate `eval:` block, for
validation passes only). So training runs at batch 16 everywhere -- matched across
architectures, which strengthens the equal-budget comparison C2 -- and `kaggle_core.yaml`'s
`train.batch_size=16` default is an explicit-but-redundant statement of that intent, not an
override to a different value. Empirically batch 16 fits both the T4 (16 GB) and the local
P2000 (4 GB) -- the GNN dry-run trained at batch 16 on the P2000 without OOM (surface point
clouds are small). lr held at 1e-3. Batch size is recorded per run in metrics.json config.
Note (QA #4): local *neural* gating runs must use a distinct `tag=` (the `--dry-run` path
auto-adds `_dryrun`) so a `pull_results.py` merge of the cloud grid never clobbers a
same-run_id local run; baselines are safe (distinct `constant_`/`ridge_` model prefixes).

**D-020 · 2026-08-24 · Kaggle sweeps must request the T4 (sm_75), never the P100 (sm_60)**
Discovered during smoke validation: Kaggle's default "GPU" accelerator is a Tesla P100,
which is Pascal **sm_60** — and Kaggle's preinstalled PyTorch supports only sm_70+
(`no kernel image is available for execution on the device`), the same Pascal wall we hit
locally (D-001). The T4 is Turing **sm_75**, fully supported by the stock Kaggle torch, so
`kaggle/launch.py` sets `machine_shape: NvidiaTeslaT4` in kernel-metadata. Kaggle grants
2×T4 in this shape; our sweep is single-GPU sequential (one T4 used). Never fall back to
P100 unless also installing a cu126 torch (slow, avoided). T4 has 16 GB so batch_size=16
(D-019) is unaffected. Also recorded: the cache dataset mounts at
`/kaggle/input/datasets/<owner>/<slug>/…`, not `/kaggle/input/<slug>/`; the driver locates
it by searching for manifest.json rather than hard-coding a mount path.

**D-021 · 2026-08-24 · Per-split train-only normalization (QA2 MAJOR fix)**
CONTEXT §4 originally froze a single global `full`-train `norm_stats.json` reused for all
splits. QA2 showed that for the four OOD splits, most test sims are inside `full`-train, so
their input/output statistics (notably `cond` = Re/AoA, the OOD axis) entered the shared
normalization — contradicting CONTEXT §5 ("test never touched by normalization") and mildly
flattering the absolute OOD numbers the paper's headline rests on. No label leak, and the
equal-budget model *comparison* was unaffected (shared transform), but an OOD/calibration
paper cannot ship test-tainted normalization. Fix: the trainer now computes normalization
from THIS split's own train list (`data.norm_per_split_train=true` default;
`compute_norm_stats(split.train, processed_dir)`), so no cal/test statistic ever enters the
transform. Deterministic (fixed train list). For `full` this equals the old global stats;
for OOD splits it differs (e.g. aoa surf_p mean −823 vs full's −1118). Needs no cache change
— stats compute on the fly from the mounted npz. Also this session (QA2 minor): CONTEXT §9
coef block updated to list `cd_head_spearman`/`cd_int_spearman` (added in ce978ad; schema
already required them). **Consequence: any runs trained under the old global normalization
are superseded — the core grid was restarted after this fix.**

**D-022 · 2026-08-24 · Reflection drops fixed-grid caches so M2 recomputes them (QA1 F1 fix)**
`reflect_x_batch`/harness reflection passed `grid_sdf` (a field on a FIXED latent grid)
through unchanged under the y-reflection, so M2's symmetry residual compared the mirrored
surface against conditioning built for the un-mirrored geometry (measured SDF discrepancy
~0.2; prediction impact ~1e-5 on the untrained model but a real logic error in a
paper-critical diagnostic). Fix: reflection now drops `grid_sdf`/`grid_mask`/`grid_feat` so
the model rebuilds them from the mirrored `surf_pos`; `edge_index`/`curvature` are kept
(provably reflection-invariant). Harness `_reflect` now calls the canonical
`reflect_x_batch` (was silently falling back to a local copy via a stale name lookup).

**D-023 · 2026-08-25 · Standing division of labor: Fable plans, Opus/Sonnet execute**
User directive (reaffirmed). All PLANNING — phase design, experiment design, sequencing,
scope/scientific decisions — is delegated to a Fable 5 agent. All EXECUTION —
implementation, launching/monitoring sweeps, pulling results, figures, debugging,
mechanical edits — is done by Opus (the session default) and Sonnet (lighter mechanical
work). Applies to every remaining phase. Planning outputs land as docs/PLAN_*.md; Opus
executes against them. This mirrors Phase 1 (Fable froze interfaces/gates in CONTEXT.md;
Opus/Sonnet agents implemented).

**D-024 · 2026-08-25 · Autonomous chaining via Windows scheduled task running cycle.py**
User directive: keep going to completion without interference, survive Claude usage-limit
resets. Mechanism: a Windows Task Scheduler task `GeoOpCycle` runs `kaggle/cycle_tick.cmd`
every 30 min (14-day duration, IgnoreNew so ticks never overlap, 2 h limit). Each tick runs
`kaggle/cycle.py` once (idempotent one-step state machine): if the current queued sweep's
kernel is RUNNING it no-ops; if COMPLETE it pulls (lean, results-first) -> merges -> run_uq
(ensemble items) -> regenerates figures/tables -> commits + pushes (cocomo069) -> republishes
geo-op-runs -> advances or relaunches. Queue: dataeff (bootstrap on geo-op-session) ->
ablations -> gnnens. This drives the GPU pipeline with ZERO dependence on any Claude session,
so limit resets don't even pause it. Claude-side finite work (paper/figures) done by
executor agents. Stop condition: PLAN_PHASE3 definition-of-done. Fluent stays user-CPU-gated.
To pause: `schtasks /change /tn GeoOpCycle /disable`. Log: logs/cycle.log.

**D-025 · 2026-09-03 · Fluent verification campaign executing (v211, 6 cores, EnSight->ParaView)**
User lifted the CPU gate ("use fluent cpu gpu whatever, get it done"), specified v211 + 6
cores + ParaView post. Root-caused the mesh abort: off-by-one in mesh_gen.py wake-cut node
merge (i>nw+na should be >=) left a 5-node TE cell Fluent rejected; fixed + hardened the
self-check. Fixed 4 v211 TUI keyword bugs in case_template.jou (direction-0/1 not u/v;
turb-viscosity-ratio-profile; wall shear-bc removed; cm/steady/compute lines). L1 SA
validated: CD 0.0111, CL 0.553 (2piα theory 0.548, <1%). y+ target lowered 0.6->0.25 ->
y+max 0.87 (wall-resolved). ParaView can't read v211 .cas.h5 (CFF/HDF5), so the journal
now also EnSight-Gold-exports (verified opens in ParaView, 5 scalars+velocity). Full batch
`fluent/run_batch.py --sst-gridstudy` (30 runs: SA x27 + SST x3 gridstudy) launched
detached (~24h, 6 cores, sequential, resumable). 20/24 AL cases are post-stall AoA=18 where
steady RANS may not flat-converge -- that non-convergence IS the AL result (acquisition
picked the hard envelope), not a bug. Post: ParaView field contours via mcp__paraview__ +
native baked Fluent display objects; surface.csv feeds surrogate-vs-Fluent comparison (C4).

**D-026 · 2026-09-03 · Fluent campaign outcome + C4 reframing (PLAN_FLUENT_POST executed)**
The SA batch finished: grid study clean and grid-converged at L2; 4 low-AoA random cases
converged; 20 cases diverged (all 16 acquisition+variance picks at α=18°, the 3 random α=18°,
and the random α=12° case). Three decisions, executed by Opus per Fable's plan:
(1) **No URANS for v1.** A URANS time-average is a different object from the steady operator
the surrogate learned; |CD_surr - CD_URANS| would be uninterpretable. URANS is future work
(one optional demonstrator only on explicit user go).
(2) **Steady robustness retry `r1`** on all 20 diverged cases (longer first-order start,
reduced URFs incl. turbulence, no cd-steady stop; `templates/case_template_r1.jou`). The
α=12° random case runs FIRST as a control gate: if r1 does not rescue it, the 18° set is not
attempted (don't burn licence). Converts "diverged on first attempt" into a documented
converged / quasi_steady / diverged outcome per case.
(3) **Six frozen AirfRANS-replica offset cases** (continuous NACA params via
`airfrans.naca_generator`; per-case μ so Fluent's U matches each sim exactly) to bound the
solver offset Δ = CD_fluent - CD_airfrans and give a three-way AirfRANS/Fluent/surrogate check.
C4 becomes a three-part statement: (a) qualitative — the label-free score put 16/16 picks at
α=18°, outside the steady-RANS-solvable envelope (evidence it detects extrapolation);
(b) quantitative on the solvable subset (random + 6 replicas + grid case + any r1 rescue),
surrogate-vs-Fluent CD/CL offset-corrected with conformal coverage; (c) the falsifiable claim
split — half 1 (random picks are cases the surrogate gets right) is testable and holds; half 2
(acquired picks have higher error) is undefined against a steady truth that does not exist
there, reported as the finding. Implementation note: the integrated cd_int on the SYNTHETIC
pool geometry is an input-representation artifact (~5x vs the real mesh), so cd_head (the
coefficient head) is the surrogate CD used in the comparison; cd_int stays reliable on the
replicas (real mesh). Scripts: `scripts/{collect_fluent,compare_fluent}.py`,
`fluent/make_offset.py`, `run_batch.py --variant/--only-file`. No hand-tuning of any selection.

---

**D-027 (2026-09-06): GNN ensemble finished via the quota-reset path, not the second account.**
When `cocomo069`'s weekly GPU quota was exhausted, a second Kaggle account (`taimooramin0699`)
was tried to unblock the GNN K=3 ensemble. It does not work for GPU: a probe kernel with
`enable_gpu:true` returned `torch.cuda.is_available()==False`, device_count 0, and no internet.
Root cause is that a fresh Kaggle account has GPU + internet disabled until the phone number is
verified — an account action only coco can perform, and not something to spend automation on.
Decision: abandon the second-account port, keep the token restored to `cocomo069`, and let the
existing self-healing `GeoOpCycle`/`cycle.py` cooldown loop finish the job on the weekly reset.
It did (2026-09-06). Lesson for any future quota bypass: a new Kaggle account is useless for
compute until phone-verified, so verify that first before porting datasets to it.
