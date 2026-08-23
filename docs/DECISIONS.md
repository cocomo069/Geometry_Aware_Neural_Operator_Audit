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
