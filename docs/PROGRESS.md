# PROGRESS.md — Running status log

Append-only. Agents: add dated entries under your section; flag interface-change
proposals with `⚠ PROPOSAL`.

## 2026-08-24 — Phase 0 (Fable)

- Hardware probed: Quadro P2000 4 GB (sm_61), i7-8850H, 32 GB RAM, 120 GB free on D:.
- Repo initialized (git, MIT, skeleton), docs written (CONTEXT/PLAN/DECISIONS).
- venv `.venv` (py3.11) creating; torch 2.8.0+cu126 + scientific stack installing in background.
- AirfRANS download pending size verification.

## Agent log

(append below)

## 2026-08-24 — A2 (physics/geometry) — DONE

- `src/geometry/quadrature.py`: closed-contour facet lengths, per-point ds
  (half-sum of adjacent facets, Σds = perimeter exactly), outward unit normals
  (central-difference tangent, orientation fixed via signed area — CW or CCW
  input both give outward), signed Menger curvature (+1/R on a circle),
  duplicate-closing-point handling. numpy/float64.
- `src/geometry/normals.py`: angle-error, outward-fraction and
  `orient_normals_outward` helpers for validating/fixing dataset normals
  (AirfRANS .vtp normals are INWARD per DATA_NOTES — cache must flip; these
  helpers verify).
- `src/geometry/sdf.py`: `sdf_on_grid` (exact point-to-segment distance,
  chunked full sweep, cKDTree midpoint prefilter for >4096 segments; sign via
  vectorized crossing-number test; **negative inside body**),
  `occupancy_on_grid`, `make_grid`, `points_in_polygon`.
- `src/geometry/symmetry.py`: `reflect_x_batch` (y-flip of pos/normal/tau/u/
  cond ⇒ AoA→−AoA), `reflect_prediction` (R#: p invariant, tau_y negated,
  coef_head CL→−CL), `symmetry_residual(model, batch)` per spec §5.5
  (per-sample rel-L2 over concatenated p,tau; differentiable, usable as
  L_sym), `antisymmetry_check` (CL odd / CD even gaps).
- `src/physics/force_integration.py` (FROZEN §6): F = Σ(−p·n_out + τ)ds;
  drag = F·ê∞, lift = F·ê⊥, ê⊥ = rot90_ccw(ê∞) = (−ey, ex) — matches the
  AirfRANS rotation-matrix convention in DATA_NOTES. `pressure_mode=
  'static'|'kinematic'` absorbs the dataset p/ρ convention (kinematic: rho
  ignored, ½|u∞|² is the dynamic pressure). Batched (batch_idx/index_add)
  and unbatched; differentiable.
- `src/physics/residuals.py`: `knn_ls_weights` (precomputable per-sample
  cKDTree kNN + pinv LS gradients, exact for affine fields),
  `pointwise_divergence`, `divergence_residual` (per-sample mean div²),
  `noslip_residual` (spec §5.5 weighted near-wall form).
- Tests: `tests/test_geometry.py` (18) + `tests/test_force_integration.py`
  (15 + dataset gate): circle normals ≤1e-6 rad, Σds=2π, constant-p zero
  force, p=n·ê ⇒ F=−πê, d'Alembert (cd=cl=0) at 3 AoAs, rotating cylinder
  Kutta–Joukowski CL=−4πγ̂ at 2 AoAs (lift-sign check), reflection flips CL
  preserves CD, shear-only drag, kinematic mode, batched==unbatched ≤1e-12,
  grad flow, exact divergence on linear fields, no-slip values.
  **33 passed, 1 skipped** (gate test auto-skips until A1's cache exists;
  it asserts median rel err ≤1% vs cl_true/cd_true, pressure_mode=kinematic).
