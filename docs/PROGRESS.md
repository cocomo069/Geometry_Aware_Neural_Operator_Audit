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

## 2026-08-24 — A1 (data) — Phase 1 complete

### Delivered

| Path | What |
|---|---|
| `src/__init__.py`, `src/data/__init__.py` | package init (were missing) |
| `src/data/splits.py` | `parse_sim_name`/`SimName`, `carve_cal`, `build_splits_from`, `write_all_splits`, `load_split`; CLI `python -m src.data.splits` |
| `src/data/airfrans_loader.py` | `NormStats`, `AirfransSurfaceDataset`, `collate`, `make_dataloader` |
| `scripts/download_data.py` | HEAD size-check gate + airfrans download wrapper |
| `scripts/build_cache.py` | raw VTU/VTP -> per-sim `.npz` + `manifest.json` + `norm_stats.json` |
| `docs/DATA_NOTES.md` | merged: Fable's pre-download block preserved verbatim, sections A1.0-A1.10 appended |
| `tests/test_splits_no_leakage.py`, `tests/test_loader.py` | 64 tests |

### Test results

`.venv/Scripts/python.exe -m pytest tests/test_splits_no_leakage.py tests/test_loader.py -q`
-> **64 passed, 2 skipped** (the 2 skips are the real-`data/splits/*.json`
manifest checks, which arm themselves automatically once the splits are built).
Both files pass on CPU with no dataset present.

Full suite: `311 passed, 7 failed` — all 7 failures are in `tests/test_active.py`
and `tests/test_infra.py` (A4/A5 code). None import `src/data`; see the proposal
below for one of them.

### Download size check (task 1) — done, not executed

`scripts/download_data.py --check-only` resolves the URL by introspecting
`airfrans.dataset.download` (hard-coded fallback only if that fails) and issues
a HEAD, falling back to a ranged 1-byte GET:

```
Dataset URL: https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip
Remote size: 10029067577 bytes = 9.34 GiB
```

Under the 20 GB ceiling and under PLAN.md's 15 GB reassess threshold. The abort
path was exercised with `--max-gb 1` (exits 2 with a clear message).
`--skip-if-present` works. **No download was started.**

### Pre-download validation of `build_cache.py`

Since the real data had not landed, I generated a **synthetic simulation in the
exact AirfRANS on-disk format** (O-grid around a circle: `_internal.vtu` with
`U`/`p`/`nut`/`implicit_distance` + closed-polyline `_aerofoil.vtp` with inward
`Normals` + `manifest.json`). `airfrans.Simulation` ingests it unmodified, so
`surface` masking, `reorganize`, `wallshearstress`, `force_coefficient` and the
`ptc` quadrature all really run. Scripts:
`scratchpad/{synth_dataset,verify_cache}.py`.

Verified end to end (raw -> cache -> splits -> dataset -> batch): outward-normal
flip (`mean(n . r_hat)` = 1.000000), `sum(surf_ds)` = perimeter, closed-surface
identity `norm(sum(n ds))/perimeter` = 1.8e-8, CCW contour starting at the TE,
float32 + C-order + exact section 4 shapes, `--limit`, idempotent re-run,
`norm_stats.json` from the `full` train list only, `batch_idx` correctness,
denormalise round-trip, `num_workers` clamped to 2.

### Findings worth other agents' attention

1. **Gate G1 is reachable, and the quadrature is exact.** Reproducing the
   package's coefficients from the cached arrays with the CONTEXT.md section 6
   formula gives **0.000 % error in float64** on every synthetic sim. The
   worry in DATA_NOTES A1.7 (reference uses cell-averaged `p_bar * N_bar`,
   we use point weights) is empirically nil. The float32 residual is pure
   cancellation and scales as `K * 1e-7` with
   `K = max|p| * perimeter / (0.5 U^2 |coef|)`; the synthetic circle is
   pathological (`K ~ 1e6`), real airfoils sit at `K ~ 1e2..1e3`, predicting
   **1e-5..1e-4** relative error — 2-3 orders inside the 1 % gate. Full analysis
   in DATA_NOTES A1.10. **A2: if G1 fails, it is not precision or quadrature** —
   check sign conventions, the drag-first return order of `force_coefficient`,
   or accidental inclusion of `nu_t` in the wall stress.
2. **`force_coefficient` returns DRAG FIRST**: `((cd,cdp,cdv),(cl,clp,clv))`.
   The cache stores unambiguous `cl_true`/`cd_true` plus the pressure/viscous
   split `cl_p/cl_v/cd_p/cd_v` so a G1 failure can be localised to one term.
3. **Wall shear stress uses molecular `NU` only**, not `nu + nu_t`, and needs the
   volume-mesh velocity jacobian — it cannot be recomputed from surface data, so
   `surf_tau` must come from the cache.
4. **Two viscosities exist.** The dataset was generated with `nu = 1.56e-5`;
   `Simulation.NU` recomputes ~1.54981e-5 from T. Re-derived split logic uses
   1.56e-5 (`splits.NU_DATASET`) to match the official `reynolds` band; the
   per-sim `nu` in the cache is `Simulation.NU` for physics.
5. **`scarce` has no test set of its own** — `airfrans.dataset.load` maps it onto
   `full_test`. Replicated.
6. NACA series is discriminated **only** by parameter count (3 -> 4-digit,
   4 -> 5-digit); the parameters are continuous reals, not integers.

### Proposals

- **PROPOSAL (A4, bug):** `tests/test_infra.py::test_parse_naca_params_...`
  fails because `scripts`' `baselines.parse_naca_params` slices the sim name
  from field **3** instead of **4**, so it returns `[aoa, d1, d2]` instead of the
  NACA parameters (observed: `[6.917, 3.16, 3.542]` for
  `airFoil2D_SST_57.872_6.917_3.16_3.542_11.436`, expected
  `[3.16, 3.542, 11.436]`). There is now a single canonical parser —
  `src.data.splits.parse_sim_name` — which is unit-tested against the upstream
  documentation's own worked examples. A4/A5 should import it rather than
  maintain duplicates. (A5's `tests/test_active.py` NACA failures are in its own
  `naca_generator` port and are unrelated to name parsing.)
- **PROPOSAL (additive, no FROZEN change):** the cache carries extra keys
  beyond CONTEXT.md section 4 — `vol_is_surf`, `cl_p/cl_v/cd_p/cd_v`,
  `rho`, `nu`, `perimeter`, `n_vol_full`, `cache_version`. All additive;
  documented in DATA_NOTES A1.8.
- **PROPOSAL (clarification, not a change):** `AirfransSurfaceDataset` takes a
  keyword `subset="train"` in addition to the frozen positional
  `(split_file, processed_dir, normalize_stats)`, since a manifest holds all
  three lists. `collate` also emits `ptr`, `n_surf`, `num_graphs` and `*_phys`
  physical copies alongside the frozen keys.
- **PROPOSAL (default worth a decision):** positions are **not** normalised by
  default. Chord = 1 m so `surf_pos` is already O(1), and per-component
  standardisation would stretch y ~10x relative to x, destroying the aspect
  ratio the geometry-aware models rely on. `surf_ds`/`surf_normal`/`vol_sdf` are
  always physical because `integrate_forces` consumes them directly. Stats for
  those fields are still written, so a config can opt in.
- **PROPOSAL (test promotion):** `scratchpad/synth_dataset.py` +
  `verify_cache.py` would make a good `tests/test_build_cache.py` — it exercises
  the whole raw->cache path without the 10 GB download. Left out because
  `tests/test_build_cache.py` is outside A1's assigned paths.

### Bug fixed during validation

`np.savez_compressed` appends `.npz` to any path whose name does not already end
in it, so the atomic-write temp file `<sim>.npz.tmp` was silently written as
`<sim>.npz.tmp.npz` and the subsequent `replace()` raised `FileNotFoundError`.
Now writes through an open handle. This would have broken the very first real
cache build.

### Still open / to verify after the download lands

- Per-sim `N` (internal nodes) and `Na` (surface nodes); exact `reynolds`/`aoa`
  train/test counts.
- Re-run the `verify_cache.py` G1 block on ~20 real sims; expect median relative
  error <= 1e-4.
- Wall-clock for the full 1000-sim build (0.39 s/sim on the 1.3k-node synthetic
  mesh; real meshes are ~100x larger, so budget roughly 1-3 s/sim, i.e.
  30-50 min single-process — I/O bound, no multiprocessing per the CPU
  constraint).
- No packages were installed by A1.
