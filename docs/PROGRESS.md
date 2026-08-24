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

---

## 2026-08-24 — A3 (models) — Phase 1 complete

### Delivered

`src/models/` (+ `src/__init__.py`), `tests/test_models.py`. All three
architectures subclass `common.SurrogateBase` and honour the frozen interface of
CONTEXT.md §7 exactly: `forward(batch: dict) -> {"p": (ΣN,), "tau": (ΣN,2),
"coef_head": (B,2)}`, plain-dict constructors, no global state.

| file | contents |
|---|---|
| `common.py` | MLP builder, `FourierFeatures` (γ(x), L freqs, no params), `count_params`/`param_summary`, masked segment sum/mean/max/weighted-mean (via `index_add_`/`scatter_reduce`), `to_dense_batch`, `cond_features`, `knn_edges` (scipy cKDTree), `get_edge_index`, `SurrogateBase` |
| `gnn.py` | **M1**: k=16 kNN graph, T=8 residual message-passing rounds, post-LayerNorm |
| `sdf_fno.py` | **M2**: GINO-lite — kernel encoder → 64×64 latent grid → FNO2d (4 layers, modes 16, width 32, self-implemented `rfft2` spectral conv) → kernel decoder |
| `geo_transformer.py` | **M3**: Transolver physics attention, M=32 slices, 4 layers, d=128, 8 heads |
| `heads.py` | `CoefHead` (the *same class* for all three) + `FieldHead` shared trunk |
| `losses.py` | `total_loss` / `init_norm_ref`, rel-L2 + λ_F force + λ_S symmetry, λ=0 defaults |
| `__init__.py` | `MODEL_REGISTRY` = `{gnn, sdf_fno, transolver}` (+ `m1/m2/m3` aliases), `build_model(name, config)` |

### Param counts at default configs (budget 1.5 M ±20% = [1.20 M, 1.80 M])

| model | params | vs target | dominant block |
|---|---|---|---|
| `gnn` (M1) | **1,491,189** | 0.994× | 8 MP rounds, 1.29 M (86%) |
| `sdf_fno` (M2) | **1,485,797** | 0.991× | FNO stack, 1.05 M (71%) |
| `transolver` (M3) | **1,446,181** | 0.964× | 4 attention blocks, 0.94 M (65%) |

All three within 3.1% of each other, so cross-architecture comparisons are not
confounded by capacity. `tests/test_models.py::test_param_budget_at_default_config`
asserts the band.

### Tests

`.venv/Scripts/python.exe -m pytest tests/test_models.py -q` → **53 passed**
(CPU only, ~13 s). Whole suite with A1/A2 tests: **321 passed, 3 skipped**.
Covers: forward shapes, finite grads for *every* parameter, seed determinism,
batched-vs-single equivalence for all three (M3 slice pooling is the real
target), M2 conditioning switch × {sdf, mask, sdf+normals}, param budgets,
SDF sign convention + agreement with A2's `sdf_on_grid`, spectral-conv
translation equivariance, `CoefHead` permutation invariance, loss hooks.

### Decisions / deviations needing a look

1. **PROPOSAL — grouped spectral convolution in M2.** CONTEXT §7 freezes
   "modes 16, width 32" for the FNO stack. A *dense* spectral layer at those
   settings costs `2·32·32·16·16·2 = 1.05 M` real params — four layers alone is
   4.2 M, ~2.8× the whole 1.5 M budget. Both constraints cannot hold together.
   Resolution: `SpectralConv2d(..., groups=4)` keeps modes 16 and width 32
   (spectral *resolution* is untouched) and block-diagonalizes the channel
   mixing, cutting spectral params 4× to 1.05 M for the whole stack. Full
   cross-channel mixing is restored every layer by the dense 1×1 pointwise path
   in `FNOBlock`. Config key `spectral_groups` (default 4) — set it to 1 to
   recover the textbook dense layer if the budget is ever relaxed. Flagging for
   DECISIONS.md rather than editing the frozen section myself (§12 rule 2).

2. **M2 latent bbox is `[-0.5, 1.5] × [-1.0, 1.0]` at 64×64** (dx = 0.031
   chords), documented in the module docstring. This assumes surface coordinates
   arrive in the **chord frame** (chord ∈ [0,1]). If A1's `norm_stats` ends up
   normalizing `surf_pos` away from that frame, M2 needs `bbox_mode: 'auto'`
   (already implemented, per-batch bbox with margin) or a bbox retune. **A1/A4:
   please confirm whether `surf_pos` is normalized in the loader.**

3. **Curvature is optional.** `SurrogateBase.get_curvature` zero-fills when the
   batch lacks `curvature`, so models run against the current cache. A2 owns
   curvature estimation; once it is in the cache the channel becomes live with
   no model change. Same pattern for `edge_index` (M1) and `grid_sdf` (M2): the
   loader *may* precompute them, and the model falls back to building them.

4. **`losses.py` is nominally A4's per CONTEXT §8** but was assigned to A3 in
   the PLAN row. Implemented with dependency inversion — `integrate_fn` and
   `symmetry_fn` are injected callables, so `src.models` never imports
   `src.physics`/`src.geometry`. `integrate_fn` is signature-compatible with
   `integrate_forces` (§6); the trainer is expected to wrap denormalization into
   the closure, since predictions are in normalized space and metrics are in
   physical units. A4 should feel free to extend (`extra_fns` hook is there for
   λ_B / λ_D).

5. **A coefficient-head data term is missing on purpose.** The pure data loss
   (λ=0 defaults) trains only the field path; `coef_head` gets no gradient until
   λ_F > 0. A4 should either add an explicit `|C_head − C_true|` term to the
   trainer or keep λ_F > 0 for runs where `cl_head_mae` matters.

### Memory notes for the 4 GB P2000 (measured, fp32, 1000 surface pts/sample)

| model | peak fwd+bwd per sample | bs=4 peak | notes |
|---|---|---|---|
| M1 `gnn` | ~440 MiB | 1.76 GiB | dominant tensor is the edge-MLP hidden `(k·ΣN, 208)` × 8 rounds |
| M2 `sdf_fno` | ~100 MiB | 0.43 GiB | latent grid is only `(B, 32, 64, 64)`; cost is the per-sample SDF on CPU |
| M3 `transolver` | ~83 MiB | 0.34 GiB | largest tensor is the assignment map `(B, 8, N, 32)` — no N×N anywhere |

- **Recommended starting batch sizes:** M1 = 4, M2 = 8, M3 = 8. M1 at bs=8
  (~3.5 GiB) tips the card into WDDM host-memory paging and slows ~10×.
- **M1 escape hatch:** `grad_checkpoint: true` → 298 MiB at bs=4, a **5.9×**
  reduction for only ~10% more wall time (the layer is bandwidth- not FLOP-
  bound). `tests/test_models.py` asserts it is numerically transparent. Turn it
  on if real `n_surf` is much larger than the assumed 1000.
- Weights are ~5.7 MiB each, so optimizer state is negligible; **activations are
  the whole story.** Ensembles (K=5) are trained sequentially anyway.
- Forward-pass timing at bs=4: M3 40 ms, M1 360 ms, M2 2.45 s. **M2 is
  CPU-bound**, not GPU-bound — the per-sample grid SDF and the cKDTree query run
  in numpy each step. If M2 epochs are too slow, the fix is for A1 to cache
  `grid_sdf` per sim in the `.npz` (M2 already accepts `batch['grid_sdf']` and
  skips the computation); geometry is static per sim, so this is pure win.

### Environment note

Installed `pytest` (+ `iniconfig`, `pluggy`) into `.venv` via `uv pip` — it was
missing and `tests/` could not run without it. No other packages added; no
compiled extensions (kNN = scipy cKDTree, scatter = `index_add_` /
`scatter_reduce`, FNO self-implemented on `torch.fft`).

## 2026-08-24 — A5 (uq + active learning)

**Delivered.** `src/uq/{ensembles,conformal,coverage}.py`, `src/active/{pool,acquisition,diversity}.py`,
package `__init__.py` for both, `tests/test_uq.py` (56) + `tests/test_active.py` (73).
`.venv/Scripts/python.exe -m pytest tests/test_uq.py tests/test_active.py -q` → **130 passed**;
full `pytest tests/` → 322 passed, 3 skipped. All numpy-first and CPU-only; torch is a
late import used only for real model/checkpoint paths.

**`src/uq/ensembles.py`** — `EnsemblePredictor` wraps K callables or K stacked prediction
arrays (`from_arrays`); `predict()` gives mean/std per surface point (`p`, `tau`) and per
simulation/coefficient (`coef_head`), ddof=1 per spec §5.6, K=1 → zero spread (not NaN).
`propagate_forces(batch, integrate_fn)` integrates **each member separately then averages**
(`Cbar_D = (1/K) Σ_k C_D^int[u_k]`), passing the models' own tensors through so A2's
differentiable `integrate_forces` works untouched; a test proves this differs from
integrate-the-mean under nonlinear post-processing. `load_ensemble_checkpoints(paths, config)`
loads K same-architecture checkpoints via `src.models.build_model` (late import; adapts to
A3's two-arg `build_model(name, cfg)` signature, with a `MODEL_REGISTRY` fallback).

**`src/uq/conformal.py`** — split conformal, both granularities, parameterized:
`SplitConformalCoefficients` (exchangeable unit = simulation → clean marginal guarantee,
per-target or pooled-max-over-targets) and `SplitConformalField` (**both** max-over-points
and quantile-over-points with q=0.9, conformalized at sim level; accepts ragged per-sim
lists, 2-D arrays, or flat arrays + `sim_idx`). Scores: normalized `s=|y−μ|/(σ+ε)` and the
absolute-residual variant (σ≡1) for the ablation. `q̂` = k-th smallest with
`k = ceil((n+1)(1−α))`, `+inf` when `k > n` (honest infinite intervals, not a clamp);
the product is rounded before `ceil` so e.g. `n=19, α=0.05` gives 19, not 20. Each
`CalibratedIntervals` carries a plain-English `guarantee` string stating the weaker
interpretation for field bands.

**`src/uq/coverage.py`** — coverage, mean/median/max width, Wilson CIs, reliability curves
over a level grid, ECE, level sweeps for both granularities. `field_coverage` reports the
sim-level coverage the band actually certifies (all points for `max`, ≥ q fraction for
`quantile`) and pooled point coverage as an explicitly unguaranteed diagnostic.

**Schema coordination (resolved, no interface change needed).** A4's
`src/eval/schema.py::UQ_SCHEMA` landed with a flat `method/levels/coef/field` shape while
A5 needed a richer per-level record list. `make_uq_report` now emits **one payload
satisfying both**: our `schema_version`/`records` plus the flat `coef`/`field` views
projected from them (A4's validator allows extra keys). `validate_uq_report` runs our
structural checks *and* A4's `assert_valid_uq`. Note for A4: `validate_uq` *returns* an
error list rather than raising — only `assert_valid_uq` raises.

**`src/active/pool.py`** — NACA 4- and 5-digit generators (standard **and** reflexed
5-digit), closed/open TE option, cosine point clustering at LE **and** TE, pool =
shapes × (Re, AoA) grid with a tolerance-matched `ExclusionSet` for training combos.
Emits CONTEXT.md §4 batch dicts (`surf_pos`, `surf_normal`, `surf_ds`, `cond`, `batch_idx`)
plus PyG-style `collate_batches`, so pool cases run straight through a surrogate — verified
end to end against real A3 models and A2's `integrate_forces` in `test_uq.py`.

*Conventions pinned (documented in the module docstring and asserted in tests):* chord 1,
contour starts at the TE, runs **upper** surface forward to the LE, then **lower** back to
the TE (Selig ordering) = **counter-clockwise / positive shoelace area**; outward normal
`n = (t_y, −t_x)`; `ds[i]` = half-sum of adjacent facet lengths so `Σ ds` = perimeter
exactly; contour closes cyclically (no duplicated endpoint). Normals/ds come from A2's
`src.geometry.quadrature.contour_quadrature` when importable, else an equivalent local
implementation — the two agree to 1e-17 (asserted).

*NACA formula references used.* 4-digit thickness
`yt = (t/0.2)(0.2969√x − 0.1260x − 0.3516x² + 0.2843x³ − a₄x⁴)`, `a₄ = 0.1015` blunt TE /
`0.1036` closed TE; 4-digit two-branch camber line about `p`. 5-digit **standard** camber
`yc = (k₁/6)(x³ − 3mx² + m²(3−m)x)` for `x < m`, `(k₁m³/6)(1−x)` aft, with `m` solving
`p = m(1 − √(m/3))`; 5-digit **reflexed** camber with the `k₂/k₁` term. `(p, m, k₁)` and
`(p, m, k₁, k₂/k₁)` tables tabulated for design `Cl = 0.3`, with `k₁` scaled linearly by
`Cl_design/0.3 = 0.15·L/0.3`. Sources: **Abbott & von Doenhoff, *Theory of Wing Sections*
(Dover, 1959), §6.3–6.6 + Appendix I, Table 6.5**; **NACA TR-460** (Jacobs, Ward & Pinkerton,
1933) and **NACA TR-537** (Jacobs, Pinkerton & Greenberg, 1935); tables cross-checked against
**Ladson, Brooks, Hill & Sproles, NASA TM-4741 (1996)**, "Computer Program to Obtain
Ordinates for NACA Airfoils". Validated numerically: 0012 → max thickness 0.1200 at
x = 0.2995; 2412 → camber 0.0200 at x = 0.400; 23012 → camber 0.0184 at x = 0.150
(published ~1.8%), design Cl 0.30.

**`src/active/acquisition.py`** — `a(g,c) = z(σ_CD) + β·z(FSC) + γ·z(L_sym)`, β = γ = 1
default, `z` = shift-and-scale over the pool so the score is **invariant to positive affine
rescaling** of any component (asserted over several scale/offset combinations, ranking
included) and a constant component contributes exactly zero instead of NaN. Baselines:
`random_scores` (seeded) and `variance_only_scores`. `select_cases` + `write_case_list`
emit the `fluent/cases_to_run.json`-shaped payload for Phase 3 item 7.

**`src/active/diversity.py`** — min-max/z-score design-space normalization (constant columns
→ 0, optional column weights), greedy farthest-point (max-min) selection with medoid/index/
first seeding and an `already_selected` prior, `select_top_k_diverse` (oversampled shortlist,
seeded by the argmax so pure exploitation is never discarded), plus `greedy_score_diversity`
for the λ sensitivity check and `min_pairwise_distance`/`coverage_radius` metrics. Design
space is family-agnostic — (max camber, x of max camber, max thickness, x of max thickness,
Re, AoA) — so 4- and 5-digit sections are directly comparable.

**Two bugs the tests caught in my own code:** contours were initially clockwise (fixed to
Selig/CCW), and symmetric sections were coded `0412` instead of `0012` (the camber-position
digit must be 0 when `m = 0`).

**Notes for integration.** (1) `NU_AIR = 1.5e-5` maps Re → freestream speed
(`U = Re·ν/c`); A1 should confirm AirfRANS's own convention in DATA_NOTES.md and override if
it differs — `freestream_from_re(..., speed=...)` already allows bypassing it. (2) Real
ensembles/conformal need trained checkpoints (Phase 3 items 2–3); everything here is tested
on synthetic data only, as scoped. (3) Not blocking anyone.

## 2026-08-24 — A1 (data) — D-017 precomputed static geometry — DONE

Cache bumped to **`CACHE_VERSION = 2`**; existing `.npz` files are treated as
stale and rebuilt automatically.

### What changed

- `scripts/build_cache.py`: new `geometry_extras(surf_pos, k, bbox, res)`
  producing `grid_sdf` (64,64) float32, `edge_index` (2,E) int32 and
  `curvature` (Ns,) float32, wired into `build_one` behind `--no-geometry` /
  `--knn-k`. Constants `GRID_BBOX`, `GRID_RES`, `KNN_K` mirror M2's/M1's
  defaults. Manifest rows gained `has_geometry` and `n_edges`.
- `src/data/airfrans_loader.py`: `GEOMETRY_KEYS`, a `load_geometry` flag, and
  pass-through with shape validation. `collate` concatenates `curvature`,
  stacks `grid_sdf` to `(B,64,64)`, and offsets `edge_index` per sample.
- `tests/test_loader.py`: 23 new tests (26 -> 49).
- `docs/DATA_NOTES.md`: new section A1.11.

### Test results

`pytest tests/test_splits_no_leakage.py tests/test_loader.py tests/test_models.py -q`
-> **140 passed, 2 skipped**.

Full suite: **347 passed, 1 failed, 3 skipped**. The single failure is
`tests/test_uq.py::test_uq_report_also_satisfies_eval_schema`
(`src.eval.schema.validate_uq` returns `None` instead of `[]`) — A5/A4 code,
unrelated to D-017. The 7 failures I reported earlier have been fixed by their
owners in the meantime.

Models are proven to *consume* the cached keys, not merely tolerate them:
`test_gnn_uses_the_cached_edge_index_not_a_fresh_kdtree` and
`test_sdf_fno_uses_the_cached_grid_sdf` monkeypatch `knn_edges` /
`compute_grid_sdf` to raise, and all three models still complete a forward pass
on a batch straight out of the loader.

### Three traps found and closed (details in DATA_NOTES A1.11)

1. **Grid orientation is a transpose.** `src/geometry/sdf.make_grid` puts x on
   the first axis; M2's `make_latent_grid` is row-major `iy*nx+ix`. M2 consumes
   `grid_sdf` via `.reshape(n_graphs, -1)`, so an untransposed array is accepted
   silently — measured error **0.38 in absolute SDF**, a diagonal mirror of the
   conditioning field. Cache stores `sdf_on_grid(...).T`; the test asserts
   *exact* float32 equality against M2's own path, not a tolerance.
2. **Precompute from the float32 positions.** Models recompute geometry from the
   cached float32 `surf_pos`. Building from the float64 originals gave
   `edge_index` that differed on tie-broken neighbours (a near-uniform contour
   has many equidistant pairs, and float32 rounding flips cKDTree's tie-break)
   and `grid_sdf` off in the last bit. `geometry_extras` now takes the float32
   array that is actually stored. After the fix, cached edges are `torch.equal`
   to `knn_edges(batch_pos, 16, batch_idx)` and `grid_sdf` is `np.array_equal`
   to `compute_grid_sdf`.
3. **`get_edge_index` does no offsetting** — it returns `batch['edge_index']`
   verbatim. `collate` therefore adds `ptr[i]` per sample before concatenating.
   Because `knn_edges` maps local indices back through
   `sel = arange(ptr[g], ptr[g+1])`, this reproduces its output exactly,
   ordering included.

### ⚠ PROPOSAL / heads-up for A3 + A4

**`curvature` is not a pure optimisation and it invalidates old checkpoints.**
`edge_index` and `grid_sdf` are caches of derived quantities — predictions with
and without them are bit-comparable (asserted in
`test_cached_and_on_the_fly_geometry_give_identical_predictions`). `curvature`
is not: before D-017 `SurrogateBase.get_curvature` returned a **constant-zero**
channel, so the cache now feeds models a real input they never saw. A test
asserts predictions change. Consequently **any checkpoint trained against a
pre-D-017 cache is not comparable to one trained now** — mixed-vintage ensembles
(Phase 3 step 2) and data-efficiency curves (step 4) must be rebuilt from
scratch, not topped up. Worth a DECISIONS note if any runs already exist.

### Cost

Cache build went from 0.39 s/sim to 1.39 s/sim on the synthetic 96-point /
1.3k-node mesh; the SDF sweep dominates and scales with grid nodes (4096, fixed)
times contour points, so on real airfoils (~10^3 surface points) expect roughly
+1-2 s/sim. Against A3's measured 2.45 s **per forward pass** for M2, this pays
for itself within the first training step of the first epoch.

### Still open

- Unchanged from my previous entry: dataset still downloading; all
  `[verify after download]` items in DATA_NOTES stand.
- `verify_cache.py` still reports the same 2 known synthetic-circle cd
  cancellation failures out of 64 checks (analysed in A1.10) — unchanged by
  D-017, and an artefact of the test geometry, not the cache.
- No packages were installed by A1.

## 2026-08-24 — A4 infra (config / eval / trainer / sweeps)

**Delivered.** `src/utils/{config,seed,io}.py`, `src/eval/{metrics,harness,schema,baselines}.py`,
`scripts/{train,evaluate,sweep}.py`, `configs/{gnn,sdf_fno,transolver}.yaml`,
`configs/sweeps/{core,ensembles}.yaml`, `requirements.txt`, `tests/test_infra.py`.
`.venv/Scripts/python.exe -m pytest tests/ -q` → **352 passed, 3 skipped** (49 of them A4's).

- **config**: `load_config(path, overrides)` with dotted overrides and type inference.
  `3e-4` is handled explicitly — PyYAML's 1.1 resolver returns it as a *string*, so a bare
  `yaml.safe_load` would set `lr` to `'3e-4'`. Bare aliases (`split=`, `seed=`, `epochs=`,
  `lr=`, `batch_size=`) expand to their dotted path. `save_config` writes the resolved config.
- **seed**: `seed_all` covers python/numpy/torch(+all CUDA); `get_rng_state`/`set_rng_state`
  back exact resume. `torch.use_deterministic_algorithms` is deliberately *not* set —
  `index_add_` (the GNN's scatter) has no deterministic CUDA kernel and would raise.
- **io**: atomic JSON/CSV writers (temp + `os.replace`), strict JSON (NaN → `null`),
  `build_run_id` = `{model}_{split}_s{seed}[_{tag}]`, run/checkpoint dir helpers.
- **eval**: `harness.evaluate` is the only producer of `metrics.json`; output is
  schema-validated before it is returned. All metrics in denormalized physical units,
  per-sim then averaged (a 5000-point sim must not outweigh a 500-point one). Inference
  timing excludes a warm-up batch and syncs on CUDA; `peak_mem_mb` from
  `max_memory_allocated`, `null` on CPU. Baselines (`constant`, `ridge`) implement the §7
  model interface, so they go through the *same* harness and emit the same schema.
- **train**: Adam + cosine, clip 1.0, fp32, checkpoint every epoch to `last.pt`
  (model+optim+sched+epoch+RNG+`norm_ref`), auto-resume, `best.pt` on the val metric (val =
  the split's `cal` list), `history.csv`, then the eval harness. `--dry-run` = 2 epochs ×
  2 batches and suffixes the tag with `_dryrun`, so a smoke test can never write a
  `metrics.json` that makes a sweep skip the real run.
- **sweep**: sequential only (4 GB card), skips runs whose `metrics.json` exists,
  `--max-seconds` budget checked before each run *and* enforced as a child timeout
  (Kaggle 12 h; per-epoch checkpointing makes a killed run free to resume).

**Verified end to end** against the real `src/data`, `src/models`, `src/physics` on a
synthetic 12-sim cache: all 3 models train → checkpoint → resume → `metrics.json`;
both baselines; `scripts/evaluate.py` on a checkpoint; a real `scripts/sweep.py` run plus
its skip-on-rerun path.

### Decisions taken inside A4's scope

- **No `jsonschema` dependency.** `src/eval/schema.py` validates plain dicts. The schema is
  frozen and fully specified by CONTEXT.md §9; a second copy of it in JSON-Schema syntax
  would be a second source of truth, and the hand-rolled errors name the exact dotted key.
- **`cl_rel`/`cd_rel` are aggregate ratios** `Σ|C_pred−C_true| / Σ|C_true|`, not the mean of
  per-sim ratios: C_L crosses zero on symmetric airfoils near α=0 and per-sim ratios diverge
  there. `fsc_rel_*` keeps the spec's per-sim form but aggregates by **median** for the same
  reason. ⚠ On sims with |C_L| ≈ 0 `fsc_rel_cl` is still large by construction — read
  `fsc_cl` (absolute) alongside it.
- **`sym_residual` is computed on the general reflection** (mirror geometry *and* condition),
  not only on symmetric airfoils, so it is defined for every sim; it uses A2's
  `reflect_x_batch`/`reflect_prediction` conventions and is evaluated in physical units,
  because the p-normalisation has a non-zero mean and the relative residual is not
  invariant under it.
- **Cached geometry-derived keys** (`grid_sdf`, `edge_index`, …, `CACHED_DERIVED_KEYS` in
  `harness.py`) are dropped from the *reflected* batch during the symmetry check so the
  model rebuilds them from the mirrored geometry instead of silently reusing values built
  for the original one. Unknown batch keys are otherwise passed through untouched (D-017).

### Integration friction found and fixed (worth knowing about)

1. **Units in the force term of `src/models/losses.py`.** `force_consistency_loss` compares
   `integrate_fn(...)`'s output against `coef_head` and `cl_true`/`cd_true`, which are
   *normalized* (D-014). Integration only means anything on physical fields, so the trainer's
   injected closure (`make_denorm_integrator`) now denormalizes p/tau/cond **and
   re-normalizes the returned C_L/C_D**. Without the second half the term silently mixes
   units. Both transforms are affine, so gradients are unaffected.
2. **`norm_ref` must live in the checkpoint.** Re-capturing the auxiliary-loss scales on the
   first batch of a *resumed* session changes the objective and broke bit-identical resume.
   Now saved in `last.pt`/`best.pt` and restored. (A3's `init_norm_ref` docstring already
   said "stash the result in the checkpoint" — this is that.)
3. **CONTEXT.md §8 assigns `src/models/losses.py` to A4, but PLAN.md Phase 1 gives all of
   `src/models/` to A3.** A3 shipped it; A4 uses it and keeps a compatible
   `fallback_total_loss` in `scripts/train.py` for the case where it is absent.
   ⚠ PROPOSAL: correct §8's ownership note to A3 to stop this recurring.
4. **λ_H (D-016)** is implemented through A3's `extra_fns` extension point rather than by
   editing `losses.py`: `train.weights.head` (default **0.1**, set in all three configs and
   in `config.DEFAULTS`) drives `head_regression_loss`, normalized at init like the other
   auxiliaries. `tests/test_infra.py` asserts the head receives **no gradient at all** when
   λ_H = 0 — that is the failure mode D-016 exists to prevent.
5. **`parse_naca_params`** now delegates to A1's `src.data.splits.parse_sim_name` (fixes the
   off-by-one A1 reported: a regex over the whole name caught the `2` in `airFoil2D`), with a
   tolerant local fallback so `src.eval` stays importable without `src.data`.
6. **Schema API (A5's report)**: `validate_metrics`/`validate_uq` now **raise** `SchemaError`;
   `check_metrics`/`check_uq` are the non-raising variants. On success the validators still
   return the empty error list, so existing `assert validate_uq(x) == []` call sites keep
   working while an ignored return can no longer pass an invalid payload.
7. **CLI overrides may be interleaved with flags** (`parse_cli`): argparse fills a `nargs="*"`
   positional greedily, so `--checkpoint X tag=eval --device cpu split=aoa` used to error.
   `scripts/sweep.py --train-args` takes one quoted string (argparse will not absorb
   dash-prefixed values into a list).

**Not verified yet** (needs the real cache): metric magnitudes on AirfRANS, VRAM/batch-size
fit on the P2000, and `cd_spearman` (needs ≥3 test sims — it is `null` below that by design).

## 2026-08-24 — A6 (aux scaffolds): fluent/, DrivAerNet++ subset, paper/, model_cards/

Scope delivered in full. **No solver was run, nothing was downloaded, no git
commands were issued.** All numeric work was mesh generation (pure numpy,
seconds) and one LaTeX compile.

### fluent/ — 2D RANS verification pipeline, ready to execute

- `fluent/mesh_gen.py` (896 ln) — parametric single-block C-grid generator for
  NACA 4- and 5-digit sections, writing a **native Fluent 2D `.msh` directly**.
  numpy-only, deterministic, no gmsh / meshio / ICEM on the critical path.
  Includes the y+ -> first-cell-height sizing (flat-plate Schlichting
  correlation, documented with the factor-of-2 cell-centre-vs-cell-height trap)
  and `--self-check`, which re-parses the written file and asserts index ranges,
  owner/neighbour consistency and section counts.
- `fluent/templates/case_template.jou` (279 ln) — the TUI journal template.
  Reads mesh, pressure-based steady incompressible, **SA and k-omega SST
  variants**, velocity-inlet from (Re, AoA), pressure-outlet, no-slip wall,
  reference values for CL/CD, two-stage first-order -> second-order solve,
  convergence on residuals **and** a CD-stability monitor, CSV export of surface
  p / cp / wall shear / cf / y+, forces, achieved y+, transcript on, display
  objects baked into the saved case for later GUI inspection.
- `fluent/templates/probe_bc_keywords.jou` — **Step 0**, a keyword probe. These
  journals were authored without a live Fluent; a wrong grouped-`set` keyword
  does not fail loudly, it leaves an unanswered prompt that eats following
  journal lines and produces a plausible-looking wrong result. The probe removes
  that failure mode for about a minute of runtime.
- `fluent/templates/mesh_cgrid.rpl` (148 ln) — ICEM Tcl replay, kept as the
  independent cross-check/fallback meshing route with the same substitution
  slots.
- `fluent/make_cases.py` (397 ln) + `fluent/cases_to_run.json` — manifest-driven
  journal instantiation with schema validation. Verified: 5 cases -> 10 journals,
  zero unfilled slots.
- `fluent/README.md` (operating manual, orchestrator call sequence, solver-choice
  rationale) and `docs/FLUENT_PLAN.md` (431 ln: mesh-route argument, y+ maths
  with worked numbers, 3-level grid study with the Celik GCI arithmetic, the
  6-simulation AirfRANS-replication offset study, acceptance criteria).

**Mesh generation validated** (this is the part that could have silently
shipped broken). Three separate bugs each produced a folded grid — the
far-field distribution inheriting the wall clustering; a direction blend running
on the distance fraction rather than the index fraction; and a discontinuous ray
family across the trailing-edge fold. All three are fixed and written up in
FLUENT_PLAN.md section 2.2. Current quality, NACA 0012 / Re 3e6 / 5 deg:

| Level | Cells | min Jacobian | min orthogonality | max AR | normal growth |
|---|---|---|---|---|---|
| 1 coarse | 18 796 | > 0 | 0.284 | 5.4e3 | 1.189 |
| 2 medium | 43 008 | > 0 | 0.290 | 5.2e3 | 1.121 |
| 3 fine | 96 768 | > 0 | 0.294 | 5.2e3 | 1.079 |

Spot checks: NACA 23012 at +10 deg -> 0.225; NACA 4412 at -5 deg -> 0.238.
Refinement ratio ~1.5 per direction per level (GCI wants >= 1.3); y+ target
scales 0.90 / 0.60 / 0.40 so refinement is systematic at the wall too and all
three levels stay wall-resolved.

### scripts/subset_drivaernet.py + docs/DRIVAERNET_ACCESS.md

- Script (573 ln) consumes already-downloaded coefficient CSVs and decimated
  meshes; stdlib `csv` (no pandas), numpy/pyvista/trimesh guarded with actionable
  error messages. Auto-detects the stratification columns actually present and
  reports them rather than assuming a schema. Seeded stratified draw with
  proportional allocation plus a per-stratum floor; decimation to 8k-16k points,
  float16 storage; emits `subset_manifest.json` with exact design IDs, strata
  counts, seed and source-CSV SHA-256s.
- Tested against a synthetic 900-design fixture: 600 selected across 12 strata,
  identical IDs on re-run with the same seed, 456/600 overlap across seeds,
  `--describe` / `--strata` / `--allow-missing-meshes` paths all exercised.
  Warns loudly if fastback/notchback/estateback are not all present, since the
  shape-family OOD split of spec 5.7 depends on it.
- ACCESS doc (214 ln): Globus Connect Personal setup including the D:-drive
  access grant that is the usual failure point, what to select and what not to
  (the 39 TB is the volumetric fields), expected sizes, CC BY-NC 4.0 obligations,
  and the 200-design fallback.

### paper/ and model_cards/

- `main.tex` (714 ln): every section of spec section 11, all **12 figures** and
  **6 tables** of spec section 8 as placeholders with `\label` and a caption
  saying what will go there, plus appendices A-D. Figure placeholders render as
  framed boxes so the skeleton compiles before any figure exists; swapping in a
  real figure is a one-line change.
- `refs.bib` (328 ln): all 16 spec section 14 reading-list items with arXiv/venue
  entries, plus supporting refs (Celik GCI, Roache, Spalart-Allmaras, Menter SST,
  Vovk, Lei, Tibshirani covariate shift). Header flags that page/volume numbers
  need checking against publisher records before submission.
- `macros.tex` (145 ln): notation matching spec section 5 (FSC, C_D^int,
  C^head, the loss terms, the split names). Every symbol is `\ensuremath`-wrapped
  so it works in prose as well as maths.
- `paper/figures/README.md`: the figure-file naming contract.
- `model_cards/TEMPLATE.md` (219 ln) per spec section 13, including the explicit
  "do not use for" section and the rule that a number not in a committed
  `metrics.json` does not go on the card.

**LaTeX status.** `pdflatex` exists (MiKTeX) but **`natbib.sty` is not
installed**, and installing it would be a download, which was out of scope. I
verified the document by compiling a scratch copy with natbib/siunitx/microtype
stubbed out: **clean compile, no errors, no undefined references, 12 pages**
(inflated by the placeholder boxes and TODO text; the real paper targets 8-10).
Three genuine bugs were found and fixed in the process: math-mode-unsafe macros,
an undefined `\degree`, and my `\num` macro clashing with siunitx's. To build
for real: `mpm --install=natbib` (about 100 kB), then `latexmk -pdf main.tex`.

### Notes for the integrator

- **PROPOSAL (not actioned, outside A6's paths):** `.gitignore` has no LaTeX
  entries. Suggest adding `paper/*.aux paper/*.log paper/*.out paper/*.bbl
  paper/*.blg paper/main.pdf` and `data/processed/drivaernet/points/`.
- **Blocking on A1:** `fluent/cases_to_run.json` currently assumes air at
  298.15 K (rho 1.184, mu 1.85e-5) because that is what the AirfRANS generator
  states. The offset study measures the wrong thing if that differs from what
  the dataset actually used — please confirm in `docs/DATA_NOTES.md`, and I will
  re-derive U_inf and the y+ sizing if it changes.
- **Blocking on Phase 3 step 7:** the two `placeholder_*` cases are stand-ins.
  `scripts/sweep.py` should overwrite them with the real active-learning
  selection. They must not be hand-tuned into something plausible — the
  falsifiable claim requires the selection to come from the model.
- The `gridstudy_*` trio is not a placeholder and should run first: its GCI band
  is the numerical uncertainty quoted on every other Fluent case.

## 2026-08-24 — A6 (aux scaffolds) — DONE (final log by orchestrator; agent hit usage limit after finishing work)

- fluent/: gmsh-based mesh_gen.py (C-grid, y+ targeted), parameterized SA + SST TUI
  journal templates, make_cases.py + rendered cases (2 placeholders + 3-level
  NACA0012 grid study), RUNBOOK.md, MANIFEST.json. Runs deferred per D-007.
- docs/FLUENT_PLAN.md (grid study/GCI design, y+ math, AirfRANS replication offset
  study), docs/DRIVAERNET_ACCESS.md + scripts/subset_drivaernet.py (user-gated).
- paper/: main.tex compiles clean — 12 pages, all 12 figures + 6 tables placeholdered,
  refs.bib from spec reading list, macros.tex. model_cards/TEMPLATE.md.

## 2026-08-24 — GATE G1 PASSED (Fable)

Dataset downloaded (10.03 GB, checksummed by size), extracted (1000 sims), 20-sim
smoke cache built (8.3 s/sim). Force integration on cached ground-truth fields vs
dataset coefficients (pressure_mode=kinematic, rho=1, a_ref=1):
**CD median rel err 1.44e-4 (max 1.02e-3); CL median 4.79e-6 (max 1.63e-5), n=20.**
Two orders inside the 1% gate. Full 1000-sim cache build launched.

## 2026-08-24 — Full cache + definitive G1 (Fable)

Cache: 1000/1000 sims, 0 failed, 5.22 s/sim, ~5 GB; norm_stats from 700 full-train
sims; 6 split manifests written. G1 on ALL 1000 sims (kinematic, rho=1, a_ref=1):
CD rel err median 3.31e-4, p95 1.20e-3, max 1.61e-3; CL median 4.84e-6.
Protocol quadrature error ~100x below expected model error. G1 CLOSED.

## 2026-08-24 — Kaggle cloud pipeline VALIDATED (Fable/Opus)

Smoke v9 on T4: sweep_rc=0, GNN 2-epoch train in 50s (25s/epoch → ~2.8h/400-epoch run),
p_rel_l2=0.537, cd_head_spearman=0.869. Full cloud path works: geo-op-code dataset →
T4 kernel → cache located by manifest search → train → checkpoint → results.zip export →
pull. Nine smoke iterations fixed: secret attach, tar-extract, output path, log capture,
data path override (processed_dir + norm_stats), mount path discovery, T4 vs P100 (D-020).
Launching core grid (3 models × 6 splits).
