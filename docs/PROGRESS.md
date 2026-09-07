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
  rationale) and `docs/plans/FLUENT_PLAN.md` (431 ln: mesh-route argument, y+ maths
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

### scripts/subset_drivaernet.py + docs/plans/DRIVAERNET_ACCESS.md

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
- docs/plans/FLUENT_PLAN.md (grid study/GCI design, y+ math, AirfRANS replication offset
  study), docs/plans/DRIVAERNET_ACCESS.md + scripts/subset_drivaernet.py (user-gated).
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

## 2026-08-25 — Core grid session 1 results (Fable/Opus)

Session 1 (T4, ~11h) completed 5 GNN runs @400ep + partial 6th (resumes). First real
neural numbers, per-split train-only norm (D-021) applied:
| split | p_relL2 | cd_head_sp | fsc_cd | sym |
|---|---|---|---|---|
| full | 0.074 | 0.999 | 0.0053 | 4.49 |
| scarce | 0.154 | 0.996 | 0.0086 | 5.02 |
| reynolds | 0.223 | 0.977 | 0.0164 | 7.22 |
| aoa | 0.188 | 0.945 | 0.0104 | 8.21 |
| shape5 | 0.130 | 0.983 | 0.0089 | 6.76 |

Findings: OOD hierarchy full<shape5<scarce<aoa<reynolds; **FSC grows with shift** (C1
label-free OOD detector, visible in model 1); GNN cd_spearman beats ridge (0.83-0.90)
on every split. Fig4/5/12 + Tables 1-2 regenerated with real data. Session 2 launched
with --runs-dataset resume (fixed path): finishes combined + transolver + sdf_fno.

## 2026-08-25 — Paper draft: sections needing only the core grid (Sonnet, PLAN_PHASE2 Phase E/1b)

Drafted `paper/main.tex` abstract, introduction, related work (cleanup only, content
was already complete), problem setup (§5: G1 number filled in, $\Lossbc$/$\Lossdiv$
added, shift-magnitude $\shift$ subsection written against the real
`src/viz/shift.py` definition with computed per-split means), models and training
(§6: matched-budget paragraph, reimplementation note, training details), datasets and
splits (§7: split paragraph with real train/cal/test counts, Table 6 datacard
populated from `data/splits/*.json`, DrivAerNet++ reframed as not-yet-run per D-005),
results in-distribution + OOD + consistency (§8.1-8.3: prose against Tables 1-2 and
Figs 4-5, which are `\input`/`\includegraphics`d from the committed
`paper/tables/tab{1,2}_*.tex` and `paper/figures/fig{04,05}_*.pdf` rather than
hand-copied; Fig 12 cost-accuracy also activated since it needs only the core grid),
and limitations (single-seed-on-core-grid, DrivAerNet++/Fluent not run, compute
budget, aoa pre/post-stall breakout not computed — all flagged explicitly rather than
implied). Calibration, data-efficiency (Fig 9), ablations and active-learning stay
`\todo` stubs with correct figure/table numbers, per scope. One real finding surfaced
while writing: field-error hardest-single-axis ordering is NOT stable across
architectures (GNN finds `reynolds` harder than `aoa`; SDF-FNO/Transolver find the
reverse) — written up honestly rather than rounded into one clean ranking. Also: the
$\shift$ magnitude ordering (`shape5` largest at 1.00) does not track empirical
difficulty (`reynolds`/`aoa`, $\shift$=0.42/0.23, are the two hardest splits for every
model) — used as the "these models interpolate within a regime, not across one"
evidence. Citation check: every `\cite{}` key resolves in `refs.bib` and every
`refs.bib` entry is cited — no gaps. Compile check: `microtype` needed
`[expansion=false]` to work around a MiKTeX font-expansion fatal error in this
environment (unrelated to content); with that, `pdflatex`+`bibtex`+`pdflatex`×2
produces an 18-page PDF, zero undefined refs/citations, two minor over/underfull-box
warnings only. Did not run git per instructions.

## 2026-08-25 — Phase A UQ glue: run_uq.py + Figs 6–8 + Table 3 (Opus, PLAN_PHASE2 1a)

Built the missing glue that turns ensemble checkpoints into the C3 calibration results.
No training, no GPU training, no git, no Kaggle; inference on the local P2000. CPU-light
(single-process, `num_workers=0`, small batches) to coexist with the user Fluent job.

**`scripts/run_uq.py`** (new). Per `(model, split)`: reads `results/{model}_{split}_s0/
config.yaml`, recomputes per-split **train-only** norm stats (D-021) locally, loads the K
available `best.pt` via `src.uq.ensembles.load_ensemble_checkpoints`, runs inference over
cal+test, and reduces each ensemble to per-sim coefficient members (`cd_int`/`cl_int` via
the frozen `integrate_forces`, integrate-then-average per spec 5.6; `cd_head`/`cl_head`
from `coef_head`) and per-point field members (denormalized `p`). Each member is run once
per batch (`member_outputs(raw=True)`); member arrays are stored once and sliced for the
K-subset ablation (no re-inference). Fits split conformal on the cal set, sweeps levels
{0.8,0.9,0.95} on test, writes schema-valid `results/uq/*.json` via
`coverage.write_uq_report` (satisfies both `uq-1` and `src/eval/schema.py::UQ_SCHEMA`).
- Two calibration modes: **matched** (`{model}_{split}_k{K}.json`, cal = split's own cal)
  and **transfer** (`..._calfull.json`, cal = `full`'s cal, served *through* the split-X
  model with split-X normalization so inputs match how it trained). For `full` the two
  coincide by construction.
- Zero-GPU ablation knobs all folded in for free: K∈{1,3,5} (member subsets, one file
  each; K=1 → `absolute` score only since σ≡0), score `normalized`∨`absolute` (both fitted
  per file, distinguished by each record's `score`), field aggregation `max`∨`quantile`
  (granularities `field_max`/`field_quantile`).
- CLI: `--model/--split` or `--all` (default 3 models × {full,reynolds,aoa,shape5}),
  `--seeds`, `--modes`, `--levels`, `--k-subsets`, `--agg-q`, `--device`, path overrides.
  Config's Kaggle `processed_dir`/`norm_stats` are overridden to local paths.

**Figures** (`scripts/make_figures.py`): added **fig7** (coverage vs shift @0.9 for
`cd_int`; solid=transfer, dashed=matched, hline at nominal) and **fig8** (interval width
ID vs OOD, coef `cd_int` + field `p`, grouped bars per model, log-y). Rewrote **fig6** to
read `results/uq` correctly — it previously mixed all coefficient records into one zigzag;
now selects matched, largest-K, `cd_int`, preferred score, one reliability curve per
(model,split). All three gate cleanly (return None) on an empty `results/uq`.

**Table 3** (`scripts/make_tables.py`): `table3_calibration` — rows = model×split, cols =
matched coverage@{.8,.9,.95}, transfer coverage@.9, mean width@.9, ECE (LaTeX + MD).

**Tests** (`tests/test_run_uq.py`, new, CPU-only, 8 tests): pure record-assembly on a
controlled K=5 synthetic ensemble (matched coverage lands inside the Wilson CI of nominal;
coverage/width monotone in level), K=1→absolute-only, both field granularities/scores,
full-report schema roundtrip, and a genuine end-to-end through `infer_dataset` with a fake
3-member callable ensemble + the real `integrate_forces` → schema-valid JSON. `pytest
tests/test_uq.py tests/test_run_uq.py -q` → 65 passed. `run_uq --help` OK.

**Demo (K=1, seed-0 core checkpoints), committed to `results/uq/`:** 24 reports (3 models
× 4 splits × {matched,transfer}), all schema-valid. Figs 6/7/8 + Tab 3 regenerated from
them. The intended C3 story is already visible at K=1: on `full`, matched≈transfer≈nominal
(cd_int cov@.9 = 0.93/0.96/0.96 for GNN/SDF-FNO/Transolver); under shift, coverage falls
well below nominal (e.g. matched cd_int cov@.9: reynolds 0.70/0.62/0.46, aoa 0.79/0.62/0.48
for GNN/SDF-FNO/Transolver) — "conformal delivers nominal coverage only when exchangeable;
the guarantee breaks under shift." Transfer (cal=full) partially recovers on some splits.

**Waiting on real ensembles** (seeds 1–4 training on Kaggle): re-run
`run_uq --all --seeds 0,1,2,3,4` after `pull_results.py` merges them to get K∈{1,3,5} files
(σ>0 → `normalized` score activates, fig7/tab3 auto-pick the largest K and normalized
score). Everything downstream already handles K>1 — verified by the K=5 synthetic tests.
The single-member (K=1) demo uses the `absolute` conformal score throughout.

**Gaps / notes:** (1) matched↔transfer mode is carried in the `ensemble_id` suffix
(`_calfull`) and mirrored into a `mode` field in the report; figures/tables derive it from
the suffix rather than a new `src.viz.data` column (kept viz/data untouched). (2) Integrator
called with the harness defaults (`rho=1, a_ref=1, pressure_mode="static"`) so `cd_int`
matches the committed `metrics.json`. (3) Did not run git.

---

## 2026-09-03 — Data-efficiency + ablation sweep infrastructure (Phase B/C glue, Opus)

Built the CPU-light glue so the data-efficiency and ablation sweeps can launch on Kaggle.
No GPU, no training, no Kaggle launches, no git.

**Split manifests (Task 1).** Added `build_data_efficiency_splits` + `write_data_efficiency_splits`
to `src/data/splits.py` (constants `DATA_EFFICIENCY_SIZES={25,50,100,200,400}`,
`DATA_EFFICIENCY_SEEDS={0,1,2}`). Each `data/splits/full_n{size}_s{seed}.json` keeps full's
cal (100) and test (200) verbatim and shrinks only train to a deterministic `rng(seed)`
subset of full-train (700); subsets are nested within a seed (n25 ⊂ n50 ⊂ … ⊂ n400) and
differ across seeds. Wrote all **15** manifests via a new `--data-efficiency[-only]` flag on
`python -m src.data.splits`. Verified nesting, subset⊆full-train, and cal/test identity.

**Leak test (Task 2).** Extended `tests/test_splits_no_leakage.py`: synthetic logic tests
(subset/shared-cal-test/nesting/determinism/oversize-reject) and real-manifest tests over
`full_n*_s*.json` (train⊆full-train, cal/test == full's, disjoint, size-from-filename ==
n_train, full grid present). Passes (100 passed across split+viz files).

**viz train_size (Task 3).** `src/viz/data.py::_train_size_from` gained a 4th fallback:
read `n_train` from the run's split manifest (`data.split_file`, else `splits_dir/<split>.json`)
via new helper `_n_train_from_split`. This is what lets the FREE `{model}_full_s{0,1,2}` runs
land at x=700 on Fig 9 (previously NaN). Data-eff runs resolve via the `_n\d+_` token in the
run_id, unchanged. `test_fig9_...` stays green.

**Sweep specs (Task 4).** `configs/sweeps/kaggle_dataeff.yaml` — 45 runs (3 models × 5 sizes
× 3 seeds), `runs:` list (matrix cannot pair `data.split_file` with seed), model-major,
`train.batch_size=16`, `split=full_n{size}` + `data.split_file=…_s{seed}.json` + `seed={seed}`
so run_id = `{model}_full_n{size}_s{seed}`. `configs/sweeps/kaggle_ablations.yaml` — 6 tagged
runs: M2 conditioning `mask` (tag=cond_mask) and `sdf+normals` (tag=cond_sdfnrm) on {full,
shape5} seed 0; M1 `train.weights.force=1.0` (tag=lamF) on {full, combined} seed 0.

**Code correction.** The M2 conditioning string is `sdf+normals` (with `+`), verified against
`src/models/sdf_fno.py::CONDITIONINGS=("sdf","mask","sdf+normals")`. The `sdf_normals`
(underscore) in the `configs/sdf_fno.yaml` line-15 comment was WRONG — corrected to `sdf+normals`.
Confirmed the `+` survives config parsing and is accepted by the model.

**Smoke checks (Task 5).** `expand_spec` + `run_id_for` on both specs → 45 and 6 unique
run_ids, none colliding with the core grid. Built the `gnn_full_n25_s0` dataset
(cache_in_ram=false, missing=skip, no training): train=25, cal=100, test=200.

**Pre-existing failure (NOT introduced here, flagged):** `tests/test_viz.py::test_tables_write_latex_and_markdown`
fails with `ValueError: truth value of a Series is ambiguous` in `scripts/make_tables.py::_render`.
Cause: the committed K=5 ensembles put 5 `sdf_fno_full_s*` (and transolver) rows in `results/`;
`table1_indist` groups by model on the full split without aggregating seeds, so `best[col]`
has a duplicate `sdf_fno` index. Independent of this task — verified it still fails with
`_train_size_from` forced to all-NaN, and nothing here touches `make_tables.py` or adds run dirs.
Needs a seed-aggregation fix in `make_tables.py`.

Did not run git.

## 2026-09-03 — Phase D now-part: active-learning scoring (Opus, executor)

Built `scripts/run_active.py`, the glue that scores an unseen NACA pool with a trained
ensemble's uncertainty and emits the Fluent case list (design doc 5.8, PLAN_PHASE2 Phase D).
Inference only: no training, no Fluent execution, no Kaggle, no Ansys MCP. Single-process,
`torch.no_grad`, GPU-guarded (used the free P2000 for the demo; CPU stayed light for the
user's Fluent job).

**What it does.** (1) Builds a parametric pool of NACA 4- and 5-digit airfoils
(`src.active.pool.default_pool_shapes` = 21 sections) × an (Re, AoA) grid that deliberately
overshoots the AirfRANS envelope — `re_grid=(2..7)e6`, `aoa_grid=(-6,0,6,12,18)` — so the
score can pick genuinely OOD cases; excludes every training combination of the scorer's split
via a new `training_exclusions()` (rounds each continuous training sim to a NACA code + Re/AoA
with tolerances, feeds `ExclusionSet`). (2) Loads a K-member ensemble via
`src.uq.ensembles.load_ensemble_checkpoints` and scores the pool: `sigma_CD` = ensemble std of
the *integrated* C_D (integrate-then-average through the frozen `integrate_forces`), FSC =
|mean C_D^int − mean C_D^head|, L_sym = mean mirror-symmetry residual (`src.geometry.symmetry`).
Model is fed **normalized** cond (as trained), forces use **physical** cond; p/tau denormalized
before integration. All three components z-normalized over the pool; `a = z(σ_CD) + β·z(FSC) +
γ·z(L_sym)`, β=γ=1 default, CLI-overridable (`--skip-sym` drops the γ term). (3) Three arms of
k=8 via `select_cases` farthest-point diversity: acquisition, variance-only, random. (4) Writes
`fluent/cases_to_run.json` in make_cases.py's schema (case_id/naca_digits/re/aoa_deg/mesh_level +
`arm` tag), **preserving the `gridstudy_*` trio and `defaults`**, replacing only the placeholders;
plus `results/active/acquisition_ranking.csv` (full pool + components + z-scores + per-arm
selection flags) and `results/active/selection_summary.json`.

**Normalization note (D-021).** Uses per-split train-only stats recomputed via
`scripts.build_cache.compute_norm_stats` (matches how the ensemble trained); for `full` this
equals the committed `norm_stats.json`. Re→u_inf uses `NU_DATASET=1.56e-5` so cond magnitudes
match the training distribution, not `pool.NU_AIR`.

**Tests.** `tests/test_run_active.py` (7 tests, CPU, tiny fake K=3 ensemble + small pool +
real integrator + real symmetry residual): components finite/shaped, `--skip-sym` zeroes L_sym,
acquisition score scale-invariant under positive affine rescaling of each component, 3 arms each
return k distinct diverse cases, acquisition arm keeps the argmax, and the emitted
`cases_to_run.json` **validates against `fluent/make_cases.py::load_manifest`+`validate`** (imports
the module). All 7 pass (~20 s).

**Demo (real, transolver_full K=5, k=8, β=γ=1, P2000).** Pool = 630 entries, 0 dropped as
training combos (legitimate: exact-million Re + integer AoA never land within tol of a continuous
training sim; exclusion verified to fire on a fabricated on-training entry). Wrote 27 cases
(3 gridstudy + 24 AL, unique ids, 8/arm); `make_cases.py` validation OK. The acquisition and
variance arms concentrate at the OOD extreme (AoA=18 deg, Re 6–7e6, mix of 4- and 5-digit),
while the random arm spreads across the whole envelope including low-AoA in-distribution cases —
exactly the contrast the falsifiable claim ("selected cases are harder than random") needs.
Committed `fluent/cases_to_run.json`, `results/active/acquisition_ranking.csv`,
`results/active/selection_summary.json`.

Did NOT render `fluent/cases/` journals or run any Fluent/Ansys tool (user CPU green-light still
gated, D-007). Did not run git.

## 2026-09-03 — Kaggle infra fix + self-sustaining cycle (Opus, PLAN_PHASE3 1-2)

Implemented the checkpoint-pull fix and the autonomous chaining loop. CPU-light only:
no training, no git, no real Kaggle session launched (a data-eff session was RUNNING
throughout; confirmed via read-only `kaggle kernels status`, so these driver edits take
effect on the NEXT launch, which is correct).

**Files changed / added**
- `kaggle/session_driver.py`: LEAN export. Records `pre_done` (runs restored already
  finished) before the sweep; new pure `select_export_set()` picks per-run checkpoints for
  runs touched THIS session only. Exports small `results.zip` always, then one
  `ckpt_<run_id>.zip` (ZIP_STORED) per run + `EXPORT_MANIFEST.json` (size+sha256, done flag,
  session_id). Finished runs export `best.pt` only; the one unfinished run exports `last.pt`
  (+best if present). `SWEEP` is now comma-separated -> multiple `--spec` to sweep.py. STATUS.txt
  carries session_id. No more cumulative 2 GB `checkpoints.zip`.
- `scripts/sweep.py`: `--spec` is repeatable (`action="append"`); specs drain in order, name =
  `"+".join(stems)`. Skip/budget logic unchanged.
- `kaggle/launch.py`: `--sweep` repeatable (joined with commas); prints the push output line and
  appends `{utc,slug,specs,commit,runs_dataset}` to `kaggle/launches.jsonl`. Default slug still
  `geo-op-session`; the cycle passes an explicit per-sweep slug.
- `kaggle/pull_results.py`: full rewrite. `fetch_small()` grabs results.zip/session.log/STATUS/
  EXPORT_MANIFEST first and merges results immediately (with a metrics.json clobber guard, D-019);
  `needed_checkpoints()` fetches only core/ensemble best.pt (feeds UQ/AL) + unfinished last.pt,
  skipping data-eff/tagged and already-local files; `fetch_zip()` pulls per-run zips one at a time
  with retry + sha/bytes verify; `republish()` stages results/** + only unfinished last.pt (lean,
  ~10-50 MB) and skips when unchanged (`kaggle/runs_dataset_state.json`). Dedupes merged sessions
  via `kaggle/pulled_sessions.jsonl`. Flags: `--checkpoints {needed,all,none}`, `--no-republish`,
  `--dest`, `--force`. Legacy (no-manifest) sessions: merge results only, never touch checkpoints.zip.
- `kaggle/fetch_zip_members.py` (new): HTTP-Range member extractor. `HttpRangeFile` (seek/tell/read
  via `Range: bytes=`) fed to `zipfile.ZipFile` reads only the central directory + requested members
  from a kernel-output zip (e.g. one 16 MB best.pt out of a 2 GB archive). Recovery tool for the
  legacy fat zip (PLAN_PHASE3 1.8) and CLI fallback. Best-effort; prints the curl -C - fallback.
- `kaggle/cycle.py` (new): idempotent one-step loop. Reads `kaggle/cycle_queue.json`, acts on the
  first not-done item: RUNNING/QUEUED -> exit 0 untouched; COMPLETE/ERROR -> pull (lean) then
  `sweep --dry-run` decides advance (0 to run) vs relaunch; never launched -> push_code + launch.
  Pure `classify_status`/`next_action`/`current_index` are unit-tested; verified `classify_status`
  parses the real CLI form `... has status "KernelWorkerStatus.RUNNING"`. `--dry-run` prints the
  queue with no kaggle/git call. Stops on any non-zero subprocess; push_code runs before every launch.
- `kaggle/cycle_queue.json` (new): ordered queue. core + ensembles marked done; dataeff, ablations,
  gnnens pending (owner cocomo069, runs_dataset geo-op-runs). gnnens spec `kaggle_gnn_ens_k3.yaml`
  is pending creation (noted); cycle refuses to launch a missing spec.
- `tests/test_cycle.py` (new, 14 tests, CPU, all kaggle/git mocked): select_export_set names/files,
  needed_checkpoints policy + skip-if-local, status classification, next_action (advance only on
  COMPLETE), current_index, step() no-op while RUNNING / launch when absent / advance vs relaunch on
  COMPLETE / drained queue / dry-run makes no calls, and sweep multi-spec ordering.

**Verification.** `py_compile` all kaggle/*.py + sweep.py OK; `pytest tests/test_cycle.py` 14 passed;
`pytest tests/test_infra.py` 49 passed (multi-spec change is backward-compatible); `cycle.py --help`
and `--dry-run` OK (current item = geo-op-dataeff).

**Expected sizes after the fix.** Per session export: results.zip (small, unchanged) + one
`ckpt_<rid>.zip` per NEW run (~16-33 MB each, ZIP_STORED) instead of one cumulative ~2.1 GB
checkpoints.zip. Republished geo-op-runs drops from ~1.5 GB to ~10-50 MB (results/** + only unfinished
last.pt). A stalled pull now costs one 16-33 MB retry, never the session.

**How to run the cycle once per scheduler tick:** `.venv/Scripts/python.exe kaggle/cycle.py`
(one step, safe while a session is mid-flight). `--dry-run` inspects state without any call.

Did NOT run git.

## 2026-09-03 — PLAN_PHASE3 section 4 "NOW" items: figures 1/2/3/10/11a, calibration/AL/
conclusion/repro/appendix prose (Sonnet, executor)

Pure writing + matplotlib, CPU-only, no GPU touched, no training, no Kaggle, no git. Did
what PLAN_PHASE3.md section 4 marks NOW from data already committed locally
(`results/`, `results/uq/`, `results/active/`, `checkpoints/`).

**Figures.** Added to `scripts/make_figures.py` (pure, gated exactly like the existing
functions): `fig1_schematic` (protocol diagram, matplotlib patches, no data dependency),
`fig10_symmetry` (grouped bars of `consistency.sym_residual` by model x split, log-y),
`fig11_active_pool` (acquisition-score-vs-AoA scatter over the 630-candidate pool, coloured
by Re, the three k=8 arms marked, from `results/active/acquisition_ranking.csv`). New file
`scripts/render_examples.py` (kept separate from `make_figures.py` because it needs
`torch`/checkpoint loading, a different dependency shape than the pure-dataframe figure
functions) does real CPU inference: loads a config + one or more seed checkpoints, builds
the split's test dataset via `scripts.train.build_datasets`/`build_model`/`load_checkpoint`,
runs a forward pass on one representative test simulation, denormalises, and plots
`c_p = p / (0.5 U_inf^2)` (kinematic pressure convention, DATA_NOTES.md A1.2). Produces
`fig02_cp_profiles.pdf` (Transolver, prediction vs. truth, K=5 ensemble band, one panel per
split: full/reynolds/aoa/shape5 — real committed K=5 checkpoints exist for all four) and
`fig03_surface_fields.pdf` (repurposes the DrivAerNet++ "three cars" slot, out of v1 scope
per D-005: truth/prediction/|error| c_p scattered on the airfoil contour, full vs. combined
split, seed 0). Both are genuine model output, not illustrative sketches; fig2 in particular
visibly reproduces the paper's own finding (tight band and near-perfect fit in-distribution,
systematic offset and wider-but-insufficient band under Reynolds/AoA shift, tight fit under
shape5 — matching the "shape5 is the mildest single-axis shift" result).

**Bug fixed in passing.** `fig9_data_efficiency`'s gate (`train_size.nunique() >= 3`) was a
false positive on the *real* results tree: `_train_size_from`'s split-manifest fallback gives
every OOD split its own `n_train` (700/160/404/704/391/191), so six different splits looked
like >=3 distinct training-set sizes and the function would have plotted a spurious
data-efficiency curve that actually conflates distribution shift with training-set size —
before any data-efficiency sweep has run. Fixed by filtering to `split == "full"` first
(matching `fig12`'s existing pattern and the true intent: data-eff varies size *within*
`full`). Verified: `tests/test_viz.py -q` still 9/9 green (its synthetic fig9 test already
uses split=full with distinct `n<size>` tags, so it was and remains a true positive); fig9
now correctly TODO-skips against the real tree.

**Regenerated figures.** `python -m scripts.make_figures --outdir paper/figures` (9 files:
fig01, 04, 05, 06, 07, 08, 10, 11, 12; fig09 correctly TODO-skipped) then
`python -m scripts.render_examples --outdir paper/figures` (fig02, fig03). All from
committed JSON/checkpoints, none hand-edited.

**Paper (`paper/main.tex`).** Wrote: Fig 1/2/3/10 captions and uncommented their
`\includegraphics` (deleting the matching `\figph` lines); Fig 6/7/8 uncommented (they were
still `\figph` placeholders despite the PDFs already existing) and the fig08 filename
mismatch fixed (`fig08_width_dists.pdf` -> `fig08_interval_width.pdf`, the name
`make_figures.py` actually writes); replaced the hand-typed placeholder Table 3 with
`\input{tables/tab3_calibration}` (matching how Table 1/2 already work); Section 8.4
Calibration rewritten from `results/uq` + `tab3_calibration` + docs/RESULTS.md 3.1-3.2 (in-
distribution near nominal for every model; GNN is K=1 for now, absolute score not
normalized, stated explicitly; Reynolds/AoA are the worst shifts, GNN collapses to 0.698
coverage on reynolds; matched-vs-transfer calibration does not order cleanly, reported as
found rather than rounded into a clean story; width widens under shift but under-
compensates). Section 9 Active Learning: real narrative of the actual run (Transolver K=5
scorer, 630-pool, three k=8 arms), new `paper/tables/tab5_selected_cases.tex` (the 8
acquisition-arm cases with sigma_CD/FSC/L_sym/score from `results/active/`, replacing the
old Fluent-blocked Table 5 placeholder per PLAN_PHASE3 4.2, labelled "selected, verification
pending"), Figure 11 restructured into two subfigures (11a live = the new pool figure, 11b
still `\figph`, genuinely blocked on Fluent), falsifiable-claim paragraph restated precisely
(what can and cannot be said before an independent solver runs). Conclusion written (three
findings + a "what to do differently Monday" paragraph, explicitly a protocol contribution
not a leaderboard position). Reproducibility statement written (MIT/DOI-at-release, what the
release contains, AirfRANS ODbL-1.0 / DrivAerNet++ CC-BY-NC obligations, the reimplementation
note per D-002, the per-split normalisation note per D-021, seeds). Discussion updated: the
"single seed" bullet now states K=5 (M2/M3) vs K=1 (M1, ensembles still training) precisely;
the "calibration not yet measured" bullet rewritten to describe what was actually measured
plus its small-sample caveat; the Fluent bullet updated to the real 27-case count; the
"negative and mixed results" paragraph gained the matched-vs-transfer non-ordering as a
concrete mixed result. Appendix A (hyperparameters) filled from `configs/*.yaml` (two
tables: per-architecture, and the shared training recipe). Appendix B (CFD setup) written as
a structured stub with real design content from `docs/plans/FLUENT_PLAN.md` (meshing route,
solver setup, GCI plan, offset-study plan, acceptance criteria) and `\todo{}`-tagged only the
actual measured numbers, which do not exist yet. Appendix D (compute accounting) computed
directly from `train_time_s` summed over all 50 committed `metrics.json` files that carry it
(23.27 GPU-h total: GNN 10.45h/6 runs, SDF-FNO 6.08h/22 runs, Transolver 6.74h/22 runs),
stated as a lower bound, not an estimate, since GNN-ensemble sessions still in flight are not
in the sum. Table 4 (ablations) caption trimmed to the lean v1 set (no numbers filled in —
ablations were explicitly out of scope for this pass per the task's data-availability note).

**Not touched, on instruction:** `kaggle/`, `scripts/run_*.py`, `src/` training code,
`scripts/make_tables.py` (existing tab1/2/3 `.tex`/`.md` files used as committed static
assets, not regenerated — regenerating tab1 is a known pre-existing landmine per the
2026-09-02 entry above, unrelated to this session), ablation numbers, data-efficiency
numbers, Fluent numbers, DrivAerNet++.

**Compile.** `pdflatex` + `bibtex` + `pdflatex` x2: clean, zero undefined references or
citations, 24 pages. `\todo` count 23 -> 18, `\pending` count roughly unchanged (not in
scope), `\figph` usages in the main body 12 -> 2 (fig9 data-efficiency, fig11b Fluent
verification — both genuinely blocked on data this session does not have).

Did NOT run git.

## 2026-09-03 — Fluent execution (Opus): mesh fix + first solving case

First real Fluent run of the study. Goal was one NACA case meshing cleanly AND
solving to convergence in v211, producing the surface CSV plus CD/CL. Done for
`gridstudy_naca0012_re3e6_a5_L1` (SA).

### Root cause of the "Build Grid: Aborted" abort (it was NOT winding)

The prior diagnosis suspected wake-cut face winding or a degenerate sharp-TE
cell. Both were checked and ruled out:

- A geometric winding check on the written `.msh` (reconstruct cell centroids,
  test that each face's CW normal `(dx,dy)->(dy,-dx)` points out of owner c0)
  found 0 bad faces out of 37878. Winding is correct everywhere, wake cut
  included. Zone-type codes are exact (interior 2, wall 3, pressure-outlet 5,
  velocity-inlet 10) and Fluent read every zone.
- The min-Jacobian 5.27e-9 cell at the TE is tiny but strictly positive
  (not folded); Fluent's own `/mesh/check` reports all-positive volumes.

The actual bug was an off-by-one in the wake-cut node identification in
`write_fluent_msh`: the condition `i > nw + na` skipped k = 0 (i = nw+na, the
trailing edge on the upper branch), so the upper-branch TE node was never merged
with the lower-branch TE node. That left a duplicate node at the TE that split
the cell fan, so the upper TE cell (cell (nw+na, 0)) referenced 5 nodes instead
of 4 and never closed. Fluent aborts "Build Grid: Aborted due to critical error"
on exactly that. Fix: `i >= nw + na` (the docstring's own formula
`node(nw+na+k,0) === node(nw-k,0)` for k = 0..nw already required it). Node count
dropped 19083 -> 19082 (the one duplicate removed). `validate_msh` was hardened
with a closed-quad check (every cell must have exactly 4 nodes each shared by 2
of its faces) so this class of bug can't reach the solver silently again; the
old self-check passed the broken mesh because it never verified cell closure.
The fix applies to all three grid levels (all regenerated, all min_jacobian > 0).

### v211 TUI keyword reconciliation (probed live, template was authored blind)

`probe_bc_keywords.jou` plus two targeted probes found four template bugs, all of
which silently eat following journal lines (the exact failure mode the template
warns about). Fixed in `case_template.jou` and `make_cases.py`, journals
regenerated:

1. velocity-inlet components are `direction-0` (X-Velocity) / `direction-1`
   (Y-Velocity), NOT `u`/`v` (which don't exist in this build's grouped `set`
   menu). velocity-spec cycles: `no` (skip Magnitude+Direction) then `yes`
   (select Components). Each profile field answers "Use Profile? [no]" first.
2. SA inlet turbulence key is `turb-viscosity-ratio-profile`, not
   `turb-viscosity-ratio`. On the pressure-outlet that key is not settable from
   the menu; with prevent-reverse-flow walls its backflow value is never used, so
   the outlet turb block is omitted.
3. The `set/wall` block was removed entirely: `shear-bc` prompts "Change current
   value? [no]" (the wall is already no-slip), and an answer of `no-slip` is not
   the yes/no it wants -> the repeating prompt ate the whole reference-values and
   report-definitions blocks. Fluent's wall default is exactly stationary no-slip,
   so nothing is lost.
4. The moment (cm) report-definition was dropped: v211's `moment` report-def
   rejects `about-point` and the failure cascaded, eating the entire
   report-files block (so no coeffs.out was written). v211 uses `mom-center`
   <x y z> + `mom-axis` <x y z>; left as a flagged TODO since CD/CL are the
   reported quantities and cm was not required. `/define/models/steady` takes no
   argument (a trailing `yes` is an orphaned invalid command); the invalid
   `/solve/report-definitions/compute` line was removed (CD/CL come from
   coeffs.out). SST inlet keywords are updated to the best reconstruction but are
   NOT yet verified on a live SST solve - probe before the first SST case.

### L1 SA result (converged, physically sane)

- CD = 0.011056  (pressure 0.003577 + viscous 0.007479; viscous is ~68% of drag,
  so the no-slip wall took effect and CD is well above the 5e-3 missing-viscosity
  red flag)
- CL = 0.55299   (pressure 0.55303, viscous ~0). Thin-airfoil 2*pi*alpha = 0.548,
  so within ~1%.
- Surface Cp: stagnation +1.01, suction peak -1.89 at x/c ~ 0.009, attached
  (Cf > 0 everywhere).
- y+ max = 2.10, area-avg = 0.98. This EXCEEDS the plan's wall-resolved target
  (max < 1, avg < 0.7). L1 is the coarse level (y+ target 0.9) and the a-priori
  flat-plate sizing underpredicts the LE/suction-peak y+ by ~2.3x, exactly as
  FLUENT_PLAN section 3.2 anticipated. To make all three levels strictly
  wall-resolved, lower the y+ targets by ~2.3x (or accept L1 as the coarse end).
- Convergence: continuity plateaued ~1.5e-6, momentum ~1e-8, nut ~1.4e-5. CD is
  mildly unsteady (the CD-steady range oscillates 1-4e-5 around the 1e-5 stop
  criterion), so it ran the full stage-2 budget to iter 4300 rather than
  auto-stopping. CD flat to ~4 sig figs; accepted per FLUENT_PLAN section 6
  (a plateau on a mildly-unsteady flow is expected, declared behaviour). Runtime
  ~30 min on 6 cores (2ddp) for 18.8k cells.

### Files produced (L1 SA)

`_sa.cas.h5`, `_sa.dat.h5` (display objects ct_pressure/ct_velocity/ct_cp/
vec_velocity baked in for GUI screenshots), `_sa_coeffs.out`, `_sa_force_x.txt`,
`_sa_force_y.txt`, `_sa_yplus_max.txt`, `_sa_yplus_avg.txt`, `_sa_surface.csv`
(columns mapped by header name: y-plus, skin-friction-coef, wall-shear x/y/mag,
pressure-coefficient, pressure), and a matplotlib cp/cf/yplus verification plot
`_sa_surface_cp_cf_yplus.png`. Large binaries (.msh, .cas.h5, .dat.h5, .trn) are
gitignored; small result text files and the plot are kept.

### ParaView note

ParaView's built-in Fluent reader cannot open the v211 `.cas.h5` (CFF/HDF5)
format (open timed out, no source registered). For ParaView field contours at
scale, add an EnSight-Gold or CGNS export to the journal after the solve, or open
the `.cas.h5` in the Fluent GUI (the display objects are already baked in). The
surface cp/cf here was plotted with matplotlib from surface.csv.

### Recipe for the remaining cases (orchestrator)

Meshes + journals for all 27 cases are regenerated and valid. Per case, per
model, run the case journal via `fluent_run_journal` (2ddp, processors 6), poll
`ansys_job_status` + `fluent_check_convergence`, then read the last row of
`_coeffs.out` for CD/CL and the yplus/force/surface files. The SA path is fully
verified. Before the first SST case, run `probe_bc_keywords.jou` on it to confirm
the SST inlet turbulence keywords (turb-intensity / turb-viscosity-ratio and
their use-profile prompts) and the 5-equation residual count, since those were
reconstructed, not measured.

## 2026-09-03 — Fluent full-campaign prep (Opus): EnSight, y+ fix, SST, batch runner

Prepared the whole 2D RANS campaign to run unattended as a standalone background
process. No long batch was run (that is ~17 h and is left for the user); only the
machinery was built and validated with short probes and one capped solve.

### 1. ParaView bridge via EnSight Gold export (verified end to end)

ParaView cannot open a v211 `.cas.h5` (CFF/HDF5), so the case template now writes
an EnSight Gold export after the solve+save. The exact v211 TUI arg order was
found by probing the live solver on the existing L1 case+data (three iterations,
`runs/probe_ensight*.trn`): after the filename the **scalar list comes first**,
then binary?, then cell zones, then interior surfaces, then cell-centred. The
naive "surfaces/binary first" ordering silently exports velocity only. The
verified one-liner baked into `templates/case_template.jou`:

    /file/export/ensight-gold "<out>.encas" pressure pressure-coefficient \
        velocity-magnitude x-wall-shear y-wall-shear () yes * () () no

Output opens cleanly in ParaView (`paraview_open_data`): all 5 scalars + the
velocity vector, cp range [-1.89, 0.999], matching the known suction peak /
stagnation. The baked-in native Fluent display objects are kept as well (GUI
screenshots). EnSight artifacts (`*.encas/.geo/.scl*/.vel`, ensight `.xml`) added
to `.gitignore` — regenerable like the `.cas/.dat`.

### 2. y+ fix — target lowered 0.6 → 0.25, all meshes regenerated

The first L1 solve gave achieved y+max 2.10 against a 0.90 target (ratio ~2.33),
over the wall-resolved <1 requirement. Lowered the medium-level y+ target 0.6 →
0.25 (factor ~2.4) in `cases_to_run.json` defaults, `make_cases.py` and
`mesh_gen.py`, and regenerated all 27 meshes + 54 journals. Confirmed on a
regenerated L1 (target 0.375) with a capped solve through the new batch runner:
**achieved y+max 0.87, y+avg 0.41 — wall-resolved.** Since L1 carries the largest
target, all three grid levels are now < 1. `mesh_gen.build_cgrid` fix: at the
smaller first cell the mid-chord streamwise spacing can exceed
`first_cell*AR_CAP`, which tripped the aspect-ratio-cap assertion on the section;
replaced the assertion with an explicit `fc[section] = first_cell` (the cap must
never move the wall cell there — that IS the guarantee it was only checking).
Max AR rose to ~6.5e3 (still < the 1e4 acceptance floor); all min_jacobian > 0.

### 3. SST keywords + residual count (probed live, `runs/probe_sst.trn`)

Residual header is exactly `continuity x-velocity y-velocity k omega` = **5
equations** (n_residuals=5 was right; the old "6" comment was wrong). The SST
velocity-inlet keys `turb-intensity` / `turb-viscosity-ratio` are PLAIN numeric
fields with NO "Use Profile?" prompt (unlike SA's `-profile` key), so the value
follows the key directly; the previous `no` line threw "eval: unbound variable"
and only recovered by luck. Removed the `no` lines from the SST `inlet_block` in
`make_cases.py` and re-rendered. SST velocity-inlet default turbulence-spec is
already "Intensity and Viscosity Ratio", so ke-spec need not be set.

### 4. Standalone batch runner `fluent/run_batch.py`

Resumable, sequential, drives `fluent.exe` directly (not the MCP). Per case:
builds the mesh if missing (runs the case's own `mesh.cmd`), skips if
`_coeffs.out` has a final row AND `.cas.h5` exists, else launches
`fluent.exe 2ddp -g -t6 -i <journal> -wait` blocking, appends to
`logs/fluent_batch.log`, continues on error/timeout. Flags: `--dry-run`,
`--only`, `--models`, `--sst-gridstudy`, `--smoke-iters` (validation cap),
`--timeout-min`. Validated: dry-run lists 27 SA (30 with `--sst-gridstudy`,
gridstudy first); skip logic detects a complete case; one real capped launch
(L1 SA, 800 iters, 2.9 min) produced cd 0.0112 / cl 0.550, the y+ files above and
the EnSight export. Smoke artifacts were cleared afterward so the real run
re-solves L1 to full convergence.

### 5. Template cleanups

Removed the untested `/mesh/repair-improve/report-poor-elements` line (the
proven L1 journal never had it) and added a final `/exit yes`.

### Cases expected to be hard

20 of the 24 AL cases sit at **α = 18° (post-stall)**; steady RANS may not
converge to a flat CD there and several of those meshes have min orthogonality
0.13–0.19 (inherent to high-AoA thick-section C-grids, below the 0.15 acceptance
floor — flag, not fold; all Jacobians positive). The seven Re = 7×10⁶ α = 18°
cases also sit at M ≈ 0.32, marginally past the incompressible assumption. The
random arm's low-α / α = 0 cases (`al_rand_*_a0`, `_a6`, `_am6`, `_a12`) are the
safe ones. Their non-convergence, if it happens, is itself the active-learning
result (acquisition picked the hard corner of the envelope), not a setup bug.

---

## 2026-09-03 (Opus) — Fluent post-processing + C4 salvage (PLAN_FLUENT_POST)

Executing Fable's `docs/plans/PLAN_FLUENT_POST.md` after the main SA batch (grid study
clean; 4 low-AoA random converged; 20 diverged).

### collect_fluent.py (step 1, done)
`scripts/collect_fluent.py` parses every `fluent/cases/*/*_coeffs.out`, classifies
converged / quasi_steady / diverged per PLAN section 2, computes the Celik GCI on
the gridstudy trio, and runs the batch-integrity check (cell count via `.trn`,
h5py absent). Classifications reproduce PLAN section 0 exactly: gridstudy 3/3
converged, random 4 converged (low-AoA) / 4 diverged, acquisition 8 diverged,
variance 8 diverged. The literal "≤1e-5 over 150 iters" converged rule was too
strict (real converged cases creep ~3e-4 over 150 iters while flat to ~1e-6 per
iter), so the classifier keys on window oscillation amplitude (p2p/|mean| < 1%,
no drift) — documented in the script. GCI: SA CD ~0% (L2/L3 identical, p_obs
noise-dominated — use L1→L2 3.3% as the conservative band per PLAN §0), SST CD
0.26%, CL SA 3.4% / SST 0.22%.

### Offset study (steps 2–3, batch running)
`mesh_gen.py --naca-params` (via `airfrans.naca_generator`) + `make_cases.py`
naca_params/per-case mu. `fluent/make_offset.py` froze the 6 replicas, passed the
geometry gate (dense contour vs cached `surf_pos`, max NN < 5e-4 c — all ~7.7e-5),
and appended per-case mu/re (reproducing each sim's U exactly) to
`cases_to_run.json`. All 6 meshes: min orthogonality ≥ 0.216, positive Jacobian.
Launched DETACHED: `run_batch.py --models sa,sst --only-file fluent/offset_cases.txt`
→ `logs/fluent_offset.log` (12 runs, ~2.5–3 h). offset_1 SA converged
(cd 0.0105, cl 0.261); offset_1 SST classified diverged (to re-check on completion).

### r1 retry (step 4 done, step 5 pending offset completion)
`templates/case_template_r1.jou` (stage-1 1000 first-order iters @ URF p0.2/mom0.5/
turb0.5; stage-2 6000 @ p0.2/mom0.4; no cd-steady stop). `make_cases.py --variant`
writes `<case>_<model>_r1` journals + `MANIFEST_r1.json` without touching S0.
`fluent/diverged_cases.txt` (20 cases, control-first order) generated from the
collected results. r1 journals rendered for all 20. NOT yet launched — waits for
the offset batch to free the single licence; will run the α=12 control
(`al_rand_naca22012_re3e6_a12`) FIRST as the gate.

### compare_fluent.py (step 4 done)
`scripts/compare_fluent.py` computes the solver offset (offset.csv) and scores
every Fluent case with the 3 surrogate ensembles, offset-corrected, with 90%
conformal-band coverage. **Key finding:** the integrated `cd_int` on the SYNTHETIC
pool geometry (score_pool path) is an input-representation artifact (~0.05 vs
0.0099 on the real mesh for a checked in-envelope shape; unchanged by point count)
— `cd_head` (the coefficient-head regression) is input-robust and is the PRIMARY
surrogate CD; `cd_int` is kept as an FSC/physics diagnostic and is reliable only
on the replicas (per_sim, real mesh). `tests/test_compare_fluent.py` pins schema,
speed convention and offset arithmetic (CPU, no dataset). Full 3-model run is
deferred until the offset + r1 batches complete.

### ParaView (step 5)
Not run here (per brief). `fluent/paraview/PARAVIEW_TODO.md` lists the ready
`.encas` cases and the exact figure specs (B1 grid, B3 the 4 random, B4 offset_2
SA vs SST, B5 any r1-rescued) for the orchestrator's ParaView step.

### 2026-09-03 (Opus) — background tasks killed; moved to Task Scheduler (durable)

The environment kills Claude-session background tasks (Bash AND PowerShell) after
~1 h, so the detached offset/r1 batches did not survive (three kills). Completed
work was preserved each time (batches are resumable). Switched the whole campaign
to a **Windows Task Scheduler** job `FluentCampaign` (D-024 GeoOpCycle pattern):
`fluent/campaign_tick.py` runs ONE ~10-21 min solve per 8-min tick
(MultipleInstances=IgnoreNew serialises on the single licence), priority offset
SA+SST -> r1 control gate -> the 19 (only if the alpha=12 control reaches a
(quasi-)steady solution). On completion it auto-runs collect_fluent --gci +
compare_fluent + make_tables + make_figures and disables its own task. Survives
session idle and any hourly reaper (each tick is well under an hour).

State at handoff: offset study essentially done (offset_1..5 SA+SST + offset_6 SA;
offset_6 SST + the r1 retry run under the scheduler). Measured solver offset
(SA, n>=5): Delta_CD ~ +0.0009 (s 0.0003), all replicas positive; SA<->SST spread
< |Delta_CD|. RESULTS section 6 rewritten with the qualitative-external-check
reframe + limitation text; FLUENT_PLAN 5.2 + D-026 updated.

**Orchestrator handoff:** when `logs/fluent_campaign.FINALIZED` appears, the
scheduled task has regenerated `results/fluent/*` + Table 5 + Fig 11. Then:
(1) `git add results/fluent/*.csv *.json paper/tables/tab5_fluent.* paper/figures/fig11_active_verify.*`
and commit; (2) splice the final surrogate MAE/coverage numbers + the r1 outcome
into RESULTS section 6.4 / Table 5a (values in surrogate_vs_fluent_summary.json);
(3) run the ParaView step (fluent/paraview/PARAVIEW_TODO.md); (4) remove the
`FluentCampaign` scheduled task if it did not self-disable
(`schtasks /Delete /TN FluentCampaign /F`). A collect classifies a still-running
solve from its partial coeffs, so trust `has_cas`/the FINALIZED run, not an
interim snapshot.

## 2026-09-03 — D-025 ParaView field figures (Opus execution)

Rendered the ParaView field-contour figures from the EnSight Gold `.encas` exports
(ParaView is the post-processing tool of record). Fluent was NOT launched (solves
running under the FluentCampaign scheduler); ParaView-only, light load. 9 PNGs at
300 dpi in `paper/figures/fluent/`:
- B1 grid-independence C_p (NACA0012 Re3e6 a5, SA): L1/L2/L3
  (`figB1{a,b,c}_gridstudy_L{1,2,3}_cp.png`), fixed range [-2,1], Cool to Warm.
- B3 velocity-magnitude + black streamlines (SA), Viridis, per-case range:
  `figB3{a,b,c,d}_al_rand_{naca0010_re7e6_a6, naca2415_re6e6_a0,
  naca33012_re6e6_am6, naca4415_re2e6_a0}_vel.png`.
- B4 SA-vs-SST C_p on offset_2 (`figB4{a,b}_offset2_naca0109_a8p6_{sa,sst}_cp.png`),
  fixed range [-4,1]; SA/SST fields near-identical (peak -6.72 vs -6.66).
No required cases skipped (all `.encas` readable). B5 (optional r1-rescued) not
produced: no `*_sa_r1.encas` present yet. `.encas` fields are already node-based
POINT data with friendly names, so CellDataToPointData was unnecessary. Legend
persistence handled by hiding prior reps/scalar bars before each export; verified
each figure by reading the PNG back. Details + captions in
`fluent/paraview/RENDERED.md`. Did NOT run git.

---

## Session 4 (2026-09-06) — GNN K=3 ensemble finished, project complete

The last open item (GNN K=3 deep ensemble for Table 3) landed via the autonomous
`cocomo069` path. The second account `taimooramin0699` was abandoned for GPU work: a probe
kernel proved `torch.cuda.is_available()==False` there because the account is not
phone-verified (Kaggle gates GPU + internet behind phone verification), which only coco can
clear. So the finish waited on the weekly quota reset instead.

Quota reset on 2026-09-06 (my Sat-00:00-UTC estimate was ~a day early). `GeoOpCycle` then
auto-launched `geo-op-gnnens`. It took **two kernel versions**: v1 ran ~8 h and completed 5/8
runs; v2 resumed through the runs-dataset skip logic and finished the last 3 (`gnn_aoa_s2`,
`gnn_shape5_s1`, `gnn_shape5_s2`). `cycle.py` pulled results, reran `run_uq`, regenerated
figures + Table 3, and committed+pushed (`6ddee8e`, `f06a494`).

Verified: 12 GNN ensemble runs (4 splits × seeds s0/s1/s2 = K=3). Table 3 fully populated for
all three models. Headline (RESULTS.md §3): K=3 lifts GNN matched cov@.9 on Reynolds 0.70→0.97
and AoA 0.79→0.89. Both scheduled tasks (GeoOpCycle, FluentCampaign) disabled; project scratch
removed; `cocomo069` token active. Reusable C: caches left for coco to decide on.

---

## Session 5 (2026-09-07) — paper written end to end, figures brought to research grade, repo tidied

Worked in the `research-paper-org-publish` worktree. Three things happened.

**QA baseline re-verified.** Full suite 449 passed before any change (and again after).
Tables regenerate bit-identical from `results/`; `gnn_fake_s0` was a dead config-only dir
(deleted), `geo-op-session.log` was empty (deleted), and the design doc moved to
`docs/05_geometry_aware_neural_operator_surrogate_uq.md` with all references updated.

**Figures fixed at the source** (`make_figures.py` / `render_examples.py`), then regenerated:
fig01 box was clipped off-canvas; fig02's `invert_yaxis()` toggled per panel on shared axes,
so with 4 panels the published figure was NOT suction-up despite its label (now inverted once);
fig03 had duplicate colorbars and dead whitespace; fig04/11a had title/legend/colorbar
collisions; fig09's log ticks overlapped; fig06 was a 12-line legend on an empty unit square
(now faceted per model); fig05 was an identity-line blob (now quantile-zoomed); fig10 dropped
a pointless log axis. All regenerated PDFs+PNGs inspected page by page.

**Paper rewritten from RESULTS.md** into a shared `paper/body.tex` with two wrappers:
`main.tex` (single-column, 26 pp) and `main_twocol.tex` (two-column, 19 pp). All stale
2026-09-03 claims replaced (GNN K=3 with band-width disclosure, cd_int counterpoint,
data-efficiency crossing, ablations, complete Fluent §7 with grid study/offset/regime split,
honest C4). Table emitters now write float bodies (caption+tabular, no table env) so the same
generated file serves both layouts; Table 5 became summary+offset+per-case; everything
LaTeX-escaped. Both builds compile with ZERO overfull boxes and no undefined refs. Compute
appendix re-summed from metrics: 61.0 GPU-h over 110 runs (GNN 42.8/32, SDF-FNO 9.8/41,
Transolver 8.4/37). Pooled FSCrel-vs-field-error correlation recomputed over 58 cells:
Pearson 0.89, Spearman 0.87 (replaces the draft's pending r=0.94).

### Session 5 continuation (2026-09-07, morning) — deep-clean reorg + the "cycle gremlin" root cause

Deep-clean reorganization landed (PR #2): docs/plans/ hierarchy with all cross-references
rewritten, per-folder README indexes, model cards generated from committed metrics
(scripts/make_model_cards.py), root README repository map. Local folder cleanup: dead
checkpoint dirs (gnn_fake_s0, gnn_full_s0_g2, smoke_dryrun) and kaggle/pulls (181 MB)
removed, stray data/ logs consolidated into logs/, redundant Dataset.zip deleted per coco
(9.34 GB freed; re-download via scripts/download_data.py if ever needed).

Root cause of the mystery "[results] cycle" commits found and fixed: the two completion-path
tests in tests/test_cycle.py monkeypatched everything cycle.step() calls EXCEPT
regenerate_and_commit, so every full pytest run executed the real regen + git commit
(cycle's author/message) inside whatever checkout pytest ran from, then attempted a push of
local main (silently rejected, non-fast-forward). Four such commits observed on 2026-09-07,
two of which swallowed staged unrelated work and were squashed/re-authored before pushing.
Both tests now stub it; the scheduled tasks were never the culprit and remain disabled.
