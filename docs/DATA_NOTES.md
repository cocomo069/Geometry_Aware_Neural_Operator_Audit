# DATA_NOTES.md — verified facts from airfrans_lib source (Fable, pre-download)

Source inspected: github.com/Extrality/airfrans_lib `src/airfrans/simulation.py`, `dataset.py` (v0.1.5.1).

- Download URL (preprocessed): https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip — **Content-Length 10,029,067,577 B ≈ 9.34 GiB** (HEAD-verified 2026-08-24).
- Per-sim files: `<name>_internal.vtu` (volume) + `<name>_aerofoil.vtp` (surface), read via pyvista.
- Sim name encodes conditions: `..._<inlet_velocity>_<aoa_deg>_...` at fields 2,3 of `name.split('_')`; NACA params follow (4 or 5 numbers → 4/5-digit family). Verify exact tail format after download.
- `Simulation.surface` mask = `internal.point_data['U'][:,0] == 0` (no-slip nodes). SDF = `-implicit_distance` (positive in fluid per their sign flip; verify).
- **Surface normals stored in the .vtp are INWARD-pointing** (their force uses `F_p = +Σ p·n·ds`, which equals −∮p n_out ds). Our cache (CONTEXT.md §4) requires OUTWARD normals → `build_cache.py` must flip sign. A1: verify empirically (normals at LE should point −x).
- **Pressure is kinematic (p/ρ)**, OpenFOAM incompressible convention. Their `force(compressible=False)` multiplies by RHO = P_ref·MOL/(R·T) at T=298.15 K, then coefficients divide by 0.5·RHO·U² — so ρ cancels: using kinematic p directly with rho=1 in our integrate_forces reproduces their coefficients. a_ref = chord = 1 (per unit span; no area division beyond 0.5ρU²).
- Wall shear stress: NOT stored; computed as τ = 2ν·S_dev·(−n_inward) from the velocity jacobian (pyvista `compute_derivative` point-wise), with **ν = molecular kinematic viscosity only** (NU(T) polynomial, ~1.56e-5), NOT ν+ν_t. Deviatoric: S − tr(S)/3·I. `wallshearstress(over_airfoil=True)` maps internal-mesh surface nodes to the .vtp airfoil mesh via `reorganize`. A1 must cache `surf_tau` computed exactly this way (reference=True fields).
- Their quadrature: point→cell (`ptc`) averaging, then Σ cell_value·cell_Length. Our point-weight quadrature (half-sum adjacent facets) is mathematically equivalent to first order; G1 tolerance 1% median accommodates the difference.
- Drag/lift directions: rotation by AoA matrix `[[cosα, sinα], [−sinα, cosα]]` applied to force → (drag, lift). Matches CONTEXT.md ê∞/ê⊥ with ê⊥ = rot90(ê∞) counterclockwise.
- `cl_true`/`cd_true` for the cache = `force_coefficient(reference=True)` totals (not stored in files; computed).
- `dataset.load(root, task, train)` reads manifest text files (`manifest.json` in the zip? verify) — A1: introspect after unzip.

---

# A1 — full package introspection (2026-08-24)

Source of truth: `github.com/Extrality/airfrans_lib` v0.1.5.1, files
`src/airfrans/{simulation,dataset,naca_generator,reorganize}.py` plus
`docs/source/notes/{dataset,simulation}.rst`. The package installed in `.venv`
matches. Items marked **[verify after download]** are the only ones not
derivable from source.

## A1.0 Resolutions of the pre-download "verify" flags above

| Flag | Resolution |
|---|---|
| "manifest text files (`manifest.json` in the zip? verify)" | **Confirmed**: `dataset.load` opens `osp.join(root, 'manifest.json')` — a single JSON dict at the dataset root, keys `"<task>_train"` / `"<task>_test"`. |
| "NACA params tail format — verify" | **Confirmed from upstream docs**: the name has 7 fields (3 NACA params, 4-digit series) or 8 fields (4 NACA params, 5-digit series). See A1.4. |
| "SDF positive in fluid (verify)" | **Confirmed**: `sdf = -implicit_distance`; upstream calls it "the distance to the airfoil (one component in meter)", non-negative in the fluid. |
| "normals INWARD — A1 verify empirically" | **Confirmed from upstream docs, not merely inferred**: `simulation.rst` opens with *"Since version 0.1.4, the normals in the airfrans.Simulation class are inward-pointing instead of outward-pointing as previously"*, and `dataset.rst` records that the `.vtp` was built with `compute_normals(flip_normals=False)` "leading to inward-pointing normals". `build_cache.py` stores `-airfoil_normals` and asserts the flip two independent ways (A1.6). |
| "quadrature equivalent to first order" | **Refined — see A1.7**: the viscous term is *exactly* equivalent; only the pressure term carries a second-order difference. |

## A1.1 `Simulation` attributes — exact names, shapes, units

Built as `af.Simulation(root, name, T=298.15)`, where `root` is the directory
holding `manifest.json` and one sub-directory per sim (i.e. `data/raw/Dataset`).
All arrays are `float64`.

Air properties (functions of `T`, scalars):

| Attribute | Value at T = 298.15 K | Units |
|---|---|---|
| `MOL` | 28.965338e-3 | kg/mol |
| `P_ref` | 1.01325e5 | Pa |
| `RHO` | approx **1.1840** | kg/m^3 |
| `NU` | approx **1.54981e-5** | m^2/s |
| `C` | approx 346.1 | m/s |

Boundary conditions (parsed from the **name**, not read from the files):

| Attribute | Shape | Units |
|---|---|---|
| `inlet_velocity` | scalar | m/s |
| `angle_of_attack` | scalar | **rad** (the name stores degrees; the class converts) |

Mesh objects: `internal` (`pv.UnstructuredGrid` from `<name>_internal.vtu`,
`compute_cell_sizes(length=False, volume=False)` giving `cell_data['Area']`) and
`airfoil` (`pv.PolyData` from `<name>_aerofoil.vtp`,
`compute_cell_sizes(area=False, volume=False)` giving `cell_data['Length']`).

Fields, with `N` = internal nodes, `Na` = airfoil-patch nodes:

| Attribute | Shape | Units | Notes |
|---|---|---|---|
| `position` | (N, 2) | m | `internal.points[:, :2]` |
| `surface` | (N,) bool | — | `internal.point_data['U'][:,0] == 0` (no-slip) |
| `sdf` | (N, **1**) | m | `-implicit_distance`, >= 0 in the fluid |
| `normals` | (N, 2) | — | **inward**; exactly 0 off the surface |
| `input_velocity` | (N, 2) | m/s | freestream broadcast to every node |
| `velocity` | (N, 2) | m/s | target |
| `pressure` | (N, **1**) | **m^2/s^2** | target, **kinematic** p/rho |
| `nu_t` | (N, **1**) | m^2/s | target (upstream docs say "m^2 s^-2" — a typo) |
| `airfoil_position` | (Na, 2) | m | patch nodes |
| `airfoil_normals` | (Na, 2) | — | **inward** |

Note the trailing singleton axis on `sdf`, `pressure`, `nu_t` — a frequent
source of silent broadcasting bugs. The cache flattens them to `(Nv,)`.

`Na` equals `surface.sum()`: the same physical points **in a different order**.
`af.reorganize(in_order_points, out_order_points, q)` re-indexes between the two
by exact float equality of coordinates (an O(Na x Ns) NumPy loop; milliseconds).

**[verify after download]** actual `N` per sim (pre-crop about 180k; the release
is clipped to the box `[(-2,4), (-1.5,1.5), (0,1)]` and sliced at z = 0.5) and
`Na` (expected of order 10^3).

## A1.2 Pressure normalisation convention — kinematic, and rho cancels

`pressure` is the OpenFOAM incompressible variable, i.e. **p/rho in m^2/s^2**.
Three independent confirmations in source:

1. `simulation.rst`: "the air pressure on the internal mesh (divided by the
   specific mass in the incompressible case)".
2. `force(compressible=False)` multiplies the integral by `self.RHO`; the
   `compressible=True` branch does not.
3. `boundary_layer(compressible=False)` normalises by `0.5*inlet_velocity**2`
   (no rho); the `True` branch by `0.5*RHO*inlet_velocity**2`.

Because `force_coefficient` divides by `0.5*RHO*U^2` **after** `force` has
multiplied by `RHO`, **rho cancels exactly**. Consequence for `src/physics`
(CONTEXT.md section 6): calling `integrate_forces` on the *native* cached
`surf_p` / `surf_tau` with `rho=1.0, a_ref=1.0` reproduces the dataset
coefficients. Do **not** insert 1.184 anywhere.

## A1.3 Reference force / coefficient computation (what A2 must reproduce)

`Simulation.force_coefficient(compressible=False, reference=True)`, verbatim
chain:

```
wss = wallshearstress(over_airfoil=True)          # (Na,2), airfoil-patch order
p   = reorganize(position[surface], airfoil_position, pressure[surface])
airfoil.point_data['wallShearStress'] = wss
airfoil.point_data['p'] = p
airfoil = airfoil.ptc(pass_point_data=False)      # point data -> cell data
wp_int  = (-p_cell[:,None] * Normals_cell[:, :2] * Length[:,None]).sum(0)
wss_int = ( wss_cell                             * Length[:,None]).sum(0)
force_p = -wp_int  * RHO
force_v =  wss_int * RHO
basis   = [[ cos a, sin a],
           [-sin a, cos a]]
cp, cv  = (basis @ force_p)/(.5*RHO*U**2), (basis @ force_v)/(.5*RHO*U**2)
return ((cd, cdp, cdv), (cl, clp, clv))           # cd = cp[0]+cv[0], cl = cp[1]+cv[1]
```

Consequences:

* `force_p = +RHO * sum(p * N_inward * ds) = -RHO * closed_integral(p * n_out ds)`
  — the standard pressure force, which independently confirms `N` is inward.
  CONTEXT.md section 6's convention (`F = sum(-p n + tau) ds`, n outward) is
  therefore **already correct** once the cache stores outward normals.
* `basis` rotates by -alpha, so row 0 is the drag axis
  `e_inf = (cos a, sin a)` and row 1 the lift axis
  `e_perp = (-sin a, cos a) = rot90_ccw(e_inf)`. Matches CONTEXT.md section 6.
* Reference values are per unit span with chord = 1 m; there is **no area
  division** beyond `0.5 rho U^2`, hence `a_ref = 1.0`.
* **Return order is DRAG FIRST**: `((cd, cdp, cdv), (cl, clp, clv))`. Easy to
  transpose. The cache stores them under unambiguous names `cl_true`, `cd_true`,
  plus the pressure/viscous split `cl_p, cl_v, cd_p, cd_v`.

### Wall shear stress — not stored, derived exactly as the package does

`Simulation.wallshearstress`:

```
jacobian = internal.compute_derivative(scalars='U', gradient='jacobian')
           .point_data['jacobian'].reshape(-1,3,3)[surface, :2, :2]
s   = 0.5*(J + J.T);  s -= trace(s)/3 * I2        # deviatoric
wss = -(2*NU*s . N_inward)  ==  +2*NU*s . n_out
```

* Viscosity is the **molecular** `NU` only — `nu_t` is **not** included. The
  reference C_D therefore carries laminar wall stress only; do not "improve" it
  with an effective viscosity or gate G1 will fail.
* The gradient is taken on the **internal (volume) mesh** — wall shear stress
  cannot be recovered from surface data alone. This is why `build_cache.py` must
  run against the raw `.vtu`, and why `surf_tau` is cached rather than
  recomputed downstream.
* `over_airfoil=True` maps the result into airfoil-patch order via `reorganize`.
* Units: **m^2/s^2 (kinematic traction)** — multiply by `rho` for Pa.
* The deviatoric subtraction divides the trace by 3 (a 3-D convention applied to
  a 2-D tensor). For incompressible flow the trace is near zero so it is nearly
  a no-op, but it is not exactly zero numerically. Reproduce it verbatim.

## A1.4 Simulation-name grammar and NACA digit parsing

Format `airFoil2D_SST_<U>_<AoA_deg>_<naca params...>`, split on `_`:

| Fields | index 2 | index 3 | index 4: | Series |
|---|---|---|---|---|
| **7** | U (m/s) | AoA (deg) | 3 params `(M, P, XX)` | **NACA 4-digit** |
| **8** | U (m/s) | AoA (deg) | 4 params `(L, P, Q, XX)` | **NACA 5-digit** |

Examples from upstream docs:
`airFoil2D_SST_43.597_5.932_3.551_3.1_1.0_18.252` has 8 fields, so it is
**5-digit** with params `(3.551, 3.1, 1.0, 18.252)`; a NACA 0012 ends in
`_0_0_12`, 7 fields, so **4-digit**.

Critical detail: the parameters are **continuous real numbers, not integers** —
AirfRANS samples the NACA design space continuously, so `3.551` is a legitimate
"first digit". Any parser assuming integer digits will break. The last parameter
is always thickness in percent of chord; the preceding ones are the camber-line
parameters consumed by `airfrans.naca_generator.camber_line`, which itself
dispatches on `len(params) == 2` (4-digit) vs `== 3` (5-digit). This
3-vs-4 parameter count is the **only** discriminator; there is no explicit
series field anywhere in the dataset.

Implemented in `src/data/splits.py::parse_sim_name`, returning a `SimName` with
`u_inf, aoa_deg, naca_params, n_digits, thickness, camber_params, re`.

### Reynolds number, and the two viscosities

`Re = U*c/nu` with `c = 1 m`. **Two viscosities are in play** (upstream
`simulation.rst` note): the dataset was *generated* with `nu = 1.56e-5`, whereas
`Simulation.NU` recomputes about `1.54981e-5` from `T`. The paper's
Re in [2e6, 6e6] and the official `reynolds` band [3e6, 5e6] follow the
**1.56e-5** convention, so `src/data/splits.py::NU_DATASET = 1.56e-5` drives all
Re-derived split logic (about a 0.7 % offset from `Simulation.NU`, immaterial to
band membership away from the edges). `Simulation.NU` is still cached per sim as
`nu` for physics use.

## A1.5 Official task split mechanism

`af.dataset.load(root, task, train)` reads **`<root>/manifest.json`** — a single
JSON object mapping `"<task>_<split>"` to a list of sim names. Keys used:
`full_train`, `full_test`, `scarce_train`, `reynolds_train`, `reynolds_test`,
`aoa_train`, `aoa_test`.

The `scarce` task **has no test set of its own**: `load` contains
`taskk = 'full' if task == 'scarce' and not train else task`, so scarce reuses
`full_test`. `src/data/splits.py` replicates this exactly.

Task definitions (upstream `dataset.rst`):

| Task | Train | Test |
|---|---|---|
| `full` | 800 sims | 200 sims, same distribution (interpolation) |
| `scarce` | 200 sims | identical to `full_test` (low-data interpolation) |
| `reynolds` | Re in [3e6, 5e6] | Re outside (extrapolation) |
| `aoa` | AoA in [-2.5, +12.5] deg | AoA outside (extrapolation) |

`src/data/splits.py` reads this JSON **directly**; it never calls
`dataset.load`, which would open all 1000 `.vtu` files (minutes of I/O and many
GB of RAM) merely to recover a list of names. Splits therefore need neither
`airfrans` nor `pyvista` installed.

**[verify after download]** exact train/test counts for `reynolds` and `aoa`.

### Derived splits (CONTEXT.md section 5)

* `shape5` — train = every 4-digit sim, test = every 5-digit sim.
* `combined` — train = 4-digit **and** Re in [3e6, 5e6]; test = 5-digit **and**
  Re outside that band. Sims in neither bucket are deliberately unused, which
  maximises the joint shape + Reynolds shift. The band is pinned to the official
  `reynolds` train band so `combined` is a strict strengthening of that shift.
* Calibration carve, one rule for all six splits:
  `cal_n = min(100, max(1, round(0.20 * n_train)))`, taking the **last**
  `cal_n` entries of `shuffle(sorted(train), seed=0)`. For `full` (800 train)
  this is exactly CONTEXT.md's "last 100 of shuffled(seed=0) train"; for
  `scarce` (200 train) exactly "20 % of train". Sorting before shuffling makes
  the outcome independent of the ordering inside the upstream manifest.

## A1.6 Surface point ordering and outward normals (our own construction)

The package provides **no contour ordering** — `airfoil_position` is in VTK file
order. CONTEXT.md section 4 requires "ordered along the airfoil contour
(TE -> around -> TE)", so `build_cache.py` derives it:

1. Read the `.vtp` line connectivity (`airfoil.lines`, the flat VTK array
   `[n, i0, i1, ...]`) into edges and build adjacency.
2. Assert every node has degree 2 (closed contour) or that exactly two nodes
   have degree 1 (open contour, blunt TE). Anything else raises.
3. Start at `argmax(x)` — the trailing edge — and walk the curve.
4. If the shoelace signed area is negative, reverse the walk (keeping the TE
   first) so it is **counter-clockwise**: TE -> extrado -> LE -> intrado -> TE.

Outward normals are `-airfoil_normals`, renormalised, then re-indexed by the
walk. Two independent assertions guard the flip:

* **alignment** — a CCW tangent implies `n_geo = (t_y, -t_x)`; we require
  `mean(n_out . n_geo) > 0.9`. This is purely geometric and does not trust the
  `.vtp` normals at all.
* **closure** — `norm(sum(n_out * ds)) / perimeter < 5e-3`, the divergence
  theorem on a closed curve.

Both are recorded per sim in the cache `manifest.json` as `normal_alignment` and
`normal_closure`; `--no-strict` downgrades failures to warnings.

## A1.7 Quadrature weights — exactness analysis

The cache stores `surf_ds[i] = 0.5 * sum(length of facets incident to i)`, so
`sum(surf_ds) == perimeter` (also cached, as `perimeter`).

Compared with the reference `ptc`-then-`sum(v_cell * L_cell)` scheme:

* **Viscous term: exactly identical.** For a two-node line cell, `ptc` averaging
  is `(v_a + v_b)/2`, and `sum_c ((v_a+v_b)/2 * L_c)` is algebraically identical
  to `sum_p (v_p * ds_p)`. The tau contribution matches to machine precision.
* **Pressure term: a second-order difference.** The reference multiplies the
  cell-averaged pressure by the cell-averaged *normal* (`ptc` averages `Normals`
  too, and does **not** renormalise), i.e. `p_bar_c * N_bar_c`, whereas the
  point scheme integrates `(p_a N_a + p_b N_b)/2`. The per-cell discrepancy is
  `L_c * dp * dN / 4`, i.e. O(h^2); with of order 10^3 surface points it sits
  far inside the 1 % G1 tolerance. **A2: if G1 lands just above 1 %, check this
  first** — reproducing `p_bar_c * N_bar_c` exactly would require cell-wise
  arrays, which the frozen section 4 schema does not carry.

## A1.8 Cache schema as built (`data/processed/airfrans/<sim>.npz`)

Every CONTEXT.md section 4 key is present with the mandated shapes and dtypes
(`float32`, C-order). `build_cache.py` additionally stores, purely additively:

| Extra key | Shape | Why |
|---|---|---|
| `vol_is_surf` | (Nv,) uint8 | which subsampled volume nodes lie on the wall (no-slip residual) |
| `cl_p, cl_v, cd_p, cd_v` | scalar | pressure/viscous split of the reference coefficients, so A2 can localise a G1 failure to one term |
| `rho`, `nu` | scalar | `Simulation.RHO` / `.NU`, so nothing downstream re-derives them |
| `perimeter` | scalar | `sum(surf_ds)`, a cheap cache-integrity check |
| `n_vol_full` | scalar | node count before subsampling |
| `cache_version` | scalar | idempotency and staleness detection |

Volume subsampling: uniform without replacement to at most 32768 nodes, seeded
per sim by `blake2s(sim_name)` (platform-independent, unlike Python's `hash`),
with the selected indices sorted so on-disk order is stable.

`norm_stats.json` is computed **only** over the `train` list of
`data/splits/full.json`, so `cal` and `test` never touch normalisation (CONTEXT
section 5). It streams count / sum / sum-of-squares, so no 1000-sim array is
ever materialised.

**Positions are deliberately NOT normalised by default.** Chord = 1 m so
`surf_pos` is already O(1), and per-component standardisation would stretch y by
roughly 10x relative to x, destroying the aspect ratio that the geometry-aware
models depend on. The default normalised set is
`{surf_p, surf_tau, cl_true, cd_true, cond, vol_u, vol_p, vol_nut}`, while
`surf_ds`, `surf_normal` and `vol_sdf` are **always physical** because
`integrate_forces` and the no-slip residual consume them directly. Statistics
for the un-normalised fields are still written, so a config can opt in.

## A1.9 Miscellany worth knowing

* `dataset.load` returns `(list of (N, 12) arrays, list of names)` with columns
  `[pos(2), inlet_velocity(2), sdf(1), normals(2), velocity(2), pressure(1),
  nu_t(1), surface(1)]`. We do not use it — see A1.5.
* `Simulation.sampling_volume / sampling_surface / sampling_mesh` provide
  mesh-free sampling; `sampling_surface` normals are inward as well. Unused
  here, but the upstream note that *"the test scores of models for the four
  target fields have to be computed at the position of the nodes of the
  simulation meshes"* is a **protocol constraint**: never report field metrics
  on resampled points.
* Preprocessed release: the internal mesh is clipped to
  `[(-2,4), (-1.5,1.5), (0,1)]` with `crinkle=True` and sliced at z = 0.5; the
  `.vtp` patches are sliced likewise. The airfoil patch retains `p`, `U`, `nut`
  and `Normals` as point data, but the cache takes `p` via `reorganize` from the
  internal mesh so that it matches `force()` exactly.
* A `<name>_freestream.vtp` (external boundary) also ships. Unused.
* Licence **ODbL-1.0** — we commit split manifests and code, never processed
  field data.

## A1.10 Pre-download validation harness, and the float32 precision budget

Because the 10 GB download had not landed, `build_cache.py` was validated
against a **synthetic simulation written in the exact AirfRANS on-disk format**
(an O-grid around a circle: `<name>_internal.vtu` with `U`, `p`, `nut`,
`implicit_distance`, plus a closed-polyline `<name>_aerofoil.vtp` carrying
inward `Normals`, and a `manifest.json`). `airfrans.Simulation` ingests it
unmodified, so every code path — `surface` masking, `reorganize`,
`wallshearstress`, `force_coefficient`, `ptc` quadrature — is genuinely
exercised. The generator lives at
`scratchpad/synth_dataset.py` + `scratchpad/verify_cache.py`; it is worth
promoting to `tests/test_build_cache.py` (outside A1's assigned paths, so left
as a proposal).

Verified on the circle, where every quantity is analytic:

* `sum(surf_ds)` = 3.141032 vs `2*pi*R` = 3.141593 (a 96-point polygon
  under-resolves the arc by exactly the expected O(h^2) chord deficit);
* `mean(n_out . r_hat)` = 1.000000 — the inward->outward flip is right;
* `norm(sum(n*ds))/perimeter` = 1.8e-8 — the closed-surface identity;
* contour walk starts at `argmax(x)` and has positive signed area (CCW);
* dtypes `float32`, C-contiguous, shapes exactly per section 4.

### Force reproduction: the quadrature scheme is exact, float32 is the budget

Reproducing the reference coefficients from the cached arrays via the
CONTEXT.md section 6 formula (`F = sum(-p n + tau) ds`, `rho=1`, `a_ref=1`):

| Precision of the cached surface fields | rel. error on C_D |
|---|---|
| float64 | **0.000 %** (all four synthetic sims) |
| float32 | 3.5 % / 5.6 % on the circle |

So the concern raised in A1.7 — that the reference's `p_bar_c * N_bar_c`
cell-product differs from our point quadrature — is **empirically nil**: in
float64 the two agree to every printed digit. The entire float32 residual is
**catastrophic cancellation**, and it is quantifiable. Define

```
cancellation ratio  K = max|p| * perimeter / (0.5 * U^2 * |coefficient|)
```

which is how many orders of magnitude the summands exceed the result. float32
carries ~7 significant decimal digits, so the expected relative error on the
coefficient is roughly `K * 1e-7`.

* The synthetic circle is a pathological case: its pressure field is nearly
  fore-aft symmetric, so C_D is pure numerical residue and **K ~ 1e6**,
  predicting a few-percent error — exactly what is observed. This is an artefact
  of the test geometry, **not** of the cache.
* Real AirfRANS airfoils have `C_D ~ 5e-3 .. 2e-2`, `C_L ~ 0.1 .. 1.5`,
  `max|p|/(0.5 U^2) ~ O(1)` and perimeter ~2, giving **K ~ 1e2 .. 1e3** and a
  predicted float32 error of **1e-5 .. 1e-4 relative** — two to three orders of
  magnitude inside the 1 % G1 tolerance.

**Conclusion for A2 / gate G1:** float32 storage (frozen by CONTEXT.md section
4) is comfortably adequate for the real dataset, and the point-wise `surf_ds`
quadrature is not merely first-order-equivalent to the package's scheme but
numerically indistinguishable from it. If G1 nevertheless fails, the cause is
*not* precision or quadrature — look at sign conventions, the drag-first return
order of `force_coefficient` (A1.3), or the inclusion of `nu_t` in the wall
stress (A1.3).

**[verify after download]** re-run `verify_cache.py`'s G1 block on 20 real sims
and confirm the median relative error lands at 1e-4 or better.
