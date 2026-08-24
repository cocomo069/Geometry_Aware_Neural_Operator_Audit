# FLUENT_PLAN.md — 2D RANS verification: design, grid study, offset study

Companion to [`../fluent/README.md`](../fluent/README.md) (the operating manual).
This file holds the *engineering design*: why the meshing route was chosen, the
y+ and boundary-layer spacing mathematics, the grid-independence study and its
GCI arithmetic, the AirfRANS-replication offset study, and the acceptance
criteria that decide whether a case counts.

Status: **authored, not executed.** Fluent runs are deferred by decision D-007.
Owner: agent A6. Consumers: spec §5.8 (active learning), §8 Table 5, and the
paper's appendix B.

---

## 1. What this component has to deliver

Three numbers, in order of importance:

1. **The solver offset** Δ between our Fluent setup and the OpenFOAM setup
   behind AirfRANS, with an uncertainty. Without it, every surrogate-vs-Fluent
   disagreement is confounded by solver, turbulence model, mesh topology and
   convergence criteria, and no statement about surrogate error survives review.
2. **A numerical uncertainty band** on our own Fluent coefficients, from a
   three-level grid study. A CFD reviewer will not read past its absence.
3. **Independent CD and CL** for the 6–12 active-learning-selected cases, so
   the falsifiable claim — that high-acquisition cases really do have higher
   surrogate error than random ones — can be tested against something that did
   not generate the training data.

Everything else in `fluent/` exists to make those three reproducible.

---

## 2. Meshing route: why a direct writer, not gmsh and not ICEM

The brief allowed either an ICEM Tcl replay or a gmsh-based generator. The
implemented primary route is **neither**: `fluent/mesh_gen.py` builds the C-grid
algebraically in numpy and writes the native Fluent 2D `.msh` itself. The
reasoning, since this is a deviation worth defending:

**The topology is not something a mesher needs to discover.** A C-grid around an
airfoil is a single structured block with a known index map. An unstructured
mesher's value is handling geometry you did not parametrise; here we parametrise
the geometry ourselves from the NACA equations. There is nothing to discover.

**Both alternative routes still need a hand-written Fluent writer, or lose
control.** `meshio`'s Fluent writer silently drops boundary face zones and is
unusable. gmsh cannot write a Fluent `.msh` at all — a gmsh route would produce
a gmsh `.msh` that then needs exactly the writer in `mesh_gen.py`, so gmsh would
only replace the (trivial) algebra while adding a ~60 MB binary wheel that
`CONTEXT.md` §1's "no compiled extensions" rule is written to avoid. ICEM *can*
write the Fluent mesh through its own output interface, which is a real
advantage — but it makes the mesh a product of an interactive tool's replay
state rather than of a readable script, and it puts a licensed product on the
critical path of *mesh generation* as well as solving.

**The two things this study actually needs are exactly the two things a direct
writer gives.** The y+ target is a statement about the first cell height, and
GCI is a statement about a controlled refinement ratio. Both are one-line
parameters here and are guaranteed exactly; through a mesher's sizing functions
they are requests that get approximately honoured.

**It is deterministic and diffable.** No seed, no tool version, no replay state.
The same command produces the same bytes, which is what "reproducible CFD" has
to mean if it is going to be claimed in a paper.

**Cost of the choice, stated honestly.** A hand-written mesh cannot be validated
against Fluent's own grid checker at authoring time, and an algebraic C-grid has
no elliptic smoother, so its worst cells (the trailing-edge/wake fold) are worse
than ICEM's would be. Two mitigations:

- `mesh_gen.py --self-check` re-parses the written file and asserts index
  ranges, owner/neighbour consistency, non-degenerate faces, no orphaned cells
  and matching section counts. It does *not* check face winding, which only
  Fluent can confirm — that is enforced by construction (see §2.2).
- `fluent/templates/mesh_cgrid.rpl` is a complete ICEM Tcl replay kept as an
  independent cross-check route, to be used if the quality metrics below are
  ever judged inadequate. It is a fallback, not dead weight: `icemcfd_run_script`
  is available in-session.

### 2.1 Topology

Single-block C-grid, index `(i, j)`, right-handed everywhere:

```
i = 0      .. nw          lower wake cut, outlet plane -> trailing edge
i = nw     .. nw+na       airfoil, TE -> lower -> LE -> upper -> TE
i = nw+na  .. ni          upper wake cut, trailing edge -> outlet plane
j = 0                     body + wake-cut line
j = nj                    far-field C: upstream semicircle + two downstream lines
```

The wake cut is handled the way a real C-grid handles it: the `j = 0` nodes of
the upper branch are given **the same node ids** as the lower branch's, so the
faces along the cut emerge as ordinary interior faces. No periodic or interface
boundary condition is needed, and the cut is invisible to the solver.

Zones written: `airfoil` (wall, type 3), `farfield` (velocity-inlet, type 10),
`outlet` (pressure-outlet, type 5), `interior` (type 2).

### 2.2 Three construction details that had to be got right

These are recorded because each one produced a *folded* mesh (negative cell
Jacobian) before it was fixed, and a folded mesh is not a quality problem, it is
an invalid mesh:

1. **The far-field distribution must relax toward uniform.** The inner line is
   exponentially clustered (at the TE along the wake, at the LE and TE on the
   section). Propagating that clustering out to r = 30 c leaves far-field
   spacings of ~1e-5 c beside cells 30 c tall. Each block's outer distribution
   is therefore relaxed to uniform (`--outer-uniformity`, default 1), which is
   what a hyperbolic or elliptic C-grid does anyway.
2. **The wall-normal/radial direction blend must run on the index fraction, not
   the distance fraction.** The radial distribution is exponentially stretched,
   so a distance-based blend is still tilted away from the far-field ray at
   `j = nj-1` and then snaps to it at `j = nj` — a lateral jump that folds cells
   across the whole airfoil block. The blend is `smoothstep(j/nj)`, which has
   zero derivative at both ends: orthogonal at the wall (what the y+ sizing
   assumes) and kink-free at the far field.
3. **Each wake branch's normal must relax from the trailing-edge surface normal
   to the vertical**, over ~1 chord. Otherwise the ray family is discontinuous
   across the trailing-edge fold — the last wake node carries exactly `(0,-1)`
   while the TE node one index away carries the lower-surface normal a few
   degrees off, and the near-parallel rays 1e-4 c apart produce inverted slivers.

Two further sizing rules, for quality rather than validity:

- The wake's first cell is seeded from the **mean** surface panel length, not
  the first panel. The loop is cosine-clustered so the first panel is
  quadratically small (~3e-4 c); seeding the wake with it drives the wake growth
  ratio to 1.27 and drops minimum orthogonality from 0.29 to 0.12.
- The wall-normal first cell is allowed to **grow** where the streamwise spacing
  is large (`AR_CAP = 5000`). The y+ target is a statement about the airfoil
  boundary layer; a 1e-5 m first cell 29 chords downstream in the wake buys
  nothing and costs an aspect ratio of ~4e5. An assertion enforces that the cap
  never moves the first cell **on the section**, so the y+ sizing is untouched.

Face orientation follows the verified convention: the clockwise normal of the
edge `n0 → n1`, i.e. `(dx, dy) → (dy, -dx)`, points **out of** the owner cell
`c0` for every face, boundaries included. If the interior i-faces and j-faces
disagree on that convention Fluent aborts with "Build Grid: Aborted due to
critical error" even though the zones read fine.

### 2.3 Measured mesh quality (NACA 0012, Re 3×10⁶, α = 5°)

| Level | Cells | ni × nj | min Jacobian | min orthogonality | max aspect ratio | wall-normal growth |
|---|---|---|---|---|---|---|
| 1 coarse | 18 796 | 254 × 74 | > 0 | 0.284 | 5.4×10³ | 1.189 |
| 2 medium | 43 008 | 384 × 112 | > 0 | 0.290 | 5.2×10³ | 1.121 |
| 3 fine | 96 768 | 576 × 168 | > 0 | 0.294 | 5.2×10³ | 1.079 |

Spot checks: NACA 23012 at α = 10° → min orthogonality 0.225; NACA 4412 at
α = −5° → 0.238. The worst cells are always the two sliver columns at the
trailing-edge fold, which is characteristic of any C-grid. Growth ratios stay
below the 1.2 rule of thumb at every level.

---

## 3. y+ target and boundary-layer spacing

### 3.1 The sizing rule

Flat-plate turbulent correlation evaluated at `x = c`:

```
Re_c   = ρ U∞ c / μ
C_f    = 0.026 · Re_c^(−1/7)          (Schlichting 1/7-power law)
τ_w    = C_f · ½ ρ U∞²
u_τ    = √(τ_w / ρ)
y₁     = y⁺ · μ / (ρ u_τ)             (wall to first cell CENTRE)
Δy₁    = 2 y₁                         (first cell HEIGHT — what the mesh uses)
```

The factor of two matters and is a common silent error: y+ is defined at the
first cell *centre*, so a mesh built with the first cell *height* equal to `y₁`
lands at y+ ≈ 0.5 of the target, not at the target.

Worked example, the baseline case (NACA 0012, Re = 3×10⁶, c = 1 m,
ρ = 1.184 kg/m³, μ = 1.85×10⁻⁵ Pa·s, so U∞ = 46.875 m/s):

```
C_f  = 0.026 · (3×10⁶)^(−1/7) = 3.09×10⁻³
τ_w  = 4.02 Pa
u_τ  = 1.842 m/s
y₁   = 0.6 · 1.85×10⁻⁵ / (1.184 · 1.842) = 5.09×10⁻⁶ m
Δy₁  = 1.02×10⁻⁵ m  ( = 1.0×10⁻⁵ c )
```

### 3.2 Why the target is 0.6 and not 1.0

The correlation is a flat-plate estimate at `x = c`. On an airfoil the real y+
peaks near the leading edge and under the suction peak, typically 1.5–2.5× the
flat-plate value, and the correlation ignores the pressure gradient entirely. A
target of 0.6 leaves headroom so the *achieved maximum* stays below 1 and both
turbulence models run genuinely wall-resolved, with no switch to wall functions
part-way along the chord.

This is an **a priori sizing rule only**. The journals report the achieved y+
(`*_yplus_max.txt`, `*_yplus_avg.txt`) and those are the numbers Table 5 quotes.
If the achieved maximum exceeds 1, the case is rejected and re-meshed — see §6.

### 3.3 Growth and far-field extent

Given `Δy₁`, the number of wall-normal cells `nj` and the far-field radius, the
geometric growth ratio is solved by bisection **per ray** so that each ray's
`nj` cells span its own length starting from its own first cell. The generator
warns if the ratio on the section exceeds 1.2. Far field at 30 c and outlet at
30 c downstream: far enough that the velocity-inlet condition on the whole C
does not distort the near field at the angles of attack in scope (|α| ≤ 15°).

---

## 4. Grid-independence study

### 4.1 Design

Three levels, on **one representative case**: NACA 0012, Re = 3×10⁶, α = 5°
(`gridstudy_naca0012_re3e6_a5_L{1,2,3}` in `fluent/cases_to_run.json`). The case
is chosen to be boring on purpose — symmetric section, mid-Re, attached flow, so
CL and CD are both well posed and neither sits near a bifurcation where the grid
study would be measuring physics instead of discretisation.

Both turbulence models are run at all three levels: a grid study that only holds
for SA tells you nothing about the SST results reported beside it.

**Refinement is systematic in every direction, including at the wall.** Cell
counts scale by ≈1.5 per direction per level, and the y+ target scales by 1/1.5
alongside them (0.90 / 0.60 / 0.40), so `Δy₁` refines with everything else. All
three levels stay wall-resolved (y+ < 1), so no wall-treatment switch
contaminates the comparison. Holding `Δy₁` fixed across levels — common RANS
practice — would have broken the systematic-refinement assumption that
Richardson extrapolation rests on, and is deliberately not done here.

| Level | Cells N | h ∝ N^(−1/2) (rel.) | y+ target | Δy₁ (m) |
|---|---|---|---|---|
| 3 fine (φ₁) | 96 768 | 1.000 | 0.40 | 6.79×10⁻⁶ |
| 2 medium (φ₂) | 43 008 | 1.500 | 0.60 | 1.02×10⁻⁵ |
| 1 coarse (φ₃) | 18 796 | 2.269 | 0.90 | 1.53×10⁻⁵ |

Refinement ratios `r₂₁ = h₂/h₁ = 1.500` and `r₃₂ = h₃/h₂ = 1.513`, both above
the 1.3 minimum the GCI procedure requires, and near enough to equal that the
constant-r simplification is defensible (the variable-r form is used anyway).

### 4.2 GCI arithmetic (Celik et al., ASME JFE 2008)

Applied separately to CD and CL, and separately per turbulence model.
Representative cell size in 2D, with `ΔA_i` the cell area:

```
h = [ (1/N) Σ ΔA_i ]^(1/2)
```

With `φ₁` fine, `φ₂` medium, `φ₃` coarse, `ε₂₁ = φ₂ − φ₁`, `ε₃₂ = φ₃ − φ₂`,
`s = sign(ε₃₂/ε₂₁)`, the apparent order `p` solves, by fixed-point iteration:

```
p = |ln|ε₃₂/ε₂₁| + q(p)| / ln(r₂₁),    q(p) = ln[ (r₂₁^p − s) / (r₃₂^p − s) ]
```

then

```
φ_ext = (r₂₁^p φ₁ − φ₂) / (r₂₁^p − 1)          Richardson extrapolate
e_a   = |(φ₁ − φ₂) / φ₁|                        approximate relative error
GCI_fine = 1.25 · e_a / (r₂₁^p − 1)             reported uncertainty band
```

The factor 1.25 is the standard three-grid safety factor.

### 4.3 What is reported, including when it goes wrong

- `p` alongside the formal second order. `p` far from 2 is informative, not
  embarrassing: `p` well below 2 means the solution is not yet in the asymptotic
  range; `p` above ~2.5 usually means the three points are noise-dominated
  because the differences are near the convergence tolerance.
- **Oscillatory convergence** (`ε₃₂/ε₂₁ < 0`) is reported as such, with the GCI
  computed but flagged, not quietly hidden.
- The GCI on the **medium** grid is the numerical uncertainty band attached to
  every other case in the study, since every other case runs at level 2.
- **Target:** GCI_fine ≤ 1% on CD and ≤ 0.5% on CL. If CD misses it, the
  production level moves to 3 and the cost is absorbed; the band is reported as
  measured either way, and a band that fails the target is reported rather than
  the target being relaxed after the fact.

A separate, larger uncertainty is the **SA↔SST spread** on the same grid. It is
not numerical uncertainty and is never folded into the GCI; it is reported
alongside as turbulence-model sensitivity, and it will almost certainly dominate.

---

## 5. AirfRANS-replication offset study

### 5.1 The point

AirfRANS was generated with OpenFOAM `simpleFoam` and Spalart–Allmaras on its
own meshes and convergence criteria. Our Fluent setup differs in solver,
discretisation, mesh topology and stopping rule. Until that difference is
measured, a surrogate-vs-Fluent gap cannot be attributed to the surrogate. This
is the study's single most confounding risk, and spec §5.8 marks it essential.

### 5.2 Which simulations to replicate

**Six sims, drawn only from AirfRANS `full`-task test splits**, so no sim used
here has been seen by any trained model. Selection is stratified rather than
random, because with n = 6 a random draw is likely to miss an axis entirely:

| # | Selection rule | Why this cell |
|---|---|---|
| 1 | symmetric 4-digit, low α (≈0–2°), mid Re | the cleanest possible comparison: attached, thin wake, minimal model sensitivity — isolates *setup* differences from *physics* differences |
| 2 | symmetric 4-digit, high α (≈10–12°), mid Re | where turbulence-model differences first bite; the offset is expected to grow here and that growth is itself a result |
| 3 | cambered 4-digit, mid α, mid Re | camber activates the CL offset, which case 1 cannot see |
| 4 | 4-digit at the **low** Re edge (≈2×10⁶) | the offset may be Re-dependent; two edge points test that |
| 5 | 4-digit at the **high** Re edge (≈6×10⁶) | as above, and the high-Re case is where our y+ sizing is most stressed |
| 6 | a **5-digit** section, mid conditions | the `shape5` split tests on 5-digit, so an offset measured only on 4-digit sections would not transfer to the split that matters most |

Exact sim ids are pinned once `docs/DATA_NOTES.md` exists (A1 owns the sim-name
parsing) and are then **frozen and committed** — this list must not be re-drawn
after seeing results.

Replication means matching NACA digits, Re, α, chord and fluid properties
exactly. The fluid properties in `fluent/cases_to_run.json` (ρ = 1.184 kg/m³,
μ = 1.85×10⁻⁵ Pa·s, air at 298.15 K) are what the AirfRANS generator states it
used and **must be verified against `DATA_NOTES.md`** before these runs: if our
freestream state differs from theirs, the offset study measures the wrong thing.

### 5.3 What is measured

For each replicated sim, both turbulence models:

```
Δ_CD = CD_fluent − CD_airfrans          Δ_CL = CL_fluent − CL_airfrans
δ_CD = Δ_CD / CD_airfrans               δ_CL = Δ_CL / |CL_airfrans|
```

Reported as mean ± sample standard deviation over the six, **per model**, with
all six values shown individually — six points is far too few for the summary
statistic to stand alone.

Beyond the integrated coefficients, the **surface distributions** are compared:
cp and cf against arc length, ours vs AirfRANS on the same axes. This is what
localises the disagreement — a leading-edge suction-peak mismatch means mesh
resolution, a trailing-edge recovery mismatch means turbulence model, and a
uniform offset means reference-value or convention mismatch. Integrated
coefficients alone cannot distinguish those three.

### 5.4 How the offset is used

The SA offset (like-for-like with AirfRANS's own model) is the one applied. On
the active-learning verification cases, the surrogate prediction is compared
against `CD_fluent − Δ̄_CD`, and the uncertainty on `Δ̄` is propagated into the
comparison. The paper states plainly that with n = 6 this correction carries a
wide interval and that the Fluent set is a qualitative external check rather
than a statistical test.

If the SA↔SST spread exceeds |Δ̄|, that is reported as the headline caveat: it
would mean turbulence-model choice matters more than the solver offset, and no
solver-offset correction can rescue a comparison at that precision.

---

## 6. Acceptance criteria

A case counts only if **all** of these hold. They are checked from the files on
disk, not from an assumption that a completed run means correct data.

**Mesh**
- `min_jacobian > 0` in `*_mesh.json`. Non-positive means folded cells — never
  solve on it, no exceptions.
- `min_orthogonality ≥ 0.15`, `max_aspect_ratio ≤ 10⁴`, wall-normal growth
  ratio on the section ≤ 1.2.
- `--self-check` passes on the written `.msh`.
- Fluent's own `/mesh/check` reports positive volumes and no critical errors.

**Wall resolution**
- achieved **max** y+ < 1.0 over the airfoil (`*_yplus_max.txt`).
- achieved area-averaged y+ < 0.7 (`*_yplus_avg.txt`).
- Failure → re-mesh with a smaller y+ target; do **not** reinterpret it as
  a wall-function case, because the two models' near-wall treatments would then
  differ from the grid study's.

**Convergence — both conditions, not either**
- scaled residuals at or below the stated criteria (1×10⁻⁶ per equation), and
- CD flat to the stop criterion (relative change ≤ 1×10⁻⁵) over the last 150
  iterations, from `*_coeffs.out`.

A converged residual with a drifting CD is **not** a converged case: external-aero
residuals routinely plateau while the coefficients still move in the third
significant figure, and CD is the reported quantity. If residuals plateau above
criterion but CD is genuinely flat, the case is accepted with the plateau value
stated explicitly in the appendix — a plateau on a physically mildly-unsteady
flow is expected behaviour, not a failure, but it has to be declared.

**Physical plausibility**
- CL sign and rough magnitude consistent with thin-airfoil theory
  (`CL ≈ 2π·α_rad` within a factor of ~1.3 at small α for the symmetric case).
- CD in the 10⁻²–10⁻¹ decade; a CD below 5×10⁻³ on a 12%-thick section at
  Re 10⁶ means the viscous contribution is missing, most likely from a wall
  boundary condition that did not take.
- No reverse flow reported at the outlet.

**Grid study**
- all three levels converged by the criteria above **before** any GCI is
  computed — a GCI over unconverged solutions is arithmetic on noise;
- apparent order `p` reported, oscillatory convergence flagged;
- GCI_fine ≤ 1% on CD, ≤ 0.5% on CL, or the production level is raised.

**Batch integrity**
- cell count read from the saved `.cas.h5` itself matches the mesh that case was
  supposed to use. The overwrite/discard prompt failure mode silently makes a
  case solve on the *previous* mesh and save plausible-looking results under the
  right filename; the only reliable defence is checking the file, and this
  exact failure has corrupted a real multi-case batch before.
- `*_surface.csv` columns mapped by header name, never by position.

---

## 7. Cost estimate and scheduling

Rough, to be replaced by measurements (honest cost accounting is itself a
contribution for a small-compute study, per spec §6):

| Item | Cells | Runs | Est. wall-clock each |
|---|---|---|---|
| Grid study | 18.8k / 43k / 96.8k | 6 (3 levels × 2 models) | 5 / 12 / 30 min |
| Offset study | 43k | 12 (6 sims × 2 models) | ~12 min |
| Active-learning verification | 43k | 16–24 (8–12 cases × 2 models) | ~12 min |

Total ≈ 8–12 CPU-hours on the 6-core i7-8850H, sequential. That fits comfortably
inside the spec's "1 to 3 days" allowance for the Fluent component and can be
run overnight in one background batch — but only once the user has green-lit CPU
use (D-007), and only after the Step 0 keyword probe.
