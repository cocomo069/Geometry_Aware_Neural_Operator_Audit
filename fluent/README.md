# `fluent/` — 2D RANS verification pipeline

Independent solver-side verification for the neural-operator study: a
parameterised, reproducible Ansys Fluent pipeline for 2D steady incompressible
RANS over NACA 4- and 5-digit airfoils.

**The campaign has run** (D-025, D-026; v211, 6 cores). The grid study is
grid-converged at L2; the 4 low-AoA random cases converged; 20 cases (all 16
acquisition+variance α=18° picks, 3 random α=18°, 1 random α=12°) diverged and
are being retried with the conservative `r1` variant. The solver-offset study
(6 AirfRANS replicas) is running. Post-processing lives in
`scripts/collect_fluent.py` (classify + GCI) and `scripts/compare_fluent.py`
(surrogate vs Fluent). The design rationale, grid study, y+ mathematics and
acceptance criteria live in [`../docs/plans/FLUENT_PLAN.md`](../docs/plans/FLUENT_PLAN.md);
this file is the operating manual.

## Why this component exists

Two things, neither of which is "another CFD dataset":

1. **Solver-to-solver offset.** AirfRANS was generated with OpenFOAM
   (`simpleFoam`, Spalart–Allmaras). Any disagreement between a surrogate and
   *our* Fluent result is confounded by solver, turbulence model, mesh topology
   and convergence criteria unless that offset is measured first. Replicating a
   handful of AirfRANS test cases and quantifying the offset is what makes every
   later comparison interpretable (spec §5.8, point ii).
2. **Independence for the active-learning loop.** The falsifiable claim is that
   cases chosen by the acquisition score are genuinely harder than randomly
   chosen ones. Verifying that against the same generator that produced the
   training data would be circular; it needs an independent solver.

The set is small (6–12 cases plus a grid study) and the paper says so: it is a
qualitative external check, not a statistical test.

## Contents

| Path | What it is |
|---|---|
| `mesh_gen.py` | Parametric C-grid generator. Pure numpy → native Fluent 2D `.msh`. y+ → first-cell-height sizing + self-check. `--naca DDDD` for integer digits, or `--naca-params M,P,T` / `L,P,Q,T` (continuous, via `airfrans.naca_generator`) for the offset replicas. |
| `make_cases.py` | Reads `cases_to_run.json`, instantiates journals + params into `cases/<case_id>/`, writes `cases/MANIFEST.json` + `RUNBOOK.md`. `--variant r1` renders the `<case>_<model>_r1` retry journals (+ `MANIFEST_r1.json`); `--only-file` restricts to a case-id list; per-case `mu`/`naca_params` supported. |
| `make_offset.py` | Freezes the 6 AirfRANS-replica offset cases, runs the geometry gate vs cached `surf_pos`, appends them (per-case `mu`/`re`) to `cases_to_run.json`, writes `offset_cases.txt`. |
| `cases_to_run.json` | The case manifest: grid-independence trio + 24 active-learning picks (from `run_active.py`) + 6 offset replicas. |
| `templates/case_template.jou` | The Fluent TUI journal template, one `{{SLOT}}` per parameter. |
| `templates/case_template_r1.jou` | The `r1` robustness-retry template (longer first-order start, reduced URFs, no cd-steady stop). |
| `diverged_cases.txt` / `offset_cases.txt` | Case-id lists for `run_batch.py --only-file` (the 20 diverged / the 6 offset). |
| `templates/probe_bc_keywords.jou` | **Step 0.** Dumps the TUI keyword names this Fluent build actually accepts. |
| `templates/mesh_cgrid.rpl` | ICEM CFD Tcl replay, the fallback meshing route. |
| `cases/` | Generated output. Regenerate at will; nothing here is hand-edited. |

## How the orchestrator runs it

The Ansys MCP tools are available in-session. All run tools are **asynchronous**:
they return a `job_id` immediately and the job continues in the background.

```
Step 0  probe          fluent_run_journal(templates/probe_bc_keywords.jou)
                       → read the transcript, reconcile keywords, fix the template
Step 1  meshes         .venv/Scripts/python.exe fluent/make_cases.py --write-mesh
                       → check every cases/*/**_mesh.json
Step 2  solve          for each journal in cases/RUNBOOK.md order:
                         fluent_run_journal(journal_text=...)
                         ansys_job_status(job_id)          # poll, do not guess
                         fluent_check_convergence(job_id)  # structured residuals
Step 3  accept         apply the criteria in FLUENT_PLAN.md §6
```

Useful notes for whoever drives this:

- **One solve at a time.** A single solver licence means concurrent jobs collide.
  A clean-looking early death (returncode 1, healthy residuals, right at the
  start) is usually a licence collision from an orphaned process, not a physics
  problem — check `ansys_list_jobs` before debugging the case.
- **Many cases in one launch.** Concatenating journals into one `.jou` and
  invoking it with a single `/file/read-journal "<abs path>"` avoids a licensed
  startup per case.
- **The transcript on disk lags the solver**, sometimes by minutes. Trust job
  status and the saved `.cas.h5`/`.dat.h5`, not the transcript, for real-time
  truth.
- **Headless Fluent cannot render.** The journals bake contour and vector
  objects into the saved case so it can be opened in the GUI later and
  screenshotted with nothing to rebuild; actual PNGs come from CFD-Post
  (`cfdpost_run_session`) or from matplotlib over the exported CSV.

### Step 0 is not optional

These journals were authored **without a live Fluent session**. Three things in
them are version-sensitive: the grouped `set` menu keyword names, the number of
residual equations (which differs between SA and SST and is consumed
positionally), and the discretisation-scheme integer codes.

A wrong keyword in a grouped `set` menu does not fail loudly. The unanswered
prompt **eats the following journal lines**, and the case then solves with
silently wrong boundary conditions and saves a physically plausible but wrong
result under the right filename. Run the probe, read the transcript, reconcile.

## Solver settings and why

| Choice | Reason |
|---|---|
| Pressure-based, steady, 2D planar | M ≤ 0.28 across the AirfRANS Re range at these conditions — incompressible. A density-based solver would add stiffness and cost for nothing. |
| Spalart–Allmaras **and** k-ω SST | SA matches what generated AirfRANS, so it isolates the *setup* offset. SST is the independent model, so the SA↔SST spread is an honest estimate of turbulence-model sensitivity. Reporting one without the other would understate the uncertainty. |
| Velocity-inlet on the far-field C, pressure-outlet downstream | Standard incompressible external-aero pairing. `pressure-far-field` is compressible-only. Reverse flow is prevented at the outlet. |
| Angle of attack baked into the **mesh** | The airfoil is rotated by −α about the quarter chord, so the freestream is always +x. The wake cut then stays aligned with the wake (better resolution than a diagonally crossing wake), and drag = Fx, lift = Fy with no direction bookkeeping. |
| Reference area = chord × 1 m | Fluent 2D uses unit depth, so the coefficients come out in the standard 2D convention and are directly comparable with AirfRANS. |
| SIMPLE, conservative under-relaxation | Robust for an unattended batch. Coupled is faster on a clean structured mesh but has been observed to diverge on the turbulence equations near skewed cells, and an unattended sweep cannot babysit that. |
| Two stages: first order → second order | Stage 1 is startup robustness only and is never reported. Stage 2 is second-order and **mandatory**: a grid study on a first-order solution measures numerical diffusion, not discretisation error, and GCI's Richardson extrapolation assumes a formal order that first-order upwind does not deliver. |
| Convergence on **CD**, not residuals alone | External-aero residuals plateau while coefficients still drift in the third digit. CD is what the paper reports, so CD is what has to stop moving. Both criteria must be met. |

## Outputs per case

Written into `cases/<case_id>/`:

| File | Contents |
|---|---|
| `*_coeffs.out` | CL, CD, CM every iteration — the convergence history for the appendix |
| `*_force_x.txt`, `*_force_y.txt` | pressure/viscous force split |
| `*_yplus_max.txt`, `*_yplus_avg.txt` | achieved y+ (the mesh-adequacy evidence, Table 5) |
| `*_surface.csv` | surface x, y, p, cp, τ_wx, τ_wy, cf, y+ — feeds the offset study and the surrogate comparison |
| `*.cas.h5`, `*.dat.h5` | case + data, with display objects baked in |
| `*.trn` | full transcript |

**When reading `*_surface.csv`: map columns by the header line, never by
position.** `/file/export/ascii` silently reorders the requested columns.

## Reproducing a mesh by hand

```
.venv/Scripts/python.exe fluent/mesh_gen.py --naca 0012 --re 3e6 --aoa 5 \
    --level 2 --self-check --out /tmp/m.msh
```

`--self-check` re-parses the written file and asserts index ranges, owner/
neighbour consistency and section counts. Since Fluent is not available at
authoring time, that check is the only thing standing between a malformed mesh
and a wasted solver run — use it.

## Licence

The meshes, journals, convergence histories and extracted surface data produced
here are **new work under this repository's MIT licence**, deliberately
unencumbered by AirfRANS's ODbL-1.0 or DrivAerNet++'s CC BY-NC-4.0. They are
intended for release on Zenodo with a DOI as a standalone verification set.
