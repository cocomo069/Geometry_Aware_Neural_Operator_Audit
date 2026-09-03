# ParaView field figures -- TODO for the orchestrator (D-025)

Per the user's directive, field figures come from the EnSight Gold exports via the
`mcp__paraview__*` tools. This execution agent did NOT run ParaView (per its brief).
Every converged/accepted case already has its `<case>_<model>.encas` (+ `.geo`,
`.scl*`, `.vel`) on disk and the native Fluent display objects (`ct_pressure`,
`ct_velocity`, `ct_cp`, `vec_velocity`) baked into the saved `.cas.h5`.

Run these AFTER the offset and r1 batches finish (re-run `scripts/collect_fluent`
first, then confirm each `.encas` exists). Diverged cases have no `.dat.h5`/`.encas`
and get no figure.

## Figures to render (PLAN_FLUENT_POST section 5)

Absolute case-dir root: `D:\Personal Projects\geom_aware_neural_operator\fluent\cases\`

| figure | case(s) | model | content |
|---|---|---|---|
| B1 (a,b,c) | `gridstudy_naca0012_re3e6_a5_L1` / `_L2` / `_L3` | sa | cp contour + surface-cp overlay (grid-independence) |
| B3 (a,b,c,d) | `al_rand_naca0010_re7e6_a6`, `al_rand_naca2415_re6e6_a0`, `al_rand_naca33012_re6e6_am6`, `al_rand_naca4415_re2e6_a0` | sa | velocity-magnitude + streamlines, cp legend (the (b) comparison cases) |
| B4 | `offset_2_naca0109_a8p6` | sa vs sst side-by-side | turbulence-model sensitivity (alpha 8.6 deg; may need cp range [-4,1]) |
| B5 (optional) | any a12/a18 case rescued to quasi_steady by r1 | sa_r1 | separated-flow field, hollow-marker cases in Fig 11b |

## Per-case MCP pipeline (in order)
1. `paraview_open_data({"path": "<abs>/<case>/<case>_sa.encas", "name": "<case>"})`; read back array names (expect `pressure`, `pressure-coefficient`, `velocity-magnitude`, `x-wall-shear`, `y-wall-shear`, `velocity`).
2. `paraview_set_coloring({"name":"<case>","array":"pressure-coefficient","association":"POINTS","colormap":"Cool to Warm","rescale":[-2.0,1.0]})` (same fixed range on every panel; state exceptions in the caption).
3. velocity panels: `paraview_apply_filter` StreamTracer, `Vectors=["POINTS","velocity"]`, SeedType Line from [-1,-1,0] to [-1,1,0], Resolution 60 (z=0 plane).
4. `paraview_set_camera` +Z full view, then a second zoomed call at the LE (focal 0.05,0, height +/-0.15).
5. `paraview_add_color_legend` (title `C_p`); `paraview_export_figure` to `<abs>/paper/figures/figB{1,3,4}_<case>_cp.png` (width 3.2 in, height 2.4 in, dpi 300) + a `.pdf` for the vector build.
6. `paraview_save_state` to `fluent/paraview/<case>.pvsm`.
7. B1 surface-cp overlay: matplotlib from the three `_surface.csv` (map columns BY HEADER: `x-coordinate`, `pressure-coefficient`), un-rotate the Fluent surface by -alpha about (0.25,0) before plotting x/c.

Before the batch of exports, verify one export renders (open the PNG with Read) and
that the airfoil outline is visible (add a `Contour` of `velocity-magnitude` at 0 or
`ExtractSurface` of the wall zone if the EnSight part list includes `airfoil`).

## `.encas` inventory (regenerate after batches complete: `ls fluent/cases/*/*.encas`)
Ready at last check (SA, converged): gridstudy L1/L2/L3, the 4 converged random
cases, offset_1 (offset_2..6 SA+SST appear as the offset batch finishes; r1-rescued
cases appear as `<case>_sa_r1.encas` if any converge).
