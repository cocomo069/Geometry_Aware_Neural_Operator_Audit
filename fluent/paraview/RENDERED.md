# ParaView field figures — RENDERED (D-025)

Rendered 2026-09-03 (Opus execution agent) from the EnSight Gold `.encas` exports
via `mcp__paraview__*`. Output: `paper/figures/fluent/`, all PNG at 300 dpi,
3.2 in x 2.4 in (960 x 720 px), white background, parallel (orthographic)
projection, hand-computed 2D camera (focal (0.5, 0), parallel_scale 0.85, +Z look).

The `.encas` exports already carry node-based (POINT) fields with friendly names
(`pressure_coefficient`, `velocity_magnitude`, `velocity`, ...), so rendering is
already Gouraud-smooth; no CellDataToPointData pass was needed (there is no cell
data to convert). Native Fluent display objects remain baked in the saved
`.cas.h5` per PARAVIEW_TODO.md; Fluent was not launched (solves running).

## B1 — grid-independence, pressure-coefficient contour (NACA0012, Re 3e6, alpha 5, SA)
Fixed C_p range [-2, 1], "Cool to Warm" colormap, identical camera on all three.

| file | case | mesh cells | caption |
|---|---|---|---|
| `figB1a_gridstudy_L1_cp.png` | gridstudy_naca0012_re3e6_a5_L1 | 19,368 | Coarse grid (L1): C_p field, NACA0012 at alpha 5 deg; LE stagnation and suction peak resolved on the coarse mesh. |
| `figB1b_gridstudy_L2_cp.png` | gridstudy_naca0012_re3e6_a5_L2 | 43,872 | Medium grid (L2): same C_p field; sharper LE suction-peak gradient than L1. |
| `figB1c_gridstudy_L3_cp.png` | gridstudy_naca0012_re3e6_a5_L3 | 98,064 | Fine grid (L3): converged C_p field; suction-peak gradient essentially grid-independent from L2. |

## B3 — converged random cases, velocity-magnitude + streamlines (SA)
"Viridis" colormap (perceptually uniform), per-case |U| range, black streamlines
(StreamTracer, seed line x=-1 from y=-1 to y=1, resolution 60), identical camera.

| file | case | |U| range (m/s) | caption |
|---|---|---|---|
| `figB3a_al_rand_naca0010_re7e6_a6_vel.png` | al_rand_naca0010_re7e6_a6 | 0–215 | NACA0010, Re 7e6, alpha 6 deg: velocity magnitude with streamlines; attached flow, upper-LE acceleration. |
| `figB3b_al_rand_naca2415_re6e6_a0_vel.png` | al_rand_naca2415_re6e6_a0 | 0–120 | NACA2415, Re 6e6, alpha 0 deg: cambered-section acceleration over the upper surface at zero incidence. |
| `figB3c_al_rand_naca33012_re6e6_am6_vel.png` | al_rand_naca33012_re6e6_am6 | 0–210 | NACA33012, Re 6e6, alpha -6 deg: negative incidence, suction/acceleration on the lower surface. |
| `figB3d_al_rand_naca4415_re2e6_a0_vel.png` | al_rand_naca4415_re2e6_a0 | 0–42 | NACA4415, Re 2e6, alpha 0 deg: strongly cambered section, upper-surface acceleration, attached streamlines. |

## B4 — turbulence-model sensitivity, C_p contour (offset_2, NACA0109, alpha 8.6)
Fixed C_p range [-4, 1] on both panels for direct comparison; "Cool to Warm",
identical camera. Suction peak saturates (true min ~ -6.7 both models); the peak
differs negligibly between models (SA -6.72 vs SST -6.66), i.e. the SA/SST
sensitivity is small for this near-attached case.

| file | case | model | caption |
|---|---|---|---|
| `figB4a_offset2_naca0109_a8p6_sa_cp.png` | offset_2_naca0109_a8p6 | SA | C_p field, Spalart–Allmaras. |
| `figB4b_offset2_naca0109_a8p6_sst_cp.png` | offset_2_naca0109_a8p6 | SST | C_p field, k-omega SST; field essentially indistinguishable from SA. |

## Cases skipped
None of the required cases were skipped. All target `.encas` were readable:
grid L1/L2/L3 (SA), the 4 converged random cases (SA), offset_2 (SA + SST).
Note: offset_2 SST `.encas` is present (needed for B4). offset_6 has SA only;
not required by the B1/B3/B4 spec. B5 (optional, r1-rescued a12/a18) was not
produced: no `*_sa_r1.encas` present at render time.

## Notes / gotchas
- Legends persist across sources in this connector: hid every representation +
  the prior source's scalar bar before each new case, and explicitly hid the SA
  legend before the SST panel, so no stacked/overlapping legends. Verified by
  reading back the exported PNGs.
- `SeedType: "High Resolution Line Source"` is rejected as obsolete; use
  `"Line"`.
- Full farfield domain bounds ([-29.75, 30] x [-30, 30]); airfoil is unit chord
  near origin, so the camera was hand-framed (auto-reset does not fit reliably).
