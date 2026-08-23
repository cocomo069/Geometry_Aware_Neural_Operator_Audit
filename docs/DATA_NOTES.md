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
