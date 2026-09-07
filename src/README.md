# src/ — the library

Pure-Python package (no compiled extensions, by constraint: `docs/CONTEXT.md` §1).
Absolute imports (`from src.physics.force_integration import ...`), run from repo
root. Interfaces marked FROZEN in `docs/CONTEXT.md` §§4–10.

| Package | Contents |
|---|---|
| `data/` | AirfRANS loader (`airfrans_loader.py`), split construction + sim-name parsing (`splits.py`) |
| `geometry/` | SDF, normals, quadrature weights, symmetry/reflection ops |
| `physics/` | Force integration (the G1-gated quadrature) and physics residuals |
| `models/` | The three architectures (`gnn.py`, `sdf_fno.py`, `geo_transformer.py`), shared heads, `build_model` registry |
| `eval/` | Metrics, evaluation harness, results schema, non-learned baselines |
| `uq/` | Deep-ensemble aggregation, split conformal prediction, coverage/ECE |
| `active/` | NACA pool generator, acquisition scoring, farthest-point diversity |
| `utils/` | Config loading/overrides, seeding, checkpoint IO |
| `viz/` | Figure/table data loading (`data.py`), the single matplotlib style (`style.py`), shift magnitudes (`shift.py`) |
