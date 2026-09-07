# results/ — committed run artifacts

One directory per run, named `<model>_<split>_s<seed>[_<tag>]`, each holding
`config.yaml`, `metrics.json`, `per_sim.csv` and `history.csv` (checkpoints are
gitignored). Every table and figure in the paper regenerates from these files
via `scripts/make_tables.py`, `scripts/make_figures.py` and
`scripts/render_examples.py`; the schema is frozen in `docs/CONTEXT.md` §9.

## Run families

| Pattern | What it is |
|---|---|
| `{gnn,sdf_fno,transolver}_<split>_s0` | Core grid: 3 models × 6 splits (`full`, `scarce`, `reynolds`, `aoa`, `shape5`, `combined`), 400 epochs |
| `..._s1 ... _s4` | Deep-ensemble members on {full, reynolds, aoa, shape5}: seeds 0–4 (K=5) for SDF-FNO/Transolver, 0–2 (K=3) for the GNN |
| `..._full_n{25,50,100,200,400}_s{0,1,2}` | Data-efficiency sweep (training-set size × 3 seeds) |
| `sdf_fno_*_cond_{mask,sdfnrm}` | M2 geometry-conditioning ablation (tagged; excluded from core tables) |
| `{gnn}_*_lamF` | M1 force-consistency-loss ablation (tagged) |
| `{constant,ridge}_<split>_s0` | Non-learned baselines |
| `gnn_full_s0_smoke_dryrun` | 2-epoch pipeline smoke test (tagged; kept as evidence, excluded everywhere) |

Tagged runs and non-canonical splits are filtered out of every core table and
figure (`make_tables.py` / `make_figures.py` `_neural()`); only untagged runs on
the six canonical splits enter Tables 1–3.

## Non-run directories

| Dir | Contents |
|---|---|
| `uq/` | Split-conformal calibration reports per (model, split, K, score, matched/transfer), written by `scripts/run_uq.py`; source for Table 3 and Figs 6–8 |
| `active/` | Acquisition pool ranking + the three k=8 arm selections, written by `scripts/run_active.py`; source for Fig 11a |
| `fluent/` | Collected Fluent campaign outputs: `fluent_summary.csv` (per-case convergence), `offset.csv`/`offset_summary.json` (solver offset), `gci.json` (grid study), `surrogate_vs_fluent*.{csv,json}` (the verified comparison); source for Table 5 and Fig 11b |
