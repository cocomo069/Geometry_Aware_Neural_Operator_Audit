# configs/ — run configuration

One YAML per architecture (`gnn.yaml`, `sdf_fno.yaml`, `transolver.yaml`);
split, seed, tag and any other field are CLI overrides
(`python -m scripts.train --config configs/gnn.yaml split=full seed=0`).
Defaults are the paper's matched protocol (`docs/CONTEXT.md` §10): Adam 1e-3
cosine, batch 16, 400 epochs, fp32, `lambda_H=0.1`, physics-loss weights 0.

`sweeps/` holds sweep specs for `scripts/sweep.py` and the Kaggle drivers:
`core.yaml`/`kaggle_core.yaml` (3 models × 6 splits), `ensembles.yaml`/
`kaggle_ensembles.yaml` (K=5 seeds), `kaggle_gnn_ens_k3.yaml` (GNN K=3),
`kaggle_dataeff.yaml` (train-size sweep), `kaggle_ablations.yaml`,
`kaggle_smoke.yaml` (pipeline smoke test).
