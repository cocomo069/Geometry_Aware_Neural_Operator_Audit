"""Re-run evaluation on an existing checkpoint (or fit a baseline).

    # re-evaluate a finished run (config is read back from results/<run_id>/config.yaml)
    .venv/Scripts/python.exe -m scripts.evaluate --run-id gnn_full_s0

    # evaluate a specific checkpoint with an explicit config
    .venv/Scripts/python.exe -m scripts.evaluate --config configs/gnn.yaml \
        --checkpoint checkpoints/gnn_full_s0/best.pt split=aoa

    # the two non-learned baselines (PLAN.md 3.6 / gate G2), same metrics schema
    .venv/Scripts/python.exe -m scripts.evaluate --config configs/gnn.yaml --baseline ridge

Writes ``results/<run_id>/metrics.json`` (+ ``per_sim.csv``) through the same
harness that ``scripts/train.py`` uses -- CONTEXT.md 9 allows no other producer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config import get_in, load_config, save_config  # noqa: E402
from src.utils.io import build_run_id, results_dir  # noqa: E402
from src.utils.seed import seed_all  # noqa: E402

from scripts.train import (  # noqa: E402
    build_datasets,
    build_model,
    count_params,
    load_checkpoint,
    parse_cli,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts/evaluate.py",
        description="Evaluate a checkpoint or a baseline into results/<run_id>/metrics.json",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="YAML config, e.g. configs/gnn.yaml")
    src.add_argument("--run-id", help="existing run; reads results/<run_id>/config.yaml")
    p.add_argument("overrides", nargs="*", default=[], help="dotted key=value overrides")
    p.add_argument("--checkpoint", default=None,
                   help="checkpoint path (default: checkpoints/<run_id>/best.pt)")
    p.add_argument("--baseline", choices=["constant", "ridge"], default=None,
                   help="fit a non-learned baseline on train instead of loading a model")
    p.add_argument("--subset", choices=["test", "cal", "train"], default="test")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--no-symmetry", action="store_true")
    p.add_argument("--out", default=None, help="output directory (default results/<run_id>)")
    p.add_argument("--results-root", default="results")
    p.add_argument("--checkpoints-root", default="checkpoints")
    p.add_argument("--tag", default=None, help="override the run tag (e.g. 'eval_aoa')")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_cli(build_parser(), argv)

    if args.run_id:
        cfg_path = Path(args.results_root) / args.run_id / "config.yaml"
        if not cfg_path.is_file():
            raise SystemExit(f"no config at {cfg_path}; pass --config instead")
        cfg = load_config(cfg_path, args.overrides)
    else:
        cfg = load_config(args.config, args.overrides)

    if args.device:
        cfg["device"] = args.device
    if args.tag is not None:
        cfg["tag"] = args.tag
    seed = int(get_in(cfg, "seed", 0))
    seed_all(seed)
    device = resolve_device(cfg.get("device", "auto"))
    split = str(get_in(cfg, "data.split", "full"))

    data = build_datasets(cfg)
    collate = data.get("collate")
    eval_ds = data.get(args.subset)
    if eval_ds is None or len(eval_ds) == 0:
        raise SystemExit(f"split {split!r} has no usable {args.subset!r} subset")

    batch_size = args.batch_size or int(get_in(cfg, "eval.batch_size", 8))

    if args.baseline:
        from src.eval.baselines import run_baseline

        run_id = build_run_id(args.baseline, split, seed, cfg.get("tag"))
        out = Path(args.out) if args.out else results_dir(run_id, args.results_root)
        save_config(cfg, out / "config.yaml")
        metrics = run_baseline(
            args.baseline, data["train"], eval_ds, split,
            seed=seed, tag=cfg.get("tag"), device=str(device),
            batch_size=batch_size, collate_fn=collate, save_dir=out,
            max_batches=args.max_batches,
            compute_symmetry=not args.no_symmetry,
        )
    else:
        from src.eval.harness import evaluate

        model_name = str(get_in(cfg, "model.name"))
        run_id = args.run_id or build_run_id(model_name, split, seed, cfg.get("tag"))
        ckpt = Path(args.checkpoint) if args.checkpoint else (
            Path(args.checkpoints_root) / run_id / "best.pt"
        )
        if not ckpt.is_file():
            alt = ckpt.with_name("last.pt")
            if not alt.is_file():
                raise SystemExit(f"no checkpoint at {ckpt} (nor {alt})")
            print(f"[eval] {ckpt.name} missing, falling back to {alt.name}")
            ckpt = alt

        model = build_model(cfg)
        state = load_checkpoint(ckpt, model, map_location=device, restore_rng=False)
        out = Path(args.out) if args.out else results_dir(run_id, args.results_root)
        save_config(cfg, out / "config.yaml")
        metrics = evaluate(
            model, eval_ds, split,
            run_meta={
                "run_id": run_id,
                "model": model_name,
                "seed": seed,
                "tag": cfg.get("tag"),
                "params": count_params(model),
                "epochs": int(state.get("epoch", 0) or 0),
            },
            device=device,
            batch_size=batch_size,
            collate_fn=collate,
            max_batches=args.max_batches,
            compute_symmetry=not args.no_symmetry,
            symmetry_max_sims=get_in(cfg, "eval.symmetry_max_sims", 64),
            save_dir=out,
        )

    print(f"[eval] {run_id} -> {out / 'metrics.json'}")
    print(f"       p_rel_l2={metrics['field']['p_rel_l2']} "
          f"cd_int_mae={metrics['coef']['cd_int_mae']} "
          f"cd_spearman={metrics['coef']['cd_spearman']} "
          f"fsc_cd={metrics['consistency']['fsc_cd']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
