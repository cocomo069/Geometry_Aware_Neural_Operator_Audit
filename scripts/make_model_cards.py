"""Generate model cards (spec C5) from committed metrics.

    .venv/Scripts/python.exe -m scripts.make_model_cards [--results results] [--outdir model_cards]

One card per architecture family (model_cards/{gnn,sdf_fno,transolver}.md).
Every number comes from a committed ``results/<run_id>/metrics.json`` or
``results/uq/*.json`` (seed means over untagged core runs, matching the
paper's Tables 1--3); the prose sections are fixed text maintained here.
Numbers are regenerated on every run -- do not hand-edit the tables.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.make_tables import _largest_k, _lv, _pick_record
from src.viz import data as vdata
from src.viz import style

SPLITS = ("full", "scarce", "reynolds", "aoa", "shape5", "combined")

REFERENCE = {
    "gnn": "kNN message passing in the MeshGraphNets style (Pfaff et al., 2021)",
    "sdf_fno": "SDF-conditioned FNO in the GINO style (Li et al., 2023; Li et al., 2021)",
    "transolver": "Transolver-style physics attention (Wu et al., 2024)",
}

ENCODER = {
    "gnn": "k-nearest-neighbour (k=16) message passing on the surface point cloud; "
           "node features: position, outward normal, signed Menger curvature, "
           "broadcast flow condition",
    "sdf_fno": "kernel encoder scatters surface features onto a 64x64 latent grid "
               "conditioned on the signed distance function; FNO2d stack (16 modes, "
               "width 32); kernel decoder queries back to surface points",
    "transolver": "physics attention: soft-assign N points to M=32 slices across "
                  "4 attention layers at width 128, attend among slice tokens, "
                  "scatter back",
}

# The point of a model card is the honesty section. Fixed prose, reviewed with
# the paper; keep in sync with docs/RESULTS.md if the findings change.
DO_NOT_USE = {
    "gnn": (
        "- Any application that consumes the *integrated* drag of the predicted "
        "field: the GNN's field-implied drag ranks designs far worse than a "
        "29-parameter ridge baseline (integrated Spearman 0.36 in distribution), "
        "and its head/field drags disagree by ~24% of the true value already "
        "in distribution (RESULTS.md 2.2-2.3).\n"
        "- Conformal-interval-backed decisions in distribution: the GNN "
        "*under-covers* at nominal 90% (0.82, Wilson CI excludes 0.90).\n"
        "- Reading its high shifted normalized-score coverage as calibration: "
        "the Reynolds-shift 0.97 rides on a near-vacuous band ~23x the size of "
        "the quantity; absolute-score coverage stays ~0.70 (RESULTS.md 3.1).\n"
        "- Anything post-stall or outside the AirfRANS envelope: verified "
        "errors ~600-900x in-distribution against an independent solver."
    ),
    "sdf_fno": (
        "- Shape-family extrapolation with conformal backing: SDF-FNO's worst "
        "calibration split is shape5 (matched coverage 0.76 at nominal 90).\n"
        "- Data-scarce settings below ~100 training simulations, where it is "
        "the most data-hungry of the three architectures (RESULTS.md 4).\n"
        "- Any use of the integrated drag under the combined shift (integrated "
        "Spearman ~0 there).\n"
        "- Anything post-stall or outside the AirfRANS envelope: verified "
        "errors ~600-900x in-distribution against an independent solver."
    ),
    "transolver": (
        "- Treating its accuracy lead as trust: on the two flow-regime shifts "
        "its 90% conformal intervals covered only 0.42 of independent Fluent "
        "cases at moderate angles (Table 5 of the paper).\n"
        "- Any use of the integrated drag as a ranking signal: 0.89 in "
        "distribution (no better than ridge), collapsing to 0.28 under the "
        "combined shift.\n"
        "- Anything post-stall or outside the AirfRANS envelope: verified "
        "errors ~600-900x in-distribution against an independent solver."
    ),
}


def _fmt(x, prec=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{float(x):.{prec}g}"


def _seed_stats(df: pd.DataFrame, model: str, split: str, col: str):
    sub = df[(df["model"] == model) & (df["split"] == split)]
    vals = pd.to_numeric(sub[col], errors="coerce").dropna()
    if vals.empty:
        return None, None, 0
    sd = vals.std(ddof=1) if len(vals) > 1 else None
    return float(vals.mean()), (float(sd) if sd is not None and np.isfinite(sd) else None), len(vals)


def make_card(model: str, df: pd.DataFrame, uq_reports, outdir: Path) -> Path:
    core = df[df["tag"].isna() & df["split"].isin(SPLITS)] if "tag" in df.columns else df
    full = core[(core["model"] == model) & (core["split"] == "full")]
    params = int(full["params"].dropna().iloc[0]) if not full.empty else 0
    gpu_h_mean = pd.to_numeric(full.get("gpu_hours"), errors="coerce").mean()
    infer_ms = pd.to_numeric(full.get("cost_infer_ms_per_sim"), errors="coerce").mean()

    lines = [
        f"# Model card — {style.model_label(model)}",
        "",
        f"*Generated by `scripts/make_model_cards.py` on {date.today().isoformat()} "
        "from committed `results/` metrics (seed means over untagged core runs). "
        "Regenerate rather than hand-editing the tables.*",
        "",
        "## 1. Model details",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Architecture | {style.model_label(model)} |",
        f"| Design follows | {REFERENCE[model]} |",
        "| Reimplementation? | **Yes** — reimplemented in plain PyTorch, not run from the authors' code; absolute numbers are not comparable with published ones |",
        f"| Encoder | {ENCODER[model]} |",
        f"| Parameters | {params:,} |",
        "| Task | surface fields (p, tau_w) + coefficient head (C_L, C_D); C^int by quadrature of the predicted fields |",
        "| Training | Adam 1e-3 cosine, batch 16, 400 epochs, fp32, lambda_H=0.1 head term; per-split train-only normalisation |",
        f"| Training cost | {_fmt(gpu_h_mean, 2)} GPU-h per run (Tesla T4, `full` split) |",
        f"| Inference | {_fmt(infer_ms, 3)} ms per simulation (CPU-measured harness value) |",
        "",
        "## 2. Accuracy, consistency and ranking (seed mean ± std)",
        "",
        "| Split | n seeds | p rel-L2 | C_D MAE (head) | C_D Spearman (head) | FSC_rel C_D | sym residual |",
        "|---|---|---|---|---|---|---|",
    ]
    for split in SPLITS:
        row = [style.split_label(split)]
        m0, s0, n = _seed_stats(core, model, split, "field_p_rel_l2")
        row.append(str(n))
        for col, prec in (("field_p_rel_l2", 3), ("coef_cd_head_mae", 3),
                          ("coef_cd_head_spearman", 3), ("consistency_fsc_rel_cd", 3),
                          ("consistency_sym_residual", 3)):
            m, sd, _ = _seed_stats(core, model, split, col)
            cell = _fmt(m, prec)
            if sd is not None and sd > 0:
                cell += f" ± {_fmt(sd, 2)}"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## 3. Calibration (split conformal, C_D^int coefficient band, matched)",
        "",
        "| Split | K | cov@.9 (normalized) | mean width@.9 | ECE |",
        "|---|---|---|---|---|",
    ]
    matched = _largest_k(uq_reports, "matched")
    for split in ("full", "shape5", "reynolds", "aoa"):
        rep = matched.get((model, split))
        rec = _pick_record(rep, "cd_int", "coefficient") if rep else None
        if rec is None:
            continue
        lines.append("| " + " | ".join([
            style.split_label(split), _fmt(rep.get("k"), 1),
            _fmt(_lv(rec, 0.9, "coverage"), 3),
            _fmt(_lv(rec, 0.9, "mean_width"), 3),
            _fmt(rec.get("ece"), 3),
        ]) + " |")

    lines += [
        "",
        "Coverage must be read next to width; a wide band can cover without "
        "informing (see the paper's calibration section and docs/RESULTS.md 3.1).",
        "",
        "## 4. Intended use",
        "",
        "Surrogate predictions of surface pressure/shear and drag for NACA "
        "4/5-digit sections *inside* the AirfRANS envelope (Re 2-6e6, alpha "
        "-5..+15 deg, attached flow), with FSC monitored as a label-free "
        "out-of-distribution flag on every prediction.",
        "",
        "## 5. Do not use for",
        "",
        DO_NOT_USE[model],
        "",
        "## 6. Provenance",
        "",
        "- Training data: AirfRANS (ODbL-1.0); split manifests in `data/splits/`.",
        "- Every number above: `results/<run_id>/metrics.json` and "
        "`results/uq/*.json` at the commit this file was generated at.",
        "- Full protocol, caveats and the external Fluent check: the paper "
        "(`paper/`) and `docs/RESULTS.md`.",
        "",
    ]
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{model}.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="model_cards")
    args = ap.parse_args(argv)

    df = vdata.load_runs(args.results)
    uq = vdata.load_uq_reports(Path(args.results) / "uq")
    written = [make_card(m, df, uq, Path(args.outdir))
               for m in ("gnn", "sdf_fno", "transolver")]
    print(f"{len(written)} model card(s) written:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
