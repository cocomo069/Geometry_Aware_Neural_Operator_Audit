"""Generate paper figures from committed run metrics (PLAN.md Phase 5).

    .venv/Scripts/python.exe -m scripts.make_figures [--results results] [--outdir paper/figures]

Each figure function is pure: it takes the tidy DataFrame(s) built by
``src.viz.data`` and writes ``paper/figures/fig<NN>_<slug>.{pdf,png}``.  A figure
whose data does not exist yet is skipped with a printed ``TODO`` line rather than
erroring, so this runs at any stage of the project and fills in as runs land.

Spec figure list (05_..._uq.md section 8):
  4  error vs shift magnitude, one panel per OOD axis            [live: needs neural runs]
  5  force self-consistency scatter, C_D int vs head             [live: any runs with both]
  6  reliability diagram (empirical vs nominal coverage)         [gated: results/uq]
  7  coverage vs shift magnitude                                 [gated: results/uq]
  8  interval-width distributions, ID vs OOD                     [gated: results/uq]
  9  data-efficiency curves, log-log                             [gated: data-eff runs]
  12 cost-accuracy Pareto (rel L2 vs GPU-hours)                  [live: needs neural runs]
Figures 1,2,3,10,11 need schematics / field renders / AL results produced elsewhere.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.viz import data as vdata  # noqa: E402
from src.viz import style  # noqa: E402

# OOD axis grouping for Fig 4 (spec section 5.7): each panel holds ID + the splits
# that vary one axis away from it.
OOD_PANELS: dict[str, list[str]] = {
    "scarcity": ["full", "scarce"],
    "Reynolds": ["full", "reynolds"],
    "angle of attack": ["full", "aoa"],
    "shape family": ["full", "shape5", "combined"],
}


def _save(fig: plt.Figure, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = style.figure_path(outdir, name, "pdf")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(style.figure_path(outdir, name, "png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return pdf


def _neural(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["is_baseline"]] if "is_baseline" in df.columns else df


# --------------------------------------------------------------------------- #
def fig4_error_vs_shift(df: pd.DataFrame, outdir: Path) -> Path | None:
    """Field rel-L2 (and CD rel err) against OOD split, one panel per axis."""
    if df.empty or df["model"].nunique() == 0:
        print("TODO fig4: no runs")
        return None
    metric = "field_p_rel_l2"
    panels = list(OOD_PANELS.items())
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.4), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    models = style.sort_models(df["model"].unique())
    for ax, (axis_name, splits) in zip(axes, panels):
        present = [s for s in splits if s in set(df["split"])]
        for m in models:
            sub = df[(df["model"] == m) & (df["split"].isin(present))]
            if sub.empty or sub[metric].isna().all():
                continue
            g = sub.groupby("split")[metric].mean().reindex(present).dropna()
            xs = list(range(len(g)))
            ax.plot(xs, g.values, marker=style.model_marker(m), color=style.model_color(m),
                    label=style.model_label(m), lw=1.6)
            ax.set_xticks(xs)
            ax.set_xticklabels([style.split_label(s) for s in g.index], rotation=20, ha="right")
        ax.set_title(axis_name)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"field $p$ rel. $L_2$")
    handles = style.model_legend(models, with_marker=True)
    fig.legend(handles=handles, loc="upper center", ncol=len(models), frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Error vs distribution shift", y=1.02)
    return _save(fig, outdir, "fig04_error_vs_shift")


def fig5_fsc_scatter(df: pd.DataFrame, outdir: Path) -> Path | None:
    """C_D from surface integration vs from the direct head, coloured by split."""
    need = {"coef_cd_int_mae", "coef_cd_head_mae"}
    # We plot the actual predicted values via per-sim if available; else fall back
    # to the integrated-vs-head MAE as a proxy scatter across runs.
    per = vdata.load_all_per_sim(runs=df) if not df.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    plotted = False
    if not per.empty and {"cd_int", "cd_head"}.issubset(per.columns):
        for split in style.sort_splits(per["split"].dropna().unique()):
            sub = per[per["split"] == split]
            ax.scatter(sub["cd_head"], sub["cd_int"], s=10, alpha=0.5,
                       color=style.split_color(split), label=style.split_label(split))
        lo = float(np.nanmin([per["cd_head"].min(), per["cd_int"].min()]))
        hi = float(np.nanmax([per["cd_head"].max(), per["cd_int"].max()]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7)
        ax.set_xlabel(r"$C_D^{\mathrm{head}}$")
        ax.set_ylabel(r"$C_D^{\mathrm{int}}$")
        plotted = True
    if not plotted:
        print("TODO fig5: no per-sim cd_int/cd_head yet (needs neural runs with both heads)")
        plt.close(fig)
        return None
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Force self-consistency")
    return _save(fig, outdir, "fig05_fsc_scatter")


def fig9_data_efficiency(df: pd.DataFrame, outdir: Path) -> Path | None:
    """Rel-L2 vs training-set size, log-log, one line per model with seed bands."""
    d = _neural(df)
    if d.empty or d["train_size"].nunique() < 3:
        print("TODO fig9: need >=3 distinct train_size values (data-efficiency sweep)")
        return None
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for m in style.sort_models(d["model"].unique()):
        sub = d[d["model"] == m]
        g = sub.groupby("train_size")["field_p_rel_l2"].agg(["mean", "std"]).dropna()
        if g.empty:
            continue
        ax.plot(g.index, g["mean"], marker=style.model_marker(m), color=style.model_color(m),
                label=style.model_label(m))
        ax.fill_between(g.index, g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0),
                        color=style.model_color(m), alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training simulations")
    ax.set_ylabel(r"field $p$ rel. $L_2$")
    ax.set_title("Data efficiency")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    return _save(fig, outdir, "fig09_data_efficiency")


def fig12_cost_accuracy(df: pd.DataFrame, outdir: Path) -> Path | None:
    """Pareto: field rel-L2 against training GPU-hours (ID/full split)."""
    d = _neural(df)
    d = d[d["split"] == "full"] if "split" in d.columns else d
    if d.empty or d["gpu_hours"].isna().all():
        print("TODO fig12: no neural runs with gpu_hours on the full split yet")
        return None
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for m in style.sort_models(d["model"].unique()):
        sub = d[d["model"] == m]
        ax.scatter(sub["gpu_hours"], sub["field_p_rel_l2"], s=40,
                   color=style.model_color(m), marker=style.model_marker(m),
                   label=style.model_label(m))
    ax.set_xlabel("training GPU-hours")
    ax.set_ylabel(r"field $p$ rel. $L_2$")
    ax.set_title("Cost vs accuracy")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    return _save(fig, outdir, "fig12_cost_accuracy")


# ---- UQ figures (gated on results/uq) ------------------------------------- #
def fig6_reliability(outdir: Path, results: str) -> Path | None:
    rel = vdata.reliability_frame(vdata.load_uq(results))
    if rel is None or rel.empty:
        print("TODO fig6: no results/uq reliability data")
        return None
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7, label="ideal")
    for key, sub in rel.groupby([c for c in ("model", "split") if c in rel.columns]):
        label = " ".join(str(k) for k in (key if isinstance(key, tuple) else (key,)))
        ax.plot(sub["nominal"], sub["empirical"], marker="o", label=label)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability")
    ax.legend(frameon=False, fontsize=7)
    return _save(fig, outdir, "fig06_reliability")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="paper/figures")
    args = ap.parse_args(argv)

    style.apply_style()
    outdir = Path(args.outdir)
    df = vdata.load_runs(args.results)

    written: list[Path] = []
    for fn in (fig4_error_vs_shift, fig5_fsc_scatter, fig9_data_efficiency, fig12_cost_accuracy):
        p = fn(df, outdir)
        if p:
            written.append(p)
    p = fig6_reliability(outdir, args.results)
    if p:
        written.append(p)

    print(f"\n{len(written)} figure(s) written to {outdir}:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
