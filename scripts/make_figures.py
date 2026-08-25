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
from matplotlib.lines import Line2D  # noqa: E402

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
#: The split axis of Figs 7/8, ordered by increasing distribution shift.
UQ_SPLIT_ORDER = ("full", "shape5", "reynolds", "aoa", "combined")


def _uq_mode(ensemble_id: str) -> str:
    """``transfer`` for a ``*_calfull`` report, else ``matched``."""
    return "transfer" if str(ensemble_id).endswith("_calfull") else "matched"


def _select_uq_reports(reports, *, mode: str | None = None):
    """Keep, per ``(model, split, mode)``, only the largest-K report.

    The ensemble-size ablation writes one file per K; the headline figures use
    the fullest ensemble available (K=1 in the demo, K=5 once ensembles land).
    """
    best: dict[tuple[str, str, str], dict] = {}
    for rep in reports:
        m = _uq_mode(rep.get("ensemble_id", ""))
        if mode is not None and m != mode:
            continue
        key = (str(rep.get("model", "")), str(rep.get("split", "")), m)
        cur = best.get(key)
        if cur is None or float(rep.get("k", 0) or 0) > float(cur.get("k", 0) or 0):
            best[key] = rep
    return list(best.values())


def _pick_record(rep, target: str, granularity: str):
    """One record for ``(target, granularity)``, preferring the normalized score."""
    recs = [
        r for r in rep.get("records", [])
        if isinstance(r, dict)
        and str(r.get("target")) == target
        and str(r.get("granularity")) == granularity
    ]
    if not recs:
        return None
    recs.sort(key=lambda r: 0 if r.get("score") == "normalized" else 1)
    return recs[0]


def _level_row(rec, nominal: float):
    for lv in rec.get("levels", []):
        if abs(float(lv.get("nominal", -1)) - nominal) < 1e-9:
            return lv
    return None


def fig6_reliability(outdir: Path, results: str) -> Path | None:
    """Empirical vs nominal coverage for the matched ``cd_int`` band."""
    uq_dir = Path(results) / "uq"
    reports = _select_uq_reports(vdata.load_uq_reports(uq_dir), mode="matched")
    curves = []
    for rep in reports:
        rec = _pick_record(rep, "cd_int", "coefficient")
        if rec is None:
            continue
        rel = rec.get("reliability") or {}
        nom, emp = rel.get("nominal"), rel.get("empirical")
        if nom and emp and len(nom) == len(emp):
            curves.append((str(rep.get("model", "")), str(rep.get("split", "")), nom, emp))
    if not curves:
        print("TODO fig6: no results/uq reliability data")
        return None
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7, label="ideal")
    for model, split, nom, emp in curves:
        ax.plot(nom, emp, marker="o", ms=3.5, lw=1.3,
                color=style.model_color(model), alpha=0.9,
                label=f"{style.model_label(model)} / {style.split_label(split)}")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title(r"Reliability ($C_D^{\mathrm{int}}$, matched)")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    return _save(fig, outdir, "fig06_reliability")


def fig7_coverage_vs_shift(outdir: Path, results: str) -> Path | None:
    """Empirical coverage at nominal 0.9 vs shift; matched (dashed) vs transfer (solid)."""
    uq_dir = Path(results) / "uq"
    reports = vdata.load_uq_reports(uq_dir)
    if not reports:
        print("TODO fig7: no results/uq data")
        return None
    nominal = 0.9
    per_mode = {m: _select_uq_reports(reports, mode=m) for m in ("matched", "transfer")}
    splits_present = sorted(
        {str(r.get("split", "")) for reps in per_mode.values() for r in reps},
        key=lambda s: UQ_SPLIT_ORDER.index(s) if s in UQ_SPLIT_ORDER else 99,
    )
    if not splits_present:
        print("TODO fig7: no splits in results/uq")
        return None
    models = style.sort_models(
        {str(r.get("model", "")) for reps in per_mode.values() for r in reps}
    )
    xpos = {s: i for i, s in enumerate(splits_present)}

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.axhline(nominal, color="k", lw=1, ls=":", alpha=0.7, label=f"nominal {nominal:g}")
    plotted = False
    for mode, ls in (("matched", "--"), ("transfer", "-")):
        for model in models:
            xs, ys = [], []
            for rep in per_mode[mode]:
                if str(rep.get("model")) != model:
                    continue
                rec = _pick_record(rep, "cd_int", "coefficient")
                lv = _level_row(rec, nominal) if rec else None
                if lv is None:
                    continue
                xs.append(xpos[str(rep.get("split", ""))])
                ys.append(float(lv.get("coverage", float("nan"))))
            if not xs:
                continue
            order = np.argsort(xs)
            xs = np.asarray(xs)[order]
            ys = np.asarray(ys)[order]
            ax.plot(xs, ys, ls=ls, marker=style.model_marker(model),
                    color=style.model_color(model), lw=1.6,
                    label=(style.model_label(model) if mode == "transfer" else None))
            plotted = True
    if not plotted:
        print("TODO fig7: no cd_int coverage records")
        plt.close(fig)
        return None
    ax.set_xticks(range(len(splits_present)))
    ax.set_xticklabels([style.split_label(s) for s in splits_present], rotation=20, ha="right")
    ax.set_ylabel(f"empirical coverage @ {nominal:g}")
    ax.set_title(r"Coverage vs shift ($C_D^{\mathrm{int}}$)")
    handles = (style.model_legend(models, with_marker=True)
               + [Line2D([], [], color="k", ls="--", label="matched cal"),
                  Line2D([], [], color="k", ls="-", label="transfer (cal=full)"),
                  Line2D([], [], color="k", ls=":", label=f"nominal {nominal:g}")])
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)
    return _save(fig, outdir, "fig07_coverage_vs_shift")


def fig8_interval_width(outdir: Path, results: str) -> Path | None:
    """Interval width ID vs OOD: matched bands, coef C_D^int and field p, per model."""
    uq_dir = Path(results) / "uq"
    reports = _select_uq_reports(vdata.load_uq_reports(uq_dir), mode="matched")
    if not reports:
        print("TODO fig8: no results/uq data")
        return None
    nominal = 0.9
    splits_present = sorted(
        {str(r.get("split", "")) for r in reports},
        key=lambda s: UQ_SPLIT_ORDER.index(s) if s in UQ_SPLIT_ORDER else 99,
    )
    models = style.sort_models({str(r.get("model", "")) for r in reports})
    lut = {(str(r.get("model")), str(r.get("split"))): r for r in reports}

    panels = (
        (r"$C_D^{\mathrm{int}}$ interval width", "cd_int", "coefficient"),
        (r"$p$ field-band width", "p", "field_quantile"),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.6))
    if len(panels) == 1:
        axes = [axes]
    any_bar = False
    nsp = max(1, len(splits_present))
    width = 0.8 / max(1, len(models))
    for ax, (title, target, gran) in zip(axes, panels):
        for mi, model in enumerate(models):
            heights, xs = [], []
            for si, split in enumerate(splits_present):
                rep = lut.get((model, split))
                rec = _pick_record(rep, target, gran) if rep else None
                lv = _level_row(rec, nominal) if rec else None
                if lv is None:
                    continue
                w = float(lv.get("mean_width", float("nan")))
                if not np.isfinite(w):
                    continue
                xs.append(si + (mi - (len(models) - 1) / 2) * width)
                heights.append(w)
            if heights:
                ax.bar(xs, heights, width=width, color=style.model_color(model),
                       label=style.model_label(model), alpha=0.9)
                any_bar = True
        ax.set_xticks(range(nsp))
        ax.set_xticklabels([style.split_label(s) for s in splits_present], rotation=20, ha="right")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, which="both", axis="y", alpha=0.3)
    if not any_bar:
        print("TODO fig8: no width records")
        plt.close(fig)
        return None
    axes[0].set_ylabel(f"mean interval width @ {nominal:g}")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("Interval width, ID vs OOD (matched)", y=1.02)
    return _save(fig, outdir, "fig08_interval_width")


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
    for fn in (fig6_reliability, fig7_coverage_vs_shift, fig8_interval_width):
        p = fn(outdir, args.results)
        if p:
            written.append(p)

    print(f"\n{len(written)} figure(s) written to {outdir}:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
