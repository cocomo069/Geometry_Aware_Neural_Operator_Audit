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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

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
    """Core neural runs only: drop the constant/ridge baselines AND the smoke /
    ablation tagged runs (smoke_dryrun, lamF, cond_*), which are not core results
    and polluted Table 1 and the figures before the round-1/2 fixes. Data-efficiency
    runs are KEPT: the real ones are untagged (split=full_n*) and the legacy/test
    convention tags them n<size> (split=full) -- both are legitimate for fig9, so a
    tag matching ^n\\d+$ is retained; every other tag is dropped."""
    d = df
    if "tag" in d.columns:
        tag = d["tag"]
        is_dataeff = tag.astype("string").str.fullmatch(r"n\d+").fillna(False)
        d = d[tag.isna() | is_dataeff]
    if "is_baseline" in d.columns:
        d = d[~d["is_baseline"]]
    return d


# --------------------------------------------------------------------------- #
def fig4_error_vs_shift(df: pd.DataFrame, outdir: Path) -> Path | None:
    """Field rel-L2 (and CD rel err) against OOD split, one panel per axis."""
    if df.empty or df["model"].nunique() == 0:
        print("TODO fig4: no runs")
        return None
    df = _neural(df)  # core runs only: no baselines, no smoke/ablation tags
    metric = "field_p_rel_l2"
    panels = list(OOD_PANELS.items())
    # 2x2 grid at the final printed full width (7.05 in, REVTeX text block).
    # The old 1x4 was drawn 16 in wide and shrank to ~43% on insertion,
    # leaving ~4 pt effective text (research-figures: draw at final width,
    # at most two panels per row).
    fig, axgrid = plt.subplots(2, 2, figsize=(7.05, 5.2), sharey=True)
    axes = axgrid.ravel()
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
    for ax in (axes[0], axes[2]):
        ax.set_ylabel(r"field $p$ rel. $L_2$")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    handles = style.model_legend(models, with_marker=True)
    # Legend alone above the panels; the caption carries the description, so no
    # suptitle (it used to collide with this legend).
    fig.legend(handles=handles, loc="upper center", ncol=len(models), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    return _save(fig, outdir, "fig04_error_vs_shift")


def fig5_fsc_scatter(df: pd.DataFrame, outdir: Path) -> Path | None:
    """C_D from surface integration vs from the direct head, coloured by split."""
    need = {"coef_cd_int_mae", "coef_cd_head_mae"}
    # We plot the actual predicted values via per-sim if available; else fall back
    # to the integrated-vs-head MAE as a proxy scatter across runs.
    core = _neural(df)  # drop baselines + tagged runs
    if "split" in core.columns:
        core = core[core["split"].isin(style.SPLIT_ORDER)]  # canonical splits only (no full_n*)
    per = vdata.load_all_per_sim(runs=core) if not core.empty else pd.DataFrame()
    # Final printed size: one REVTeX column (3.4 in).
    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    plotted = False
    if not per.empty and {"cd_int", "cd_head"}.issubset(per.columns):
        for split in style.sort_splits(per["split"].dropna().unique()):
            sub = per[per["split"] == split]
            ax.scatter(sub["cd_head"], sub["cd_int"], s=6, alpha=0.5,
                       color=style.split_color(split), label=style.split_label(split))
        # Zoom to the central mass of the data (0.5-99.5% quantiles, padded);
        # without this the identity line's aspect stretch turned the cloud into
        # an unreadable vertical blob dominated by a handful of cd_int outliers.
        qx = per["cd_head"].quantile([0.005, 0.995]).to_numpy(dtype=float)
        qy = per["cd_int"].quantile([0.005, 0.995]).to_numpy(dtype=float)
        lo = float(min(qx[0], qy[0]))
        hi = float(max(qx[1], qy[1]))
        pad = 0.06 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, alpha=0.7)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        n_out = int(((per["cd_int"] < lo - pad) | (per["cd_int"] > hi + pad)
                     | (per["cd_head"] < lo - pad) | (per["cd_head"] > hi + pad)).sum())
        if n_out:
            ax.annotate(f"{n_out} outlier(s) beyond view", (0.98, 0.02),
                        xycoords="axes fraction", ha="right", va="bottom",
                        fontsize=6.5, color="dimgray")
        ax.set_xlabel(r"$C_D^{\mathrm{head}}$")
        ax.set_ylabel(r"$C_D^{\mathrm{int}}$")
        plotted = True
    if not plotted:
        print("TODO fig5: no per-sim cd_int/cd_head yet (needs neural runs with both heads)")
        plt.close(fig)
        return None
    # Legend OUTSIDE the axes, one block above: inside upper-left it sat on
    # top of the point cloud (research-figures: an in-axes legend position
    # must be verified empty, and none is here).
    ax.legend(frameon=False, fontsize=6.5, loc="lower left", ncol=3,
              bbox_to_anchor=(-0.05, 1.02), columnspacing=0.9,
              handletextpad=0.4)
    return _save(fig, outdir, "fig05_fsc_scatter")


def fig1_schematic(outdir: Path) -> Path | None:
    """Protocol schematic: three encoders feeding one shared evaluation harness.

    Pure drawing, no data dependency -- always renders.  Kept as a plain
    matplotlib patches diagram (no TikZ/graphviz dependency) so it builds on
    the same Agg-only, no-extra-deps stack as every other figure.
    """
    fig, ax = plt.subplots(figsize=(7.05, 3.1))
    ax.set_xlim(-0.1, 10.1)
    ax.set_ylim(0, 4.3)
    ax.axis("off")

    enc_y0 = 2.75  # bottom of the encoder boxes (compact layout, no dead band)
    # Subtitles kept SHORT: "GINO-style, SDF-conditioned" filled the box wall
    # to wall and read as overflowing at print size (research-figures fix
    # hierarchy rule 1: shorten the text before resizing anything).
    encoders = [
        ("M1  GNN\n$k$NN message passing", style.MODEL_COLORS["gnn"], 1.85),
        ("M2  SDF-FNO\nGINO-style", style.MODEL_COLORS["sdf_fno"], 5.0),
        ("M3  Transolver\nphysics attention", style.MODEL_COLORS["transolver"], 8.15),
    ]
    box_w, box_h = 2.95, 1.0
    for label, color, cx in encoders:
        b = FancyBboxPatch((cx - box_w / 2, enc_y0), box_w, box_h,
                            boxstyle="round,pad=0.06,rounding_size=0.08",
                            linewidth=1.3, edgecolor=color, facecolor=color, alpha=0.16)
        ax.add_patch(b)
        ax.text(cx, enc_y0 + box_h / 2, label, ha="center", va="center", fontsize=8.2,
                 color="black", linespacing=1.5)

    shared_y0, shared_h = 0.55, 1.30
    shared = FancyBboxPatch((0.5, shared_y0), 9.0, shared_h,
                             boxstyle="round,pad=0.06,rounding_size=0.1",
                             linewidth=1.3, edgecolor=style.OKABE_ITO["black"],
                             facecolor="white")
    ax.add_patch(shared)
    # Below the box, clear of the three encoder->box arrows.
    ax.text(5.0, shared_y0 - 0.32, "one shared evaluation protocol",
            ha="center", va="center", fontsize=9, style="italic")

    stages = [
        "force\nintegration\n→ FSC",
        "6 OOD\nsplits\n(shift $\\Delta$)",
        "deep\nensembles +\nsplit conformal",
        "acquisition\n+ Fluent\nverification",
    ]
    n = len(stages)
    slot = 9.0 / n
    for i, s in enumerate(stages):
        cx = 0.5 + slot * (i + 0.5)
        ax.text(cx, shared_y0 + shared_h / 2, s, ha="center", va="center", fontsize=7.6,
                 linespacing=1.4)
        if i > 0:
            ax.plot([0.5 + slot * i, 0.5 + slot * i],
                     [shared_y0 + 0.12, shared_y0 + shared_h - 0.12],
                     color=style.OKABE_ITO["grey"], lw=0.8, alpha=0.6)

    for _, color, cx in encoders:
        arrow = FancyArrowPatch((cx, enc_y0), (cx, shared_y0 + shared_h + 0.02),
                                 arrowstyle="-|>", mutation_scale=12, lw=1.2,
                                 color=color, alpha=0.85)
        ax.add_patch(arrow)

    ax.text(5.0, enc_y0 + box_h + 0.32,
            "same data pipeline, loss family, optimiser, schedule, budget",
            ha="center", va="center", fontsize=7.8, color=style.OKABE_ITO["grey"])
    return _save(fig, outdir, "fig01_schematic")


def fig10_symmetry(df: pd.DataFrame, outdir: Path) -> Path | None:
    """Symmetry (mirror-equivariance) residual per model, grouped bars by split."""
    d = _neural(df)
    metric = "consistency_sym_residual"
    if d.empty or metric not in d.columns or d[metric].isna().all():
        print("TODO fig10: no consistency.sym_residual in results")
        return None
    splits_present = [s for s in style.SPLIT_ORDER if s in set(d["split"])]
    models = style.sort_models(d["model"].unique())
    if not splits_present or not models:
        print("TODO fig10: no (model, split) cells with symmetry residual")
        return None

    fig, ax = plt.subplots(figsize=(7.05, 2.9))
    width = 0.8 / max(1, len(models))
    any_bar = False
    for mi, m in enumerate(models):
        sub = d[d["model"] == m]
        g = sub.groupby("split")[metric].mean().reindex(splits_present)
        xs = np.array([i + (mi - (len(models) - 1) / 2) * width for i in range(len(splits_present))])
        heights = g.to_numpy(dtype=float)
        mask = np.isfinite(heights)
        if not mask.any():
            continue
        ax.bar(xs[mask], heights[mask], width=width, color=style.model_color(m),
               label=style.model_label(m), alpha=0.9)
        any_bar = True
    if not any_bar:
        print("TODO fig10: no finite symmetry-residual values")
        plt.close(fig)
        return None
    ax.set_xticks(range(len(splits_present)))
    ax.set_xticklabels([style.split_label(s) for s in splits_present], rotation=20, ha="right")
    # Linear scale: the residuals span only ~4.4-8.2, a log axis added nothing
    # but a misleading "log scale" label.
    ax.set_ylabel("symmetry residual")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, outdir, "fig10_symmetry")


def fig9_data_efficiency(df: pd.DataFrame, outdir: Path) -> Path | None:
    """Rel-L2 vs training-set size, log-log, one line per model with seed bands.

    Restricted to the ``full`` split before checking distinctness: the
    data-efficiency sweep varies training-set size *within* ``full`` (tagged
    ``n<size>`` runs); without this filter, the fallback ``train_size`` reader
    (CONTEXT.md's per-split manifest ``n_train``) makes the six *different*
    OOD splits look like >=3 distinct sizes and would plot a spurious
    data-efficiency curve that actually mixes distribution shift with data
    scarcity -- the sweep has not run yet (docs/RESULTS.md section 4).
    """
    d = _neural(df)
    # data-eff runs use split labels "full_n25".."full_n400"; the core run is "full"
    # (train_size 700). Match both, but NOT the OOD splits (reynolds/aoa/...), so the
    # curve is pure data scarcity within the full distribution.
    d = d[d["split"].astype(str).str.startswith("full")] if "split" in d.columns else d
    if d.empty or d["train_size"].nunique() < 3:
        print("TODO fig9: need >=3 distinct train_size values (data-efficiency sweep)")
        return None
    # Half of a full-width composite float (paired with fig12); "(a)" is
    # stamped in-figure because REVTeX has no subcaption support.
    fig, ax = plt.subplots(figsize=(3.42, 3.0))
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
    # Explicit ticks at the actual sweep sizes: matplotlib's default log ticks
    # ("3x10^1 4x10^1 6x10^1") overlapped each other at this figure width.
    sizes = sorted(int(s) for s in d["train_size"].dropna().unique())
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("training simulations")
    ax.set_ylabel(r"field $p$ rel. $L_2$")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    style.panel_letter(fig, "a")
    return _save(fig, outdir, "fig09_data_efficiency")


def fig12_cost_accuracy(df: pd.DataFrame, outdir: Path) -> Path | None:
    """Pareto: field rel-L2 against training GPU-hours (ID/full split)."""
    d = _neural(df)
    d = d[d["split"] == "full"] if "split" in d.columns else d
    if d.empty or d["gpu_hours"].isna().all():
        print("TODO fig12: no neural runs with gpu_hours on the full split yet")
        return None
    # Half of a full-width composite float (paired with fig09).
    fig, ax = plt.subplots(figsize=(3.42, 3.0))
    for m in style.sort_models(d["model"].unique()):
        sub = d[d["model"] == m]
        ax.scatter(sub["gpu_hours"], sub["field_p_rel_l2"], s=40,
                   color=style.model_color(m), marker=style.model_marker(m),
                   label=style.model_label(m))
    ax.set_xlabel("training GPU-hours")
    ax.set_ylabel(r"field $p$ rel. $L_2$")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, alpha=0.3)
    style.panel_letter(fig, "b")
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
    # One panel per model, lines coloured by split, axes zoomed to where the
    # data actually lives. The old single-panel version buried 12 near-identical
    # lines under a 12-entry legend in an almost entirely empty [0,1] square.
    models = style.sort_models({m for m, _, _, _ in curves})
    all_vals = [v for _, _, nom, emp in curves for v in list(nom) + list(emp)]
    lo = max(0.0, min(all_vals) - 0.05)
    hi = min(1.005, max(all_vals) + 0.02)
    fig, axes = plt.subplots(1, len(models), figsize=(min(7.05, 2.35 * len(models)), 2.7),
                             sharex=True, sharey=True)
    if len(models) == 1:
        axes = [axes]
    splits_seen: list[str] = []
    for ax, model in zip(axes, models):
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7)
        for m, split, nom, emp in curves:
            if m != model:
                continue
            ax.plot(nom, emp, marker="o", ms=3.5, lw=1.3,
                    color=style.split_color(split), alpha=0.9)
            if split not in splits_seen:
                splits_seen.append(split)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(style.model_label(model), fontsize=8.5)
        ax.set_xlabel("nominal coverage")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("empirical coverage")
    handles = [Line2D([], [], color="k", ls="--", label="ideal")] + [
        Line2D([], [], color=style.split_color(s), marker="o", ms=3.5,
               label=style.split_label(s))
        for s in style.sort_splits(splits_seen)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, 1.06))
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

    # Final printed size: one REVTeX column (3.4 in).
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
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
    handles = (style.model_legend(models, with_marker=True)
               + [Line2D([], [], color="k", ls="--", label="matched cal"),
                  Line2D([], [], color="k", ls="-", label="transfer (cal=full)"),
                  Line2D([], [], color="k", ls=":", label=f"nominal {nominal:g}")])
    # Legend below the axes so it can never sit on top of the coverage lines.
    ax.legend(handles=handles, frameon=False, fontsize=6, loc="upper center",
              bbox_to_anchor=(0.5, -0.28), ncol=2, columnspacing=0.9,
              handletextpad=0.4)
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
    fig, axes = plt.subplots(1, len(panels), figsize=(min(7.05, 3.5 * len(panels)), 3.0))
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
    return _save(fig, outdir, "fig08_interval_width")


def fig11_active_pool(outdir: Path, results: str) -> Path | None:
    """Acquisition-score landscape over the 630-case AL pool, with the three arms marked."""
    path = Path(results) / "active" / "acquisition_ranking.csv"
    if not path.is_file():
        print("TODO fig11: no results/active/acquisition_ranking.csv (run_active.py not run yet)")
        return None
    try:
        pool = pd.read_csv(path)
    except Exception as exc:
        print(f"TODO fig11: unreadable acquisition_ranking.csv ({exc})")
        return None
    need = {"aoa_deg", "re", "acq_score",
            "selected_acquisition", "selected_variance", "selected_random"}
    if pool.empty or not need.issubset(pool.columns):
        print("TODO fig11: acquisition_ranking.csv missing expected columns")
        return None

    # Half of a full-width composite float (paired with fig11b).
    fig, ax = plt.subplots(figsize=(3.42, 3.2))
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.35, 0.35, size=len(pool))
    # Re in millions so the colorbar needs no "1e6" offset text (which used to
    # collide with the figure title).
    sc = ax.scatter(pool["aoa_deg"] + jitter, pool["acq_score"], c=pool["re"] / 1e6,
                     cmap="viridis", s=6, alpha=0.45, linewidths=0)
    # Deterministic per-arm x offsets: the acquisition and variance arms pick
    # overlapping alpha=18 cases, and without the offsets their markers stacked
    # on top of each other.
    arms = (
        ("selected_acquisition", "acquisition (k=8)", "*", 75, style.OKABE_ITO["vermillion"], -0.55),
        ("selected_variance", "ensemble variance (k=8)", "^", 34, style.OKABE_ITO["blue"], 0.55),
        ("selected_random", "random (k=8)", "s", 26, style.OKABE_ITO["grey"], 0.0),
    )
    for col, label, marker, size, color, dx in arms:
        sub = pool[pool[col] == 1]
        if sub.empty:
            continue
        ax.scatter(sub["aoa_deg"] + dx, sub["acq_score"], marker=marker, s=size,
                   facecolor=color, edgecolor="black", linewidths=0.6, label=label, zorder=5)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$\mathrm{Re}\ (\times 10^6)$", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    ax.set_xlabel(r"angle of attack $\alpha$ (deg)")
    ax.set_ylabel("acquisition score $a$")
    ax.legend(frameon=False, fontsize=6, loc="upper left", handletextpad=0.3,
              borderaxespad=0.2)
    ax.grid(True, alpha=0.3)
    style.panel_letter(fig, "a")
    return _save(fig, outdir, "fig11_active_pool")


def fig11_active_verify(outdir: Path, results: str) -> Path | None:
    """Fig 11(b): surrogate-vs-Fluent |CD error| by arm on the accepted set, with
    the acquisition/variance arms shown as a 'no steady solution' band + their
    ensemble sigma. Uses cd_head (input-robust surrogate CD; see compare_fluent)."""
    svf = Path(results) / "fluent" / "surrogate_vs_fluent.csv"
    if not svf.is_file():
        print("TODO fig11b: no results/fluent/surrogate_vs_fluent.csv (compare_fluent not run)")
        return None
    df = pd.read_csv(svf)
    t = df[df["model"] == "transolver"].copy()
    if t.empty:
        print("TODO fig11b: no transolver rows in surrogate_vs_fluent.csv")
        return None

    acc = t[t["status"].isin(["converged", "quasi_steady"])].copy()
    div = t[t["status"] == "diverged"].copy()
    xlabels = {"gridstudy": "grid", "offset": "offset", "random": "random",
               "variance": "variance", "acquisition": "acq."}
    order = ["gridstudy", "offset", "random", "variance", "acquisition"]
    xpos = {a: i for i, a in enumerate(order)}

    # Half of a full-width composite float (paired with fig11a).
    fig, ax = plt.subplots(figsize=(3.42, 3.5))
    # 90% conformal half-width reference band, on the SAME quantity as the y-axis
    # (cd_head). Round-2 fix: the figure previously drew the cd_int half-width under
    # cd_head errors, which disagreed with the reported cd_head coverage.
    half = None
    if "q90_halfwidth_cd_head" in t and t["q90_halfwidth_cd_head"].notna().any():
        half = float(t["q90_halfwidth_cd_head"].dropna().iloc[0])

    rng = np.random.default_rng(1)
    plotted_any = False
    counts = {}
    for arm in order:
        sub = acc[acc["arm"] == arm]
        ndiv = int((div["arm"] == arm).shape[0]) if div.empty else int((div["arm"] == arm).sum())
        counts[arm] = (len(sub), ndiv)
        for _, r in sub.iterrows():
            err = abs(r["err_cd_head"]) if np.isfinite(r["err_cd_head"]) else np.nan
            if not np.isfinite(err) or err <= 0:
                continue
            filled = r["status"] == "converged"
            deep = float(r.get("aoa_deg", 0) or 0) >= 17.0   # post-stall pick
            jit = rng.uniform(-0.16, 0.16)
            col = style.OKABE_ITO["vermillion"] if deep else style.OKABE_ITO["blue"]
            ax.errorbar(xpos[arm] + jit, err, yerr=float(r.get("cd_head_std", 0) or 0),
                        marker=("s" if deep else "o"), ms=5, capsize=1.5,
                        mfc=(col if filled else "white"), mec=col,
                        ecolor=style.OKABE_ITO["grey"], lw=0.8, zorder=5)
            plotted_any = True

    if plotted_any:
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 1.4)
    if half is not None:
        ax.axhspan(ax.get_ylim()[0], half, color=style.OKABE_ITO["green"], alpha=0.12, zorder=0)
        ax.axhline(half, color=style.OKABE_ITO["green"], lw=1.0, ls="--")
    # Legend proxies for the regime marker. Labels stay SHORT and the legend
    # sits in the verified-empty lower-right region (no accepted point below
    # the half-width band for the variance/acquisition columns): at column
    # width the old center-right legend sat on the random-column data.
    from matplotlib.lines import Line2D
    proxies = [Line2D([0], [0], marker="o", color="w", mec=style.OKABE_ITO["blue"],
                      mfc=style.OKABE_ITO["blue"], label="moderate-α accepted"),
               Line2D([0], [0], marker="s", color="w", mec=style.OKABE_ITO["vermillion"],
                      mfc=style.OKABE_ITO["vermillion"], label="post-stall accepted")]
    if half is not None:
        proxies.append(Line2D([0], [0], color=style.OKABE_ITO["green"], ls="--",
                              label="90% half-width"))

    # Per-arm accepted/non-settling counts go into the tick labels: as tiny
    # in-axes annotations they crammed against the panel edge and each other
    # at column width (caught by the reader, not the QA pass).
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels(
        [f"{xlabels[a]}\n{counts.get(a, (0, 0))[0]}/{counts.get(a, (0, 0))[1]}"
         for a in order], fontsize=6.5)
    ax.set_ylabel(r"$|C_D^{\mathrm{head}} - C_D^{\mathrm{Fluent,corr}}|$")
    ax.legend(handles=proxies, frameon=False, fontsize=6, loc="lower right",
              handletextpad=0.3, borderaxespad=0.3)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    style.panel_letter(fig, "b")
    return _save(fig, outdir, "fig11_active_verify")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="paper/figures")
    args = ap.parse_args(argv)

    style.apply_style()
    outdir = Path(args.outdir)
    df = vdata.load_runs(args.results)

    written: list[Path] = []
    p = fig1_schematic(outdir)
    if p:
        written.append(p)
    for fn in (fig4_error_vs_shift, fig5_fsc_scatter, fig9_data_efficiency,
               fig10_symmetry, fig12_cost_accuracy):
        p = fn(df, outdir)
        if p:
            written.append(p)
    for fn in (fig6_reliability, fig7_coverage_vs_shift, fig8_interval_width,
               fig11_active_pool, fig11_active_verify):
        p = fn(outdir, args.results)
        if p:
            written.append(p)

    print(f"\n{len(written)} figure(s) written to {outdir}:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
