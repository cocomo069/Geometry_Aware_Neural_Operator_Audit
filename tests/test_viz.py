"""Tests for the figure/table pipeline (src.viz + scripts.make_figures/tables).

CPU-only, headless (Agg). Uses the committed baseline runs under results/ as live
fixtures and synthesises extra runs in tmp_path for cases the baselines can't cover.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from src.viz import data as vdata  # noqa: E402
from src.viz import style  # noqa: E402
from scripts import make_figures as mf  # noqa: E402
from scripts import make_tables as mt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


# --------------------------------------------------------------------------- #
# Data loading on the real committed runs
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not RESULTS.exists(), reason="no results/ committed")
def test_load_runs_parses_baselines():
    df = vdata.load_runs(RESULTS)
    assert not df.empty
    # the 12 committed baseline runs: constant/ridge x 6 splits
    base = df[df["is_baseline"]] if "is_baseline" in df.columns else df
    assert base["model"].isin({"constant", "ridge"}).any()
    for col in ("model", "split", "field_p_rel_l2", "coef_cd_head_spearman"):
        assert col in df.columns


@pytest.mark.skipif(not RESULTS.exists(), reason="no results/ committed")
def test_per_sim_has_fsc_columns():
    df = vdata.load_runs(RESULTS)
    per = vdata.load_all_per_sim(runs=df)
    if per.empty:
        pytest.skip("no per_sim.csv present")
    for col in ("cd_int", "cd_head", "fsc_cd"):
        assert col in per.columns


# --------------------------------------------------------------------------- #
# Style invariants
# --------------------------------------------------------------------------- #
def test_style_colors_stable_and_distinct():
    style.apply_style()
    cols = {m: style.model_color(m) for m in style.MODEL_ORDER}
    assert len(set(cols.values())) == len(cols)  # one distinct colour per model
    # unknown model still returns a colour (fallback), never raises
    assert isinstance(style.model_color("mystery"), str)


def test_sort_models_orders_known_first():
    out = style.sort_models(["ridge", "gnn", "zzz", "constant"])
    assert out.index("gnn") < out.index("ridge") < out.index("constant")
    assert out[-1] == "zzz"


# --------------------------------------------------------------------------- #
# Figures run headless on real data and write files
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not RESULTS.exists(), reason="no results/ committed")
def test_fig4_and_fig5_write_files(tmp_path):
    style.apply_style()
    df = vdata.load_runs(RESULTS)
    p4 = mf.fig4_error_vs_shift(df, tmp_path)
    p5 = mf.fig5_fsc_scatter(df, tmp_path)
    # baselines cover both field error (fig4) and cd_int/cd_head (fig5)
    assert p4 is not None and p4.exists()
    assert p5 is not None and p5.exists()
    assert (tmp_path / "fig04_error_vs_shift.png").exists()


def test_gated_figures_skip_cleanly_when_empty(tmp_path):
    empty = vdata.load_runs(tmp_path)  # no runs
    assert mf.fig9_data_efficiency(empty, tmp_path) is None
    assert mf.fig12_cost_accuracy(empty, tmp_path) is None
    assert mf.fig6_reliability(tmp_path, str(tmp_path)) is None


def test_fig9_renders_with_synthetic_data_efficiency(tmp_path):
    """Synthesise runs at >=3 train sizes so the data-efficiency figure renders."""
    rows = []
    for size in (25, 100, 400):
        for seed in (0, 1):
            rid = f"gnn_full_s{seed}_n{size}"
            d = tmp_path / rid
            d.mkdir()
            (d / "metrics.json").write_text(json.dumps({
                "run_id": rid, "model": "gnn", "split": "full", "seed": seed,
                "tag": f"n{size}", "params": 1_400_000, "epochs": 400,
                "field": {"p_rel_l2": 0.5 / size ** 0.2, "tau_rel_l2": 0.6,
                          "p_mae": 0.1, "tau_mae": 0.1},
                "coef": {"cl_head_mae": 0.1, "cd_head_mae": 0.01, "cl_int_mae": 0.1,
                         "cd_int_mae": 0.01, "cd_spearman": 0.9, "cd_head_spearman": 0.9,
                         "cd_int_spearman": 0.9, "cl_rel": 0.1, "cd_rel": 0.1},
                "consistency": {"fsc_cl": 0.01, "fsc_cd": 0.001, "fsc_rel_cl": 0.1,
                                "fsc_rel_cd": 0.1, "sym_residual": 0.01, "antisym_cl_gap": 0.01},
                "cost": {"infer_ms_per_sim": 5.0, "peak_mem_mb": 500.0},
            }))
            (d / "config.yaml").write_text(f"data:\n  train_size: {size}\n")
            rows.append(rid)
    df = vdata.load_runs(tmp_path)
    p9 = mf.fig9_data_efficiency(df, tmp_path)
    assert p9 is not None and p9.exists()


# --------------------------------------------------------------------------- #
# Tables render LaTeX + Markdown, compile-safe
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not RESULTS.exists(), reason="no results/ committed")
def test_tables_write_latex_and_markdown(tmp_path):
    df = vdata.load_runs(RESULTS)
    t1 = mt.table1_indist(df, tmp_path)
    t2 = mt.table2_ood(df, tmp_path)
    assert any(p.suffix == ".tex" for p in t1)
    assert any(p.suffix == ".md" for p in t2)
    tex = (tmp_path / "tab1_indist.tex").read_text()
    assert r"\begin{tabular}" in tex and r"\bottomrule" in tex
    # Float BODY only: the paper wraps each \input in its own table/table*
    # environment (single- vs two-column layout), so the generated file must
    # carry caption+label+tabular but NOT a \begin{table} of its own.
    assert r"\caption{" in tex and r"\label{" in tex
    assert "\\begin{table}" not in tex


def test_table_best_is_bolded():
    df = pd.DataFrame({
        "model": ["gnn", "ridge"], "split": ["full", "full"],
        "field_p_rel_l2": [0.2, 0.9], "is_baseline": [False, True],
    })
    md = mt._render(df, mt.TABLE1_COLS, "model", caption="c", label="l", latex=False)
    # gnn has the lower (better) rel-L2 -> bolded in markdown
    assert "**0.2**" in md
