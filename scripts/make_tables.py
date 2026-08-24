"""Generate paper tables (LaTeX + Markdown) from run metrics (PLAN.md Phase 5).

    .venv/Scripts/python.exe -m scripts.make_tables [--results results] [--outdir paper/tables]

Tables (spec section 8):
  1  in-distribution accuracy, consistency, cost (full split)
  2  OOD results, models x splits (field + coefficient error)
  3  calibration coverage/width (gated on results/uq)

Each table is written as ``tab<N>_<slug>.tex`` (booktabs, \\input-able) and
``tab<N>_<slug>.md`` (README mirror). Missing cells render as ``--``; the best
value per column is bolded. Runs that do not exist yet simply produce a table of
whatever is present, so this is safe to run at any project stage.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.viz import data as vdata
from src.viz import style

DASH = "--"


def _fmt(x: float | None, prec: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return DASH
    return f"{x:.{prec}g}"


def _best_mask(series: pd.Series, lower_is_better: bool) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() == 0:
        return pd.Series(False, index=series.index)
    target = vals.min() if lower_is_better else vals.max()
    return vals == target


# column spec: (dataframe column, header, lower_is_better, precision)
TABLE1_COLS = [
    ("field_p_rel_l2", r"$p$ rel-$L_2$", True, 3),
    ("field_tau_rel_l2", r"$\tau_w$ rel-$L_2$", True, 3),
    ("coef_cd_head_mae", r"$C_D$ MAE", True, 3),
    ("coef_cd_head_spearman", r"$C_D$ $\rho$", False, 3),
    ("consistency_fsc_cd", "FSC $C_D$", True, 3),
    ("consistency_sym_residual", "sym", True, 3),
    ("gpu_hours", "GPU-h", True, 3),
]


def _render(df: pd.DataFrame, cols, row_key: str, *, caption: str, label: str,
            latex: bool) -> str:
    """Render a model/row table with per-column best bolded."""
    rows = style.sort_models(df[row_key].unique()) if row_key == "model" else sorted(df[row_key].unique())
    best = {c: _best_mask(df.set_index(row_key)[c], lo) for c, _, lo, _ in cols if c in df.columns}

    def cell(rk: str, col: str, prec: int) -> str:
        sub = df[df[row_key] == rk]
        if sub.empty or col not in sub.columns:
            return DASH
        val = pd.to_numeric(sub[col], errors="coerce").mean()
        s = _fmt(val, prec)
        if s != DASH and col in best and rk in best[col].index and bool(best[col].get(rk, False)):
            s = (r"\textbf{" + s + "}") if latex else f"**{s}**"
        return s

    headers = [row_key] + [h for _, h, _, _ in cols]
    body = [[style.model_label(rk) if row_key == "model" else str(rk)]
            + [cell(rk, c, p) for c, _, _, p in cols] for rk in rows]

    if latex:
        ncol = len(headers)
        out = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}",
               f"\\label{{{label}}}", r"\begin{tabular}{l" + "r" * (ncol - 1) + "}",
               r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
        out += [" & ".join(r) + r" \\" for r in body]
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        return "\n".join(out)
    # markdown
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out) + "\n"


def _write(outdir: Path, slug: str, df: pd.DataFrame, cols, *, caption: str, label: str) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, latex in (("tex", True), ("md", False)):
        p = outdir / f"{slug}.{ext}"
        p.write_text(_render(df, cols, "model", caption=caption, label=label, latex=latex),
                     encoding="utf-8")
        paths.append(p)
    return paths


def table1_indist(df: pd.DataFrame, outdir: Path) -> list[Path]:
    sub = df[df["split"] == "full"] if "split" in df.columns else df
    if sub.empty:
        print("TODO table1: no full-split runs")
        return []
    return _write(outdir, "tab1_indist", sub, TABLE1_COLS,
                  caption="In-distribution accuracy, consistency and cost (AirfRANS \\texttt{full}).",
                  label="tab:indist")


def table2_ood(df: pd.DataFrame, outdir: Path) -> list[Path]:
    """One row per model, columns = field rel-L2 per split."""
    if df.empty:
        print("TODO table2: no runs")
        return []
    splits = style.sort_splits(df["split"].unique())
    piv = (df.pivot_table(index="model", columns="split", values="field_p_rel_l2", aggfunc="mean")
             .reindex(columns=splits))
    models = style.sort_models(piv.index)
    best = {s: _best_mask(piv[s], True) for s in splits}

    def cell(m, s, latex):
        v = piv.loc[m, s] if (m in piv.index and s in piv.columns) else None
        txt = _fmt(v, 3)
        if txt != DASH and bool(best[s].get(m, False)):
            txt = (r"\textbf{" + txt + "}") if latex else f"**{txt}**"
        return txt

    paths = []
    for ext, latex in (("tex", True), ("md", False)):
        headers = ["model"] + [style.split_label(s) for s in splits]
        body = [[style.model_label(m)] + [cell(m, s, latex) for s in splits] for m in models]
        if latex:
            lines = [r"\begin{table}[t]", r"\centering",
                     r"\caption{Out-of-distribution field $p$ rel-$L_2$ by split.}",
                     r"\label{tab:ood}", r"\begin{tabular}{l" + "r" * len(splits) + "}",
                     r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
            lines += [" & ".join(r) + r" \\" for r in body]
            lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
            txt = "\n".join(lines)
        else:
            lines = ["| " + " | ".join(headers) + " |",
                     "| " + " | ".join("---" for _ in headers) + " |"]
            lines += ["| " + " | ".join(r) + " |" for r in body]
            txt = "\n".join(lines) + "\n"
        p = outdir / f"tab2_ood.{ext}"
        p.write_text(txt, encoding="utf-8")
        paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="paper/tables")
    args = ap.parse_args(argv)

    df = vdata.load_runs(args.results)
    outdir = Path(args.outdir)
    written: list[Path] = []
    written += table1_indist(df, outdir)
    written += table2_ood(df, outdir)

    print(f"\n{len(written)} table file(s) written to {outdir}:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
