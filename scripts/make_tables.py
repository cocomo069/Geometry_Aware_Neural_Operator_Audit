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
    """Render a model/row table with per-column best bolded.

    Multiple rows per ``row_key`` (e.g. ensemble seeds on the same split) are
    averaged to one row first, so seed-0 core runs and their seed-1..4 ensemble
    members collapse to a single per-model number instead of colliding.
    """
    num_cols = [c for c, _, _, _ in cols if c in df.columns]
    df = (df.groupby(row_key, as_index=False)[num_cols].mean()
          if num_cols else df.drop_duplicates(subset=[row_key]))
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


# --------------------------------------------------------------------------- #
# Table 3 -- calibration (reads results/uq, not the run metrics)
# --------------------------------------------------------------------------- #
_UQ_SPLIT_ORDER = ("full", "scarce", "shape5", "reynolds", "aoa", "combined")


def _uq_mode(ensemble_id: str) -> str:
    return "transfer" if str(ensemble_id).endswith("_calfull") else "matched"


def _largest_k(reports, mode: str) -> dict:
    """``{(model, split): report}`` keeping the largest-K report for ``mode``."""
    best: dict[tuple[str, str], dict] = {}
    for rep in reports:
        if _uq_mode(rep.get("ensemble_id", "")) != mode:
            continue
        key = (str(rep.get("model", "")), str(rep.get("split", "")))
        cur = best.get(key)
        if cur is None or float(rep.get("k", 0) or 0) > float(cur.get("k", 0) or 0):
            best[key] = rep
    return best


def _pick_record(rep, target: str, granularity: str):
    recs = [
        r for r in (rep.get("records", []) if rep else [])
        if isinstance(r, dict)
        and str(r.get("target")) == target
        and str(r.get("granularity")) == granularity
    ]
    if not recs:
        return None
    recs.sort(key=lambda r: 0 if r.get("score") == "normalized" else 1)
    return recs[0]


def _lv(rec, nominal: float, key: str):
    if rec is None:
        return None
    for lv in rec.get("levels", []):
        if abs(float(lv.get("nominal", -1)) - nominal) < 1e-9:
            return lv.get(key)
    return None


def table3_calibration(results: str, outdir: Path) -> list[Path]:
    """Coverage / width / ECE for the C_D^int band, per model x split.

    Columns: matched coverage at 0.8/0.9/0.95, transfer coverage at 0.9, matched
    mean width at 0.9, matched ECE.  Reads ``results/uq/*.json`` (produced by
    ``scripts/run_uq.py``); missing cells render as ``--``.
    """
    reports = vdata.load_uq_reports(Path(results) / "uq")
    if not reports:
        print("TODO table3: no results/uq calibration data")
        return []
    matched = _largest_k(reports, "matched")
    transfer = _largest_k(reports, "transfer")
    def _order(ms: tuple[str, str]) -> tuple[int, int]:
        model, split = ms
        mi = style.MODEL_ORDER.index(model) if model in style.MODEL_ORDER else 99
        si = _UQ_SPLIT_ORDER.index(split) if split in _UQ_SPLIT_ORDER else 99
        return (mi, si)

    keys = sorted(set(matched) | set(transfer), key=_order)
    if not keys:
        print("TODO table3: no calibration records")
        return []

    headers = ["model", "split", "cov@.8", "cov@.9", "cov@.95",
               "cov@.9 (tr)", "width@.9", "ECE"]
    rows = []
    for model, split in keys:
        m_rep = matched.get((model, split))
        t_rep = transfer.get((model, split))
        m_rec = _pick_record(m_rep, "cd_int", "coefficient")
        t_rec = _pick_record(t_rep, "cd_int", "coefficient")
        rows.append([
            style.model_label(model),
            style.split_label(split),
            _fmt(_lv(m_rec, 0.8, "coverage"), 3),
            _fmt(_lv(m_rec, 0.9, "coverage"), 3),
            _fmt(_lv(m_rec, 0.95, "coverage"), 3),
            _fmt(_lv(t_rec, 0.9, "coverage"), 3),
            _fmt(_lv(m_rec, 0.9, "mean_width"), 3),
            _fmt(None if m_rec is None else m_rec.get("ece"), 3),
        ])

    caption = (r"Split-conformal calibration of the integrated drag band "
               r"($C_D^{\mathrm{int}}$): empirical coverage, mean interval width and "
               r"ECE. Matched calibrates on the split's own cal set; transfer (tr) "
               r"calibrates on \texttt{full}'s cal.")
    paths = []
    for ext, latex in (("tex", True), ("md", False)):
        if latex:
            lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}",
                     r"\label{tab:calibration}",
                     r"\begin{tabular}{ll" + "r" * (len(headers) - 2) + "}",
                     r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
            lines += [" & ".join(r) + r" \\" for r in rows]
            lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
            txt = "\n".join(lines)
        else:
            lines = ["| " + " | ".join(headers) + " |",
                     "| " + " | ".join("---" for _ in headers) + " |"]
            lines += ["| " + " | ".join(r) + " |" for r in rows]
            txt = "\n".join(lines) + "\n"
        outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / f"tab3_calibration.{ext}"
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
    written += table3_calibration(args.results, outdir)

    print(f"\n{len(written)} table file(s) written to {outdir}:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
