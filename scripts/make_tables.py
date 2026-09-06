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

    Multiple rows per ``row_key`` (e.g. the 3-5 seeds of a core run on the same
    split) are reported as ``mean ± std`` over those seeds; a single row shows
    the bare mean. The per-column "best" (bolded) is decided on the seed mean.
    Callers are expected to have already dropped tagged runs (smoke / ablation)
    and non-canonical splits so only comparable core runs reach here.
    """
    num_cols = [c for c, _, _, _ in cols if c in df.columns]
    means = (df.groupby(row_key, as_index=True)[num_cols].mean()
             if num_cols else pd.DataFrame())
    rows = style.sort_models(df[row_key].unique()) if row_key == "model" else sorted(df[row_key].unique())
    best = {c: _best_mask(means[c], lo) for c, _, lo, _ in cols if c in means.columns}
    pm = r"$\pm$" if latex else "±"

    def cell(rk: str, col: str, prec: int) -> str:
        if col not in df.columns:
            return DASH
        vals = pd.to_numeric(df.loc[df[row_key] == rk, col], errors="coerce").dropna()
        if vals.empty:
            return DASH
        s = _fmt(vals.mean(), prec)
        if s == DASH:
            return DASH
        if len(vals) > 1:
            sd = vals.std(ddof=1)
            if np.isfinite(sd) and sd > 0:
                s = f"{s} {pm} {_fmt(sd, prec)}"
        if col in best and rk in best[col].index and bool(best[col].get(rk, False)):
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
    core = df[df["tag"].isna()] if "tag" in df.columns else df
    sub = core[core["split"] == "full"] if "split" in core.columns else core
    if sub.empty:
        print("TODO table1: no full-split runs")
        return []
    return _write(outdir, "tab1_indist", sub, TABLE1_COLS,
                  caption="In-distribution accuracy, consistency and cost (AirfRANS \\texttt{full}).",
                  label="tab:indist")


def table2_ood(df: pd.DataFrame, outdir: Path) -> list[Path]:
    """One row per model, columns = field rel-L2 per split (mean ± std over seeds).

    Only untagged core runs on the canonical OOD splits are shown; smoke and
    ablation runs (tagged) and the data-efficiency sub-splits (``full_n*``,
    which belong to Figure 9, not the OOD table) are excluded.
    """
    core = df[df["tag"].isna()] if "tag" in df.columns else df
    core = core[core["split"].isin(style.SPLIT_ORDER)]
    if core.empty:
        print("TODO table2: no runs")
        return []
    splits = style.sort_splits(core["split"].unique())
    piv = (core.pivot_table(index="model", columns="split", values="field_p_rel_l2", aggfunc="mean")
              .reindex(columns=splits))
    piv_sd = (core.pivot_table(index="model", columns="split", values="field_p_rel_l2", aggfunc="std")
                 .reindex(columns=splits))
    models = style.sort_models(piv.index)
    best = {s: _best_mask(piv[s], True) for s in splits}

    def cell(m, s, latex):
        v = piv.loc[m, s] if (m in piv.index and s in piv.columns) else None
        txt = _fmt(v, 3)
        if txt == DASH:
            return DASH
        sd = piv_sd.loc[m, s] if (m in piv_sd.index and s in piv_sd.columns) else None
        pm = r"$\pm$" if latex else "±"
        if sd is not None and np.isfinite(sd) and sd > 0:
            txt = f"{txt} {pm} {_fmt(sd, 3)}"
        if bool(best[s].get(m, False)):
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


def _read_csv_dicts(path: Path):
    import csv
    if not path.exists():
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _num(x, prec=5):
    try:
        v = float(x)
        return DASH if (v != v) else f"{v:.{prec}g}"
    except (TypeError, ValueError):
        return DASH


def table5_fluent(results: str, outdir: Path) -> list[Path]:
    """Table 5 (a/b/c): AL selection+outcome, solver offset, verified comparison.

    Reads results/fluent/{fluent_summary.csv, offset.csv, surrogate_vs_fluent.csv}.
    Bespoke stacked table (case rows), independent of the model-row _render helper.
    Uses cd_head as the surrogate CD (input-robust; see compare_fluent).
    """
    import json as _json
    fdir = Path(results) / "fluent"
    summ = _read_csv_dicts(fdir / "fluent_summary.csv")
    off = _read_csv_dicts(fdir / "offset.csv")
    svf = _read_csv_dicts(fdir / "surrogate_vs_fluent.csv")
    if not summ:
        print("TODO table5: results/fluent/fluent_summary.csv missing")
        return []

    # ---- 5a: acquisition selection + steady outcome (S0/r1) + surrogate cd_head
    def status_of(cid, model, variant):
        for r in summ:
            if r["case_id"] == cid and r["model"] == model and r["variant"] == variant:
                return r["status"]
        return "not_run"
    acq = sorted({r["case_id"] for r in summ if r["arm"] == "acquisition"})
    svf_t = {r["case_id"]: r for r in svf if r["model"] == "transolver"}
    a_rows = []
    for cid in acq:
        base = next(r for r in summ if r["case_id"] == cid and r["model"] == "sa")
        s0 = status_of(cid, "sa", "s0")
        r1 = status_of(cid, "sa", "r1")
        sr = svf_t.get(cid, {})
        cdh = _num(sr.get("cd_head_mean"), 4)
        cds = _num(sr.get("cd_head_std"), 2)
        a_rows.append([cid.replace("al_acq_", ""), base["naca"], _num(base["re"], 2),
                       _num(base["aoa_deg"], 3), s0, r1,
                       (cdh + " $\\pm$ " + cds) if cdh != DASH else DASH])
    a_head = ["case", "NACA", "Re", "alpha", "S0", "r1", "surrogate CD_head"]

    # ---- 5b: solver offset
    b_rows = []
    off_sa = {r["sim"]: r for r in off if r["model"] == "sa"}
    off_sst = {r["sim"]: r for r in off if r["model"] == "sst"}
    for sim in sorted(off_sa):
        r = off_sa[sim]
        rst = off_sst.get(sim, {})
        b_rows.append([r["case_id"], _num(r["cd_airfrans"], 4), _num(r["cl_airfrans"], 3),
                       _num(r["cd_fluent"], 4), _num(rst.get("cd_fluent"), 4),
                       _num(r["delta_cd"], 3), _num(r["reldelta_cd"], 3), _num(r["delta_cl"], 3)])
    b_head = ["replica", "CD_AF", "CL_AF", "CD_SA", "CD_SST", "dCD", "rel dCD", "dCL"]
    # footer mean+/-s
    osum = {}
    p = fdir / "offset_summary.json"
    if p.exists():
        osum = _json.loads(p.read_text(encoding="utf-8"))

    # ---- 5c: verified comparison on the accepted set (all models, cd_head)
    svf_by_case = {}
    for r in svf:
        if r["status"] in ("converged", "quasi_steady"):
            svf_by_case.setdefault(r["case_id"], {})[r["model"]] = r
    c_rows = []
    for cid in sorted(svf_by_case):
        d = svf_by_case[cid]
        ref = d.get("transolver") or next(iter(d.values()))
        def sc(m):
            r = d.get(m)
            if not r:
                return DASH
            return _num(r["cd_head_mean"], 4) + "$\\pm$" + _num(r["cd_head_std"], 1)
        cov = d.get("transolver", {}).get("covered90_cd_head", "")
        c_rows.append([cid, ref["arm"], ref["naca"], _num(ref["re"], 2), _num(ref["aoa_deg"], 3),
                       ref["status"], _num(ref["cd_fluent"], 4), _num(ref["cd_fluent_corrected"], 4),
                       sc("transolver"), sc("sdf_fno"), sc("gnn"), str(cov)])
    c_head = ["case", "arm", "NACA", "Re", "alpha", "status", "CD_fl", "CD_fl-dbar",
              "Transolver", "SDF-FNO", "GNN", "cov90 (Transolver)"]

    def md_table(head, rows):
        L = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
        L += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        return "\n".join(L)

    n_acc = len(c_rows)
    gci_note = ""
    gp = fdir / "gci.json"
    if gp.exists():
        g = _json.loads(gp.read_text(encoding="utf-8"))
        if "sa" in g:
            gci_note = f"GCI_fine CD(SA)={100*g['sa']['cd']['gci_fine']:.2f}%"
    md = ["# Table 5 -- Fluent verification (a selection+outcome, b offset, c verified)",
          "",
          f"Caption: grid-converged at L2 ({gci_note}); n = {n_acc}, "
          "qualitative external check. Surrogate CD is the input-robust cd_head.",
          "", "## 5a Acquisition selection + steady outcome",
          md_table(a_head, a_rows), "",
          "## 5b Solver offset (6 AirfRANS replicas)",
          md_table(b_head, b_rows)]
    if osum:
        for m in ("sa", "sst"):
            if m in osum:
                s = osum[m]
                md.append(f"- {m.upper()}: dbar_CD={s['dbar_cd']:+.4g} (s={s['s_cd']:.2g}), "
                          f"dbar_CL={s['dbar_cl']:+.3g} (s={s['s_cl']:.2g}), n={s['n']}")
    md += ["", "## 5c Verified surrogate-vs-Fluent comparison (accepted set)",
           md_table(c_head, c_rows), ""]

    outdir.mkdir(parents=True, exist_ok=True)
    md_path = outdir / "tab5_fluent.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    # a compact LaTeX version (5c is the headline; 5a/5b as separate tabulars)
    def tex_table(head, rows, caption, label):
        ncol = len(head)
        L = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", r"\small",
             r"\begin{tabular}{" + "l" * ncol + "}", r"\toprule",
             " & ".join(h.replace("_", r"\_") for h in head) + r" \\", r"\midrule"]
        L += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        return "\n".join(L)
    tex = tex_table(c_head, c_rows,
                    f"Surrogate vs Fluent on the accepted steady set (n={n_acc}, "
                    f"grid-converged at L2, {gci_note}; qualitative external check). "
                    "Surrogate CD is the input-robust coefficient head.",
                    "tab:fluent_verify")
    tex += "\n" + tex_table(b_head, b_rows, "Solver offset: six AirfRANS replicas.",
                            "tab:fluent_offset")
    tex_path = outdir / "tab5_fluent.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return [md_path, tex_path]


def _cell_metric(results: str, run_id: str, path: tuple) -> str:
    p = Path(results) / run_id / "metrics.json"
    if not p.exists():
        return DASH
    import json
    d = json.loads(p.read_text())
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return _fmt(d, 4) if isinstance(d, (int, float)) else DASH


def table4_ablation(results: str, outdir: Path) -> list[Path]:
    """Ablations: M2 conditioning (sdf/mask/sdf+normals) and M1 physics-loss (lamF).

    Reads the tagged ablation run dirs against their core-grid baselines. Skips
    with a TODO if the ablation runs are not present yet.
    """
    r = Path(results)
    have = any((r / f"sdf_fno_full_s0_cond_mask").exists() for _ in [0]) or \
        (r / "gnn_full_s0_lamF").exists()
    if not have:
        print("TODO table4: ablation runs (cond_mask/cond_sdfnrm/lamF) not present")
        return []
    P = ("field", "p_rel_l2")
    F = ("consistency", "fsc_cd")
    lines_md = ["### M2 geometry conditioning (field p rel-L2)", "",
                "| conditioning | full | shape5 (OOD) |", "| --- | --- | --- |",
                f"| SDF (baseline) | {_cell_metric(results,'sdf_fno_full_s0',P)} | {_cell_metric(results,'sdf_fno_shape5_s0',P)} |",
                f"| binary mask | {_cell_metric(results,'sdf_fno_full_s0_cond_mask',P)} | {_cell_metric(results,'sdf_fno_shape5_s0_cond_mask',P)} |",
                f"| SDF + normals | {_cell_metric(results,'sdf_fno_full_s0_cond_sdfnrm',P)} | {_cell_metric(results,'sdf_fno_shape5_s0_cond_sdfnrm',P)} |",
                "", "### M1 physics/force-consistency loss", "",
                "| GNN variant | full p rel-L2 | full FSC | combined p rel-L2 | combined FSC |",
                "| --- | --- | --- | --- | --- |",
                f"| baseline (λ_F=0) | {_cell_metric(results,'gnn_full_s0',P)} | {_cell_metric(results,'gnn_full_s0',F)} | {_cell_metric(results,'gnn_combined_s0',P)} | {_cell_metric(results,'gnn_combined_s0',F)} |",
                f"| + force loss | {_cell_metric(results,'gnn_full_s0_lamF',P)} | {_cell_metric(results,'gnn_full_s0_lamF',F)} | {_cell_metric(results,'gnn_combined_s0_lamF',P)} | {_cell_metric(results,'gnn_combined_s0_lamF',F)} |",
                ""]
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "tab4_ablation.md").write_text("\n".join(lines_md) + "\n", encoding="utf-8")
    # LaTeX: two small tabulars in one file
    tex = (lines_md[0] + "\n\n" + "\n".join(lines_md)).replace("### ", "% ")
    (outdir / "tab4_ablation.tex").write_text(
        "% Table 4 -- ablations (see tab4_ablation.md for the readable form)\n"
        "% M2 conditioning and M1 force-loss; values are field p rel-L2 / FSC.\n",
        encoding="utf-8")
    return [outdir / "tab4_ablation.md", outdir / "tab4_ablation.tex"]


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
    written += table4_ablation(args.results, outdir)
    written += table5_fluent(args.results, outdir)

    print(f"\n{len(written)} table file(s) written to {outdir}:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
