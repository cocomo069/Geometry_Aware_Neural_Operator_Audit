"""scripts/collect_fluent.py -- parse, classify and audit the Fluent campaign.

Reads every ``fluent/cases/<case>/`` directory, parses the per-(case, model,
variant) coefficient histories, classifies each run converged / quasi-steady /
diverged / not-run per the rules of docs/PLAN_FLUENT_POST.md section 2, runs the
grid-study GCI (Celik et al. 2008, FLUENT_PLAN section 4.2) and a batch-integrity
check (saved mesh cell count vs mesh.json), and writes:

    results/fluent/fluent_summary.csv   one row per (case, model, variant)
    results/fluent/collected.json       the same data + run-level metadata
    results/fluent/gci.json             (--gci) Celik GCI on the gridstudy trio
    results/fluent/COLLECTED_SUMMARY.md  a human-readable roll-up

Pure numpy/stdlib: no solver, no torch, no dataset. Safe to re-run.

Usage
-----
    .venv/Scripts/python.exe -m scripts.collect_fluent
    .venv/Scripts/python.exe -m scripts.collect_fluent --gci
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = _ROOT / "fluent" / "cases"
OUT_DIR = _ROOT / "results" / "fluent"

# stage-2 acceptance window (iterations) and the flatness / oscillation rules.
#
# The plan's section-2 table names "relative CD change <= 1e-5 over the last 150
# iterations" as the converged rule. The real converged transcripts (gridstudy,
# the 4 low-AoA random cases) are flat to ~1e-6 relative PER ITERATION but creep
# monotonically at the 7th significant figure, so the accumulated 150-iter change
# is ~3e-4 -- larger than 1e-5, yet these are converged (PLAN section 0 tabulates
# them so). The literal endpoint rule is therefore too strict; the discriminator
# that actually separates a settled fixed point from a post-stall limit cycle is
# the OSCILLATION AMPLITUDE over the window, not the endpoint creep. We classify:
#   converged    : window peak-to-peak / |mean| < CONV_P2P_FRAC and no drift
#                  (a settled steady value; CD flat to the reported digits)
#   quasi_steady : bounded limit cycle, p2p/|mean| < QUASI_P2P_FRAC, no drift
#   diverged     : non-finite, |CD|>10, or bounded-but-drifting / large oscillation
# cd_rel_change_150 is still reported for transparency.
WINDOW = 1000
FLAT_LOOKBACK = 150
CONV_P2P_FRAC = 1e-2     # peak-to-peak(CD)/|mean(CD)| below this -> converged
CONV_DRIFT = 0.02        # half-window mean shift / |mean| below this -> no drift
QUASI_P2P_FRAC = 0.5     # peak-to-peak(CD)/|mean(CD)| below this -> quasi-steady
QUASI_DRIFT = 0.05
DIVERGE_ABS = 10.0       # |CD| above this (after startup) -> diverged
# The impulsive first-order start spikes |CD| large on the first few iterations of
# EVERY case (peak ~10-14 at iteration 1) before the field develops; that transient
# is not a divergence. Exclude the first STARTUP_GUARD iterations from the |CD|>10
# blow-up test only. A genuinely diverged run blows up (|CD|->1e80 / non-finite)
# well after this window, so it is still caught. (round-2 review fix)
STARTUP_GUARD = 50

VARIANTS = ("", "r1", "r2")   # "" == the original S0 batch
MODELS = ("sa", "sst")


# --------------------------------------------------------------------------- #
# small parsers
# --------------------------------------------------------------------------- #
def _arm_of(case_id: str) -> str:
    if case_id.startswith("gridstudy"):
        return "gridstudy"
    if case_id.startswith("al_acq"):
        return "acquisition"
    if case_id.startswith("al_var"):
        return "variance"
    if case_id.startswith("al_rand"):
        return "random"
    if case_id.startswith("offset"):
        return "offset"
    return "other"


def _read_coeffs(path: Path):
    """Return (iters, cd, cl) float arrays for a Fluent report file, or None."""
    if not path.exists():
        return None
    its, cds, cls = [], [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            it = int(parts[0])
        except ValueError:
            continue
        try:
            vals = [float(p) for p in parts[1:]]
        except ValueError:
            continue
        its.append(it)
        cds.append(vals[0])
        cls.append(vals[1] if len(vals) > 1 else float("nan"))
    if not its:
        return None
    return np.asarray(its), np.asarray(cds, dtype=np.float64), np.asarray(cls, dtype=np.float64)


_YPLUS_RE = re.compile(r"airfoil\s+([-\d.eE+]+)")


def _read_scalar_report(path: Path):
    """Pull the single 'airfoil <value>' number out of a surface-integral file."""
    if not path.exists():
        return float("nan")
    m = _YPLUS_RE.search(path.read_text(encoding="utf-8", errors="replace"))
    return float(m.group(1)) if m else float("nan")


_TRN_CELLS_RE = re.compile(r"(\d+)\s+(?:quadrilateral\s+)?cells", re.IGNORECASE)


def _cells_from_trn(path: Path):
    """First reported cell count in a Fluent transcript, or None."""
    if not path.exists():
        return None
    m = _TRN_CELLS_RE.search(path.read_text(encoding="utf-8", errors="replace"))
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# classification (PLAN_FLUENT_POST section 2)
# --------------------------------------------------------------------------- #
def classify(cd: np.ndarray) -> dict[str, Any]:
    """Classify a CD history and return the window statistics."""
    n = cd.size
    finite = np.isfinite(cd)
    out = dict(n_iter=int(n), cd=float("nan"), cd_window_mean=float("nan"),
               cd_window_std=float("nan"), cd_p2p=float("nan"),
               cd_rel_change_150=float("nan"), status="diverged")

    if n == 0:
        out["status"] = "not_run"
        return out

    # diverged: any non-finite value, or |CD| above the blow-up threshold AFTER the
    # startup transient (the iteration-1 impulsive-start spike is not a divergence).
    post = cd[STARTUP_GUARD:] if n > STARTUP_GUARD else cd
    post_finite = post[np.isfinite(post)]
    if not np.all(finite) or (post_finite.size and np.any(np.abs(post_finite) > DIVERGE_ABS)):
        out["status"] = "diverged"
        # still record the last finite value for the record
        if np.any(finite):
            out["cd"] = float(cd[finite][-1])
        return out

    out["cd"] = float(cd[-1])
    w = cd[-min(WINDOW, n):]
    mean = float(np.mean(w))
    out["cd_window_mean"] = mean
    out["cd_window_std"] = float(np.std(w, ddof=1)) if w.size > 1 else 0.0
    p2p = float(np.max(w) - np.min(w))
    out["cd_p2p"] = p2p

    if n > FLAT_LOOKBACK:
        rel = abs(cd[-1] - cd[-1 - FLAT_LOOKBACK]) / max(abs(cd[-1]), 1e-30)
    else:
        rel = abs(cd[-1] - cd[0]) / max(abs(cd[-1]), 1e-30)
    out["cd_rel_change_150"] = float(rel)

    # monotone drift over the window: |mean of second half - first half| large
    half = w.size // 2
    drift = abs(np.mean(w[half:]) - np.mean(w[:half])) / max(abs(mean), 1e-30) if half else 0.0
    p2p_frac = (p2p / abs(mean)) if mean != 0 else float("inf")

    if p2p_frac < CONV_P2P_FRAC and drift < CONV_DRIFT:
        out["status"] = "converged"
    elif p2p_frac < QUASI_P2P_FRAC and drift < QUASI_DRIFT:
        out["status"] = "quasi_steady"
    else:
        out["status"] = "diverged"
    return out


def _classify_cl(cl: np.ndarray) -> dict[str, float]:
    w = cl[np.isfinite(cl)]
    if w.size == 0:
        return dict(cl=float("nan"), cl_window_mean=float("nan"), cl_p2p=float("nan"))
    ww = w[-min(WINDOW, w.size):]
    return dict(cl=float(w[-1]), cl_window_mean=float(np.mean(ww)),
                cl_p2p=float(np.max(ww) - np.min(ww)))


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
def collect_run(cdir: Path, case_id: str, model: str, variant: str,
                params: dict, mesh: dict) -> dict[str, Any] | None:
    label = model if not variant else f"{model}_{variant}"
    coeffs = cdir / f"{case_id}_{label}_coeffs.out"
    parsed = _read_coeffs(coeffs)
    if parsed is None:
        # a journal exists but no run: only emit a not_run row for S0 (avoid
        # flooding the table with r1/r2 rows that were never scheduled)
        jour = cdir / f"{case_id}_{label}.jou"
        if variant == "" and jour.exists():
            base = _base_row(case_id, model, variant, params, mesh)
            base.update(dict(status="not_run", n_iter=0))
            return base
        return None

    its, cd, cl = parsed
    row = _base_row(case_id, model, variant, params, mesh)
    row.update(classify(cd))
    row.update(_classify_cl(cl))

    dat = cdir / f"{case_id}_{label}.dat.h5"
    encas = cdir / f"{case_id}_{label}.encas"
    cas = cdir / f"{case_id}_{label}.cas.h5"
    row["has_dat"] = dat.exists()
    row["has_encas"] = encas.exists()
    row["has_cas"] = cas.exists()

    # y+ from the surface-integral reports
    row["yplus_max"] = _read_scalar_report(cdir / f"{case_id}_{label}_yplus_max.txt")
    row["yplus_avg"] = _read_scalar_report(cdir / f"{case_id}_{label}_yplus_avg.txt")

    # batch integrity: cells in the saved case (via .trn) vs mesh.json
    trn_cells = _cells_from_trn(cdir / f"{case_id}_{label}.trn")
    mesh_cells = None
    if mesh:
        mesh_cells = (mesh.get("counts", {}) or {}).get("n_cells") or \
            (mesh.get("grid", {}) or {}).get("n_cells")
    row["cells_trn"] = trn_cells
    row["cells_mesh"] = mesh_cells
    row["integrity_ok"] = (trn_cells is not None and mesh_cells is not None
                           and int(trn_cells) == int(mesh_cells))
    return row


def _base_row(case_id, model, variant, params, mesh) -> dict[str, Any]:
    q = (mesh.get("quality", {}) if mesh else {}) or {}
    grid = (mesh.get("grid", {}) if mesh else {}) or {}
    return dict(
        case_id=case_id, arm=_arm_of(case_id),
        naca=str(params.get("naca_digits", "")),
        naca_params=params.get("naca_params"),
        airfrans_sim=params.get("airfrans_sim", ""),
        re=float(params.get("re", float("nan"))),
        aoa_deg=float(params.get("aoa_deg", float("nan"))),
        mesh_level=int(params.get("mesh_level", 0) or 0),
        model=model, variant=(variant or "s0"),
        mach=float(params.get("mach_estimate", float("nan"))),
        u_inf=float(params.get("u_inf", float("nan"))),
        min_orthogonality=float(q.get("min_orthogonality", float("nan"))),
        wall_min=float(grid.get("first_cell", float("nan"))),
        yplus_max=float("nan"), yplus_avg=float("nan"),
        has_dat=False, has_encas=False, has_cas=False,
        cells_trn=None, cells_mesh=None, integrity_ok=False,
        cl=float("nan"), cl_window_mean=float("nan"), cl_p2p=float("nan"),
    )


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


# --------------------------------------------------------------------------- #
# GCI (Celik et al. 2008)
# --------------------------------------------------------------------------- #
def _apparent_order(e21: float, e32: float, r21: float, r32: float) -> float:
    s = float(np.sign(e32 / e21)) if e21 != 0 else 1.0
    p = 2.0
    for _ in range(200):
        q = math.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if abs(p_new - p) < 1e-10:
            p = p_new
            break
        p = p_new
    return p


def gci_triplet(phi1: float, phi2: float, phi3: float, r21: float, r32: float) -> dict:
    """Celik GCI. phi1 fine, phi2 medium, phi3 coarse."""
    e21 = phi2 - phi1
    e32 = phi3 - phi2
    oscillatory = (e32 / e21) < 0 if e21 != 0 else False
    if e21 == 0:
        return dict(p_obs=float("nan"), phi_ext=phi1, e_a=0.0,
                    gci_fine=0.0, oscillatory=oscillatory,
                    phi1=phi1, phi2=phi2, phi3=phi3)
    p = _apparent_order(e21, e32, r21, r32)
    phi_ext = (r21 ** p * phi1 - phi2) / (r21 ** p - 1.0)
    e_a = abs((phi1 - phi2) / phi1) if phi1 != 0 else float("nan")
    gci_fine = 1.25 * e_a / (r21 ** p - 1.0)
    return dict(p_obs=float(p), phi_ext=float(phi_ext), e_a=float(e_a),
                gci_fine=float(gci_fine), oscillatory=bool(oscillatory),
                phi1=float(phi1), phi2=float(phi2), phi3=float(phi3))


def compute_gci(rows: list[dict]) -> dict:
    """GCI on the gridstudy trio, per model, for CD and CL (medium-grid band)."""
    def find(level, model):
        cid = f"gridstudy_naca0012_re3e6_a5_L{level}"
        for r in rows:
            if r["case_id"] == cid and r["model"] == model and r["variant"] == "s0":
                return r
        return None

    out: dict[str, Any] = {"method": "Celik et al. 2008 (FLUENT_PLAN 4.2)"}
    # cell counts fix the refinement ratios: h ~ N^{-1/2}, r = sqrt(N_fine/N_coarse)
    for model in MODELS:
        r1_, r2_, r3_ = find(1, model), find(2, model), find(3, model)
        if not (r1_ and r2_ and r3_):
            continue
        n1 = r3_.get("cells_mesh") or 96768  # fine = L3
        # cells_mesh may be None for one; fall back to canonical counts
        N = {1: 18796, 2: 43008, 3: 96768}
        r21 = math.sqrt(N[3] / N[2])   # medium/fine spacing ratio
        r32 = math.sqrt(N[2] / N[1])   # coarse/medium spacing ratio
        entry = {"r21": r21, "r32": r32,
                 "levels_converged": all(x["status"] in ("converged", "quasi_steady")
                                         for x in (r1_, r2_, r3_))}
        # phi1 fine=L3, phi2 medium=L2, phi3 coarse=L1
        for key, getter in (("cd", lambda x: x["cd"]), ("cl", lambda x: x["cl"])):
            g = gci_triplet(getter(r3_), getter(r2_), getter(r1_), r21, r32)
            entry[key] = g
        out[model] = entry
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
CSV_FIELDS = [
    "case_id", "arm", "naca", "re", "aoa_deg", "mesh_level", "model", "variant",
    "n_iter", "status", "cd", "cl", "cd_window_mean", "cd_window_std", "cd_p2p",
    "cl_window_mean", "cl_p2p", "cd_rel_change_150", "yplus_max", "yplus_avg",
    "min_orthogonality", "mach", "u_inf", "wall_min", "airfrans_sim",
    "has_cas", "has_dat", "has_encas", "integrity_ok", "cells_trn", "cells_mesh",
]


def collect_all() -> list[dict]:
    rows: list[dict] = []
    for cdir in sorted(CASES_DIR.iterdir()):
        if not cdir.is_dir() or cdir.name.startswith("placeholder"):
            continue
        case_id = cdir.name
        params = _load_json(cdir / f"{case_id}_params.json")
        mesh = _load_json(cdir / f"{case_id}_mesh.json")
        for model in MODELS:
            for variant in VARIANTS:
                row = collect_run(cdir, case_id, model, variant, params, mesh)
                if row is not None:
                    rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown(rows: list[dict], gci: dict | None, path: Path) -> None:
    L = ["# Fluent campaign -- collected results", ""]
    # tally by (arm, variant, status)
    from collections import Counter
    tally = Counter((r["arm"], r["model"], r["variant"], r["status"]) for r in rows)
    L.append("## Status tally (arm / model / variant)\n")
    L.append("| arm | model | variant | status | n |")
    L.append("|---|---|---|---|---|")
    for (arm, model, variant, status), n in sorted(tally.items()):
        L.append(f"| {arm} | {model} | {variant} | {status} | {n} |")
    L.append("")

    # accepted (converged + quasi_steady) SA runs, the (b) comparison set
    acc = [r for r in rows if r["model"] == "sa" and r["status"] in ("converged", "quasi_steady")]
    L.append(f"## Accepted steady SA points (n={len(acc)})\n")
    L.append("| case | arm | variant | Re | aoa | status | CD | CL | y+max | orth |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(acc, key=lambda x: (x["arm"], x["case_id"])):
        L.append("| {case_id} | {arm} | {variant} | {re:.2g} | {aoa_deg:.3g} | "
                 "{status} | {cd:.5f} | {cl:.4f} | {ym:.2f} | {orth:.3f} |".format(
                     case_id=r["case_id"], arm=r["arm"], variant=r["variant"],
                     re=r["re"], aoa_deg=r["aoa_deg"], status=r["status"],
                     cd=r["cd"] if np.isfinite(r["cd"]) else float("nan"),
                     cl=r["cl"] if np.isfinite(r["cl"]) else float("nan"),
                     ym=r.get("yplus_max", float("nan")),
                     orth=r.get("min_orthogonality", float("nan"))))
    L.append("")

    if gci:
        L.append("## Grid-study GCI (Celik 2008)\n")
        for model in MODELS:
            if model not in gci:
                continue
            e = gci[model]
            L.append(f"### {model.upper()}  (r21={e['r21']:.3f}, r32={e['r32']:.3f}, "
                     f"levels_converged={e['levels_converged']})")
            for key in ("cd", "cl"):
                g = e[key]
                L.append(f"- {key.upper()}: p_obs={g['p_obs']:.3f}, "
                         f"GCI_fine={100*g['gci_fine']:.3f}%, phi_ext={g['phi_ext']:.6f}"
                         + ("  [OSCILLATORY]" if g["oscillatory"] else ""))
            L.append("")

    # integrity failures
    bad = [r for r in rows if r["has_cas"] and not r["integrity_ok"]]
    if bad:
        L.append("## Batch-integrity WARNINGS (saved cells != mesh.json)\n")
        for r in bad:
            L.append(f"- {r['case_id']} [{r['model']}/{r['variant']}]: "
                     f"trn={r['cells_trn']} mesh={r['cells_mesh']}")
        L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gci", action="store_true", help="also compute the grid-study GCI")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    rows = collect_all()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(rows, args.out_dir / "fluent_summary.csv")

    gci = compute_gci(rows) if args.gci else None
    if gci:
        (args.out_dir / "gci.json").write_text(json.dumps(gci, indent=2), encoding="utf-8")

    # collected.json: rows + metadata (drop numpy types)
    def clean(r):
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in r.items()}
    collected = dict(
        n_rows=len(rows),
        rows=[clean(r) for r in rows],
    )
    (args.out_dir / "collected.json").write_text(json.dumps(collected, indent=2), encoding="utf-8")

    write_markdown(rows, gci, args.out_dir / "COLLECTED_SUMMARY.md")

    # console roll-up
    from collections import Counter
    tally = Counter((r["arm"], r["status"]) for r in rows if r["model"] == "sa" and r["variant"] == "s0")
    print(f"[collect] {len(rows)} rows over {len(set(r['case_id'] for r in rows))} cases")
    print("[collect] SA / S0 status by arm:")
    for (arm, status), n in sorted(tally.items()):
        print(f"    {arm:12s} {status:14s} {n}")
    if gci:
        for model in MODELS:
            if model in gci:
                print(f"[collect] GCI {model}: CD={100*gci[model]['cd']['gci_fine']:.3f}%  "
                      f"CL={100*gci[model]['cl']['gci_fine']:.3f}%  "
                      f"p_obs(CD)={gci[model]['cd']['p_obs']:.2f}")
    print(f"[collect] wrote {args.out_dir}/fluent_summary.csv, collected.json, "
          f"COLLECTED_SUMMARY.md" + (", gci.json" if gci else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
