"""scripts/compare_fluent.py -- surrogate vs Fluent on the accepted cases (C4).

Runs the trained surrogate ensembles (transolver / sdf_fno / gnn) at the
geometry + condition of every Fluent case and compares the predicted CD/CL
against the independent Fluent solution, offset-corrected by the AirfRANS-replica
solver offset. Reuses the *exact* inference path of scripts/run_active
(``load_ensemble_checkpoints`` + ``score_pool`` + the frozen force integrator) so
no second prediction path exists.

Outputs (results/fluent/):
  offset.csv                        one row per replica x model (Delta_CD, Delta_CL)
  offset_summary.json               mean +/- s offset per model
  surrogate_vs_fluent.csv           one row per (case, model): predictions, errors,
                                    conformal-band coverage
  surrogate_vs_fluent_summary.json  per-model MAE / max|err| / coverage by arm

Freestream convention (PLAN section 4.2): the surrogate is fed the EXACT Fluent
velocity (params.json ``u_inf``) via ``score_pool(speeds=...)``; the pool's nu
default is not allowed to recompute it.

Offset replicas: the surrogate already ran on the AirfRANS geometry, so its
prediction is read from ``results/<model>_full_s*/per_sim.csv`` (mean/std over
seeds) rather than re-meshed -- ``pred_source`` records which path each row used.

Conformal coverage: the calibrated 90% coefficient interval width is read from
``results/uq/<model>_full_k*.json`` (``coef.<target>.0.9.width_mean``). That file
stores the aggregate interval width, not a serialized CalibratedIntervals /
q_hat; width_mean == width_median there (the matched-mode band is effectively
constant across the test sims), so the 90% half-width is width_mean/2 and a case
is covered when |err| <= half-width. This is the honest use of what the
calibration study actually persisted.

Usage
-----
    .venv/Scripts/python.exe -m scripts.compare_fluent               # all models + offset
    .venv/Scripts/python.exe -m scripts.compare_fluent --offset-only
    .venv/Scripts/python.exe -m scripts.compare_fluent --models transolver
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RESULTS = _ROOT / "results"
OUT_DIR = RESULTS / "fluent"
CASES_DIR = _ROOT / "fluent" / "cases"
PROCESSED = _ROOT / "data" / "processed" / "airfrans"

# ensemble seeds per model (gnn ensemble is K=3, others K=5; see D-019)
MODEL_SEEDS = {
    "transolver": [0, 1, 2, 3, 4],
    "sdf_fno": [0, 1, 2, 3, 4],
    "gnn": [0, 1, 2],
}
MODEL_K_JSON = {
    "transolver": "transolver_full_k5.json",
    "sdf_fno": "sdf_fno_full_k5.json",
    "gnn": "gnn_full_k3.json",
}
ACCEPTED = ("converged", "quasi_steady")


def _is_accepted(row: dict) -> bool:
    """A usable steady point: (quasi-)steady classification AND the .cas.h5 was
    written (the journal saves only at the end), so a still-running or crashed
    solve with a transiently-flat CD is never mistaken for a converged one."""
    return row.get("status") in ACCEPTED and str(row.get("has_cas", "")).lower() == "true"


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def load_summary() -> list[dict]:
    path = OUT_DIR / "fluent_summary.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run scripts/collect_fluent first")
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def sa_rows(summary: list[dict]) -> list[dict]:
    """One SA row per case (the like-for-like model vs AirfRANS), any variant that
    reached the best status. Prefer r1 over s0 when r1 rescued the case."""
    by_case: dict[str, dict] = {}
    rank = {"converged": 3, "quasi_steady": 2, "diverged": 1, "not_run": 0}
    for r in summary:
        if r["model"] != "sa":
            continue
        cid = r["case_id"]
        cur = by_case.get(cid)
        if cur is None or rank.get(r["status"], 0) > rank.get(cur["status"], 0):
            by_case[cid] = r
    return list(by_case.values())


# --------------------------------------------------------------------------- #
# offset study (section 3)
# --------------------------------------------------------------------------- #
def compute_offset(summary: list[dict]) -> dict:
    """Delta = CD_fluent - CD_airfrans on the accepted replicas, per model."""
    rows_out = []
    per_model: dict[str, dict[str, list]] = {"sa": {"dcd": [], "dcl": []},
                                             "sst": {"dcd": [], "dcl": []}}
    for r in summary:
        if r["arm"] != "offset" or not _is_accepted(r):
            continue
        sim = r["airfrans_sim"]
        if not sim:
            continue
        d = np.load(PROCESSED / f"{sim}.npz", allow_pickle=True)
        cd_af, cl_af = float(d["cd_true"]), float(d["cl_true"])
        cd_fl, cl_fl = _f(r["cd"]), _f(r["cl"])
        model = r["model"]
        dcd, dcl = cd_fl - cd_af, cl_fl - cl_af
        rows_out.append(dict(
            case_id=r["case_id"], sim=sim, model=model,
            cd_airfrans=cd_af, cl_airfrans=cl_af, cd_fluent=cd_fl, cl_fluent=cl_fl,
            delta_cd=dcd, delta_cl=dcl,
            reldelta_cd=dcd / cd_af if cd_af else float("nan"),
            reldelta_cl=dcl / abs(cl_af) if cl_af else float("nan"),
        ))
        if model in per_model:
            per_model[model]["dcd"].append(dcd)
            per_model[model]["dcl"].append(dcl)

    summary_out = {}
    for model, dd in per_model.items():
        if dd["dcd"]:
            summary_out[model] = dict(
                n=len(dd["dcd"]),
                dbar_cd=float(np.mean(dd["dcd"])), s_cd=float(np.std(dd["dcd"], ddof=1) if len(dd["dcd"]) > 1 else 0.0),
                dbar_cl=float(np.mean(dd["dcl"])), s_cl=float(np.std(dd["dcl"], ddof=1) if len(dd["dcl"]) > 1 else 0.0),
            )
    return {"rows": rows_out, "summary": summary_out}


def write_offset(offset: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "sim", "model", "cd_airfrans", "cl_airfrans",
              "cd_fluent", "cl_fluent", "delta_cd", "delta_cl",
              "reldelta_cd", "reldelta_cl"]
    with (OUT_DIR / "offset.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in offset["rows"]:
            w.writerow(r)
    (OUT_DIR / "offset_summary.json").write_text(json.dumps(offset["summary"], indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# conformal half-widths (from the calibration study)
# --------------------------------------------------------------------------- #
def conformal_halfwidths(model: str, level: str = "0.9") -> dict[str, float]:
    path = RESULTS / "uq" / MODEL_K_JSON[model]
    d = json.loads(path.read_text(encoding="utf-8"))
    coef = d["coef"]
    out = {}
    for tgt in ("cd_int", "cd_head", "cl_int", "cl_head"):
        out[tgt] = 0.5 * float(coef[tgt][level]["width_mean"])
    return out


# --------------------------------------------------------------------------- #
# surrogate predictions
# --------------------------------------------------------------------------- #
def _replica_prediction(model: str, sim: str) -> dict[str, float]:
    """Mean/std over seeds of the per-sim surrogate prediction for a replica."""
    cd_i, cl_i, cd_h, cl_h = [], [], [], []
    for s in MODEL_SEEDS[model]:
        p = RESULTS / f"{model}_full_s{s}" / "per_sim.csv"
        if not p.exists():
            continue
        for row in csv.DictReader(p.open(encoding="utf-8")):
            if row["sim_name"] == sim:
                cd_i.append(_f(row["cd_int"])); cl_i.append(_f(row["cl_int"]))
                cd_h.append(_f(row["cd_head"])); cl_h.append(_f(row["cl_head"]))
                break
    if not cd_i:
        return {}
    def ms(a):
        a = np.asarray(a)
        return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0)
    cdim, cdis = ms(cd_i); clim, clis = ms(cl_i); cdhm, cdhs = ms(cd_h); clhm, clhs = ms(cl_h)
    return dict(cd_int_mean=cdim, cd_int_std=cdis, cl_int_mean=clim, cl_int_std=clis,
                cd_head_mean=cdhm, cd_head_std=cdhs, cl_head_mean=clhm, cl_head_std=clhs)


def _score_pool_cases(model: str, cases: list[dict]) -> dict[str, dict]:
    """Score the non-replica cases with the model's ensemble; keyed by case_id."""
    import torch
    from scripts.run_active import (_find_checkpoints, _load_config, _norm_stats,
                                    score_pool)
    from src.active.pool import PoolEntry, parse_naca
    from src.physics.force_integration import integrate_forces
    from src.uq.ensembles import load_ensemble_checkpoints

    seeds = MODEL_SEEDS[model]
    found = _find_checkpoints(model, "full", seeds, _ROOT / "checkpoints")
    if not found:
        raise SystemExit(f"no checkpoints for {model}_full")
    paths = [p for _, p in found]
    cfg = _load_config(model, "full", seeds, RESULTS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = load_ensemble_checkpoints(paths, cfg, map_location="cpu")
    for m in predictor.models:
        try:
            m.to(device)
        except Exception:  # noqa: BLE001
            pass
    stats = _norm_stats("full", PROCESSED, _ROOT / "data" / "splits")

    entries, speeds, ids = [], [], []
    for c in cases:
        entries.append(PoolEntry(parse_naca(c["naca"]), float(c["re"]), float(c["aoa_deg"])))
        speeds.append(float(c["u_inf"]))
        ids.append(c["case_id"])

    comp = score_pool(
        predictor, entries,
        integrate=integrate_forces,
        denorm=lambda n, x: stats.denormalize(n, x),
        normalize_cond=lambda x: stats.normalize("cond", x),
        device=device, batch_size=8, compute_sym=False, n_points=200,
        speeds=speeds, return_members=True, log=lambda *_: None,
    )
    out = {}
    for i, cid in enumerate(ids):
        out[cid] = dict(
            cd_int_mean=float(comp["cd_int"][i]),
            cd_int_std=float(comp["cd_int_members"][:, i].std(ddof=1)) if comp["cd_int_members"].shape[0] > 1 else 0.0,
            cd_head_mean=float(comp["cd_head"][i]),
            cd_head_std=float(comp["cd_head_members"][:, i].std(ddof=1)) if comp["cd_head_members"].shape[0] > 1 else 0.0,
            cl_int_mean=float(comp["cl_int"][i]),
            cl_int_std=float(comp["cl_int_members"][:, i].std(ddof=1)) if comp["cl_int_members"].shape[0] > 1 else 0.0,
            cl_head_mean=float(comp["cl_head"][i]),
            cl_head_std=float(comp["cl_head_members"][:, i].std(ddof=1)) if comp["cl_head_members"].shape[0] > 1 else 0.0,
        )
    return out


# --------------------------------------------------------------------------- #
# main comparison
# --------------------------------------------------------------------------- #
CSV_FIELDS = [
    "case_id", "arm", "model", "K", "naca", "re", "aoa_deg", "status",
    "cd_fluent", "cd_fluent_corrected", "cl_fluent", "cl_fluent_corrected",
    "cd_int_mean", "cd_int_std", "cd_head_mean", "cd_head_std",
    "cl_int_mean", "cl_int_std", "cl_head_mean", "cl_head_std",
    "err_cd_int", "err_cd_head", "err_cl_int", "err_cl_head",
    "fsc", "q90_halfwidth_cd_int", "covered90_cd_int", "covered90_cd_head",
    "pred_source",
]


def build_rows(model: str, summary: list[dict], offset: dict) -> list[dict]:
    rows = sa_rows(summary)
    # skip gridstudy L1/L3 (keep L2 as the one low-AoA grid point) -- L2 is the
    # production level and the (b) comparison point; L1/L3 are grid-study only.
    rows = [r for r in rows if not (r["case_id"].startswith("gridstudy")
                                    and not r["case_id"].endswith("_L2"))]
    dbar_cd = offset["summary"].get("sa", {}).get("dbar_cd", 0.0)
    dbar_cl = offset["summary"].get("sa", {}).get("dbar_cl", 0.0)
    half = conformal_halfwidths(model)
    K = len(MODEL_SEEDS[model])

    # split into replica (per_sim path) and non-replica (score_pool path)
    non_replica = [r for r in rows if r["arm"] != "offset"]
    preds_np = _score_pool_cases(model, non_replica)

    out = []
    for r in rows:
        cid = r["case_id"]
        status = r["status"]
        cd_fl, cl_fl = _f(r["cd"]), _f(r["cl"])
        if r["arm"] == "offset":
            pred = _replica_prediction(model, r["airfrans_sim"])
            src = "per_sim"
        else:
            pred = preds_np.get(cid, {})
            src = "score_pool"
        if not pred:
            continue

        # offset correction only where a Fluent steady value exists
        cd_corr = (cd_fl - dbar_cd) if math.isfinite(cd_fl) else float("nan")
        cl_corr = (cl_fl - dbar_cl) if math.isfinite(cl_fl) else float("nan")
        accepted = _is_accepted(r)
        err_cd_int = pred["cd_int_mean"] - cd_corr if accepted else float("nan")
        err_cd_head = pred["cd_head_mean"] - cd_corr if accepted else float("nan")
        err_cl_int = pred["cl_int_mean"] - cl_corr if accepted else float("nan")
        err_cl_head = pred["cl_head_mean"] - cl_corr if accepted else float("nan")
        cov_cd_int = (abs(err_cd_int) <= half["cd_int"]) if math.isfinite(err_cd_int) else None
        cov_cd_head = (abs(err_cd_head) <= half["cd_head"]) if math.isfinite(err_cd_head) else None

        out.append(dict(
            case_id=cid, arm=r["arm"], model=model, K=K, naca=r["naca"],
            re=_f(r["re"]), aoa_deg=_f(r["aoa_deg"]), status=status,
            cd_fluent=cd_fl, cd_fluent_corrected=cd_corr,
            cl_fluent=cl_fl, cl_fluent_corrected=cl_corr,
            cd_int_mean=pred["cd_int_mean"], cd_int_std=pred["cd_int_std"],
            cd_head_mean=pred["cd_head_mean"], cd_head_std=pred["cd_head_std"],
            cl_int_mean=pred["cl_int_mean"], cl_int_std=pred["cl_int_std"],
            cl_head_mean=pred["cl_head_mean"], cl_head_std=pred["cl_head_std"],
            err_cd_int=err_cd_int, err_cd_head=err_cd_head,
            err_cl_int=err_cl_int, err_cl_head=err_cl_head,
            fsc=abs(pred["cd_int_mean"] - pred["cd_head_mean"]),
            q90_halfwidth_cd_int=half["cd_int"],
            covered90_cd_int=cov_cd_int, covered90_cd_head=cov_cd_head,
            pred_source=src,
        ))
    return out


def summarize(all_rows: list[dict]) -> dict:
    out = {}
    for model in sorted(set(r["model"] for r in all_rows)):
        mrows = [r for r in all_rows if r["model"] == model]
        acc = [r for r in mrows if r["status"] in ACCEPTED]
        def mae(key):
            v = [abs(r[key]) for r in acc if math.isfinite(r[key])]
            return float(np.mean(v)) if v else None
        def mx(key):
            v = [abs(r[key]) for r in acc if math.isfinite(r[key])]
            return float(np.max(v)) if v else None
        # in-distribution yardstick
        yard = {}
        mp = RESULTS / f"{model}_full_s0" / "metrics.json"
        if mp.exists():
            md = json.loads(mp.read_text(encoding="utf-8")).get("coef", {})
            yard = {"cd_int_mae": md.get("cd_int_mae"), "cd_head_mae": md.get("cd_head_mae")}
        # coverage
        cov_int = [r["covered90_cd_int"] for r in acc if r["covered90_cd_int"] is not None]
        cov_head = [r["covered90_cd_head"] for r in acc if r["covered90_cd_head"] is not None]
        by_arm = {}
        for arm in sorted(set(r["arm"] for r in acc)):
            arows = [r for r in acc if r["arm"] == arm]
            vh = [abs(r["err_cd_head"]) for r in arows if math.isfinite(r["err_cd_head"])]
            vi = [abs(r["err_cd_int"]) for r in arows if math.isfinite(r["err_cd_int"])]
            by_arm[arm] = dict(n=len(arows),
                               mae_cd_head=float(np.mean(vh)) if vh else None,
                               mae_cd_int=float(np.mean(vi)) if vi else None)
        out[model] = dict(
            n_accepted=len(acc), n_total=len(mrows),
            # cd_head is the PRIMARY surrogate CD: the coefficient-head regression
            # is robust to the input representation, whereas the integrated cd_int
            # on the SYNTHETIC pool geometry (score_pool path, non-replica cases)
            # carries a large input-representation artifact (~5x on a checked
            # in-envelope shape) -- it is a physics-consistency / FSC diagnostic
            # there, not a CD prediction. On the offset replicas (per_sim path,
            # real AirfRANS mesh) cd_int is reliable.
            primary_metric="cd_head",
            mae_cd_head=mae("err_cd_head"), max_cd_head=mx("err_cd_head"),
            mae_cl_head=mae("err_cl_head"),
            mae_cd_int=mae("err_cd_int"), max_cd_int=mx("err_cd_int"),
            mae_cl_int=mae("err_cl_int"),
            coverage90_cd_head=float(np.mean(cov_head)) if cov_head else None,
            coverage90_cd_int=float(np.mean(cov_int)) if cov_int else None,
            by_arm=by_arm, indist_yardstick=yard,
        )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="transolver,sdf_fno,gnn",
                    help="comma list of surrogate models (default all three)")
    ap.add_argument("--offset-only", action="store_true",
                    help="compute the solver offset (offset.csv) and exit")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    summary = load_summary()
    offset = compute_offset(summary)
    write_offset(offset)
    print("[compare] offset per model:")
    for m, s in offset["summary"].items():
        print(f"    {m}: dbar_CD={s['dbar_cd']:+.5f} (s={s['s_cd']:.5f}), "
              f"dbar_CL={s['dbar_cl']:+.4f} (s={s['s_cl']:.4f}), n={s['n']}")
    if args.offset_only:
        return 0

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_rows: list[dict] = []
    for model in models:
        print(f"[compare] scoring {model} ...")
        all_rows += build_rows(model, summary, offset)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "surrogate_vs_fluent.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    summ = summarize(all_rows)
    (args.out_dir / "surrogate_vs_fluent_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")

    print("[compare] per-model MAE on accepted set (offset-corrected); cd_head is primary:")
    for model, s in summ.items():
        print(f"    {model:11s} n={s['n_accepted']:2d}  MAE_CD_head={s['mae_cd_head']}  "
              f"cov90_head={s['coverage90_cd_head']}  (indist CD_head MAE {s['indist_yardstick'].get('cd_head_mae')})")
    print(f"[compare] wrote {args.out_dir}/surrogate_vs_fluent.csv + summary.json + offset.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
