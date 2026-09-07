"""fluent/make_offset.py -- define the 6 AirfRANS-replica offset cases (D-3).

The solver-offset study (docs/plans/PLAN_FLUENT_POST.md section 3) reruns six frozen
AirfRANS *test*-split simulations in Fluent so the solver offset
``Delta = CD_fluent - CD_airfrans`` can be measured on cases that carry an
independent OpenFOAM truth. The six sims are FROZEN in ``OFFSET_SIMS`` below;
they are not re-drawn after seeing results.

This script:
  1. reads each sim's cached ``.npz`` for U_inf, nu, AoA and the AirfRANS truth,
     and parses its continuous NACA parameters from the sim name;
  2. runs the geometry-validation GATE: a dense contour from the same parameters
     vs the cached ``surf_pos`` (994 pts, unordered), max nearest-neighbour
     distance must be < 5e-4 c, else the parameter convention is wrong -- written
     to ``results/fluent/offset_geometry_check.json``;
  3. appends the six cases to ``fluent/cases_to_run.json`` (per-case ``mu`` and
     ``re`` so the Fluent freestream reproduces the sim's U exactly), preserving
     everything already there;
  4. writes ``fluent/offset_cases.txt`` (the case-id list run_batch --only-file
     consumes).

Usage
-----
    .venv/Scripts/python.exe fluent/make_offset.py            # gate + write manifest
    .venv/Scripts/python.exe fluent/make_offset.py --gate-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PROCESSED = REPO / "data" / "processed" / "airfrans"
CASES_JSON = HERE / "cases_to_run.json"
OUT_DIR = REPO / "results" / "fluent"

# Frozen replica list (PLAN_FLUENT_POST section 3). Each: (case_id, sim_name, rule).
OFFSET_SIMS = [
    ("offset_1_naca0016_a2p4", "airFoil2D_SST_60.541_2.401_0.081_0.0_16.295",
     "near-symmetric 4-digit, low alpha, mid Re"),
    ("offset_2_naca0109_a8p6", "airFoil2D_SST_62.607_8.578_0.875_0.0_8.812",
     "near-symmetric 4-digit, higher alpha, mid Re"),
    ("offset_3_naca2211_a3p3", "airFoil2D_SST_71.226_3.333_2.424_2.411_10.928",
     "cambered 4-digit, mid alpha, mid Re"),
    ("offset_4_naca2710_am1p6", "airFoil2D_SST_31.863_-1.58_1.567_6.956_10.028",
     "4-digit, low-Re edge"),
    ("offset_5_naca0010_a3p8", "airFoil2D_SST_93.213_3.79_0.418_0.0_9.665",
     "4-digit, high-Re edge"),
    ("offset_6_naca5_17_a0p9", "airFoil2D_SST_64.144_0.932_1.956_3.115_0.0_17.336",
     "5-digit (non-reflex), mid conditions"),
]

RHO = 1.184
GATE_TOL = 5.0e-4        # max nearest-neighbour distance, chords
GATE_NPTS = 20000        # dense reference contour so NN measures shape, not spacing


def parse_sim_params(sim: str):
    """(U_inf, aoa_deg, [M,P,T] or [L,P,Q,T]) from an AirfRANS sim name."""
    toks = sim.replace("airFoil2D_SST_", "").split("_")
    u_inf = float(toks[0])
    aoa = float(toks[1])
    rest = [float(t) for t in toks[2:]]
    return u_inf, aoa, rest


def geometry_gate() -> dict:
    """Densified contour vs cached surf_pos for each sim; max NN distance."""
    from scipy.spatial import cKDTree
    sys.path.insert(0, str(HERE))
    from mesh_gen import airfoil_loop

    results = {}
    all_ok = True
    for case_id, sim, _rule in OFFSET_SIMS:
        npz = PROCESSED / f"{sim}.npz"
        d = np.load(npz, allow_pickle=True)
        surf = d["surf_pos"].astype(np.float64)
        _u, _a, params = parse_sim_params(sim)
        loop = airfoil_loop("params", GATE_NPTS, 1.0, 0.0, naca_params=params)
        tree = cKDTree(loop)
        dd, _ = tree.query(surf)
        ok = bool(dd.max() < GATE_TOL)
        all_ok = all_ok and ok
        results[case_id] = dict(sim=sim, naca_params=params,
                                max_nn=float(dd.max()), mean_nn=float(dd.mean()),
                                tol=GATE_TOL, ok=ok, n_surf=int(surf.shape[0]))
    results["_all_ok"] = all_ok
    return results


def build_cases() -> list[dict]:
    cases = []
    for i, (case_id, sim, rule) in enumerate(OFFSET_SIMS, start=1):
        d = np.load(PROCESSED / f"{sim}.npz", allow_pickle=True)
        u_inf = float(d["u_inf_mag"])
        nu = float(d["nu"])
        aoa = float(d["aoa_deg"])
        _u, _a, params = parse_sim_params(sim)
        mu = RHO * nu
        re = u_inf / nu            # so u_inf = re*mu/rho reproduces the sim's U
        # display label
        if len(params) == 3:
            code = f"{round(params[0])}{round(params[1])}{round(params[2]):02d}"
        else:
            code = f"5_{round(params[-1]):02d}"
        cases.append(dict(
            case_id=case_id, naca_digits=code, naca_params=params,
            re=re, aoa_deg=aoa, mesh_level=2, mu=mu,
            models=["sa", "sst"], arm="offset",
            airfrans_sim=sim,
            cd_true=float(d["cd_true"]), cl_true=float(d["cl_true"]),
            note=(f"Offset replica {i} ({rule}). AirfRANS test sim {sim}; "
                  f"continuous NACA params {params}, U={u_inf:.3f} m/s, "
                  f"nu={nu:.6g}, Re={re:.4g}, AoA={aoa:.3f} deg. Per-case mu "
                  f"reproduces the sim freestream exactly. NOT hand-tuned."),
        ))
    return cases


def append_to_manifest(cases: list[dict]) -> None:
    raw = json.loads(CASES_JSON.read_text(encoding="utf-8"))
    existing = raw["cases"]
    ids = {c["case_id"] for c in existing}
    added = 0
    for c in cases:
        if c["case_id"] in ids:
            # replace in place (idempotent re-run)
            existing[:] = [e for e in existing if e["case_id"] != c["case_id"]]
        existing.append(c)
        added += 1
    raw["cases"] = existing
    CASES_JSON.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    print(f"[offset] appended/updated {added} offset cases in {CASES_JSON}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-only", action="store_true",
                    help="run the geometry gate and write the check json; do not touch the manifest")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gate = geometry_gate()
    (OUT_DIR / "offset_geometry_check.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    for case_id, sim, _rule in OFFSET_SIMS:
        r = gate[case_id]
        print(f"[offset] gate {case_id:26s} maxNN={r['max_nn']:.2e}  "
              f"{'OK' if r['ok'] else 'FAIL'}")
    if not gate["_all_ok"]:
        print("[offset] GEOMETRY GATE FAILED -- parameter convention wrong; not writing manifest")
        return 1
    print(f"[offset] geometry gate PASSED (all max NN < {GATE_TOL:g} c)")

    if args.gate_only:
        return 0

    cases = build_cases()
    append_to_manifest(cases)
    ids = [c["case_id"] for c in cases]
    (HERE / "offset_cases.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"[offset] wrote fluent/offset_cases.txt ({len(ids)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
