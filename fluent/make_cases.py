"""fluent/make_cases.py -- instantiate Fluent journals from a case manifest.

Reads `fluent/cases_to_run.json`, expands every case into
`fluent/cases/<case_id>/` containing:

    <case_id>_params.json     every derived quantity (U_inf, y+ sizing, ...)
    <case_id>_sa.jou          instantiated journal, Spalart-Allmaras
    <case_id>_sst.jou         instantiated journal, k-omega SST
    <case_id>_mesh.cmd        the exact mesh_gen.py command line for this case
    <case_id>.msh             only with --write-mesh

and at the top level `fluent/cases/MANIFEST.json` + `fluent/cases/RUNBOOK.md`
(the ordered list of orchestrator calls).

By default NO mesh is generated and nothing is solved -- this script only
writes text. Pass --write-mesh to also build the meshes (pure numpy, a few
seconds and a few MB each; still no solver).

Manifest schema
---------------
Top level is either a bare list of case objects, or an object with `defaults`
and `cases`. Each case:

    {
      "case_id":     "naca0012_re3e6_a5_L2",   # required, unique, filesystem-safe
      "naca_digits": "0012",                   # required, 4 or 5 digits
      "re":          3.0e6,                    # required, chord Reynolds number
      "aoa_deg":     5.0,                      # required
      "mesh_level":  2,                        # required, 1 coarse / 2 medium / 3 fine
      "models":      ["sa", "sst"],            # optional, default both
      "note":        "..."                     # optional, free text
    }

`defaults` may set chord, rho, mu, y_plus, r_far, x_out, iter_stage1,
iter_stage2, cd_stop_criterion, smooth_sweeps, models.

Usage
-----
    .venv/Scripts/python.exe fluent/make_cases.py
    .venv/Scripts/python.exe fluent/make_cases.py --write-mesh
    .venv/Scripts/python.exe fluent/make_cases.py --only naca0012_re3e6_a5_L2
"""

from __future__ import annotations

import argparse
import json
import re as _re
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TEMPLATE = HERE / "templates" / "case_template.jou"

# ---------------------------------------------------------------------------
# Defaults. Fluid properties are air at 298.15 K / 1 atm, which is what the
# AirfRANS generator states it used -- VERIFY against docs/DATA_NOTES.md once
# A1 has introspected the dataset, because the solver-offset study is
# meaningless if our freestream state differs from theirs.
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    chord=1.0,
    rho=1.184,
    mu=1.85e-5,
    y_plus=0.6,
    r_far=30.0,
    x_out=30.0,
    smooth_sweeps=0,
    iter_stage1=300,
    iter_stage2=4000,
    cd_stop_criterion=1e-5,
    models=["sa", "sst"],
)

# ---------------------------------------------------------------------------
# Per-turbulence-model journal blocks.
#
# `n_residuals` matters: /solve/monitors/residual/convergence-criteria takes
# the per-equation criteria POSITIONALLY, and the equation count differs
# between the two models (SA: continuity, x-vel, y-vel, nut = 4; SST adds k and
# omega = 5). Getting the count wrong leaves an unanswered prompt that eats the
# next journal lines.
#
# Every keyword below is flagged in templates/probe_bc_keywords.jou -- run that
# probe once and reconcile before trusting these.
# ---------------------------------------------------------------------------
MODELS = {
    "sa": dict(
        label="sa",
        pretty="Spalart-Allmaras",
        cmd="spalart-allmaras yes",
        n_residuals=4,     # continuity, x-velocity, y-velocity, nut (verified v211)
        # VERIFIED against Fluent 2021 R1 (v211), 2026-09-03: the velocity-inlet
        # turbulence key for SA is `turb-viscosity-ratio-profile`, and every
        # profile-capable field first answers "Use Profile? [no]" (the `no`).
        inlet_block="turb-viscosity-ratio-profile\nno\n3",
        scheme_first="/solve/set/discretization-scheme/nut 0",
        scheme_second="/solve/set/discretization-scheme/nut 1",
        urf_block=("/solve/set/under-relaxation/nut 0.7\n"
                   "/solve/set/under-relaxation/turb-viscosity 1"),
    ),
    "sst": dict(
        label="sst",
        pretty="k-omega SST",
        cmd="kw-sst yes",
        n_residuals=5,     # continuity, x-velocity, y-velocity, k, omega
        # NOT YET VERIFIED on a live v211 SST solve. The `no` (Use Profile?)
        # answers match the confirmed SA pattern; the SST velocity-inlet key
        # names (turb-intensity / turb-viscosity-ratio) must be confirmed with
        # probe_bc_keywords.jou before the first SST case is trusted.
        inlet_block="turb-intensity\nno\n0.1\nturb-viscosity-ratio\nno\n3",
        scheme_first=("/solve/set/discretization-scheme/k 0\n"
                      "/solve/set/discretization-scheme/omega 0"),
        scheme_second=("/solve/set/discretization-scheme/k 1\n"
                       "/solve/set/discretization-scheme/omega 1"),
        urf_block=("/solve/set/under-relaxation/k 0.7\n"
                   "/solve/set/under-relaxation/omega 0.7\n"
                   "/solve/set/under-relaxation/turb-viscosity 1"),
    ),
}

SAFE_ID = _re.compile(r"^[A-Za-z0-9_.-]+$")


def fluent_path(p):
    """Fluent's TUI wants forward slashes even on Windows."""
    return str(Path(p).resolve()).replace("\\", "/")


def load_manifest(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return dict(DEFAULTS), raw
    defaults = dict(DEFAULTS)
    defaults.update(raw.get("defaults", {}))
    return defaults, raw["cases"]


def validate(cases, defaults):
    """Fail loudly and early -- a bad manifest must not reach the solver."""
    seen = set()
    errs = []
    for i, c in enumerate(cases):
        where = "case[%d]" % i
        for k in ("case_id", "naca_digits", "re", "aoa_deg", "mesh_level"):
            if k not in c:
                errs.append("%s: missing required key '%s'" % (where, k))
        cid = c.get("case_id", "")
        if cid and not SAFE_ID.match(cid):
            errs.append("%s: case_id '%s' is not filesystem-safe" % (where, cid))
        if cid in seen:
            errs.append("%s: duplicate case_id '%s'" % (where, cid))
        seen.add(cid)
        d = str(c.get("naca_digits", ""))
        if len(d) not in (4, 5) or not d.isdigit():
            errs.append("%s: naca_digits '%s' must be 4 or 5 digits" % (where, d))
        if len(d) == 5 and d[2] == "1":
            errs.append("%s: NACA %s is a reflex 5-digit mean line, which "
                        "fluent/mesh_gen.py does not implement" % (where, d))
        if c.get("mesh_level") not in (1, 2, 3):
            errs.append("%s: mesh_level must be 1, 2 or 3" % where)
        for m in c.get("models", defaults["models"]):
            if m not in MODELS:
                errs.append("%s: unknown model '%s' (have %s)"
                            % (where, m, sorted(MODELS)))
        if not (0.0 < float(c.get("re", 0)) < 1e9):
            errs.append("%s: implausible Reynolds number %r" % (where, c.get("re")))
        if abs(float(c.get("aoa_deg", 0))) > 25.0:
            errs.append("%s: |aoa_deg| > 25 -- deeply stalled, steady RANS will "
                        "not converge meaningfully; remove or justify" % where)
    if errs:
        raise SystemExit("cases_to_run.json is invalid:\n  " + "\n  ".join(errs))


def derive(case, defaults):
    """All numbers the journal needs, computed once, recorded in params.json."""
    sys.path.insert(0, str(HERE))
    from mesh_gen import (GRID_LEVELS, LEVEL_YPLUS_SCALE, first_cell_height,
                          solve_growth_ratio)

    p = dict(defaults)
    p.update({k: v for k, v in case.items() if k != "models"})

    chord = float(p["chord"])
    rho, mu = float(p["rho"]), float(p["mu"])
    re = float(p["re"])
    level = int(p["mesh_level"])

    # U_inf from Re: the manifest specifies Re (matching how AirfRANS indexes
    # its sims); the solver needs a velocity.
    u_inf = re * mu / (rho * chord)
    yp = float(p["y_plus"]) * LEVEL_YPLUS_SCALE[level]
    bl = first_cell_height(u_inf, chord, rho, mu, yp)

    grid = dict(GRID_LEVELS[level])
    # Report the wall-normal growth ratio up front so an unreasonable mesh is
    # visible in params.json without generating it.
    growth = solve_growth_ratio(grid["nj"], bl["first_cell_height"],
                                float(p["r_far"]) * chord)

    return dict(
        case_id=p["case_id"], naca_digits=str(p["naca_digits"]),
        re=re, aoa_deg=float(p["aoa_deg"]), mesh_level=level,
        chord=chord, rho=rho, mu=mu, nu=mu / rho,
        u_inf=u_inf, mach_estimate=u_inf / 346.0,
        a_ref=chord * 1.0, x_moment=0.25 * chord,
        y_plus=float(p["y_plus"]), y_plus_target=yp, boundary_layer=bl,
        grid=dict(grid, n_cells=(2 * grid["nw"] + grid["na"]) * grid["nj"],
                  normal_growth=growth),
        r_far=float(p["r_far"]), x_out=float(p["x_out"]),
        smooth_sweeps=int(p["smooth_sweeps"]),
        iter_stage1=int(p["iter_stage1"]), iter_stage2=int(p["iter_stage2"]),
        cd_stop_criterion=float(p["cd_stop_criterion"]),
        models=list(case.get("models", defaults["models"])),
        note=case.get("note", ""),
    )


def instantiate(template, d, model, mesh_path, out_dir):
    tight = " ".join(["1e-6"] * model["n_residuals"])
    loose = " ".join(["1e-4"] * model["n_residuals"])
    slots = {
        "CASE_ID": d["case_id"],
        "TURB_LABEL": model["label"],
        "MESH_PATH": fluent_path(mesh_path),
        "OUT_DIR": fluent_path(out_dir),
        "RHO": "%.6g" % d["rho"],
        "MU": "%.6g" % d["mu"],
        "U_INF": "%.6f" % d["u_inf"],
        "U_X": "%.6f" % d["u_inf"],
        "U_Y": "0",
        "CHORD": "%.6g" % d["chord"],
        "A_REF": "%.6g" % d["a_ref"],
        "X_MOMENT": "%.6g" % d["x_moment"],
        "TURB_MODEL_CMD": model["cmd"],
        "TURB_INLET_BLOCK": model["inlet_block"],
        "TURB_SCHEME_BLOCK_FIRST": model["scheme_first"],
        "TURB_SCHEME_BLOCK_SECOND": model["scheme_second"],
        "TURB_URF_BLOCK": model["urf_block"],
        "RESIDUAL_CRITERIA": tight,
        "RESIDUAL_CRITERIA_LOOSE": loose,
        "ITER_STAGE1": str(d["iter_stage1"]),
        "ITER_STAGE2": str(d["iter_stage2"]),
        "CD_STOP_CRITERION": "%.3g" % d["cd_stop_criterion"],
    }
    text = template
    for k, v in slots.items():
        text = text.replace("{{" + k + "}}", str(v))
    left = _re.findall(r"\{\{([A-Z0-9_]+)\}\}", text)
    if left:
        raise SystemExit("%s/%s: unfilled slots %s"
                         % (d["case_id"], model["label"], sorted(set(left))))
    return text


def mesh_command(d, mesh_path, json_path):
    return (".venv/Scripts/python.exe fluent/mesh_gen.py"
            " --naca %s --re %g --aoa %g --level %d --chord %g"
            " --rho %g --mu %g --y-plus %g --r-far %g --x-out %g"
            " --smooth-sweeps %d --out %s --json %s"
            % (d["naca_digits"], d["re"], d["aoa_deg"], d["mesh_level"],
               d["chord"], d["rho"], d["mu"], d["y_plus"], d["r_far"],
               d["x_out"], d["smooth_sweeps"],
               mesh_path.relative_to(REPO).as_posix(),
               json_path.relative_to(REPO).as_posix()))


def write_runbook(out_root, rows, total_cells, n_jou):
    L = []
    L.append("# RUNBOOK -- generated by fluent/make_cases.py, do not edit by hand\n")
    L.append("Generated %s. %d cases, %d journals, %d cells summed over all runs.\n"
             % (date.today(), len(rows), n_jou, total_cells))
    L.append("Nothing here has been executed. Fluent runs are deferred by "
             "decision D-007 until the user green-lights CPU use.\n")
    L.append("## Step 0 -- probe the TUI keywords (once, ~1 min)\n")
    L.append("The journals were authored without a live Fluent. Substitute "
             "`{{MESH_PATH}}` / `{{OUT_DIR}}` in "
             "`fluent/templates/probe_bc_keywords.jou`, run it, and reconcile "
             "every grouped-`set` keyword and the residual-equation count against "
             "`fluent/templates/case_template.jou` before anything else. An "
             "unanswered prompt silently eats following journal lines; this step "
             "is what makes that impossible.\n")
    L.append("## Step 1 -- build the meshes (pure numpy, no solver)\n")
    L.append("```")
    L.append(".venv/Scripts/python.exe fluent/make_cases.py --write-mesh")
    L.append("```")
    L.append("Check each `*_mesh.json`: `min_jacobian` must be > 0 (a "
             "non-positive value means the grid folded -- never solve on it), "
             "`min_orthogonality` above ~0.2, `normal_growth` below ~1.2.\n")
    L.append("## Step 2 -- solve, one case at a time\n")
    L.append("One solver licence means one solve at a time. Run these in order "
             "via `mcp__ansys__fluent_run_journal` (asynchronous -- it returns a "
             "job_id; poll `ansys_job_status`), then "
             "`mcp__ansys__fluent_check_convergence` on each.\n")
    for r in rows:
        L.append("### `%s`  (NACA %s, Re %g, AoA %g deg, level %d, %d cells)"
                 % (r["case_id"], r["naca"], r["re"], r["aoa_deg"],
                    r["mesh_level"], r["n_cells"]))
        if r["note"]:
            L.append("\n%s\n" % r["note"])
        for j in r["journals"]:
            L.append("- `%s`" % j)
        L.append("")
    L.append("## Step 3 -- acceptance checks per case\n")
    L.append("Per `docs/FLUENT_PLAN.md` section 6. In short: max y+ < 1, "
             "residuals at or below the stated criteria AND CD flat to the stop "
             "criterion over the last 150 iterations, and the GCI on the 3-level "
             "family below the stated band. A converged residual with a drifting "
             "CD is not a converged case.\n")
    (out_root / "RUNBOOK.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Instantiate Fluent journals from fluent/cases_to_run.json")
    ap.add_argument("--manifest", type=Path, default=HERE / "cases_to_run.json")
    ap.add_argument("--out-dir", type=Path, default=HERE / "cases")
    ap.add_argument("--only", action="append", default=None,
                    help="restrict to these case_ids (repeatable)")
    ap.add_argument("--write-mesh", action="store_true",
                    help="also generate the .msh (pure numpy; still no solver)")
    args = ap.parse_args(argv)

    defaults, cases = load_manifest(args.manifest)
    validate(cases, defaults)
    if args.only:
        cases = [c for c in cases if c["case_id"] in set(args.only)]
        if not cases:
            raise SystemExit("--only matched no case_id in %s" % args.manifest)

    template = TEMPLATE.read_text(encoding="utf-8")
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows, total_cells, n_jou = [], 0, 0
    for case in cases:
        d = derive(case, defaults)
        cdir = out_root / d["case_id"]
        cdir.mkdir(parents=True, exist_ok=True)
        mesh_path = cdir / (d["case_id"] + ".msh")
        mjson = cdir / (d["case_id"] + "_mesh.json")

        (cdir / (d["case_id"] + "_params.json")).write_text(
            json.dumps(d, indent=2), encoding="utf-8")
        cmd = mesh_command(d, mesh_path, mjson)
        (cdir / (d["case_id"] + "_mesh.cmd")).write_text(cmd + "\n", encoding="utf-8")

        jous = []
        for m in d["models"]:
            model = MODELS[m]
            text = instantiate(template, d, model, mesh_path, cdir)
            jp = cdir / ("%s_%s.jou" % (d["case_id"], m))
            jp.write_text(text, encoding="utf-8")
            jous.append(jp.relative_to(REPO).as_posix())
            n_jou += 1

        if args.write_mesh:
            from mesh_gen import generate
            rec = generate(d["naca_digits"], d["re"], d["aoa_deg"], d["mesh_level"],
                           mesh_path, chord=d["chord"], rho=d["rho"], mu=d["mu"],
                           y_plus=d["y_plus"], r_far=d["r_far"], x_out=d["x_out"],
                           smooth_sweeps=d["smooth_sweeps"])
            mjson.write_text(json.dumps(rec, indent=2), encoding="utf-8")

        total_cells += d["grid"]["n_cells"] * len(d["models"])
        manifest_rows.append(dict(
            case_id=d["case_id"], naca=d["naca_digits"], re=d["re"],
            aoa_deg=d["aoa_deg"], mesh_level=d["mesh_level"],
            u_inf=d["u_inf"], y_plus_target=d["y_plus_target"],
            first_cell_height=d["boundary_layer"]["first_cell_height"],
            n_cells=d["grid"]["n_cells"], normal_growth=d["grid"]["normal_growth"],
            models=d["models"], journals=jous,
            mesh=mesh_path.relative_to(REPO).as_posix(),
            mesh_cmd=cmd, note=d["note"]))
        print("[cases] %-30s U=%7.2f m/s  cells=%7d  y+=%.2f  dy1=%.2e m  "
              "g=%.4f  models=%s"
              % (d["case_id"], d["u_inf"], d["grid"]["n_cells"],
                 d["y_plus_target"], d["boundary_layer"]["first_cell_height"],
                 d["grid"]["normal_growth"], ",".join(d["models"])))
        if d["grid"]["normal_growth"] > 1.2:
            print("[cases]   WARNING: wall-normal growth ratio %.3f > 1.2 -- "
                  "raise nj or lower r_far" % d["grid"]["normal_growth"])
        if d["mach_estimate"] > 0.3:
            print("[cases]   WARNING: M ~ %.2f > 0.3 -- the incompressible "
                  "assumption in case_template.jou is marginal"
                  % d["mach_estimate"])

    (out_root / "MANIFEST.json").write_text(json.dumps(dict(
        generated=str(date.today()), source=str(args.manifest),
        defaults=defaults, n_cases=len(manifest_rows), n_journals=n_jou,
        total_cells_all_runs=total_cells, cases=manifest_rows), indent=2),
        encoding="utf-8")
    write_runbook(out_root, manifest_rows, total_cells, n_jou)
    print("\n[cases] %d cases, %d journals, %d cells summed over all runs"
          % (len(manifest_rows), n_jou, total_cells))
    print("[cases] wrote %s and %s"
          % (out_root / "MANIFEST.json", out_root / "RUNBOOK.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
