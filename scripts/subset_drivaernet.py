"""scripts/subset_drivaernet.py -- reproducible DrivAerNet++ subset manifest.

DrivAerNet++ is ~8000 designs and over 39 TB in full. This study needs 400-800
designs' worth of *surface geometry plus drag coefficients* and nothing else.
This script turns an already-downloaded pile of coefficient CSVs and decimated
meshes into a seeded, stratified subset with an exact, committable list of
design IDs -- because that manifest is what lets someone else reproduce the
study without the 39 TB.

**This script downloads nothing.** It requires that the user has already
completed the Globus / Harvard Dataverse steps in `docs/plans/DRIVAERNET_ACCESS.md`
and can point `--coeff-csv` and `--design-dir` at local files. That separation
is deliberate: the download needs an interactive Globus login and a CC BY-NC-4.0
acceptance, neither of which an agent can or should do.

What it does
------------
1. Reads one or more coefficient CSVs (stdlib `csv`, no pandas dependency).
2. Auto-detects the stratification columns actually present -- car category
   (fastback / notchback / estateback), wheel configuration, underbody
   configuration -- and reports what it found rather than assuming a schema.
3. Intersects the design IDs against the mesh/point-cloud files actually on
   disk, so the manifest can never name a design that is not there.
4. Draws a **seeded, stratified** subset with proportional allocation plus a
   floor per stratum, so shape-family holdout stays possible.
5. Optionally decimates each selected design to 8k-16k points and stores it as
   float16.
6. Writes `subset_manifest.json` with the exact design IDs, the strata counts,
   the source-file checksums and the licence note.

Stratification and why it is not optional
-----------------------------------------
The 3D leg exists to repeat the OOD protocol on a second dataset. Its key split
is shape-family holdout: train on fastback + notchback, test on estateback
(spec section 5.7). A subset that happened to contain twelve estatebacks would
make that split meaningless. Proportional allocation with a floor guarantees
every stratum survives at a usable size, and the seed makes the draw auditable.

Usage
-----
    # manifest only (default; touches no mesh files)
    .venv/Scripts/python.exe scripts/subset_drivaernet.py \
        --coeff-csv data/raw/drivaernet/AeroCoefficients.csv \
        --design-dir data/raw/drivaernet/meshes \
        --out data/processed/drivaernet --n 600 --seed 0

    # also decimate + store float16 point clouds
    ... --decimate --points 12288

    # inspect what columns/strata the CSVs actually offer, then stop
    ... --describe
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

LICENCE = ("DrivAerNet++ is CC BY-NC 4.0: non-commercial research use only, "
           "attribution required. This repository redistributes design IDs and "
           "code, never the dataset itself. Cite Elrefaie, Morar, Dai & Ahmed, "
           "NeurIPS 2024.")

# Column-name patterns. DrivAerNet++ ships several CSVs whose headers differ in
# case, spacing and exact wording between releases, so match loosely and REPORT
# what was matched rather than hard-coding one release's schema.
ID_PATTERNS = [r"^design[_ ]?id$", r"^design$", r"^id$", r"^name$",
               r"^sample[_ ]?id$", r"^experiment[_ ]?name$"]
CD_PATTERNS = [r"^c[_ ]?d$", r"^cd$", r"^average[_ ]?cd$", r"^drag$",
               r"^drag[_ ]?coefficient$"]
CL_PATTERNS = [r"^c[_ ]?l$", r"^cl$", r"^average[_ ]?cl$", r"^lift$",
               r"^lift[_ ]?coefficient$"]
AREA_PATTERNS = [r"^frontal[_ ]?area$", r"^area$", r"^a[_ ]?ref$",
                 r"^projected[_ ]?frontal[_ ]?area$"]

# Candidate stratification columns, most important first.
STRATA_PATTERNS = [
    ("category", [r"^category$", r"^car[_ ]?category$", r"^design[_ ]?category$",
                  r"^body[_ ]?style$", r"^variant$", r"^type$"]),
    ("wheels", [r"^wheels?$", r"^wheel[_ ]?config(uration)?$",
                r"^wheel[_ ]?type$", r"^tyres?$"]),
    ("underbody", [r"^underbody$", r"^under[_ ]?body[_ ]?config(uration)?$",
                   r"^underbody[_ ]?type$", r"^floor$"]),
]

# Body styles the shape-family split depends on. Used only to sanity-report; a
# category column with different vocabulary is still accepted and stratified on.
KNOWN_CATEGORIES = ("fastback", "notchback", "estateback", "estate", "wagon")

MESH_SUFFIXES = (".stl", ".ply", ".obj", ".vtk", ".vtp", ".vtu",
                 ".npy", ".npz", ".pt", ".paddle_tensor")


# ---------------------------------------------------------------------------
# Optional dependencies: guarded, never imported at module scope
# ---------------------------------------------------------------------------

def _need_numpy():
    try:
        import numpy as np
        return np
    except ImportError:
        raise SystemExit(
            "numpy is required for --decimate (float16 storage needs it).\n"
            "  uv pip install numpy")


def _mesh_backend():
    """Return ('pyvista', module) or ('trimesh', module), else raise."""
    try:
        import pyvista as pv
        return "pyvista", pv
    except ImportError:
        pass
    try:
        import trimesh
        return "trimesh", trimesh
    except ImportError:
        pass
    raise SystemExit(
        "--decimate needs pyvista or trimesh, neither of which is installed.\n"
        "  uv pip install pyvista      (or)   uv pip install trimesh\n"
        "Both ship pure wheels on Windows. Run without --decimate to produce\n"
        "the manifest alone -- that is the part the paper actually commits.")


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def _match(header, patterns):
    """First header column matching any pattern, case-insensitively."""
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for h in header:
            if rx.match(h.strip()):
                return h
    return None


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_coeff_csvs(paths):
    """Merge coefficient CSVs into {design_id: {field: value}} plus provenance.

    Later files update earlier ones field-by-field, so a coefficients CSV and a
    separate metadata CSV can be passed together and are joined on design id.
    """
    table = {}
    sources = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise SystemExit("coefficient CSV not found: %s" % p)
        with open(p, newline="", encoding="utf-8-sig") as f:
            rdr = csv.DictReader(f)
            header = [h for h in (rdr.fieldnames or []) if h]
            if not header:
                raise SystemExit("%s: no header row" % p)
            id_col = _match(header, ID_PATTERNS)
            if id_col is None:
                raise SystemExit(
                    "%s: could not find a design-id column among %s.\n"
                    "Rename the column or extend ID_PATTERNS in this script."
                    % (p, header))
            cd_col = _match(header, CD_PATTERNS)
            cl_col = _match(header, CL_PATTERNS)
            area_col = _match(header, AREA_PATTERNS)
            strata_cols = {}
            for name, pats in STRATA_PATTERNS:
                col = _match(header, pats)
                if col:
                    strata_cols[name] = col

            n_rows = 0
            for row in rdr:
                did = (row.get(id_col) or "").strip()
                if not did:
                    continue
                rec = table.setdefault(did, {"design_id": did})
                if cd_col:
                    rec["cd"] = _num(row.get(cd_col))
                if cl_col:
                    rec["cl"] = _num(row.get(cl_col))
                if area_col:
                    rec["frontal_area"] = _num(row.get(area_col))
                for name, col in strata_cols.items():
                    v = (row.get(col) or "").strip()
                    if v:
                        rec[name] = v
                n_rows += 1

            sources.append(dict(
                path=str(p), sha256=sha256(p), rows=n_rows,
                id_column=id_col, cd_column=cd_col, cl_column=cl_col,
                area_column=area_col,
                strata_columns=strata_cols, header=header))
    return table, sources


def _num(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def infer_category(rec):
    """Fall back to parsing the body style out of the design id.

    DrivAerNet++ ids encode the variant (e.g. `..._Fastback_...`). If no
    category column exists, the id is the next-best source; if that fails too
    the design lands in an explicit 'unknown' stratum rather than being silently
    dropped or silently merged.
    """
    if rec.get("category"):
        return rec["category"].strip().lower()
    did = rec["design_id"].lower()
    for c in KNOWN_CATEGORIES:
        if c in did:
            return c
    return "unknown"


# ---------------------------------------------------------------------------
# Mesh discovery
# ---------------------------------------------------------------------------

def index_designs(dirs, glob_pat):
    """Map design_id -> file path for every mesh/point-cloud file on disk."""
    found = {}
    for d in dirs:
        d = Path(d)
        if not d.exists():
            raise SystemExit("design directory not found: %s" % d)
        for f in sorted(d.rglob(glob_pat)):
            if f.is_file() and f.suffix.lower() in MESH_SUFFIXES:
                found.setdefault(f.stem, f)
    return found


# ---------------------------------------------------------------------------
# Stratified selection
# ---------------------------------------------------------------------------

def stratify(records, keys):
    """Group design records into strata keyed by a tuple of metadata values."""
    strata = {}
    for r in records:
        key = tuple(str(r.get(k, "unknown")).strip().lower() or "unknown"
                    for k in keys)
        strata.setdefault(key, []).append(r)
    return strata


def allocate(strata, n_target, min_per_stratum):
    """Proportional allocation with a floor, resolved by largest remainder.

    Deterministic given the strata contents: ties break on the stratum key, not
    on dict order, so the allocation is reproducible across Python versions.
    """
    total = sum(len(v) for v in strata.values())
    if total == 0:
        raise SystemExit("no designs left after intersecting CSVs with files on disk")
    n_target = min(n_target, total)

    keys = sorted(strata)
    alloc = {}
    for k in keys:
        alloc[k] = min(len(strata[k]), min_per_stratum)
    remaining = n_target - sum(alloc.values())
    if remaining < 0:
        # the floors alone overshoot: trim from the largest allocations first
        while remaining < 0:
            k = max(keys, key=lambda k: (alloc[k], k))
            if alloc[k] == 0:
                break
            alloc[k] -= 1
            remaining += 1
        return alloc

    room = {k: len(strata[k]) - alloc[k] for k in keys}
    pool = sum(room.values())
    if pool == 0:
        return alloc
    exact = {k: remaining * room[k] / pool for k in keys}
    for k in keys:
        alloc[k] += int(exact[k])
    short = n_target - sum(alloc.values())
    order = sorted(keys, key=lambda k: (-(exact[k] - int(exact[k])), k))
    i = 0
    while short > 0 and i < len(order) * 4:
        k = order[i % len(order)]
        if alloc[k] < len(strata[k]):
            alloc[k] += 1
            short -= 1
        i += 1
    return alloc


def select(strata, alloc, seed):
    """Seeded draw within each stratum. Sorted first, so the RNG is the only
    source of variation -- filesystem or dict ordering must not leak in."""
    rng = random.Random(seed)
    picked = []
    for k in sorted(strata):
        pool = sorted(strata[k], key=lambda r: r["design_id"])
        n = min(alloc.get(k, 0), len(pool))
        picked.extend(rng.sample(pool, n) if n else [])
    picked.sort(key=lambda r: r["design_id"])
    return picked


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------

def decimate_design(path, n_points, seed, backend, mod, np):
    """Return (points float16 (N,3), n_source_points, method)."""
    suffix = path.suffix.lower()

    if suffix in (".npy", ".npz"):
        arr = np.load(path)
        if hasattr(arr, "files"):
            arr = arr[arr.files[0]]
        pts = np.asarray(arr, dtype=np.float32).reshape(-1, 3)
        n_src = len(pts)
        if n_src > n_points:
            rng = np.random.default_rng(seed)
            pts = pts[np.sort(rng.choice(n_src, n_points, replace=False))]
        return pts.astype(np.float16), n_src, "random-subsample"

    if backend == "pyvista":
        mesh = mod.read(str(path))
        surf = mesh.extract_surface() if hasattr(mesh, "extract_surface") else mesh
        n_src = surf.n_points
        pts = np.asarray(surf.points, dtype=np.float32)
        if n_src > n_points:
            # decimate_pro preserves the surface far better than dropping
            # vertices at random, which would bias toward densely tessellated
            # regions (wheel arches, mirrors) and thin out flat panels.
            try:
                red = 1.0 - float(n_points) / float(n_src)
                dec = surf.triangulate().decimate_pro(red, preserve_topology=True)
                pts = np.asarray(dec.points, dtype=np.float32)
                method = "pyvista.decimate_pro"
            except Exception:
                rng = np.random.default_rng(seed)
                pts = pts[np.sort(rng.choice(n_src, n_points, replace=False))]
                method = "pyvista.random-subsample"
        else:
            method = "pyvista.as-is"
        return pts.astype(np.float16), n_src, method

    # trimesh: area-weighted surface sampling gives an even point density
    mesh = mod.load(str(path), force="mesh")
    n_src = int(getattr(mesh, "vertices", []).__len__())
    rng = np.random.default_rng(seed)
    try:
        pts, _ = mod.sample.sample_surface_even(mesh, n_points, seed=seed)
        if len(pts) < n_points:
            pts, _ = mod.sample.sample_surface(mesh, n_points)
        method = "trimesh.sample_surface_even"
    except Exception:
        pts, _ = mod.sample.sample_surface(mesh, n_points)
        method = "trimesh.sample_surface"
    return np.asarray(pts, dtype=np.float16), n_src, method


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build a seeded, stratified DrivAerNet++ subset manifest "
                    "from already-downloaded files. Downloads nothing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--coeff-csv", type=Path, action="append", required=True,
                    help="coefficient / metadata CSV (repeatable; joined on design id)")
    ap.add_argument("--design-dir", type=Path, action="append", default=None,
                    help="directory of meshes or point clouds (repeatable)")
    ap.add_argument("--mesh-glob", default="*",
                    help="glob applied inside --design-dir")
    ap.add_argument("--out", type=Path, default=Path("data/processed/drivaernet"))
    ap.add_argument("--n", type=int, default=600,
                    help="target subset size (spec asks for 400-800)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-per-stratum", type=int, default=10,
                    help="floor so shape-family holdout stays possible")
    ap.add_argument("--strata", action="append", default=None,
                    help="stratification field (repeatable). Default: whichever "
                         "of category/wheels/underbody are present.")
    ap.add_argument("--points", type=int, default=12288,
                    help="target points per design (spec asks for 8k-16k)")
    ap.add_argument("--decimate", action="store_true",
                    help="also write float16 point clouds; needs pyvista or trimesh")
    ap.add_argument("--describe", action="store_true",
                    help="report the columns and strata found, then exit")
    ap.add_argument("--allow-missing-meshes", action="store_true",
                    help="build the manifest from the CSVs alone (no --design-dir)")
    args = ap.parse_args(argv)

    if not (400 <= args.n <= 800):
        print("[subset] NOTE: --n %d is outside the 400-800 the spec asks for "
              "(200 is the documented bandwidth fallback)." % args.n)
    if args.decimate and not (8192 <= args.points <= 16384):
        print("[subset] NOTE: --points %d is outside the 8k-16k the spec asks for."
              % args.points)

    table, sources = read_coeff_csvs(args.coeff_csv)
    print("[subset] %d designs across %d CSV(s)" % (len(table), len(sources)))
    for s in sources:
        print("[subset]   %s: id=%s cd=%s strata=%s"
              % (Path(s["path"]).name, s["id_column"], s["cd_column"],
                 s["strata_columns"] or "NONE FOUND"))

    for rec in table.values():
        rec["category"] = infer_category(rec)

    # ---- intersect with what is actually on disk -------------------------
    files = {}
    if args.design_dir:
        files = index_designs(args.design_dir, args.mesh_glob)
        print("[subset] %d mesh/point-cloud files on disk" % len(files))
        before = len(table)
        table = {k: v for k, v in table.items() if k in files}
        print("[subset] %d designs have both coefficients and geometry "
              "(dropped %d)" % (len(table), before - len(table)))
        if not table:
            raise SystemExit(
                "No design id in the CSVs matched a filename stem in "
                "--design-dir.\nCheck that the mesh filenames are the design "
                "ids (e.g. <design_id>.stl); adjust --mesh-glob if they carry "
                "a suffix.")
    elif not args.allow_missing_meshes:
        raise SystemExit(
            "--design-dir is required so the manifest cannot name a design "
            "whose geometry is absent.\nPass --allow-missing-meshes to build "
            "from coefficients alone (coefficient-only ablations).")

    records = sorted(table.values(), key=lambda r: r["design_id"])
    with_cd = sum(1 for r in records if r.get("cd") is not None)
    if with_cd < len(records):
        print("[subset] WARNING: %d/%d designs have no CD value"
              % (len(records) - with_cd, len(records)))

    # ---- choose stratification keys --------------------------------------
    if args.strata:
        keys = list(args.strata)
    else:
        present = set()
        for r in records:
            present.update(k for k in ("category", "wheels", "underbody") if r.get(k))
        keys = [k for k in ("category", "wheels", "underbody") if k in present]
    if not keys:
        keys = ["category"]
    print("[subset] stratifying on: %s" % ", ".join(keys))

    strata = stratify(records, keys)
    print("[subset] %d strata:" % len(strata))
    for k in sorted(strata):
        print("[subset]   %-40s %5d available" % ("|".join(k), len(strata[k])))

    cats = sorted({r["category"] for r in records})
    print("[subset] categories seen: %s" % ", ".join(cats))
    missing = [c for c in ("fastback", "notchback", "estateback")
               if not any(c in x for x in cats)]
    if missing:
        print("[subset] WARNING: %s absent -- the shape-family OOD split of "
              "spec section 5.7 needs all three." % ", ".join(missing))

    if args.describe:
        print("[subset] --describe: stopping before selection.")
        return 0

    # ---- allocate and draw ------------------------------------------------
    alloc = allocate(strata, args.n, args.min_per_stratum)
    picked = select(strata, alloc, args.seed)
    print("[subset] selected %d designs (target %d)" % (len(picked), args.n))
    for k in sorted(strata):
        if alloc.get(k):
            print("[subset]   %-40s %5d selected of %d"
                  % ("|".join(k), alloc[k], len(strata[k])))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- optional decimation ----------------------------------------------
    designs = []
    if args.decimate:
        np = _need_numpy()
        backend, mod = _mesh_backend()
        pts_dir = out / "points"
        pts_dir.mkdir(parents=True, exist_ok=True)
        print("[subset] decimating with %s -> %d points, float16" % (backend, args.points))

    for idx, r in enumerate(picked):
        entry = dict(design_id=r["design_id"],
                     category=r.get("category"),
                     wheels=r.get("wheels"),
                     underbody=r.get("underbody"),
                     cd=r.get("cd"), cl=r.get("cl"),
                     frontal_area=r.get("frontal_area"))
        src = files.get(r["design_id"])
        if src is not None:
            entry["source_file"] = str(src)
        if args.decimate and src is not None:
            # per-design seed: re-running one design reproduces its own cloud
            dseed = (args.seed * 1000003 + idx) & 0xFFFFFFFF
            try:
                pts, n_src, method = decimate_design(src, args.points, dseed,
                                                     backend, mod, np)
            except Exception as exc:
                print("[subset]   FAILED %s: %s" % (r["design_id"], exc))
                entry["decimation_error"] = str(exc)
                designs.append(entry)
                continue
            dst = pts_dir / (r["design_id"] + ".npy")
            np.save(dst, pts)
            entry.update(points_file=str(dst.relative_to(out).as_posix()),
                         n_points=int(len(pts)), n_points_source=int(n_src),
                         decimation=method, dtype="float16")
            if (idx + 1) % 25 == 0:
                print("[subset]   %d/%d decimated" % (idx + 1, len(picked)))
        designs.append(entry)

    # ---- manifest ----------------------------------------------------------
    manifest = dict(
        name="drivaernet_subset",
        created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        generator="scripts/subset_drivaernet.py",
        seed=args.seed,
        n_requested=args.n,
        n_selected=len(picked),
        n_candidates=len(records),
        strata_keys=keys,
        min_per_stratum=args.min_per_stratum,
        strata={"|".join(k): dict(available=len(v), selected=alloc.get(k, 0))
                for k, v in sorted(strata.items())},
        decimation=dict(enabled=bool(args.decimate),
                        target_points=args.points, dtype="float16"),
        sources=sources,
        licence=LICENCE,
        designs=designs,
    )
    mpath = out / "subset_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[subset] wrote %s (%d designs)" % (mpath, len(designs)))
    print("[subset] COMMIT THIS FILE. It is what makes the 3D leg reproducible "
          "without the 39 TB.")
    print("[subset] %s" % LICENCE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
