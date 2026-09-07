# DRIVAERNET_ACCESS.md — getting the DrivAerNet++ subset

**This is a user task, not an agent task.** It needs an interactive Globus
login and an explicit licence acceptance, neither of which an agent can or
should do on your behalf. Decision **D-005** makes AirfRANS the self-sufficient
core of the paper and DrivAerNet++ a user-gated extension precisely so this
never blocks anything.

The end state you are working toward: two kinds of local file, and nothing else.

1. **Coefficient / metadata CSVs** — drag (and where available lift), frontal
   area, and the design-category labels. Tens of MB at most.
2. **Surface geometry, one file per design** — the decimated-mesh or
   point-cloud modality. A few GB.

Then run `scripts/subset_drivaernet.py`, commit the resulting
`subset_manifest.json`, and the 3D leg is reproducible by anyone without the
39 TB.

---

## 0. Before anything: the licence

DrivAerNet++ is **CC BY-NC 4.0**.

- **Non-commercial research use only.** Fine for a preprint and this study.
- **Attribution required.** Cite Elrefaie, Morar, Dai & Ahmed, *DrivAerNet++*,
  NeurIPS 2024 (arXiv:2406.09624).
- **Do not redistribute the data**, processed or otherwise, through this
  repository. We commit *design IDs and code*. That is both licence-safe and, as
  it happens, the more useful artefact.
- State the licence in the paper's dataset section. AirfRANS (ODbL-1.0) and
  DrivAerNet++ (CC BY-NC-4.0) have different obligations and both must appear.
- The Fluent verification set we generate ourselves stays under this repo's MIT
  licence, deliberately unencumbered by either.

Accepting the licence is a click on the Dataverse landing page. Read it rather
than clicking through.

---

## 1. Scope: what to download, and what not to

The full release is **> 39 TB** because it includes volumetric flow fields and
full-resolution CFD meshes. This study needs neither.

| Modality | Rough size | Take it? |
|---|---|---|
| Aerodynamic coefficient CSVs (Cd, Cl, frontal area) | < 100 MB | **Yes** — required |
| Design metadata / category labels / official splits | < 100 MB | **Yes** — required |
| Point clouds or decimated surface meshes (~8k–16k pts) | ~2–10 GB for the whole set | **Yes** — this is the geometry modality |
| Full-resolution STL surface meshes | ~100 GB+ | No |
| Volumetric flow-field data (`.vtu`) | tens of TB | **No.** This is the 39 TB. |
| Surface-field data (pressure/shear on the body) | large | Not for the current protocol — the 3D leg predicts Cd from geometry |

If bandwidth or disk becomes the binding constraint, the documented fallback is
**200 designs**, and the 3D leg is reframed as a smaller transfer study. The 2D
paper stands alone either way; do not let this component grow into the critical
path.

Disk check before starting: `CONTEXT.md` §1 records ~120 GB free on D:, and
AirfRANS already claims ~20 GB of it. A 600-design geometry subset at a few GB
is comfortable; a full-resolution mesh download is not.

---

## 2. Step by step

### 2.1 Read the repo first

Start at **<https://github.com/Mohamedelrefaie/DrivAerNet>**. It is the
authoritative index and it changes more often than any third-party description
of it. Specifically, get from it:

- the current Harvard Dataverse DOI / landing-page link,
- the **file names** of the coefficient CSVs and of the point-cloud archives
  (these have been renamed between releases — take them from the repo, do not
  trust a filename quoted in this document or anywhere else),
- the **official train/val/test split files**, which are plain ID lists,
- the task definitions and split conventions (spec reading-list item 6 says read
  these *before* downloading anything, and that is good advice: it determines
  which modality you actually need).

### 2.2 Get a Globus account and endpoint

The bulk data moves over **Globus**, not HTTP.

1. Create an account at <https://www.globus.org/> — free, and institutional
   logins (university SSO) work; ORCID and Google also work.
2. Install **Globus Connect Personal** on this machine
   (<https://www.globus.org/globus-connect-personal>). This turns the laptop
   into a named endpoint that transfers can target.
3. Start it and give it a collection name you will recognise.
4. **Grant it access to the D: drive.** By default Globus Connect Personal only
   shares your home directory. In its Preferences → Access tab, add the target
   directory (e.g. `D:\Personal Projects\geom_aware_neural_operator\data\raw\drivaernet`)
   as a writable path. Forgetting this is the single most common way the
   transfer appears to start and then fails.

### 2.3 Accept the licence and open the transfer

1. Open the Harvard Dataverse landing page from the GitHub repo.
2. Accept the CC BY-NC-4.0 terms.
3. Follow its Globus transfer link. This opens the Globus File Manager with the
   dataset endpoint on one side.
4. Set your personal collection as the destination and navigate to
   `data/raw/drivaernet/` inside it.

### 2.4 Select only what you need

In the Globus File Manager, **select individual files, never the dataset root.**
Selecting the root queues the full 39 TB and it will not be obvious that it has
happened until the transfer has been running for a day.

Queue, in this order:

1. the coefficient CSV(s),
2. the metadata / split files,
3. the point-cloud or decimated-surface archive.

Start the transfer. Globus is asynchronous, resumable, and emails you on
completion; it survives sleeping the laptop and losing the network, which is why
the dataset is distributed this way. Monitor it in the Globus web UI's Activity
tab, not by watching the folder.

### 2.5 Verify before unpacking

```bash
du -sh data/raw/drivaernet/*
```

Sanity checks: the CSV should be tens of MB, not tens of GB; the geometry
archive should be single-digit GB. If either is wildly off, stop and re-check
what got selected rather than unpacking 39 TB of `.vtu`.

Unpack the geometry archive so that **each design is one file whose stem is its
design ID** — e.g. `data/raw/drivaernet/meshes/<design_id>.stl`. The subsetting
script matches design IDs against filename stems; if the release nests them
differently, either flatten the tree or pass `--mesh-glob` to reach them.

### 2.6 Build the subset

```bash
# First: look before selecting. Reports which columns and strata exist.
.venv/Scripts/python.exe scripts/subset_drivaernet.py \
    --coeff-csv data/raw/drivaernet/<coefficients>.csv \
    --design-dir data/raw/drivaernet/meshes \
    --describe

# Then: the real draw.
.venv/Scripts/python.exe scripts/subset_drivaernet.py \
    --coeff-csv data/raw/drivaernet/<coefficients>.csv \
    --coeff-csv data/raw/drivaernet/<metadata>.csv \
    --design-dir data/raw/drivaernet/meshes \
    --out data/processed/drivaernet \
    --n 600 --seed 0 --min-per-stratum 10

# Optionally decimate to float16 point clouds (needs pyvista or trimesh):
    ... --decimate --points 12288
```

Read the output, do not just let it scroll:

- **`stratifying on:`** should list `category, wheels, underbody`. If it only
  says `category`, the metadata CSV was not passed or its columns are named
  differently — the script prints the header it saw so you can extend the
  patterns near the top of the file.
- **`categories seen:`** must include fastback, notchback **and** estateback.
  The script warns if one is missing, because the shape-family OOD split (spec
  §5.7) trains on fastback + notchback and tests on estateback. A subset without
  estatebacks silently deletes the most important 3D experiment.
- **`designs have both coefficients and geometry`** should be close to the CSV
  row count. A large drop means the filename-stem-to-design-ID match is failing.

### 2.7 Commit the manifest

```bash
git add data/processed/drivaernet/subset_manifest.json
```

`subset_manifest.json` holds the exact design IDs, the strata counts, the seed,
and SHA-256 checksums of the source CSVs. **This file is the deliverable.** It is
what lets a reader reproduce the 3D leg without the 39 TB, and it is the
"exact subset IDs" row of the paper's Table 6. The float16 point clouds under
`points/` are a local cache and stay gitignored.

---

## 3. Reproducibility notes

- **The seed is part of the result.** Re-running with `--seed 0` reproduces the
  same 600 IDs exactly; the script sorts every pool before sampling so
  filesystem and dict ordering cannot leak into the draw.
- **Never re-draw after seeing results.** If the subset needs to change, that is
  a new manifest with a new name and a note in `DECISIONS.md`, not a quiet
  re-run of the same command.
- **float16 is a storage decision, not a precision claim.** Coordinates are
  O(1 m) with ~1 mm of meaningful resolution, so float16's ~1e-3 relative
  precision is adequate for geometry; the *coefficients* stay float64 in the
  manifest. Say this in the paper rather than leaving a reviewer to wonder.
- **Decimation method is recorded per design** in the manifest
  (`pyvista.decimate_pro`, `trimesh.sample_surface_even`, or a random subsample
  fallback), because a mixture of methods across designs would be a confound and
  needs to be visible rather than inferred.

---

## 4. If access does not happen

Nothing downstream breaks. Per D-005 the paper is scoped as 2D-complete and
3D-extension-ready. The honest framing in that case is a Limitations sentence
saying the protocol was validated on one dataset family and that the 3D
transfer study is future work — which is a materially weaker claim than "it
generalises", and is the correct one to make.
