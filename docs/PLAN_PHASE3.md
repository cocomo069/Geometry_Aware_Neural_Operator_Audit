# PLAN_PHASE3.md, Drive to completion (arXiv v1) with minimal user interference

Planned by Fable 5, 2026-09-03. Executors: Opus (orchestrator: infra, Kaggle cycle,
UQ, merges, commits) and Sonnet (paper prose, schematics, mechanical figure/table
work). Supersedes the sequencing in PLAN_PHASE2.md; the per-phase designs there
(Phase A UQ, Phase B data-eff, Phase C ablations, Phase D AL) stay valid and are
referenced, not repeated.

Everything below is dependency-ordered. Section 7 is the ordered to-do list the
orchestrator actually runs; sections 1 to 6 are the design behind each item.

---

## 0. State snapshot (verified on disk and via the Kaggle API, 2026-09-03)

**Local repo (`main`, HEAD 22114e0):**
- `results/`: 18 core runs (3 models x 6 splits, s0), 12 baselines, K=5 ensembles for
  transolver + sdf_fno on {full, reynolds, aoa, shape5} (32 run dirs), plus `gnn_full_s1`.
  GNN ensembles beyond that: **absent locally** (missing: `gnn_full_s{2,3,4}`,
  `gnn_{reynolds,aoa,shape5}_s{1,2,3,4}` = 15 runs).
- `results/uq/`: 60 reports. M2/M3 at K in {1,3,5}; GNN at K=1 (K=2 on `full`).
- `checkpoints/`: 1.6 GB, 53 run dirs. Every run dir holds `best.pt` AND `last.pt`, each
  16 to 17 MB, **both carrying optimizer + scheduler + RNG state** (`scripts/train.py::
  save_checkpoint` writes the full payload to both). Model weights alone are ~6 MB.
- Uncommitted, from the AL agent (in flight): `scripts/run_active.py`,
  `tests/test_run_active.py`, `results/active/{acquisition_ranking.csv,
  selection_summary.json}`, modified `fluent/cases_to_run.json` (now 27 cases: 3
  gridstudy + 3 arms x 8, scored by `transolver_full_k5`, pool 630). Commit these once
  that agent reports done.
- Paper: `paper/main.tex` compiles; 23 `\todo`, 22 `\pending`, 12 `\figph` placeholders.

**Kaggle (account cocomo069):**
- Kernel `cocomo069/geo-op-session` is RUNNING (data-eff sweep, pushed 2026-09-02 21:36
  UTC together with a fresh `geo-op-code`). Every launch so far pushed a new *version*
  of this one slug.
- Dataset `geo-op-runs`: last version 2026-08-25 17:56 UTC, 1.56 GB. It holds core +
  M2/M3 ensembles + `gnn_full_s1`, i.e. **no GNN ensemble progress after 08-25**. Every
  GNN-ensemble session launched since resumed from this same state, so each one
  re-trained the same leading GNN runs from scratch and its output was never merged.
- Measured T4 wall-clock per 400-epoch run (from `metrics.json::train_time_s`):
  GNN full 3.0 h, aoa 3.0 h, reynolds 1.8 h, shape5 1.7 h, scarce 0.7 h, combined 0.2 h;
  sdf_fno full 0.36 h; transolver full 0.41 h. **PLAN_PHASE2's "GNN 1.5 h" was wrong by
  2x; every GPU estimate below uses the measured numbers.**

**Root cause of the pull stall (read from `.venv/Lib/site-packages/kaggle/api/
kaggle_api_extended.py`, Kaggle CLI 2.2.4):**
1. `KaggleApi.kernels_output()` (line 6640) does `requests.get(item.url, stream=True)`
   and then `out.write(download_response.content)`: the whole 2 GB `checkpoints.zip`
   is one HTTP GET, buffered in RAM, no chunking, no timeout, no retry, no Range resume.
   On this connection a single 2 GB GET goes idle and never completes.
2. `kernels_output()` parses the `owner/slug/<version>` suffix but never sends it
   (`request.user_name`, `request.kernel_slug` only): the CLI serves the **latest**
   session's output only. Older versions are reachable only through the web UI.
3. `kaggle/session_driver.py` step 5 re-zips the **entire cumulative** `checkpoints/`
   (restored prior runs + new runs) every session, and `kaggle/pull_results.py` re-uploads
   the entire local `checkpoints/` (1.6 GB) to `geo-op-runs` every cycle. Both grow with
   every session.

---

## 1. Infra fix: make large-checkpoint sessions pull reliably

### 1.1 Decision

Adopt all four options, each in the form that removes a whole failure class:

| option | adopted as | why |
|---|---|---|
| (b) export only NEW/changed | driver exports a run's checkpoints only if that run was **not already finished when the session started** | stops the cumulative growth at its source |
| (d) drop one of best/last | finished runs export **best.pt only**; unfinished runs export **last.pt + best.pt** | last.pt exists for resume; a run with `metrics.json` never resumes again. Halves finished-run bytes |
| (a) per-run zips + manifest | one `ckpt_<run_id>.zip` per exported run + `EXPORT_MANIFEST.json` | lets the puller fetch selectively, retry per file, and skip what is already local |
| (c) small files first, checkpoints lazily | `pull_results.py` always fetches `results.zip`, `session.log`, `STATUS.txt`, `EXPORT_MANIFEST.json` first, merges results immediately, then fetches only the per-run zips it needs | metrics land even when checkpoints fail; a stall now costs one 6 to 33 MB file |

Plus two cheap changes that remove the remaining sharp edges:
- **One kernel slug per queued sweep** (`geo-op-dataeff`, `geo-op-ablations`,
  `geo-op-gnnens`), because the CLI can only see the latest version's output.
  `launch.py --kernel-slug` already exists; the cycle script (section 2) maps spec to slug.
- **Republish only what the next session needs**: `results/**` plus `checkpoints/<rid>/
  last.pt` for runs WITHOUT `metrics.json`. `scripts/sweep.py` skips on `metrics.json`
  and `scripts/train.py` resumes from `last.pt`; finished checkpoints are never read
  on Kaggle. `geo-op-runs` drops from 1.5 GB to ~10 MB and the upload stall vanishes.
  Local `checkpoints/` becomes the only copy of finished `best.pt`; it is gitignored
  and on D:, so make one archive copy (`checkpoints_archive_<date>.zip` on another
  drive) before deleting anything, and keep the 08-25 `geo-op-runs` version as the
  cloud archive of core + M2/M3 ensembles.

Rejected: (a) alone (still cumulative), (c) alone (still a 2 GB file somewhere),
saving `best.pt` weights-only from `train.py` (touches the frozen training path for a
2.5x gain the export filter already gets; leave `train.py` alone).

### 1.2 `kaggle/session_driver.py` (executor: Opus)

Keep steps 1 to 4 as they are. Change step 3 and step 5:

- After the runs-dataset restore (step 3), record the finished set:
  `pre_done = {p.parent.name for p in (REPO/"results").glob("*/metrics.json")}`.
  Also `SESSION_ID = uuid.uuid4().hex[:12]` at import time, printed and written into
  `STATUS.txt`.
- New pure helper (module level, unit-testable, no Kaggle imports):
  ```python
  def select_export_set(results_dir, ckpt_dir, pre_done) -> dict[str, dict]:
      """{run_id: {"done": bool, "files": ["best.pt"] | ["last.pt","best.pt"]}}
      for every checkpoints/<run_id>/ that is NOT in pre_done. done := results/<run_id>/
      metrics.json exists. Finished -> ["best.pt"] (fallback ["last.pt"] if best is
      missing). Unfinished -> every *.pt present."""
  ```
- Step 5 export: `results.zip` unchanged (all of `results/`, small). Then for each
  `run_id, spec` in `select_export_set(...)`: write `WORK/ckpt_<run_id>.zip` with
  `ZIP_STORED` (torch files do not compress; STORED saves minutes at export and makes
  member byte ranges trivial) containing `checkpoints/<run_id>/<file>`. Then write
  `WORK/EXPORT_MANIFEST.json`:
  ```json
  {"session_id": "...", "sweep": "...", "commit": "...", "started_utc": "...",
   "finished_utc": "...", "sweep_rc": 0, "pre_done_count": 53,
   "runs": {"gnn_reynolds_s2": {"done": true, "zip": "ckpt_gnn_reynolds_s2.zip",
             "files": ["best.pt"], "bytes": 16806073, "sha256": "..."}}}
  ```
  Keep output filenames flat in `/kaggle/working` (the CLI `--file-pattern` regex
  matches `file_name`).
- `SWEEP` becomes a comma-separated list; step 4 passes each as a separate `--spec`
  (section 1.3). Budget check between specs is sweep.py's job.
- Log one line per exported zip with its size, and a final
  `[driver] exported N run zips, M MB total` so `session.log` documents export size.

### 1.3 `scripts/sweep.py`: multiple specs in priority order (executor: Opus, small)

`--spec` gets `action="append"`; `load_spec`/`expand_spec` are called per spec and the
plans concatenated in order, `name` = `"+".join(stems)`. Skip/budget logic unchanged.
One session can then drain `kaggle_dataeff.yaml` then `kaggle_ablations.yaml` then a
GNN-ensemble spec without a relaunch. Add a test in `tests/test_infra.py` (two tiny
specs, expansion order preserved, done-skipping still per run_id).

### 1.4 `kaggle/launch.py` (executor: Opus, small)

- `--sweep` accepts repeats (`action="append"`), joined with commas into the driver.
- `--kernel-slug` default stays `geo-op-session` for backward compatibility, but the
  cycle script always passes an explicit per-queue slug.
- Print the pushed version number (the `kernels push` output contains it) and append a
  line to `kaggle/launches.jsonl`: `{"utc", "slug", "specs", "commit", "runs_dataset"}`.
  This is the orchestrator's memory of what is in flight.

### 1.5 `kaggle/pull_results.py` rewrite (executor: Opus)

CLI: `--kernel <owner/slug>`, `--runs-dataset`, `--checkpoints {needed,all,none}`
(default `needed`), `--no-republish`, `--dest <dir>` (default a per-session folder under
`kaggle/pulls/<session_id>/` so a re-run never re-downloads what it already has).

Functions:
1. `fetch_small(kernel, dest)`: subprocess
   `kaggle kernels output <kernel> -p <dest> --file-pattern "^(results\.zip|session\.log|STATUS\.txt|EXPORT_MANIFEST\.json|MOUNT_TREE\.txt)$"`.
   Always first, always small. Parse `EXPORT_MANIFEST.json` (if absent: legacy session,
   see 1.7).
2. `merge_results(results_zip, root)`: extract `results/**` into the repo (as today).
   Also refuse to overwrite a local `results/<rid>/metrics.json` that differs from the
   incoming one unless `--force` (guards the local-vs-cloud clobber risk noted in D-019).
3. `needed_checkpoints(manifest, root, policy)`: the run ids to fetch. `needed` =
   run id matches `^(gnn|sdf_fno|transolver)_(full|scarce|reynolds|aoa|shape5|combined)_s\d$`
   (core/ensemble runs, whose `best.pt` feeds `run_uq.py` and `run_active.py`) OR
   `done == false` (its `last.pt` must be republished for resume), AND the target file is
   not already present locally with the manifest's byte size. Data-eff (`_n\d+_`) and
   tagged ablation runs are metrics-only downstream, so `needed` skips them.
4. `fetch_zip(kernel, name, dest, attempts=4, timeout=900)`: subprocess
   `kaggle kernels output <kernel> -p <dest> --file-pattern "^<name>$"` with a
   per-call timeout and retry; verify `bytes` and `sha256` against the manifest before
   extracting into `checkpoints/<rid>/`. One stalled 33 MB file costs one retry, never
   the session.
5. `republish(root, runs_slug)`: stage `results/**` + `checkpoints/<rid>/last.pt` for
   every `rid` lacking `metrics.json` into a temp dir, `kaggle datasets version
   --dir-mode zip -m "<session_id> merge"`. Skip when nothing changed since the last
   republish (compare a hash of the staged file list + sizes stored in
   `kaggle/runs_dataset_state.json`).
6. Record `{session_id, kernel, utc, merged_runs, fetched_zips}` in
   `kaggle/pulled_sessions.jsonl`; refuse to merge a `session_id` already listed unless
   `--force` (idempotent cycle, section 2).

### 1.6 Tests (CPU, `tests/test_infra.py` or new `tests/test_kaggle_export.py`)

`select_export_set` on a synthetic tree (finished, unfinished, pre-done cases);
`needed_checkpoints` policy table; manifest sha/bytes verification rejects a truncated
zip; sweep multi-spec ordering. No network in tests.

### 1.7 Timing: the fix only lands on the NEXT launch

The data-eff session running now (`geo-op-session`, old driver) will still export a
single ~2.1 GB `checkpoints.zip`. Pull policy for that session: `fetch_small` only;
merge `results/`; **do not attempt `checkpoints.zip`**. Data-eff checkpoints are not
used by anything downstream. If `session.log` shows a run cut by the wall clock
(`hit the wall-clock budget`), either let it retrain in the chained session (worst case
one GNN n400 run, ~1.7 h) or pull its `last.pt` with the Range extractor of 1.8
(~17 MB), then republish so the chained session resumes it.

### 1.8 One-off recovery of the already-trained GNN ensemble runs (do this FIRST)

The CLI can only see the latest session's output, and the running data-eff session will
become "latest" the moment it completes. So the window to pull the last GNN-ensemble
session through the API is **now**, and it may already be closed if the listing is
being served for the running version. Steps, in order:

1. **Discover what exists** (2 minutes, tiny downloads):
   `kaggle kernels output cocomo069/geo-op-session -p kaggle/pulls/recovery --file-pattern "^(results\.zip|session\.log|STATUS\.txt)$"`.
   Read the `[sweep] kaggle_ensembles: 48 runs, N already complete, M to run` header and
   the `[sweep] OK gnn_*` lines. Unzip `results.zip` to a scratch dir and list
   `results/gnn_*_s[1-4]/metrics.json`. Expect at most ~4 finished GNN runs per 11 h
   session in matrix order (`gnn_full_s2,s3,s4` = 9 h, then `gnn_reynolds_s1`), and the
   same runs in every GNN session because none was ever merged. If the header says
   `kaggle_dataeff`, the API is already serving the running session: go to step 4.
2. **Extract only the needed zip members over HTTP Range** (new script
   `kaggle/fetch_zip_members.py`, ~80 lines, stdlib + requests):
   - obtain the signed URL of `checkpoints.zip` exactly the way the CLI does
     (`KaggleApi().build_kaggle_client()` then
     `ApiListKernelSessionOutputRequest(user_name, kernel_slug)` then
     `kaggle.kernels.kernels_api_client.list_kernel_session_output(req).files[i].url`);
   - wrap the URL in a file-like `HttpRangeFile(url)` implementing `seek/tell/read`
     via `Range: bytes=a-b` requests (GCS signed URLs advertise `Accept-Ranges: bytes`;
     the CLI's own `download_file` relies on that). Refresh the URL if a request
     returns 403 (signed URLs expire);
   - `zipfile.ZipFile(HttpRangeFile(url))` reads the central directory from the tail,
     then `zf.extract(member)` for every `checkpoints/gnn_*_s[1-4]/best.pt` present.
     ~16.8 MB per run, so 4 runs = ~70 MB instead of 2.1 GB. Also grab
     `checkpoints/<rid>/last.pt` for any unfinished GNN run to resume it;
   - merge the matching `results/gnn_*` dirs from step 1's `results.zip`.
   The same script is the fallback for any legacy fat zip later (1.7).
3. **Fallback if Range is refused**: `curl -L -C - --retry 30 --retry-all-errors
   --retry-delay 20 -o checkpoints.zip "<url>"` (re-fetch the URL when it expires) and
   extract the members locally. Slow but resumable, unlike the CLI.
4. **Fallback if the output is no longer reachable via the API**: the web UI
   (kernel page, Versions, pick the GNN-ensemble version, Output tab) exposes per-file
   download links for older versions; the `results.zip` there is small, and
   `checkpoints.zip` can be fed to the same `curl -C -` loop. If that also fails,
   accept the loss: the runs are re-trained under the new driver (section 2 queue),
   cost = whatever step 1 found (roughly 4 runs, ~11 GPU-h).
5. After merging: `python -m scripts.run_uq --models gnn --seeds 0,1,2,3,4` for the
   splits that gained members, republish `geo-op-runs` with the new lean layout (1.5),
   commit `results/gnn_*`.

---

## 2. Self-sustaining sweep-chaining loop

### 2.1 The cycle (one script, idempotent): `kaggle/cycle.py`

Run it at the start of every orchestrator session and whenever a Kaggle session is
expected to have finished. It never launches while something is RUNNING.

```
for slug in queue (kaggle/queue.yaml, ordered):
    status = kaggle kernels status <owner>/<slug>
    RUNNING / QUEUED       -> print ETA (launch utc + 11 h from launches.jsonl); stop.
    COMPLETE / ERROR       -> if session not in pulled_sessions.jsonl:
                                  pull_results.py --kernel <owner>/<slug>   (1.5)
                                  (merge results; needed checkpoints; republish lean runs dataset)
                              then decide "done?" for this queue item:
                                  sweep.py --spec <specs> --dry-run  ->  "0 to run"  => done
                              done  -> mark item done in queue.yaml, continue to next item
                              !done -> relaunch same item: launch.py --sweep <specs...>
                                       --kernel-slug <slug> --runs-dataset <owner>/geo-op-runs; stop.
    never launched         -> launch as above; stop.
after any merge:
    python -m scripts.make_figures --outdir paper/figures
    python -m scripts.make_tables  --outdir paper/tables
    python -m pytest tests/test_viz.py -q
    update docs/RESULTS.md numbers touched by the merge (executor: Sonnet, mechanical)
    git add results/ paper/figures paper/tables docs/RESULTS.md kaggle/*.jsonl; git commit -m "[results] <slug> session <id>: <n> runs"
    gh auth switch -u cocomo069; git push origin main; gh auth switch back to the default account
```

`--dry-run` of the queue item's specs is the single source of truth for "done"; it is
exactly the skip logic the cloud session uses, so local and cloud never disagree.

Guardrails: one Kaggle session in flight at a time; `push_code.py` runs before every
launch (the driver reads code from `geo-op-code`, so any repo change, including the
driver fix itself, must be pushed first); `pull_results.py` refuses duplicate session
ids; the cycle prints and exits on any non-zero subprocess rather than continuing.

Cadence without user involvement: Kaggle sessions are ~11 h and the T4 quota is ~30 h
per week (resets weekly). The orchestrator checks `cycle.py` at every session start;
optionally the `/loop` skill or a scheduled task can run `cycle.py` every 2 h during a
work day. The user never has to touch Kaggle.

### 2.2 Ordered queue (`kaggle/queue.yaml`)

| # | slug | specs (in-session order) | runs left | GPU-h (measured rates) | notes |
|---|---|---|---|---|---|
| 1 | `geo-op-session` (legacy, in flight) | `kaggle_dataeff.yaml` | 45 (GNN rows first) | ~12.7 total: GNN 3 seeds x 3.3 h = 10 h; M2, M3 ~1.3 h each | will almost surely be truncated at 11 h. Pull per 1.7. Remainder moves to item 2 |
| 2 | `geo-op-dataeff` | `kaggle_dataeff.yaml`, then `kaggle_ablations.yaml` | data-eff remainder (~2 to 5 h) + 6 ablation runs (~4.4 h: M2 4 x 0.2 to 0.36 h, GNN lamF full 3.0 h + combined 0.2 h) | ~7 to 9 h | fits ONE session (multi-spec, 1.3). New driver from here on |
| 3 | `geo-op-gnnens` | `kaggle_gnn_ens_k3.yaml` (new spec: `gnn` x {full s2, reynolds s1, s2, aoa s1, s2, shape5 s1, s2}, minus whatever 1.8 recovered) | up to 7 | up to 16.5 h (full 3.0, reynolds 2 x 1.8, aoa 2 x 3.0, shape5 2 x 1.7) | K=3 for GNN on all four splits. Two sessions if nothing was recovered |
| 4 | `geo-op-gnnens` | `kaggle_gnn_ens_k5.yaml` (`gnn` x {full s3, s4, reynolds s3, s4, aoa s3, s4, shape5 s3, s4}) | 8 | ~19 h | **optional top-up to K=5**, only if quota allows before the v1 freeze (section 6). Not a v1 blocker |

Why K=3 first for the GNN (a change from PLAN_PHASE2): at the measured 1.7 to 3.0 h
per GNN run, the full K=5 remainder is ~35 GPU-h, more than a week of quota, for a
row of Table 3 and one line of Fig 7 that already exist at K=1. K=3 gives sigma > 0
(the `normalized` score activates), seed error bars for the GNN on Figs 4/5, and costs
16 h. `run_uq.py` reports `k` per file and the paper states "K=5 for M2/M3, K=3 for M1"
in the setup. Order items 3 and 4 so seeds 1 to 2 land on all splits before any seed 3.
The old `kaggle_ensembles.yaml` (48-run matrix) must NOT be relaunched: its GNN block is
split-major (all four seeds of `full` first), the wrong order for this budget.

Quota calendar (assuming the weekly window opened around 09-02): week 1 = item 1 (11 h)
+ item 2 (~9 h) + first half of item 3 (~10 h); week 2 = rest of item 3 (~6 h) + item 4
if chosen. Local CPU/GPU is never used for training.

### 2.3 What "merge" regenerates, every cycle

`make_figures` (fig4, fig5, fig12 gain seed bands as ensembles land; fig9 appears once
>= 3 sizes exist; fig6 to fig8 refresh from `results/uq`), `make_tables` (tab1 to tab3;
tab4 once section 4 adds it), `docs/RESULTS.md` sections 3 to 5 status flags, and the
`\pending{}` numbers in `paper/main.tex` (Sonnet, from the regenerated tables, never
hand-typed from memory).

---

## 3. UQ completion (CPU lane, minutes)

Trigger: any cycle that merges new `gnn_*_s[1-4]` checkpoints (recovery in 1.8, or
queue item 3/4).

1. `.venv/Scripts/python.exe -m scripts.run_uq --models gnn --seeds 0,1,2,3,4`
   (M2/M3 reports are already final at K=5; rerunning `--all` is harmless but wastes
   ~20 minutes of P2000 time). `run_uq` picks whatever seeds exist, so K=3 yields
   `gnn_<split>_k{1,3}{,_calfull}.json`; K=5 adds `k5`.
2. Delete the stale `results/uq/gnn_full_k2*.json` (superseded; `_largest_k` would
   ignore them anyway, but they clutter the committed set).
3. Regenerate fig6/7/8 + tab3; the figure/table code auto-selects the largest K and the
   `normalized` score.
4. Expected change and how to read it: the GNN row currently comes from a single member
   with the `absolute` score (no sigma). With K >= 3 the `normalized` score becomes
   available, so matched coverage on `reynolds`/`aoa` should move toward the M2/M3 rows
   (0.90 to 0.95) and interval width should become input-dependent (fig8 spreads).
   If the GNN row does NOT improve, that is itself reportable: graph-net members that
   agree with each other while being wrong under-estimate sigma, and conformal cannot
   repair a score that carries no information. Either way the C3 sentence ("conformal
   fixes calibration, not robustness") stands; the paper's calibration subsection
   (section 4) is written so only the numbers change.
5. Commit `results/uq/`, figures, tables; update RESULTS.md section 3.1 table (the
   "GNN (K=1)" row becomes K=3 or K=5).

---

## 4. Figures, tables and paper sections still open (CPU lane)

Executor: Sonnet for prose and mechanical figure code; Opus reviews numbers against
committed JSON. Nothing here needs the user. "NOW" = all data exists today.

### 4.1 Figures

| fig | file | what | data dependency | status | executor task |
|---|---|---|---|---|---|
| 1 | `fig01_schematic.pdf` | protocol overview: three model families feeding one evaluation harness (FSC, OOD splits, conformal, AL) | none | NOW | draw as a matplotlib/patches or TikZ diagram in a new `scripts/make_schematics.py` (pure drawing, no metrics); `main.tex` line 149 uncomment |
| 2 | `fig02_cp_profiles.pdf` | cp vs x/c for 4 sims (1 ID + reynolds + aoa + shape5 test sims), truth + 3 models (s0 `best.pt`) | core checkpoints, cache npz | NOW | new `scripts/render_examples.py` (loads `checkpoints/{model}_{split}_s0/best.pt` via `src.uq.ensembles.load_ensemble_checkpoints` with K=1, denormalizes p, plots per sim); CPU inference, seconds |
| 3 | `fig03_surface_fields.pdf` (rename from `fig03_car_surfaces`) | surface p on 2 airfoils (ID, combined): truth / Transolver prediction / abs error, colour along the contour | same as fig 2 | NOW | same script; **repurpose**: DrivAerNet++ is out of v1 (D-005), so the "3 cars" figure is replaced and the caption in `main.tex` line 502 rewritten. No 3D claim anywhere |
| 4, 5, 12 | exist | seed bands appear automatically as ensembles merge | ensembles | done (refresh each cycle) | none |
| 6, 7, 8 | exist (fig08 is written as `fig08_interval_width.pdf`; `main.tex` line 692 references `fig08_width_dists.pdf`) | | `results/uq` | done; GNN row refreshes per section 3 | fix the filename mismatch in `main.tex`, uncomment lines 675/681/692 |
| 9 | `fig09_data_efficiency.pdf` | log-log error vs n_train, 6 sizes, 3 models, seed bands, slope in caption | data-eff runs (queue items 1 to 2) | BLOCKED (days) | `fig9_data_efficiency` exists; after the merge verify x=700 points from `{model}_full_s{0,1,2}` and quote fitted slopes per model in the caption |
| 10 | `fig10_symmetry.pdf` | symmetry residual (and `antisym_cl_gap`) per model x split, grouped bars, log-y | `consistency.sym_residual` in all 18 core `metrics.json` (values 4.4 to 8.2 already tabulated in RESULTS.md 2.4) | NOW | add `fig10_symmetry(df, outdir)` to `make_figures.py` next to fig5; wire into `main()`; uncomment `main.tex` line 619 |
| 11 | `fig11_active_learning.pdf` | panel (a) acquisition-score landscape over the 630-case pool (score vs AoA, coloured by Re, the 3 x 8 selections marked per arm); panel (b) surrogate error on Fluent-verified cases by arm | (a) `results/active/acquisition_ranking.csv` NOW; (b) Fluent, user-gated | (a) NOW, (b) BLOCKED (user CPU) | `fig11_active_pool(outdir, results)` in `make_figures.py` reading the CSV; panel (b) is added by the Fluent phase. For v1 the figure is panel (a) with the caption stating verification is pending |

### 4.2 Tables

| table | file | contents | dependency | status |
|---|---|---|---|---|
| 1, 2, 3 | exist | | | done; refresh per cycle |
| 4 | `tab4_ablations.{tex,md}` | rows = (axis, setting): M2 conditioning {sdf (core), mask, sdf+normals} on {full, shape5}; M1 lambda_F {0 (core), 1.0} on {full, combined}; ensemble size K {1, 3, 5}; conformal score {normalized, absolute}; field aggregation {max, quantile 0.9}. cols = p rel-L2, cd_int MAE, fsc_rel_cd, coverage@0.9 (where a UQ record exists) | Tier-0 rows (K, score, aggregation) from `results/uq` NOW; Tier-1 rows from the 6 tagged runs (queue item 2) | half NOW, half BLOCKED (days). Add `table4_ablations(df, results, outdir)` to `make_tables.py`; the tagged runs are matched to their core counterpart by `(model, split, seed)` with `tag` stripped. Rewrite the `main.tex` line 746 caption to the lean v1 set (drop lambda in {0.01, 0.1}, K=10, volume-then-trace, augmentation) |
| 5 | `tab5_fluent` | Fluent verification cases | Fluent runs | BLOCKED (user CPU). For v1: replace with the selected-case table (3 arms x 8: NACA code, Re, AoA, sigma_CD, FSC, acquisition score) generated from `results/active/selection_summary.json`, labelled "selected, verification pending" |

### 4.3 Paper sections (`paper/main.tex`)

| section | line | dependency | status | notes |
|---|---|---|---|---|
| 8.4 Calibration | 627 | tab3 + fig6 to 8 (exist) | NOW | replace the `\todo` with the matched vs transfer story from RESULTS.md 3.1 to 3.2; state K per model; numbers from `tab3_calibration.md`; re-touch after section 3 |
| Data efficiency | 700 | fig9 | BLOCKED (days) | the cost half (fig12) is already written; keep the `\pending` numbers until tab1 regenerates |
| Ablations | 741 | tab4 | half NOW | write the Tier-0 paragraph (K, score, aggregation) now; Tier-1 paragraph after queue item 2 |
| 9 Active learning | 762 | `results/active/*`, `fluent/cases_to_run.json` | NOW (selection) | pool definition (630 = NACA 4/5-digit x Re {2..7}e6 x AoA {-6,0,6,12,18}), scorer `transolver_full_k5`, beta = gamma = 1, three arms, the selected-case table; the solver-offset and falsifiable-claim paragraphs stay but say "pending". Note honestly that the acquisition arm picked eight AoA = 18 deg cases (post-stall), which is what a shift-sensitive score should do and also the hardest regime for steady RANS |
| Discussion | 830 | | mostly written | update the two `\todo`s: the "single seed" limitation becomes "K=5 (M2/M3) and K=3 (M1) seeds"; drop the "calibration not yet measured" bullet; add the AoA breakout only if done (it is optional; not a v1 blocker) |
| Conclusion | 927 | all results | NOW (draft), final pass at freeze | half a page: protocol not leaderboard; the three headline sentences of RESULTS.md 8; what a practitioner does differently |
| Reproducibility statement | 935 | | NOW | MIT code, split manifests in `data/splits/`, reimplementation note (D-002), AirfRANS ODbL obligation, no processed copy redistributed, DOI at release (Zenodo happens at public release per D-018) |
| Acknowledgements | 943 | user | user writes (names rule) | leave one line for the user |
| App. A hyperparameters | 951 | `configs/*.yaml` | NOW | one table per model from the YAMLs (widths, depths, k, grid, modes, slices, lr, schedule, batch 16, 400 epochs, clip 1.0, seeds) |
| App. B CFD setup | 960 | `docs/FLUENT_PLAN.md` | NOW (design), results pending | import y+ sizing, C-grid topology, GCI arithmetic, the six replicated cases plan; mark measured values pending |
| App. C additional figures | 971 | figs 2/10 extras | after 4.1 | per-split cp profiles, acquisition sensitivity over beta/gamma (a 3 x 3 grid of selection overlap from `run_active.py --beta/--gamma`, minutes) |
| App. D compute accounting | 979 | `metrics.json::train_time_s`, `session.log`s | NOW | sum train_time_s per phase (core 15.4 h, M2/M3 ensembles ~10 h, ...), add the sessions lost to restarts (the re-trained GNN ensemble sessions count, honestly), local P2000 inference hours |

Front-half hygiene at the same time: `\pending{}` numbers in sections 5 to 8 replaced
from the regenerated tables; `fig08` filename mismatch; Fig 3 caption; Table 4 caption.

---

## 5. Fluent verification (CPU-gated, deferred, D-007)

Restated: nothing in the Fluent lane runs until the user says the CPU is free. When
they do, the handoff is complete and needs no design work:

- Case list: `fluent/cases_to_run.json` (27 cases: `gridstudy_naca0012_re3e6_a5_L{1,2,3}`
  first, then 3 arms x 8 AL cases from `results/active/selection_summary.json`).
- Render: `.venv/Scripts/python.exe fluent/make_cases.py --write-mesh` (numpy only)
  regenerates `fluent/cases/<case>/` (mesh, `_sa.jou`, `_sst.jou`, params) and
  `fluent/cases/RUNBOOK.md`.
- Execute per RUNBOOK step 0 to 3: keyword probe once, then one solve at a time via
  `mcp__ansys__fluent_run_journal` + `ansys_job_status` + `fluent_check_convergence`.
  Two turbulence models x 27 cases at ~43k cells: roughly 10 to 20 min each, ~10 to 18 h
  of CPU total, resumable case by case.
- Then: extract CD/CL per case, compute the AirfRANS-replication offset (6 test sims,
  FLUENT_PLAN section 1), GCI on the trio, the "were the selected cases harder than
  random" comparison, fig11 panel (b), tab5, appendix B numbers.
- Known risk to state up front: all 8 acquisition-arm cases sit at AoA = 18 deg
  (post-stall). Steady RANS may not meet the RUNBOOK convergence criteria there; a
  non-converged case is reported as such, not tuned. Do not alter the selection.

If the green light arrives before the v1 freeze, fold the results in; otherwise v1
ships with the selection-as-protocol version (section 4) and Fluent becomes v2.

---

## 6. arXiv v1 scope and definition of done

### 6.1 Scope (minimal complete, unchanged from PLAN_PHASE2 except the GNN K)

Core grid (Tables 1 to 2, Figs 4, 5, 10, 12) + full C3 calibration (Table 3, Figs 6 to 8;
K=5 M2/M3, K>=3 M1) + data efficiency (Fig 9) + ablations Table 4 (Tier 0 + Tier 1) +
AL selection as protocol (section 9, Fig 11a, selected-case table) + schematics (Figs 1
to 3) + appendices A, B (design), D. **Not in v1**: Fluent results, DrivAerNet++, GNN
K=5 top-up, lambda_F sweep, AoA stall breakout, combined-split ensembles.

### 6.2 Definition of done (the orchestrator STOPS when every box is ticked)

Infra
- [ ] `session_driver.py`, `sweep.py`, `launch.py`, `pull_results.py`, `cycle.py`
      merged with tests green (`pytest tests/ -q`); one full cycle exercised end to end
      on queue item 2 (pull of a lean export, republish of a ~10 MB runs dataset).
- [ ] GNN-ensemble recovery attempted and its outcome logged in PROGRESS.md.

Results (all committed under `results/`)
- [ ] 45 data-eff `metrics.json` present (`sweep --spec kaggle_dataeff.yaml --dry-run`
      prints `0 to run`).
- [ ] 6 ablation `metrics.json` present (same check on `kaggle_ablations.yaml`).
- [ ] GNN K>=3 on {full, reynolds, aoa, shape5}; `results/uq/gnn_*_k3*.json` (or k5).
- [ ] `results/active/` + `fluent/cases_to_run.json` + rendered `fluent/cases/` committed.

Figures and tables (all regenerated from committed JSON by `make_figures`/`make_tables`,
none hand-edited)
- [ ] figs 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11(a), 12 exist as PDF in `paper/figures/`.
- [ ] tabs 1, 2, 3, 4 exist; tab5 replaced by the selected-case table.
- [ ] `docs/RESULTS.md` status flags all complete except section 6 (Fluent, pending).

Paper
- [ ] `grep -c '\\todo' paper/main.tex` = 0 in the main body (appendix B/C may keep
      explicitly "pending Fluent" sentences, no `\todo` macro); `\pending` count = 0;
      `\figph` count = 0 in the main body.
- [ ] Every number in the text traceable to a committed `paper/tables/*.md` or
      `results/**/*.json` (Opus spot-checks 10 numbers at random).
- [ ] `pdflatex` + `bibtex` + `pdflatex` x2 clean: zero undefined refs/citations.
- [ ] One internal read-through pass (Sonnet drafts, Opus reviews for overclaiming
      against the Discussion's own limitations list).
- [ ] Global style rules applied to every generated document: no em dashes, no names
      inside the report body (author block is the user's to fill).

Handoff to the user (the only user actions in the whole phase)
- [ ] Message with: the compiled PDF path, the DoD checklist ticked, the two optional
      items (Fluent green light; GNN K=5 top-up), and the request for sign-off before any
      public push (GITHUB_UPLOAD_RULES staging copy + Zenodo happen only after sign-off,
      D-018).

Stop rule: when the DoD is ticked, the orchestrator does NOT start Fluent, DrivAerNet++,
the K=5 top-up, or any Tier-2 ablation. It reports and waits.

---

## 7. Execution order (single sequence; CPU and GPU lanes interleave)

| # | lane | action | gate |
|---|---|---|---|
| 0 | CPU, now | Section 1.8 steps 1 to 2: discover the reachable session, Range-extract GNN `best.pt`s, merge, `run_uq --models gnn`. Time-critical: the running data-eff session will hide the previous output when it completes | outcome logged |
| 1 | CPU | Commit the AL agent's files once it reports (`run_active.py`, tests, `results/active/`, `cases_to_run.json`, rendered `fluent/cases/`) | tests green |
| 2 | CPU | Implement section 1.2 to 1.6 (driver, sweep multi-spec, launch, pull_results, tests) + `cycle.py` + `queue.yaml`; `push_code.py` | tests green, code dataset pushed |
| 3 | GPU | When `geo-op-session` completes: pull per 1.7 (small files only), merge data-eff results, republish lean `geo-op-runs`, launch queue item 2 (`geo-op-dataeff`: data-eff remainder + ablations) with the new driver | fig9 renders if >= 3 sizes landed |
| 4 | CPU (Sonnet) | Section 4 NOW items: figs 1, 2, 3, 10, 11a; tab4 Tier-0 half; sections 8.4, 9, conclusion, reproducibility, appendices A, B, D; filename/caption fixes | main.tex compiles, `\todo` count falling |
| 5 | GPU | Item 2 completes: pull, merge, tab4 full, fig9 final, paper data-eff + ablations text; launch item 3 (`geo-op-gnnens`, K=3 spec) | 45 + 6 runs committed |
| 6 | GPU | Item 3 completes (1 to 2 sessions): pull, `run_uq --models gnn`, refresh figs 6 to 8 + tab3, RESULTS.md, paper calibration numbers | GNN K>=3 |
| 7 | CPU | v1 assembly: regenerate everything, `\pending`/`\todo`/`\figph` to zero, read-through, PDF | DoD 6.2 |
| 8 | user | Sign-off message (and, whenever they choose, the Fluent green light; the K=5 top-up runs only if quota is idle before sign-off) | stop |

Total remaining Kaggle GPU for v1: ~2 to 5 h (data-eff tail) + ~4.4 h (ablations) +
~16.5 h (GNN K=3) = **~25 GPU-h, about one weekly quota**; the optional K=5 top-up is
another ~19 h. Local machine: inference and figures only, CPU-light throughout.
