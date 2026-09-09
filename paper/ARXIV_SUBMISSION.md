# arXiv submission runbook

Everything below is staged; the actual submission needs your arXiv login, so the
last mile is yours. Rebuild + repackage at any time with:

```bash
.venv/Scripts/python.exe -m scripts.make_tables
.venv/Scripts/python.exe -m scripts.make_figures
.venv/Scripts/python.exe -m scripts.render_examples
cd paper && latexmk -pdf single_column.tex && latexmk -pdf two_column.tex && cd ..
.venv/Scripts/python.exe -m scripts.make_arxiv --variant single_column
```

## What to upload

- **`paper/arxiv_single_column.zip`** is the submission package (REVTeX 4.2
  `reprint,onecolumn` build, the canonical arXiv version): wrapper +
  `body.tex` + `preamble_shared.tex` + `macros.tex` + the generated tables +
  the 13 PDF figures + `refs.bib` + a prebuilt `single_column.bbl` so arXiv's
  pdflatex pass needs no bibtex. Verified to compile standalone from a clean
  directory with two pdflatex passes (arXiv ships REVTeX 4.2).
- `paper/pdf/paper_two_column.pdf` is the two-column rendering of the same
  body; keep it for the repo/README or reviewer copies. arXiv itself takes one
  source package, and single-column is the one to submit.

## Suggested metadata (edit as you see fit)

- **Title**: Geometry-Aware Neural Operator Surrogates for External
  Aerodynamics: Physics Consistency, Out-of-Distribution Generalization and
  Calibrated Uncertainty
- **Primary category**: `physics.flu-dyn` (Fluid Dynamics).
  **Cross-list**: `cs.LG`. (The reverse, primary cs.LG with a flu-dyn
  cross-list, is also defensible; primary flu-dyn matches where the
  contribution lands, an evaluation protocol for CFD surrogates.)
- **Comments field**: "26 pages, 11 figures, 11 tables. Code and data
  manifests: https://github.com/cocomo069/Geometry_Aware_Neural_Operator_Audit"
- **License**: arXiv's default non-exclusive license is the usual choice
  unless you want CC BY 4.0.
- **Abstract**: paste from `paper/body.tex` (the `abstract` block), minus the
  LaTeX markup arXiv does not render (`\FSC` becomes FSC, `$...$` math is fine).

## Before you press submit

1. **Make the GitHub repo public** (or accept that the paper links a private
   repo until you flip it). The paper's reproducibility statement and the
   author footnote both point at
   `github.com/cocomo069/Geometry_Aware_Neural_Operator_Audit`.
   Per your own upload rules: `gh auth switch -u cocomo069` first, and check
   `D:\Personal Projects\GITHUB_UPLOAD_RULES.md` for whether the current
   private repo layout is the one you want public, or whether a clean staging
   copy goes out instead (HANDOFF.md notes the public copy was planned as a
   one-time restructure).
2. **Author block**: the paper carries `T. Amin` with your email, as it
   already did in the repo. Change to your full name in the author block at
   the top of `paper/body.tex` if you prefer it spelled out, then rebuild +
   repackage.
3. arXiv account must be endorsed for the chosen primary category; a first
   submission in `physics.flu-dyn` may require endorsement.

## Decisions (locked in by coco, 2026-09-07)

1. Repo: made public as-is (note: the pre-scrub history still contains the
   client `gh` account name in old doc versions; ask for a history rewrite if
   that matters).
2. Author line: "T. Amin", no affiliation. No rebuild needed.
3. Primary category `physics.flu-dyn`, cross-list `cs.LG`.
4. License: arXiv default non-exclusive.
