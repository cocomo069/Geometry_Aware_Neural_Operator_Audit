"""Assemble the arXiv submission package from paper/.

    .venv/Scripts/python.exe -m scripts.make_arxiv [--variant main|main_twocol]

arXiv compiles the source itself, so the package contains: the wrapper .tex,
body.tex, macros.tex, the generated tables, the PDF figures actually
\\includegraphics'd by the body, and the pre-built .bbl (arXiv runs pdflatex
but not bibtex reliably with custom setups; shipping the .bbl is the standard
safe route -- refs.bib is included too for completeness).

Output: paper/arxiv_<variant>.zip plus a printed manifest. The zip is flat
except for figures/ and tables/, matching the \\input/\\includegraphics paths.
"""
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

PAPER = Path("paper")


def referenced_figures(body: str) -> list[str]:
    return sorted(set(re.findall(r"\\includegraphics\[[^\]]*\]\{(figures/[^}]+)\}", body)))


def referenced_tables(body: str) -> list[str]:
    return sorted(set(t + ".tex" for t in re.findall(r"\\input\{(tables/[^}]+)\}", body)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="main", choices=["main", "main_twocol"])
    args = ap.parse_args(argv)

    body = (PAPER / "body.tex").read_text(encoding="utf-8")
    files = [f"{args.variant}.tex", "body.tex", "macros.tex", "refs.bib",
             f"{args.variant}.bbl"]
    files += referenced_tables(body)
    files += referenced_figures(body)

    missing = [f for f in files if not (PAPER / f).is_file()]
    if missing:
        for f in missing:
            print(f"MISSING: paper/{f}")
        if f"{args.variant}.bbl" in missing:
            print("  -> run latexmk first so the .bbl exists")
        return 1

    out = PAPER / f"arxiv_{args.variant}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(PAPER / f, arcname=f)
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB, {len(files)} files):")
    for f in files:
        print(f"  {f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
