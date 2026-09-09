"""The single matplotlib style for every figure in the paper.

Design rules, chosen once here so no figure re-litigates them:

* **Backend is always ``Agg``.**  Figures are produced headless in CI/tests and
  on a machine whose GPU must stay free (CONTEXT.md section 12).
* **One colour per model family, fixed across all twelve figures.**  The
  palette is Okabe--Ito, which is safe for deuteranopia/protanopia and stays
  distinguishable in greyscale print.  ``gnn``/``sdf_fno``/``transolver`` get
  the three most saturated hues, the two baselines get purple and grey so they
  read as "reference lines" rather than competitors.
* **Split colours run cool to warm with increasing shift** so Figure 4 and
  Figure 7 are legible even without reading the legend.
* Serif text (matching the paper body font), 300 dpi, vector PDF **and** a PNG
  mirror for the README and for eyeballing during drafting.

Nothing here reads data; nothing in :mod:`src.viz.data` draws.  That split is
what makes the figure functions in ``scripts/make_figures.py`` pure and
testable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # noqa: E402  -- must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

__all__ = [
    "OKABE_ITO",
    "MODEL_ORDER",
    "MODEL_COLORS",
    "MODEL_MARKERS",
    "MODEL_LABELS",
    "SPLIT_ORDER",
    "SPLIT_COLORS",
    "SPLIT_LABELS",
    "OOD_AXES",
    "FALLBACK_COLORS",
    "apply_style",
    "model_color",
    "model_marker",
    "model_label",
    "split_color",
    "split_label",
    "sort_models",
    "sort_splits",
    "model_legend",
    "save_figure",
    "figure_path",
    "annotate_todo",
]

#: Okabe--Ito colourblind-safe qualitative palette (Okabe & Ito 2008).
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#7F7F7F",
}

#: Canonical model ordering: the three surrogates, then the two baselines.
MODEL_ORDER: tuple[str, ...] = ("gnn", "sdf_fno", "transolver", "ridge", "constant")

#: FROZEN across every figure in the paper.  Do not reassign per-figure.
MODEL_COLORS: dict[str, str] = {
    "gnn": OKABE_ITO["blue"],
    "sdf_fno": OKABE_ITO["vermillion"],
    "transolver": OKABE_ITO["green"],
    "ridge": OKABE_ITO["purple"],
    "constant": OKABE_ITO["grey"],
}

MODEL_MARKERS: dict[str, str] = {
    "gnn": "o",
    "sdf_fno": "s",
    "transolver": "^",
    "ridge": "D",
    "constant": "v",
}

MODEL_LABELS: dict[str, str] = {
    "gnn": "M1 GNN",
    "sdf_fno": "M2 SDF-FNO",
    "transolver": "M3 Transolver",
    "ridge": "Ridge",
    "constant": "Constant",
}

#: Splits ordered by increasing distribution shift (CONTEXT.md section 5).
SPLIT_ORDER: tuple[str, ...] = (
    "full",
    "scarce",
    "reynolds",
    "aoa",
    "shape5",
    "combined",
)

#: Cool -> warm with increasing shift; also Okabe--Ito, so still CVD-safe.
SPLIT_COLORS: dict[str, str] = {
    "full": OKABE_ITO["blue"],
    "scarce": OKABE_ITO["sky"],
    "reynolds": OKABE_ITO["green"],
    "aoa": OKABE_ITO["orange"],
    "shape5": OKABE_ITO["vermillion"],
    "combined": OKABE_ITO["purple"],
}

SPLIT_LABELS: dict[str, str] = {
    "full": "full (in-dist.)",
    "scarce": "scarce",
    "reynolds": "Reynolds shift",
    "aoa": "AoA shift",
    "shape5": "shape family",
    "combined": "combined",
}

#: The OOD axes that get one panel each in Figure 4 (spec section 5.7).
OOD_AXES: tuple[str, ...] = ("reynolds", "aoa", "shape5", "combined")

#: Used for any model/split name we have not seen before, so an unexpected run
#: still plots (in a colour that is visibly "not one of ours").
FALLBACK_COLORS: tuple[str, ...] = (
    OKABE_ITO["yellow"],
    OKABE_ITO["sky"],
    OKABE_ITO["black"],
)

_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "savefig.transparent": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    # Serif to match the paper body; DejaVu Serif ships with matplotlib so this
    # never falls back to a warning-generating missing family.
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral", "serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "errorbar.capsize": 2.0,
    "figure.autolayout": False,
}


def apply_style() -> None:
    """Install the paper rcParams.  Idempotent; call once per process."""
    plt.rcParams.update(_RC)


def _fallback(name: str) -> str:
    return FALLBACK_COLORS[hash(str(name)) % len(FALLBACK_COLORS)]


def model_color(model: str) -> str:
    """Colour for a model family; stable across every figure."""
    return MODEL_COLORS.get(str(model), _fallback(model))


def model_marker(model: str) -> str:
    return MODEL_MARKERS.get(str(model), "P")


def model_label(model: str) -> str:
    return MODEL_LABELS.get(str(model), str(model))


def split_color(split: str) -> str:
    return SPLIT_COLORS.get(str(split), _fallback(split))


def split_label(split: str) -> str:
    return SPLIT_LABELS.get(str(split), str(split))


def sort_models(models: Iterable[str]) -> list[str]:
    """Canonical order, unknown names appended alphabetically at the end."""
    seen = list(dict.fromkeys(str(m) for m in models))
    known = [m for m in MODEL_ORDER if m in seen]
    rest = sorted(m for m in seen if m not in MODEL_ORDER)
    return known + rest


def sort_splits(splits: Iterable[str]) -> list[str]:
    seen = list(dict.fromkeys(str(s) for s in splits))
    known = [s for s in SPLIT_ORDER if s in seen]
    rest = sorted(s for s in seen if s not in SPLIT_ORDER)
    return known + rest


def model_legend(models: Sequence[str], *, with_marker: bool = True) -> list[Line2D]:
    """Proxy handles so every panel can share one legend."""
    handles = []
    for m in sort_models(models):
        handles.append(
            Line2D(
                [],
                [],
                color=model_color(m),
                marker=model_marker(m) if with_marker else None,
                label=model_label(m),
            )
        )
    return handles


def figure_path(outdir: str | Path, name: str, ext: str) -> Path:
    return Path(outdir) / f"{name}.{ext.lstrip('.')}"


def save_figure(
    fig: Figure,
    name: str,
    outdir: str | Path,
    *,
    formats: Sequence[str] = ("pdf", "png"),
    close: bool = True,
) -> list[Path]:
    """Write ``<outdir>/<name>.{pdf,png}`` at 300 dpi and return the paths.

    ``name`` is the bare stem, e.g. ``"fig04_error_vs_shift"``; the caller owns
    the ``figNN_slug`` convention so the paper's ``\\includegraphics`` paths
    stay predictable.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in formats:
        path = figure_path(out, name, ext)
        fig.savefig(path)
        written.append(path)
    if close:
        plt.close(fig)
    return written


def panel_letter(fig: Figure, letter: str) -> None:
    """Stamp "(a)" style panel lettering into the top-left corner of a figure.

    Used by figures that are placed side by side as one composite float in the
    paper (REVTeX has no subcaption support), so the letter must live inside
    the image itself, per the research-figures skill.
    """
    fig.text(0.02, 0.98, f"({letter})", ha="left", va="top",
             fontsize=9, fontweight="bold")


def annotate_todo(ax, message: str) -> None:
    """Stamp an empty panel with why it is empty, instead of shipping a lie."""
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=8,
        color=OKABE_ITO["grey"],
        wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
