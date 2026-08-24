"""Plotting + results-access helpers for the paper figures and tables (A7).

Two concerns, two modules:

* :mod:`src.viz.style` -- the single matplotlib style for every figure in the
  paper, a colourblind-safe palette with **one fixed colour per model family**
  so a reader can learn the legend once, and PDF+PNG save helpers at 300 dpi.
* :mod:`src.viz.data`  -- read ``results/*/metrics.json``, ``per_sim.csv`` and
  ``results/uq/*.json`` into tidy pandas frames.  Robust to missing/partial
  runs: a run that has not finished yet is simply absent from the frame.
* :mod:`src.viz.shift` -- the shift magnitude ``Delta`` of spec section 5.7,
  the x-axis of the paper's central figure (Figure 4).

Everything here is import-time side-effect free apart from selecting the
``Agg`` backend (CONTEXT.md section 12: no GUI, no GPU, CPU-light).
"""

from __future__ import annotations

from . import data, shift, style  # noqa: F401

__all__ = ["style", "data", "shift"]
