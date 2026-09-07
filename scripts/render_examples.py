"""Render real predicted-vs-truth surface pressure figures from trained checkpoints.

Pure CPU inference, no training and no Kaggle: loads a model config plus one or
more seed checkpoints, builds the split's *test* dataset, runs a forward pass on
one representative test simulation, and plots surface pressure coefficient
``c_p = p / (0.5 * U_inf**2)`` -- ``p`` is the OpenFOAM kinematic pressure
(rho already divided out, docs/DATA_NOTES.md A1.2), so this is the standard
convention with rho = 1.

Feeds paper Figures 2 and 3 (05_..._uq.md section 8; PLAN_PHASE3.md section 4.1):

    fig 2  c_p profile, prediction vs truth (+ ensemble band where K>1 seeds
           exist), one panel per split: full (ID), reynolds, aoa, shape5.
    fig 3  surface field view (repurposing the "3 cars" slot -- DrivAerNet++ is
           out of v1, D-005): truth / prediction / |error| c_p scattered on the
           airfoil contour, colour = value, for one in-distribution and one
           combined-shift simulation.

Every function is gated exactly like scripts/make_figures.py: missing
checkpoints or cache -> a printed TODO and ``None``, never an exception.

    .venv/Scripts/python.exe -m scripts.render_examples --outdir paper/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.viz import style  # noqa: E402


def _save(fig: plt.Figure, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = style.figure_path(outdir, name, "pdf")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(style.figure_path(outdir, name, "png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return pdf


def _load_sim(
    model: str,
    split: str,
    seeds: list[int],
    *,
    configs_root: str = "configs",
    checkpoints_root: str = "checkpoints",
    device: str = "cpu",
    sim_index: int = 0,
) -> dict[str, Any] | None:
    """Run one representative test sim through every available seed checkpoint.

    Returns ``None`` (with a printed reason) if the config, cache or every
    checkpoint for ``(model, split)`` is unavailable -- this is the gating
    contract every caller relies on.  On success, returns physical-unit
    arrays: ``x, y`` (surface coordinates, chord = 1 so ``x`` is ``x/c``),
    ``normal_y`` (sign gives upper/lower surface), ``cp_true``,
    ``cp_pred_mean``, ``cp_pred_std`` (zero when only one seed is available),
    ``sim_name``, ``n_members``.
    """
    ckpts = [
        Path(checkpoints_root) / f"{model}_{split}_s{s}" / "best.pt" for s in seeds
    ]
    ckpts = [c for c in ckpts if c.is_file()]
    if not ckpts:
        print(f"  [render_examples] no checkpoint for {model}_{split}_s* under {checkpoints_root}")
        return None

    try:
        import torch

        from scripts.train import build_datasets, build_model, load_checkpoint
        from src.data.airfrans_loader import collate
        from src.utils.config import load_config

        cfg = load_config(Path(configs_root) / f"{model}.yaml", [f"data.split={split}"])
        data = build_datasets(cfg)
        test_ds = data.get("test")
        if test_ds is None or len(test_ds) <= sim_index:
            print(f"  [render_examples] {model}_{split}: no usable test sim at index {sim_index}")
            return None

        item = test_ds[sim_index]
        batch = collate([item])

        pos = test_ds.denormalize("surf_pos", item["surf_pos"]).numpy()
        normal = test_ds.denormalize("surf_normal", item["surf_normal"]).numpy()
        u_inf = float(item["cond_phys"][0].item())
        q = 0.5 * u_inf**2
        cp_true = item["surf_p_phys"].numpy().reshape(-1) / q

        preds = []
        for ckpt in ckpts:
            net = build_model(cfg)
            load_checkpoint(ckpt, net, map_location=device, restore_rng=False)
            net.eval()
            with torch.no_grad():
                out = net(batch)
            p_phys = test_ds.denormalize("surf_p", out["p"]).reshape(-1).numpy()
            preds.append(p_phys / q)
        preds = np.stack(preds, axis=0)
        cp_mean = preds.mean(axis=0)
        cp_std = preds.std(axis=0) if preds.shape[0] > 1 else np.zeros_like(cp_mean)

        return {
            "x": pos[:, 0],
            "y": pos[:, 1],
            "normal_y": normal[:, 1],
            "cp_true": cp_true,
            "cp_pred_mean": cp_mean,
            "cp_pred_std": cp_std,
            "sim_name": str(item.get("sim_name", "")),
            "n_members": preds.shape[0],
        }
    except Exception as exc:  # pragma: no cover - defensive, mirrors the gating pattern
        print(f"  [render_examples] {model}_{split}: inference failed ({type(exc).__name__}: {exc})")
        return None


def _profile_panel(ax, sim: dict[str, Any], color: str, *, band: bool) -> None:
    """One c_p-vs-x/c panel: truth (black) vs prediction (colour), upper/lower split."""
    x, cp_t, cp_m, cp_s = sim["x"], sim["cp_true"], sim["cp_pred_mean"], sim["cp_pred_std"]
    upper = sim["normal_y"] >= 0
    for mask, ls in ((upper, "-"), (~upper, "--")):
        if not mask.any():
            continue
        order = np.argsort(x[mask])
        xs = x[mask][order]
        ax.plot(xs, cp_t[mask][order], color="black", lw=1.3, ls=ls,
                label="truth" if ls == "-" else None)
        ax.plot(xs, cp_m[mask][order], color=color, lw=1.3, ls=ls,
                label="prediction" if ls == "-" else None)
        if band:
            lo = (cp_m[mask] - cp_s[mask])[order]
            hi = (cp_m[mask] + cp_s[mask])[order]
            ax.fill_between(xs, lo, hi, color=color, alpha=0.22, linewidth=0)
    # NOTE: do NOT call ax.invert_yaxis() here. With sharey=True the inversion
    # toggles once per panel, so an even panel count silently cancelled it and
    # the published figure ended up NOT inverted despite its axis label. The
    # caller inverts the shared axis exactly once.
    ax.set_xlabel("$x/c$")
    ax.grid(True, alpha=0.3)


def fig2_cp_profiles(
    outdir: Path,
    *,
    model: str = "transolver",
    splits: tuple[str, ...] = ("full", "reynolds", "aoa", "shape5"),
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    configs_root: str = "configs",
    checkpoints_root: str = "checkpoints",
) -> Path | None:
    """c_p, prediction vs truth, one panel per split, ensemble band where K>1."""
    sims = {}
    for split in splits:
        sim = _load_sim(model, split, list(seeds),
                         configs_root=configs_root, checkpoints_root=checkpoints_root)
        if sim is not None:
            sims[split] = sim
    if not sims:
        print("TODO fig2: no checkpoints/cache available for any requested split")
        return None

    present = [s for s in splits if s in sims]
    fig, axes = plt.subplots(1, len(present), figsize=(3.4 * len(present), 3.4), sharey=True)
    if len(present) == 1:
        axes = [axes]
    for ax, split in zip(axes, present):
        sim = sims[split]
        _profile_panel(ax, sim, style.split_color(split), band=sim["n_members"] > 1)
        k = sim["n_members"]
        ax.set_title(f"{style.split_label(split)}\n(K={k})", fontsize=8)
    # Invert the shared c_p axis exactly once (aerodynamics convention,
    # suction up); see the note in _profile_panel.
    if not axes[0].yaxis_inverted():
        axes[0].invert_yaxis()
    axes[0].set_ylabel(r"$c_p$ (suction up)")
    handles = [
        plt.Line2D([], [], color="black", lw=1.3, label="truth"),
        plt.Line2D([], [], color=style.model_color(model), lw=1.3, label="prediction ($\\pm 1$ std)"),
        plt.Line2D([], [], color="black", lw=1.3, ls="-", label="upper surface"),
        plt.Line2D([], [], color="black", lw=1.3, ls="--", label="lower surface"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    return _save(fig, outdir, "fig02_cp_profiles")


def fig3_surface_fields(
    outdir: Path,
    *,
    model: str = "transolver",
    splits: tuple[str, ...] = ("full", "combined"),
    seed: int = 0,
    configs_root: str = "configs",
    checkpoints_root: str = "checkpoints",
) -> Path | None:
    """Truth / prediction / |error| c_p scattered on the airfoil contour.

    Replaces the DrivAerNet++ "three cars" slot (out of v1 scope, D-005): two
    AirfRANS simulations (one in-distribution, one on the hardest shift,
    ``combined``), single seed (no ensemble claim here -- that is Figure 2's job).
    """
    rows = []
    for split in splits:
        sim = _load_sim(model, split, [seed],
                         configs_root=configs_root, checkpoints_root=checkpoints_root)
        if sim is not None:
            rows.append((split, sim))
    if not rows:
        print("TODO fig3: no checkpoint/cache available for any requested split")
        return None

    # Compact layout: truth and prediction share one colorbar per row (they are
    # on the same scale by construction), the error panel keeps its own. The
    # old version drew three tall colorbars per row -- two of them identical --
    # and left most of the canvas as whitespace.
    fig, axes = plt.subplots(len(rows), 3, figsize=(9.0, 1.7 * len(rows)),
                             squeeze=False, constrained_layout=True)
    for ri, (split, sim) in enumerate(rows):
        x, y = sim["x"], sim["y"]
        cp_t, cp_m = sim["cp_true"], sim["cp_pred_mean"]
        err = np.abs(cp_m - cp_t)
        vmin, vmax = float(min(cp_t.min(), cp_m.min())), float(max(cp_t.max(), cp_m.max()))
        panels = ((cp_t, "truth ($c_p$)", "coolwarm", vmin, vmax),
                  (cp_m, "prediction ($c_p$)", "coolwarm", vmin, vmax),
                  (err, r"$|$error$|$", "inferno", 0.0, float(err.max()) or 1.0))
        scs = []
        for ci, (vals, title, cmap, lo, hi) in enumerate(panels):
            ax = axes[ri][ci]
            sc = ax.scatter(x, y, c=vals, cmap=cmap, vmin=lo, vmax=hi, s=7, linewidths=0)
            scs.append(sc)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if ri == 0:
                ax.set_title(title, fontsize=9)
            if ci == 0:
                ax.set_ylabel(style.split_label(split), fontsize=9)
        fig.colorbar(scs[0], ax=[axes[ri][0], axes[ri][1]], fraction=0.05,
                     pad=0.02, shrink=0.85).ax.tick_params(labelsize=6)
        fig.colorbar(scs[2], ax=axes[ri][2], fraction=0.05,
                     pad=0.02, shrink=0.85).ax.tick_params(labelsize=6)
    return _save(fig, outdir, "fig03_surface_fields")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="paper/figures")
    ap.add_argument("--configs-root", default="configs")
    ap.add_argument("--checkpoints-root", default="checkpoints")
    ap.add_argument("--model", default="transolver")
    args = ap.parse_args(argv)

    style.apply_style()
    outdir = Path(args.outdir)

    written: list[Path] = []
    for fn in (fig2_cp_profiles, fig3_surface_fields):
        p = fn(outdir, model=args.model,
                configs_root=args.configs_root, checkpoints_root=args.checkpoints_root)
        if p:
            written.append(p)

    print(f"\n{len(written)} figure(s) written to {outdir}:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
