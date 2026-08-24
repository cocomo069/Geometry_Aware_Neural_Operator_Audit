"""Training objective (CONTEXT.md section 8; spec sections 5.3-5.5).

.. math::

    \\mathcal{L} = \\mathcal{L}_{\\text{data}}
                 + \\lambda_F \\mathcal{L}_{\\text{force}}
                 + \\lambda_S \\mathcal{L}_{\\text{sym}}

with :math:`\\mathcal{L}_{\\text{data}}` the per-sample **relative L2** on the
surface pressure and wall shear stress, and every auxiliary term divided by its
value at initialization (``norm_ref``) so all terms start at order one.
**Default lambdas are zero** -- the physics terms are ablation flags, per
CONTEXT.md section 8.

Dependency inversion
--------------------
``total_loss`` never imports ``src.physics`` or ``src.geometry``: the force
integrator and the symmetry residual arrive as *callables*, which keeps
``src.models`` free of circular imports and lets the trainer wrap
denormalization into the closure it passes.

``integrate_fn(p, tau, normal, ds, cond, batch_idx) -> {"cl": (B,), "cd": (B,)}``
    Signature-compatible with ``src.physics.force_integration.integrate_forces``
    (CONTEXT.md section 6).  Predictions are in *normalized* space, so the
    trainer is expected to hand over a closure that denormalizes ``p``/``tau``
    first -- metrics are defined in physical units (CONTEXT.md section 9).

``symmetry_fn(pred, batch) -> scalar tensor``
    Hook for ``src.geometry.symmetry``; typically a ``functools.partial`` that
    has already captured the model so it can evaluate the mirrored geometry.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional

import torch

from src.models.common import segment_sum

__all__ = [
    "DEFAULT_WEIGHTS",
    "rel_l2_per_sample",
    "rel_l2",
    "force_consistency_loss",
    "data_loss",
    "total_loss",
    "init_norm_ref",
]


#: lambda defaults -- pure data loss (CONTEXT.md section 8).
DEFAULT_WEIGHTS: dict = {
    "p": 1.0,
    "tau": 1.0,
    "force": 0.0,     # lambda_F
    "sym": 0.0,       # lambda_S
}

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# data term
# --------------------------------------------------------------------------- #

def rel_l2_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: int,
    eps: float = _EPS,
) -> torch.Tensor:
    """Per-sample relative L2 ``||pred - target||_2 / (||target||_2 + eps)``.

    Shapes: ``pred``/``target`` ``(sumN,)`` or ``(sumN, C)``; returns ``(B,)``.
    The reduction is a masked segment sum, so samples never leak into each
    other and empty graphs give exactly zero.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} "
                         f"vs target {tuple(target.shape)}")
    if pred.dim() == 1:
        pred = pred.unsqueeze(-1)
        target = target.unsqueeze(-1)
    if pred.dim() != 2:
        raise ValueError(f"expected (sumN,) or (sumN, C), got {tuple(pred.shape)}")
    diff2 = ((pred - target) ** 2).sum(dim=-1, keepdim=True)
    ref2 = (target ** 2).sum(dim=-1, keepdim=True)
    num = segment_sum(diff2, batch_idx, num_graphs).squeeze(-1)
    den = segment_sum(ref2, batch_idx, num_graphs).squeeze(-1)
    return torch.sqrt(num + eps * eps) / (torch.sqrt(den) + eps)


def rel_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: int,
    eps: float = _EPS,
) -> torch.Tensor:
    """Batch mean of :func:`rel_l2_per_sample` -- a scalar tensor."""
    return rel_l2_per_sample(pred, target, batch_idx, num_graphs, eps=eps).mean()


def data_loss(pred: Mapping, batch: Mapping, weights: Mapping,
              num_graphs: int) -> dict:
    """Relative-L2 data terms on ``p`` and ``tau``.

    Returns ``{"data_p": scalar, "data_tau": scalar, "data": scalar}``; a term
    is skipped (contributing an exact zero) when its target is absent from the
    batch, so the same function serves surface-only and field-ablation runs.
    """
    batch_idx = batch["batch_idx"]
    device = pred["p"].device
    zero = torch.zeros((), device=device, dtype=pred["p"].dtype)

    lp = zero
    if batch.get("surf_p", None) is not None:
        lp = rel_l2(pred["p"], batch["surf_p"].reshape(-1), batch_idx, num_graphs)
    lt = zero
    if batch.get("surf_tau", None) is not None:
        lt = rel_l2(pred["tau"], batch["surf_tau"].reshape(-1, 2), batch_idx, num_graphs)

    total = float(weights.get("p", 1.0)) * lp + float(weights.get("tau", 1.0)) * lt
    return {"data_p": lp, "data_tau": lt, "data": total}


# --------------------------------------------------------------------------- #
# force-consistency term (spec section 5.4)
# --------------------------------------------------------------------------- #

def force_consistency_loss(
    pred: Mapping,
    batch: Mapping,
    integrate_fn: Callable,
    num_graphs: int,
) -> dict:
    """``|C_int - C_true| + |C_head - C_true| + |C_int - C_head|``.

    Averaged over the batch and over the two coefficients.  When ground-truth
    coefficients are missing from the batch only the label-free self-
    consistency part ``|C_int - C_head|`` is used -- that is exactly the FSC
    diagnostic of spec section 5.4, which is what makes this usable on
    unlabelled pool geometries in the active-learning loop.

    Returns ``{"force": scalar, "cl_int": (B,), "cd_int": (B,)}``.
    """
    coefs = integrate_fn(pred["p"], pred["tau"], batch["surf_normal"],
                         batch["surf_ds"], batch["cond"], batch["batch_idx"])
    cl_int = coefs["cl"].reshape(-1)
    cd_int = coefs["cd"].reshape(-1)
    if cl_int.shape[0] != num_graphs:
        raise ValueError(f"integrate_fn returned {cl_int.shape[0]} samples, "
                         f"expected {num_graphs}")

    head = pred["coef_head"]
    c_int = torch.stack([cl_int, cd_int], dim=-1)

    term = (c_int - head).abs().mean()
    cl_true = batch.get("cl_true", None)
    cd_true = batch.get("cd_true", None)
    if cl_true is not None and cd_true is not None:
        c_true = torch.stack([cl_true.reshape(-1), cd_true.reshape(-1)], dim=-1)
        term = term + (c_int - c_true).abs().mean() + (head - c_true).abs().mean()
    return {"force": term, "cl_int": cl_int, "cd_int": cd_int}


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #

def _resolve_weights(weights: Optional[Mapping]) -> dict:
    merged = dict(DEFAULT_WEIGHTS)
    if weights:
        merged.update({k: v for k, v in weights.items() if v is not None})
    # tolerate the lambda spelling used in the spec
    for alias, key in (("lambda_F", "force"), ("lambda_S", "sym")):
        if alias in merged:
            merged[key] = merged.pop(alias)
    return merged


def total_loss(
    pred: Mapping,
    batch: Mapping,
    weights: Optional[Mapping] = None,
    norm_ref: Optional[Mapping] = None,
    integrate_fn: Optional[Callable] = None,
    symmetry_fn: Optional[Callable] = None,
    extra_fns: Optional[Mapping[str, Callable]] = None,
) -> tuple:
    """Full training objective.

    Parameters
    ----------
    pred:
        Model output ``{"p", "tau", "coef_head"}`` (CONTEXT.md section 7).
    batch:
        Loader batch; targets ``surf_p``, ``surf_tau``, ``cl_true``, ``cd_true``
        are all optional (missing ones simply drop out of the objective).
    weights:
        ``{"p", "tau", "force", "sym", ...}``; defaults in
        :data:`DEFAULT_WEIGHTS` (all lambdas zero).
    norm_ref:
        ``{term: float}`` scale factors from :func:`init_norm_ref`.  Auxiliary
        terms are divided by these so each starts at order one.  ``None`` means
        no normalization.
    integrate_fn, symmetry_fn, extra_fns:
        Callables injected by the trainer; see the module docstring.  A term
        whose weight is zero is **not evaluated at all**, so a lambda-zero run
        pays nothing for the physics hooks.

    Returns
    -------
    ``(loss, terms)`` -- ``loss`` is a scalar tensor carrying grad; ``terms`` is
    a dict of detached scalar tensors for logging (``history.csv``).
    """
    if "p" not in pred or "tau" not in pred or "coef_head" not in pred:
        raise KeyError("pred must contain 'p', 'tau' and 'coef_head'")
    w = _resolve_weights(weights)
    num_graphs = int(batch["cond"].shape[0])
    refs = dict(norm_ref or {})

    parts = data_loss(pred, batch, w, num_graphs)
    loss = parts["data"]
    terms = {"data_p": parts["data_p"], "data_tau": parts["data_tau"],
             "data": parts["data"]}

    lam_f = float(w.get("force", 0.0))
    if lam_f != 0.0:
        if integrate_fn is None:
            raise ValueError("weights['force'] != 0 requires integrate_fn "
                             "(src.physics.force_integration.integrate_forces)")
        fc = force_consistency_loss(pred, batch, integrate_fn, num_graphs)
        scaled = fc["force"] / max(float(refs.get("force", 1.0)), _EPS)
        loss = loss + lam_f * scaled
        terms["force"] = fc["force"]
        terms["force_scaled"] = scaled

    lam_s = float(w.get("sym", 0.0))
    if lam_s != 0.0:
        if symmetry_fn is None:
            raise ValueError("weights['sym'] != 0 requires symmetry_fn "
                             "(src.geometry.symmetry.symmetry_residual)")
        sym = symmetry_fn(pred, batch)
        scaled = sym / max(float(refs.get("sym", 1.0)), _EPS)
        loss = loss + lam_s * scaled
        terms["sym"] = sym
        terms["sym_scaled"] = scaled

    for name, fn in (extra_fns or {}).items():
        lam = float(w.get(name, 0.0))
        if lam == 0.0:
            continue
        value = fn(pred, batch)
        scaled = value / max(float(refs.get(name, 1.0)), _EPS)
        loss = loss + lam * scaled
        terms[name] = value
        terms[name + "_scaled"] = scaled

    terms["loss"] = loss
    return loss, {k: v.detach() for k, v in terms.items()}


def init_norm_ref(
    pred: Mapping,
    batch: Mapping,
    weights: Optional[Mapping] = None,
    integrate_fn: Optional[Callable] = None,
    symmetry_fn: Optional[Callable] = None,
    extra_fns: Optional[Mapping[str, Callable]] = None,
    floor: float = 1e-6,
) -> dict:
    """Capture each auxiliary term value at initialization (spec section 5.3).

    Call once on the first batch with the freshly initialized model, stash the
    result in the checkpoint, and pass it to :func:`total_loss` thereafter.
    Terms whose lambda is zero are skipped; values are floored at ``floor`` so a
    term that happens to start at zero cannot blow the objective up.
    """
    w = _resolve_weights(weights)
    num_graphs = int(batch["cond"].shape[0])
    refs: dict = {}
    with torch.no_grad():
        if float(w.get("force", 0.0)) != 0.0 and integrate_fn is not None:
            fc = force_consistency_loss(pred, batch, integrate_fn, num_graphs)
            refs["force"] = max(float(fc["force"].item()), floor)
        if float(w.get("sym", 0.0)) != 0.0 and symmetry_fn is not None:
            refs["sym"] = max(float(symmetry_fn(pred, batch).item()), floor)
        for name, fn in (extra_fns or {}).items():
            if float(w.get(name, 0.0)) != 0.0:
                refs[name] = max(float(fn(pred, batch).item()), floor)
    return refs
