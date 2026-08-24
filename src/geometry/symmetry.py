"""Mirror-symmetry operators and residuals (spec section 5.5).

The reflection R is about the x-axis (the AirfRANS chord line): positions
and every vector field flip their y-component; the freestream condition
``cond`` flips its y-component too, which maps AoA -> -AoA. Scalars (p, ds,
sdf) are invariant. Reflection reverses the contour's orientation
(CCW <-> CW) but keeps the per-point index correspondence, so pointwise
comparison between an original and a reflected prediction is valid.

The equivariance statement being tested: a physically consistent surrogate
must satisfy  u_hat(g, c) == R# u_hat(R g, R c)  for every (g, c), where
R# is the action of R on the output fields:

* scalar field p: unchanged;
* vector field tau: y-component negated;
* coefficient head (CL, CD): CL -> -CL, CD unchanged.

The CL sign flip follows from the force-integration convention
(src/physics/force_integration.py): with e_perp = rot90_ccw(e_inf), the
reflected problem has F -> (Fx, -Fy) and e_inf -> (ex, -ey), so
D = F . e_inf is invariant while L = F . e_perp changes sign.

``symmetry_residual`` implements the spec's relative L2 form
    || u(g,c) - R# u(Rg, Rc) ||_2 / || u(g,c) ||_2
per sample over the concatenated (p, tau) surface fields. It is exactly
zero for an equivariant model on ANY input; on a symmetric geometry with a
symmetric condition it measures the model's deviation from the physical
mirror symmetry of the true solution.
"""
from __future__ import annotations

import torch

__all__ = [
    "reflect_x_batch",
    "reflect_prediction",
    "symmetry_residual",
    "antisymmetry_check",
]

# per-point or per-sample 2-vector quantities whose y-component flips under R
_FLIP_Y_KEYS = (
    "surf_pos", "surf_normal", "surf_tau",
    "vol_pos", "vol_u",
    "pos", "normal", "tau", "u",
    "cond",
)
# per-sample scalars that negate under R
_NEGATE_KEYS = ("aoa_deg",)

# fixed-grid / connectivity caches (D-017) that are NOT per-point 2-vectors and
# must be RECOMPUTED from the mirrored geometry, not passed through: a reflected
# airfoil needs a reflected grid SDF. Dropping them makes the model rebuild them
# from the mirrored surf_pos (QA F1). edge_index/curvature are reflection-
# invariant (isometry preserves kNN and signed curvature) so they are NOT dropped.
_DROP_UNDER_REFLECTION = ("grid_sdf", "grid_mask", "grid_feat")


def _flip_y(t: torch.Tensor) -> torch.Tensor:
    assert t.shape[-1] == 2, f"expected trailing dim 2, got {tuple(t.shape)}"
    out = t.clone()
    out[..., 1] = -out[..., 1]
    return out


def reflect_x_batch(batch: dict) -> dict:
    """Reflect a batch dict about the x-axis: R g and R c.

    Flips the y-component of every known 2-vector key (positions, normals,
    shear, velocity, cond) and negates ``aoa_deg`` if present; everything
    else (scalars, ds, sdf, batch_idx, names) is passed through unchanged.
    Returns a new dict; input tensors are not modified.
    """
    out = {}
    for key, val in batch.items():
        if key in _DROP_UNDER_REFLECTION:
            continue  # force recompute from the mirrored geometry (F1)
        if key in _FLIP_Y_KEYS and isinstance(val, torch.Tensor):
            out[key] = _flip_y(val)
        elif key in _NEGATE_KEYS and isinstance(val, torch.Tensor):
            out[key] = -val
        else:
            out[key] = val
    return out


def reflect_prediction(pred: dict) -> dict:
    """Apply R# to a model prediction dict {p, tau, coef_head}.

    p unchanged; tau y-component negated; coef_head = (CL, CD) -> (-CL, CD).
    Returns a new dict.
    """
    out = dict(pred)
    if "tau" in out and isinstance(out["tau"], torch.Tensor):
        out["tau"] = _flip_y(out["tau"])
    if "coef_head" in out and isinstance(out["coef_head"], torch.Tensor):
        coef = out["coef_head"].clone()
        coef[..., 0] = -coef[..., 0]
        out["coef_head"] = coef
    return out


def _per_sample_rel_l2(diff_sq, ref_sq, batch_idx, eps):
    """sqrt(sum diff^2) / sqrt(sum ref^2) grouped by batch_idx (or global)."""
    if batch_idx is None:
        num = diff_sq.sum()
        den = ref_sq.sum()
        return torch.sqrt(num) / (torch.sqrt(den) + eps)
    assert batch_idx.dtype == torch.long, "batch_idx must be torch.long"
    n_batch = int(batch_idx.max()) + 1
    num = torch.zeros(n_batch, dtype=diff_sq.dtype, device=diff_sq.device)
    den = torch.zeros_like(num)
    num = num.index_add(0, batch_idx, diff_sq)
    den = den.index_add(0, batch_idx, ref_sq)
    return torch.sqrt(num) / (torch.sqrt(den) + eps)


def symmetry_residual(model, batch: dict, eps: float = 1e-12) -> torch.Tensor:
    """Relative-L2 mirror-symmetry residual of a model on a batch.

        res_b = || u(g,c) - R# u(Rg,Rc) ||_2 / || u(g,c) ||_2

    computed per sample over the concatenated surface fields (p, tau) --
    each surface point contributes the 3-vector (p, tau_x, tau_y).

    Parameters
    ----------
    model : callable, batch dict -> pred dict with "p" (N,) and/or
        "tau" (N, 2) (SurrogateBase interface, CONTEXT.md section 7).
    batch : batch dict with optional "batch_idx" (N,) long.

    Returns
    -------
    (B,) tensor if the batch carries ``batch_idx``, else a 0-dim tensor.
    Differentiable, so it can also serve as the training-time L_sym term.
    """
    pred = model(batch)
    pred_ref = reflect_prediction(model(reflect_x_batch(batch)))

    parts, parts_ref = [], []
    if "p" in pred and isinstance(pred["p"], torch.Tensor):
        assert pred["p"].dim() == 1, f"p must be (N,), got {tuple(pred['p'].shape)}"
        parts.append(pred["p"].unsqueeze(1))
        parts_ref.append(pred_ref["p"].unsqueeze(1))
    if "tau" in pred and isinstance(pred["tau"], torch.Tensor):
        assert pred["tau"].dim() == 2 and pred["tau"].shape[1] == 2, (
            f"tau must be (N, 2), got {tuple(pred['tau'].shape)}")
        parts.append(pred["tau"])
        parts_ref.append(pred_ref["tau"])
    assert parts, "model prediction has neither 'p' nor 'tau' fields"

    v = torch.cat(parts, dim=1)          # (N, F)
    v_ref = torch.cat(parts_ref, dim=1)  # (N, F)
    diff_sq = (v - v_ref).pow(2).sum(dim=1)
    ref_sq = v.pow(2).sum(dim=1)
    return _per_sample_rel_l2(diff_sq, ref_sq, batch.get("batch_idx"), eps)


def antisymmetry_check(model, batch: dict) -> dict:
    """Coefficient antisymmetry under reflection: CL(-a) ~ -CL(a), CD even.

    Evaluates the model's coefficient head on the batch and on its x-axis
    reflection and returns per-sample gaps

        cl_gap = | CL(g, c) + CL(Rg, Rc) |
        cd_gap = | CD(g, c) - CD(Rg, Rc) |

    For a *symmetric* airfoil (R g == g up to point re-indexing) this is
    exactly the spec's CL(-alpha) ~ -CL(alpha) / CD even check; for cambered
    shapes it is the general reflection-equivariance gap of the head.

    Returns dict {"cl_gap": (B,), "cd_gap": (B,)} (0-dim when unbatched).
    """
    pred = model(batch)
    pred_r = model(reflect_x_batch(batch))
    assert "coef_head" in pred, "model must return a 'coef_head' (B, 2) = (CL, CD)"
    coef, coef_r = pred["coef_head"], pred_r["coef_head"]
    assert coef.shape[-1] == 2, f"coef_head must be (..., 2), got {tuple(coef.shape)}"
    return {
        "cl_gap": (coef[..., 0] + coef_r[..., 0]).abs(),
        "cd_gap": (coef[..., 1] - coef_r[..., 1]).abs(),
    }
