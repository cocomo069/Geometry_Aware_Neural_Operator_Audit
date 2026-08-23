"""Aerodynamic force integration from surface fields (FROZEN, CONTEXT.md s6).

Force convention
----------------
Force per unit span exerted by the fluid on the body:

    F = sum_i (-p_i * n_i + tau_i) * ds_i

with n_i the **outward** unit normal at surface point i (pointing from the
body into the fluid), p_i the surface pressure (freestream-relative or not:
any constant offset integrates to zero over a closed contour), tau_i the
wall shear-stress vector (traction exerted by the fluid on the wall), and
ds_i the per-point quadrature weight (facet-length share, sums to the
perimeter).

Drag / lift directions and the lift sign
----------------------------------------
    e_inf  = cond / |cond|                     (unit freestream direction)
    e_perp = rot90_ccw(e_inf) = (-e_inf_y, e_inf_x)
    drag D = F . e_inf,   lift L = F . e_perp

Derivation of the lift sign: with angle of attack alpha, the freestream is
cond = U (cos a, sin a), so e_perp = (-sin a, cos a) and

    L = -Fx sin a + Fy cos a .

At a = 0 this reduces to L = Fy: a body deflecting fluid downward feels
Fy > 0, hence L > 0 -- so a cambered airfoil at positive AoA gets CL > 0,
the standard wind-axes convention where (e_inf, e_perp) is the right-handed
(drag, lift) frame. Two independent checks enforced by the tests:
(1) reflecting the whole problem about the x-axis (AoA -> -AoA, y -> -y)
flips the sign of CL and preserves CD; (2) a potential-flow cylinder with
counterclockwise circulation Gamma has L = -rho U Gamma (Kutta-Joukowski:
positive lift requires clockwise circulation, as around a lifting airfoil),
which this routine reproduces at any freestream angle.

Coefficients
------------
    CD = D / (0.5 * rho_eff * |u_inf|^2 * a_ref),   CL likewise.

``pressure_mode`` absorbs the dataset's pressure convention:

* ``"static"``    -- p and tau are true stresses (units Pa); rho_eff = rho.
* ``"kinematic"`` -- p and tau are density-normalized (p/rho, tau/rho; the
  AirfRANS / incompressible-OpenFOAM convention). The integrated F is then
  force-per-density, and 0.5*|u_inf|^2*a_ref plays the role of the dynamic
  pressure: rho_eff = 1 and the ``rho`` argument is ignored (rho cancels
  between numerator and denominator).
"""
from __future__ import annotations

import torch

__all__ = ["integrate_forces"]


def integrate_forces(
    p: torch.Tensor,
    tau: torch.Tensor,
    normal: torch.Tensor,
    ds: torch.Tensor,
    cond: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    rho: float = 1.0,
    a_ref: float = 1.0,
    pressure_mode: str = "static",
    eps: float = 1e-12,
) -> dict:
    """Integrate surface fields to forces and force coefficients.

    Parameters
    ----------
    p : (N,) surface pressure at each surface point.
    tau : (N, 2) wall shear-stress vector.
    normal : (N, 2) outward unit normals (into the fluid).
    ds : (N,) per-point quadrature weights (facet length share).
    cond : (2,) freestream velocity vector if unbatched, (B, 2) if batched.
    batch_idx : optional (N,) long tensor mapping each point to its sample
        (PyG-style concatenated batch). None => single sample.
    rho : fluid density (used only when pressure_mode == "static").
    a_ref : reference area (chord length in 2-D; AirfRANS chord = 1).
    pressure_mode : "static" or "kinematic" (see module docstring).
    eps : numerical floor for |u_inf| normalization.

    Returns
    -------
    dict with keys ``cl``, ``cd``, ``fx``, ``fy``; each a 0-dim tensor when
    unbatched, a (B,) tensor when batched. Fully differentiable w.r.t.
    ``p`` and ``tau`` (and the geometry inputs).
    """
    assert p.dim() == 1, f"p must be (N,), got {tuple(p.shape)}"
    n_pts = p.shape[0]
    assert tau.shape == (n_pts, 2), f"tau must be ({n_pts}, 2), got {tuple(tau.shape)}"
    assert normal.shape == (n_pts, 2), (
        f"normal must be ({n_pts}, 2), got {tuple(normal.shape)}")
    assert ds.shape == (n_pts,), f"ds must be ({n_pts},), got {tuple(ds.shape)}"
    if pressure_mode not in ("static", "kinematic"):
        raise ValueError(f"pressure_mode must be 'static' or 'kinematic', got {pressure_mode!r}")

    # per-point force contribution (N, 2)
    f_pt = (-p.unsqueeze(1) * normal + tau) * ds.unsqueeze(1)

    if batch_idx is None:
        assert cond.shape == (2,), (
            f"unbatched cond must be (2,), got {tuple(cond.shape)}")
        force = f_pt.sum(dim=0, keepdim=True)          # (1, 2)
        cond_b = cond.unsqueeze(0)                     # (1, 2)
        squeeze = True
    else:
        assert batch_idx.shape == (n_pts,), (
            f"batch_idx must be ({n_pts},), got {tuple(batch_idx.shape)}")
        assert batch_idx.dtype == torch.long, "batch_idx must be torch.long"
        assert cond.dim() == 2 and cond.shape[1] == 2, (
            f"batched cond must be (B, 2), got {tuple(cond.shape)}")
        n_batch = cond.shape[0]
        assert int(batch_idx.max()) < n_batch, "batch_idx exceeds cond batch size"
        force = torch.zeros(n_batch, 2, dtype=f_pt.dtype, device=f_pt.device)
        force = force.index_add(0, batch_idx, f_pt)    # (B, 2)
        cond_b = cond
        squeeze = False

    u_mag = torch.linalg.vector_norm(cond_b, dim=1)                    # (B,)
    e_inf = cond_b / (u_mag.clamp_min(eps)).unsqueeze(1)               # (B, 2)
    e_perp = torch.stack([-e_inf[:, 1], e_inf[:, 0]], dim=1)           # rot90 CCW

    rho_eff = rho if pressure_mode == "static" else 1.0
    q_dyn = (0.5 * rho_eff * a_ref) * u_mag.pow(2)                     # (B,)
    q_dyn = q_dyn.clamp_min(eps)

    drag = (force * e_inf).sum(dim=1)
    lift = (force * e_perp).sum(dim=1)
    out = {
        "cl": lift / q_dyn,
        "cd": drag / q_dyn,
        "fx": force[:, 0],
        "fy": force[:, 1],
    }
    if squeeze:
        out = {k: v.squeeze(0) for k, v in out.items()}
    return out
