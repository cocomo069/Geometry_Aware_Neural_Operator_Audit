"""Consistency residuals: divergence and no-slip (spec section 5.5).

``divergence_residual`` estimates div(u) at scattered points with
least-squares linear gradients on kNN neighborhoods. Neighbor search uses
scipy's cKDTree on CPU and never crosses sample boundaries in a
concatenated (PyG-style) batch. The neighbor indices and the LS
pseudo-inverse depend only on the (fixed) point positions, so they can be
precomputed once per geometry via ``knn_ls_weights`` and reused; the
residual is then a fixed linear map of ``u`` and fully differentiable
w.r.t. ``u``. The estimator is exact for affine velocity fields.

``noslip_residual`` implements the spec's weighted near-wall penalty
    L_bc = (1/|N_delta|) * sum_{x in N_delta} ||u(x)||^2 * w(x),
    w(x) = 1 - sdf(x)/delta,  N_delta = {x : sdf(x) < delta},
with sdf >= 0 in the fluid (CONTEXT.md section 4).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

__all__ = [
    "knn_ls_weights",
    "pointwise_divergence",
    "divergence_residual",
    "noslip_residual",
]


def _to_numpy(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def knn_ls_weights(pos, batch_idx=None, k: int = 8):
    """Precompute kNN indices and least-squares gradient weights.

    For each point i with neighbor offsets D_i (k, 2) (rows x_j - x_i), the
    LS estimate of the gradient of a scalar field f is
        grad f_i = P_i @ (f[nbr_i] - f_i),   P_i = pinv(D_i)  (2, k).

    Parameters
    ----------
    pos : (N, 2) point positions (torch tensor or numpy array).
    batch_idx : optional (N,) long; neighborhoods are restricted to each
        sample. None => single sample.
    k : neighbors per point (excluding the point itself); each sample must
        contain more than k points, and points within a sample must be
        distinct.

    Returns
    -------
    (idx, weights): idx (N, k) torch.long global indices; weights (N, 2, k)
    torch.float64 pseudo-inverse matrices.
    """
    pos_np = _to_numpy(pos).astype(np.float64)
    assert pos_np.ndim == 2 and pos_np.shape[1] == 2, (
        f"pos must be (N, 2), got {pos_np.shape}")
    assert k >= 2, "k >= 2 required for a 2-D gradient"
    n_pts = pos_np.shape[0]

    if batch_idx is None:
        groups = [np.arange(n_pts)]
    else:
        bi = _to_numpy(batch_idx).astype(np.int64)
        assert bi.shape == (n_pts,), f"batch_idx must be ({n_pts},), got {bi.shape}"
        groups = [np.flatnonzero(bi == b) for b in range(int(bi.max()) + 1)]

    idx = np.empty((n_pts, k), dtype=np.int64)
    for g in groups:
        assert g.size > k, (
            f"sample with {g.size} points cannot provide k={k} neighbors")
        tree = cKDTree(pos_np[g])
        _, nn = tree.query(pos_np[g], k=k + 1)   # first column is self
        idx[g] = g[nn[:, 1:]]

    offsets = pos_np[idx] - pos_np[:, None, :]   # (N, k, 2)
    weights = np.linalg.pinv(offsets)            # batched -> (N, 2, k)
    device = pos.device if isinstance(pos, torch.Tensor) else "cpu"
    return (torch.as_tensor(idx, dtype=torch.long, device=device),
            torch.as_tensor(weights, dtype=torch.float64, device=device))


def pointwise_divergence(u, pos=None, batch_idx=None, k: int = 8, weights=None):
    """LS estimate of div(u) at every point.

    Parameters
    ----------
    u : (N, 2) velocity (torch tensor; differentiable path).
    pos : (N, 2) positions; required unless ``weights`` is given.
    weights : optional precomputed (idx, P) from ``knn_ls_weights``.

    Returns
    -------
    (N,) tensor of divergence estimates, in u's dtype.
    """
    assert isinstance(u, torch.Tensor) and u.dim() == 2 and u.shape[1] == 2, (
        "u must be a (N, 2) torch tensor")
    if weights is None:
        assert pos is not None, "pos is required when weights are not precomputed"
        weights = knn_ls_weights(pos, batch_idx=batch_idx, k=k)
    idx, p_mat = weights
    assert idx.shape[0] == u.shape[0], "weights were computed for a different N"
    p_mat = p_mat.to(dtype=u.dtype, device=u.device)
    du = u[idx] - u.unsqueeze(1)                 # (N, k, 2)
    # div = d(u_x)/dx + d(u_y)/dy = P[:,0,:] . du_x + P[:,1,:] . du_y
    div = (p_mat[:, 0, :] * du[..., 0]).sum(dim=1) \
        + (p_mat[:, 1, :] * du[..., 1]).sum(dim=1)
    return div


def divergence_residual(u, pos, batch_idx=None, k: int = 8, weights=None):
    """Mean squared divergence, per sample (spec's L_div).

    Returns a (B,) tensor when ``batch_idx`` is given, else a 0-dim tensor.
    Differentiable w.r.t. ``u``.
    """
    div = pointwise_divergence(u, pos=pos, batch_idx=batch_idx, k=k, weights=weights)
    sq = div.pow(2)
    if batch_idx is None:
        return sq.mean()
    assert batch_idx.dtype == torch.long, "batch_idx must be torch.long"
    n_batch = int(batch_idx.max()) + 1
    num = torch.zeros(n_batch, dtype=sq.dtype, device=sq.device)
    num = num.index_add(0, batch_idx, sq)
    cnt = torch.zeros(n_batch, dtype=sq.dtype, device=sq.device)
    cnt = cnt.index_add(0, batch_idx, torch.ones_like(sq))
    return num / cnt.clamp_min(1.0)


def noslip_residual(u, sdf, delta: float, batch_idx=None):
    """Weighted near-wall no-slip penalty (spec section 5.5).

        L_bc = (1/|N_delta|) sum_{sdf < delta} ||u||^2 (1 - sdf/delta)

    Parameters
    ----------
    u : (N, 2) velocity at volume points.
    sdf : (N,) distance to the wall, >= 0 in the fluid.
    delta : near-wall band width, > 0.
    batch_idx : optional (N,) long for per-sample means.

    Returns
    -------
    (B,) tensor when batched, else 0-dim. Samples with no points inside the
    band return 0. Differentiable w.r.t. ``u``.
    """
    assert isinstance(u, torch.Tensor) and u.dim() == 2 and u.shape[1] == 2, (
        "u must be a (N, 2) torch tensor")
    assert sdf.shape == (u.shape[0],), (
        f"sdf must be ({u.shape[0]},), got {tuple(sdf.shape)}")
    assert delta > 0, "delta must be positive"
    w = (1.0 - sdf / delta).clamp_min(0.0)       # zero outside the band
    in_band = (sdf < delta).to(u.dtype)
    contrib = u.pow(2).sum(dim=1) * w
    if batch_idx is None:
        cnt = in_band.sum()
        return contrib.sum() / cnt.clamp_min(1.0)
    assert batch_idx.dtype == torch.long, "batch_idx must be torch.long"
    n_batch = int(batch_idx.max()) + 1
    num = torch.zeros(n_batch, dtype=contrib.dtype, device=contrib.device)
    num = num.index_add(0, batch_idx, contrib)
    cnt = torch.zeros_like(num).index_add(0, batch_idx, in_band)
    return num / cnt.clamp_min(1.0)
