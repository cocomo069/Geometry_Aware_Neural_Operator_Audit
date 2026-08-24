"""M2 -- GINO-lite: SDF-conditioned FNO on a fixed latent grid (spec section 5.2).

Pipeline
--------
1. **Kernel encoder.**  Every surface quadrature point ``y`` deposits its
   features into the ``k_enc`` nearest latent-grid nodes ``x`` with a learned
   kernel ``kappa_enc(x - y, ||x - y||, s_g(x))`` -- a discretization of

   .. math::
       h_{\\text{lat}}(x) = \\int_{B_r(x)\\cap\\Gamma_g}
           \\kappa_{\\text{enc}}(x, y, s_g(x))\\, a(y)\\, \\mathrm{d}y .

2. **FNO2d stack** on the ``64 x 64`` latent grid: 4 layers, ``modes = 16``,
   ``width = 32``, spectral convolutions implemented here with
   :func:`torch.fft.rfft2` (no ``neuraloperator`` dependency).

3. **Kernel decoder** interpolating the latent field back to the (arbitrary)
   surface query points with ``kappa_dec(z - x, ||z - x||)``.

Both transfers are quadrature approximations of continuum integrals, so
discretization invariance is preserved.

Latent grid / bounding box
--------------------------
**Fixed bbox** ``[-0.5, 1.5] x [-1.0, 1.0]`` at ``64 x 64``.  Rationale: the
AirfRANS airfoil has unit chord spanning ``x in [0, 1]`` (CONTEXT.md section 2),
so half a chord of margin upstream and downstream and one chord above/below
covers the near field where the surface trace lives, at a uniform spacing of
``dx = 2/64 = 0.031`` chords -- fine enough that a ``modes = 16`` spectral
truncation resolves ~4 chord-length-scale wavenumbers per direction.  Set
``bbox_mode: 'auto'`` in the config to derive the box per batch instead (useful
if the loader ever normalizes coordinates away from the chord frame).

SDF
---
The grid SDF is recomputed per sample from ``surf_pos``.  When A2's
``src/geometry/sdf.py`` is importable its ``sdf_on_grid(surf_pos, grid_xy)`` is
used; otherwise this module falls back to an equivalent local numpy routine
(exact point-to-segment distance + even-odd inside test).  Both share one
convention: **negative inside the body, positive in the fluid.**  A precomputed
``batch['grid_sdf']`` ``(B, H, W)`` or ``(B, H*W)`` short-circuits the whole
computation -- that is the documented hand-off if the loader ever caches it.

Conditioning ablation (spec section 5.2, PLAN Phase-3 item 5)
-------------------------------------------------------------
``conditioning: 'sdf' | 'mask' | 'sdf+normals'``.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.models.common import (
    FourierFeatures,
    SurrogateBase,
    build_mlp,
    segment_sum,
)
from src.models.heads import CoefHead, FieldHead

__all__ = [
    "SpectralConv2d",
    "FNOBlock",
    "SDFFNOSurrogate",
    "CONDITIONINGS",
    "DEFAULT_CONFIG",
    "compute_grid_sdf",
    "make_latent_grid",
]


CONDITIONINGS = ("sdf", "mask", "sdf+normals")
CONDITIONING_DIM = {"sdf": 1, "mask": 1, "sdf+normals": 3}

DEFAULT_CONFIG = {
    "width": 32,
    "modes": 16,               # per direction
    "n_fno_layers": 4,
    "spectral_groups": 4,      # grouped spectral channel mixing (budget, see below)
    "fno_residual": True,
    "grid_res": (64, 64),      # (n_y, n_x)
    "bbox": (-0.5, 1.5, -1.0, 1.0),   # (xmin, xmax, ymin, ymax)
    "bbox_mode": "fixed",      # 'fixed' | 'auto'
    "bbox_margin": 0.25,       # only used by bbox_mode='auto'
    "conditioning": "sdf",
    "k_enc": 8,
    "k_dec": 8,
    "enc_radius_cells": 4.0,   # encoder support radius, in grid cells
    "kernel_hidden": 256,
    "point_hidden": 256,
    "n_freqs": 6,
    "activation": "gelu",
    "use_external_sdf": True,
    "trunk_dim": 320,
    "coef_hidden": 256,
    "coef_layers": 2,
    "coef_pooling": "ds_mean_max",
}


# --------------------------------------------------------------------------- #
# SDF helpers (numpy; A2's src/geometry/sdf.py used opportunistically)
# --------------------------------------------------------------------------- #

#: A2 owns ``src/geometry/sdf.py``.  Its public entry point is
#: ``sdf_on_grid(surf_pos, grid_xy) -> (...,)`` with the **same** convention we
#: use here (negative inside the body).  The alternates are tried only so a
#: rename on A2's side degrades to the local fallback instead of crashing.
_EXTERNAL_SDF_NAMES = (
    "sdf_on_grid",
    "signed_distance_to_polygon",
    "signed_distance_polygon",
    "signed_distance",
    "sdf_from_polygon",
    "grid_sdf",
    "polygon_sdf",
)

_EXTERNAL_SDF_STATE: dict = {"resolved": False, "fn": None}


def _resolve_external_sdf():
    """Best-effort import of A2's SDF routine; ``None`` when unavailable."""
    if _EXTERNAL_SDF_STATE["resolved"]:
        return _EXTERNAL_SDF_STATE["fn"]
    fn = None
    try:  # pragma: no cover - depends on A2 landing
        import importlib

        module = importlib.import_module("src.geometry.sdf")
        for name in _EXTERNAL_SDF_NAMES:
            candidate = getattr(module, name, None)
            if callable(candidate):
                fn = candidate
                break
    except Exception:
        fn = None
    _EXTERNAL_SDF_STATE["resolved"] = True
    _EXTERNAL_SDF_STATE["fn"] = fn
    return fn


def _unsigned_distance_to_polyline(
    query: np.ndarray, poly: np.ndarray, chunk: int = 2048
) -> np.ndarray:
    """Distance from ``query`` ``(G, 2)`` to the closed polyline ``poly`` ``(S, 2)``.

    Exact point-to-segment distance, chunked over query points so peak memory is
    ``O(chunk * S)`` rather than ``O(G * S)``.
    """
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a                                     # (S, 2)
    denom = np.einsum("sd,sd->s", ab, ab)
    denom = np.where(denom > 0.0, denom, 1.0)
    out = np.empty(query.shape[0], dtype=np.float64)
    for start in range(0, query.shape[0], chunk):
        q = query[start:start + chunk]             # (C, 2)
        aq = q[:, None, :] - a[None, :, :]         # (C, S, 2)
        t = np.einsum("csd,sd->cs", aq, ab) / denom
        np.clip(t, 0.0, 1.0, out=t)
        proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        d = np.linalg.norm(q[:, None, :] - proj, axis=-1)
        out[start:start + chunk] = d.min(axis=1)
    return out


def _points_inside_polygon(query: np.ndarray, poly: np.ndarray,
                           chunk: int = 4096) -> np.ndarray:
    """Even-odd (ray casting) inside test; ``(G,)`` bool.

    The surface points are ordered along the contour (CONTEXT.md section 4), so
    they form a simple closed polygon when the last vertex is joined to the
    first.
    """
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    out = np.zeros(query.shape[0], dtype=bool)
    for start in range(0, query.shape[0], chunk):
        q = query[start:start + chunk]
        px, py = q[:, 0:1], q[:, 1:2]                      # (C, 1)
        straddles = (y0[None, :] > py) != (y1[None, :] > py)
        denom = np.where(np.abs(y1 - y0) > 0.0, y1 - y0, 1.0)[None, :]
        x_cross = x0[None, :] + (py - y0[None, :]) * (x1 - x0)[None, :] / denom
        crossings = straddles & (px < x_cross)
        out[start:start + chunk] = (crossings.sum(axis=1) % 2) == 1
    return out


def compute_grid_sdf(
    surf_pos: np.ndarray,
    grid: np.ndarray,
    use_external: bool = True,
) -> np.ndarray:
    """Signed distance from ``grid`` ``(G, 2)`` to the airfoil ``surf_pos`` ``(S, 2)``.

    Returns ``(G,)`` float32, **negative inside the body**.
    """
    surf_pos = np.ascontiguousarray(surf_pos, dtype=np.float64)
    grid = np.ascontiguousarray(grid, dtype=np.float64)
    if surf_pos.ndim != 2 or surf_pos.shape[1] != 2:
        raise ValueError(f"surf_pos must be (S, 2), got {surf_pos.shape}")
    if surf_pos.shape[0] < 3:
        raise ValueError("need at least 3 surface points to define a polygon")

    if use_external:
        fn = _resolve_external_sdf()
        if fn is not None:
            try:
                raw = np.asarray(fn(surf_pos, grid), dtype=np.float64).reshape(-1)
                if raw.shape[0] == grid.shape[0] and np.all(np.isfinite(raw)):
                    return raw.astype(np.float32)
            except Exception:
                # never retry a broken external hook on every forward pass
                _EXTERNAL_SDF_STATE["fn"] = None

    dist = _unsigned_distance_to_polyline(grid, surf_pos)
    inside = _points_inside_polygon(grid, surf_pos)
    return np.where(inside, -dist, dist).astype(np.float32)


def make_latent_grid(bbox: Sequence[float], res: Sequence[int]) -> np.ndarray:
    """Row-major ``(n_y * n_x, 2)`` node coordinates for the latent grid.

    Index layout is ``iy * n_x + ix`` so a reshape to ``(n_y, n_x)`` gives the
    ``(H, W)`` image the FNO expects.
    """
    xmin, xmax, ymin, ymax = (float(v) for v in bbox)
    ny, nx = int(res[0]), int(res[1])
    if nx < 2 or ny < 2:
        raise ValueError("grid_res must be >= 2 in both directions")
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)


# --------------------------------------------------------------------------- #
# spectral convolution
# --------------------------------------------------------------------------- #

class SpectralConv2d(nn.Module):
    """2-D Fourier layer: truncated spectral multiplier via ``rfft2``.

    Complex weights are stored as **real** tensors with a trailing size-2 axis
    and read through :func:`torch.view_as_complex`, so optimizer state, fp32
    determinism and :func:`~src.models.common.count_params` all behave
    ordinarily.

    ``groups`` splits the channel mixing into block-diagonal groups.  With
    ``width = 32``, ``modes = 16`` a dense layer costs
    ``2 * 32 * 32 * 16 * 16 * 2 = 1.05 M`` real parameters -- four such layers
    alone would be 4.2 M, nearly 3x the 1.5 M budget of CONTEXT.md section 7.
    ``groups = 4`` keeps the frozen ``modes = 16`` / ``width = 32`` spectral
    resolution while cutting spectral parameters 4x; full cross-channel mixing
    is restored every layer by the dense pointwise ``1x1`` path in
    :class:`FNOBlock`.  See docs/PROGRESS.md.

    Shapes: ``(B, in_ch, H, W) -> (B, out_ch, H, W)``.
    """

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int,
                 groups: int = 1) -> None:
        super().__init__()
        if in_ch % groups or out_ch % groups:
            raise ValueError(f"channels {in_ch}/{out_ch} not divisible by groups {groups}")
        if modes1 < 1 or modes2 < 1:
            raise ValueError("modes must be >= 1")
        self.in_ch, self.out_ch, self.groups = int(in_ch), int(out_ch), int(groups)
        self.gi, self.go = self.in_ch // self.groups, self.out_ch // self.groups
        self.modes1, self.modes2 = int(modes1), int(modes2)
        scale = 1.0 / (self.gi * self.go)
        # axis 0 indexes the two retained k1 blocks (low positive / high negative)
        shape = (2, self.groups, self.gi, self.go, self.modes1, self.modes2, 2)
        self.weight = nn.Parameter(scale * (2.0 * torch.rand(shape) - 1.0))

    def _mul(self, x_ft: torch.Tensor, w: torch.Tensor, m1: int, m2: int) -> torch.Tensor:
        b = x_ft.shape[0]
        x = x_ft.reshape(b, self.groups, self.gi, m1, m2)
        out = torch.einsum("bgixy,gioxy->bgoxy", x, w)
        return out.reshape(b, self.out_ch, m1, m2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != self.in_ch:
            raise ValueError(f"expected (B, {self.in_ch}, H, W), got {tuple(x.shape)}")
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")                 # (B, C, H, W//2+1)
        m1 = min(self.modes1, h // 2)
        m2 = min(self.modes2, w // 2 + 1)
        weight = torch.view_as_complex(self.weight)             # (2,G,gi,go,M1,M2)

        out_ft = torch.zeros((b, self.out_ch, h, x_ft.shape[-1]),
                             dtype=x_ft.dtype, device=x.device)
        out_ft[:, :, :m1, :m2] = self._mul(
            x_ft[:, :, :m1, :m2], weight[0, ..., :m1, :m2], m1, m2)
        out_ft[:, :, -m1:, :m2] = self._mul(
            x_ft[:, :, -m1:, :m2], weight[1, ..., :m1, :m2], m1, m2)
        return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (f"in_ch={self.in_ch}, out_ch={self.out_ch}, "
                f"modes=({self.modes1}, {self.modes2}), groups={self.groups}")


class FNOBlock(nn.Module):
    """``x -> act(SpectralConv(x) + Conv1x1(x))`` with an optional residual."""

    def __init__(self, width: int, modes1: int, modes2: int, groups: int = 1,
                 activation: str = "gelu", residual: bool = True,
                 use_activation: bool = True) -> None:
        super().__init__()
        from src.models.common import get_activation

        self.spectral = SpectralConv2d(width, width, modes1, modes2, groups=groups)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.act = get_activation(activation) if use_activation else nn.Identity()
        self.residual = bool(residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.spectral(x) + self.pointwise(x))
        return x + y if self.residual else y


# --------------------------------------------------------------------------- #
# the surrogate
# --------------------------------------------------------------------------- #

class SDFFNOSurrogate(SurrogateBase):
    """M2: SDF-conditioned GINO-lite surrogate.

    Config keys: see :data:`DEFAULT_CONFIG`.  The ablation switch is
    ``conditioning in {'sdf', 'mask', 'sdf+normals'}``.

    ``forward(batch) -> {"p": (sumN,), "tau": (sumN, 2), "coef_head": (B, 2)}``.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        merged = dict(DEFAULT_CONFIG)
        merged.update(config or {})
        super().__init__(merged)

        self.conditioning = str(self.cfg("conditioning", "sdf"))
        if self.conditioning not in CONDITIONINGS:
            raise KeyError(f"conditioning must be one of {CONDITIONINGS}, "
                           f"got {self.conditioning!r}")
        self.cond_dim = CONDITIONING_DIM[self.conditioning]

        res = self.cfg("grid_res", (64, 64))
        self.grid_res = (int(res[0]), int(res[1]))
        self.bbox = tuple(float(v) for v in self.cfg("bbox", (-0.5, 1.5, -1.0, 1.0)))
        if len(self.bbox) != 4:
            raise ValueError("bbox must be (xmin, xmax, ymin, ymax)")
        self.bbox_mode = str(self.cfg("bbox_mode", "fixed"))
        if self.bbox_mode not in ("fixed", "auto"):
            raise KeyError("bbox_mode must be 'fixed' or 'auto'")
        self.bbox_margin = float(self.cfg("bbox_margin", 0.25))
        self.use_external_sdf = bool(self.cfg("use_external_sdf", True))

        width = int(self.cfg("width", 32))
        modes = self.cfg("modes", 16)
        modes1, modes2 = (int(modes), int(modes)) if np.isscalar(modes) else \
            (int(modes[0]), int(modes[1]))
        n_fno = int(self.cfg("n_fno_layers", 4))
        act = str(self.cfg("activation", "gelu"))
        self.width = width
        self.k_enc = int(self.cfg("k_enc", 8))
        self.k_dec = int(self.cfg("k_dec", 8))
        self.enc_radius_cells = float(self.cfg("enc_radius_cells", 4.0))
        kernel_hidden = int(self.cfg("kernel_hidden", 128))
        point_hidden = int(self.cfg("point_hidden", 128))

        self.pos_encoder = FourierFeatures(2, n_freqs=int(self.cfg("n_freqs", 6)))

        # --- encoder: surface features -> latent grid ------------------------
        self.surf_proj = build_mlp(self.point_input_dim(), width, point_hidden,
                                   n_hidden_layers=1, activation=act)
        # kappa_enc(x - y, ||x - y||, s_g(x)) -> (width,)
        self.enc_kernel = build_mlp(3 + self.cond_dim, width, kernel_hidden,
                                    n_hidden_layers=1, activation=act)
        # --- lifting on the grid ---------------------------------------------
        lift_in = width + self.cond_dim + self.pos_encoder.out_dim
        self.lift = build_mlp(lift_in, width, point_hidden,
                              n_hidden_layers=1, activation=act)
        # --- FNO stack --------------------------------------------------------
        groups = int(self.cfg("spectral_groups", 4))
        self.fno = nn.ModuleList([
            FNOBlock(width, modes1, modes2, groups=groups, activation=act,
                     residual=bool(self.cfg("fno_residual", True)),
                     use_activation=(i < n_fno - 1))
            for i in range(n_fno)
        ])
        self.grid_proj = build_mlp(width, width, point_hidden,
                                   n_hidden_layers=1, activation=act)
        # --- decoder: latent grid -> surface points ---------------------------
        self.dec_kernel = build_mlp(3, width, kernel_hidden,
                                    n_hidden_layers=1, activation=act)
        # --- readouts ---------------------------------------------------------
        trunk_dim = int(self.cfg("trunk_dim", 256))
        self.field_head = FieldHead(width + self.point_input_dim(),
                                    trunk_dim=trunk_dim, n_hidden_layers=2,
                                    activation=act)
        self.coef_head = CoefHead(trunk_dim,
                                  hidden_dim=int(self.cfg("coef_hidden", 256)),
                                  n_hidden_layers=int(self.cfg("coef_layers", 2)),
                                  activation=act,
                                  pooling=str(self.cfg("coef_pooling", "ds_mean_max")))

        self._grid_cache: dict = {}

    # ------------------------------------------------------------------ #
    # latent grid plumbing
    # ------------------------------------------------------------------ #
    @property
    def n_grid(self) -> int:
        return self.grid_res[0] * self.grid_res[1]

    def _grid_numpy(self, bbox) -> np.ndarray:
        key = tuple(round(float(v), 9) for v in bbox)
        entry = self._grid_cache.get(key)
        if entry is None:
            from scipy.spatial import cKDTree

            grid = make_latent_grid(key, self.grid_res)
            entry = {"grid": grid, "tree": cKDTree(grid)}
            self._grid_cache = {key: entry}   # single-entry cache; bbox rarely moves
        return entry["grid"]

    def _grid_tree(self, bbox):
        self._grid_numpy(bbox)
        key = tuple(round(float(v), 9) for v in bbox)
        return self._grid_cache[key]["tree"]

    def _resolve_bbox(self, surf_pos: torch.Tensor) -> tuple:
        if self.bbox_mode == "fixed":
            return self.bbox
        pos = surf_pos.detach().cpu().numpy()
        span = float(max(pos[:, 0].ptp(), pos[:, 1].ptp(), 1e-6))
        pad = self.bbox_margin * span
        return (float(pos[:, 0].min()) - pad, float(pos[:, 0].max()) + pad,
                float(pos[:, 1].min()) - pad, float(pos[:, 1].max()) + pad)

    # ------------------------------------------------------------------ #
    # geometry preprocessing (numpy / scipy, no autograd)
    # ------------------------------------------------------------------ #
    def _grid_conditioning(self, batch: dict, n_graphs: int, bbox) -> torch.Tensor:
        """``(B, n_grid, cond_dim)`` conditioning channels on the latent grid."""
        pos = batch["surf_pos"]
        device, dtype = pos.device, pos.dtype
        grid = self._grid_numpy(bbox)

        pre = batch.get("grid_sdf", None)
        sdf_list = []
        normals_list = []
        need_normals = self.conditioning == "sdf+normals"

        if pre is not None:
            sdf_t = torch.as_tensor(pre, device=device, dtype=dtype).reshape(n_graphs, -1)
            if sdf_t.shape[1] != self.n_grid:
                raise ValueError(f"grid_sdf has {sdf_t.shape[1]} nodes, "
                                 f"expected {self.n_grid}")
        else:
            batch_np = batch["batch_idx"].detach().cpu().numpy()
            pos_np = pos.detach().cpu().numpy()
            for g in range(n_graphs):
                sel = np.nonzero(batch_np == g)[0]
                if sel.size < 3:
                    raise ValueError(f"graph {g} has {sel.size} surface points; "
                                     "M2 needs at least 3 to form a polygon")
                sdf_list.append(compute_grid_sdf(pos_np[sel], grid,
                                                 use_external=self.use_external_sdf))
            sdf_t = torch.from_numpy(np.stack(sdf_list)).to(device=device, dtype=dtype)

        if self.conditioning == "sdf":
            return sdf_t.unsqueeze(-1)
        if self.conditioning == "mask":
            return (sdf_t < 0).to(dtype).unsqueeze(-1)

        # 'sdf+normals': nearest-surface-point normal at every grid node
        from scipy.spatial import cKDTree

        batch_np = batch["batch_idx"].detach().cpu().numpy()
        pos_np = pos.detach().cpu().numpy()
        nrm_np = batch["surf_normal"].detach().cpu().numpy()
        for g in range(n_graphs):
            sel = np.nonzero(batch_np == g)[0]
            _, idx = cKDTree(pos_np[sel]).query(grid, k=1, workers=1)
            normals_list.append(nrm_np[sel][np.asarray(idx).reshape(-1)])
        nrm_t = torch.from_numpy(np.stack(normals_list)).to(device=device, dtype=dtype)
        return torch.cat([sdf_t.unsqueeze(-1), nrm_t], dim=-1)

    def _surface_grid_neighbours(self, batch: dict, n_graphs: int, bbox):
        """Nearest latent-grid nodes for every surface point.

        Returns ``(flat_grid_idx (sumN, k), delta (sumN, k, 2), dist (sumN, k))``
        where ``flat_grid_idx`` indexes the flattened ``(B * n_grid)`` node stack
        so a single :func:`index_add_` handles the whole batch.
        """
        pos = batch["surf_pos"]
        device, dtype = pos.device, pos.dtype
        k = max(self.k_enc, self.k_dec)
        grid = self._grid_numpy(bbox)
        tree = self._grid_tree(bbox)
        pos_np = pos.detach().cpu().numpy()
        _, idx = tree.query(pos_np, k=min(k, grid.shape[0]), workers=1)
        idx = np.asarray(idx).reshape(pos_np.shape[0], -1)

        idx_t = torch.from_numpy(np.ascontiguousarray(idx, dtype=np.int64)).to(device)
        grid_t = torch.from_numpy(grid).to(device=device, dtype=dtype)
        # delta = x_grid - y_surf
        delta = grid_t[idx_t.reshape(-1)].reshape(idx_t.shape[0], idx_t.shape[1], 2) \
            - pos.unsqueeze(1)
        dist = torch.linalg.norm(delta, dim=-1)
        flat = batch["batch_idx"].view(-1, 1) * self.n_grid + idx_t
        return flat, idx_t, delta, dist

    # ------------------------------------------------------------------ #
    def forward(self, batch: dict) -> dict:
        n_graphs = self.check_batch(batch)
        pos = batch["surf_pos"]
        ds = batch["surf_ds"]
        batch_idx = batch["batch_idx"]
        device, dtype = pos.device, pos.dtype
        ny, nx = self.grid_res
        n_grid = self.n_grid

        bbox = self._resolve_bbox(pos)
        cond_grid = self._grid_conditioning(batch, n_graphs, bbox)      # (B, G, Cc)
        flat_idx, local_idx, delta, dist = self._surface_grid_neighbours(
            batch, n_graphs, bbox)                                      # (sumN, k, ...)

        dx = (bbox[1] - bbox[0]) / (nx - 1)
        dy = (bbox[3] - bbox[2]) / (ny - 1)
        radius = self.enc_radius_cells * float(max(dx, dy))

        # ---- kernel encoder: surface -> grid --------------------------------
        ke = self.k_enc
        a = self.surf_proj(self.point_input_features(batch))            # (sumN, W)
        d_e, dist_e = delta[:, :ke], dist[:, :ke]
        flat_e, local_e = flat_idx[:, :ke], local_idx[:, :ke]
        cond_at_node = cond_grid[batch_idx.view(-1, 1).expand_as(local_e).reshape(-1),
                                 local_e.reshape(-1)]                   # (sumN*ke, Cc)
        kin = torch.cat([d_e.reshape(-1, 2), dist_e.reshape(-1, 1), cond_at_node], dim=-1)
        kappa = self.enc_kernel(kin).reshape(d_e.shape[0], ke, self.width)
        support = (dist_e <= radius).to(dtype).unsqueeze(-1)            # (sumN, ke, 1)
        w_quad = ds.view(-1, 1, 1) * support
        contrib = (kappa * a.unsqueeze(1)) * w_quad                     # (sumN, ke, W)

        flat_flat = flat_e.reshape(-1)
        h_lat = segment_sum(contrib.reshape(-1, self.width), flat_flat, n_graphs * n_grid)
        norm = segment_sum(w_quad.reshape(-1, 1), flat_flat, n_graphs * n_grid)
        h_lat = h_lat / (norm + 1e-8)                                   # (B*G, W)

        # ---- lifting + FNO stack --------------------------------------------
        grid_np = self._grid_numpy(bbox)
        grid_t = torch.from_numpy(grid_np).to(device=device, dtype=dtype)
        grid_pe = self.pos_encoder(grid_t).unsqueeze(0).expand(n_graphs, -1, -1)
        lift_in = torch.cat([h_lat.view(n_graphs, n_grid, self.width),
                             cond_grid, grid_pe], dim=-1)
        v = self.lift(lift_in)                                          # (B, G, W)
        v = v.view(n_graphs, ny, nx, self.width).permute(0, 3, 1, 2).contiguous()
        for block in self.fno:
            v = block(v)
        v = v.permute(0, 2, 3, 1).reshape(n_graphs, n_grid, self.width)
        v = self.grid_proj(v)
        v_flat = v.reshape(n_graphs * n_grid, self.width)

        # ---- kernel decoder: grid -> surface --------------------------------
        kd = self.k_dec
        d_d, dist_d = delta[:, :kd], dist[:, :kd]
        flat_d = flat_idx[:, :kd]
        dkin = torch.cat([d_d.reshape(-1, 2), dist_d.reshape(-1, 1)], dim=-1)
        kappa_d = self.dec_kernel(dkin).reshape(d_d.shape[0], kd, self.width)
        gathered = v_flat[flat_d.reshape(-1)].reshape(d_d.shape[0], kd, self.width)
        feat = (kappa_d * gathered).mean(dim=1)                         # (sumN, W)

        # ---- readouts --------------------------------------------------------
        field, trunk = self.field_head(
            torch.cat([feat, self.point_input_features(batch)], dim=-1))
        coef = self.coef_head(trunk, batch_idx, n_graphs, batch["cond"], ds=ds)
        return self.pack_outputs(field, coef)
