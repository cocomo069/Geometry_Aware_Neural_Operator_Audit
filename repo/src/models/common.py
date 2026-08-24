"""Shared building blocks for the surrogate models (M1/M2/M3).

Everything here is pure PyTorch + numpy/scipy: no compiled graph extensions
(``torch_scatter``, ``torch_cluster``, PyG) are used anywhere -- see CONTEXT.md
section 1.  Scatter reductions go through :func:`torch.Tensor.index_add_` /
:func:`torch.Tensor.scatter_reduce_`; k-nearest-neighbour graphs go through
:class:`scipy.spatial.cKDTree` on CPU.

Shape conventions (CONTEXT.md sections 4 and 7)
----------------------------------------------
A *batch* is a plain ``dict`` with concatenated (ragged, un-padded) points:

===============  =================  =========================================
key              shape              meaning
===============  =================  =========================================
``surf_pos``     ``(sumN, 2)``      surface point coordinates
``surf_normal``  ``(sumN, 2)``      outward unit normals (into the fluid)
``surf_ds``      ``(sumN,)``        quadrature weight (facet length share)
``cond``         ``(B, 2)``         freestream velocity vector (u_x, u_y)
``batch_idx``    ``(sumN,)`` long   graph membership, non-decreasing
``curvature``    ``(sumN,)``        *optional*; zeros are used if absent
``edge_index``   ``(2, E)`` long    *optional*; precomputed kNN graph (M1)
``grid_sdf``     ``(B, H, W)``      *optional*; precomputed latent-grid SDF (M2)
===============  =================  =========================================

Models return ``{"p": (sumN,), "tau": (sumN, 2), "coef_head": (B, 2)}`` in
normalized space, with ``coef_head = (C_L, C_D)``.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "ACTIVATIONS",
    "get_activation",
    "build_mlp",
    "FourierFeatures",
    "count_params",
    "param_summary",
    "segment_sum",
    "segment_count",
    "segment_mean",
    "segment_max",
    "segment_weighted_mean",
    "to_dense_batch",
    "cond_features",
    "COND_FEATURE_DIM",
    "knn_edges",
    "get_edge_index",
    "SurrogateBase",
]


# --------------------------------------------------------------------------- #
# activations / MLP builder
# --------------------------------------------------------------------------- #

ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
}


def get_activation(name: str) -> nn.Module:
    """Return a fresh activation module by (lowercase) name."""
    key = str(name).lower()
    if key not in ACTIVATIONS:
        raise KeyError(f"unknown activation {name!r}; choose from {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[key]()


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    n_hidden_layers: int = 1,
    activation: str = "gelu",
    layer_norm: bool = False,
    final_activation: bool = False,
    bias: bool = True,
) -> nn.Sequential:
    """Build ``in_dim -> [hidden_dim] * n_hidden_layers -> out_dim``.

    Parameters
    ----------
    n_hidden_layers:
        Number of *hidden* Linear layers.  ``0`` yields a bare linear map.
    layer_norm:
        Apply :class:`~torch.nn.LayerNorm` after every hidden activation.
    final_activation:
        Whether to apply the activation after the output layer too.
    """
    if in_dim <= 0 or out_dim <= 0 or hidden_dim <= 0:
        raise ValueError(f"bad MLP dims: in={in_dim} hidden={hidden_dim} out={out_dim}")
    if n_hidden_layers < 0:
        raise ValueError("n_hidden_layers must be >= 0")

    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_hidden_layers):
        layers.append(nn.Linear(d, hidden_dim, bias=bias))
        layers.append(get_activation(activation))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim, bias=bias))
    if final_activation:
        layers.append(get_activation(activation))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# Fourier positional features
# --------------------------------------------------------------------------- #

class FourierFeatures(nn.Module):
    r"""Deterministic axis-aligned Fourier positional encoding.

    :math:`\gamma(x) = [\,x,\ \sin(\sigma b^{\ell} x),\ \cos(\sigma b^{\ell} x)
    \,]_{\ell=0}^{L-1}` applied elementwise to each of the ``in_dim``
    coordinates.  No learnable parameters, hence exactly reproducible.

    Output dimension is ``in_dim * (int(include_input) + 2 * n_freqs)``.
    """

    def __init__(
        self,
        in_dim: int = 2,
        n_freqs: int = 6,
        include_input: bool = True,
        base: float = 2.0,
        scale: float = math.pi,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        if n_freqs < 0:
            raise ValueError("n_freqs must be >= 0")
        self.in_dim = int(in_dim)
        self.n_freqs = int(n_freqs)
        self.include_input = bool(include_input)
        freqs = scale * (base ** torch.arange(self.n_freqs, dtype=torch.float32))
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        return self.in_dim * (int(self.include_input) + 2 * self.n_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., in_dim) -> (..., out_dim)``."""
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"expected last dim {self.in_dim}, got {tuple(x.shape)}")
        outs = [x] if self.include_input else []
        if self.n_freqs > 0:
            scaled = x.unsqueeze(-1) * self.freqs.to(x.dtype)      # (..., in_dim, L)
            flat = scaled.reshape(*x.shape[:-1], self.in_dim * self.n_freqs)
            outs.append(torch.sin(flat))
            outs.append(torch.cos(flat))
        return torch.cat(outs, dim=-1)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"in_dim={self.in_dim}, n_freqs={self.n_freqs}, out_dim={self.out_dim}"


# --------------------------------------------------------------------------- #
# parameter counting
# --------------------------------------------------------------------------- #

def count_params(module: nn.Module, trainable_only: bool = True) -> int:
    """Total number of (real-valued) parameter entries in ``module``.

    Complex parameters are stored as real tensors with a trailing size-2 axis
    (see :class:`~src.models.sdf_fno.SpectralConv2d`), so this counts real
    degrees of freedom, which is the meaningful budget quantity.
    """
    if trainable_only:
        params = (p for p in module.parameters() if p.requires_grad)
    else:
        params = module.parameters()
    return int(sum(p.numel() for p in params))


def param_summary(module: nn.Module, top: int = 16) -> str:
    """Human-readable per-submodule parameter breakdown (for tuning budgets)."""
    rows = [(name, count_params(child, trainable_only=False))
            for name, child in module.named_children()]
    rows.sort(key=lambda r: -r[1])
    total = count_params(module, trainable_only=False)
    direct = sum(p.numel() for p in module.parameters(recurse=False))
    lines = [f"total: {total:,}"]
    if direct:
        lines.append(f"  {'<direct params>':<24s} {direct:>10,d}")
    for name, n in rows[:top]:
        lines.append(f"  {name:<24s} {n:>10,d}  ({100.0 * n / max(total, 1):5.1f}%)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# segment (scatter) reductions -- native torch only
# --------------------------------------------------------------------------- #

def _check_segment_args(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if index.dim() != 1:
        raise ValueError(f"index must be 1-D, got {tuple(index.shape)}")
    if src.shape[0] != index.shape[0]:
        raise ValueError(f"src/index length mismatch: {src.shape[0]} vs {index.shape[0]}")
    return index if index.dtype == torch.long else index.long()


def segment_sum(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Scatter-add ``src`` ``(N, ...)`` into ``(num_segments, ...)`` by ``index``."""
    index = _check_segment_args(src, index)
    out = src.new_zeros((num_segments,) + tuple(src.shape[1:]))
    return out.index_add(0, index, src)


def segment_count(index: torch.Tensor, num_segments: int,
                  dtype=torch.float32, device=None) -> torch.Tensor:
    """Number of entries per segment, shape ``(num_segments,)``."""
    device = index.device if device is None else device
    return torch.bincount(index.long(), minlength=num_segments).to(dtype=dtype, device=device)


def segment_mean(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Masked segment mean; **empty segments yield exactly zero** (no NaNs)."""
    index = _check_segment_args(src, index)
    total = segment_sum(src, index, num_segments)
    cnt = segment_count(index, num_segments, dtype=src.dtype, device=src.device)
    cnt = cnt.view((num_segments,) + (1,) * (src.dim() - 1))
    return total / cnt.clamp(min=1.0)


def segment_max(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Masked segment max; empty segments yield zero."""
    index = _check_segment_args(src, index)
    out = src.new_zeros((num_segments,) + tuple(src.shape[1:]))
    idx = index.view((-1,) + (1,) * (src.dim() - 1)).expand_as(src)
    return out.scatter_reduce(0, idx, src, reduce="amax", include_self=False)


def segment_weighted_mean(
    src: torch.Tensor,
    weight: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Quadrature-weighted segment mean, ``sum(w * x) / sum(w)``."""
    index = _check_segment_args(src, index)
    w = weight.reshape((-1,) + (1,) * (src.dim() - 1)).to(src.dtype)
    num = segment_sum(src * w, index, num_segments)
    den = segment_sum(w.expand_as(src), index, num_segments)
    return num / (den + eps)


def to_dense_batch(
    x: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: Optional[int] = None,
    fill_value: float = 0.0,
):
    """Pad a ragged ``(sumN, C)`` stack into ``(B, Nmax, C)`` plus a bool mask.

    Returns ``(dense, mask, slot)`` where ``slot`` ``(sumN,)`` is the position
    of each point inside its own graph, so the inverse is simply
    ``dense[batch_idx, slot]``.  Requires ``batch_idx`` to be non-decreasing
    (guaranteed by the loader's concatenating collate, CONTEXT.md section 4).

    Memory is ``O(B * Nmax * C)`` -- never ``O(N^2)``.
    """
    batch_idx = batch_idx.long()
    if num_graphs is None:
        num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0
    counts = torch.bincount(batch_idx, minlength=num_graphs)
    nmax = int(counts.max().item()) if counts.numel() else 0
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    slot = torch.arange(x.shape[0], device=x.device) - offsets[batch_idx]
    dense = x.new_full((num_graphs, nmax) + tuple(x.shape[1:]), fill_value)
    dense[batch_idx, slot] = x
    mask = torch.zeros((num_graphs, nmax), dtype=torch.bool, device=x.device)
    mask[batch_idx, slot] = True
    return dense, mask, slot


# --------------------------------------------------------------------------- #
# conditioning features
# --------------------------------------------------------------------------- #

COND_FEATURE_DIM = 5


def cond_features(cond: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Expand the raw freestream vector into scale/direction features.

    ``cond`` ``(B, 2)`` = ``(u_inf_x, u_inf_y)`` -> ``(B, 5)`` =
    ``[u_x, u_y, |u|, cos(alpha), sin(alpha)]``.  The direction cosines make the
    angle of attack directly available without the network having to divide.
    """
    if cond.dim() != 2 or cond.shape[-1] != 2:
        raise ValueError(f"cond must be (B, 2), got {tuple(cond.shape)}")
    mag = torch.linalg.norm(cond, dim=-1, keepdim=True)
    direction = cond / (mag + eps)
    return torch.cat([cond, mag, direction], dim=-1)


# --------------------------------------------------------------------------- #
# kNN graph construction (scipy cKDTree, CPU, no compiled torch extensions)
# --------------------------------------------------------------------------- #

def knn_edges(
    pos: torch.Tensor,
    k: int,
    batch_idx: Optional[torch.Tensor] = None,
    num_graphs: Optional[int] = None,
    loop: bool = False,
) -> torch.Tensor:
    """Build a per-graph k-nearest-neighbour edge list with ``scipy.cKDTree``.

    Parameters
    ----------
    pos: ``(sumN, D)`` point coordinates (any device; moved to CPU numpy).
    k: neighbours per node, excluding self unless ``loop``.
    batch_idx: ``(sumN,)`` graph membership; ``None`` means a single graph.

    Returns
    -------
    ``edge_index`` ``(2, E)`` long tensor on ``pos.device`` with row 0 = source
    ``j`` and row 1 = destination ``i`` (messages flow ``j -> i``).  Edges never
    cross graphs.
    """
    from scipy.spatial import cKDTree  # local import keeps module import cheap

    if pos.dim() != 2:
        raise ValueError(f"pos must be (N, D), got {tuple(pos.shape)}")
    if k < 1:
        raise ValueError("k must be >= 1")
    n = pos.shape[0]
    device = pos.device
    pos_np = np.ascontiguousarray(pos.detach().cpu().numpy(), dtype=np.float64)
    if batch_idx is None:
        batch_np = np.zeros(n, dtype=np.int64)
        num_graphs = 1
    else:
        batch_np = np.ascontiguousarray(batch_idx.detach().cpu().numpy(), dtype=np.int64)
        if num_graphs is None:
            num_graphs = int(batch_np.max()) + 1 if n else 0

    srcs: list[np.ndarray] = []
    dsts: list[np.ndarray] = []
    for g in range(num_graphs):
        sel = np.nonzero(batch_np == g)[0]
        if sel.size == 0:
            continue
        pts = pos_np[sel]
        kq = int(min(k + (0 if loop else 1), sel.size))
        tree = cKDTree(pts)
        _, nbr = tree.query(pts, k=kq, workers=1)
        nbr = np.asarray(nbr).reshape(sel.size, -1)
        if not loop and nbr.shape[1] > 1:
            nbr = nbr[:, 1:]          # column 0 is the point itself
        dst_local = np.repeat(np.arange(sel.size), nbr.shape[1])
        src_local = nbr.reshape(-1)
        srcs.append(sel[src_local])
        dsts.append(sel[dst_local])

    if not srcs:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    edge = np.stack([np.concatenate(srcs), np.concatenate(dsts)], axis=0)
    return torch.from_numpy(np.ascontiguousarray(edge, dtype=np.int64)).to(device)


def get_edge_index(
    batch: dict,
    k: int,
    pos_key: str = "surf_pos",
    num_graphs: Optional[int] = None,
) -> torch.Tensor:
    """Use ``batch['edge_index']`` if the loader precomputed one, else build it.

    This is the documented hand-off with A1's loader: it *may* ship an
    ``edge_index`` ``(2, E)`` with the batch to skip per-step kD-tree work, and
    the model degrades gracefully to building the graph on the fly.
    """
    edge_index = batch.get("edge_index", None)
    if edge_index is not None:
        edge_index = torch.as_tensor(edge_index, device=batch[pos_key].device).long()
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be (2, E), got {tuple(edge_index.shape)}")
        return edge_index
    return knn_edges(batch[pos_key], k=k, batch_idx=batch.get("batch_idx"),
                     num_graphs=num_graphs)


# --------------------------------------------------------------------------- #
# base class
# --------------------------------------------------------------------------- #

class SurrogateBase(nn.Module):
    """Common interface for M1/M2/M3 (CONTEXT.md section 7, FROZEN).

    Subclasses implement ``forward(batch: dict) -> dict`` returning
    ``{"p": (sumN,), "tau": (sumN, 2), "coef_head": (B, 2)}``.  Constructors
    take a plain config ``dict``; no global state; deterministic given
    ``src/utils/seed.py::seed_all(seed)``.
    """

    REQUIRED_KEYS: Sequence[str] = ("surf_pos", "surf_normal", "surf_ds", "cond", "batch_idx")

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__()
        self.config = dict(config or {})

    # -- config helper ----------------------------------------------------- #
    def cfg(self, key: str, default):
        """Config lookup with a default; missing keys fall back silently."""
        value = self.config.get(key, default)
        return default if value is None else value

    # -- batch validation -------------------------------------------------- #
    @staticmethod
    def check_batch(batch: dict) -> int:
        """Validate the batch dict; return the number of graphs ``B``.

        Asserts early and loudly -- no silent shape coercion (CONTEXT.md s.11).
        """
        for key in SurrogateBase.REQUIRED_KEYS:
            if key not in batch:
                raise KeyError(f"batch is missing required key {key!r}")
        pos = batch["surf_pos"]
        normal = batch["surf_normal"]
        ds = batch["surf_ds"]
        cond = batch["cond"]
        batch_idx = batch["batch_idx"]
        if pos.dim() != 2 or pos.shape[1] != 2:
            raise ValueError(f"surf_pos must be (sumN, 2), got {tuple(pos.shape)}")
        n = pos.shape[0]
        if tuple(normal.shape) != (n, 2):
            raise ValueError(f"surf_normal must be {(n, 2)}, got {tuple(normal.shape)}")
        if tuple(ds.shape) != (n,):
            raise ValueError(f"surf_ds must be {(n,)}, got {tuple(ds.shape)}")
        if tuple(batch_idx.shape) != (n,):
            raise ValueError(f"batch_idx must be {(n,)}, got {tuple(batch_idx.shape)}")
        if cond.dim() != 2 or cond.shape[1] != 2:
            raise ValueError(f"cond must be (B, 2), got {tuple(cond.shape)}")
        if batch_idx.dtype != torch.long:
            raise TypeError(f"batch_idx must be int64, got {batch_idx.dtype}")
        if n and bool((batch_idx[1:] < batch_idx[:-1]).any()):
            raise ValueError("batch_idx must be non-decreasing (concatenating collate)")
        n_graphs = int(cond.shape[0])
        if n and int(batch_idx.max().item()) >= n_graphs:
            raise ValueError("batch_idx refers to more graphs than cond has rows")
        curv = batch.get("curvature", None)
        if curv is not None and tuple(curv.shape) not in ((n,), (n, 1)):
            raise ValueError(f"curvature must be {(n,)} or {(n, 1)}, got {tuple(curv.shape)}")
        return n_graphs

    @staticmethod
    def get_curvature(batch: dict) -> torch.Tensor:
        """``(sumN, 1)`` curvature, zero-filled when the loader omits it.

        A2 owns curvature estimation (``src/geometry/normals.py``); until the
        cache carries it, models see a constant-zero channel, which is a no-op
        for the first Linear layer rather than a silent error.
        """
        pos = batch["surf_pos"]
        curv = batch.get("curvature", None)
        if curv is None:
            return pos.new_zeros((pos.shape[0], 1))
        curv = torch.as_tensor(curv, device=pos.device, dtype=pos.dtype)
        return curv.reshape(-1, 1)

    # -- conditioning ------------------------------------------------------ #
    def make_cond_encoder(
        self,
        out_dim: int,
        hidden_dim: int = 64,
        n_hidden_layers: int = 1,
        activation: str = "gelu",
    ) -> nn.Module:
        """Build the shared ``cond -> embedding`` MLP (call from ``__init__``)."""
        return build_mlp(COND_FEATURE_DIM, out_dim, hidden_dim,
                         n_hidden_layers=n_hidden_layers, activation=activation)

    @staticmethod
    def broadcast_cond(cond_feat: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """``(B, C)`` graph-level features -> ``(sumN, C)`` per-point features."""
        return cond_feat.index_select(0, batch_idx.long())

    @staticmethod
    def cond_features(cond: torch.Tensor) -> torch.Tensor:
        """See :func:`cond_features`."""
        return cond_features(cond)

    def point_input_features(self, batch: dict) -> torch.Tensor:
        """Assemble ``[gamma(pos), normal, curvature, cond]`` -> ``(sumN, F)``.

        Requires ``self.pos_encoder`` (a :class:`FourierFeatures`) to exist.
        ``self.POINT_FEATURE_DIM`` reports ``F`` for layer sizing.
        """
        pos = batch["surf_pos"]
        feats = [self.pos_encoder(pos),
                 batch["surf_normal"],
                 self.get_curvature(batch),
                 self.broadcast_cond(cond_features(batch["cond"]), batch["batch_idx"])]
        return torch.cat(feats, dim=-1)

    def point_input_dim(self) -> int:
        """Dimension produced by :meth:`point_input_features`."""
        return self.pos_encoder.out_dim + 2 + 1 + COND_FEATURE_DIM

    # -- misc -------------------------------------------------------------- #
    def count_params(self, trainable_only: bool = True) -> int:
        return count_params(self, trainable_only=trainable_only)

    def param_summary(self) -> str:  # pragma: no cover - diagnostic
        return param_summary(self)

    def forward(self, batch: dict) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def pack_outputs(field: torch.Tensor, coef: torch.Tensor) -> dict:
        """``field`` ``(sumN, 3)`` + ``coef`` ``(B, 2)`` -> the frozen output dict."""
        if field.dim() != 2 or field.shape[1] != 3:
            raise ValueError(f"field head must be (sumN, 3), got {tuple(field.shape)}")
        if coef.dim() != 2 or coef.shape[1] != 2:
            raise ValueError(f"coef_head must be (B, 2), got {tuple(coef.shape)}")
        return {
            "p": field[:, 0].contiguous(),
            "tau": field[:, 1:3].contiguous(),
            "coef_head": coef,
        }
