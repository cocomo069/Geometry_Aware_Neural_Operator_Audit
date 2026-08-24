"""M1 -- kNN message-passing GNN baseline (spec section 5.2, "M1, graph baseline").

Node features are :math:`(x_i, \\mathbf{n}_i, \\kappa_i, c)` -- position (Fourier
encoded), outward normal, local curvature and the broadcast flow condition -- on
a k-nearest-neighbour graph over the airfoil surface points.  ``T`` rounds of

.. math::

    m_{ij}^{(t)} = \\phi_e(h_i^{(t)}, h_j^{(t)}, x_j - x_i, \\|x_j - x_i\\|),
    \\qquad
    h_i^{(t+1)} = h_i^{(t)} + \\phi_v(h_i^{(t)}, \\mathrm{agg}_j m_{ij}^{(t)})

with residual connections and post-LayerNorm.

Implementation notes
--------------------
* The graph is built with :class:`scipy.spatial.cKDTree` (CONTEXT.md section 1
  forbids compiled extensions such as ``torch_cluster``).  The loader *may*
  precompute ``batch['edge_index']``; if present it is used verbatim.
* Aggregation is :func:`torch.Tensor.index_add_` based (``segment_sum`` /
  ``segment_mean``), never ``torch_scatter``.
* Edge features are recomputed once per forward and reused across all ``T``
  rounds, so memory is ``O(E * width)`` per round with ``E = k * sumN``.

Memory on the 4 GB Pascal card (measured, 1000 surface points per sample)
------------------------------------------------------------------------
Activations dominate: the per-round edge-MLP hidden tensor is
``(k * sumN, mlp_hidden)``, i.e. ``16 * 1000 * 208`` floats per sample per
round, times 8 rounds.  Measured forward+backward peak is ~440 MiB per sample,
so ``batch_size = 4`` (1.76 GiB) is comfortable and ``batch_size = 8`` runs the
card out of VRAM into WDDM host memory (a ~10x slowdown).  Setting
``grad_checkpoint: true`` recomputes each round during backward: measured
298 MiB at ``batch_size = 4`` -- a **5.9x** reduction for only ~10% more wall
time, since the layer is memory-bandwidth bound rather than FLOP bound.  M1 is
by far the most memory-hungry of the three; M2 and M3 sit near 100 and 83 MiB
per sample.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint

from src.models.common import (
    FourierFeatures,
    SurrogateBase,
    build_mlp,
    get_edge_index,
    segment_mean,
    segment_sum,
)
from src.models.heads import CoefHead, FieldHead

__all__ = ["MessagePassingLayer", "GNNSurrogate", "DEFAULT_CONFIG"]


DEFAULT_CONFIG = {
    "hidden_dim": 128,
    "mlp_hidden": 208,
    "n_rounds": 8,          # T = 8 message-passing rounds
    "k": 16,                # kNN degree
    "n_freqs": 6,           # Fourier positional frequencies
    "activation": "gelu",
    "aggr": "mean",         # 'mean' | 'sum'
    "layer_norm": True,
    "grad_checkpoint": False,   # trade ~2.5x compute for ~5x less activation memory
    "trunk_dim": 128,
    "coef_hidden": 256,
    "coef_layers": 2,
    "coef_pooling": "ds_mean_max",
}


class MessagePassingLayer(nn.Module):
    """One residual message-passing round.

    Shapes
    ------
    ``h`` ``(sumN, H)``, ``edge_index`` ``(2, E)`` (row 0 = source ``j``,
    row 1 = destination ``i``), ``edge_attr`` ``(E, 3)`` = ``[dx, dy, |dx|]``.
    """

    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden: int,
        edge_attr_dim: int = 3,
        activation: str = "gelu",
        aggr: str = "mean",
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if aggr not in ("mean", "sum"):
            raise KeyError(f"aggr must be 'mean' or 'sum', got {aggr!r}")
        self.aggr = aggr
        self.edge_mlp = build_mlp(2 * hidden_dim + edge_attr_dim, hidden_dim, mlp_hidden,
                                  n_hidden_layers=1, activation=activation)
        self.node_mlp = build_mlp(2 * hidden_dim, hidden_dim, mlp_hidden,
                                  n_hidden_layers=1, activation=activation)
        self.norm = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        if edge_index.numel() == 0:
            agg = h.new_zeros(n, h.shape[1])
        else:
            src, dst = edge_index[0], edge_index[1]
            msg = self.edge_mlp(torch.cat([h.index_select(0, dst),
                                           h.index_select(0, src),
                                           edge_attr], dim=-1))
            reduce = segment_mean if self.aggr == "mean" else segment_sum
            agg = reduce(msg, dst, n)
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h)


class GNNSurrogate(SurrogateBase):
    """M1: kNN message-passing surrogate over airfoil surface points.

    Config keys (all optional, see :data:`DEFAULT_CONFIG`)::

        hidden_dim, mlp_hidden, n_rounds, k, n_freqs, activation, aggr,
        layer_norm, grad_checkpoint, trunk_dim, coef_hidden, coef_layers,
        coef_pooling

    ``forward(batch) -> {"p": (sumN,), "tau": (sumN, 2), "coef_head": (B, 2)}``.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        merged = dict(DEFAULT_CONFIG)
        merged.update(config or {})
        super().__init__(merged)

        hidden = int(self.cfg("hidden_dim", 128))
        mlp_hidden = int(self.cfg("mlp_hidden", 192))
        n_rounds = int(self.cfg("n_rounds", 8))
        act = str(self.cfg("activation", "gelu"))
        self.k = int(self.cfg("k", 16))
        self.grad_checkpoint = bool(self.cfg("grad_checkpoint", False))
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if n_rounds < 1:
            raise ValueError("n_rounds must be >= 1")

        self.pos_encoder = FourierFeatures(2, n_freqs=int(self.cfg("n_freqs", 6)))
        self.encoder = build_mlp(self.point_input_dim(), hidden, mlp_hidden,
                                 n_hidden_layers=1, activation=act)
        self.layers = nn.ModuleList([
            MessagePassingLayer(hidden, mlp_hidden, edge_attr_dim=3, activation=act,
                                aggr=str(self.cfg("aggr", "mean")),
                                layer_norm=bool(self.cfg("layer_norm", True)))
            for _ in range(n_rounds)
        ])
        trunk_dim = int(self.cfg("trunk_dim", hidden))
        self.field_head = FieldHead(hidden, trunk_dim=trunk_dim, n_hidden_layers=2,
                                    activation=act)
        self.coef_head = CoefHead(trunk_dim,
                                  hidden_dim=int(self.cfg("coef_hidden", 256)),
                                  n_hidden_layers=int(self.cfg("coef_layers", 2)),
                                  activation=act,
                                  pooling=str(self.cfg("coef_pooling", "ds_mean_max")))

    # ------------------------------------------------------------------ #
    @staticmethod
    def edge_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """``[x_j - x_i, ||x_j - x_i||]`` -> ``(E, 3)``."""
        if edge_index.numel() == 0:
            return pos.new_zeros((0, 3))
        src, dst = edge_index[0], edge_index[1]
        delta = pos.index_select(0, src) - pos.index_select(0, dst)
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        return torch.cat([delta, dist], dim=-1)

    def forward(self, batch: dict) -> dict:
        n_graphs = self.check_batch(batch)
        pos = batch["surf_pos"]
        batch_idx = batch["batch_idx"]

        edge_index = get_edge_index(batch, k=self.k, num_graphs=n_graphs)
        edge_attr = self.edge_features(pos, edge_index)

        h = self.encoder(self.point_input_features(batch))
        checkpointing = (self.grad_checkpoint and self.training
                         and torch.is_grad_enabled())
        for layer in self.layers:
            if checkpointing:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, edge_index, edge_attr, use_reentrant=False)
            else:
                h = layer(h, edge_index, edge_attr)

        field, trunk = self.field_head(h)
        coef = self.coef_head(trunk, batch_idx, n_graphs, batch["cond"],
                              ds=batch["surf_ds"])
        return self.pack_outputs(field, coef)
