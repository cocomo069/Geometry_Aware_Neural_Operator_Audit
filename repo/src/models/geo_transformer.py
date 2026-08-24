"""M3 -- Transolver-style geometry transformer (spec section 5.2, "M3").

Per layer, the ``N`` surface points are *softly assigned* to ``M << N`` physics
slices, standard multi-head attention runs **among the slice tokens only**, and
the attended tokens are scattered back with the same assignment weights:

.. math::

    w_{im} = \\frac{\\exp\\langle q_m, h_i\\rangle}{\\sum_{m'}\\exp\\langle q_{m'},
    h_i\\rangle},\\qquad
    z_m = \\frac{\\sum_i w_{im} h_i}{\\sum_i w_{im}},\\qquad
    h_i' = \\sum_m w_{im}\\tilde z_m .

Cost is :math:`\\mathcal{O}(NM + M^2)`, never :math:`\\mathcal{O}(N^2)`: the
largest tensor materialized is the assignment map ``(B, heads, Nmax, M)`` and
the slice-attention map ``(B, heads, M, M)``.  With ``M = 32``, ``heads = 8``
and ~1k surface points per airfoil this is a few MB -- comfortable inside the
4 GB Pascal budget (CONTEXT.md section 1).

Ragged batches are densified to ``(B, Nmax, d)`` with a boolean mask; the mask
zeroes the assignment weights of padded slots so slice statistics see only real
points.  Consequently the per-sample outputs are identical whether a sample is
run alone or inside a batch (asserted by ``tests/test_models.py``).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import (
    FourierFeatures,
    SurrogateBase,
    build_mlp,
    to_dense_batch,
)
from src.models.heads import CoefHead, FieldHead

__all__ = ["PhysicsAttention", "TransolverBlock", "GeoTransformerSurrogate",
           "DEFAULT_CONFIG"]


DEFAULT_CONFIG = {
    "dim": 128,
    "n_layers": 4,
    "n_heads": 8,
    "n_slices": 32,         # M
    "ffn_mult": 4,
    "n_freqs": 6,
    "activation": "gelu",
    "slice_temperature": 0.5,
    "trunk_dim": 384,
    "coef_hidden": 256,
    "coef_layers": 2,
    "coef_pooling": "ds_mean_max",
}


class PhysicsAttention(nn.Module):
    """Soft slice assignment -> attention among slices -> scatter back.

    Shapes
    ------
    ``forward(x (B, N, d), mask (B, N) bool) -> (B, N, d)``.

    Intermediates: assignment ``(B, H, N, M)``, slice tokens ``(B, H, M, Dh)``,
    slice-attention logits ``(B, H, M, M)``.  No ``N x N`` tensor exists.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        n_slices: int = 32,
        temperature: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by n_heads {n_heads}")
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.dim // self.n_heads
        self.n_slices = int(n_slices)
        self.eps = float(eps)

        self.in_project_x = nn.Linear(dim, dim)
        self.in_project_fx = nn.Linear(dim, dim)
        self.slice_weight = nn.Parameter(torch.empty(self.n_heads, self.head_dim,
                                                     self.n_slices))
        self.slice_bias = nn.Parameter(torch.zeros(self.n_heads, self.n_slices))
        # learnable per-head assignment temperature (Transolver's "slice temp")
        self.log_temperature = nn.Parameter(
            torch.full((self.n_heads,), math.log(max(temperature, 1e-3))))
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.slice_weight, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, d = x.shape
        if d != self.dim:
            raise ValueError(f"expected width {self.dim}, got {d}")
        h, dh, m = self.n_heads, self.head_dim, self.n_slices

        fx = self.in_project_fx(x).view(b, n, h, dh).permute(0, 2, 1, 3)   # (B,H,N,Dh)
        px = self.in_project_x(x).view(b, n, h, dh).permute(0, 2, 1, 3)    # (B,H,N,Dh)

        logits = torch.einsum("bhnd,hdm->bhnm", px, self.slice_weight)
        logits = logits + self.slice_bias.view(1, h, 1, m)
        temp = torch.exp(self.log_temperature).clamp(min=1e-3).view(1, h, 1, 1)
        w = torch.softmax(logits / temp, dim=-1)                           # (B,H,N,M)
        if mask is not None:
            w = w * mask.view(b, 1, n, 1).to(w.dtype)

        den = w.sum(dim=2)                                                 # (B,H,M)
        num = torch.einsum("bhnm,bhnd->bhmd", w, fx)                       # (B,H,M,Dh)
        z = num / (den.unsqueeze(-1) + self.eps)                           # slice tokens

        # standard MHA among the M slice tokens: O(M^2) with M = 32
        z_flat = z.permute(0, 2, 1, 3).reshape(b, m, d)                    # (B,M,d)
        qkv = self.to_qkv(z_flat).view(b, m, 3, h, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                   # (B,H,M,Dh)
        z_att = F.scaled_dot_product_attention(q, k, v)                    # (B,H,M,Dh)

        out = torch.einsum("bhnm,bhmd->bhnd", w, z_att)                    # (B,H,N,Dh)
        out = out.permute(0, 2, 1, 3).reshape(b, n, d)
        return self.to_out(out)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (f"dim={self.dim}, n_heads={self.n_heads}, "
                f"head_dim={self.head_dim}, n_slices={self.n_slices}")


class TransolverBlock(nn.Module):
    """Pre-norm physics-attention block: ``x + Attn(LN x)``, then ``x + FFN(LN x)``."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        n_slices: int = 32,
        ffn_mult: int = 4,
        activation: str = "gelu",
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PhysicsAttention(dim, n_heads=n_heads, n_slices=n_slices,
                                     temperature=temperature)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = build_mlp(dim, dim, dim * int(ffn_mult), n_hidden_layers=1,
                             activation=activation)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class GeoTransformerSurrogate(SurrogateBase):
    """M3: Transolver-style surrogate, ``M = 32`` slices, 4 layers, ``d = 128``.

    Config keys (see :data:`DEFAULT_CONFIG`)::

        dim, n_layers, n_heads, n_slices, ffn_mult, n_freqs, activation,
        slice_temperature, trunk_dim, coef_hidden, coef_layers, coef_pooling

    ``forward(batch) -> {"p": (sumN,), "tau": (sumN, 2), "coef_head": (B, 2)}``.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        merged = dict(DEFAULT_CONFIG)
        merged.update(config or {})
        super().__init__(merged)

        dim = int(self.cfg("dim", 128))
        n_layers = int(self.cfg("n_layers", 4))
        act = str(self.cfg("activation", "gelu"))
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        self.pos_encoder = FourierFeatures(2, n_freqs=int(self.cfg("n_freqs", 6)))
        self.encoder = build_mlp(self.point_input_dim(), dim, 2 * dim,
                                 n_hidden_layers=1, activation=act)
        self.blocks = nn.ModuleList([
            TransolverBlock(dim,
                            n_heads=int(self.cfg("n_heads", 8)),
                            n_slices=int(self.cfg("n_slices", 32)),
                            ffn_mult=int(self.cfg("ffn_mult", 4)),
                            activation=act,
                            temperature=float(self.cfg("slice_temperature", 0.5)))
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        trunk_dim = int(self.cfg("trunk_dim", 256))
        self.field_head = FieldHead(dim, trunk_dim=trunk_dim, n_hidden_layers=2,
                                    activation=act)
        self.coef_head = CoefHead(trunk_dim,
                                  hidden_dim=int(self.cfg("coef_hidden", 256)),
                                  n_hidden_layers=int(self.cfg("coef_layers", 2)),
                                  activation=act,
                                  pooling=str(self.cfg("coef_pooling", "ds_mean_max")))

    def forward(self, batch: dict) -> dict:
        n_graphs = self.check_batch(batch)
        batch_idx = batch["batch_idx"]

        h = self.encoder(self.point_input_features(batch))          # (sumN, d)
        dense, mask, slot = to_dense_batch(h, batch_idx, n_graphs)  # (B, Nmax, d)
        for block in self.blocks:
            dense = block(dense, mask)
        dense = self.norm(dense)
        h = dense[batch_idx, slot]                                  # back to (sumN, d)

        field, trunk = self.field_head(h)
        coef = self.coef_head(trunk, batch_idx, n_graphs, batch["cond"],
                              ds=batch["surf_ds"])
        return self.pack_outputs(field, coef)
