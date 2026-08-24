"""Shared output heads.

`CoefHead` is the **same module class** used by all three backbones (M1/M2/M3),
per CONTEXT.md section 7: a pooled MLP head predicting ``(C_L, C_D)`` directly
from backbone features plus the flow condition.  Keeping it identical across
models is what makes the force-self-consistency (FSC) diagnostic of spec section
5.4 an architecture comparison rather than a head comparison.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.common import (
    COND_FEATURE_DIM,
    build_mlp,
    cond_features,
    segment_max,
    segment_mean,
    segment_weighted_mean,
)

__all__ = ["CoefHead", "FieldHead", "POOLINGS"]


POOLINGS = ("mean", "max", "mean_max", "ds_mean", "ds_mean_max")


def _pool_multiplier(pooling: str) -> int:
    if pooling not in POOLINGS:
        raise KeyError(f"unknown pooling {pooling!r}; choose from {POOLINGS}")
    return 2 if pooling.endswith("mean_max") or pooling == "mean_max" else 1


class CoefHead(nn.Module):
    """Pooled point features + flow condition -> ``(C_L, C_D)``.

    Parameters
    ----------
    feat_dim:
        Width of the per-point backbone features handed in.
    hidden_dim, n_hidden_layers, activation:
        MLP geometry.
    pooling:
        ``mean`` | ``max`` | ``mean_max`` | ``ds_mean`` | ``ds_mean_max``.
        The ``ds_*`` variants weight by the quadrature measure ``surf_ds``, so
        the pooled descriptor is a discrete surface integral rather than a
        point-density average -- the physically right thing on non-uniform
        airfoil discretizations.  Default ``ds_mean_max``.

    Shapes
    ------
    ``forward(feat (sumN, feat_dim), batch_idx (sumN,), num_graphs, cond (B, 2),
    ds (sumN,) | None) -> (B, 2)``.

    Empty graphs pool to zero (masked segment reductions), never NaN.
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2,
        activation: str = "gelu",
        pooling: str = "ds_mean_max",
        cond_dim: int = COND_FEATURE_DIM,
    ) -> None:
        super().__init__()
        if feat_dim <= 0:
            raise ValueError("feat_dim must be positive")
        if pooling not in POOLINGS:
            raise KeyError(f"unknown pooling {pooling!r}; choose from {POOLINGS}")
        self.feat_dim = int(feat_dim)
        self.pooling = pooling
        self.cond_dim = int(cond_dim)
        in_dim = self.feat_dim * _pool_multiplier(pooling) + self.cond_dim
        self.mlp = build_mlp(in_dim, 2, hidden_dim,
                             n_hidden_layers=n_hidden_layers, activation=activation)

    @property
    def in_dim(self) -> int:
        return self.feat_dim * _pool_multiplier(self.pooling) + self.cond_dim

    def pool(
        self,
        feat: torch.Tensor,
        batch_idx: torch.Tensor,
        num_graphs: int,
        ds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Masked segment pooling -> ``(B, feat_dim * multiplier)``."""
        if feat.dim() != 2:
            raise ValueError(f"feat must be (sumN, C), got {tuple(feat.shape)}")
        use_ds = self.pooling.startswith("ds_")
        if use_ds and ds is None:
            raise ValueError(f"pooling={self.pooling!r} requires surf_ds")
        parts = []
        if self.pooling in ("mean", "mean_max"):
            parts.append(segment_mean(feat, batch_idx, num_graphs))
        elif self.pooling in ("ds_mean", "ds_mean_max"):
            parts.append(segment_weighted_mean(feat, ds, batch_idx, num_graphs))
        if self.pooling in ("max", "mean_max", "ds_mean_max"):
            parts.append(segment_max(feat, batch_idx, num_graphs))
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        feat: torch.Tensor,
        batch_idx: torch.Tensor,
        num_graphs: int,
        cond: torch.Tensor,
        ds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``(B, 2)`` = ``(C_L, C_D)`` in normalized space."""
        if feat.shape[-1] != self.feat_dim:
            raise ValueError(f"expected feat width {self.feat_dim}, got {feat.shape[-1]}")
        pooled = self.pool(feat, batch_idx, num_graphs, ds=ds)
        cfeat = cond_features(cond) if cond.shape[-1] == 2 else cond
        if cfeat.shape[0] != num_graphs:
            raise ValueError(f"cond has {cfeat.shape[0]} rows, expected {num_graphs}")
        return self.mlp(torch.cat([pooled, cfeat], dim=-1))

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"feat_dim={self.feat_dim}, pooling={self.pooling}, in_dim={self.in_dim}"


class FieldHead(nn.Module):
    """Per-point trunk + linear projection to ``(p, tau_x, tau_y)``.

    Exposes the trunk output as well, so the same features feed
    :class:`CoefHead` -- that shared trunk is what makes ``C^int`` (integrated
    from ``p``/``tau``) and ``C^head`` genuinely two readouts of one backbone.
    """

    def __init__(
        self,
        in_dim: int,
        trunk_dim: int = 256,
        n_hidden_layers: int = 2,
        activation: str = "gelu",
        out_dim: int = 3,
    ) -> None:
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError("FieldHead needs at least one trunk layer")
        self.trunk = build_mlp(in_dim, trunk_dim, trunk_dim,
                               n_hidden_layers=n_hidden_layers - 1,
                               activation=activation, final_activation=True)
        self.out = nn.Linear(trunk_dim, out_dim)
        self.trunk_dim = int(trunk_dim)

    def forward(self, x: torch.Tensor):
        """``(sumN, in_dim) -> (field (sumN, 3), trunk (sumN, trunk_dim))``."""
        h = self.trunk(x)
        return self.out(h), h
