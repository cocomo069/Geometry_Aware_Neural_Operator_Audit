"""Surrogate architectures M1/M2/M3 and the shared training objective.

All three subclass :class:`~src.models.common.SurrogateBase` and honour the
frozen interface of CONTEXT.md section 7::

    forward(batch: dict) -> {"p": (sumN,), "tau": (sumN, 2), "coef_head": (B, 2)}

Registry keys match the config filenames of CONTEXT.md section 10
(``configs/{gnn,sdf_fno,transolver}.yaml``); ``m1``/``m2``/``m3`` and a couple of
module-name spellings are accepted as aliases.
"""

from __future__ import annotations

from typing import Optional, Type

from src.models.common import (
    FourierFeatures,
    SurrogateBase,
    build_mlp,
    count_params,
    segment_mean,
    segment_sum,
)
from src.models.geo_transformer import GeoTransformerSurrogate
from src.models.gnn import GNNSurrogate
from src.models.heads import CoefHead, FieldHead
from src.models.sdf_fno import SDFFNOSurrogate

__all__ = [
    "SurrogateBase",
    "GNNSurrogate",
    "SDFFNOSurrogate",
    "GeoTransformerSurrogate",
    "CoefHead",
    "FieldHead",
    "FourierFeatures",
    "build_mlp",
    "count_params",
    "segment_mean",
    "segment_sum",
    "MODEL_REGISTRY",
    "build_model",
    "available_models",
]


#: canonical name -> class
MODEL_REGISTRY: dict = {
    "gnn": GNNSurrogate,
    "sdf_fno": SDFFNOSurrogate,
    "transolver": GeoTransformerSurrogate,
}

_ALIASES: dict = {
    "m1": "gnn",
    "m2": "sdf_fno",
    "m3": "transolver",
    "graph": "gnn",
    "gino": "sdf_fno",
    "fno": "sdf_fno",
    "geo_transformer": "transolver",
    "geotransformer": "transolver",
}


def available_models() -> list:
    """Canonical model names accepted by :func:`build_model`."""
    return sorted(MODEL_REGISTRY)


def build_model(name: str, config: Optional[dict] = None) -> SurrogateBase:
    """Instantiate a surrogate by name from a plain config dict.

    ``build_model('gnn', {'hidden_dim': 32})`` -- unspecified keys fall back to
    the model DEFAULT_CONFIG, so tiny CPU test configs are one-key overrides.
    """
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    cls: Optional[Type[SurrogateBase]] = MODEL_REGISTRY.get(key)
    if cls is None:
        raise KeyError(f"unknown model {name!r}; available: {available_models()} "
                       f"(aliases: {sorted(_ALIASES)})")
    return cls(config or {})
