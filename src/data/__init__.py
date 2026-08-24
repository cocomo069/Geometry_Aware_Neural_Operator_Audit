"""Data layer: official/derived splits, raw->npz cache, torch datasets.

Modules
-------
splits
    Deterministic construction of the six split manifests (CONTEXT.md sec. 5).
airfrans_loader
    ``AirfransSurfaceDataset`` + ``collate`` producing PyG-style batches
    (CONTEXT.md sec. 4).
"""

from src.data.splits import (
    SPLIT_NAMES,
    build_all_splits,
    parse_sim_name,
    read_raw_manifest,
    write_all_splits,
)

__all__ = [
    "SPLIT_NAMES",
    "build_all_splits",
    "parse_sim_name",
    "read_raw_manifest",
    "write_all_splits",
]
