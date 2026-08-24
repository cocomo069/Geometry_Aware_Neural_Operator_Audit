"""Torch dataset + PyG-style collate over the AirfRANS ``.npz`` cache.

Implements CONTEXT.md section 4 (FROZEN).  Nothing here imports ``airfrans`` or
``pyvista``: the raw dataset is consumed once by ``scripts/build_cache.py`` and
everything downstream reads the cache.

Batching convention
-------------------
Point clouds are **concatenated, never padded**.  A batch of ``B`` simulations
with ``Ns_i`` surface points each carries::

    surf_pos    (sum Ns, 2)     surf_normal (sum Ns, 2)
    surf_ds     (sum Ns,)       surf_p      (sum Ns,)
    surf_tau    (sum Ns, 2)     batch_idx   (sum Ns,)  int64, values in [0, B)
    cond        (B, 2)          cl_true     (B,)       cd_true (B,)
    ptr         (B + 1,)        n_surf      (B,)       num_graphs = B
    sim_name    list[str] of length B

``batch_idx[j] == i`` means point ``j`` belongs to simulation ``i``; this is
what ``torch.index_add_`` based scatter reductions in ``src/models`` and the
per-sample force integration in ``src/physics`` consume.

Normalisation
-------------
Stats live in ``data/processed/airfrans/norm_stats.json`` and are computed on
the **train subset of the ``full`` split only**.  Fields are standardised
``(x - mean) / std`` per component.  Only the fields in
:data:`DEFAULT_NORMALIZE_FIELDS` are touched by default:

* ``surf_p``, ``surf_tau``, ``cl_true``, ``cd_true`` -- targets, so model
  outputs live in normalised space as CONTEXT.md section 7 requires;
* ``cond``, ``vol_u``, ``vol_p``, ``vol_nut`` -- inputs whose physical scale is
  large (tens of m/s, thousands of m^2/s^2).

Deliberately **not** normalised by default:

* ``surf_pos`` / ``vol_pos`` -- the chord is 1 m so positions are already O(1),
  and per-component standardisation would stretch y ~10x relative to x,
  destroying the aspect ratio that the geometry-aware models depend on;
* ``surf_normal`` -- unit vectors already;
* ``surf_ds``, ``vol_sdf`` -- metric quantities in metres, consumed directly by
  the force integral and the no-slip residual.

``surf_ds`` and ``surf_normal`` are therefore *always* physical, which is what
``src/physics/force_integration.py`` expects.  Physical copies of every
normalised field are additionally exposed under a ``_phys`` suffix so the eval
harness never has to guess.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.splits import load_split

__all__ = [
    "DEFAULT_NORMALIZE_FIELDS",
    "SURFACE_KEYS",
    "VOLUME_KEYS",
    "NormStats",
    "AirfransSurfaceDataset",
    "collate",
    "make_dataloader",
]

#: Point-wise surface arrays concatenated along dim 0 by :func:`collate`.
SURFACE_KEYS: tuple[str, ...] = (
    "surf_pos",
    "surf_normal",
    "surf_ds",
    "surf_p",
    "surf_tau",
)

#: Point-wise volume arrays (only present when ``load_volume=True``).
VOLUME_KEYS: tuple[str, ...] = (
    "vol_pos",
    "vol_u",
    "vol_p",
    "vol_nut",
    "vol_sdf",
)

#: Per-simulation scalars/vectors stacked to a leading batch dimension.
_SIM_VECTOR_KEYS: tuple[str, ...] = ("cond",)
_SIM_SCALAR_KEYS: tuple[str, ...] = (
    "cl_true",
    "cd_true",
    "re",
    "aoa_deg",
    "u_inf_mag",
    "rho",
    "nu",
    "perimeter",
)

#: See the module docstring for the rationale behind this exact set.
DEFAULT_NORMALIZE_FIELDS: tuple[str, ...] = (
    "surf_p",
    "surf_tau",
    "cl_true",
    "cd_true",
    "cond",
    "vol_u",
    "vol_p",
    "vol_nut",
)


class NormStats:
    """Per-field standardisation loaded from ``norm_stats.json``.

    Parameters
    ----------
    stats : Mapping or str or Path or None
        Either the parsed ``norm_stats.json`` payload, a path to it, or
        ``None`` for an identity transform.
    fields : iterable of str, optional
        Which fields to actually transform.  Defaults to
        :data:`DEFAULT_NORMALIZE_FIELDS`.  Fields outside this set (or absent
        from ``stats``) pass through untouched.

    Notes
    -----
    ``normalize`` and ``denormalize`` are exact inverses, and both broadcast a
    ``(C,)`` mean/std against a trailing dimension of size ``C`` (or against a
    scalar field of shape ``(...)`` when ``C == 1``).
    """

    def __init__(
        self,
        stats: Mapping[str, Any] | str | Path | None = None,
        fields: Iterable[str] | None = None,
    ) -> None:
        if stats is None:
            payload: dict[str, Any] = {"fields": {}}
        elif isinstance(stats, (str, Path)):
            with Path(stats).open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        else:
            payload = dict(stats)

        raw_fields = payload.get("fields", {})
        self.meta = {k: v for k, v in payload.items() if k != "fields"}
        self.fields: set[str] = set(
            DEFAULT_NORMALIZE_FIELDS if fields is None else fields
        )
        self._mean: dict[str, np.ndarray] = {}
        self._std: dict[str, np.ndarray] = {}
        for name, entry in raw_fields.items():
            mean = np.asarray(entry["mean"], dtype=np.float32).reshape(-1)
            std = np.asarray(entry["std"], dtype=np.float32).reshape(-1)
            std = np.where(np.abs(std) < 1e-8, np.float32(1.0), std)
            self._mean[name] = mean
            self._std[name] = std

    # -- introspection ------------------------------------------------------
    def has(self, field: str) -> bool:
        """True when ``field`` is both known and enabled for transformation."""
        return field in self.fields and field in self._mean

    def mean_std(self, field: str) -> tuple[np.ndarray, np.ndarray]:
        if field not in self._mean:
            raise KeyError(f"no normalisation stats for field {field!r}")
        return self._mean[field], self._std[field]

    # -- transforms ---------------------------------------------------------
    def _params(self, field: str, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.as_tensor(self._mean[field], dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self._std[field], dtype=x.dtype, device=x.device)
        if mean.numel() == 1:
            return mean.reshape(()), std.reshape(())
        if x.shape[-1] != mean.numel():
            raise ValueError(
                f"field {field!r}: tensor trailing dim {x.shape[-1]} does not "
                f"match stats width {mean.numel()}"
            )
        return mean, std

    def normalize(self, field: str, x: torch.Tensor) -> torch.Tensor:
        """``(x - mean) / std`` if ``field`` is enabled, else ``x`` unchanged."""
        if not self.has(field):
            return x
        mean, std = self._params(field, x)
        return (x - mean) / std

    def denormalize(self, field: str, x: torch.Tensor) -> torch.Tensor:
        """``x * std + mean`` if ``field`` is enabled, else ``x`` unchanged."""
        if not self.has(field):
            return x
        mean, std = self._params(field, x)
        return x * std + mean

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"NormStats(known={sorted(self._mean)}, "
            f"active={sorted(self.fields & set(self._mean))})"
        )


class AirfransSurfaceDataset(Dataset):
    """Surface-primary AirfRANS dataset backed by the ``.npz`` cache.

    Parameters
    ----------
    split_file : str or Path
        A manifest written by ``src/data/splits.py``.
    processed_dir : str or Path
        Directory holding ``<sim_name>.npz`` (``data/processed/airfrans``).
    normalize_stats : NormStats or Mapping or str or Path or None
        Normalisation statistics; ``None`` disables normalisation.  A path is
        loaded as JSON.  Defaults to ``processed_dir/norm_stats.json`` when
        that file exists and the argument is omitted entirely.
    subset : {"train", "cal", "test"}
        Which list inside the manifest to serve.
    load_volume : bool
        Also return the ``vol_*`` arrays (CONTEXT.md section 7 volume variants).
    normalize_fields : iterable of str, optional
        Override :data:`DEFAULT_NORMALIZE_FIELDS`.
    cache_in_ram : bool
        Keep decoded items in a dict.  The whole surface set is a few hundred
        MB (CONTEXT.md section 1: "fits in RAM"); volume arrays are not.
    missing : {"error", "skip"}
        What to do about split entries with no ``.npz`` on disk.

    Item schema
    -----------
    ``surf_pos (Ns,2) surf_normal (Ns,2) surf_ds (Ns,) surf_p (Ns,)
    surf_tau (Ns,2) cond (2,) cl_true () cd_true () sim_name str``
    plus ``re aoa_deg u_inf_mag rho nu perimeter n_surf`` and physical copies
    ``surf_p_phys surf_tau_phys cond_phys cl_true_phys cd_true_phys``.
    """

    def __init__(
        self,
        split_file: str | Path,
        processed_dir: str | Path,
        normalize_stats: Any = "auto",
        subset: str = "train",
        load_volume: bool = False,
        normalize_fields: Iterable[str] | None = None,
        cache_in_ram: bool = False,
        missing: str = "error",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if subset not in ("train", "cal", "test"):
            raise ValueError(
                f"subset must be one of train/cal/test, got {subset!r}"
            )
        if missing not in ("error", "skip"):
            raise ValueError(f"missing must be 'error' or 'skip', got {missing!r}")

        self.split_file = Path(split_file)
        self.processed_dir = Path(processed_dir)
        self.subset = subset
        self.load_volume = bool(load_volume)
        self.dtype = dtype
        self._ram: dict[int, dict[str, Any]] | None = {} if cache_in_ram else None

        payload = load_split(self.split_file)
        self.split_name: str = payload["name"]
        names: list[str] = list(payload[subset])

        present, absent = [], []
        for n in names:
            (present if (self.processed_dir / f"{n}.npz").is_file() else absent).append(n)
        if absent and missing == "error":
            raise FileNotFoundError(
                f"{len(absent)} of {len(names)} sims in "
                f"{self.split_file.name}[{subset}] have no .npz under "
                f"{self.processed_dir} (e.g. {absent[:3]}). Run "
                f"scripts/build_cache.py, or pass missing='skip'."
            )
        self.sim_names: list[str] = present
        self.missing_sims: list[str] = absent

        if normalize_stats == "auto":
            default_path = self.processed_dir / "norm_stats.json"
            normalize_stats = default_path if default_path.is_file() else None
        if isinstance(normalize_stats, NormStats):
            self.stats = normalize_stats
        else:
            self.stats = NormStats(normalize_stats, fields=normalize_fields)

    # -- plumbing -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sim_names)

    def path_for(self, sim_name: str) -> Path:
        return self.processed_dir / f"{sim_name}.npz"

    def _t(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(arr), dtype=self.dtype)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._ram is not None and index in self._ram:
            return self._ram[index]

        name = self.sim_names[index]
        with np.load(self.path_for(name)) as d:
            item: dict[str, Any] = {"sim_name": name}

            for key in SURFACE_KEYS:
                item[key] = self._t(d[key])
            if self.load_volume:
                for key in VOLUME_KEYS:
                    item[key] = self._t(d[key])
                item["vol_is_surf"] = torch.as_tensor(
                    np.ascontiguousarray(d["vol_is_surf"]), dtype=torch.bool
                )

            for key in _SIM_VECTOR_KEYS:
                item[key] = self._t(np.atleast_1d(d[key]))
            for key in _SIM_SCALAR_KEYS:
                item[key] = torch.as_tensor(float(d[key]), dtype=self.dtype)

            item["n_surf"] = torch.as_tensor(int(d["n_surf"]), dtype=torch.long)
            if self.load_volume:
                item["n_vol"] = torch.as_tensor(int(d["n_vol"]), dtype=torch.long)

        # Physical copies BEFORE normalisation, so eval never has to invert.
        for key in ("surf_p", "surf_tau", "cond", "cl_true", "cd_true"):
            item[f"{key}_phys"] = item[key].clone()

        for key in list(item):
            if key.endswith("_phys") or key == "sim_name":
                continue
            if isinstance(item[key], torch.Tensor) and self.stats.has(key):
                item[key] = self.stats.normalize(key, item[key])

        self._validate(item, name)
        if self._ram is not None:
            self._ram[index] = item
        return item

    def _validate(self, item: dict[str, Any], name: str) -> None:
        ns = int(item["n_surf"])
        checks = {
            "surf_pos": (ns, 2),
            "surf_normal": (ns, 2),
            "surf_ds": (ns,),
            "surf_p": (ns,),
            "surf_tau": (ns, 2),
            "cond": (2,),
        }
        for key, shape in checks.items():
            got = tuple(item[key].shape)
            if got != shape:
                raise ValueError(
                    f"{name}: cached {key} has shape {got}, expected {shape}"
                )
        if self.load_volume:
            nv = int(item["n_vol"])
            for key, shape in {
                "vol_pos": (nv, 2),
                "vol_u": (nv, 2),
                "vol_p": (nv,),
                "vol_nut": (nv,),
                "vol_sdf": (nv,),
            }.items():
                got = tuple(item[key].shape)
                if got != shape:
                    raise ValueError(
                        f"{name}: cached {key} has shape {got}, expected {shape}"
                    )

    # -- normalisation passthrough -----------------------------------------
    def normalize(self, field: str, x: torch.Tensor) -> torch.Tensor:
        """Apply the dataset's normalisation for ``field``."""
        return self.stats.normalize(field, x)

    def denormalize(self, field: str, x: torch.Tensor) -> torch.Tensor:
        """Invert the dataset's normalisation for ``field``.

        CONTEXT.md section 4: models emit normalised values; every reported
        metric and every force integral must go through this first.
        """
        return self.stats.denormalize(field, x)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"AirfransSurfaceDataset(split={self.split_name!r}, "
            f"subset={self.subset!r}, n={len(self)}, "
            f"volume={self.load_volume})"
        )


def collate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Concatenate per-simulation items into a PyG-style batch.

    Parameters
    ----------
    items : sequence of item dicts
        As produced by :meth:`AirfransSurfaceDataset.__getitem__`.

    Returns
    -------
    dict
        Point-wise keys concatenated along dim 0; per-sim keys stacked to a
        leading dimension of size ``B``; plus ``batch_idx`` ``(sum Ns,)``,
        ``ptr`` ``(B+1,)``, ``n_surf`` ``(B,)``, ``num_graphs`` and
        ``sim_name`` (a ``list[str]``).  When the items carry volume arrays,
        ``vol_batch_idx`` and ``vol_ptr`` are added too.

    Raises
    ------
    ValueError
        On an empty batch or heterogeneous keys.
    """
    if len(items) == 0:
        raise ValueError("cannot collate an empty list of items")

    keys = set(items[0])
    for i, it in enumerate(items[1:], 1):
        if set(it) != keys:
            raise ValueError(
                f"item {i} has keys {sorted(set(it) ^ keys)} not shared with "
                "item 0; datasets must be configured identically"
            )

    batch: dict[str, Any] = {}
    point_keys = [k for k in keys if k.startswith("surf_")]
    vol_keys = [
        k for k in keys if k.startswith("vol_") and k not in ("n_vol",)
    ]

    for key in sorted(point_keys):
        batch[key] = torch.cat([it[key] for it in items], dim=0)

    counts = torch.as_tensor(
        [int(it["n_surf"]) for it in items], dtype=torch.long
    )
    batch["n_surf"] = counts
    batch["batch_idx"] = torch.repeat_interleave(
        torch.arange(len(items), dtype=torch.long), counts
    )
    batch["ptr"] = torch.cat(
        [torch.zeros(1, dtype=torch.long), torch.cumsum(counts, dim=0)]
    )
    if batch["batch_idx"].numel() != batch["surf_pos"].shape[0]:
        raise ValueError(
            "n_surf disagrees with the number of concatenated surface points"
        )

    if vol_keys:
        for key in sorted(vol_keys):
            batch[key] = torch.cat([it[key] for it in items], dim=0)
        vcounts = torch.as_tensor(
            [int(it["n_vol"]) for it in items], dtype=torch.long
        )
        batch["n_vol"] = vcounts
        batch["vol_batch_idx"] = torch.repeat_interleave(
            torch.arange(len(items), dtype=torch.long), vcounts
        )
        batch["vol_ptr"] = torch.cat(
            [torch.zeros(1, dtype=torch.long), torch.cumsum(vcounts, dim=0)]
        )

    for key in sorted(keys):
        if key in batch or key == "sim_name":
            continue
        val = items[0][key]
        if isinstance(val, torch.Tensor):
            batch[key] = torch.stack([it[key] for it in items], dim=0)

    batch["sim_name"] = [it["sim_name"] for it in items]
    batch["num_graphs"] = len(items)
    return batch


def make_dataloader(
    dataset: AirfransSurfaceDataset,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int | None = None,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    """``DataLoader`` wired to :func:`collate`.

    ``num_workers`` is hard-clamped to 2 (CONTEXT.md sections 1 and 12: the
    machine must stay responsive; no multiprocessing pools).
    """
    workers = max(0, min(int(num_workers), 2))
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate,
        drop_last=drop_last,
        pin_memory=pin_memory,
        generator=generator,
        persistent_workers=workers > 0,
    )
