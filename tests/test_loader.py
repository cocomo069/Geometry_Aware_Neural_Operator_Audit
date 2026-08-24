"""Loader / collate tests (CONTEXT.md sections 4 and 11).

Everything runs on CPU against **synthetic ``.npz`` files created in
``tmp_path``** that mirror the frozen cache schema, so the suite is green long
before the 10 GB download lands and never touches a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.airfrans_loader import (
    DEFAULT_NORMALIZE_FIELDS,
    SURFACE_KEYS,
    VOLUME_KEYS,
    AirfransSurfaceDataset,
    NormStats,
    collate,
    make_dataloader,
)

CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# synthetic cache
# ---------------------------------------------------------------------------
def _fake_sim_arrays(
    name: str, n_surf: int, n_vol: int, seed: int
) -> dict[str, np.ndarray]:
    """Build one cache entry with the exact CONTEXT.md section 4 schema.

    Geometry is a circle so the invariants the real builder asserts
    (``sum(surf_ds) == perimeter``, ``sum(n * ds) == 0``) hold here too.
    """
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n_surf, endpoint=False)
    radius = 0.5
    surf_pos = np.stack(
        [0.5 + radius * np.cos(theta), radius * np.sin(theta)], axis=1
    )
    surf_normal = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    surf_ds = np.full(n_surf, 2.0 * np.pi * radius / n_surf)

    u_inf = 30.0 + 60.0 * rng.random()
    aoa_deg = -5.0 + 20.0 * rng.random()
    aoa = np.deg2rad(aoa_deg)

    return {
        "surf_pos": surf_pos.astype(np.float32),
        "surf_normal": surf_normal.astype(np.float32),
        "surf_ds": surf_ds.astype(np.float32),
        "surf_p": rng.normal(0.0, 400.0, n_surf).astype(np.float32),
        "surf_tau": rng.normal(0.0, 0.5, (n_surf, 2)).astype(np.float32),
        "vol_pos": rng.uniform(-2.0, 4.0, (n_vol, 2)).astype(np.float32),
        "vol_u": rng.normal(u_inf, 5.0, (n_vol, 2)).astype(np.float32),
        "vol_p": rng.normal(0.0, 400.0, n_vol).astype(np.float32),
        "vol_nut": rng.uniform(0.0, 1e-3, n_vol).astype(np.float32),
        "vol_sdf": rng.uniform(0.0, 3.0, n_vol).astype(np.float32),
        "vol_is_surf": (rng.random(n_vol) < 0.05).astype(np.uint8),
        "cond": np.array(
            [u_inf * np.cos(aoa), u_inf * np.sin(aoa)], dtype=np.float32
        ),
        "re": np.float32(u_inf / 1.56e-5),
        "aoa_deg": np.float32(aoa_deg),
        "u_inf_mag": np.float32(u_inf),
        "cl_true": np.float32(rng.normal(0.5, 0.3)),
        "cd_true": np.float32(abs(rng.normal(0.02, 0.005))),
        "cl_p": np.float32(0.0),
        "cl_v": np.float32(0.0),
        "cd_p": np.float32(0.0),
        "cd_v": np.float32(0.0),
        "rho": np.float32(1.184),
        "nu": np.float32(1.54981e-5),
        "perimeter": np.float32(surf_ds.sum()),
        "n_surf": np.int64(n_surf),
        "n_vol": np.int64(n_vol),
        "n_vol_full": np.int64(n_vol * 4),
        "cache_version": np.int64(CACHE_VERSION),
    }


#: (name, n_surf, n_vol) -- deliberately ragged so batching cannot be padded.
_SPEC = [
    ("airFoil2D_SST_40.0_1.0_2.0_4.0_12.0", 17, 40),
    ("airFoil2D_SST_55.0_3.0_2.0_3.0_1.0_15.0", 23, 31),
    ("airFoil2D_SST_70.0_-2.0_1.0_4.0_9.0", 11, 55),
    ("airFoil2D_SST_85.0_9.0_3.0_3.0_0.0_18.0", 29, 22),
]


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny processed cache + split manifest + norm stats."""
    root = tmp_path_factory.mktemp("processed")
    for seed, (name, ns, nv) in enumerate(_SPEC):
        np.savez_compressed(
            root / f"{name}.npz", **_fake_sim_arrays(name, ns, nv, seed)
        )

    names = [s[0] for s in _SPEC]
    (root / "split.json").write_text(
        json.dumps(
            {
                "name": "synthetic",
                "train": names[:2],
                "cal": names[2:3],
                "test": names[3:],
                "seed": 0,
                "description": "synthetic fixture",
            }
        ),
        encoding="utf-8",
    )

    stats = {
        "split": "synthetic",
        "subset": "train",
        "n_sims": 2,
        "fields": {
            "surf_p": {"mean": [10.0], "std": [400.0], "count": 40},
            "surf_tau": {"mean": [0.1, -0.2], "std": [0.5, 0.5], "count": 40},
            "cond": {"mean": [60.0, 2.0], "std": [15.0, 3.0], "count": 4},
            "cl_true": {"mean": [0.5], "std": [0.3], "count": 4},
            "cd_true": {"mean": [0.02], "std": [0.005], "count": 4},
            "vol_u": {"mean": [60.0, 0.0], "std": [10.0, 5.0], "count": 100},
            "vol_p": {"mean": [0.0], "std": [400.0], "count": 100},
            "vol_nut": {"mean": [5e-4], "std": [3e-4], "count": 100},
            # Present in the file but NOT in DEFAULT_NORMALIZE_FIELDS: the
            # loader must leave these physical.
            "surf_pos": {"mean": [0.5, 0.0], "std": [0.35, 0.35], "count": 40},
            "surf_ds": {"mean": [0.1], "std": [0.02], "count": 40},
            "surf_normal": {"mean": [0.0, 0.0], "std": [0.7, 0.7], "count": 40},
            "vol_sdf": {"mean": [1.5], "std": [0.9], "count": 100},
        },
    }
    (root / "norm_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def split_file(cache: Path) -> Path:
    return cache / "split.json"


def _ds(cache: Path, split_file: Path, **kw) -> AirfransSurfaceDataset:
    kw.setdefault("normalize_stats", None)
    kw.setdefault("subset", "train")
    return AirfransSurfaceDataset(split_file, cache, **kw)


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
def test_subsets_have_the_manifest_lengths(cache: Path, split_file: Path) -> None:
    assert len(_ds(cache, split_file, subset="train")) == 2
    assert len(_ds(cache, split_file, subset="cal")) == 1
    assert len(_ds(cache, split_file, subset="test")) == 1


def test_rejects_an_unknown_subset(cache: Path, split_file: Path) -> None:
    with pytest.raises(ValueError, match="subset"):
        _ds(cache, split_file, subset="valid")


def test_item_shapes_and_dtypes(cache: Path, split_file: Path) -> None:
    ds = _ds(cache, split_file)
    item = ds[0]
    ns = int(item["n_surf"])
    assert ns == _SPEC[0][1]
    assert item["surf_pos"].shape == (ns, 2)
    assert item["surf_normal"].shape == (ns, 2)
    assert item["surf_ds"].shape == (ns,)
    assert item["surf_p"].shape == (ns,)
    assert item["surf_tau"].shape == (ns, 2)
    assert item["cond"].shape == (2,)
    assert item["cl_true"].shape == ()
    assert item["cd_true"].shape == ()
    assert item["sim_name"] == _SPEC[0][0]
    for key in (*SURFACE_KEYS, "cond", "cl_true", "cd_true"):
        assert item[key].dtype == torch.float32, key


def test_volume_arrays_only_when_requested(cache: Path, split_file: Path) -> None:
    plain = _ds(cache, split_file)[0]
    assert not any(k.startswith("vol_") for k in plain)

    with_vol = _ds(cache, split_file, load_volume=True)[0]
    nv = int(with_vol["n_vol"])
    assert nv == _SPEC[0][2]
    assert with_vol["vol_pos"].shape == (nv, 2)
    assert with_vol["vol_u"].shape == (nv, 2)
    for key in ("vol_p", "vol_nut", "vol_sdf"):
        assert with_vol[key].shape == (nv,)
    assert with_vol["vol_is_surf"].dtype == torch.bool


def test_missing_npz_is_an_error_by_default(
    cache: Path, tmp_path: Path
) -> None:
    manifest = tmp_path / "s.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "s",
                "train": [_SPEC[0][0], "airFoil2D_SST_1.0_1.0_0.0_0.0_1.0"],
                "cal": [],
                "test": [],
                "seed": 0,
                "description": "one sim is absent",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="no .npz"):
        AirfransSurfaceDataset(manifest, cache, normalize_stats=None)

    ds = AirfransSurfaceDataset(
        manifest, cache, normalize_stats=None, missing="skip"
    )
    assert len(ds) == 1
    assert len(ds.missing_sims) == 1


def test_ram_cache_returns_equal_items(cache: Path, split_file: Path) -> None:
    ds = _ds(cache, split_file, cache_in_ram=True)
    first, second = ds[0], ds[0]
    assert torch.equal(first["surf_p"], second["surf_p"])


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
def test_normalisation_is_applied_to_the_default_fields(
    cache: Path, split_file: Path
) -> None:
    raw = _ds(cache, split_file)[0]
    norm = _ds(cache, split_file, normalize_stats="auto")[0]

    expect = (raw["surf_p"] - 10.0) / 400.0
    assert torch.allclose(norm["surf_p"], expect, atol=1e-5)

    tau_mean = torch.tensor([0.1, -0.2])
    tau_std = torch.tensor([0.5, 0.5])
    assert torch.allclose(
        norm["surf_tau"], (raw["surf_tau"] - tau_mean) / tau_std, atol=1e-5
    )
    assert torch.allclose(
        norm["cl_true"], (raw["cl_true"] - 0.5) / 0.3, atol=1e-5
    )


def test_geometry_is_never_normalised(cache: Path, split_file: Path) -> None:
    """``surf_ds`` / ``surf_normal`` / positions feed the force integral raw.

    Stats for them exist in the fixture's ``norm_stats.json`` precisely so that
    a regression which widened the default field set would fail here.
    """
    raw = _ds(cache, split_file)[0]
    norm = _ds(cache, split_file, normalize_stats="auto")[0]
    for key in ("surf_pos", "surf_ds", "surf_normal"):
        assert key not in DEFAULT_NORMALIZE_FIELDS
        assert torch.equal(norm[key], raw[key]), key


def test_physical_copies_survive_normalisation(
    cache: Path, split_file: Path
) -> None:
    raw = _ds(cache, split_file)[0]
    norm = _ds(cache, split_file, normalize_stats="auto")[0]
    for key in ("surf_p", "surf_tau", "cond", "cl_true", "cd_true"):
        assert torch.allclose(norm[f"{key}_phys"], raw[key], atol=1e-6), key
        assert not torch.allclose(norm[key], raw[key], atol=1e-6), key


def test_denormalize_inverts_normalize(cache: Path, split_file: Path) -> None:
    ds = _ds(cache, split_file, normalize_stats="auto")
    item = ds[0]
    for key in ("surf_p", "surf_tau", "cl_true", "cd_true", "cond"):
        back = ds.denormalize(key, item[key])
        assert torch.allclose(back, item[f"{key}_phys"], atol=1e-4), key


def test_denormalize_passes_untracked_fields_through(
    cache: Path, split_file: Path
) -> None:
    ds = _ds(cache, split_file, normalize_stats="auto")
    x = torch.randn(5, 2)
    assert torch.equal(ds.denormalize("surf_normal", x), x)
    assert torch.equal(ds.denormalize("not_a_field", x), x)


def test_normstats_identity_when_none() -> None:
    stats = NormStats(None)
    x = torch.randn(4)
    assert torch.equal(stats.normalize("surf_p", x), x)
    assert torch.equal(stats.denormalize("surf_p", x), x)
    assert not stats.has("surf_p")


def test_normstats_guards_a_width_mismatch(cache: Path) -> None:
    stats = NormStats(cache / "norm_stats.json")
    with pytest.raises(ValueError, match="trailing dim"):
        stats.normalize("surf_tau", torch.randn(4, 3))


def test_normalize_fields_override(cache: Path, split_file: Path) -> None:
    ds = _ds(
        cache,
        split_file,
        normalize_stats=str(cache / "norm_stats.json"),
        normalize_fields=("surf_p",),
    )
    raw = _ds(cache, split_file)[0]
    item = ds[0]
    assert not torch.allclose(item["surf_p"], raw["surf_p"], atol=1e-6)
    assert torch.equal(item["surf_tau"], raw["surf_tau"])


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------
def test_collate_concatenates_without_padding(
    cache: Path, split_file: Path
) -> None:
    ds = _ds(cache, split_file, subset="train")
    items = [ds[i] for i in range(len(ds))]
    batch = collate(items)

    total = sum(int(it["n_surf"]) for it in items)
    assert batch["surf_pos"].shape == (total, 2)
    assert batch["surf_normal"].shape == (total, 2)
    assert batch["surf_ds"].shape == (total,)
    assert batch["surf_p"].shape == (total,)
    assert batch["surf_tau"].shape == (total, 2)
    assert batch["num_graphs"] == len(items)
    assert batch["sim_name"] == [it["sim_name"] for it in items]


def test_batch_idx_is_correct(cache: Path, split_file: Path) -> None:
    """The load-bearing invariant: batch_idx must select each sim's own rows."""
    ds = _ds(cache, split_file, subset="train")
    items = [ds[i] for i in range(len(ds))]
    batch = collate(items)

    bidx = batch["batch_idx"]
    assert bidx.dtype == torch.long
    assert bidx.shape == (batch["surf_pos"].shape[0],)
    assert int(bidx.min()) == 0
    assert int(bidx.max()) == len(items) - 1
    assert torch.all(bidx[1:] >= bidx[:-1]), "batch_idx must be non-decreasing"

    for i, item in enumerate(items):
        mask = bidx == i
        assert int(mask.sum()) == int(item["n_surf"])
        for key in SURFACE_KEYS:
            assert torch.equal(batch[key][mask], item[key]), key


def test_ptr_matches_batch_idx(cache: Path, split_file: Path) -> None:
    ds = _ds(cache, split_file, subset="train")
    items = [ds[i] for i in range(len(ds))]
    batch = collate(items)
    ptr, counts = batch["ptr"], batch["n_surf"]
    assert ptr.shape == (len(items) + 1,)
    assert int(ptr[0]) == 0
    assert int(ptr[-1]) == batch["surf_pos"].shape[0]
    for i in range(len(items)):
        lo, hi = int(ptr[i]), int(ptr[i + 1])
        assert hi - lo == int(counts[i])
        assert torch.all(batch["batch_idx"][lo:hi] == i)


def test_per_sim_quantities_are_stacked(cache: Path, split_file: Path) -> None:
    ds = _ds(cache, split_file, subset="train")
    items = [ds[i] for i in range(len(ds))]
    batch = collate(items)
    b = len(items)
    assert batch["cond"].shape == (b, 2)
    assert batch["cl_true"].shape == (b,)
    assert batch["cd_true"].shape == (b,)
    assert batch["n_surf"].shape == (b,)
    for i, item in enumerate(items):
        assert torch.equal(batch["cond"][i], item["cond"])
        assert torch.equal(batch["cl_true"][i], item["cl_true"])


def test_collate_handles_volume_arrays(cache: Path, split_file: Path) -> None:
    ds = _ds(cache, split_file, subset="train", load_volume=True)
    items = [ds[i] for i in range(len(ds))]
    batch = collate(items)

    total = sum(int(it["n_vol"]) for it in items)
    for key in VOLUME_KEYS:
        assert batch[key].shape[0] == total, key
    vidx = batch["vol_batch_idx"]
    assert vidx.dtype == torch.long
    assert vidx.shape == (total,)
    for i, item in enumerate(items):
        mask = vidx == i
        assert int(mask.sum()) == int(item["n_vol"])
        assert torch.equal(batch["vol_pos"][mask], item["vol_pos"])
    # Surface and volume indices are independent.
    assert batch["batch_idx"].shape[0] != batch["vol_batch_idx"].shape[0]


def test_collate_of_a_single_item(cache: Path, split_file: Path) -> None:
    ds = _ds(cache, split_file, subset="cal")
    batch = collate([ds[0]])
    assert batch["num_graphs"] == 1
    assert torch.all(batch["batch_idx"] == 0)
    assert batch["cond"].shape == (1, 2)


def test_collate_rejects_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty"):
        collate([])


def test_collate_rejects_heterogeneous_items(
    cache: Path, split_file: Path
) -> None:
    plain = _ds(cache, split_file)[0]
    with_vol = _ds(cache, split_file, load_volume=True)[0]
    with pytest.raises(ValueError, match="keys"):
        collate([plain, with_vol])


# ---------------------------------------------------------------------------
# dataloader
# ---------------------------------------------------------------------------
def test_dataloader_yields_well_formed_batches(
    cache: Path, split_file: Path
) -> None:
    ds = _ds(cache, split_file, subset="train")
    loader = make_dataloader(ds, batch_size=2, shuffle=False, num_workers=0)
    batches = list(loader)
    assert len(batches) == 1
    batch = batches[0]
    assert batch["num_graphs"] == 2
    assert batch["batch_idx"].shape[0] == batch["surf_pos"].shape[0]
    assert set(batch["sim_name"]) == {s[0] for s in _SPEC[:2]}


def test_dataloader_clamps_num_workers(cache: Path, split_file: Path) -> None:
    """CONTEXT.md sections 1 and 12: never more than two workers."""
    ds = _ds(cache, split_file, subset="train")
    assert make_dataloader(ds, num_workers=8).num_workers == 2
    assert make_dataloader(ds, num_workers=-3).num_workers == 0


def test_dataloader_shuffle_is_seeded(cache: Path, split_file: Path) -> None:
    ds = _ds(cache, split_file, subset="train")

    def order(seed: int) -> list[str]:
        loader = make_dataloader(
            ds, batch_size=1, shuffle=True, num_workers=0, seed=seed
        )
        return [b["sim_name"][0] for b in loader]

    assert order(0) == order(0)


def test_everything_stays_on_cpu(cache: Path, split_file: Path) -> None:
    ds = _ds(cache, split_file, subset="train")
    batch = collate([ds[i] for i in range(len(ds))])
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            assert val.device.type == "cpu", key
