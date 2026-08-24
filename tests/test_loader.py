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

from scripts.build_cache import GRID_BBOX, GRID_RES, KNN_K, geometry_extras
from src.data.airfrans_loader import (
    DEFAULT_NORMALIZE_FIELDS,
    GEOMETRY_KEYS,
    SURFACE_KEYS,
    VOLUME_KEYS,
    AirfransSurfaceDataset,
    NormStats,
    collate,
    make_dataloader,
)
from src.models.common import get_edge_index, knn_edges
from src.models.sdf_fno import compute_grid_sdf, make_latent_grid

CACHE_VERSION = 2


# ---------------------------------------------------------------------------
# synthetic cache
# ---------------------------------------------------------------------------
def _fake_sim_arrays(
    name: str, n_surf: int, n_vol: int, seed: int, geometry: bool = True
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

    arrays = {
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
    if geometry:
        # Built by the real builder, from the float32 array that is stored --
        # exactly what scripts/build_cache.py does.
        arrays.update(geometry_extras(arrays["surf_pos"], k=KNN_K))
    return arrays


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
# D-017 precomputed static geometry
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def plain_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A pre-D-017 cache: no grid_sdf / edge_index / curvature."""
    root = tmp_path_factory.mktemp("processed_nogeom")
    names = [s[0] for s in _SPEC[:2]]
    for seed, (name, ns, nv) in enumerate(_SPEC[:2]):
        np.savez_compressed(
            root / f"{name}.npz",
            **_fake_sim_arrays(name, ns, nv, seed, geometry=False),
        )
    (root / "split.json").write_text(
        json.dumps(
            {
                "name": "nogeom",
                "train": names,
                "cal": [],
                "test": [],
                "seed": 0,
                "description": "pre-D-017 cache",
            }
        ),
        encoding="utf-8",
    )
    return root


def test_dataset_detects_the_geometry_keys(cache: Path, split_file: Path) -> None:
    ds = _ds(cache, split_file)
    assert set(ds.geometry_keys) == set(GEOMETRY_KEYS)


def test_geometry_item_shapes_and_dtypes(cache: Path, split_file: Path) -> None:
    item = _ds(cache, split_file)[0]
    ns = int(item["n_surf"])
    assert item["curvature"].shape == (ns,)
    assert item["curvature"].dtype == torch.float32
    assert item["grid_sdf"].shape == tuple(GRID_RES)
    assert item["grid_sdf"].dtype == torch.float32
    edge = item["edge_index"]
    assert edge.shape == (2, ns * min(KNN_K, ns - 1))
    # long, because that is what the models index with
    assert edge.dtype == torch.long
    assert int(edge.max()) < ns


def test_cached_curvature_matches_the_circle(cache: Path, split_file: Path) -> None:
    """The fixture geometry is a radius-0.5 circle, so kappa == +2 everywhere."""
    item = _ds(cache, split_file)[0]
    assert torch.allclose(
        item["curvature"], torch.full_like(item["curvature"], 2.0), atol=1e-3
    )


def test_collate_concatenates_curvature(cache: Path, split_file: Path) -> None:
    ds = _ds(cache, split_file, subset="train")
    items = [ds[i] for i in range(len(ds))]
    batch = collate(items)
    total = sum(int(it["n_surf"]) for it in items)
    assert batch["curvature"].shape == (total,)
    for i, item in enumerate(items):
        mask = batch["batch_idx"] == i
        assert torch.equal(batch["curvature"][mask], item["curvature"])


def test_collate_stacks_grid_sdf(cache: Path, split_file: Path) -> None:
    ds = _ds(cache, split_file, subset="train")
    items = [ds[i] for i in range(len(ds))]
    batch = collate(items)
    assert batch["grid_sdf"].shape == (len(items), *GRID_RES)
    for i, item in enumerate(items):
        assert torch.equal(batch["grid_sdf"][i], item["grid_sdf"])


def test_batched_edge_index_equals_on_the_fly_knn(
    cache: Path, split_file: Path
) -> None:
    """The load-bearing D-017 invariant.

    ``common.get_edge_index`` uses ``batch['edge_index']`` *verbatim*, so the
    per-sim cached edges must be offset here into the concatenated point array.
    Anything else silently wires messages between the wrong nodes.
    """
    ds = _ds(cache, split_file, subset="train")
    items = [ds[i] for i in range(len(ds))]
    batch = collate(items)

    reference = knn_edges(
        batch["surf_pos"],
        k=KNN_K,
        batch_idx=batch["batch_idx"],
        num_graphs=len(items),
    )
    assert torch.equal(batch["edge_index"], reference)


def test_edge_index_never_crosses_samples(cache: Path, split_file: Path) -> None:
    ds = _ds(cache, split_file, subset="train")
    items = [ds[i] for i in range(len(ds))]
    batch = collate(items)
    src, dst = batch["edge_index"]
    bidx = batch["batch_idx"]
    assert torch.equal(bidx[src], bidx[dst]), "an edge crosses two simulations"
    assert int(batch["edge_index"].max()) < batch["surf_pos"].shape[0]


def test_edge_index_offsets_are_per_sample(cache: Path, split_file: Path) -> None:
    """Sample i's edges must land in [ptr[i], ptr[i+1])."""
    ds = _ds(cache, split_file, subset="train")
    items = [ds[i] for i in range(len(ds))]
    batch = collate(items)
    ptr = batch["ptr"]
    consumed = 0
    for i, item in enumerate(items):
        e = item["edge_index"]
        got = batch["edge_index"][:, consumed : consumed + e.shape[1]]
        assert torch.equal(got, e + int(ptr[i]))
        consumed += e.shape[1]
    assert consumed == batch["edge_index"].shape[1]


def test_get_edge_index_uses_the_cache_verbatim(
    cache: Path, split_file: Path
) -> None:
    ds = _ds(cache, split_file, subset="train")
    batch = collate([ds[i] for i in range(len(ds))])
    assert torch.equal(
        get_edge_index(batch, k=KNN_K, num_graphs=batch["num_graphs"]),
        batch["edge_index"],
    )


def test_cached_grid_sdf_is_bit_identical_to_m2(
    cache: Path, split_file: Path
) -> None:
    """Cached grid_sdf must equal what M2 computes on the fly.

    ``src/geometry/sdf.make_grid`` puts **x** on the first axis while M2's
    ``make_latent_grid`` is row-major ``iy * nx + ix``; the two differ by a
    transpose.  Skipping it is a silent ~0.4-magnitude error, not a crash, so
    this asserts exact float32 equality rather than a tolerance.
    """
    ds = _ds(cache, split_file, subset="train")
    items = [ds[i] for i in range(len(ds))]
    batch = collate(items)
    grid = make_latent_grid(GRID_BBOX, GRID_RES)

    for i in range(len(items)):
        sel = (batch["batch_idx"] == i).numpy()
        reference = compute_grid_sdf(batch["surf_pos"].numpy()[sel], grid)
        cached = batch["grid_sdf"][i].numpy().reshape(-1)
        assert np.array_equal(cached, reference), f"sample {i}"


def test_grid_sdf_is_negative_inside_the_body(
    cache: Path, split_file: Path
) -> None:
    item = _ds(cache, split_file)[0]
    grid_sdf = item["grid_sdf"].numpy()
    n_y, n_x = GRID_RES
    xs = np.linspace(GRID_BBOX[0], GRID_BBOX[1], n_x)
    ys = np.linspace(GRID_BBOX[2], GRID_BBOX[3], n_y)
    # fixture body is the circle centred at (0.5, 0) with radius 0.5
    iy = int(np.argmin(np.abs(ys - 0.0)))
    ix = int(np.argmin(np.abs(xs - 0.5)))
    assert grid_sdf[iy, ix] < 0.0, "grid_sdf must be negative inside the body"
    assert grid_sdf[0, 0] > 0.0, "a far corner must be outside"


def test_geometry_can_be_switched_off(cache: Path, split_file: Path) -> None:
    ds = _ds(cache, split_file, load_geometry=False)
    assert ds.geometry_keys == ()
    item = ds[0]
    for key in GEOMETRY_KEYS:
        assert key not in item


def test_pre_d017_cache_still_loads(plain_cache: Path) -> None:
    """Absence of the optional keys is never an error (models fall back)."""
    ds = AirfransSurfaceDataset(
        plain_cache / "split.json", plain_cache, normalize_stats=None
    )
    assert ds.geometry_keys == ()
    batch = collate([ds[i] for i in range(len(ds))])
    for key in GEOMETRY_KEYS:
        assert key not in batch
    # and the model-side fallback still produces a usable graph
    edges = get_edge_index(batch, k=KNN_K, num_graphs=batch["num_graphs"])
    assert edges.shape[0] == 2


def test_geometry_survives_normalisation(cache: Path, split_file: Path) -> None:
    """Curvature/grid_sdf are geometry, not fields: never standardised."""
    raw = _ds(cache, split_file)[0]
    norm = _ds(cache, split_file, normalize_stats="auto")[0]
    for key in GEOMETRY_KEYS:
        assert key not in DEFAULT_NORMALIZE_FIELDS
        assert torch.equal(norm[key], raw[key]), key


def test_dataloader_carries_geometry(cache: Path, split_file: Path) -> None:
    ds = _ds(cache, split_file, subset="train")
    loader = make_dataloader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert batch["grid_sdf"].shape == (2, *GRID_RES)
    assert batch["edge_index"].shape[0] == 2
    assert batch["curvature"].shape[0] == batch["surf_pos"].shape[0]


# ---------------------------------------------------------------------------
# the models actually consume the cached geometry (D-017 hand-off)
# ---------------------------------------------------------------------------
def _real_batch(cache: Path, split_file: Path) -> dict:
    ds = _ds(cache, split_file, subset="train", normalize_stats="auto")
    return collate([ds[i] for i in range(len(ds))])


@pytest.mark.parametrize("name", ["gnn", "sdf_fno", "transolver"])
def test_models_forward_on_a_cached_batch(
    cache: Path, split_file: Path, name: str
) -> None:
    """End-to-end: a batch straight out of the loader drives every model."""
    from src.models import build_model

    cfgs = {
        "gnn": {"hidden_dim": 16, "mlp_hidden": 16, "n_rounds": 2,
                "k": KNN_K, "n_freqs": 2, "trunk_dim": 16,
                "coef_hidden": 16, "coef_layers": 1},
        "sdf_fno": {"width": 8, "modes": 4, "n_fno_layers": 2,
                    "spectral_groups": 2, "grid_res": GRID_RES, "k_enc": 4,
                    "k_dec": 4, "kernel_hidden": 16, "point_hidden": 16,
                    "n_freqs": 2, "trunk_dim": 16, "coef_hidden": 16,
                    "coef_layers": 1},
        "transolver": {"dim": 16, "n_layers": 2, "n_heads": 2, "n_slices": 4,
                       "ffn_mult": 2, "n_freqs": 2, "trunk_dim": 16,
                       "coef_hidden": 16, "coef_layers": 1},
    }
    torch.manual_seed(0)
    model = build_model(name, cfgs[name])
    batch = _real_batch(cache, split_file)

    out = model(batch)
    n_pts = batch["surf_pos"].shape[0]
    b = batch["num_graphs"]
    assert out["p"].shape == (n_pts,)
    assert out["tau"].shape == (n_pts, 2)
    assert out["coef_head"].shape == (b, 2)
    assert torch.isfinite(out["p"]).all()
    assert torch.isfinite(out["coef_head"]).all()


def test_gnn_uses_the_cached_edge_index_not_a_fresh_kdtree(
    cache: Path, split_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proof of the speedup: no kD-tree is built when the cache supplies edges."""
    import src.models.common as common_mod
    from src.models import build_model

    def boom(*a, **kw):  # pragma: no cover - must never run
        raise AssertionError("knn_edges was called despite a cached edge_index")

    monkeypatch.setattr(common_mod, "knn_edges", boom)

    torch.manual_seed(0)
    model = build_model(
        "gnn",
        {"hidden_dim": 16, "mlp_hidden": 16, "n_rounds": 2, "k": KNN_K,
         "n_freqs": 2, "trunk_dim": 16, "coef_hidden": 16, "coef_layers": 1},
    )
    out = model(_real_batch(cache, split_file))
    assert torch.isfinite(out["p"]).all()


def test_sdf_fno_uses_the_cached_grid_sdf(
    cache: Path, split_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proof of the speedup: no per-step SDF sweep when grid_sdf is cached."""
    import src.models.sdf_fno as fno_mod
    from src.models import build_model

    def boom(*a, **kw):  # pragma: no cover - must never run
        raise AssertionError("compute_grid_sdf ran despite a cached grid_sdf")

    monkeypatch.setattr(fno_mod, "compute_grid_sdf", boom)

    torch.manual_seed(0)
    model = build_model(
        "sdf_fno",
        {"width": 8, "modes": 4, "n_fno_layers": 2, "spectral_groups": 2,
         "grid_res": GRID_RES, "k_enc": 4, "k_dec": 4, "kernel_hidden": 16,
         "point_hidden": 16, "n_freqs": 2, "trunk_dim": 16,
         "coef_hidden": 16, "coef_layers": 1},
    )
    out = model(_real_batch(cache, split_file))
    assert torch.isfinite(out["p"]).all()


def test_cached_and_on_the_fly_geometry_give_identical_predictions(
    cache: Path, split_file: Path
) -> None:
    """``edge_index`` / ``grid_sdf`` are pure caches: identical predictions.

    ``curvature`` is deliberately excluded -- it is *not* an optimisation. Before
    D-017 ``SurrogateBase.get_curvature`` zero-filled the channel, so caching it
    feeds the models real information they previously did not have. Predictions
    with and without it differ by design; see ``test_curvature_changes_...``.
    """
    from src.models import build_model

    batch = _real_batch(cache, split_file)
    pure_caches = ("edge_index", "grid_sdf")
    stripped = {k: v for k, v in batch.items() if k not in pure_caches}

    for name, cfg in (
        ("gnn", {"hidden_dim": 16, "mlp_hidden": 16, "n_rounds": 2,
                 "k": KNN_K, "n_freqs": 2, "trunk_dim": 16,
                 "coef_hidden": 16, "coef_layers": 1}),
        ("sdf_fno", {"width": 8, "modes": 4, "n_fno_layers": 2,
                     "spectral_groups": 2, "grid_res": GRID_RES, "k_enc": 4,
                     "k_dec": 4, "kernel_hidden": 16, "point_hidden": 16,
                     "n_freqs": 2, "trunk_dim": 16, "coef_hidden": 16,
                     "coef_layers": 1}),
    ):
        torch.manual_seed(0)
        model = build_model(name, cfg).eval()
        with torch.no_grad():
            a = model(batch)
            b = model(stripped)
        assert torch.allclose(a["p"], b["p"], atol=1e-5), name
        assert torch.allclose(
            a["coef_head"], b["coef_head"], atol=1e-5
        ), name


def test_curvature_changes_predictions_versus_the_zero_fill(
    cache: Path, split_file: Path
) -> None:
    """Caching curvature is a model-input change, not just a speedup.

    Consequence for A3/A4: checkpoints trained against a pre-D-017 cache saw a
    constant-zero curvature channel and are not comparable to ones trained now.
    """
    from src.models import build_model

    batch = _real_batch(cache, split_file)
    without = {k: v for k, v in batch.items() if k != "curvature"}
    torch.manual_seed(0)
    model = build_model(
        "gnn",
        {"hidden_dim": 16, "mlp_hidden": 16, "n_rounds": 2, "k": KNN_K,
         "n_freqs": 2, "trunk_dim": 16, "coef_hidden": 16, "coef_layers": 1},
    ).eval()
    with torch.no_grad():
        assert not torch.allclose(model(batch)["p"], model(without)["p"], atol=1e-6)


def test_curvature_channel_reaches_the_model(
    cache: Path, split_file: Path
) -> None:
    """``SurrogateBase.get_curvature`` must pick the cached values up.

    Without the cache it returns a constant-zero channel, so a non-zero cached
    curvature has to change the assembled input features.
    """
    from src.models.common import SurrogateBase

    batch = _real_batch(cache, split_file)
    curv = SurrogateBase.get_curvature(batch)
    assert curv.shape == (batch["surf_pos"].shape[0], 1)
    assert torch.allclose(curv.reshape(-1), batch["curvature"])
    assert curv.abs().sum() > 0, "fixture curvature is +2, not zero"

    stripped = {k: v for k, v in batch.items() if k != "curvature"}
    assert SurrogateBase.get_curvature(stripped).abs().sum() == 0


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
