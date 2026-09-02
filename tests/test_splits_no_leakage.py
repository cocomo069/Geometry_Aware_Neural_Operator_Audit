"""Split integrity tests (CONTEXT.md sections 5 and 11).

Two layers:

1. **Logic tests** on a synthetic raw manifest built in ``tmp_path``.  These
   exercise every rule in ``src/data/splits.py`` and pass with no dataset
   downloaded, no ``airfrans``/``pyvista`` installed, and no GPU.
2. **Manifest tests** over whatever real ``data/splits/*.json`` happen to exist,
   asserting the CONTEXT.md rule: *cal never overlaps train or test; test is
   never touched.*  They skip cleanly before the first cache build.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.data.splits import (
    DATA_EFFICIENCY_SEEDS,
    DATA_EFFICIENCY_SIZES,
    NU_DATASET,
    OFFICIAL_TASKS,
    RE_BAND_MID,
    SPLIT_NAMES,
    build_data_efficiency_splits,
    build_splits_from,
    carve_cal,
    load_split,
    parse_sim_name,
    write_all_splits,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "data" / "splits"


# ---------------------------------------------------------------------------
# synthetic dataset fixtures
# ---------------------------------------------------------------------------
def _make_name(u: float, aoa: float, params: tuple[float, ...]) -> str:
    tail = "_".join(f"{p}" for p in params)
    return f"airFoil2D_SST_{u}_{aoa}_{tail}"


def _synthetic_names(n: int = 120) -> list[str]:
    """A miniature AirfRANS universe spanning both NACA series and all Re."""
    names: list[str] = []
    for i in range(n):
        # Re from 2e6 to 6e6 <-> u from ~31.2 to ~93.6 at nu = 1.56e-5.
        u = round(31.2 + (93.6 - 31.2) * i / (n - 1), 3)
        aoa = round(-5.0 + 20.0 * ((i * 7) % n) / (n - 1), 3)
        if i % 2 == 0:  # 4-digit series: 3 NACA parameters
            params = (round(2.0 + 0.01 * i, 3), 4.0, round(9.0 + 0.05 * i, 3))
        else:  # 5-digit series: 4 NACA parameters
            params = (
                round(2.0 + 0.01 * i, 3),
                3.0,
                float(i % 2),
                round(12.0 + 0.05 * i, 3),
            )
        names.append(_make_name(u, aoa, params))
    assert len(set(names)) == len(names), "synthetic names collided"
    return names


@pytest.fixture(scope="module")
def universe() -> list[str]:
    return _synthetic_names()


@pytest.fixture(scope="module")
def raw_manifest(universe: list[str]) -> dict[str, list[str]]:
    """A stand-in for the dataset's own ``manifest.json``."""
    n = len(universe)
    full_train = universe[: int(0.8 * n)]
    full_test = universe[int(0.8 * n) :]

    def re_of(name: str) -> float:
        return parse_sim_name(name).re

    def aoa_of(name: str) -> float:
        return parse_sim_name(name).aoa_deg

    lo, hi = RE_BAND_MID
    return {
        "full_train": full_train,
        "full_test": full_test,
        "scarce_train": full_train[: n // 5],
        "reynolds_train": [s for s in universe if lo <= re_of(s) <= hi],
        "reynolds_test": [s for s in universe if not lo <= re_of(s) <= hi],
        "aoa_train": [s for s in universe if -2.5 <= aoa_of(s) <= 12.5],
        "aoa_test": [s for s in universe if not -2.5 <= aoa_of(s) <= 12.5],
    }


@pytest.fixture(scope="module")
def built(raw_manifest: dict[str, list[str]]) -> dict[str, dict]:
    return build_splits_from(raw_manifest, seed=0)


# ---------------------------------------------------------------------------
# name parsing
# ---------------------------------------------------------------------------
def test_parse_sim_name_5digit() -> None:
    """The upstream documentation's own worked example."""
    s = parse_sim_name("airFoil2D_SST_43.597_5.932_3.551_3.1_1.0_18.252")
    assert s.u_inf == pytest.approx(43.597)
    assert s.aoa_deg == pytest.approx(5.932)
    assert s.naca_params == pytest.approx((3.551, 3.1, 1.0, 18.252))
    assert s.n_digits == 5
    assert s.thickness == pytest.approx(18.252)
    assert s.camber_params == pytest.approx((3.551, 3.1, 1.0))
    assert s.re == pytest.approx(43.597 / NU_DATASET)


def test_parse_sim_name_4digit() -> None:
    """A NACA 0012 name ends in ``_0_0_12`` -> three parameters."""
    s = parse_sim_name("airFoil2D_SST_54.2_2.1_0.0_0.0_12.0")
    assert s.n_digits == 4
    assert s.naca_params == pytest.approx((0.0, 0.0, 12.0))
    assert s.camber_params == pytest.approx((0.0, 0.0))


@pytest.mark.parametrize(
    "bad",
    [
        "airFoil2D_SST_43.5_1.0",  # too few fields
        "airFoil2D_SST_43.5_1.0_1_2_3_4_5",  # too many fields
        "airFoil2D_SST_notanumber_1.0_0_0_12",  # unparseable
    ],
)
def test_parse_sim_name_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_sim_name(bad)


def test_series_discriminator_is_parameter_count(universe: list[str]) -> None:
    for name in universe:
        s = parse_sim_name(name)
        assert s.n_digits == (4 if len(s.naca_params) == 3 else 5)


# ---------------------------------------------------------------------------
# calibration carve
# ---------------------------------------------------------------------------
def test_carve_cal_is_deterministic() -> None:
    pool = [f"sim_{i}" for i in range(250)]
    a = carve_cal(pool, seed=0)
    b = carve_cal(list(reversed(pool)), seed=0)
    assert a == b, "carve must not depend on input ordering"


def test_carve_cal_partitions_exactly() -> None:
    pool = [f"sim_{i}" for i in range(137)]
    train, cal = carve_cal(pool, seed=0)
    assert set(train) & set(cal) == set()
    assert set(train) | set(cal) == set(pool)
    assert len(train) + len(cal) == len(pool)


def test_carve_cal_matches_frozen_rules() -> None:
    """CONTEXT.md section 5: 100 for ``full`` (800), 20 % for ``scarce`` (200)."""
    _, cal_full = carve_cal([f"s{i}" for i in range(800)], seed=0)
    assert len(cal_full) == 100
    _, cal_scarce = carve_cal([f"s{i}" for i in range(200)], seed=0)
    assert len(cal_scarce) == 40


def test_carve_cal_never_empties_train() -> None:
    train, cal = carve_cal(["only"], seed=0)
    assert train == ["only"] and cal == []


# ---------------------------------------------------------------------------
# split construction
# ---------------------------------------------------------------------------
def test_all_six_splits_built(built: dict[str, dict]) -> None:
    assert set(built) == set(SPLIT_NAMES)


@pytest.mark.parametrize("name", SPLIT_NAMES)
def test_synthetic_split_schema(built: dict[str, dict], name: str) -> None:
    payload = built[name]
    for key in ("name", "train", "cal", "test", "seed", "description"):
        assert key in payload, f"{name}: missing {key!r}"
    assert payload["name"] == name
    assert payload["seed"] == 0
    assert isinstance(payload["description"], str) and payload["description"]
    for subset in ("train", "cal", "test"):
        assert isinstance(payload[subset], list)
        assert all(isinstance(s, str) for s in payload[subset])


@pytest.mark.parametrize("name", SPLIT_NAMES)
def test_synthetic_split_pairwise_disjoint(
    built: dict[str, dict], name: str
) -> None:
    _assert_no_leakage(name, built[name])


@pytest.mark.parametrize("name", SPLIT_NAMES)
def test_synthetic_split_nonempty(built: dict[str, dict], name: str) -> None:
    payload = built[name]
    assert payload["train"], f"{name}: empty train"
    assert payload["test"], f"{name}: empty test"
    assert payload["cal"], f"{name}: empty cal"


def test_build_is_deterministic(raw_manifest: dict[str, list[str]]) -> None:
    a = build_splits_from(raw_manifest, seed=0)
    b = build_splits_from(raw_manifest, seed=0)
    assert a == b


def test_different_seed_moves_cal(raw_manifest: dict[str, list[str]]) -> None:
    a = build_splits_from(raw_manifest, seed=0)["full"]
    b = build_splits_from(raw_manifest, seed=1)["full"]
    assert a["cal"] != b["cal"]
    # ...but the test set is fixed by the official task, seed or no seed.
    assert a["test"] == b["test"]


def test_official_train_pool_is_preserved(
    built: dict[str, dict], raw_manifest: dict[str, list[str]]
) -> None:
    """train + cal must reconstitute the official train list exactly."""
    for task in OFFICIAL_TASKS:
        payload = built[task]
        recon = set(payload["train"]) | set(payload["cal"])
        assert recon == set(raw_manifest[f"{task}_train"])


def test_scarce_reuses_full_test(built: dict[str, dict]) -> None:
    """airfrans.dataset.load maps scarce/test onto full_test."""
    assert built["scarce"]["test"] == built["full"]["test"]


def test_shape5_separates_the_naca_series(built: dict[str, dict]) -> None:
    payload = built["shape5"]
    for subset in ("train", "cal"):
        assert all(parse_sim_name(s).n_digits == 4 for s in payload[subset])
    assert all(parse_sim_name(s).n_digits == 5 for s in payload["test"])


def test_combined_is_a_joint_shape_and_reynolds_shift(
    built: dict[str, dict],
) -> None:
    lo, hi = RE_BAND_MID
    payload = built["combined"]
    for subset in ("train", "cal"):
        for s in payload[subset]:
            p = parse_sim_name(s)
            assert p.n_digits == 4
            assert lo <= p.re <= hi
    for s in payload["test"]:
        p = parse_sim_name(s)
        assert p.n_digits == 5
        assert not lo <= p.re <= hi


def test_derived_splits_draw_from_the_whole_universe(
    built: dict[str, dict], universe: list[str]
) -> None:
    covered = set()
    for name in ("shape5",):
        payload = built[name]
        covered |= set(payload["train"]) | set(payload["cal"]) | set(payload["test"])
    assert covered == set(universe), "shape5 must partition every simulation"


# ---------------------------------------------------------------------------
# data-efficiency sub-splits (PLAN_PHASE2 Phase B) -- synthetic logic
# ---------------------------------------------------------------------------
_DE_SYNTH_SIZES = (10, 20, 40)


@pytest.fixture(scope="module")
def dataeff(built: dict[str, dict]) -> dict[str, dict]:
    return build_data_efficiency_splits(
        built["full"], sizes=_DE_SYNTH_SIZES, seeds=(0, 1, 2)
    )


def test_dataeff_emits_every_size_seed(dataeff: dict[str, dict]) -> None:
    expected = {
        f"full_n{size}_s{seed}"
        for size in _DE_SYNTH_SIZES
        for seed in (0, 1, 2)
    }
    assert set(dataeff) == expected


@pytest.mark.parametrize("size", _DE_SYNTH_SIZES)
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_dataeff_train_is_subset_with_shared_cal_test(
    dataeff: dict[str, dict], built: dict[str, dict], size: int, seed: int
) -> None:
    full = built["full"]
    payload = dataeff[f"full_n{size}_s{seed}"]
    assert set(payload["train"]).issubset(set(full["train"]))
    assert payload["cal"] == full["cal"]
    assert payload["test"] == full["test"]
    assert payload["n_train"] == size == len(payload["train"])
    assert payload["seed"] == seed
    _assert_no_leakage(payload["name"], payload)


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_dataeff_sizes_are_nested_within_a_seed(
    dataeff: dict[str, dict], seed: int
) -> None:
    prev: set[str] = set()
    for size in _DE_SYNTH_SIZES:
        cur = set(dataeff[f"full_n{size}_s{seed}"]["train"])
        assert prev.issubset(cur), f"n{size} not nested (seed {seed})"
        prev = cur


def test_dataeff_seeds_draw_different_subsets(dataeff: dict[str, dict]) -> None:
    a = set(dataeff["full_n10_s0"]["train"])
    b = set(dataeff["full_n10_s1"]["train"])
    assert a != b, "different seeds must not reproduce the same subset"


def test_dataeff_is_deterministic(built: dict[str, dict]) -> None:
    a = build_data_efficiency_splits(built["full"], sizes=(10, 20), seeds=(0, 1))
    b = build_data_efficiency_splits(built["full"], sizes=(10, 20), seeds=(0, 1))
    assert a == b


def test_dataeff_rejects_oversized_request(built: dict[str, dict]) -> None:
    n_full = len(built["full"]["train"])
    with pytest.raises(ValueError, match="exceeds full-train"):
        build_data_efficiency_splits(built["full"], sizes=(n_full + 1,), seeds=(0,))


# ---------------------------------------------------------------------------
# round trip through disk
# ---------------------------------------------------------------------------
def test_write_and_load_roundtrip(
    tmp_path: Path, raw_manifest: dict[str, list[str]]
) -> None:
    raw_root = tmp_path / "Dataset"
    raw_root.mkdir()
    (raw_root / "manifest.json").write_text(
        json.dumps(raw_manifest), encoding="utf-8"
    )
    out_dir = tmp_path / "splits"

    written = write_all_splits(raw_root, out_dir, seed=0)
    assert set(written) == set(SPLIT_NAMES)

    for name, path in written.items():
        assert path.is_file()
        payload = load_split(path)  # load_split re-checks disjointness
        assert payload["name"] == name
        _assert_no_leakage(name, payload)

    # Idempotent: rewriting produces byte-identical files.
    before = {p: p.read_bytes() for p in written.values()}
    write_all_splits(raw_root, out_dir, seed=0)
    assert {p: p.read_bytes() for p in written.values()} == before


def test_load_split_rejects_a_leaky_manifest(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "name": "bad",
                "train": ["a", "b"],
                "cal": ["b"],  # leak
                "test": ["c"],
                "seed": 0,
                "description": "deliberately leaky",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="overlap"):
        load_split(bad)


# ---------------------------------------------------------------------------
# the real manifests, once they exist
# ---------------------------------------------------------------------------
def _real_split_files() -> list[Path]:
    if not SPLITS_DIR.is_dir():
        return []
    return sorted(SPLITS_DIR.glob("*.json"))


_REAL = _real_split_files()


@pytest.mark.skipif(not _REAL, reason="no data/splits/*.json built yet")
@pytest.mark.parametrize(
    "path", _REAL, ids=[p.stem for p in _REAL] or ["none"]
)
def test_real_manifest_has_no_leakage(path: Path) -> None:
    payload = load_split(path)
    _assert_no_leakage(path.stem, payload)


@pytest.mark.skipif(not _REAL, reason="no data/splits/*.json built yet")
def test_real_manifests_cover_the_frozen_names() -> None:
    """Once splits are built at all, all six from CONTEXT.md must be there."""
    have = {p.stem for p in _REAL}
    missing = set(SPLIT_NAMES) - have
    assert not missing, f"missing split manifests: {sorted(missing)}"


# ---------------------------------------------------------------------------
# the real data-efficiency manifests, once they exist
# ---------------------------------------------------------------------------
def _real_dataeff_files() -> list[Path]:
    if not SPLITS_DIR.is_dir():
        return []
    return sorted(SPLITS_DIR.glob("full_n*_s*.json"))


_REAL_DE = _real_dataeff_files()
_DE_STEM_RE = re.compile(r"^full_n(\d+)_s(\d+)$")


@pytest.mark.skipif(not _REAL_DE, reason="no data-efficiency manifests built yet")
@pytest.mark.parametrize(
    "path", _REAL_DE, ids=[p.stem for p in _REAL_DE] or ["none"]
)
def test_real_dataeff_matches_full(path: Path) -> None:
    """Each full_nXX_sSEED: train ⊆ full-train; cal/test == full's; disjoint."""
    full = load_split(SPLITS_DIR / "full.json")
    payload = load_split(path)
    _assert_no_leakage(path.stem, payload)
    assert set(payload["train"]).issubset(set(full["train"])), (
        f"{path.stem}: train is not a subset of full's train"
    )
    assert payload["cal"] == full["cal"], f"{path.stem}: cal differs from full's"
    assert payload["test"] == full["test"], f"{path.stem}: test differs from full's"
    m = _DE_STEM_RE.match(path.stem)
    assert m, f"{path.stem}: unexpected data-efficiency filename"
    size, seed = int(m.group(1)), int(m.group(2))
    assert len(payload["train"]) == size, f"{path.stem}: n_train != {size}"
    assert payload["seed"] == seed, f"{path.stem}: seed field != {seed}"


@pytest.mark.skipif(not _REAL_DE, reason="no data-efficiency manifests built yet")
def test_real_dataeff_covers_every_size_seed() -> None:
    """If any data-efficiency manifest exists, the full grid must be present."""
    have = {p.stem for p in _REAL_DE}
    expected = {
        f"full_n{size}_s{seed}"
        for size in DATA_EFFICIENCY_SIZES
        for seed in DATA_EFFICIENCY_SEEDS
    }
    missing = expected - have
    assert not missing, f"missing data-efficiency manifests: {sorted(missing)}"


# ---------------------------------------------------------------------------
# shared assertion
# ---------------------------------------------------------------------------
def _assert_no_leakage(name: str, payload: dict) -> None:
    """CONTEXT.md section 5: pairwise disjoint, and no duplicates within."""
    subsets = {k: list(payload[k]) for k in ("train", "cal", "test")}
    for key, seq in subsets.items():
        assert len(set(seq)) == len(seq), f"{name}: duplicates inside {key!r}"
    keys = ("train", "cal", "test")
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            overlap = set(subsets[a]) & set(subsets[b])
            assert not overlap, (
                f"{name}: {a} and {b} share {len(overlap)} sims, "
                f"e.g. {sorted(overlap)[:3]}"
            )
