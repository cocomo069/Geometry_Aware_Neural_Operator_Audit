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
from pathlib import Path

import pytest

from src.data.splits import (
    NU_DATASET,
    OFFICIAL_TASKS,
    RE_BAND_MID,
    SPLIT_NAMES,
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
