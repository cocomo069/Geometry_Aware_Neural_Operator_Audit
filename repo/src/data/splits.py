"""Deterministic construction of the six AirfRANS split manifests.

Implements CONTEXT.md section 5 (FROZEN).  Every manifest written to
``data/splits/<name>.json`` has the shape::

    {"name": str, "train": [sim, ...], "cal": [sim, ...], "test": [sim, ...],
     "seed": int, "description": str}

Sources
-------
The four *official* tasks (``full``, ``scarce``, ``reynolds``, ``aoa``) are read
straight out of the dataset's own ``manifest.json`` (the file that
``airfrans.dataset.load`` consults), so no VTU/VTP file ever has to be opened
and neither ``pyvista`` nor ``airfrans`` is imported here.  Keys in that file
are ``"<task>_train"`` / ``"<task>_test"``; the ``scarce`` task deliberately
reuses ``full_test`` as its test set (see ``airfrans.dataset.load``).

The two *derived* tasks (``shape5``, ``combined``) are built purely from the
simulation names, which encode every boundary condition -- see
``parse_sim_name`` and ``docs/DATA_NOTES.md``.

Calibration carving
-------------------
``cal`` is always carved out of the *train* pool, never out of ``test``::

    cal_n = min(100, max(1, round(0.20 * len(train))))

applied to ``shuffle(sorted(train), seed)``, taking the **last** ``cal_n``
entries.  For the ``full`` task (800 train sims) this yields exactly the
"last 100 of shuffled(seed=0) train" rule of CONTEXT.md section 5, and for
``scarce`` (200 train sims) exactly the "20% of train" rule; the remaining
tasks inherit the same single rule.

Everything here is deterministic: sorting before shuffling removes any
dependency on the ordering inside the dataset manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "SPLIT_NAMES",
    "OFFICIAL_TASKS",
    "NU_DATASET",
    "RE_BAND_MID",
    "SimName",
    "parse_sim_name",
    "read_raw_manifest",
    "carve_cal",
    "build_all_splits",
    "write_all_splits",
]

SPLIT_NAMES: tuple[str, ...] = (
    "full",
    "scarce",
    "reynolds",
    "aoa",
    "shape5",
    "combined",
)

OFFICIAL_TASKS: tuple[str, ...] = ("full", "scarce", "reynolds", "aoa")

#: Kinematic viscosity of air actually used when the AirfRANS dataset was
#: generated (m^2/s).  The ``airfrans.Simulation`` class recomputes ~1.5498e-5
#: from the temperature, which is *not* the generation value -- see the note in
#: the package docs and ``docs/DATA_NOTES.md``.  Reynolds numbers quoted in the
#: AirfRANS paper (and hence the official ``reynolds`` task bands) follow the
#: 1.56e-5 convention, so that is what the derived splits use.
NU_DATASET: float = 1.56e-5

#: Chord length of every AirfRANS airfoil (m).
CHORD: float = 1.0

#: Mid-Reynolds band, matching the official ``reynolds`` task train region
#: ("simulations with Reynolds number between 3 and 5 million are kept for the
#: trainset").
RE_BAND_MID: tuple[float, float] = (3.0e6, 5.0e6)

_CAL_FRACTION = 0.20
_CAL_MAX = 100


class SimName:
    """Parsed AirfRANS simulation name.

    AirfRANS names look like
    ``airFoil2D_SST_<U>_<AoA>_<naca digits...>`` , e.g.

    * ``airFoil2D_SST_43.597_5.932_3.551_3.1_1.0_18.252`` -> 4 NACA parameters
      -> **5-digit** series, and
    * ``airFoil2D_SST_54.2_2.1_0.0_0.0_12.0``            -> 3 NACA parameters
      -> **4-digit** series.

    The parameters are *real numbers*, not integers: the dataset samples the
    NACA design space continuously.  The last parameter is always the thickness
    in percent of chord; the preceding 2 (4-digit) or 3 (5-digit) parameters
    define the camber line -- exactly the slicing that
    ``airfrans.naca_generator.camber_line`` expects.

    Attributes
    ----------
    name : str
        The original simulation directory name.
    u_inf : float
        Inlet velocity magnitude in m/s.
    aoa_deg : float
        Angle of attack in degrees.
    naca_params : tuple[float, ...]
        3 values (4-digit series) or 4 values (5-digit series).
    n_digits : int
        4 or 5.
    thickness : float
        Airfoil thickness in percent of chord.
    re : float
        Reynolds number ``u_inf * CHORD / NU_DATASET``.
    """

    __slots__ = ("name", "u_inf", "aoa_deg", "naca_params", "n_digits")

    def __init__(
        self,
        name: str,
        u_inf: float,
        aoa_deg: float,
        naca_params: tuple[float, ...],
    ) -> None:
        self.name = name
        self.u_inf = float(u_inf)
        self.aoa_deg = float(aoa_deg)
        self.naca_params = tuple(float(p) for p in naca_params)
        if len(self.naca_params) == 3:
            self.n_digits = 4
        elif len(self.naca_params) == 4:
            self.n_digits = 5
        else:  # pragma: no cover - guarded by parse_sim_name
            raise ValueError(
                f"{name!r}: expected 3 or 4 NACA parameters, "
                f"got {len(self.naca_params)}"
            )

    @property
    def thickness(self) -> float:
        return self.naca_params[-1]

    @property
    def camber_params(self) -> tuple[float, ...]:
        """Parameters consumed by ``airfrans.naca_generator.camber_line``."""
        return self.naca_params[:-1]

    @property
    def re(self) -> float:
        return self.u_inf * CHORD / NU_DATASET

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SimName({self.name!r}, u_inf={self.u_inf:.3f}, "
            f"aoa_deg={self.aoa_deg:.3f}, n_digits={self.n_digits})"
        )


def parse_sim_name(name: str) -> SimName:
    """Parse an AirfRANS simulation directory name.

    Parameters
    ----------
    name : str
        e.g. ``"airFoil2D_SST_43.597_5.932_3.551_3.1_1.0_18.252"``.

    Returns
    -------
    SimName

    Raises
    ------
    ValueError
        If the name does not have the expected number of underscore-separated
        fields or the numeric fields do not parse.
    """
    parts = name.split("_")
    if len(parts) not in (7, 8):
        raise ValueError(
            f"{name!r}: expected 7 (4-digit) or 8 (5-digit) '_'-separated "
            f"fields, got {len(parts)}"
        )
    try:
        u_inf = float(parts[2])
        aoa_deg = float(parts[3])
        naca_params = tuple(float(p) for p in parts[4:])
    except ValueError as exc:  # pragma: no cover - malformed name
        raise ValueError(f"{name!r}: could not parse numeric fields") from exc
    return SimName(name, u_inf, aoa_deg, naca_params)


def read_raw_manifest(raw_root: Path) -> dict[str, list[str]]:
    """Read the dataset's own ``manifest.json``.

    Parameters
    ----------
    raw_root : Path
        Directory holding the unzipped dataset, i.e. the directory that
        contains ``manifest.json`` and one sub-directory per simulation.
        Typically ``data/raw/Dataset``.

    Returns
    -------
    dict[str, list[str]]
        Mapping ``"<task>_<train|test>" -> [sim_name, ...]``.
    """
    raw_root = Path(raw_root)
    manifest_path = raw_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"AirfRANS manifest not found at {manifest_path}. Download the "
            f"preprocessed dataset first (scripts/download_data.py)."
        )
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):  # pragma: no cover - corrupt file
        raise ValueError(f"{manifest_path}: expected a JSON object")
    return {str(k): [str(s) for s in v] for k, v in manifest.items()}


def all_sim_names(manifest: dict[str, list[str]]) -> list[str]:
    """Sorted union of every simulation named anywhere in the raw manifest."""
    names: set[str] = set()
    for v in manifest.values():
        names.update(v)
    return sorted(names)


def carve_cal(
    train: Sequence[str],
    seed: int = 0,
    fraction: float = _CAL_FRACTION,
    cap: int = _CAL_MAX,
) -> tuple[list[str], list[str]]:
    """Split a train pool into ``(train, cal)``.

    ``cal`` is the last ``min(cap, max(1, round(fraction * n)))`` entries of a
    seeded shuffle of ``sorted(train)``.  Deterministic for a given seed.

    Returns
    -------
    (train_out, cal) : tuple[list[str], list[str]]
        Both sorted; disjoint by construction; their union is ``set(train)``.
    """
    pool = sorted(set(train))
    n = len(pool)
    if n == 0:
        return [], []
    cal_n = min(cap, max(1, int(round(fraction * n))))
    cal_n = min(cal_n, n - 1) if n > 1 else 0
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    shuffled = [pool[i] for i in order]
    if cal_n == 0:
        return sorted(shuffled), []
    return sorted(shuffled[:-cal_n]), sorted(shuffled[-cal_n:])


def _official_split(
    manifest: dict[str, list[str]], task: str, seed: int
) -> tuple[list[str], list[str], list[str]]:
    train_key = f"{task}_train"
    # airfrans.dataset.load: the scarce task reuses the full test set.
    test_key = "full_test" if task == "scarce" else f"{task}_test"
    for key in (train_key, test_key):
        if key not in manifest:
            raise KeyError(
                f"raw manifest has no key {key!r}; available: "
                f"{sorted(manifest)}"
            )
    train, cal = carve_cal(manifest[train_key], seed=seed)
    test = sorted(set(manifest[test_key]))
    return train, cal, test


def _derived_shape5(
    names: Iterable[str], seed: int
) -> tuple[list[str], list[str], list[str]]:
    """train = every NACA 4-digit sim, test = every NACA 5-digit sim."""
    four, five = [], []
    for n in names:
        (four if parse_sim_name(n).n_digits == 4 else five).append(n)
    train, cal = carve_cal(four, seed=seed)
    return train, cal, sorted(five)


def _derived_combined(
    names: Iterable[str],
    seed: int,
    re_band: tuple[float, float] = RE_BAND_MID,
) -> tuple[list[str], list[str], list[str]]:
    """Joint shape + Reynolds shift.

    train pool = 4-digit AND Re inside ``re_band``;
    test       = 5-digit AND Re outside ``re_band``.

    Simulations that are 4-digit/outer-Re or 5-digit/mid-Re are *unused* -- the
    split deliberately maximises the distribution shift between train and test.
    """
    lo, hi = re_band
    train_pool, test = [], []
    for n in names:
        s = parse_sim_name(n)
        mid = lo <= s.re <= hi
        if s.n_digits == 4 and mid:
            train_pool.append(n)
        elif s.n_digits == 5 and not mid:
            test.append(n)
    train, cal = carve_cal(train_pool, seed=seed)
    return train, cal, sorted(test)


_DESCRIPTIONS = {
    "full": (
        "Official AirfRANS 'full' task (800 train / 200 test, same "
        "distribution -> interpolation). cal = last 100 of shuffle(seed) of "
        "the official train list."
    ),
    "scarce": (
        "Official AirfRANS 'scarce' task (200 train, test == full_test). "
        "cal = 20% of the official train list (seeded shuffle)."
    ),
    "reynolds": (
        "Official AirfRANS 'reynolds' task: train Re in [3e6, 5e6], test "
        "outside -> Reynolds extrapolation. cal carved from train."
    ),
    "aoa": (
        "Official AirfRANS 'aoa' task: train AoA in [-2.5, 12.5] deg, test "
        "outside -> angle-of-attack extrapolation. cal carved from train."
    ),
    "shape5": (
        "DERIVED shape-shift split: train = all NACA 4-digit-series sims, "
        "test = all NACA 5-digit-series sims (series inferred from the number "
        "of NACA parameters in the sim name: 3 -> 4-digit, 4 -> 5-digit). "
        "cal carved from train."
    ),
    "combined": (
        "DERIVED joint shape+Reynolds shift: train = 4-digit AND "
        f"Re in [{RE_BAND_MID[0]:.3g}, {RE_BAND_MID[1]:.3g}] (nu = "
        f"{NU_DATASET:.3g} m^2/s, chord = 1 m); test = 5-digit AND Re outside "
        "that band. Sims in neither category are unused. cal carved from "
        "train."
    ),
}


def build_all_splits(
    raw_root: Path,
    seed: int = 0,
    names: Sequence[str] | None = None,
) -> dict[str, dict]:
    """Build every split manifest in memory.

    Parameters
    ----------
    raw_root : Path
        Directory containing the dataset ``manifest.json``.  Ignored when
        ``names`` *and* a manifest dict are supplied via ``build_splits_from``.
    seed : int
        Seed for the calibration carve (CONTEXT.md fixes this at 0).
    names : sequence of str, optional
        Override the universe of simulation names for the derived splits.
        Defaults to the union of everything in the raw manifest.

    Returns
    -------
    dict[str, dict]
        ``name -> manifest dict`` for all of ``SPLIT_NAMES``.
    """
    manifest = read_raw_manifest(raw_root)
    return build_splits_from(manifest, seed=seed, names=names)


def build_splits_from(
    manifest: dict[str, list[str]],
    seed: int = 0,
    names: Sequence[str] | None = None,
) -> dict[str, dict]:
    """Same as :func:`build_all_splits` but from an already-loaded manifest.

    Kept separate so tests can exercise the logic without the 10 GB dataset.
    """
    universe = list(names) if names is not None else all_sim_names(manifest)
    out: dict[str, dict] = {}

    for task in OFFICIAL_TASKS:
        train, cal, test = _official_split(manifest, task, seed)
        out[task] = _pack(task, train, cal, test, seed)

    train, cal, test = _derived_shape5(universe, seed)
    out["shape5"] = _pack("shape5", train, cal, test, seed)

    train, cal, test = _derived_combined(universe, seed)
    out["combined"] = _pack("combined", train, cal, test, seed)

    return out


def _pack(
    name: str,
    train: Sequence[str],
    cal: Sequence[str],
    test: Sequence[str],
    seed: int,
) -> dict:
    train, cal, test = list(train), list(cal), list(test)
    _assert_disjoint(name, train, cal, test)
    return {
        "name": name,
        "train": train,
        "cal": cal,
        "test": test,
        "seed": int(seed),
        "description": _DESCRIPTIONS[name],
        "n_train": len(train),
        "n_cal": len(cal),
        "n_test": len(test),
    }


def _assert_disjoint(
    name: str, train: Sequence[str], cal: Sequence[str], test: Sequence[str]
) -> None:
    for label, seq in (("train", train), ("cal", cal), ("test", test)):
        if len(set(seq)) != len(seq):
            raise AssertionError(f"split {name!r}: duplicates in {label!r}")
    pairs = (("train", train, "cal", cal), ("train", train, "test", test),
             ("cal", cal, "test", test))
    for a_name, a, b_name, b in pairs:
        overlap = set(a) & set(b)
        if overlap:
            raise AssertionError(
                f"split {name!r}: {a_name} and {b_name} overlap on "
                f"{len(overlap)} sims, e.g. {sorted(overlap)[:3]}"
            )


def write_all_splits(
    raw_root: Path,
    out_dir: Path,
    seed: int = 0,
) -> dict[str, Path]:
    """Build and write all six split JSONs.

    Returns
    -------
    dict[str, Path]
        ``name -> written path``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = build_all_splits(Path(raw_root), seed=seed)
    written: dict[str, Path] = {}
    for name, payload in splits.items():
        path = out_dir / f"{name}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        written[name] = path
    return written


def load_split(split_file: Path) -> dict:
    """Load and validate one split manifest."""
    split_file = Path(split_file)
    with split_file.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    for key in ("name", "train", "cal", "test", "seed", "description"):
        if key not in payload:
            raise KeyError(f"{split_file}: missing key {key!r}")
    _assert_disjoint(
        payload["name"], payload["train"], payload["cal"], payload["test"]
    )
    return payload


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Write the six AirfRANS split manifests to data/splits/. Reads "
            "only the dataset's manifest.json -- no VTU/VTP files are opened."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/Dataset"),
        help="directory containing the AirfRANS manifest.json "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/splits"),
        help="where to write <name>.json (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="calibration carve seed"
    )
    args = parser.parse_args(argv)

    written = write_all_splits(args.raw_root, args.out_dir, seed=args.seed)
    for name, path in written.items():
        payload = load_split(path)
        print(
            f"{name:9s} train={payload['n_train']:4d} "
            f"cal={payload['n_cal']:4d} test={payload['n_test']:4d} -> {path}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
