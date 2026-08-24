"""Convert raw AirfRANS simulations into the frozen per-sim ``.npz`` cache.

Implements CONTEXT.md section 4 (FROZEN).  For every simulation it writes
``data/processed/airfrans/<sim_name>.npz`` and, once all requested sims are
done, ``manifest.json`` plus ``norm_stats.json``.

Key conventions (all verified against the ``airfrans`` package source; see
``docs/DATA_NOTES.md`` for the full derivation):

* **Surface identification.**  ``Simulation.surface`` is the no-slip mask
  ``internal U_x == 0``.  The airfoil patch (``*_aerofoil.vtp``) carries the
  same points in a *different* order; we key everything to the **airfoil patch**
  order and then re-index into contour order.
* **Contour ordering.**  Walked from the ``.vtp`` line connectivity, started at
  the trailing edge (max x) and oriented **counter-clockwise** (positive
  shoelace area), i.e. TE -> extrado -> LE -> intrado -> TE.
* **Normals.**  AirfRANS ships *inward*-pointing normals (PyVista
  ``compute_normals(flip_normals=False)``, documented since package v0.1.4).
  CONTEXT.md section 4 demands **outward** (into the fluid), so we store
  ``-airfoil_normals`` and assert the flip is right two independent ways.
* **Quadrature.**  ``surf_ds[i] = 0.5 * sum(len(c) for c incident to i)``.
  Summing a point field against these weights is *identical* to the package's
  own ``point-to-cell average x cell Length`` trapezoid rule for linear
  quantities.  ``sum(surf_ds) == perimeter``.
* **Units.**  ``surf_p`` and ``vol_p`` are OpenFOAM incompressible *kinematic*
  pressure ``p/rho`` in m^2/s^2; ``surf_tau`` is the *kinematic* viscous wall
  traction ``2 nu S . n_out`` in m^2/s^2.  Multiply both by ``rho`` (stored
  per-sim) to get Pa and Pa respectively.

The builder is **idempotent**: a sim whose ``.npz`` already exists with a
matching ``cache_version`` is skipped unless ``--force`` is given.

Usage
-----
    .venv/Scripts/python.exe scripts/build_cache.py --limit 5      # smoke test
    .venv/Scripts/python.exe scripts/build_cache.py                # full build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import (  # noqa: E402
    all_sim_names,
    load_split,
    parse_sim_name,
    read_raw_manifest,
    write_all_splits,
)

#: Bump when the on-disk layout changes; stale caches are then rebuilt.
#: v2 (D-017) added the precomputed static geometry keys below.
CACHE_VERSION = 2

# -- D-017 precomputed static geometry -------------------------------------
#: M2's default latent grid (src/models/sdf_fno.py DEFAULT_CONFIG).
GRID_BBOX = (-0.5, 1.5, -1.0, 1.0)   # (xmin, xmax, ymin, ymax)
GRID_RES = (64, 64)                  # (n_y, n_x) -- y first, as M2 indexes it
#: M1's neighbourhood size (src/models/gnn.py DEFAULT_CONFIG k=16).
KNN_K = 16

#: CONTEXT.md section 4: "volume node coords (subsampled to <= 32768 if larger)".
MAX_VOL_POINTS = 32768

#: Fields the normalisation statistics cover.  Per-point fields accumulate over
#: every point of every train sim; per-sim fields contribute one sample each.
_POINT_FIELDS = {
    "surf_pos": 2,
    "surf_normal": 2,
    "surf_ds": 1,
    "surf_p": 1,
    "surf_tau": 2,
    "vol_pos": 2,
    "vol_u": 2,
    "vol_p": 1,
    "vol_nut": 1,
    "vol_sdf": 1,
    "curvature": 1,          # optional (D-017); skipped when absent
}
_SIM_FIELDS = {
    "cond": 2,
    "cl_true": 1,
    "cd_true": 1,
    "re": 1,
    "aoa_deg": 1,
    "u_inf_mag": 1,
}


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _edges_from_polydata_lines(lines: np.ndarray) -> np.ndarray:
    """Extract ``(E, 2)`` point-index pairs from a PyVista ``lines`` array.

    ``lines`` is the VTK flat connectivity ``[n0, i0, i1, ..., n1, j0, ...]``.
    The airfoil patch is a 1-D slice, so every cell has ``n == 2``.
    """
    lines = np.asarray(lines).ravel()
    edges = []
    i = 0
    while i < lines.size:
        n = int(lines[i])
        if n < 2:
            raise ValueError(f"degenerate line cell with {n} points")
        ids = lines[i + 1 : i + 1 + n].astype(np.int64)
        for a, b in zip(ids[:-1], ids[1:]):
            edges.append((int(a), int(b)))
        i += n + 1
    if not edges:
        raise ValueError("airfoil patch has no line cells")
    return np.asarray(edges, dtype=np.int64)


def _signed_area(pos: np.ndarray) -> float:
    """Shoelace signed area of the closed polygon through ``pos`` (N, 2)."""
    x, y = pos[:, 0], pos[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def contour_order(pos: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Order airfoil-patch points along the contour, TE -> around -> TE, CCW.

    Parameters
    ----------
    pos : (N, 2) float array
        Airfoil patch point coordinates.
    edges : (E, 2) int array
        Point-index pairs from the patch line cells.

    Returns
    -------
    (N,) int64 array
        Permutation ``order`` such that ``pos[order]`` walks the contour
        counter-clockwise starting at the trailing edge (largest x).

    Raises
    ------
    ValueError
        If the connectivity is not a single simple path/cycle covering every
        point.
    """
    n = pos.shape[0]
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        if b not in adj[a]:
            adj[a].append(int(b))
        if a not in adj[b]:
            adj[b].append(int(a))

    degrees = np.array([len(a) for a in adj], dtype=np.int64)
    if np.any(degrees == 0):
        raise ValueError(
            f"{int(np.sum(degrees == 0))} airfoil points are not referenced by "
            "any line cell"
        )
    if np.any(degrees > 2):
        raise ValueError("airfoil contour is not a simple curve (degree > 2)")

    endpoints = np.flatnonzero(degrees == 1)
    if endpoints.size == 0:
        start = int(np.argmax(pos[:, 0]))  # trailing edge
    elif endpoints.size == 2:
        # Open polyline (blunt/unclosed TE): must start at an endpoint.
        start = int(endpoints[np.argmax(pos[endpoints, 0])])
    else:
        raise ValueError(
            f"airfoil contour has {endpoints.size} loose ends; expected 0 "
            "(closed) or 2 (open)"
        )

    order = [start]
    prev = -1
    cur = start
    while True:
        nxt = None
        for cand in adj[cur]:
            if cand != prev:
                nxt = cand
                break
        if nxt is None or nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
        if len(order) > n:  # pragma: no cover - guarded by degree checks
            raise ValueError("contour walk did not terminate")

    order_arr = np.asarray(order, dtype=np.int64)
    if order_arr.size != n:
        raise ValueError(
            f"contour walk covered {order_arr.size}/{n} points -- the airfoil "
            "patch is not a single connected curve"
        )

    if _signed_area(pos[order_arr]) < 0.0:  # clockwise -> flip, keep TE first
        order_arr = np.concatenate([order_arr[:1], order_arr[1:][::-1]])
    return order_arr


def quadrature_weights(pos_ordered: np.ndarray, closed: bool) -> np.ndarray:
    """Per-point facet-length share, ``sum == perimeter``.

    Parameters
    ----------
    pos_ordered : (N, 2) float array
        Contour-ordered points.
    closed : bool
        Whether the last point connects back to the first.

    Returns
    -------
    (N,) float64 array
    """
    n = pos_ordered.shape[0]
    if closed:
        seg = np.linalg.norm(np.roll(pos_ordered, -1, axis=0) - pos_ordered, axis=1)
        ds = 0.5 * (seg + np.roll(seg, 1))
    else:
        seg = np.linalg.norm(np.diff(pos_ordered, axis=0), axis=1)
        ds = np.zeros(n, dtype=np.float64)
        ds[:-1] += 0.5 * seg
        ds[1:] += 0.5 * seg
    return ds


def _geometric_outward_normals(
    pos_ordered: np.ndarray, closed: bool
) -> np.ndarray:
    """Outward unit normals implied by a CCW contour: ``n = (t_y, -t_x)``."""
    if closed:
        tangent = np.roll(pos_ordered, -1, axis=0) - np.roll(pos_ordered, 1, axis=0)
    else:
        tangent = np.gradient(pos_ordered, axis=0)
    nrm = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    mag = np.linalg.norm(nrm, axis=1, keepdims=True)
    mag[mag == 0.0] = 1.0
    return nrm / mag


def geometry_extras(
    surf_pos: np.ndarray,
    k: int = KNN_K,
    bbox: Sequence[float] = GRID_BBOX,
    res: Sequence[int] = GRID_RES,
) -> dict[str, np.ndarray]:
    """Precompute the static per-sim geometry of D-017.

    These three arrays depend only on the airfoil shape, never on the flow, so
    recomputing them every training step is pure waste (A3 measured M2 at
    2.45 s/forward, CPU-bound on exactly this).

    Parameters
    ----------
    surf_pos : (Ns, 2) float array
        Contour-ordered surface points.  **Pass the float32 array that is
        actually stored**, not the float64 original: the models recompute these
        quantities from the cached float32 positions, and on a near-uniform
        contour the float32 rounding is enough to flip kNN tie-breaks and shift
        the SDF in the last bit.  Feeding float64 here yields a cache that is
        *almost* but not exactly what the model would have built.
    k : int
        kNN degree; must match ``src/models/gnn.py``'s ``k``.
    bbox, res : sequences
        M2's latent grid, ``(xmin, xmax, ymin, ymax)`` and ``(n_y, n_x)``.

    Returns
    -------
    dict with ``grid_sdf`` (n_y, n_x) float32, ``edge_index`` (2, E) int32 and
    ``curvature`` (Ns,) float32.

    Notes
    -----
    **Grid orientation is a trap.**  ``src/geometry/sdf.make_grid(bbox, (nx, ny))``
    returns ``grid[ix, iy] = (x_ix, y_iy)`` -- x on the *first* axis -- whereas
    M2's ``make_latent_grid`` lays nodes out row-major as ``iy * nx + ix`` so
    that a reshape to ``(n_y, n_x)`` is the ``(H, W)`` image its FNO expects.
    The two differ by a transpose; feeding the untransposed array to M2 is a
    silent 0.38-magnitude error on a unit-chord airfoil rather than a crash.
    We therefore transpose here, and ``tests/test_loader.py`` asserts the
    cached array is bit-identical to M2's own on-the-fly result.
    """
    import torch  # noqa: PLC0415  (heavy; only the builder needs it)

    from src.geometry.quadrature import curvature as _curvature  # noqa: PLC0415
    from src.geometry.sdf import make_grid, sdf_on_grid  # noqa: PLC0415
    from src.models.common import knn_edges  # noqa: PLC0415

    # Mirror the model path exactly: it receives float32 and widens to float64.
    surf_pos = np.ascontiguousarray(
        np.asarray(surf_pos, dtype=np.float32), dtype=np.float64
    )
    n_surf = surf_pos.shape[0]

    # -- grid_sdf: A2's SDF on M2's grid, in M2's (n_y, n_x) layout ----------
    n_y, n_x = int(res[0]), int(res[1])
    grid = make_grid(bbox, (n_x, n_y))                 # (n_x, n_y, 2)
    grid_sdf = np.asarray(sdf_on_grid(surf_pos, grid)).T   # -> (n_y, n_x)
    if grid_sdf.shape != (n_y, n_x):  # pragma: no cover - guarded by make_grid
        raise ValueError(
            f"grid_sdf has shape {grid_sdf.shape}, expected {(n_y, n_x)}"
        )

    # -- edge_index: exactly what src.models.common.knn_edges would build ----
    edge = knn_edges(torch.from_numpy(surf_pos), k=int(k))
    edge_np = edge.detach().cpu().numpy()
    if edge_np.ndim != 2 or edge_np.shape[0] != 2:  # pragma: no cover
        raise ValueError(f"edge_index must be (2, E), got {edge_np.shape}")
    if edge_np.size and int(edge_np.max()) >= n_surf:  # pragma: no cover
        raise ValueError("edge_index references a point outside the contour")

    # -- curvature: A2's signed Menger curvature -----------------------------
    curv = np.asarray(_curvature(surf_pos), dtype=np.float64)
    if curv.shape != (n_surf,):
        # quadrature.dedup_closed_contour drops a duplicated closing point; our
        # contour walk should never produce one, so this means coincident nodes.
        raise ValueError(
            f"curvature returned {curv.shape}, expected {(n_surf,)} -- the "
            "contour probably has coincident points"
        )

    return {
        "grid_sdf": np.ascontiguousarray(grid_sdf, dtype=np.float32),
        "edge_index": np.ascontiguousarray(edge_np, dtype=np.int32),
        "curvature": np.ascontiguousarray(curv, dtype=np.float32),
    }


def _stable_seed(name: str) -> int:
    """Platform-independent 32-bit seed from a simulation name."""
    digest = hashlib.blake2s(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2 ** 32)


# --------------------------------------------------------------------------
# per-simulation conversion
# --------------------------------------------------------------------------
def build_one(
    sim_name: str,
    raw_root: Path,
    max_vol_points: int = MAX_VOL_POINTS,
    strict: bool = True,
    geometry: bool = True,
    knn_k: int = KNN_K,
) -> tuple[dict[str, np.ndarray], dict]:
    """Convert one raw simulation into the frozen cache arrays.

    Parameters
    ----------
    sim_name : str
        Simulation directory name, e.g.
        ``"airFoil2D_SST_43.597_5.932_3.551_3.1_1.0_18.252"``.
    raw_root : Path
        Directory containing ``<sim_name>/`` and ``manifest.json``.
    max_vol_points : int
        Volume subsample ceiling.
    strict : bool
        Raise instead of warn when a geometric sanity check fails.
    geometry : bool
        Also precompute the D-017 static geometry (``grid_sdf``,
        ``edge_index``, ``curvature``).
    knn_k : int
        kNN degree for ``edge_index``.

    Returns
    -------
    (arrays, meta) : tuple[dict[str, np.ndarray], dict]
        ``arrays`` is exactly what goes into the ``.npz``; ``meta`` is the
        JSON-serialisable manifest row.
    """
    import airfrans as af  # noqa: PLC0415  (heavy: pyvista)

    sim = af.Simulation(root=str(raw_root), name=sim_name)
    parsed = parse_sim_name(sim_name)

    # ---- surface: airfoil-patch order --------------------------------------
    airfoil_pos = np.asarray(sim.airfoil_position, dtype=np.float64)  # (Na, 2)
    n_surf = airfoil_pos.shape[0]

    # AirfRANS normals point INTO the airfoil; CONTEXT.md wants outward.
    normals_inward = np.asarray(sim.airfoil_normals, dtype=np.float64)
    outward = -normals_inward
    nmag = np.linalg.norm(outward, axis=1, keepdims=True)
    nmag[nmag == 0.0] = 1.0
    outward = outward / nmag

    # Pressure: reorganise the internal-mesh surface nodes into patch order,
    # exactly as Simulation.force() does, so cl/cd stay reproducible from cache.
    surf_mask = np.asarray(sim.surface, dtype=bool)
    surf_p = af.reorganize(
        sim.position[surf_mask], airfoil_pos, sim.pressure[surf_mask]
    ).reshape(-1).astype(np.float64)

    # Wall shear stress: kinematic viscous traction with the OUTWARD normal
    # (the package folds the sign in), already in patch order.
    surf_tau = np.asarray(
        sim.wallshearstress(over_airfoil=True, reference=True), dtype=np.float64
    )[:, :2]

    if surf_p.shape[0] != n_surf or surf_tau.shape[0] != n_surf:
        raise ValueError(
            f"{sim_name}: surface field length mismatch "
            f"(pos {n_surf}, p {surf_p.shape[0]}, tau {surf_tau.shape[0]})"
        )

    # ---- contour ordering + quadrature -------------------------------------
    edges = _edges_from_polydata_lines(np.asarray(sim.airfoil.lines))
    order = contour_order(airfoil_pos, edges)
    degrees = np.bincount(edges.ravel(), minlength=n_surf)
    closed = bool(np.all(degrees == 2))

    surf_pos = airfoil_pos[order]
    surf_normal = outward[order]
    surf_p = surf_p[order]
    surf_tau = surf_tau[order]
    surf_ds = quadrature_weights(surf_pos, closed=closed)
    perimeter = float(surf_ds.sum())

    warnings: list[str] = []

    # Check 1: the stored normals must agree with the CCW-contour geometry.
    n_geo = _geometric_outward_normals(surf_pos, closed=closed)
    align = float(np.mean(np.sum(surf_normal * n_geo, axis=1)))
    if align < 0.9:
        warnings.append(
            f"outward-normal/contour alignment is {align:.3f} (expected ~1); "
            "normal orientation is suspect"
        )

    # Check 2: closed-surface identity  sum(n ds) ~ 0.
    closure = float(
        np.linalg.norm((surf_normal * surf_ds[:, None]).sum(axis=0)) / perimeter
    )
    if closure > 5e-3:
        warnings.append(
            f"sum(n*ds)/perimeter = {closure:.2e}; surface normals do not close"
        )

    if warnings and strict:
        raise ValueError(f"{sim_name}: " + "; ".join(warnings))

    # ---- volume ------------------------------------------------------------
    vol_pos_all = np.asarray(sim.position, dtype=np.float64)
    n_vol_all = vol_pos_all.shape[0]
    if n_vol_all > max_vol_points:
        rng = np.random.default_rng(_stable_seed(sim_name))
        idx = np.sort(
            rng.choice(n_vol_all, size=max_vol_points, replace=False)
        )
    else:
        idx = np.arange(n_vol_all, dtype=np.int64)

    vol_pos = vol_pos_all[idx]
    vol_u = np.asarray(sim.velocity, dtype=np.float64)[idx]
    vol_p = np.asarray(sim.pressure, dtype=np.float64).reshape(-1)[idx]
    vol_nut = np.asarray(sim.nu_t, dtype=np.float64).reshape(-1)[idx]
    vol_sdf = np.asarray(sim.sdf, dtype=np.float64).reshape(-1)[idx]
    vol_is_surf = surf_mask[idx]

    # ---- scalars -----------------------------------------------------------
    u_inf_mag = float(np.asarray(sim.inlet_velocity))
    aoa_rad = float(np.asarray(sim.angle_of_attack))
    cond = np.array(
        [u_inf_mag * np.cos(aoa_rad), u_inf_mag * np.sin(aoa_rad)],
        dtype=np.float64,
    )

    (cd, cdp, cdv), (cl, clp, clv) = sim.force_coefficient(
        compressible=False, reference=True
    )

    rho = float(np.asarray(sim.RHO))
    nu = float(np.asarray(sim.NU))

    arrays: dict[str, np.ndarray] = {
        "surf_pos": surf_pos.astype(np.float32),
        "surf_normal": surf_normal.astype(np.float32),
        "surf_ds": surf_ds.astype(np.float32),
        "surf_p": surf_p.astype(np.float32),
        "surf_tau": surf_tau.astype(np.float32),
        "vol_pos": vol_pos.astype(np.float32),
        "vol_u": vol_u.astype(np.float32),
        "vol_p": vol_p.astype(np.float32),
        "vol_nut": vol_nut.astype(np.float32),
        "vol_sdf": vol_sdf.astype(np.float32),
        "vol_is_surf": vol_is_surf.astype(np.uint8),
        "cond": cond.astype(np.float32),
        # scalars, stored as 0-d float32 so np.load gives plain arrays
        "re": np.float32(parsed.re),
        "aoa_deg": np.float32(parsed.aoa_deg),
        "u_inf_mag": np.float32(u_inf_mag),
        "cl_true": np.float32(cl),
        "cd_true": np.float32(cd),
        "cl_p": np.float32(clp),
        "cl_v": np.float32(clv),
        "cd_p": np.float32(cdp),
        "cd_v": np.float32(cdv),
        "rho": np.float32(rho),
        "nu": np.float32(nu),
        "perimeter": np.float32(perimeter),
        "n_surf": np.int64(n_surf),
        "n_vol": np.int64(vol_pos.shape[0]),
        "n_vol_full": np.int64(n_vol_all),
        "cache_version": np.int64(CACHE_VERSION),
    }
    # D-017: static geometry the models would otherwise rebuild every step.
    if geometry:
        try:
            arrays.update(
                geometry_extras(arrays["surf_pos"], k=knn_k)
            )
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise
            warnings.append(f"geometry extras skipped: {exc}")

    for key, arr in arrays.items():
        if isinstance(arr, np.ndarray) and arr.ndim > 0:
            arrays[key] = np.ascontiguousarray(arr)

    meta = {
        "name": sim_name,
        "n_surf": int(n_surf),
        "n_vol": int(vol_pos.shape[0]),
        "n_vol_full": int(n_vol_all),
        "re": float(parsed.re),
        "aoa_deg": float(parsed.aoa_deg),
        "u_inf_mag": float(u_inf_mag),
        "naca_digits": int(parsed.n_digits),
        "naca_params": list(parsed.naca_params),
        "cl_true": float(cl),
        "cd_true": float(cd),
        "cl_p": float(clp),
        "cl_v": float(clv),
        "cd_p": float(cdp),
        "cd_v": float(cdv),
        "rho": rho,
        "nu": nu,
        "perimeter": perimeter,
        "closed_contour": closed,
        "normal_alignment": align,
        "normal_closure": closure,
        "cache_version": CACHE_VERSION,
        "has_geometry": all(
            key in arrays for key in ("grid_sdf", "edge_index", "curvature")
        ),
        "n_edges": int(arrays["edge_index"].shape[1])
        if "edge_index" in arrays else 0,
    }
    if warnings:
        meta["warnings"] = warnings
    return arrays, meta


def _npz_is_current(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as data:
            return int(data["cache_version"]) == CACHE_VERSION
    except Exception:
        return False


def _meta_from_npz(path: Path, sim_name: str) -> dict:
    with np.load(path) as d:
        parsed = parse_sim_name(sim_name)
        return {
            "name": sim_name,
            "n_surf": int(d["n_surf"]),
            "n_vol": int(d["n_vol"]),
            "n_vol_full": int(d["n_vol_full"]),
            "re": float(d["re"]),
            "aoa_deg": float(d["aoa_deg"]),
            "u_inf_mag": float(d["u_inf_mag"]),
            "naca_digits": int(parsed.n_digits),
            "naca_params": list(parsed.naca_params),
            "cl_true": float(d["cl_true"]),
            "cd_true": float(d["cd_true"]),
            "cl_p": float(d["cl_p"]),
            "cl_v": float(d["cl_v"]),
            "cd_p": float(d["cd_p"]),
            "cd_v": float(d["cd_v"]),
            "rho": float(d["rho"]),
            "nu": float(d["nu"]),
            "perimeter": float(d["perimeter"]),
            "cache_version": int(d["cache_version"]),
            "has_geometry": all(
                key in d for key in ("grid_sdf", "edge_index", "curvature")
            ),
            "n_edges": int(d["edge_index"].shape[1]) if "edge_index" in d else 0,
        }


# --------------------------------------------------------------------------
# normalisation statistics
# --------------------------------------------------------------------------
class _Accumulator:
    """Streaming mean/std over concatenated arrays of fixed width."""

    def __init__(self, width: int) -> None:
        self.width = width
        self.count = 0
        self.total = np.zeros(width, dtype=np.float64)
        self.total_sq = np.zeros(width, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.width)
        self.count += x.shape[0]
        self.total += x.sum(axis=0)
        self.total_sq += np.square(x).sum(axis=0)

    def finalize(self) -> dict:
        if self.count == 0:
            return {
                "mean": [0.0] * self.width,
                "std": [1.0] * self.width,
                "count": 0,
            }
        mean = self.total / self.count
        var = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)
        std = np.sqrt(var)
        std = np.where(std < 1e-8, 1.0, std)
        return {
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
            "count": int(self.count),
        }


def compute_norm_stats(
    sim_names: Sequence[str], processed_dir: Path, split_name: str = "full"
) -> dict:
    """Compute per-field mean/std over ``sim_names`` (the ``full`` train list).

    CONTEXT.md section 4: stats come from the train split of ``full`` **only**;
    ``cal`` and ``test`` never touch normalisation.
    """
    processed_dir = Path(processed_dir)
    accs = {
        k: _Accumulator(w) for k, w in {**_POINT_FIELDS, **_SIM_FIELDS}.items()
    }
    used = 0
    for name in sim_names:
        path = processed_dir / f"{name}.npz"
        if not path.is_file():
            continue
        with np.load(path) as d:
            for key in _POINT_FIELDS:
                if key in d:          # optional keys may be absent
                    accs[key].update(d[key])
            for key in _SIM_FIELDS:
                accs[key].update(np.atleast_1d(d[key]))
        used += 1
    if used == 0:
        raise RuntimeError(
            "no cached sims from the requested split were found; build the "
            "cache for the train split before computing norm stats"
        )
    return {
        "split": split_name,
        "subset": "train",
        "n_sims": used,
        "cache_version": CACHE_VERSION,
        "fields": {k: a.finalize() for k, a in accs.items()},
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def _select_sims(
    args: argparse.Namespace, raw_manifest: dict[str, list[str]]
) -> list[str]:
    if args.sims:
        return list(args.sims)
    if args.split_file is not None:
        payload = load_split(args.split_file)
        subsets = args.subset or ["train", "cal", "test"]
        names: list[str] = []
        for s in subsets:
            names.extend(payload[s])
        return sorted(set(names))
    return all_sim_names(raw_manifest)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/Dataset"),
        help="unzipped AirfRANS directory holding manifest.json "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/airfrans"),
        help="cache destination (default: %(default)s)",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="split manifests directory (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="only convert the first N simulations (smoke test)",
    )
    parser.add_argument(
        "--sims", nargs="+", default=None, help="explicit simulation names"
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="restrict to the sims named in this split manifest",
    )
    parser.add_argument(
        "--subset",
        nargs="+",
        choices=["train", "cal", "test"],
        default=None,
        help="with --split-file, which subsets to build (default: all)",
    )
    parser.add_argument(
        "--max-vol-points",
        type=int,
        default=MAX_VOL_POINTS,
        help="volume subsample ceiling (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild sims whose .npz already exists",
    )
    parser.add_argument(
        "--no-geometry",
        action="store_true",
        help="skip the D-017 precomputed static geometry (grid_sdf, "
        "edge_index, curvature)",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=KNN_K,
        help="kNN degree for the cached edge_index; must match the GNN's k "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="downgrade geometric sanity-check failures to warnings",
    )
    parser.add_argument(
        "--skip-splits",
        action="store_true",
        help="do not (re)write data/splits/*.json first",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="do not write norm_stats.json (use for partial builds)",
    )
    parser.add_argument(
        "--stats-split",
        default="full",
        help="split whose TRAIN subset defines the norm stats "
        "(default: %(default)s)",
    )
    args = parser.parse_args(argv)

    raw_root: Path = args.raw_root
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_manifest = read_raw_manifest(raw_root)

    if not args.skip_splits:
        written = write_all_splits(raw_root, args.splits_dir, seed=0)
        print(f"splits: wrote {len(written)} manifests to {args.splits_dir}")

    sims = _select_sims(args, raw_manifest)
    if args.limit is not None:
        sims = sims[: args.limit]
    print(f"cache: {len(sims)} simulation(s) targeted -> {out_dir}")

    manifest_path = out_dir / "manifest.json"
    rows: dict[str, dict] = {}
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                prev = json.load(fh)
            for row in prev.get("sims", []):
                if row.get("cache_version") == CACHE_VERSION:
                    rows[row["name"]] = row
        except Exception:
            rows = {}

    built = skipped = failed = 0
    failures: list[tuple[str, str]] = []
    t0 = time.time()
    for i, name in enumerate(sims, 1):
        npz_path = out_dir / f"{name}.npz"
        if not args.force and _npz_is_current(npz_path):
            if name not in rows:
                try:
                    rows[name] = _meta_from_npz(npz_path, name)
                except Exception:
                    pass
            skipped += 1
            continue
        try:
            arrays, meta = build_one(
                name,
                raw_root,
                max_vol_points=args.max_vol_points,
                strict=not args.no_strict,
                geometry=not args.no_geometry,
                knn_k=args.knn_k,
            )
        except Exception as exc:  # noqa: BLE001 - one bad sim must not stop us
            failed += 1
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  [{i}/{len(sims)}] FAIL {name}: {exc}", file=sys.stderr)
            if failed <= 3:
                traceback.print_exc(file=sys.stderr)
            continue
        # NB: np.savez_compressed appends '.npz' to a *path* whose name does
        # not already end in it, so hand it an open handle instead and keep the
        # write atomic via replace().
        tmp = npz_path.with_name(npz_path.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **arrays)
        tmp.replace(npz_path)
        rows[name] = meta
        built += 1
        if built % 25 == 0 or i == len(sims):
            rate = (time.time() - t0) / max(built, 1)
            print(
                f"  [{i}/{len(sims)}] built={built} skipped={skipped} "
                f"failed={failed}  ({rate:.2f}s/sim)"
            )

    payload = {
        "cache_version": CACHE_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "raw_root": str(raw_root),
        "max_vol_points": int(args.max_vol_points),
        "n_sims": len(rows),
        "sims": [rows[k] for k in sorted(rows)],
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"cache: manifest -> {manifest_path} ({len(rows)} sims)")

    if failures:
        print(f"cache: {len(failures)} failure(s):", file=sys.stderr)
        for name, msg in failures[:10]:
            print(f"  {name}: {msg}", file=sys.stderr)

    if not args.skip_stats:
        split_path = args.splits_dir / f"{args.stats_split}.json"
        if not split_path.is_file():
            print(
                f"stats: {split_path} missing -- skipping norm_stats.json",
                file=sys.stderr,
            )
        else:
            train = load_split(split_path)["train"]
            try:
                stats = compute_norm_stats(
                    train, out_dir, split_name=args.stats_split
                )
            except RuntimeError as exc:
                print(f"stats: {exc}", file=sys.stderr)
            else:
                stats_path = out_dir / "norm_stats.json"
                with stats_path.open("w", encoding="utf-8") as fh:
                    json.dump(stats, fh, indent=2)
                    fh.write("\n")
                print(
                    f"stats: norm_stats.json from {stats['n_sims']} "
                    f"'{args.stats_split}' train sims -> {stats_path}"
                )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
