"""Signed distance and occupancy of grid points w.r.t. a closed 2-D contour.

Convention (CONTEXT.md section 4 / M2 model): SDF is **negative inside the
body** and positive in the fluid. Occupancy is 1.0 inside the body, 0.0 in
the fluid.

Distance is the exact point-to-segment distance to the polygon boundary
(not merely nearest-vertex distance). For large contours a scipy cKDTree on
segment midpoints prefilters candidate segments; otherwise all segments are
checked, chunked to bound memory. The inside test is a vectorized
crossing-number (ray casting) test, dependency-free.

Vectorized numpy; callers convert to torch.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from src.geometry.quadrature import dedup_closed_contour

__all__ = [
    "make_grid",
    "points_in_polygon",
    "distance_to_contour",
    "sdf_on_grid",
    "occupancy_on_grid",
]

# max (points x segments) pairs handled per chunk in the exact sweep
_CHUNK_PAIRS = 4_000_000
# above this many segments, use the kd-tree prefilter instead of a full sweep
_KDTREE_SEGMENT_THRESHOLD = 4096


def make_grid(bbox, shape) -> np.ndarray:
    """Regular grid of query points.

    Parameters
    ----------
    bbox : (xmin, xmax, ymin, ymax)
    shape : (nx, ny) number of points along x and y.

    Returns
    -------
    (nx, ny, 2) float64 array; grid[i, j] = (x_i, y_j) with x_0 = xmin,
    x_{nx-1} = xmax (endpoints included), i.e. 'ij' indexing, x first.
    """
    xmin, xmax, ymin, ymax = bbox
    nx, ny = shape
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx, gy], axis=-1)


def _as_points(grid_xy):
    """Accept (..., 2) query points; return flat (M, 2) + original shape."""
    grid_xy = np.asarray(grid_xy, dtype=np.float64)
    assert grid_xy.shape[-1] == 2, f"query points must be (..., 2), got {grid_xy.shape}"
    lead_shape = grid_xy.shape[:-1]
    return grid_xy.reshape(-1, 2), lead_shape


def points_in_polygon(poly: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Crossing-number inside test for a simple closed polygon.

    Parameters
    ----------
    poly : (N, 2) ordered closed contour (either orientation).
    points : (M, 2) query points.

    Returns
    -------
    (M,) bool, True strictly inside (points exactly on the boundary are
    orientation/rounding dependent, as with any ray-casting test).
    """
    poly = dedup_closed_contour(poly)
    points = np.asarray(points, dtype=np.float64)
    assert points.ndim == 2 and points.shape[1] == 2
    a = poly
    b = np.roll(poly, -1, axis=0)
    x1, y1 = a[:, 0], a[:, 1]
    x2, y2 = b[:, 0], b[:, 1]
    dy = y2 - y1
    # guard horizontal edges (cond is False there, so the value is unused)
    dy_safe = np.where(np.abs(dy) < 1e-300, 1.0, dy)

    m = points.shape[0]
    inside = np.zeros(m, dtype=bool)
    chunk = max(1, _CHUNK_PAIRS // max(1, poly.shape[0]))
    for s in range(0, m, chunk):
        px = points[s:s + chunk, 0][:, None]
        py = points[s:s + chunk, 1][:, None]
        cond = (y1[None, :] > py) != (y2[None, :] > py)
        xint = x1[None, :] + (py - y1[None, :]) * (x2 - x1)[None, :] / dy_safe[None, :]
        crossings = np.count_nonzero(cond & (px < xint), axis=1)
        inside[s:s + chunk] = (crossings % 2) == 1
    return inside


def _exact_segment_distance(points, a, b):
    """Min distance from each point to any of the segments [a_j, b_j].

    points (m, 2), a/b (S, 2) -> (m,) float64. Fully vectorized, O(m*S).
    """
    d = b - a                                   # (S, 2)
    l2 = np.maximum((d ** 2).sum(axis=1), 1e-300)
    ap = points[:, None, :] - a[None, :, :]     # (m, S, 2)
    t = np.clip((ap * d[None, :, :]).sum(axis=2) / l2[None, :], 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * d[None, :, :]
    dist = np.linalg.norm(points[:, None, :] - proj, axis=2)
    return dist.min(axis=1)


def distance_to_contour(poly: np.ndarray, points: np.ndarray, k: int = 16) -> np.ndarray:
    """Unsigned exact distance from points to the closed polygon boundary.

    For few segments: exact sweep over all segments (chunked over points).
    For many segments: cKDTree on segment midpoints selects the k nearest
    candidate segments per point, then the exact point-segment distance is
    taken over the candidates (assumes reasonably uniform contour sampling).
    """
    poly = dedup_closed_contour(poly)
    points = np.asarray(points, dtype=np.float64)
    assert points.ndim == 2 and points.shape[1] == 2
    a = poly
    b = np.roll(poly, -1, axis=0)
    n_seg = a.shape[0]
    m = points.shape[0]
    out = np.empty(m, dtype=np.float64)

    if n_seg <= _KDTREE_SEGMENT_THRESHOLD:
        chunk = max(1, _CHUNK_PAIRS // n_seg)
        for s in range(0, m, chunk):
            out[s:s + chunk] = _exact_segment_distance(points[s:s + chunk], a, b)
        return out

    mid = 0.5 * (a + b)
    tree = cKDTree(mid)
    kq = min(k, n_seg)
    _, cand = tree.query(points, k=kq)          # (m, kq) segment indices
    cand = np.atleast_2d(cand)
    chunk = max(1, _CHUNK_PAIRS // kq)
    for s in range(0, m, chunk):
        c = cand[s:s + chunk]                   # (mc, kq)
        pc = points[s:s + chunk]
        d = b[c] - a[c]                         # (mc, kq, 2)
        l2 = np.maximum((d ** 2).sum(axis=2), 1e-300)
        ap = pc[:, None, :] - a[c]
        t = np.clip((ap * d).sum(axis=2) / l2, 0.0, 1.0)
        proj = a[c] + t[:, :, None] * d
        dist = np.linalg.norm(pc[:, None, :] - proj, axis=2)
        out[s:s + chunk] = dist.min(axis=1)
    return out


def sdf_on_grid(surf_pos: np.ndarray, grid_xy: np.ndarray) -> np.ndarray:
    """Signed distance of query points to the closed contour ``surf_pos``.

    Parameters
    ----------
    surf_pos : (N, 2) ordered closed contour (airfoil surface).
    grid_xy : (..., 2) query points, e.g. a (H, W, 2) regular grid from
        ``make_grid`` or a flat (M, 2) list.

    Returns
    -------
    (...,) float64 signed distance, **negative inside the body**, positive
    in the fluid. Shape matches ``grid_xy`` without the last axis.
    """
    pts, lead_shape = _as_points(grid_xy)
    dist = distance_to_contour(surf_pos, pts)
    inside = points_in_polygon(surf_pos, pts)
    sd = np.where(inside, -dist, dist)
    return sd.reshape(lead_shape)


def occupancy_on_grid(surf_pos: np.ndarray, grid_xy: np.ndarray) -> np.ndarray:
    """Body occupancy of query points: 1.0 inside the body, 0.0 in the fluid.

    Same shapes as ``sdf_on_grid``.
    """
    pts, lead_shape = _as_points(grid_xy)
    occ = points_in_polygon(surf_pos, pts).astype(np.float64)
    return occ.reshape(lead_shape)
