"""Contour quadrature for closed 2-D surface curves.

Every function takes an ordered closed contour ``pos`` of shape (N, 2):
point ``i`` is connected to point ``i+1`` and point ``N-1`` wraps back to
point ``0``.  A duplicated closing point (``pos[-1] == pos[0]``, as in an
airfoil contour stored TE -> around -> TE) is detected and dropped, so all
outputs have one entry per *distinct* contour point.

Orientation is arbitrary (CW or CCW): the outward direction is fixed via the
signed (shoelace) area, so normals always point away from the enclosed body,
i.e. into the fluid.

Vectorized numpy, float64 internally; callers convert to torch / float32.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "dedup_closed_contour",
    "signed_area",
    "facet_lengths",
    "point_weights",
    "contour_normals",
    "curvature",
    "contour_quadrature",
]

_TINY = 1e-30


def dedup_closed_contour(pos: np.ndarray) -> np.ndarray:
    """Validate a closed contour and drop a duplicated closing point.

    Parameters
    ----------
    pos : (N, 2) array of ordered contour points.

    Returns
    -------
    (M, 2) float64 array, M = N or N-1, with pos[-1] != pos[0].
    """
    pos = np.asarray(pos, dtype=np.float64)
    assert pos.ndim == 2 and pos.shape[1] == 2, f"pos must be (N, 2), got {pos.shape}"
    if pos.shape[0] >= 2 and np.allclose(pos[0], pos[-1], rtol=0.0, atol=1e-12):
        pos = pos[:-1]
    assert pos.shape[0] >= 3, "a closed contour needs at least 3 distinct points"
    return pos


def signed_area(pos: np.ndarray) -> float:
    """Shoelace signed area of the closed polygon. > 0 for counterclockwise."""
    pos = dedup_closed_contour(pos)
    x, y = pos[:, 0], pos[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    return float(0.5 * np.sum(x * yn - xn * y))


def facet_lengths(pos: np.ndarray) -> np.ndarray:
    """Length of facet i, the segment from point i to point i+1 (wrapping).

    Returns (N,) float64; sum equals the polygon perimeter.
    """
    pos = dedup_closed_contour(pos)
    seg = np.roll(pos, -1, axis=0) - pos
    return np.linalg.norm(seg, axis=1)


def point_weights(pos: np.ndarray) -> np.ndarray:
    """Per-point quadrature weight ds: half-sum of the two adjacent facets.

    ds_i = 0.5 * (|facet_{i-1}| + |facet_i|); sum(ds) equals the perimeter
    exactly. This matches the cache key ``surf_ds`` (CONTEXT.md section 4).

    Returns (N,) float64.
    """
    ell = facet_lengths(pos)
    return 0.5 * (ell + np.roll(ell, 1))


def contour_normals(pos: np.ndarray) -> np.ndarray:
    """Outward unit normals (pointing away from the enclosed body).

    The tangent at point i is the central difference pos[i+1] - pos[i-1]
    (wrapping). For a counterclockwise contour the outward normal is the
    tangent rotated -90 degrees, (t_y, -t_x); for clockwise, the opposite.
    Orientation is decided by the signed area, so the result is
    orientation-independent.

    Returns (N, 2) float64 unit vectors.
    """
    pos = dedup_closed_contour(pos)
    t = np.roll(pos, -1, axis=0) - np.roll(pos, 1, axis=0)
    norm = np.linalg.norm(t, axis=1, keepdims=True)
    assert np.all(norm > _TINY), "degenerate contour: coincident neighbor points"
    t = t / norm
    orient = 1.0 if signed_area(pos) > 0.0 else -1.0
    return orient * np.stack([t[:, 1], -t[:, 0]], axis=1)


def curvature(pos: np.ndarray) -> np.ndarray:
    """Signed discrete (Menger) curvature per point.

    kappa_i = 2 * cross(p_i - p_{i-1}, p_{i+1} - p_i) / (|a| |b| |c|), the
    inverse circumradius of each consecutive point triple, with the sign
    normalized by contour orientation so that kappa > 0 on locally convex
    parts (center of curvature inside the body) regardless of CW/CCW input.
    A circle of radius R gives exactly +1/R at every point.

    Returns (N,) float64.
    """
    pos = dedup_closed_contour(pos)
    a = pos - np.roll(pos, 1, axis=0)      # p_{i-1} -> p_i
    b = np.roll(pos, -1, axis=0) - pos     # p_i -> p_{i+1}
    c = np.roll(pos, -1, axis=0) - np.roll(pos, 1, axis=0)  # p_{i-1} -> p_{i+1}
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    denom = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
             * np.linalg.norm(c, axis=1))
    orient = 1.0 if signed_area(pos) > 0.0 else -1.0
    return orient * 2.0 * cross / np.maximum(denom, _TINY)


def contour_quadrature(pos: np.ndarray) -> dict:
    """All contour quantities at once.

    Returns dict with keys: ``pos`` (deduped, (N,2)), ``ds`` (N,),
    ``normal`` (N,2) outward, ``curvature`` (N,), ``perimeter`` float,
    ``signed_area`` float.
    """
    pos = dedup_closed_contour(pos)
    ds = point_weights(pos)
    return {
        "pos": pos,
        "ds": ds,
        "normal": contour_normals(pos),
        "curvature": curvature(pos),
        "perimeter": float(ds.sum()),
        "signed_area": signed_area(pos),
    }
