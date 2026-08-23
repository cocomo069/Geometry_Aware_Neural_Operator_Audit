"""Normal-vector helpers for closed 2-D surface contours.

The geometric construction of outward normals from an ordered contour lives
in :mod:`src.geometry.quadrature` (``contour_normals``); this module
re-exports it and adds helpers for validating / fixing normals that come
from a dataset (AirfRANS ships surface normals whose orientation convention
must be verified, CONTEXT.md section 2).

"Outward" always means pointing away from the enclosed body, into the fluid.
Vectorized numpy.
"""
from __future__ import annotations

import numpy as np

from src.geometry.quadrature import contour_normals, dedup_closed_contour

__all__ = [
    "contour_normals",
    "normalize_rows",
    "normals_angle_error",
    "outward_fraction",
    "orient_normals_outward",
]


def normalize_rows(vec: np.ndarray) -> np.ndarray:
    """Return unit rows of a (N, 2) array; asserts no near-zero rows."""
    vec = np.asarray(vec, dtype=np.float64)
    assert vec.ndim == 2 and vec.shape[1] == 2, f"expected (N, 2), got {vec.shape}"
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    assert np.all(norm > 1e-30), "zero-length normal encountered"
    return vec / norm


def normals_angle_error(n_est: np.ndarray, n_ref: np.ndarray) -> np.ndarray:
    """Per-point angle (radians, in [0, pi]) between two unit-normal fields."""
    n_est = normalize_rows(n_est)
    n_ref = normalize_rows(n_ref)
    assert n_est.shape == n_ref.shape, "normal arrays must have equal shapes"
    dot = np.clip((n_est * n_ref).sum(axis=1), -1.0, 1.0)
    return np.arccos(dot)


def outward_fraction(pos: np.ndarray, normals: np.ndarray) -> float:
    """Fraction of the given normals pointing outward (into the fluid).

    ``pos`` must be the ordered closed contour matching ``normals`` row by
    row (after dropping a duplicated closing point, if any).
    """
    pos = dedup_closed_contour(pos)
    normals = np.asarray(normals, dtype=np.float64)
    if normals.shape[0] == pos.shape[0] + 1 and np.allclose(normals[0], normals[-1]):
        normals = normals[:-1]
    assert normals.shape == pos.shape, (
        f"normals {normals.shape} do not match contour points {pos.shape}")
    ref = contour_normals(pos)
    dot = (normalize_rows(normals) * ref).sum(axis=1)
    return float(np.mean(dot > 0.0))


def orient_normals_outward(pos: np.ndarray, normals: np.ndarray):
    """Flip any inward-pointing rows of ``normals`` to point outward.

    Returns
    -------
    (fixed_normals (N, 2) unit float64, n_flipped int)
    """
    pos = dedup_closed_contour(pos)
    normals = np.asarray(normals, dtype=np.float64)
    if normals.shape[0] == pos.shape[0] + 1 and np.allclose(normals[0], normals[-1]):
        normals = normals[:-1]
    assert normals.shape == pos.shape, (
        f"normals {normals.shape} do not match contour points {pos.shape}")
    unit = normalize_rows(normals)
    ref = contour_normals(pos)
    flip = (unit * ref).sum(axis=1) < 0.0
    out = unit.copy()
    out[flip] *= -1.0
    return out, int(np.count_nonzero(flip))
