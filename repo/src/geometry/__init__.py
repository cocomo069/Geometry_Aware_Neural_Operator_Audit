"""Geometry utilities: contour quadrature, normals, SDF, symmetry ops (A2)."""
from src.geometry.quadrature import (
    contour_quadrature,
    contour_normals,
    curvature,
    facet_lengths,
    point_weights,
    signed_area,
)
from src.geometry.sdf import make_grid, occupancy_on_grid, sdf_on_grid
