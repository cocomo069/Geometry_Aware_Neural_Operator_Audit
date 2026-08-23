"""CPU tests for src/geometry (quadrature, normals, sdf, symmetry) and
src/physics/residuals. Synthetic geometry only, no dataset required."""
import numpy as np
import torch

from src.geometry.normals import (
    normals_angle_error,
    orient_normals_outward,
    outward_fraction,
)
from src.geometry.quadrature import (
    contour_normals,
    contour_quadrature,
    curvature,
    facet_lengths,
    point_weights,
    signed_area,
)
from src.geometry.sdf import make_grid, occupancy_on_grid, points_in_polygon, sdf_on_grid
from src.geometry.symmetry import (
    antisymmetry_check,
    reflect_prediction,
    reflect_x_batch,
    symmetry_residual,
)
from src.physics.residuals import (
    divergence_residual,
    knn_ls_weights,
    noslip_residual,
    pointwise_divergence,
)


def circle(n=512, r=1.0, ccw=True, close=False):
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    if not ccw:
        th = th[::-1]
    pos = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    if close:
        pos = np.vstack([pos, pos[:1]])
    return pos


# ---------------------------------------------------------------- quadrature

def test_circle_normals_outward_within_1e6_angle():
    for ccw in (True, False):
        pos = circle(512, ccw=ccw)
        n_est = contour_normals(pos)
        n_ref = pos / np.linalg.norm(pos, axis=1, keepdims=True)  # radial = outward
        ang = normals_angle_error(n_est, n_ref)
        assert ang.max() < 1e-6, f"max normal angle error {ang.max():.3e} (ccw={ccw})"


def test_circle_ds_sums_to_perimeter():
    pos = circle(512)
    ds = point_weights(pos)
    assert ds.shape == (512,)
    assert abs(ds.sum() - 2.0 * np.pi) < 1e-3
    # per-point weights redistribute facet lengths without changing the total
    assert abs(ds.sum() - facet_lengths(pos).sum()) < 1e-12


def test_duplicated_closing_point_is_dropped():
    pos = circle(256, close=True)
    assert point_weights(pos).shape == (256,)
    assert contour_normals(pos).shape == (256, 2)


def test_signed_area_orientation():
    assert signed_area(circle(256, ccw=True)) > 0
    assert signed_area(circle(256, ccw=False)) < 0
    assert abs(abs(signed_area(circle(2048))) - np.pi) < 1e-3


def test_circle_curvature():
    for r in (1.0, 2.5):
        for ccw in (True, False):
            k = curvature(circle(512, r=r, ccw=ccw))
            assert np.allclose(k, 1.0 / r, atol=1e-3), (
                f"curvature deviates from 1/r (r={r}, ccw={ccw})")


def test_contour_quadrature_bundle():
    q = contour_quadrature(circle(512))
    assert abs(q["perimeter"] - 2 * np.pi) < 1e-3
    assert q["normal"].shape == (512, 2)
    assert q["curvature"].shape == (512,)


# ------------------------------------------------------------------- normals

def test_orient_normals_outward_fixes_flipped_rows():
    pos = circle(128)
    good = contour_normals(pos)
    bad = good.copy()
    bad[10] *= -1.0
    bad[77] *= -1.0
    assert outward_fraction(pos, bad) < 1.0
    fixed, n_flipped = orient_normals_outward(pos, bad)
    assert n_flipped == 2
    assert np.allclose(fixed, good, atol=1e-12)
    assert outward_fraction(pos, fixed) == 1.0


# ----------------------------------------------------------------------- sdf

def test_sdf_circle_sign_and_value():
    pos = circle(1024)
    grid = make_grid((-2.0, 2.0, -2.0, 2.0), (33, 33))
    sd = sdf_on_grid(pos, grid)
    assert sd.shape == (33, 33)
    r = np.linalg.norm(grid, axis=-1)
    expected = r - 1.0                       # negative inside the body
    mask = np.abs(r - 1.0) > 5e-3            # skip points straddling the polygon
    err = np.abs(sd - expected)[mask]
    assert err.max() < 1e-3, f"max sdf error {err.max():.3e}"
    assert np.all(sd[r < 0.99] < 0)
    assert np.all(sd[r > 1.01] > 0)


def test_sdf_accepts_flat_points_and_closed_contour():
    pos = circle(512, close=True)
    pts = np.array([[0.0, 0.0], [1.5, 0.0], [0.0, -3.0]])
    sd = sdf_on_grid(pos, pts)
    assert sd.shape == (3,)
    assert sd[0] < 0 and sd[1] > 0 and sd[2] > 0
    assert abs(sd[0] + 1.0) < 1e-3 and abs(sd[1] - 0.5) < 1e-3


def test_occupancy_circle():
    pos = circle(512)
    grid = make_grid((-2.0, 2.0, -2.0, 2.0), (33, 33))
    occ = occupancy_on_grid(pos, grid)
    r = np.linalg.norm(grid, axis=-1)
    assert np.all(occ[r < 0.99] == 1.0)
    assert np.all(occ[r > 1.01] == 0.0)


def test_points_in_polygon_nonconvex():
    # L-shaped polygon
    poly = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
    pts = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [2.5, 0.5]])
    inside = points_in_polygon(poly, pts)
    assert inside.tolist() == [True, True, False, True, False]


# ------------------------------------------------------------------ symmetry

class EquivariantToyModel:
    """Exactly reflection-equivariant map (see symmetry.py docstring).

    p    = |pos|^2 * |cond|      (reflection-invariant scalar)
    tau  = pos * |cond|          (R# tau(Rg, Rc) == tau(g, c))
    coef = (CL, CD) = (3 * cond_y, |cond|)   (CL odd, CD even in cond_y)
    """

    def __call__(self, batch):
        pos = batch["surf_pos"]
        cond = batch["cond"]
        bidx = batch["batch_idx"]
        cmag = torch.linalg.vector_norm(cond, dim=-1)[bidx]        # (N,)
        p = (pos ** 2).sum(dim=1) * cmag
        tau = pos * cmag.unsqueeze(1)
        coef = torch.stack(
            [cond[:, 1] * 3.0, torch.linalg.vector_norm(cond, dim=1)], dim=1)
        return {"p": p, "tau": tau, "coef_head": coef}


class BrokenToyModel(EquivariantToyModel):
    """Breaks equivariance: p depends on raw y."""

    def __call__(self, batch):
        out = super().__call__(batch)
        out["p"] = out["p"] + 5.0 * batch["surf_pos"][:, 1]
        return out


def _toy_batch():
    pos = torch.cat([
        torch.tensor(circle(64), dtype=torch.float64),
        torch.tensor(circle(48, r=0.7), dtype=torch.float64) + 0.1,
    ])
    bidx = torch.cat([torch.zeros(64, dtype=torch.long),
                      torch.ones(48, dtype=torch.long)])
    cond = torch.tensor([[30.0, 2.0], [25.0, -4.0]], dtype=torch.float64)
    return {"surf_pos": pos, "cond": cond, "batch_idx": bidx}


def test_symmetric_model_has_zero_symmetry_residual():
    res = symmetry_residual(EquivariantToyModel(), _toy_batch())
    assert res.shape == (2,)
    assert torch.all(res < 1e-10), f"residual of equivariant model: {res}"


def test_broken_model_has_positive_symmetry_residual():
    res = symmetry_residual(BrokenToyModel(), _toy_batch())
    assert torch.all(res > 1e-3)


def test_antisymmetry_check():
    gaps = antisymmetry_check(EquivariantToyModel(), _toy_batch())
    assert torch.all(gaps["cl_gap"] < 1e-12)
    assert torch.all(gaps["cd_gap"] < 1e-12)


def test_reflect_batch_and_prediction_roundtrip():
    batch = _toy_batch()
    rb = reflect_x_batch(batch)
    assert torch.allclose(rb["surf_pos"][:, 0], batch["surf_pos"][:, 0])
    assert torch.allclose(rb["surf_pos"][:, 1], -batch["surf_pos"][:, 1])
    assert torch.allclose(rb["cond"][:, 1], -batch["cond"][:, 1])
    # double reflection is identity
    rrb = reflect_x_batch(rb)
    assert torch.allclose(rrb["surf_pos"], batch["surf_pos"])
    pred = {"p": torch.randn(5), "tau": torch.randn(5, 2),
            "coef_head": torch.randn(2, 2)}
    rp = reflect_prediction(reflect_prediction(pred))
    assert torch.allclose(rp["tau"], pred["tau"])
    assert torch.allclose(rp["coef_head"], pred["coef_head"])


# ----------------------------------------------------------------- residuals

def _random_points(n, seed):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.uniform(-1, 1, size=(n, 2)), dtype=torch.float64)


def test_divergence_exact_for_linear_field():
    pos = _random_points(200, seed=1)
    a_mat = torch.tensor([[2.0, 3.0], [1.0, -5.0]], dtype=torch.float64)
    b_vec = torch.tensor([0.3, -0.7], dtype=torch.float64)
    u = pos @ a_mat.T + b_vec
    div_true = (a_mat[0, 0] + a_mat[1, 1]).item()   # = -3
    div = pointwise_divergence(u, pos, k=8)
    assert torch.allclose(div, torch.full_like(div, div_true), atol=1e-8), (
        f"max err {(div - div_true).abs().max():.3e}")
    res = divergence_residual(u, pos, k=8)
    assert abs(res.item() - div_true ** 2) < 1e-6


def test_divergence_batched_per_sample():
    pos1, pos2 = _random_points(150, 2), _random_points(120, 3)
    a1 = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)   # div 3
    a2 = torch.tensor([[4.0, 1.0], [2.0, -4.0]], dtype=torch.float64)  # div 0
    pos = torch.cat([pos1, pos2])
    u = torch.cat([pos1 @ a1.T, pos2 @ a2.T])
    bidx = torch.cat([torch.zeros(150, dtype=torch.long),
                      torch.ones(120, dtype=torch.long)])
    res = divergence_residual(u, pos, batch_idx=bidx, k=8)
    assert res.shape == (2,)
    assert abs(res[0].item() - 9.0) < 1e-6
    assert abs(res[1].item()) < 1e-10


def test_divergence_precomputed_weights_and_grad():
    pos = _random_points(100, 4)
    weights = knn_ls_weights(pos, k=8)
    u = _random_points(100, 5).clone().requires_grad_(True)
    res = divergence_residual(u, pos=None, weights=weights)
    res.backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()
    assert u.grad.abs().sum() > 0


def test_noslip_residual_values():
    u = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    sdf = torch.tensor([0.0, 0.05, 0.5], dtype=torch.float64)
    delta = 0.1
    # in band: point 0 (w=1) and point 1 (w=0.5); point 2 outside
    expected = (1.0 * 1.0 + 4.0 * 0.5) / 2.0
    res = noslip_residual(u, sdf, delta)
    assert abs(res.item() - expected) < 1e-12
    # zero velocity -> zero residual; empty band -> zero
    assert noslip_residual(torch.zeros_like(u), sdf, delta).item() == 0.0
    assert noslip_residual(u, sdf + 10.0, delta).item() == 0.0


def test_noslip_residual_batched():
    u = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0], [1.0, 1.0]],
                     dtype=torch.float64)
    sdf = torch.tensor([0.0, 0.05, 0.5, 0.02], dtype=torch.float64)
    bidx = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    res = noslip_residual(u, sdf, 0.1, batch_idx=bidx)
    assert res.shape == (2,)
    assert abs(res[0].item() - (1.0 + 4.0 * 0.5) / 2.0) < 1e-12
    assert abs(res[1].item() - (2.0 * 0.8) / 1.0) < 1e-12
