"""CPU tests for src/physics/force_integration.py (gating module, CONTEXT.md s6).

Analytic checks on circles (equispaced points + trapezoid-like point weights
give spectral accuracy for smooth periodic integrands, so tolerances are
tight in float64), plus differentiability, batched==unbatched equality, the
kinematic pressure mode, and — when the processed AirfRANS cache exists — the
gating comparison against the dataset's own coefficients.

Lift-sign cross-checks encoded here:
  * cylinder with CCW circulation Gamma: L = -rho U Gamma (Kutta-Joukowski;
    positive lift requires clockwise circulation), at any freestream angle;
  * reflecting the problem about the x-axis flips CL and preserves CD.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.geometry.quadrature import contour_normals, point_weights
from src.physics.force_integration import integrate_forces


def circle_geom(n=2048, r=1.0, dtype=torch.float64):
    """Unit-circle contour with outward normals and quadrature weights."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pos = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    normal = torch.tensor(contour_normals(pos), dtype=dtype)
    ds = torch.tensor(point_weights(pos), dtype=dtype)
    return th, torch.tensor(pos, dtype=dtype), normal, ds


def test_constant_pressure_zero_net_force():
    _, _, normal, ds = circle_geom(512)
    p = torch.full((512,), 3.7, dtype=torch.float64)
    tau = torch.zeros(512, 2, dtype=torch.float64)
    cond = torch.tensor([30.0, 0.0], dtype=torch.float64)
    out = integrate_forces(p, tau, normal, ds, cond)
    for key in ("fx", "fy", "cl", "cd"):
        assert out[key].abs().item() < 1e-5, f"{key} = {out[key].item():.3e}"


def test_pressure_n_dot_e_known_integral():
    # p = n . e  =>  F = -(closed-integral n n^T ds) e = -pi R e  (R = 1)
    _, _, normal, ds = circle_geom(1024)
    e_vec = torch.tensor([0.3, -0.7], dtype=torch.float64)
    e_vec = e_vec / e_vec.norm()
    p = normal @ e_vec
    tau = torch.zeros(1024, 2, dtype=torch.float64)
    cond = torch.tensor([10.0, 0.0], dtype=torch.float64)
    out = integrate_forces(p, tau, normal, ds, cond)
    expected = -np.pi * e_vec
    assert abs(out["fx"].item() - expected[0].item()) < 1e-4
    assert abs(out["fy"].item() - expected[1].item()) < 1e-4


@pytest.mark.parametrize("alpha_deg", [0.0, 7.0, -12.0])
def test_cylinder_potential_flow_dalembert(alpha_deg):
    # cp(theta) = 1 - 4 sin^2(theta - alpha): zero lift AND zero drag,
    # independent of the freestream angle; constant p_inf offset must vanish.
    n = 2048
    th, _, normal, ds = circle_geom(n)
    alpha = np.deg2rad(alpha_deg)
    u_inf = 30.0
    rho = 1.2
    q = 0.5 * rho * u_inf ** 2
    cp = 1.0 - 4.0 * np.sin(th - alpha) ** 2
    p = torch.tensor(q * cp + 13.7, dtype=torch.float64)   # + arbitrary p_inf
    tau = torch.zeros(n, 2, dtype=torch.float64)
    cond = torch.tensor([u_inf * np.cos(alpha), u_inf * np.sin(alpha)],
                        dtype=torch.float64)
    out = integrate_forces(p, tau, normal, ds, cond, rho=rho)
    assert out["cd"].abs().item() < 1e-6, f"cd = {out['cd'].item():.3e}"
    assert out["cl"].abs().item() < 1e-6, f"cl = {out['cl'].item():.3e}"


@pytest.mark.parametrize("alpha_deg", [0.0, 20.0])
def test_rotating_cylinder_lift_sign_and_magnitude(alpha_deg):
    # Surface speed v/U = -2 sin(theta - alpha) + g  => CCW circulation
    # Gamma = 2 pi a U g. Kutta-Joukowski: L = -rho U Gamma, so
    # CL = L / (0.5 rho U^2 a_ref) = -4 pi a g / a_ref  (negative for g > 0:
    # positive lift needs clockwise circulation, as on a lifting airfoil).
    # Drag stays zero. Result must be independent of the freestream angle.
    n = 2048
    th, _, normal, ds = circle_geom(n)
    alpha = np.deg2rad(alpha_deg)
    u_inf = 10.0
    g_hat = 0.25
    q = 0.5 * u_inf ** 2
    cp = 1.0 - (-2.0 * np.sin(th - alpha) + g_hat) ** 2
    p = torch.tensor(q * cp, dtype=torch.float64)
    tau = torch.zeros(n, 2, dtype=torch.float64)
    cond = torch.tensor([u_inf * np.cos(alpha), u_inf * np.sin(alpha)],
                        dtype=torch.float64)
    out = integrate_forces(p, tau, normal, ds, cond, rho=1.0, a_ref=1.0)
    cl_expected = -4.0 * np.pi * g_hat
    assert abs(out["cl"].item() - cl_expected) < 1e-4 * abs(cl_expected), (
        f"cl = {out['cl'].item():.6f}, expected {cl_expected:.6f}")
    assert out["cd"].abs().item() < 1e-6


def test_reflection_flips_cl_preserves_cd():
    # Reflect the whole problem about the x-axis: CL -> -CL, CD -> CD.
    n = 1024
    th, pos, normal, ds = circle_geom(n)
    rng = np.random.default_rng(0)
    coeffs = rng.normal(size=(4, 2))
    p_np = sum(coeffs[k, 0] * np.cos((k + 1) * th) + coeffs[k, 1] * np.sin((k + 1) * th)
               for k in range(4))
    p = torch.tensor(p_np, dtype=torch.float64)
    tau = 0.1 * torch.tensor(
        np.stack([np.cos(2 * th), np.sin(3 * th)], axis=1), dtype=torch.float64)
    cond = torch.tensor([20.0, 3.0], dtype=torch.float64)

    out = integrate_forces(p, tau, normal, ds, cond)

    flip = torch.tensor([1.0, -1.0], dtype=torch.float64)
    out_r = integrate_forces(p, tau * flip, normal * flip, ds, cond * flip)
    assert abs(out_r["cl"].item() + out["cl"].item()) < 1e-10
    assert abs(out_r["cd"].item() - out["cd"].item()) < 1e-10


def test_shear_only_drag():
    # tau aligned with the freestream everywhere: D = sum |tau| ds = 2 pi t0,
    # L = 0 (for constant tau in the flow direction).
    n = 1024
    _, _, normal, ds = circle_geom(n)
    u_inf = 5.0
    t0 = 0.3
    p = torch.zeros(n, dtype=torch.float64)
    tau = torch.zeros(n, 2, dtype=torch.float64)
    tau[:, 0] = t0
    cond = torch.tensor([u_inf, 0.0], dtype=torch.float64)
    out = integrate_forces(p, tau, normal, ds, cond, rho=2.0)
    q = 0.5 * 2.0 * u_inf ** 2
    assert abs(out["cd"].item() - 2 * np.pi * t0 / q) < 1e-6
    assert out["cl"].abs().item() < 1e-12


def test_kinematic_mode_ignores_rho():
    n = 512
    th, _, normal, ds = circle_geom(n)
    p = torch.tensor(np.sin(th) + 0.2 * np.cos(3 * th), dtype=torch.float64)
    tau = 0.01 * torch.tensor(np.stack([np.cos(th), np.sin(2 * th)], 1),
                              dtype=torch.float64)
    cond = torch.tensor([12.0, 1.5], dtype=torch.float64)
    ref = integrate_forces(p, tau, normal, ds, cond, rho=1.0, pressure_mode="static")
    kin = integrate_forces(p, tau, normal, ds, cond, rho=997.0,
                           pressure_mode="kinematic")
    for key in ("cl", "cd", "fx", "fy"):
        assert torch.allclose(ref[key], kin[key], atol=1e-12), key
    with pytest.raises(ValueError):
        integrate_forces(p, tau, normal, ds, cond, pressure_mode="dynamic")


def test_batched_matches_unbatched():
    sizes = (256, 512, 300)
    radii = (1.0, 0.5, 2.0)
    conds = torch.tensor([[30.0, 2.0], [25.0, -3.0], [40.0, 0.0]],
                         dtype=torch.float64)
    ps, taus, normals, dss, outs = [], [], [], [], []
    for i, (n, r) in enumerate(zip(sizes, radii)):
        th, _, normal, ds = circle_geom(n, r=r)
        rng = np.random.default_rng(i)
        p = torch.tensor(rng.normal(size=n), dtype=torch.float64)
        tau = torch.tensor(rng.normal(size=(n, 2)), dtype=torch.float64)
        outs.append(integrate_forces(p, tau, normal, ds, conds[i],
                                     rho=1.3, a_ref=0.8))
        ps.append(p); taus.append(tau); normals.append(normal); dss.append(ds)
    batch_idx = torch.cat([torch.full((n,), i, dtype=torch.long)
                           for i, n in enumerate(sizes)])
    out_b = integrate_forces(torch.cat(ps), torch.cat(taus), torch.cat(normals),
                             torch.cat(dss), conds, batch_idx=batch_idx,
                             rho=1.3, a_ref=0.8)
    for key in ("cl", "cd", "fx", "fy"):
        assert out_b[key].shape == (3,)
        for i in range(3):
            assert abs(out_b[key][i].item() - outs[i][key].item()) < 1e-12, (
                f"{key}[{i}] batched != unbatched")


def test_gradients_flow_through_integration():
    n = 256
    th, _, normal, ds = circle_geom(n, dtype=torch.float32)
    p = torch.tensor(np.sin(th), dtype=torch.float32, requires_grad=True)
    tau = torch.zeros(n, 2, dtype=torch.float32, requires_grad=True)
    cond = torch.tensor([10.0, 1.0], dtype=torch.float32)
    out = integrate_forces(p, tau, normal, ds, cond)
    (out["cl"] + out["cd"]).backward()
    for t in (p, tau):
        assert t.grad is not None
        assert torch.isfinite(t.grad).all()
        assert t.grad.abs().sum() > 0


def test_shape_asserts():
    _, _, normal, ds = circle_geom(64)
    p = torch.zeros(64, dtype=torch.float64)
    tau = torch.zeros(64, 2, dtype=torch.float64)
    with pytest.raises(AssertionError):
        integrate_forces(p.unsqueeze(1).squeeze(-1).unsqueeze(1), tau, normal, ds,
                         torch.tensor([1.0, 0.0], dtype=torch.float64))
    with pytest.raises(AssertionError):
        integrate_forces(p, tau[:32], normal, ds,
                         torch.tensor([1.0, 0.0], dtype=torch.float64))
    with pytest.raises(AssertionError):  # batched cond without batch_idx
        integrate_forces(p, tau, normal, ds,
                         torch.tensor([[1.0, 0.0]], dtype=torch.float64))


# ------------------------------------------------------- dataset gating test

_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed" / "airfrans"


@pytest.mark.skipif(not (_PROCESSED / "manifest.json").exists(),
                    reason="processed AirfRANS cache not built yet (A1/G1)")
def test_gate_reproduces_dataset_coefficients():
    """GATE G1: ground-truth fields must reproduce cl_true/cd_true.

    AirfRANS pressure is kinematic (p/rho, DATA_NOTES.md), so
    pressure_mode='kinematic'; a_ref = chord = 1.
    Requirement: median relative error <= 1% across sims (CONTEXT.md s6).
    """
    manifest = json.loads((_PROCESSED / "manifest.json").read_text())
    sims = manifest["sims"] if isinstance(manifest, dict) else manifest
    rel_cl, rel_cd = [], []
    n_checked = 0
    for entry in sims:
        name = entry["name"] if isinstance(entry, dict) else entry
        f = _PROCESSED / f"{name}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        out = integrate_forces(
            torch.tensor(d["surf_p"], dtype=torch.float64),
            torch.tensor(d["surf_tau"], dtype=torch.float64),
            torch.tensor(d["surf_normal"], dtype=torch.float64),
            torch.tensor(d["surf_ds"], dtype=torch.float64),
            torch.tensor(d["cond"], dtype=torch.float64),
            pressure_mode="kinematic",
        )
        cl_t, cd_t = float(d["cl_true"]), float(d["cd_true"])
        rel_cl.append(abs(out["cl"].item() - cl_t) / (abs(cl_t) + 1e-6))
        rel_cd.append(abs(out["cd"].item() - cd_t) / (abs(cd_t) + 1e-6))
        n_checked += 1
        if n_checked >= 200:
            break
    assert n_checked > 0, "manifest present but no npz files found"
    med_cl, med_cd = float(np.median(rel_cl)), float(np.median(rel_cd))
    assert med_cd <= 0.01, f"median CD rel err {med_cd:.4f} > 1% over {n_checked} sims"
    assert med_cl <= 0.01, f"median CL rel err {med_cl:.4f} > 1% over {n_checked} sims"
