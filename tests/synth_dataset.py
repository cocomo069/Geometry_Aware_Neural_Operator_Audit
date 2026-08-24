"""Fabricate an AirfRANS-format simulation (O-grid around a circle) so that
scripts/build_cache.py can be smoke-tested before the 10 GB download lands.

Produces <root>/<name>/{<name>_internal.vtu, <name>_aerofoil.vtp} and
<root>/manifest.json, matching every structural assumption of
airfrans.Simulation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

R0 = 0.5          # "airfoil" radius
R1 = 3.0          # far field
Z = 0.5           # the real dataset is a z=0.5 slice
VTK_QUAD = 9


def build_sim(root: Path, name: str, n_theta: int = 96, n_r: int = 14) -> None:
    u_inf = float(name.split("_")[2])

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    r = R0 * (R1 / R0) ** np.linspace(0.0, 1.0, n_r)          # geometric grading
    rr, tt = np.meshgrid(r, theta, indexing="ij")             # (n_r, n_theta)

    pts = np.stack(
        [(rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel(),
         np.full(rr.size, Z)],
        axis=1,
    ).astype(np.float64)

    def pid(j: int, i: int) -> int:
        return j * n_theta + (i % n_theta)

    cells, ctypes = [], []
    for j in range(n_r - 1):
        for i in range(n_theta):
            cells += [4, pid(j, i), pid(j, i + 1), pid(j + 1, i + 1), pid(j + 1, i)]
            ctypes.append(VTK_QUAD)
    grid = pv.UnstructuredGrid(
        np.asarray(cells, dtype=np.int64),
        np.asarray(ctypes, dtype=np.uint8),
        pts,
    )

    rv, tv = rr.ravel(), tt.ravel()
    surf = rv <= R0 * (1.0 + 1e-12)

    # Smooth field that vanishes exactly on the inner ring (the no-slip wall).
    damp = 1.0 - R0 / rv
    U = np.zeros((pts.shape[0], 3))
    U[:, 0] = u_inf * damp * (1.0 + 0.15 * np.sin(tv))
    U[:, 1] = 0.30 * u_inf * damp * np.sin(2.0 * tv)
    U[surf] = 0.0                                   # exact zeros -> surface mask

    speed = np.linalg.norm(U[:, :2], axis=1)
    grid.point_data["U"] = U
    grid.point_data["p"] = (0.5 * (u_inf ** 2 - speed ** 2)).astype(np.float64)
    grid.point_data["nut"] = (1e-4 * damp + 1e-6).astype(np.float64)
    # airfrans does sdf = -implicit_distance, so store the negated distance.
    grid.point_data["implicit_distance"] = (-(rv - R0)).astype(np.float64)

    # --- airfoil patch: the inner ring, byte-identical coordinates ----------
    ring = pts[:n_theta].copy()
    lines = []
    for i in range(n_theta):
        lines += [2, i, (i + 1) % n_theta]          # closed polyline
    poly = pv.PolyData(ring, lines=np.asarray(lines, dtype=np.int64))
    inward = np.stack(
        [-np.cos(theta), -np.sin(theta), np.zeros(n_theta)], axis=1
    )
    poly.point_data["Normals"] = inward.astype(np.float64)
    poly.point_data["p"] = grid.point_data["p"][:n_theta]
    poly.point_data["U"] = U[:n_theta]
    poly.point_data["nut"] = grid.point_data["nut"][:n_theta]

    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    grid.save(d / f"{name}_internal.vtu")
    poly.save(d / f"{name}_aerofoil.vtp")


def main() -> int:
    root = Path(sys.argv[1])
    names = [
        "airFoil2D_SST_50.0_0.0_0.0_0.0_12.0",        # 4-digit, Re ~3.2e6
        "airFoil2D_SST_80.0_4.0_2.0_3.0_1.0_15.0",    # 5-digit, Re ~5.1e6
        "airFoil2D_SST_35.0_-3.0_1.0_4.0_9.0",        # 4-digit, Re ~2.2e6
        "airFoil2D_SST_90.0_11.0_3.0_3.0_0.0_18.0",   # 5-digit, Re ~5.8e6
    ]
    root.mkdir(parents=True, exist_ok=True)
    for n in names:
        build_sim(root, n)
    manifest = {
        "full_train": names[:3],
        "full_test": names[3:],
        "scarce_train": names[:2],
        "reynolds_train": [names[0], names[1]],
        "reynolds_test": [names[2], names[3]],
        "aoa_train": names[:3],
        "aoa_test": names[3:],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    print(f"wrote {len(names)} synthetic sims to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
