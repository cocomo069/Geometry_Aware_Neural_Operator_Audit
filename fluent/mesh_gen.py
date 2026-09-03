"""fluent/mesh_gen.py -- parametric 2D C-grid generator for NACA airfoils.

Writes a **native Fluent 2D ASCII `.msh`** directly (no gmsh, no meshio, no ICEM).

Why this route rather than gmsh or an ICEM `.rpl` replay
--------------------------------------------------------
See `docs/FLUENT_PLAN.md` section 2 for the full argument. Short version:

* A C-grid around an airfoil is a *fully algebraic* topology -- there is nothing
  for an unstructured mesher to discover.  Writing it directly gives exact
  control over the first-cell height (which is the whole point of a y+ target),
  exact control over the refinement ratio (which is the whole point of a GCI
  study), and a bit-for-bit reproducible mesh from a seedless deterministic
  script.
* `meshio`'s Fluent writer drops boundary face zones and is unusable; gmsh
  cannot write a Fluent `.msh` at all.  So a gmsh route would still need this
  file's writer -- gmsh would only replace the (trivial) algebra.
* gmsh is an extra ~60 MB wheel.  `docs/CONTEXT.md` §1 forbids compiled
  extensions beyond the standard scientific stack; this module needs **numpy
  only**.  (If you ever do want gmsh: `uv pip install gmsh` ships a pure
  binary wheel on Windows -- *do not install it as part of this task*.)
* An ICEM Tcl replay template is still provided at
  `fluent/templates/mesh_cgrid.rpl` as an independent cross-check route, since
  `mcp__ansys__icemcfd_run_script` is available.  Use it if the quality metrics
  printed by this script are ever judged inadequate.

Topology
--------
Single-block C-grid.  Index (i, j); i runs along the body+wake line, j runs
outward.  The (i, j) map is right-handed everywhere (positive Jacobian).

    i = 0        .. nw          lower wake cut, from the outlet plane to the TE
    i = nw       .. nw+na       airfoil surface, TE -> lower -> LE -> upper -> TE
    i = nw+na    .. ni          upper wake cut, from the TE to the outlet plane
    j = 0                       body + wake-cut line
    j = nj                      far-field C (semicircle + two downstream lines)

`ni = 2*nw + na`.  The wake cut is handled the way a real C-grid handles it:
the j = 0 nodes of the upper branch are *identified with* (the same node ids as)
the j = 0 nodes of the lower branch, so the faces along the cut come out as
ordinary **interior** faces.  No periodic/interface BC is needed.

    node(nw + na + k, 0) === node(nw - k, 0),   k = 0 .. nw

Boundary zones written
----------------------
    interior   (type 2)   all internal faces incl. the wake cut
    airfoil    (type 3)   wall,             j = 0, i in [nw, nw+na)
    farfield   (type 10)  velocity-inlet,   j = nj
    outlet     (type 5)   pressure-outlet,  i = 0 and i = ni

Angle of attack is applied by **rotating the airfoil**, not by angling the
freestream.  The freestream is therefore always along +x, the wake cut is
aligned with the wake, drag = Fx and lift = Fy.  See FLUENT_PLAN.md §3.

Coordinates are written in **metres**; all counts and node/cell/face indices in
the `.msh` section headers and data rows are **hexadecimal** (Fluent's format).

CLI
---
    .venv/Scripts/python.exe fluent/mesh_gen.py --naca 0012 --re 3e6 --aoa 5 \
        --level 2 --out fluent/cases/demo/mesh.msh

Run `--help` for every knob.  Nothing here is expensive: the baseline 43k-cell
mesh builds in a couple of seconds and uses a few MB.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Fluid properties.  Defaults are air at 298.15 K / 1 atm, which is what the
# AirfRANS generator states it used.  VERIFY against docs/DATA_NOTES.md once
# A1 has introspected the dataset, and override via --rho / --mu if they differ.
# --------------------------------------------------------------------------
RHO_DEFAULT = 1.184        # kg/m^3
MU_DEFAULT = 1.85e-5       # Pa.s   -> nu = 1.5625e-5 m^2/s

# Grid levels for the 3-level grid-independence study (docs/FLUENT_PLAN.md §4).
# Refinement ratio r ~ 1.5 in *each* direction, so the cell count roughly
# doubles-and-a-bit per level and the GCI representative size h ~ N^{-1/2}
# shrinks by ~1.5.  `na` must be even so the leading edge lands on a node.
GRID_LEVELS = {
    1: dict(na=170, nw=42, nj=74),     # coarse   ~ 18.8k cells
    2: dict(na=256, nw=64, nj=112),    # medium   ~ 43.0k cells  (baseline)
    3: dict(na=384, nw=96, nj=168),    # fine     ~ 96.8k cells
}
LEVEL_YPLUS_SCALE = {1: 1.5, 2: 1.0, 3: 1.0 / 1.5}

# Fluent zone-type codes (see docs/FLUENT_PLAN.md §2 for the table).
ZT_INTERIOR = 2
ZT_WALL = 3
ZT_PRESSURE_OUTLET = 5
ZT_VELOCITY_INLET = 10

# Cap on the wall-normal aspect ratio away from the section (see build_cgrid).
AR_CAP = 5000.0


# ==========================================================================
# 1. Airfoil geometry
# ==========================================================================

def naca4_halves(digits: str, x: np.ndarray):
    """Camber line y_c(x), slope dy_c/dx and half-thickness y_t(x) for NACA 4-digit.

    `digits` is e.g. "0012" or "4412".  `x` in [0, 1].  Uses the **closed**
    trailing-edge coefficient (-0.1036) so the TE is a single sharp point,
    which the C-grid topology requires.
    """
    m = int(digits[0]) / 100.0
    p = int(digits[1]) / 10.0
    t = int(digits[2:4]) / 100.0

    yt = 5.0 * t * (0.2969 * np.sqrt(np.clip(x, 0.0, None))
                    - 0.1260 * x
                    - 0.3516 * x ** 2
                    + 0.2843 * x ** 3
                    - 0.1036 * x ** 4)

    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if m > 0.0 and p > 0.0:
        fwd = x < p
        aft = ~fwd
        yc[fwd] = m / p ** 2 * (2 * p * x[fwd] - x[fwd] ** 2)
        dyc[fwd] = 2 * m / p ** 2 * (p - x[fwd])
        yc[aft] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[aft] - x[aft] ** 2)
        dyc[aft] = 2 * m / (1 - p) ** 2 * (p - x[aft])
    return yc, dyc, yt


# Standard NACA 5-digit mean-line constants (normal, non-reflex series),
# tabulated for design lift coefficient Cl = 0.3 (the "2" in 230xx).
_NACA5_TABLE = {
    0.05: (0.0580, 361.400),
    0.10: (0.1260, 51.640),
    0.15: (0.2025, 15.957),
    0.20: (0.2900, 6.643),
    0.25: (0.3910, 3.230),
}


def naca5_halves(digits: str, x: np.ndarray):
    """Camber line, slope and half-thickness for a NACA 5-digit section.

    `digits` is e.g. "23012": L=2 (design Cl = 0.15*L), P=3 (mean-line position
    p = P/20), Q=0 (0 = normal, 1 = reflex), XX = thickness in percent.

    Only the **normal** (Q = 0) series is implemented; reflex mean lines need a
    second constant k2/k1 that is not tabulated here.  Raises on Q = 1 rather
    than silently producing the wrong shape.
    """
    L = int(digits[0])
    P = int(digits[1])
    Q = int(digits[2])
    t = int(digits[3:5]) / 100.0
    if Q != 0:
        raise ValueError(
            f"NACA {digits}: reflex (third digit = 1) 5-digit mean lines are not "
            "implemented -- the k2/k1 constants are not tabulated in this module."
        )

    p = P / 20.0
    keys = sorted(_NACA5_TABLE)
    if p in _NACA5_TABLE:
        m, k1 = _NACA5_TABLE[p]
    else:                                     # linear interpolation in p
        if not (keys[0] <= p <= keys[-1]):
            raise ValueError(f"NACA {digits}: mean-line position p={p} out of tabulated range")
        ms = np.interp(p, keys, [_NACA5_TABLE[k][0] for k in keys])
        k1s = np.interp(p, keys, [_NACA5_TABLE[k][1] for k in keys])
        m, k1 = float(ms), float(k1s)

    # Table is for Cl_design = 0.3; scale k1 linearly with the design Cl.
    k1 *= (0.15 * L) / 0.3

    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    fwd = x < m
    aft = ~fwd
    yc[fwd] = (k1 / 6.0) * (x[fwd] ** 3 - 3 * m * x[fwd] ** 2 + m ** 2 * (3 - m) * x[fwd])
    dyc[fwd] = (k1 / 6.0) * (3 * x[fwd] ** 2 - 6 * m * x[fwd] + m ** 2 * (3 - m))
    yc[aft] = (k1 * m ** 3 / 6.0) * (1.0 - x[aft])
    dyc[aft] = -(k1 * m ** 3 / 6.0)

    yt = 5.0 * t * (0.2969 * np.sqrt(np.clip(x, 0.0, None))
                    - 0.1260 * x
                    - 0.3516 * x ** 2
                    + 0.2843 * x ** 3
                    - 0.1036 * x ** 4)
    return yc, dyc, yt


def airfoil_loop(digits: str, na: int, chord: float, aoa_deg: float):
    """Closed airfoil contour, TE -> lower -> LE -> upper -> TE, `na` cells.

    Returns an (na+1, 2) array whose first and last rows are the *same* point
    (the sharp TE).  Cosine spacing in the loop parameter clusters points at
    both the LE and the TE, which is what a surface-pressure study needs.

    The section is rotated by -aoa about the quarter chord, so the freestream
    stays along +x (see module docstring).
    """
    if na % 2 != 0:
        raise ValueError(f"na must be even so the LE lands on a node (got {na})")

    s = np.linspace(0.0, 1.0, na + 1)
    xc = 0.5 * (1.0 + np.cos(2.0 * np.pi * s))     # 1 -> 0 -> 1, clustered at both ends
    xc = np.clip(xc, 0.0, 1.0)

    if len(digits) == 4:
        yc, dyc, yt = naca4_halves(digits, xc)
    elif len(digits) == 5:
        yc, dyc, yt = naca5_halves(digits, xc)
    else:
        raise ValueError(f"NACA designation must be 4 or 5 digits, got '{digits}'")

    theta = np.arctan(dyc)
    lower = s < 0.5                                # first half of the loop
    sgn = np.where(lower, -1.0, +1.0)              # -1 on lower surface, +1 on upper
    x = xc - sgn * yt * np.sin(theta)
    y = yc + sgn * yt * np.cos(theta)

    # Force the two TE endpoints and the LE to be exact.
    x[0] = x[-1] = 1.0
    y[0] = y[-1] = float(yc[0])
    x[na // 2] = 0.0
    y[na // 2] = 0.0

    pts = np.stack([x, y], axis=1) * chord

    # Rotate by -aoa about the quarter chord: nose up for positive aoa.
    a = math.radians(aoa_deg)
    pivot = np.array([0.25 * chord, 0.0])
    ca, sa = math.cos(a), math.sin(a)
    rot = np.array([[ca, sa], [-sa, ca]])          # rotation by -a
    pts = (pts - pivot) @ rot.T + pivot
    return pts


# ==========================================================================
# 2. Spacing helpers
# ==========================================================================

def solve_growth_ratio(n: int, first: float, total: float) -> float:
    """Geometric growth ratio g with first*(g^n - 1)/(g - 1) = total.

    Bisection; returns 1.0 (uniform) if the requested first cell already fills
    the extent.  Raises if no ratio in (1, 5] can reach `total`.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if first * n >= total:
        return 1.0

    def f(g):
        return first * (g ** n - 1.0) / (g - 1.0) - total

    lo, hi = 1.0 + 1e-12, 5.0
    if f(hi) < 0.0:
        raise ValueError(
            f"cannot span {total:g} with {n} cells from a first cell of {first:g} "
            "even at growth ratio 5 -- increase the cell count or the first-cell height"
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def geometric_nodes(n: int, first: float, total: float):
    """n+1 node positions on [0, total], geometric growth from `first`."""
    g = solve_growth_ratio(n, first, total)
    if g == 1.0:
        return np.linspace(0.0, total, n + 1), 1.0
    sizes = first * g ** np.arange(n)
    pos = np.concatenate([[0.0], np.cumsum(sizes)])
    pos *= total / pos[-1]                          # kill float drift on the last node
    return pos, g


def arclength_fractions(pts: np.ndarray) -> np.ndarray:
    """Cumulative arc length of a polyline, normalised to [0, 1]."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return cum / cum[-1]


# ==========================================================================
# 3. y+ -> first-cell height
# ==========================================================================

def first_cell_height(u_inf: float, chord: float, rho: float, mu: float,
                      y_plus_target: float) -> dict:
    """First-cell *height* (full cell, wall to first node) for a y+ target.

    Flat-plate turbulent correlation, evaluated at x = c:

        Re_c = rho * U * c / mu
        C_f  = 0.026 * Re_c ** (-1/7)              (Schlichting 1/7-power law)
        tau_w = C_f * 0.5 * rho * U**2
        u_tau = sqrt(tau_w / rho)
        y_1   = y_plus * mu / (rho * u_tau)

    y_1 is the distance from the wall to the **first cell centre** in the
    definition of y+, so the first *cell height* is 2 * y_1.  We return both and
    the mesh uses the cell height.

    Caveats, documented in docs/FLUENT_PLAN.md §3:
      * this is a flat-plate estimate; the real y+ on an airfoil peaks near the
        leading edge and near the suction peak, typically 1.5-2.5x this value,
        which is why the default target is 0.6 rather than 1.0;
      * it ignores the pressure gradient entirely;
      * it is an *a priori* sizing rule only.  The journal reports the achieved
        y+ (`yplus_max.txt` / `yplus_avg.txt`) and that is the number the paper
        quotes.
    """
    re_c = rho * u_inf * chord / mu
    cf = 0.026 * re_c ** (-1.0 / 7.0)
    tau_w = cf * 0.5 * rho * u_inf ** 2
    u_tau = math.sqrt(tau_w / rho)
    y1_centre = y_plus_target * mu / (rho * u_tau)
    return dict(re_c=re_c, cf=cf, tau_w=tau_w, u_tau=u_tau,
                y1_centre=y1_centre, first_cell_height=2.0 * y1_centre)


# ==========================================================================
# 4. C-grid construction
# ==========================================================================

def build_cgrid(surface: np.ndarray, nw: int, nj: int, chord: float,
                first_cell: float, r_far: float, x_out: float,
                wake_first: float | None = None, smooth_sweeps: int = 0,
                wake_normal_blend: float = 1.0,
                outer_uniformity: float = 1.0):
    """Build the (ni+1, nj+1, 2) node coordinate array for the C-grid.

    `surface` is the (na+1, 2) closed airfoil loop from `airfoil_loop`.
    Returns (X, Y, meta).
    """
    na = surface.shape[0] - 1
    ni = 2 * nw + na
    te = surface[0].copy()                 # trailing edge (start == end of loop)

    # ---- inner line: lower wake + airfoil + upper wake ------------------
    wake_len = x_out - te[0]
    if wake_len <= 0:
        raise ValueError("x_out must be downstream of the trailing edge")
    if wake_first is None:
        # Mean surface panel length, NOT the first panel. The loop is cosine
        # clustered, so the first panel at the trailing edge is quadratically
        # small (~3e-4 c) -- seeding the wake with it drives the wake growth
        # ratio to 1.27 and drops the minimum orthogonality from 0.29 to 0.12.
        # The mean panel is the honest "TE-region streamwise spacing".
        wake_first = float(np.linalg.norm(np.diff(surface, axis=0), axis=1).mean())
    wpos, wake_g = geometric_nodes(nw, wake_first, wake_len)   # 0 (TE) -> wake_len

    inner = np.empty((ni + 1, 2))
    # lower branch: i = 0 at the outlet plane, i = nw at the TE
    inner[:nw + 1, 0] = te[0] + wpos[::-1]
    inner[:nw + 1, 1] = te[1]
    # airfoil
    inner[nw:nw + na + 1] = surface
    # upper branch: i = nw+na at the TE, i = ni at the outlet plane
    inner[nw + na:, 0] = te[0] + wpos
    inner[nw + na:, 1] = te[1]

    # ---- outward unit normals on the inner line -------------------------
    # tangent by central differences; outward normal = (-t_y, t_x) given the
    # right-handed (i, j) orientation established in the module docstring.
    tan = np.empty_like(inner)
    tan[1:-1] = inner[2:] - inner[:-2]
    tan[0] = inner[1] - inner[0]
    tan[-1] = inner[-1] - inner[-2]
    # at the TE the loop folds: use the one-sided tangent of each branch so the
    # normal does not average to zero across the fold
    tan[nw] = surface[1] - surface[0]
    tan[nw + na] = surface[-1] - surface[-2]
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], axis=1)

    # Relax each wake branch's normal from the trailing-edge surface normal to
    # the vertical over `wake_normal_blend` chords.
    #
    # Without this the ray family is DISCONTINUOUS across the trailing-edge
    # fold: the last wake node carries a normal of exactly (0, -1) while the
    # trailing-edge node one index away carries the lower-surface normal, a few
    # degrees off. Those two rays are then near-parallel and only ~1e-4 c apart,
    # so the cells between them are slivers that cross and invert. Blending
    # makes the family continuous at the fold, which is also what a real
    # hyperbolic C-grid produces -- the wake-cut rays do leave the trailing edge
    # tilted and straighten out downstream.
    #
    # The two branches necessarily disagree at the shared trailing-edge node
    # (one relaxes to (0,-1), the other to (0,+1)). That disagreement IS the
    # fold, and it is why only the j = 0 row of nodes is shared between them.
    blend = max(wake_normal_blend * chord, 1e-12)
    n_te_lo = nrm[nw].copy()
    n_te_up = nrm[nw + na].copy()

    def _relax_normal(n_te, n_inf, dist):
        w = np.clip(dist / blend, 0.0, 1.0)
        w = w * w * (3.0 - 2.0 * w)                      # smoothstep
        v = (1.0 - w)[:, None] * n_te[None, :] + w[:, None] * n_inf[None, :]
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    nrm[:nw + 1] = _relax_normal(n_te_lo, np.array([0.0, -1.0]), wpos[::-1])
    nrm[nw + na:] = _relax_normal(n_te_up, np.array([0.0, +1.0]), wpos)

    # ---- outer boundary: matched block-by-block to the inner line -------
    # Block-by-block, NOT by global arc length: the wake is ~97% of the inner
    # line's length but only ~40% of the outer boundary's, so a global
    # arc-length match would crush the whole airfoil into a handful of i-lines.
    #
    # Within each block the outer distribution is relaxed toward UNIFORM. The
    # inner line is exponentially clustered (at the TE on the wake, at the LE
    # and TE on the section); copying that clustering out to r_far = 30c
    # collapses the far-field spacing to ~1e-5 c beside cells 30 c tall, which
    # is a fold, not a stretched cell. A real hyperbolic or elliptic C-grid
    # relaxes to near-uniform in the far field, and `outer_uniformity` is how
    # much of that relaxation this algebraic construction applies. Convex
    # combinations of monotone distributions stay monotone, so the transfinite
    # blend still cannot cross itself.
    w = float(np.clip(outer_uniformity, 0.0, 1.0))

    def _relax(frac):
        return (1.0 - w) * frac + w * np.linspace(0.0, 1.0, len(frac))

    centre = np.array([0.25 * chord, 0.0])
    outer = np.empty((ni + 1, 2))

    wfrac = _relax(wpos / wpos[-1])
    outer[:nw + 1, 0] = centre[0] + (x_out - centre[0]) * wfrac[::-1]
    outer[:nw + 1, 1] = -r_far
    outer[nw + na:, 0] = centre[0] + (x_out - centre[0]) * wfrac
    outer[nw + na:, 1] = +r_far

    # airfoil block -> upstream semicircle
    f = _relax(arclength_fractions(surface))             # 0 .. 1 along the airfoil
    ang = -0.5 * np.pi - np.pi * f                       # -90 deg -> -270 deg (= +90)
    outer[nw:nw + na + 1, 0] = centre[0] + r_far * np.cos(ang)
    outer[nw:nw + na + 1, 1] = centre[1] + r_far * np.sin(ang)

    # ---- normal distribution, one ray at a time -------------------------
    # Each i-line gets its OWN geometric distribution: `nj` cells spanning that
    # ray's own length L(i), starting from that ray's own first cell fc(i).
    # A single shared normalised distribution cannot work here because L(i) is
    # ~30 c everywhere but the streamwise spacing ds(i) varies by five orders of
    # magnitude between the leading edge and the far wake.
    #
    # fc(i) caps the wall-normal aspect ratio: the y+ target is a statement
    # about the airfoil boundary layer, and forcing a 1e-5 m first cell 29
    # chords downstream in the wake -- where the streamwise spacing is metres --
    # buys nothing and costs an aspect ratio of ~4e5. On the section itself
    # ds/AR_CAP is far below `first_cell`, so the y+ sizing there is untouched;
    # the assertion enforces that rather than assuming it.
    L = np.linalg.norm(outer - inner, axis=1)
    ds = np.empty(ni + 1)
    seg = np.linalg.norm(np.diff(inner, axis=0), axis=1)
    ds[1:-1] = 0.5 * (seg[:-1] + seg[1:])
    ds[0], ds[-1] = seg[0], seg[-1]
    fc = np.maximum(first_cell, ds / AR_CAP)
    # The y+ target is sacred ON THE SECTION: the wake-oriented aspect-ratio cap
    # must never raise the wall cell there. At small y+ targets (high Re and/or
    # fine levels) the mid-chord streamwise spacing can exceed first_cell*AR_CAP,
    # so the max() above would clip the airfoil wall cell and silently break the
    # y+ sizing. Force the section back to first_cell -- this IS the guarantee the
    # old assertion only checked -- and let its (still < 1e4) wall aspect ratio
    # stand. The cap keeps doing its job on the wake rays, where it belongs.
    fc[nw:nw + na + 1] = first_cell

    S = np.empty((ni + 1, nj + 1))
    growths = np.empty(ni + 1)
    for i in range(ni + 1):
        pos, g = geometric_nodes(nj, float(fc[i]), float(L[i]))
        S[i] = pos / pos[-1]
        growths[i] = g

    # ---- transfinite blend: wall-normal near the body, radial far away ---
    # The direction blend runs on the INDEX fraction u = j/nj, not on the
    # distance fraction. The distance fraction is exponentially stretched, so a
    # distance-based blend leaves the direction still tilted away from outer(i)
    # at j = nj-1 and then snaps to outer(i) at j = nj -- a lateral jump big
    # enough to fold cells right across the airfoil block. smoothstep has zero
    # derivative at both ends: the stack leaves the wall orthogonal (which is
    # what the y+ sizing assumes) and reaches the far field with no kink.
    u = np.arange(nj + 1) / float(nj)
    phi = u * u * (3.0 - 2.0 * u)
    dir_far = (outer - inner) / L[:, None]

    X = np.empty((ni + 1, nj + 1))
    Y = np.empty((ni + 1, nj + 1))
    for j in range(nj + 1):
        if j == nj:
            X[:, j], Y[:, j] = outer[:, 0], outer[:, 1]
            continue
        d = (1.0 - phi[j]) * nrm + phi[j] * dir_far      # (ni+1, 2)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        step = S[:, j] * L
        X[:, j] = inner[:, 0] + step * d[:, 0]
        Y[:, j] = inner[:, 1] + step * d[:, 1]
    growth = float(growths[nw:nw + na + 1].max())        # on the section itself

    # ---- light i-direction smoothing to relax the TE/wake corner --------
    # Only outside the boundary-layer stack (s > 2%), so the first-cell height
    # and wall orthogonality are untouched; only along i, so the radial
    # clustering is untouched.  Endpoints i = 0 and i = ni are fixed (outlet).
    if smooth_sweeps > 0:
        j0 = int(np.searchsorted(S[nw + na // 2], 0.02))
        j0 = max(j0, 1)
        omega = 0.3
        for _ in range(smooth_sweeps):
            for arr in (X, Y):
                interior = arr[1:-1, j0:nj]
                avg = 0.5 * (arr[:-2, j0:nj] + arr[2:, j0:nj])
                arr[1:-1, j0:nj] = interior + omega * (avg - interior)

    # ---- enforce the wake-cut identification exactly --------------------
    # (j = 0 only; the two branches share that row of nodes)
    for k in range(nw + 1):
        X[nw + na + k, 0] = X[nw - k, 0]
        Y[nw + na + k, 0] = Y[nw - k, 0]

    meta = dict(na=na, nw=nw, nj=nj, ni=ni, n_cells=ni * nj,
                n_nodes_logical=(ni + 1) * (nj + 1),
                te=te.tolist(), wake_growth=wake_g, normal_growth=growth,
                first_cell=first_cell, shortest_ray=float(L.min()),
                r_far=r_far, x_out=x_out, chord=chord,
                smooth_sweeps=smooth_sweeps, outer_uniformity=outer_uniformity,
                wake_normal_blend=wake_normal_blend)
    return X, Y, meta


# ==========================================================================
# 5. Quality metrics
# ==========================================================================

def mesh_quality(X: np.ndarray, Y: np.ndarray) -> dict:
    """Cheap structured-quad quality metrics: area, aspect ratio, skew.

    `min_jacobian` <= 0 means the grid has folded -- fatal, never send it to a
    solver.  `min_orthogonality` is |cos| of the worst corner angle deviation,
    reported the same way Fluent's "orthogonal quality" reads (1 = perfect).
    """
    p00 = np.stack([X[:-1, :-1], Y[:-1, :-1]], axis=-1)
    p10 = np.stack([X[1:, :-1], Y[1:, :-1]], axis=-1)
    p11 = np.stack([X[1:, 1:], Y[1:, 1:]], axis=-1)
    p01 = np.stack([X[:-1, 1:], Y[:-1, 1:]], axis=-1)

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    # signed area via the shoelace formula on the quad
    area = 0.5 * (cross(p10 - p00, p01 - p00) + cross(p10 - p11, p01 - p11)) * -1.0
    area = np.abs(0.5 * (cross(p00, p10) + cross(p10, p11) + cross(p11, p01) + cross(p01, p00)))
    jac = cross(p10 - p00, p01 - p00)

    e_i = np.linalg.norm(p10 - p00, axis=-1)
    e_j = np.linalg.norm(p01 - p00, axis=-1)
    ar = np.maximum(e_i, e_j) / np.maximum(np.minimum(e_i, e_j), 1e-30)

    ti = (p10 - p00) / np.maximum(e_i, 1e-30)[..., None]
    tj = (p01 - p00) / np.maximum(e_j, 1e-30)[..., None]
    ortho = 1.0 - np.abs(np.sum(ti * tj, axis=-1))

    return dict(n_cells=int(area.size),
                min_area=float(area.min()), max_area=float(area.max()),
                min_jacobian=float(jac.min()),
                max_aspect_ratio=float(ar.max()),
                min_orthogonality=float(ortho.min()),
                mean_orthogonality=float(ortho.mean()))


# ==========================================================================
# 6. Fluent .msh writer
# ==========================================================================

def write_fluent_msh(path: Path, X: np.ndarray, Y: np.ndarray, meta: dict,
                     header: str = "") -> dict:
    """Write the native Fluent 2D ASCII `.msh`.

    Face-orientation convention (verified, see the cfd-workflow skill): the CW
    normal of the edge n0 -> n1, i.e. (dx, dy) -> (dy, -dx), must point **out of**
    the owner cell c0 for *every* face, boundaries included.  If interior i- and
    j-faces disagree on that convention Fluent aborts with
    "Build Grid: Aborted due to critical error" even though the zones read fine.
    """
    na, nw, nj, ni = meta["na"], meta["nw"], meta["nj"], meta["ni"]
    NJ = nj + 1

    # ---- node ids (1-based), with the wake-cut row identified -----------
    nid = np.zeros((ni + 1, NJ), dtype=np.int64)
    counter = 0
    for i in range(ni + 1):
        for j in range(NJ):
            if j == 0 and i >= nw + na:
                # Identify the whole upper wake-cut row with the lower branch,
                # INCLUDING k = 0 (i = nw+na), which is the trailing edge itself:
                # node(nw+na+k, 0) === node(nw-k, 0) for k = 0 .. nw.
                # Using `>` here (skipping k = 0) leaves the upper-branch TE node
                # un-merged -- a duplicate node at the trailing edge that splits
                # the cell fan around the TE so the upper TE cell never closes,
                # which is exactly what makes Fluent abort with
                # "Build Grid: Aborted due to critical error".
                nid[i, 0] = nid[2 * nw + na - i, 0]      # === node(ni - i, 0)
                continue
            counter += 1
            nid[i, j] = counter
    n_nodes = counter

    coords = np.zeros((n_nodes, 2))
    for i in range(ni + 1):
        for j in range(NJ):
            k = nid[i, j] - 1
            coords[k, 0] = X[i, j]
            coords[k, 1] = Y[i, j]

    def cid(i, j):
        return i * nj + j + 1                            # 1-based cell id

    n_cells = ni * nj

    # ---- faces -----------------------------------------------------------
    interior, airfoil, farfield, outlet = [], [], [], []

    # interior faces of constant i (between cell i-1 and cell i)
    for i in range(1, ni):
        for j in range(nj):
            interior.append((nid[i, j], nid[i, j + 1], cid(i - 1, j), cid(i, j)))
    # interior faces of constant j (between cell j-1 and cell j)
    for j in range(1, nj):
        for i in range(ni):
            interior.append((nid[i, j], nid[i + 1, j], cid(i, j), cid(i, j - 1)))

    # j = 0 row: wake cut (interior, folded) then the airfoil wall
    for i in range(nw):
        # partner cell across the cut is (ni - 1 - i, 0)
        interior.append((nid[i, 0], nid[i + 1, 0], cid(i, 0), cid(ni - 1 - i, 0)))
    for i in range(nw, nw + na):
        airfoil.append((nid[i, 0], nid[i + 1, 0], cid(i, 0), 0))
    # i in [nw+na, ni) duplicates the wake cut already emitted -- skip.

    # j = nj: far field
    for i in range(ni):
        farfield.append((nid[i + 1, nj], nid[i, nj], cid(i, nj - 1), 0))

    # i = 0 and i = ni: the two halves of the downstream outlet plane
    for j in range(nj):
        outlet.append((nid[0, j + 1], nid[0, j], cid(0, j), 0))
        outlet.append((nid[ni, j], nid[ni, j + 1], cid(ni - 1, j), 0))

    zones = [
        ("interior", 3, ZT_INTERIOR, interior),
        ("airfoil", 4, ZT_WALL, airfoil),
        ("farfield", 5, ZT_VELOCITY_INLET, farfield),
        ("outlet", 6, ZT_PRESSURE_OUTLET, outlet),
    ]
    n_faces = sum(len(z[3]) for z in zones)

    # ---- assemble the file (one join, one write: never line by line) -----
    def hx(v):
        return format(int(v), "x")

    out = []
    out.append('(0 "%s")' % (header or "NACA C-grid"))
    out.append('(0 "%d cells (%d x %d), %d nodes, %d faces")'
               % (n_cells, ni, nj, n_nodes, n_faces))
    out.append("(2 2)")                                        # 2D
    out.append("(10 (0 1 %s 0 2))" % hx(n_nodes))
    out.append("(10 (1 1 %s 1 2)(" % hx(n_nodes))
    out.append("\n".join("%.9e %.9e" % (cx, cy) for cx, cy in coords))
    out.append("))")
    out.append("(12 (0 1 %s 0))" % hx(n_cells))
    out.append("(12 (2 1 %s 1 3))" % hx(n_cells))              # quad element type 3
    out.append("(13 (0 1 %s 0))" % hx(n_faces))
    first = 1
    for _name, zid, bctype, faces in zones:
        last = first + len(faces) - 1
        out.append("(13 (%s %s %s %s 2)(" % (hx(zid), hx(first), hx(last), hx(bctype)))
        out.append("\n".join("%s %s %s %s" % (hx(a), hx(b), hx(c), hx(d))
                             for (a, b, c, d) in faces))
        out.append("))")
        first = last + 1
    out.append("(45 (2 fluid fluid)())")
    out.append("(45 (3 interior interior)())")
    out.append("(45 (4 wall airfoil)())")
    out.append("(45 (5 velocity-inlet farfield)())")
    out.append("(45 (6 pressure-outlet outlet)())")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="ascii")

    return dict(n_nodes=n_nodes, n_cells=n_cells, n_faces=n_faces,
                n_faces_airfoil=len(airfoil), n_faces_farfield=len(farfield),
                n_faces_outlet=len(outlet), n_faces_interior=len(interior))


# ==========================================================================
# 6b. Self-check on the written file
# ==========================================================================

def validate_msh(path: Path) -> dict:
    """Re-parse the written `.msh` and assert it is internally consistent.

    Fluent is not available while these meshes are authored, so this is the
    only check standing between a malformed file and a wasted solver run. It
    verifies what a hand-written Fluent mesh actually gets wrong:

      * every node/cell index in every face row is in range (they are HEX);
      * boundary faces have c1 = 0 and interior faces have two distinct,
        non-zero cells;
      * every cell is referenced by at least three faces (a quad needs four,
        but a cell on two boundaries still has four -- three is a loose floor
        that catches genuinely orphaned cells);
      * the declared section counts match the number of data rows.

    It does NOT check face winding -- that is enforced by construction in
    `write_fluent_msh` and is the one thing only Fluent can confirm.
    """
    text = Path(path).read_text(encoding="ascii").splitlines()
    n_nodes = n_cells = n_faces = 0
    coords, faces = [], []
    mode, remaining = None, 0
    for line in text:
        ls = line.strip()
        if ls.startswith("(10 (0"):
            n_nodes = int(ls.split()[3], 16); continue
        if ls.startswith("(12 (0"):
            n_cells = int(ls.split()[3], 16); continue
        if ls.startswith("(13 (0"):
            n_faces = int(ls.split()[3], 16); continue
        if ls.startswith("(10 (1"):
            mode = "node"; continue
        if ls.startswith("(13 ("):
            parts = ls.replace("(", " ").replace(")", " ").split()
            remaining = int(parts[3], 16) - int(parts[2], 16) + 1
            mode = "face"; continue
        if ls in ("))", ")", ""):
            mode = None; continue
        if mode == "node":
            coords.append(tuple(float(v) for v in ls.split()))
        elif mode == "face" and remaining > 0:
            faces.append(tuple(int(v, 16) for v in ls.split()))
            remaining -= 1

    errs = []
    if len(coords) != n_nodes:
        errs.append("node rows %d != declared %d" % (len(coords), n_nodes))
    if len(faces) != n_faces:
        errs.append("face rows %d != declared %d" % (len(faces), n_faces))
    refs = {}
    cell_node_hits = {}          # cell -> {node: incidence count over its faces}
    n_bnd = n_int = 0
    for k, (a, b, c0, c1) in enumerate(faces):
        for nd in (a, b):
            if not (1 <= nd <= n_nodes):
                errs.append("face %d: node index %d out of range" % (k, nd))
        if a == b:
            errs.append("face %d: degenerate (both nodes identical)" % k)
        if not (1 <= c0 <= n_cells):
            errs.append("face %d: owner cell %d out of range" % (k, c0))
        if c1 == 0:
            n_bnd += 1
        else:
            n_int += 1
            if not (1 <= c1 <= n_cells):
                errs.append("face %d: neighbour cell %d out of range" % (k, c1))
            if c1 == c0:
                errs.append("face %d: interior face with c0 == c1" % k)
        for c in (c0, c1):
            if c:
                refs[c] = refs.get(c, 0) + 1
                d = cell_node_hits.setdefault(c, {})
                d[a] = d.get(a, 0) + 1
                d[b] = d.get(b, 0) + 1
    orphan = [c for c in range(1, n_cells + 1) if refs.get(c, 0) < 3]
    if orphan:
        errs.append("%d cells referenced by fewer than 3 faces (first: %s)"
                    % (len(orphan), orphan[:5]))

    # Every cell must be a CLOSED quad: exactly 4 distinct corner nodes, each
    # shared by exactly 2 of that cell's faces. This is the check that catches a
    # wake-cut / trailing-edge node that should have been identified but was not
    # -- a duplicate node at the same location splits the cell fan so a cell
    # references 5 nodes (two of them coincident), leaving two nodes with
    # incidence 1. validate_msh missed exactly this before, and Fluent aborts on
    # it with "Build Grid: Aborted due to critical error".
    unclosed = []
    for c in range(1, n_cells + 1):
        hits = cell_node_hits.get(c, {})
        if len(hits) != 4 or any(v != 2 for v in hits.values()):
            unclosed.append(c)
    if unclosed:
        errs.append("%d cells are not closed 4-node quads (first: %s) -- most "
                    "likely an un-identified duplicate node on the wake cut / TE"
                    % (len(unclosed), unclosed[:5]))
    if errs:
        sep = chr(10) + "  "
        raise SystemExit("mesh self-check FAILED for %s:%s%s"
                         % (path, sep, sep.join(errs)))
    return dict(n_nodes=n_nodes, n_cells=n_cells, n_faces=n_faces,
                n_boundary_faces=n_bnd, n_interior_faces=n_int)


# ==========================================================================
# 7. Driver
# ==========================================================================

def generate(naca: str, re: float, aoa_deg: float, level: int, out: Path,
             chord: float = 1.0, rho: float = RHO_DEFAULT, mu: float = MU_DEFAULT,
             y_plus: float = 0.25, r_far: float = 30.0, x_out: float = 30.0,
             smooth_sweeps: int = 0, outer_uniformity: float = 1.0,
             overrides: dict | None = None,
             verbose: bool = True) -> dict:
    """Generate one mesh; returns the full parameter/quality record."""
    res = dict(GRID_LEVELS[level])
    if overrides:
        res.update({k: v for k, v in overrides.items() if v is not None})

    u_inf = re * mu / (rho * chord)
    yp = y_plus * LEVEL_YPLUS_SCALE[level]
    bl = first_cell_height(u_inf, chord, rho, mu, yp)

    surface = airfoil_loop(naca, res["na"], chord, aoa_deg)
    X, Y, meta = build_cgrid(surface, res["nw"], res["nj"], chord,
                             bl["first_cell_height"], r_far * chord,
                             x_out * chord, smooth_sweeps=smooth_sweeps,
                             outer_uniformity=outer_uniformity)
    qual = mesh_quality(X, Y)
    hdr = (f"NACA {naca}  Re={re:.3g}  AoA={aoa_deg:g} deg  level={level}  "
           f"U_inf={u_inf:.4f} m/s  y+_target={yp:.3f}")
    counts = write_fluent_msh(Path(out), X, Y, meta, header=hdr)

    record = dict(naca_digits=naca, re=re, aoa_deg=aoa_deg, mesh_level=level,
                  chord=chord, rho=rho, mu=mu, nu=mu / rho, u_inf=u_inf,
                  a_ref=chord * 1.0, y_plus_target=yp,
                  boundary_layer=bl, grid=meta, quality=qual, counts=counts,
                  mesh_file=str(out))

    if verbose:
        print(f"[mesh] {hdr}")
        print(f"[mesh]   wrote {out}")
        print(f"[mesh]   {counts['n_cells']} cells  ({meta['ni']} x {meta['nj']}), "
              f"{counts['n_nodes']} nodes, {counts['n_faces']} faces")
        print(f"[mesh]   airfoil faces {counts['n_faces_airfoil']}, "
              f"farfield {counts['n_faces_farfield']}, outlet {counts['n_faces_outlet']}")
        print(f"[mesh]   first cell height {bl['first_cell_height']:.3e} m "
              f"(y+ target {yp:.2f}, u_tau {bl['u_tau']:.3f} m/s)")
        print(f"[mesh]   normal growth ratio {meta['normal_growth']:.4f}, "
              f"wake growth {meta['wake_growth']:.4f}")
        print(f"[mesh]   min orthogonality {qual['min_orthogonality']:.3f}, "
              f"max AR {qual['max_aspect_ratio']:.3g}, "
              f"min Jacobian {qual['min_jacobian']:.3e}")
        if qual["min_jacobian"] <= 0.0:
            print("[mesh]   *** FATAL: folded cells (min Jacobian <= 0). "
                  "Do not solve on this mesh. ***")
        if meta["normal_growth"] > 1.2:
            print(f"[mesh]   *** WARNING: normal growth ratio "
                  f"{meta['normal_growth']:.3f} > 1.2; raise nj or r_far. ***")
        if qual["min_orthogonality"] < 0.15:
            print("[mesh]   *** WARNING: min orthogonality < 0.2; raise "
                  "--smooth-sweeps or fall back to the ICEM route. ***")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Parametric 2D C-grid generator for NACA airfoils "
                    "-> native Fluent .msh",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--naca", required=True, help="4- or 5-digit designation, e.g. 0012 or 23012")
    ap.add_argument("--re", type=float, required=True, help="chord Reynolds number")
    ap.add_argument("--aoa", type=float, default=0.0, help="angle of attack, degrees")
    ap.add_argument("--level", type=int, default=2, choices=sorted(GRID_LEVELS),
                    help="grid level: 1 coarse, 2 medium (baseline), 3 fine")
    ap.add_argument("--out", type=Path, required=True, help="output .msh path")
    ap.add_argument("--chord", type=float, default=1.0, help="chord in metres")
    ap.add_argument("--rho", type=float, default=RHO_DEFAULT, help="density, kg/m^3")
    ap.add_argument("--mu", type=float, default=MU_DEFAULT, help="dynamic viscosity, Pa.s")
    ap.add_argument("--y-plus", type=float, default=0.25,
                    help="y+ target at the MEDIUM level; other levels scale by 1/r. "
                         "Lowered 0.6 -> 0.25 on 2026-09-03: the a-priori flat-plate "
                         "sizing under-predicts the LE/suction-peak y+ ~2.4x, so 0.6 "
                         "gave an achieved y+max of 2.1; 0.25 keeps achieved max < 1.")
    ap.add_argument("--r-far", type=float, default=30.0, help="far-field radius, chords")
    ap.add_argument("--x-out", type=float, default=30.0, help="outlet plane x, chords")
    ap.add_argument("--smooth-sweeps", type=int, default=0,
                    help="i-direction relaxation sweeps outside the BL stack. "
                         "OFF by default: it averages across the trailing-edge "
                         "fold, pulling the two branches together and making the "
                         "sliver cells there worse, not better.")
    ap.add_argument("--outer-uniformity", type=float, default=1.0,
                    help="0 = far-field spacing copies the wall clustering (folds), "
                         "1 = uniform per block (recommended)")
    ap.add_argument("--na", type=int, default=None, help="override: surface cells (even)")
    ap.add_argument("--nw", type=int, default=None, help="override: wake cells per branch")
    ap.add_argument("--nj", type=int, default=None, help="override: normal cells")
    ap.add_argument("--self-check", action="store_true",
                    help="re-parse the written .msh and assert consistency")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the parameter/quality record here")
    args = ap.parse_args(argv)

    rec = generate(args.naca, args.re, args.aoa, args.level, args.out,
                   chord=args.chord, rho=args.rho, mu=args.mu, y_plus=args.y_plus,
                   r_far=args.r_far, x_out=args.x_out,
                   smooth_sweeps=args.smooth_sweeps,
                   outer_uniformity=args.outer_uniformity,
                   overrides=dict(na=args.na, nw=args.nw, nj=args.nj))
    if args.self_check:
        chk = validate_msh(args.out)
        rec["self_check"] = chk
        print("[mesh]   self-check OK: %d nodes, %d cells, %d faces "
              "(%d boundary, %d interior)"
              % (chk["n_nodes"], chk["n_cells"], chk["n_faces"],
                 chk["n_boundary_faces"], chk["n_interior_faces"]))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"[mesh]   wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
