"""Parametric NACA airfoil pool for active learning (spec section 5.8).

The acquisition pool is "a parametric family of NACA 4 and 5 digit airfoils
crossed with a grid of (Re, alpha), none in the training set".  This module
generates that pool and emits batch dicts with the frozen surface keys of
CONTEXT.md section 4 (``surf_pos``, ``surf_normal``, ``surf_ds``, ``cond``,
``batch_idx``) so a trained surrogate can run inference on proposed geometries
without any further plumbing.

Conventions pinned here
-----------------------
* **Chord = 1**, leading edge at ``(0, 0)``, trailing edge at ``(1, 0)`` (mean
  line, before camber).  AirfRANS also uses chord = 1 (CONTEXT.md section 2).
* **Contour ordering / orientation**: points start at the trailing edge, run
  forward along the **upper** surface to the leading edge, then aft along the
  **lower** surface back to the trailing edge (the Selig ordering).  In the
  standard x-right / y-up frame this traversal is **counter-clockwise**
  (positive shoelace area) -- asserted by ``tests/test_active.py``.
  The contour is closed *cyclically*: the last point is not a duplicate of the
  first, edge ``N-1 -> 0`` closes it.  This satisfies CONTEXT.md section 4's
  "ordered along the airfoil contour (TE -> around -> TE)".
* **Outward normal** (pointing into the fluid) for a counter-clockwise contour
  is ``n = (t_y, -t_x)`` with ``t`` the unit tangent.  Sanity-checked against
  the outward radial direction at construction time.
* ``surf_ds[i]`` is the facet-length share of point ``i``: half the length of
  each of its two adjacent edges, so ``sum(ds) == perimeter`` exactly.

Formula references
------------------
4-digit thickness and camber, and 5-digit standard/reflexed camber, follow
Abbott & von Doenhoff, *Theory of Wing Sections* (Dover, 1959), sections 6.3-6.6
and Appendix I; the 5-digit ``(m, k1)`` tables are those of NACA TR-460 /
TR-537 as tabulated in Abbott & von Doenhoff Table 6.5 and reproduced in Ladson
et al., NASA TM-4741 (1996) "Computer Program to Obtain Ordinates for NACA
Airfoils".  Exact expressions are repeated in the docstrings below.
"""

from __future__ import annotations

import math
import re as _re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "NU_AIR",
    "NACA4",
    "NACA5",
    "AirfoilShape",
    "parse_naca",
    "cosine_spacing",
    "surface_geometry",
    "signed_area",
    "PoolEntry",
    "ExclusionSet",
    "naca4_family",
    "naca5_family",
    "default_pool_shapes",
    "reynolds_to_speed",
    "freestream_from_re",
    "build_pool",
    "entry_to_batch",
    "collate_batches",
    "design_matrix",
    "DESIGN_COLUMNS",
]

#: Kinematic viscosity of air at sea level, m^2/s.  Used only to map a target
#: Reynolds number to a freestream speed; override once A1 pins AirfRANS's own
#: convention in docs/DATA_NOTES.md.
NU_AIR = 1.5e-5

_TE_COEF_OPEN = 0.1015   # classic NACA quartic coefficient, blunt (open) TE
_TE_COEF_CLOSED = 0.1036  # adjusted so yt(1) == 0 exactly


# ==========================================================================
# shape families
# ==========================================================================
class AirfoilShape:
    """Common interface for the NACA families.

    Subclasses provide :meth:`camber_line` and :meth:`thickness`; everything
    else (coordinates, descriptors) is shared.
    """

    code: str
    t: float

    # -- to be provided by subclasses -------------------------------------
    def camber_line(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(yc, dyc_dx)`` on ``x in [0, 1]``."""
        raise NotImplementedError

    def thickness(self, x: np.ndarray, closed_te: bool = True) -> np.ndarray:
        """Half-thickness distribution ``yt(x)``.

        4- and 5-digit series share the same quartic::

            yt = (t/0.2) (0.2969 sqrt(x) - 0.1260 x - 0.3516 x^2
                          + 0.2843 x^3 - a4 x^4)

        with ``a4 = 0.1015`` for the original (blunt) trailing edge and
        ``a4 = 0.1036`` for a closed trailing edge (``yt(1) == 0``).
        """
        xa = np.asarray(x, dtype=np.float64)
        if np.any(xa < -1e-12) or np.any(xa > 1.0 + 1e-12):
            raise ValueError("x must lie in [0, 1]")
        xa = np.clip(xa, 0.0, 1.0)
        a4 = _TE_COEF_CLOSED if closed_te else _TE_COEF_OPEN
        return (self.t / 0.2) * (
            0.2969 * np.sqrt(xa)
            - 0.1260 * xa
            - 0.3516 * xa**2
            + 0.2843 * xa**3
            - a4 * xa**4
        )

    # -- shared ------------------------------------------------------------
    def coordinates(
        self,
        n_points: int = 200,
        *,
        closed_te: bool = True,
        distribution: str = "cosine",
    ) -> np.ndarray:
        """Closed airfoil contour, shape ``(N, 2)``, counter-clockwise from TE.

        ``n_points`` is the number of returned points; it is rounded up to an
        even number when ``closed_te`` is True so the two half-surfaces match.

        ``distribution``:
          * ``"cosine"``  -- ``x = (1 - cos(beta)) / 2``, ``beta`` uniform on
            ``[0, pi]``.  Clusters points at **both** the leading edge (high
            curvature) and the trailing edge (thin, sharp).
          * ``"halfcosine"`` -- ``x = 1 - cos(beta)``, ``beta`` uniform on
            ``[0, pi/2]``: clusters at the LE only.
          * ``"uniform"`` -- equal chordwise spacing (diagnostics only).
        """
        if n_points < 8:
            raise ValueError("need at least 8 contour points")
        n_side = n_points // 2 + 1
        xs = cosine_spacing(n_side, distribution=distribution)
        yc, dyc = self.camber_line(xs)
        yt = self.thickness(xs, closed_te=closed_te)
        theta = np.arctan(dyc)
        sin_t, cos_t = np.sin(theta), np.cos(theta)

        xu = xs - yt * sin_t
        yu = yc + yt * cos_t
        xl = xs + yt * sin_t
        yl = yc - yt * cos_t

        # TE -> LE along the upper surface, then LE -> TE along the lower one.
        # This is the Selig ordering and, in an x-right/y-up frame, traverses
        # the section counter-clockwise (positive shoelace area).
        upper = np.stack([xu[::-1], yu[::-1]], axis=1)    # x: 1 -> 0
        lower = np.stack([xl, yl], axis=1)                # x: 0 -> 1
        pts = np.concatenate([upper, lower[1:]], axis=0)  # drop duplicated LE
        if closed_te:
            pts = pts[:-1]  # drop the TE duplicate; contour closes cyclically
        return np.ascontiguousarray(pts, dtype=np.float64)

    # -- descriptors -------------------------------------------------------
    def max_camber(self, n: int = 2001) -> tuple[float, float]:
        """``(max |yc|, x at which it occurs)`` by dense sampling."""
        x = np.linspace(0.0, 1.0, n)
        yc, _ = self.camber_line(x)
        i = int(np.argmax(np.abs(yc)))
        return float(yc[i]), float(x[i])

    def max_thickness(self, n: int = 2001, closed_te: bool = True) -> tuple[float, float]:
        """``(max total thickness = 2*yt, x at which it occurs)``."""
        x = np.linspace(0.0, 1.0, n)
        yt = self.thickness(x, closed_te=closed_te)
        i = int(np.argmax(yt))
        return float(2.0 * yt[i]), float(x[i])

    def descriptor(self) -> dict[str, float]:
        """Family-agnostic shape descriptor used by the diversity metric."""
        camb, x_camb = self.max_camber()
        thick, x_thick = self.max_thickness()
        return {
            "max_camber": camb,
            "x_max_camber": x_camb if abs(camb) > 1e-9 else 0.0,
            "max_thickness": thick,
            "x_max_thickness": x_thick,
        }

    @property
    def is_symmetric(self) -> bool:
        camb, _ = self.max_camber()
        return abs(camb) < 1e-12

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.code})"


@dataclass(frozen=True)
class NACA4(AirfoilShape):
    """NACA 4-digit airfoil, e.g. ``2412`` -> ``m=0.02, p=0.4, t=0.12``.

    Camber line (Abbott & von Doenhoff sec. 6.4)::

        0 <= x < p :  yc = (m/p^2)     (2 p x - x^2)
                      dyc/dx = (2m/p^2)     (p - x)
        p <= x <= 1:  yc = (m/(1-p)^2) ((1 - 2p) + 2 p x - x^2)
                      dyc/dx = (2m/(1-p)^2) (p - x)

    A symmetric section (``m == 0`` or ``p == 0``, e.g. 0012) has ``yc == 0``.
    """

    m: float  # max camber, fraction of chord (first digit / 100)
    p: float  # chordwise position of max camber (second digit / 10)
    t: float  # max thickness, fraction of chord (last two digits / 100)

    def __post_init__(self) -> None:
        if not (0.0 <= self.m < 0.5):
            raise ValueError(f"max camber m={self.m} out of range")
        if not (0.0 <= self.p < 1.0):
            raise ValueError(f"camber position p={self.p} out of range")
        if not (0.0 < self.t < 0.5):
            raise ValueError(f"thickness t={self.t} out of range")
        if self.m > 0 and self.p <= 0:
            raise ValueError("cambered 4-digit section needs p > 0")

    @classmethod
    def from_code(cls, code: str) -> "NACA4":
        s = str(code).strip().upper().replace("NACA", "").strip()
        if len(s) != 4 or not s.isdigit():
            raise ValueError(f"not a 4-digit NACA code: {code!r}")
        return cls(m=int(s[0]) / 100.0, p=int(s[1]) / 10.0, t=int(s[2:]) / 100.0)

    @property
    def code(self) -> str:  # type: ignore[override]
        # A symmetric section carries no camber position, so the second digit
        # is 0 by convention: NACA 0012, never "0412".
        m_digit = int(round(self.m * 100))
        p_digit = 0 if m_digit == 0 else int(round(self.p * 10))
        return f"{m_digit:d}{p_digit:d}{int(round(self.t * 100)):02d}"

    @property
    def family(self) -> str:
        return "naca4"

    def camber_line(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xa = np.asarray(x, dtype=np.float64)
        yc = np.zeros_like(xa)
        dyc = np.zeros_like(xa)
        if self.m == 0.0 or self.p == 0.0:
            return yc, dyc
        m, p = self.m, self.p
        fore = xa < p
        aft = ~fore
        yc[fore] = (m / p**2) * (2 * p * xa[fore] - xa[fore] ** 2)
        dyc[fore] = (2 * m / p**2) * (p - xa[fore])
        yc[aft] = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * xa[aft] - xa[aft] ** 2)
        dyc[aft] = (2 * m / (1 - p) ** 2) * (p - xa[aft])
        return yc, dyc

    def params(self) -> dict[str, float]:
        return {"m": self.m, "p": self.p, "t": self.t}


# 5-digit camber tables.  ``p`` is the chordwise position of maximum camber
# (= second digit / 20); ``m`` is the camber-line parameter solving
# ``p = m (1 - sqrt(m/3))`` for the standard series; ``k1`` is scaled for a
# design lift coefficient of 0.3 (first digit 2).
#   Abbott & von Doenhoff Table 6.5 / NACA TR-537; see also NASA TM-4741.
_NACA5_STANDARD: tuple[tuple[float, float, float], ...] = (
    # (p,    m,      k1)
    (0.05, 0.0580, 361.400),
    (0.10, 0.1260, 51.640),
    (0.15, 0.2025, 15.957),
    (0.20, 0.2900, 6.643),
    (0.25, 0.3910, 3.230),
)

#: Reflexed series (third digit 1): (p, m, k1, k2_over_k1).
_NACA5_REFLEXED: tuple[tuple[float, float, float, float], ...] = (
    (0.10, 0.1300, 51.990, 0.000764),
    (0.15, 0.2170, 15.793, 0.006770),
    (0.20, 0.3180, 6.520, 0.030300),
    (0.25, 0.4410, 3.191, 0.135500),
)

_NACA5_DESIGN_CL_REF = 0.3  # the tables above are tabulated for Cl_design = 0.3


def _lookup_naca5(p: float, reflex: bool) -> tuple[float, float, float]:
    """Return ``(m, k1, k2_over_k1)`` for a max-camber position ``p``.

    Exact table hit for the standard digits (p in {0.05,...,0.25}); linear
    interpolation in ``p`` for off-table values, which only arise if a caller
    builds a non-standard section deliberately.
    """
    table = _NACA5_REFLEXED if reflex else _NACA5_STANDARD
    ps = np.array([row[0] for row in table])
    for row in table:
        if abs(row[0] - p) < 1e-12:
            return (row[1], row[2], row[3] if reflex else 0.0)
    if p < ps.min() - 1e-9 or p > ps.max() + 1e-9:
        raise ValueError(
            f"5-digit camber position p={p} outside the tabulated range "
            f"[{ps.min()}, {ps.max()}] for {'reflexed' if reflex else 'standard'} series"
        )
    m = float(np.interp(p, ps, [row[1] for row in table]))
    k1 = float(np.interp(p, ps, [row[2] for row in table]))
    k2r = float(np.interp(p, ps, [row[3] for row in table])) if reflex else 0.0
    return m, k1, k2r


@dataclass(frozen=True)
class NACA5(AirfoilShape):
    """NACA 5-digit airfoil, e.g. ``23012``.

    Digits ``L P Q TT``:
      * ``L`` -> design lift coefficient ``Cl_i = 0.15 L``
      * ``P`` -> position of max camber ``p = P / 20``
      * ``Q`` -> 0 = standard camber line, 1 = reflexed
      * ``TT`` -> thickness in percent chord

    **Standard** camber line (Q = 0), with ``m`` solving ``p = m(1-sqrt(m/3))``
    and ``k1`` from the table::

        0 <= x < m :  yc = (k1/6) (x^3 - 3 m x^2 + m^2 (3 - m) x)
                      dyc/dx = (k1/6) (3 x^2 - 6 m x + m^2 (3 - m))
        m <= x <= 1:  yc = (k1 m^3 / 6) (1 - x)
                      dyc/dx = -k1 m^3 / 6

    **Reflexed** camber line (Q = 1), with ``r = k2/k1`` from the table::

        0 <= x <= m: yc = (k1/6) [ (x-m)^3 - r (1-m)^3 x - m^3 x + m^3 ]
        m <  x <= 1: yc = (k1/6) [ r (x-m)^3 - r (1-m)^3 x - m^3 x + m^3 ]

    Both are tabulated for ``Cl_i = 0.3``; other ``L`` values scale ``k1``
    linearly by ``Cl_i / 0.3`` (thin-airfoil theory is linear in camber).
    """

    l_digit: int   # first digit; design Cl = 0.15 * l_digit
    p_digit: int   # second digit; p = p_digit / 20
    reflex: int    # third digit: 0 standard, 1 reflexed
    t: float       # thickness fraction

    def __post_init__(self) -> None:
        if self.l_digit not in range(0, 10):
            raise ValueError(f"first digit must be 0..9, got {self.l_digit}")
        if self.reflex not in (0, 1):
            raise ValueError("third digit (reflex) must be 0 or 1")
        if not (0.0 < self.t < 0.5):
            raise ValueError(f"thickness t={self.t} out of range")
        # validate the (p, reflex) combination against the tables
        _lookup_naca5(self.p_digit / 20.0, bool(self.reflex))

    @classmethod
    def from_code(cls, code: str) -> "NACA5":
        s = str(code).strip().upper().replace("NACA", "").strip()
        if len(s) != 5 or not s.isdigit():
            raise ValueError(f"not a 5-digit NACA code: {code!r}")
        return cls(l_digit=int(s[0]), p_digit=int(s[1]), reflex=int(s[2]), t=int(s[3:]) / 100.0)

    @property
    def code(self) -> str:  # type: ignore[override]
        return f"{self.l_digit:d}{self.p_digit:d}{self.reflex:d}{int(round(self.t * 100)):02d}"

    @property
    def family(self) -> str:
        return "naca5"

    @property
    def p(self) -> float:
        """Chordwise position of maximum camber."""
        return self.p_digit / 20.0

    @property
    def design_cl(self) -> float:
        return 0.15 * self.l_digit

    def _coefficients(self) -> tuple[float, float, float]:
        m, k1, k2r = _lookup_naca5(self.p, bool(self.reflex))
        scale = self.design_cl / _NACA5_DESIGN_CL_REF
        return m, k1 * scale, k2r

    def camber_line(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xa = np.asarray(x, dtype=np.float64)
        yc = np.zeros_like(xa)
        dyc = np.zeros_like(xa)
        if self.l_digit == 0:
            return yc, dyc
        m, k1, r = self._coefficients()
        if not self.reflex:
            fore = xa < m
            aft = ~fore
            yc[fore] = (k1 / 6.0) * (xa[fore] ** 3 - 3 * m * xa[fore] ** 2 + m**2 * (3 - m) * xa[fore])
            dyc[fore] = (k1 / 6.0) * (3 * xa[fore] ** 2 - 6 * m * xa[fore] + m**2 * (3 - m))
            yc[aft] = (k1 * m**3 / 6.0) * (1.0 - xa[aft])
            dyc[aft] = -(k1 * m**3) / 6.0
        else:
            c = k1 / 6.0
            lin = -r * (1 - m) ** 3 * xa - m**3 * xa + m**3
            dlin = -r * (1 - m) ** 3 - m**3
            fore = xa <= m
            aft = ~fore
            yc[fore] = c * ((xa[fore] - m) ** 3 + lin[fore])
            dyc[fore] = c * (3 * (xa[fore] - m) ** 2 + dlin)
            yc[aft] = c * (r * (xa[aft] - m) ** 3 + lin[aft])
            dyc[aft] = c * (3 * r * (xa[aft] - m) ** 2 + dlin)
        return yc, dyc

    def params(self) -> dict[str, float]:
        return {
            "design_cl": self.design_cl,
            "p": self.p,
            "reflex": float(self.reflex),
            "t": self.t,
        }


def parse_naca(code: str) -> AirfoilShape:
    """Dispatch ``"0012"`` / ``"NACA 23012"`` to :class:`NACA4` / :class:`NACA5`."""
    s = str(code).strip().upper().replace("NACA", "").strip()
    s = _re.sub(r"[^0-9]", "", s)
    if len(s) == 4:
        return NACA4.from_code(s)
    if len(s) == 5:
        return NACA5.from_code(s)
    raise ValueError(f"unrecognised NACA code {code!r} (need 4 or 5 digits)")


# ==========================================================================
# point distributions and discrete geometry
# ==========================================================================
def cosine_spacing(n: int, distribution: str = "cosine") -> np.ndarray:
    """Chordwise stations on ``[0, 1]``, ``n`` points, endpoints included."""
    if n < 2:
        raise ValueError("need at least 2 stations")
    if distribution == "cosine":
        beta = np.linspace(0.0, math.pi, n)
        return 0.5 * (1.0 - np.cos(beta))
    if distribution == "halfcosine":
        beta = np.linspace(0.0, 0.5 * math.pi, n)
        return 1.0 - np.cos(beta)
    if distribution == "uniform":
        return np.linspace(0.0, 1.0, n)
    raise ValueError(f"unknown distribution {distribution!r}")


def _own_surface_geometry(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Outward unit normals and facet-share weights for a closed CCW contour.

    ``ds[i] = (|e_{i-1}| + |e_i|) / 2`` with ``e_i = p_{i+1} - p_i`` taken
    cyclically, so ``sum(ds)`` equals the polygon perimeter exactly.  The normal
    at ``i`` is built from the ``ds``-weighted average of the two adjacent edge
    normals, which is the discrete outward normal consistent with that
    quadrature.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {p.shape}")
    n = p.shape[0]
    if n < 3:
        raise ValueError("need at least 3 contour points")
    edge = np.roll(p, -1, axis=0) - p              # e_i = p_{i+1} - p_i
    elen = np.linalg.norm(edge, axis=1)
    if np.any(elen <= 0):
        raise ValueError("contour has duplicate consecutive points")
    # Outward normal of edge i for a counter-clockwise polygon: (e_y, -e_x).
    edge_n = np.stack([edge[:, 1], -edge[:, 0]], axis=1) / elen[:, None]
    prev_len = np.roll(elen, 1)
    prev_n = np.roll(edge_n, 1, axis=0)
    nvec = prev_n * prev_len[:, None] + edge_n * elen[:, None]
    norm = np.linalg.norm(nvec, axis=1)
    degenerate = norm < 1e-14
    if np.any(degenerate):  # 180-degree cusp: fall back to the forward edge
        nvec[degenerate] = edge_n[degenerate]
        norm[degenerate] = 1.0
    nvec = nvec / norm[:, None]
    ds = 0.5 * (prev_len + elen)
    return np.ascontiguousarray(nvec), np.ascontiguousarray(ds)


def _geometry_module_normals(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Try A2's ``src.geometry`` helpers; return ``None`` if unavailable.

    A2 owns ``src/geometry/quadrature.py``, whose ``contour_quadrature(pos)``
    returns ``{"pos", "ds", "normal", "curvature", ...}`` with the same
    conventions pinned in this module's docstring (outward normals, ds =
    half-sum of adjacent facet lengths).  We still validate the result (matching
    point count, unit length, outward orientation) before trusting it; any
    mismatch falls through to :func:`_own_surface_geometry`.
    """
    try:
        import importlib

        qmod = importlib.import_module("src.geometry.quadrature")
    except Exception:
        return None
    qfn = getattr(qmod, "contour_quadrature", None)
    if not callable(qfn):
        return None
    try:
        res = qfn(points)
        nvec = np.asarray(res["normal"], dtype=np.float64)
        ds = np.asarray(res["ds"], dtype=np.float64).ravel()
    except Exception:
        return None
    if nvec.shape != points.shape or ds.shape[0] != points.shape[0]:
        return None  # e.g. a closing point was deduplicated; keep our own arrays
    lens = np.linalg.norm(nvec, axis=1)
    if not np.allclose(lens, 1.0, atol=1e-6):
        return None
    centroid = points.mean(axis=0)
    if np.mean(np.einsum("ij,ij->i", nvec, points - centroid) > 0) < 0.9:
        return None  # orientation convention differs; do not silently adopt it
    return nvec, ds


def surface_geometry(
    points: np.ndarray,
    *,
    prefer_geometry_module: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """``(normals (N,2) outward unit, ds (N,) facet shares)`` for a closed contour.

    Uses ``src/geometry`` when it exists and passes validation, otherwise the
    self-contained implementation above.
    """
    p = np.asarray(points, dtype=np.float64)
    if prefer_geometry_module:
        got = _geometry_module_normals(p)
        if got is not None:
            return got
    return _own_surface_geometry(p)


def signed_area(points: np.ndarray) -> float:
    """Shoelace signed area; positive for a counter-clockwise contour."""
    p = np.asarray(points, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# ==========================================================================
# pool
# ==========================================================================
def reynolds_to_speed(re: float, chord: float = 1.0, nu: float = NU_AIR) -> float:
    """``U = Re * nu / chord``."""
    if re <= 0:
        raise ValueError("Reynolds number must be positive")
    return float(re) * float(nu) / float(chord)


def freestream_from_re(
    re: float,
    aoa_deg: float,
    *,
    chord: float = 1.0,
    nu: float = NU_AIR,
    speed: float | None = None,
) -> np.ndarray:
    """Freestream vector ``cond = (u_inf_x, u_inf_y)`` (CONTEXT.md section 4).

    Angle of attack is encoded by rotating the freestream, not the geometry:
    ``cond = U (cos alpha, sin alpha)``, matching AirfRANS where ``cond``
    "encodes speed + AoA".
    """
    u = reynolds_to_speed(re, chord, nu) if speed is None else float(speed)
    a = math.radians(float(aoa_deg))
    return np.array([u * math.cos(a), u * math.sin(a)], dtype=np.float64)


@dataclass(frozen=True)
class PoolEntry:
    """One candidate case: a shape crossed with a flow condition."""

    shape: AirfoilShape
    re: float
    aoa_deg: float

    @property
    def code(self) -> str:
        return self.shape.code

    @property
    def family(self) -> str:
        return getattr(self.shape, "family", "naca")

    @property
    def key(self) -> str:
        return f"{self.family}{self.code}_re{self.re:.4g}_a{self.aoa_deg:+.3g}"

    def cond(self, *, chord: float = 1.0, nu: float = NU_AIR, speed: float | None = None) -> np.ndarray:
        return freestream_from_re(self.re, self.aoa_deg, chord=chord, nu=nu, speed=speed)

    def descriptor(self) -> dict[str, float]:
        d = self.shape.descriptor()
        d["re"] = float(self.re)
        d["aoa_deg"] = float(self.aoa_deg)
        return d

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "family": self.family,
            "code": self.code,
            "re": float(self.re),
            "aoa_deg": float(self.aoa_deg),
            "params": self.shape.params(),
        }


class ExclusionSet:
    """Training combinations that must not appear in the pool.

    Entries are ``(code, re, aoa_deg)`` triples, ``PoolEntry`` objects, or
    mappings with those keys.  Matching is tolerant on the continuous
    variables: a candidate is excluded when the code matches exactly and
    ``|Re - Re_train| <= re_tol`` and ``|alpha - alpha_train| <= aoa_tol``.
    Passing ``re=None``/``aoa=None`` in an entry excludes the whole shape.
    """

    def __init__(
        self,
        entries: Iterable[Any] = (),
        *,
        re_tol: float = 1.0e4,
        aoa_tol: float = 0.25,
    ) -> None:
        self.re_tol = float(re_tol)
        self.aoa_tol = float(aoa_tol)
        self._items: list[tuple[str, float | None, float | None]] = []
        for e in entries:
            self._items.append(self._normalize(e))

    @staticmethod
    def _normalize(e: Any) -> tuple[str, float | None, float | None]:
        if isinstance(e, PoolEntry):
            return (e.code, float(e.re), float(e.aoa_deg))
        if isinstance(e, Mapping):
            code = str(e.get("code") or e.get("shape") or e.get("naca"))
            re_v = e.get("re")
            ao = e.get("aoa_deg", e.get("aoa"))
            return (_clean_code(code), None if re_v is None else float(re_v), None if ao is None else float(ao))
        if isinstance(e, str):
            return (_clean_code(e), None, None)
        if isinstance(e, (tuple, list)):
            if len(e) == 1:
                return (_clean_code(str(e[0])), None, None)
            if len(e) == 3:
                return (
                    _clean_code(str(e[0])),
                    None if e[1] is None else float(e[1]),
                    None if e[2] is None else float(e[2]),
                )
        raise TypeError(f"cannot interpret exclusion entry {e!r}")

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, entry: PoolEntry) -> bool:
        return self.excludes(entry)

    def excludes(self, entry: PoolEntry) -> bool:
        code = entry.code
        for c, re_v, ao in self._items:
            if c != code:
                continue
            if re_v is not None and abs(re_v - entry.re) > self.re_tol:
                continue
            if ao is not None and abs(ao - entry.aoa_deg) > self.aoa_tol:
                continue
            return True
        return False


def _clean_code(code: str) -> str:
    return _re.sub(r"[^0-9]", "", str(code).upper().replace("NACA", ""))


def naca4_family(
    m_vals: Sequence[float] = (0.0, 0.02, 0.04),
    p_vals: Sequence[float] = (0.4,),
    t_vals: Sequence[float] = (0.10, 0.12, 0.15),
) -> list[NACA4]:
    """Cartesian product of 4-digit parameters.

    A symmetric section (``m == 0``) has no camber position, so it is emitted
    once with ``p = 0`` (NACA 00xx) instead of once per entry in ``p_vals``.
    """
    out: list[NACA4] = []
    seen: set[str] = set()
    for m in m_vals:
        ps = (0.0,) if m == 0.0 else tuple(p_vals)
        for p in ps:
            for t in t_vals:
                shape = NACA4(m=float(m), p=float(p), t=float(t))
                if shape.code not in seen:
                    seen.add(shape.code)
                    out.append(shape)
    return out


def naca5_family(
    l_vals: Sequence[int] = (2, 3),
    p_vals: Sequence[int] = (2, 3, 4),
    reflex_vals: Sequence[int] = (0,),
    t_vals: Sequence[float] = (0.12, 0.15),
) -> list[NACA5]:
    """Cartesian product of 5-digit parameters, skipping untabulated combos."""
    out: list[NACA5] = []
    seen: set[str] = set()
    for l in l_vals:
        for p in p_vals:
            for q in reflex_vals:
                for t in t_vals:
                    try:
                        shape = NACA5(l_digit=int(l), p_digit=int(p), reflex=int(q), t=float(t))
                    except ValueError:
                        continue  # (p, reflex) not in the NACA tables
                    if shape.code not in seen:
                        seen.add(shape.code)
                        out.append(shape)
    return out


def default_pool_shapes() -> list[AirfoilShape]:
    """A reasonable default shape pool spanning both families."""
    return [*naca4_family(), *naca5_family()]


def build_pool(
    shapes: Sequence[AirfoilShape | str] | None = None,
    re_grid: Sequence[float] = (2.0e6, 3.0e6, 4.0e6, 5.0e6, 6.0e6),
    aoa_grid: Sequence[float] = (-4.0, 0.0, 4.0, 8.0, 12.0),
    *,
    exclude: ExclusionSet | Iterable[Any] | None = None,
) -> list[PoolEntry]:
    """Shapes x (Re, AoA) grid, minus the training combinations.

    ``re_grid``/``aoa_grid`` default to the AirfRANS envelope (Re in [2e6, 6e6],
    AoA in [-5, +15] deg; CONTEXT.md section 2).
    """
    if shapes is None:
        shapes = default_pool_shapes()
    resolved = [parse_naca(s) if isinstance(s, str) else s for s in shapes]
    if not isinstance(exclude, ExclusionSet):
        exclude = ExclusionSet(exclude or ())
    pool: list[PoolEntry] = []
    for shape in resolved:
        for re_v in re_grid:
            for aoa in aoa_grid:
                entry = PoolEntry(shape=shape, re=float(re_v), aoa_deg=float(aoa))
                if exclude.excludes(entry):
                    continue
                pool.append(entry)
    return pool


# ==========================================================================
# batch emission (CONTEXT.md section 4 surface keys)
# ==========================================================================
def entry_to_batch(
    entry: PoolEntry,
    *,
    n_points: int = 200,
    closed_te: bool = True,
    distribution: str = "cosine",
    chord: float = 1.0,
    nu: float = NU_AIR,
    speed: float | None = None,
    as_torch: bool = False,
    dtype: Any = np.float32,
) -> dict[str, Any]:
    """Batch dict for one pool entry, ready for ``SurrogateBase.forward``.

    Keys (CONTEXT.md sections 4 and 7):
    ``surf_pos`` (Ns,2), ``surf_normal`` (Ns,2) outward unit, ``surf_ds`` (Ns,),
    ``cond`` (1,2), ``batch_idx`` (Ns,) int64 zeros.  Also carries non-tensor
    metadata under ``sim_name``, ``re``, ``aoa_deg`` for bookkeeping.
    """
    pts = entry.shape.coordinates(n_points, closed_te=closed_te, distribution=distribution)
    normals, ds = surface_geometry(pts)
    cond = entry.cond(chord=chord, nu=nu, speed=speed)[None, :]
    batch: dict[str, Any] = {
        "surf_pos": np.ascontiguousarray(pts, dtype=dtype),
        "surf_normal": np.ascontiguousarray(normals, dtype=dtype),
        "surf_ds": np.ascontiguousarray(ds, dtype=dtype),
        "cond": np.ascontiguousarray(cond, dtype=dtype),
        "batch_idx": np.zeros(pts.shape[0], dtype=np.int64),
        "sim_name": [entry.key],
        "re": np.asarray([entry.re], dtype=dtype),
        "aoa_deg": np.asarray([entry.aoa_deg], dtype=dtype),
    }
    return _maybe_torch(batch) if as_torch else batch


def collate_batches(batches: Sequence[Mapping[str, Any]], *, as_torch: bool = False) -> dict[str, Any]:
    """Concatenate per-entry batches PyG-style with a running ``batch_idx``.

    Mirrors ``src/data/airfrans_loader.py::collate`` (CONTEXT.md section 4,
    decision D-010): points are concatenated, never padded.
    """
    if not batches:
        raise ValueError("no batches to collate")
    pos, nrm, ds, cond, idx, names, re_v, aoa = [], [], [], [], [], [], [], []
    offset = 0
    for b in batches:
        p = np.asarray(b["surf_pos"])
        pos.append(p)
        nrm.append(np.asarray(b["surf_normal"]))
        ds.append(np.asarray(b["surf_ds"]).ravel())
        cond.append(np.asarray(b["cond"]).reshape(-1, 2))
        idx.append(np.full(p.shape[0], offset, dtype=np.int64))
        names.extend(b.get("sim_name", [f"item{offset}"]))
        if "re" in b:
            re_v.append(np.asarray(b["re"]).ravel())
        if "aoa_deg" in b:
            aoa.append(np.asarray(b["aoa_deg"]).ravel())
        offset += 1
    out: dict[str, Any] = {
        "surf_pos": np.concatenate(pos, axis=0),
        "surf_normal": np.concatenate(nrm, axis=0),
        "surf_ds": np.concatenate(ds, axis=0),
        "cond": np.concatenate(cond, axis=0),
        "batch_idx": np.concatenate(idx, axis=0),
        "sim_name": names,
    }
    if re_v:
        out["re"] = np.concatenate(re_v)
    if aoa:
        out["aoa_deg"] = np.concatenate(aoa)
    return _maybe_torch(out) if as_torch else out


def _maybe_torch(batch: dict[str, Any]) -> dict[str, Any]:
    """Convert array values to torch tensors (late import)."""
    import importlib

    torch = importlib.import_module("torch")
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(np.ascontiguousarray(v))
        else:
            out[k] = v
    return out


# ==========================================================================
# design space
# ==========================================================================
#: Columns of the normalized design space used for diversity selection.
DESIGN_COLUMNS: tuple[str, ...] = (
    "max_camber",
    "x_max_camber",
    "max_thickness",
    "x_max_thickness",
    "re",
    "aoa_deg",
)


def design_matrix(entries: Sequence[PoolEntry], columns: Sequence[str] = DESIGN_COLUMNS) -> np.ndarray:
    """``(n_pool, len(columns))`` raw (un-normalized) design vectors.

    Shape parameters are family-agnostic descriptors (max camber and its
    position, max thickness and its position) so 4- and 5-digit sections live
    in one comparable space; normalization happens in ``src/active/diversity.py``.
    """
    if not entries:
        raise ValueError("empty pool")
    rows = []
    for e in entries:
        d = e.descriptor()
        missing = [c for c in columns if c not in d]
        if missing:
            raise KeyError(f"descriptor missing columns {missing}")
        rows.append([float(d[c]) for c in columns])
    return np.asarray(rows, dtype=np.float64)
