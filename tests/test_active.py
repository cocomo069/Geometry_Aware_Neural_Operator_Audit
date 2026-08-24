"""CPU tests for src/active (NACA pool, acquisition score, diversity selection).

Synthetic geometry and synthetic scores only -- no dataset, no trained models.

Orientation convention under test (documented in src/active/pool.py):
contours start at the trailing edge, run forward along the **upper** surface to
the leading edge, then aft along the **lower** surface back to the TE (Selig
ordering), which is **counter-clockwise** / positive shoelace area, with
outward unit normals ``n = (t_y, -t_x)``.
"""

import json
import math

import numpy as np
import pytest

from src.active.acquisition import (
    AcquisitionResult,
    acquisition_score,
    random_scores,
    rank_pool,
    select_cases,
    select_top_k,
    variance_only_scores,
    write_case_list,
    zscore,
)
from src.active.diversity import (
    apply_normalization,
    coverage_radius,
    farthest_point_selection,
    greedy_score_diversity,
    min_pairwise_distance,
    normalize_design_space,
    select_top_k_diverse,
)
from src.active.pool import (
    DESIGN_COLUMNS,
    NACA4,
    NACA5,
    ExclusionSet,
    PoolEntry,
    build_pool,
    collate_batches,
    cosine_spacing,
    default_pool_shapes,
    design_matrix,
    entry_to_batch,
    freestream_from_re,
    naca4_family,
    naca5_family,
    parse_naca,
    reynolds_to_speed,
    signed_area,
    surface_geometry,
)

# ==========================================================================
# NACA 4-digit
# ==========================================================================


def test_naca0012_thickness_and_symmetry():
    """The canonical check: 0012 is 12% thick at ~30% chord and has no camber."""
    a = NACA4.from_code("0012")
    assert a.code == "0012"
    assert (a.m, a.p, a.t) == (0.0, 0.0, 0.12)  # symmetric -> no camber position

    thick, x_thick = a.max_thickness()
    assert thick == pytest.approx(0.12, abs=1e-3)
    assert x_thick == pytest.approx(0.30, abs=0.01)

    camber, _ = a.max_camber()
    assert camber == 0.0
    assert a.is_symmetric

    yc, dyc = a.camber_line(np.linspace(0, 1, 51))
    assert np.allclose(yc, 0.0) and np.allclose(dyc, 0.0)


@pytest.mark.parametrize("t", [0.06, 0.09, 0.12, 0.15, 0.18, 0.21])
def test_symmetric_thickness_scales_with_last_two_digits(t):
    a = NACA4(m=0.0, p=0.4, t=t)
    thick, x_thick = a.max_thickness()
    assert thick == pytest.approx(t, rel=2e-3)
    assert x_thick == pytest.approx(0.30, abs=0.01)


def test_naca0012_contour_is_mirror_symmetric():
    a = NACA4.from_code("0012")
    pts = a.coordinates(240)
    # upper half is the first 120 points (TE -> LE); lower half mirrors it
    upper = pts[:120]
    lower = pts[121:][::-1]
    assert np.allclose(upper[:, 0][1:], lower[:, 0], atol=1e-12)
    assert np.allclose(upper[:, 1][1:], -lower[:, 1], atol=1e-12)


def test_naca2412_camber():
    """2412: 2% camber at 40% chord, 12% thick."""
    a = NACA4.from_code("2412")
    camber, x_camber = a.max_camber()
    assert camber == pytest.approx(0.02, abs=1e-4)
    assert x_camber == pytest.approx(0.40, abs=0.01)
    assert a.max_thickness()[0] == pytest.approx(0.12, abs=1e-3)
    assert not a.is_symmetric
    # camber line is continuous and C1 across the knuckle at x = p
    x = np.array([0.4 - 1e-7, 0.4 + 1e-7])
    yc, dyc = a.camber_line(x)
    assert yc[0] == pytest.approx(yc[1], abs=1e-9)
    # dyc/dx -> 0 from both sides; the one-sided curvatures differ
    # (-2m/p^2 vs -2m/(1-p)^2), so both values are O(eps), not equal at O(eps^2)
    assert dyc[0] == pytest.approx(0.0, abs=1e-6)
    assert dyc[1] == pytest.approx(0.0, abs=1e-6)
    assert dyc[0] > 0.0 > dyc[1]  # sign change confirms this is the maximum


def test_naca4_rejects_bad_parameters():
    with pytest.raises(ValueError):
        NACA4(m=0.02, p=0.0, t=0.12)  # cambered but no camber position
    with pytest.raises(ValueError):
        NACA4(m=0.0, p=0.4, t=0.0)
    with pytest.raises(ValueError):
        NACA4.from_code("123")
    with pytest.raises(ValueError):
        NACA4.from_code("abcd")


# ==========================================================================
# NACA 5-digit
# ==========================================================================


def test_naca23012_standard_camber():
    """23012: design Cl = 0.3, max camber at 15% chord, ~1.8% camber, 12% thick."""
    a = NACA5.from_code("23012")
    assert a.code == "23012"
    assert (a.l_digit, a.p_digit, a.reflex) == (2, 3, 0)
    assert a.design_cl == pytest.approx(0.30)
    assert a.p == pytest.approx(0.15)

    camber, x_camber = a.max_camber()
    assert camber == pytest.approx(0.0183, abs=5e-4)  # published value ~1.8%
    assert x_camber == pytest.approx(0.15, abs=0.005)
    assert a.max_thickness()[0] == pytest.approx(0.12, abs=1e-3)


def test_naca5_camber_line_matches_closed_form():
    """Re-derive yc from the published (m, k1) table independently."""
    a = NACA5.from_code("23012")
    m, k1 = 0.2025, 15.957
    x = np.array([0.05, 0.10, 0.2025 - 1e-9, 0.3, 0.7])
    yc, dyc = a.camber_line(x)
    expect = np.where(
        x < m,
        (k1 / 6.0) * (x**3 - 3 * m * x**2 + m**2 * (3 - m) * x),
        (k1 * m**3 / 6.0) * (1.0 - x),
    )
    assert np.allclose(yc, expect, atol=1e-12)
    # aft of m the camber line is straight
    assert dyc[-1] == pytest.approx(-(k1 * m**3) / 6.0)
    assert dyc[-2] == pytest.approx(-(k1 * m**3) / 6.0)


def test_naca5_aft_camber_is_linear():
    a = NACA5.from_code("23012")
    x = np.linspace(0.3, 1.0, 40)
    yc, _ = a.camber_line(x)
    assert np.allclose(np.diff(yc, 2), 0.0, atol=1e-12)
    assert a.camber_line(np.array([1.0]))[0][0] == pytest.approx(0.0, abs=1e-12)


def test_naca5_design_cl_scales_camber_linearly():
    """First digit L scales k1 (and hence camber) linearly: 43012 vs 23012."""
    a2 = NACA5.from_code("23012")
    a4 = NACA5.from_code("43012")
    assert a4.design_cl == pytest.approx(0.60)
    assert a4.max_camber()[0] == pytest.approx(2.0 * a2.max_camber()[0], rel=1e-9)
    # zero design lift -> no camber at all
    assert NACA5.from_code("03012").max_camber()[0] == 0.0


def test_naca5_reflexed_camber():
    """Reflexed series (third digit 1) uses the k2/k1 table and unloads the TE."""
    std = NACA5.from_code("23012")
    ref = NACA5.from_code("23112")
    assert ref.reflex == 1
    # reflex bends the aft camber line back up: trailing-edge slope is positive
    _, d_std = std.camber_line(np.array([0.95]))
    _, d_ref = ref.camber_line(np.array([0.95]))
    assert d_std[0] < 0.0
    assert d_ref[0] > d_std[0]
    assert ref.max_camber()[1] == pytest.approx(0.15, abs=0.01)
    assert ref.max_thickness()[0] == pytest.approx(0.12, abs=1e-3)


def test_naca5_camber_line_is_continuous_at_m():
    for code in ("23012", "23112", "24012"):
        a = NACA5.from_code(code)
        m = a._coefficients()[0]
        x = np.array([m - 1e-8, m + 1e-8])
        yc, dyc = a.camber_line(x)
        assert yc[0] == pytest.approx(yc[1], abs=1e-8), code
        assert dyc[0] == pytest.approx(dyc[1], abs=1e-6), code


def test_naca5_rejects_untabulated_combinations():
    with pytest.raises(ValueError):
        NACA5(l_digit=2, p_digit=1, reflex=1, t=0.12)  # p=0.05 has no reflexed row
    with pytest.raises(ValueError):
        NACA5(l_digit=2, p_digit=9, reflex=0, t=0.12)  # p=0.45 off the table
    with pytest.raises(ValueError):
        NACA5(l_digit=2, p_digit=3, reflex=2, t=0.12)
    with pytest.raises(ValueError):
        NACA5.from_code("2301")


def test_parse_naca_dispatch():
    assert isinstance(parse_naca("0012"), NACA4)
    assert isinstance(parse_naca("NACA 23012"), NACA5)
    assert parse_naca("naca-2412").code == "2412"
    with pytest.raises(ValueError):
        parse_naca("123456")


# ==========================================================================
# discretization: closure, orientation, normals, quadrature
# ==========================================================================

ALL_CODES = ["0012", "2412", "4415", "23012", "23112", "43015"]


@pytest.mark.parametrize("code", ALL_CODES)
def test_contour_is_closed_and_counterclockwise(code):
    shape = parse_naca(code)
    pts = shape.coordinates(200)

    assert pts.shape == (200, 2)
    assert np.all(np.isfinite(pts))

    # starts and (cyclically) ends at the trailing edge, passes through the LE
    assert pts[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert pts[len(pts) // 2] == pytest.approx(np.array([0.0, 0.0]), abs=1e-12)

    # counter-clockwise: positive shoelace area, of the right magnitude
    area = signed_area(pts)
    assert area > 0.0, f"{code} is not counter-clockwise"
    assert 0.5 * shape.t < area < 1.0 * shape.t  # ~0.68*t*c^2 for NACA sections

    # closed cyclically: no duplicated endpoint, and the closing edge is not a gap
    assert not np.allclose(pts[0], pts[-1])
    edges = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    assert edges[-1] < 3.0 * np.median(edges), "closing edge is a gap, not a facet"
    assert np.all(edges > 0.0)


@pytest.mark.parametrize("code", ALL_CODES)
def test_normals_are_unit_and_outward(code):
    pts = parse_naca(code).coordinates(300)
    normals, ds = surface_geometry(pts)

    assert normals.shape == pts.shape
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12)

    # outward = positive dot product with the ray from the section centroid
    centroid = pts.mean(axis=0)
    assert np.all(np.einsum("ij,ij->i", normals, pts - centroid) > 0.0)

    # ds is a partition of the perimeter
    perim = float(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1).sum())
    assert ds.sum() == pytest.approx(perim, rel=1e-12)
    assert np.all(ds > 0.0)
    assert perim == pytest.approx(2.03, abs=0.15)  # ~2c for a thin section


def test_normals_agree_with_geometry_module_when_available():
    """If A2's src/geometry is present, our own computation must match it."""
    quad = pytest.importorskip("src.geometry.quadrature")
    pts = parse_naca("2412").coordinates(200)
    from src.active.pool import _own_surface_geometry

    mine_n, mine_ds = _own_surface_geometry(pts)
    theirs = quad.contour_quadrature(pts)
    assert np.allclose(mine_n, theirs["normal"], atol=1e-12)
    assert np.allclose(mine_ds, theirs["ds"], atol=1e-12)
    assert theirs["signed_area"] > 0.0


def test_closed_versus_open_trailing_edge():
    a = NACA4.from_code("0012")
    assert a.thickness(np.array([1.0]), closed_te=True)[0] == pytest.approx(0.0, abs=1e-12)
    open_te = a.thickness(np.array([1.0]), closed_te=False)[0]
    assert open_te > 0.0
    # yt(1) = (t/0.2) * 0.0021 = 0.0105 * t; the TE gap is 2 * yt(1)
    assert open_te == pytest.approx(0.0105 * 0.12, rel=1e-6)
    assert 2 * open_te == pytest.approx(0.00252, abs=2e-5)

    closed_pts = a.coordinates(200, closed_te=True)
    open_pts = a.coordinates(200, closed_te=False)
    assert closed_pts.shape[0] == 200
    assert open_pts.shape[0] == 201  # TE upper and lower are distinct points
    assert np.linalg.norm(open_pts[0] - open_pts[-1]) == pytest.approx(2 * open_te)
    assert signed_area(open_pts) > 0.0
    n, _ = surface_geometry(open_pts)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)


def test_cosine_spacing_clusters_at_both_edges():
    xs = cosine_spacing(101, "cosine")
    assert xs[0] == 0.0 and xs[-1] == pytest.approx(1.0)
    assert np.all(np.diff(xs) > 0)
    d = np.diff(xs)
    assert d[0] < 0.25 * d[len(d) // 2], "no clustering at the leading edge"
    assert d[-1] < 0.25 * d[len(d) // 2], "no clustering at the trailing edge"

    half = cosine_spacing(101, "halfcosine")
    dh = np.diff(half)
    assert dh[0] < dh[-1]  # LE clustering only

    uni = cosine_spacing(11, "uniform")
    assert np.allclose(np.diff(uni), 0.1)

    with pytest.raises(ValueError):
        cosine_spacing(101, "bogus")
    with pytest.raises(ValueError):
        cosine_spacing(1)


def test_le_points_are_denser_than_midchord_on_the_contour():
    pts = parse_naca("0012").coordinates(400)
    edges = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    le = len(pts) // 2
    quarter = len(pts) // 4
    assert edges[le] < edges[quarter]


def test_resolution_refinement_converges():
    """Perimeter and area converge as the contour is refined."""
    shape = parse_naca("2412")
    areas, perims = [], []
    for n in (100, 400, 1600):
        pts = shape.coordinates(n)
        areas.append(signed_area(pts))
        perims.append(surface_geometry(pts)[1].sum())
    assert abs(areas[2] - areas[1]) < abs(areas[1] - areas[0])
    assert abs(perims[2] - perims[1]) < abs(perims[1] - perims[0])


def test_coordinates_rejects_too_few_points():
    with pytest.raises(ValueError):
        parse_naca("0012").coordinates(4)


# ==========================================================================
# pool construction
# ==========================================================================


def test_families_and_default_pool():
    four = naca4_family()
    five = naca5_family()
    assert all(isinstance(s, NACA4) for s in four)
    assert all(isinstance(s, NACA5) for s in five)
    assert len({s.code for s in four}) == len(four)
    assert len({s.code for s in five}) == len(five)
    # symmetric sections are not duplicated across camber positions
    assert sum(1 for s in four if s.is_symmetric) == 3
    # untabulated (p, reflex) combos are skipped, not raised
    assert naca5_family(l_vals=(2,), p_vals=(1, 2, 3), reflex_vals=(1,), t_vals=(0.12,))
    assert len(default_pool_shapes()) == len(four) + len(five)


def test_build_pool_is_the_full_cross_product():
    shapes = [parse_naca("0012"), parse_naca("23012")]
    re_grid = (2e6, 4e6, 6e6)
    aoa_grid = (-4.0, 0.0, 8.0)
    pool = build_pool(shapes, re_grid, aoa_grid)
    assert len(pool) == 2 * 3 * 3
    assert len({e.key for e in pool}) == len(pool)
    assert {e.family for e in pool} == {"naca4", "naca5"}
    e = pool[0]
    assert isinstance(e, PoolEntry)
    d = e.as_dict()
    assert set(d) >= {"key", "family", "code", "re", "aoa_deg", "params"}
    assert json.loads(json.dumps(d))["code"] == e.code


def test_pool_accepts_string_codes():
    pool = build_pool(["0012", "23012"], (2e6,), (0.0,))
    assert [e.code for e in pool] == ["0012", "23012"]


def test_exclusion_list_removes_training_combinations():
    shapes = ["0012", "2412"]
    re_grid = (2e6, 4e6)
    aoa_grid = (0.0, 5.0)
    full = build_pool(shapes, re_grid, aoa_grid)
    assert len(full) == 8

    # exclude one exact combination
    pool = build_pool(shapes, re_grid, aoa_grid, exclude=[("0012", 2e6, 0.0)])
    assert len(pool) == 7
    assert not any(e.code == "0012" and e.re == 2e6 and e.aoa_deg == 0.0 for e in pool)

    # exclude a whole shape
    pool = build_pool(shapes, re_grid, aoa_grid, exclude=["2412"])
    assert len(pool) == 4
    assert all(e.code == "0012" for e in pool)

    # tolerant matching on Re / AoA
    ex = ExclusionSet([("0012", 2.0e6 + 5e3, 0.1)], re_tol=1e4, aoa_tol=0.25)
    assert len(build_pool(shapes, re_grid, aoa_grid, exclude=ex)) == 7
    strict = ExclusionSet([("0012", 2.0e6 + 5e3, 0.1)], re_tol=1.0, aoa_tol=1e-6)
    assert len(build_pool(shapes, re_grid, aoa_grid, exclude=strict)) == 8

    # PoolEntry objects and dicts are accepted directly
    assert len(build_pool(shapes, re_grid, aoa_grid, exclude=[full[0]])) == 7
    assert len(build_pool(shapes, re_grid, aoa_grid,
                          exclude=[{"code": "0012", "re": 2e6, "aoa": 0.0}])) == 7
    with pytest.raises(TypeError):
        ExclusionSet([3.14])


def test_freestream_encodes_speed_and_aoa():
    assert reynolds_to_speed(3.0e6, 1.0, 1.5e-5) == pytest.approx(45.0)
    cond = freestream_from_re(3.0e6, 0.0)
    assert cond[0] == pytest.approx(45.0) and cond[1] == pytest.approx(0.0)

    cond5 = freestream_from_re(3.0e6, 5.0)
    assert np.linalg.norm(cond5) == pytest.approx(45.0)
    assert math.degrees(math.atan2(cond5[1], cond5[0])) == pytest.approx(5.0)

    # explicit speed overrides the Re -> U map
    assert np.linalg.norm(freestream_from_re(3.0e6, 0.0, speed=10.0)) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        reynolds_to_speed(0.0)


# ==========================================================================
# batch emission (CONTEXT.md section 4 keys)
# ==========================================================================


def test_entry_to_batch_matches_frozen_surface_keys():
    entry = PoolEntry(shape=parse_naca("2412"), re=3.0e6, aoa_deg=5.0)
    batch = entry_to_batch(entry, n_points=160)

    for key in ("surf_pos", "surf_normal", "surf_ds", "cond", "batch_idx"):
        assert key in batch, key
    assert batch["surf_pos"].shape == (160, 2)
    assert batch["surf_normal"].shape == (160, 2)
    assert batch["surf_ds"].shape == (160,)
    assert batch["cond"].shape == (1, 2)
    assert batch["batch_idx"].shape == (160,)
    assert batch["batch_idx"].dtype == np.int64
    assert np.all(batch["batch_idx"] == 0)
    for key in ("surf_pos", "surf_normal", "surf_ds", "cond"):
        assert batch[key].dtype == np.float32
        assert batch[key].flags["C_CONTIGUOUS"]
    assert batch["sim_name"] == [entry.key]
    assert float(np.linalg.norm(batch["cond"])) == pytest.approx(45.0, rel=1e-5)


def test_collate_batches_concatenates_with_batch_idx():
    pool = build_pool(["0012", "2412"], (3e6,), (0.0, 5.0))
    batches = [entry_to_batch(e, n_points=100) for e in pool]
    out = collate_batches(batches)

    assert out["surf_pos"].shape == (400, 2)
    assert out["surf_ds"].shape == (400,)
    assert out["cond"].shape == (4, 2)
    assert out["batch_idx"].shape == (400,)
    assert np.array_equal(np.unique(out["batch_idx"]), np.arange(4))
    assert np.bincount(out["batch_idx"]).tolist() == [100] * 4
    assert len(out["sim_name"]) == 4
    assert out["re"].shape == (4,)
    with pytest.raises(ValueError):
        collate_batches([])


def test_entry_to_batch_as_torch():
    torch = pytest.importorskip("torch")
    entry = PoolEntry(shape=parse_naca("0012"), re=3e6, aoa_deg=0.0)
    batch = entry_to_batch(entry, n_points=64, as_torch=True)
    assert torch.is_tensor(batch["surf_pos"])
    assert batch["surf_pos"].dtype == torch.float32
    assert batch["batch_idx"].dtype == torch.int64
    assert isinstance(batch["sim_name"], list)


def test_design_matrix_is_family_agnostic():
    pool = build_pool(["0012", "2412", "23012"], (2e6, 6e6), (0.0, 8.0))
    x = design_matrix(pool)
    assert x.shape == (len(pool), len(DESIGN_COLUMNS))
    assert np.all(np.isfinite(x))
    # a symmetric section has zero camber and a zeroed camber position
    sym = [i for i, e in enumerate(pool) if e.code == "0012"]
    assert np.allclose(x[sym, 0], 0.0)
    assert np.allclose(x[sym, 1], 0.0)
    # thickness column is the real thickness
    assert np.allclose(x[:, 2], [e.shape.max_thickness()[0] for e in pool], atol=1e-3)
    with pytest.raises(ValueError):
        design_matrix([])


# ==========================================================================
# acquisition score
# ==========================================================================


def test_zscore_is_unit_variance_and_zero_mean():
    x = np.array([1.0, 2.0, 3.0, 10.0])
    z = zscore(x)
    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert z.std() == pytest.approx(1.0)


def test_zscore_of_constant_component_is_zero_not_nan():
    z = zscore(np.full(7, 3.5))
    assert np.all(z == 0.0)
    assert np.all(np.isfinite(z))
    assert np.all(zscore(np.zeros(5)) == 0.0)


def test_zscore_rejects_non_finite():
    with pytest.raises(ValueError):
        zscore(np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        zscore(np.array([1.0, np.inf]))
    with pytest.raises(ValueError):
        zscore(np.array([]))


@pytest.mark.parametrize("a_sig, a_fsc, a_sym", [(1e6, 1e-4, 3.0), (7.0, 7.0, 7.0)])
@pytest.mark.parametrize("b_sig, b_fsc, b_sym", [(0.0, 0.0, 0.0), (-5.0, 100.0, 0.25)])
def test_acquisition_is_invariant_to_affine_rescaling(a_sig, a_fsc, a_sym, b_sig, b_fsc, b_sym):
    """z-normalization makes a(g,c) invariant to positive affine rescaling.

    This is what lets sigma_CD (~1e-3), FSC (~1e-2) and L_sym (dimensionless) be
    summed without one silently dominating -- and it means the ranking cannot
    change if an upstream module changes units.
    """
    rng = np.random.default_rng(0)
    n = 120
    sigma_cd = np.abs(rng.normal(1e-3, 3e-4, n))
    fsc = np.abs(rng.normal(2e-2, 8e-3, n))
    l_sym = np.abs(rng.normal(0.4, 0.15, n))

    base = acquisition_score(sigma_cd, fsc, l_sym)
    scaled = acquisition_score(
        a_sig * sigma_cd + b_sig,
        a_fsc * fsc + b_fsc,
        a_sym * l_sym + b_sym,
    )
    assert np.allclose(base, scaled, atol=1e-9)
    assert np.array_equal(rank_pool(base), rank_pool(scaled))


def test_acquisition_sign_flips_under_negative_scaling():
    rng = np.random.default_rng(1)
    sigma_cd = np.abs(rng.normal(1.0, 0.3, 50))
    base = acquisition_score(sigma_cd)
    flipped = acquisition_score(-sigma_cd)
    assert np.allclose(base, -flipped, atol=1e-9)


def test_acquisition_is_the_weighted_sum_of_z_scores():
    rng = np.random.default_rng(2)
    n = 60
    sig, fsc, sym = (np.abs(rng.normal(1, 0.3, n)) for _ in range(3))
    expected = zscore(sig) + 2.0 * zscore(fsc) + 0.5 * zscore(sym)
    got = acquisition_score(sig, fsc, sym, beta=2.0, gamma=0.5)
    assert np.allclose(got, expected)

    # defaults are beta = gamma = 1 (spec section 5.8)
    assert np.allclose(
        acquisition_score(sig, fsc, sym),
        zscore(sig) + zscore(fsc) + zscore(sym),
    )


def test_acquisition_terms_are_optional():
    rng = np.random.default_rng(3)
    sig = np.abs(rng.normal(1, 0.3, 40))
    fsc = np.abs(rng.normal(1, 0.3, 40))
    assert np.allclose(acquisition_score(sig), zscore(sig))
    assert np.allclose(acquisition_score(sig, fsc), zscore(sig) + zscore(fsc))
    # zeroing the weights reproduces the variance-only baseline
    assert np.allclose(
        acquisition_score(sig, fsc, fsc, beta=0.0, gamma=0.0),
        variance_only_scores(sig),
    )


def test_acquisition_result_exposes_components():
    rng = np.random.default_rng(4)
    n = 30
    sig, fsc, sym = (np.abs(rng.normal(1, 0.3, n)) for _ in range(3))
    res = acquisition_score(sig, fsc, sym, return_result=True)
    assert isinstance(res, AcquisitionResult)
    assert res.n == n
    assert set(res.components) == {"sigma_cd", "fsc", "l_sym"}
    assert np.allclose(res.normalized["fsc"], zscore(fsc))
    assert res.top_k(5).tolist() == rank_pool(res.scores)[:5].tolist()
    table = res.as_table()
    assert len(table) == n and "z_sigma_cd" in table[0]


def test_acquisition_rejects_length_mismatch():
    with pytest.raises(ValueError):
        acquisition_score(np.ones(10), np.ones(9))
    with pytest.raises(ValueError):
        acquisition_score(np.ones(10), None, np.ones(11))
    with pytest.raises(ValueError):
        acquisition_score(np.array([]))


def test_rank_and_top_k_break_ties_by_index():
    scores = np.array([1.0, 3.0, 3.0, 2.0])
    assert rank_pool(scores).tolist() == [1, 2, 3, 0]
    assert select_top_k(scores, 2).tolist() == [1, 2]
    assert select_top_k(scores, 0).size == 0
    with pytest.raises(ValueError):
        select_top_k(scores, 5)


def test_random_baseline_is_seeded_and_reproducible():
    a = random_scores(50, seed=0)
    b = random_scores(50, seed=0)
    c = random_scores(50, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (50,)
    with pytest.raises(ValueError):
        random_scores(0)


# ==========================================================================
# diversity
# ==========================================================================


def test_normalize_design_space_minmax_and_zscore():
    x = np.array([[0.0, 100.0], [1.0, 300.0], [2.0, 200.0]])
    xn, stats = normalize_design_space(x, "minmax")
    assert np.allclose(xn.min(axis=0), 0.0)
    assert np.allclose(xn.max(axis=0), 1.0)
    assert np.allclose(apply_normalization(x, stats), xn)

    xz, _ = normalize_design_space(x, "zscore")
    assert np.allclose(xz.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(xz.std(axis=0), 1.0)

    xp, _ = normalize_design_space(x, "none")
    assert np.allclose(xp, x)


def test_normalize_handles_constant_columns():
    x = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    for method in ("minmax", "zscore"):
        xn, _ = normalize_design_space(x, method)
        assert np.all(np.isfinite(xn))
        assert np.allclose(xn[:, 1], 0.0)


def test_normalize_column_weights():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    xn, _ = normalize_design_space(x, "minmax", weights=[3.0, 0.0])
    assert np.allclose(xn[:, 0], [0.0, 3.0])
    assert np.allclose(xn[:, 1], 0.0)  # a zero-weighted column is ignored
    with pytest.raises(ValueError):
        normalize_design_space(x, "minmax", weights=[1.0])
    with pytest.raises(ValueError):
        normalize_design_space(np.array([[1.0, np.nan]]), "minmax")
    with pytest.raises(ValueError):
        normalize_design_space(np.zeros(5), "minmax")


def test_farthest_point_picks_the_corners_of_a_grid():
    """The headline diversity check: FPS spreads picks over a synthetic grid."""
    g = np.linspace(0.0, 1.0, 10)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)  # 100 points on [0,1]^2

    # seeded at the (0,0) corner, FPS must pick exactly the four corners
    sel = farthest_point_selection(pts, 4, start=0)
    chosen = pts[sel]
    corners = {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.1 - 1.1, 1.0)}
    got = {(round(p[0], 6), round(p[1], 6)) for p in chosen}
    assert got == {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)}
    assert min_pairwise_distance(chosen) == pytest.approx(1.0)

    # ... and it beats simply taking the first four grid points
    naive = pts[:4]
    assert min_pairwise_distance(chosen) > 8 * min_pairwise_distance(naive)
    assert coverage_radius(pts, sel) < coverage_radius(pts, np.arange(4))


def test_farthest_point_covers_the_space_as_k_grows():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 1.0, size=(400, 3))
    radii = [coverage_radius(pts, farthest_point_selection(pts, k)) for k in (2, 4, 8, 16)]
    assert radii == sorted(radii, reverse=True), radii
    assert radii[-1] < 0.6


def test_farthest_point_determinism_and_bounds():
    rng = np.random.default_rng(1)
    pts = rng.uniform(size=(50, 2))
    a = farthest_point_selection(pts, 5, start="medoid")
    b = farthest_point_selection(pts, 5, start="medoid")
    assert np.array_equal(a, b)
    assert len(set(a.tolist())) == 5  # no repeats
    assert farthest_point_selection(pts, 0).size == 0
    assert farthest_point_selection(pts, 50).size == 50
    with pytest.raises(ValueError):
        farthest_point_selection(pts, 51)
    with pytest.raises(ValueError):
        farthest_point_selection(pts, 3, start=999)
    # 'first' seeds with row 0
    assert farthest_point_selection(pts, 3, start="first")[0] == 0


def test_already_selected_pushes_picks_away():
    pts = np.array([[0.0, 0.0], [0.05, 0.0], [1.0, 1.0], [0.95, 1.0]])
    # with (0,0) already in the training set, the next pick is the far cluster
    sel = farthest_point_selection(pts, 1, start=1, already_selected=[0])
    assert sel.tolist() == [1]
    sel2 = farthest_point_selection(pts, 2, start=2, already_selected=[0])
    assert set(sel2.tolist()) == {2, 3}


def test_select_top_k_diverse_keeps_the_best_and_spreads_the_rest():
    g = np.linspace(0.0, 1.0, 10)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # scores decay away from a corner: plain top-k would pick 4 near-duplicates
    scores = -np.linalg.norm(pts - np.array([1.0, 1.0]), axis=1)
    plain = select_top_k(scores, 4)
    diverse = select_top_k_diverse(scores, pts, 4, oversample=6)

    assert len(set(diverse.tolist())) == 4
    assert diverse[0] == plain[0], "the argmax must survive"
    assert min_pairwise_distance(pts, diverse) > min_pairwise_distance(pts, plain)


def test_select_top_k_diverse_edge_cases():
    rng = np.random.default_rng(2)
    pts = rng.uniform(size=(20, 2))
    scores = rng.uniform(size=20)
    assert select_top_k_diverse(scores, pts, 0).size == 0
    assert select_top_k_diverse(scores, pts, 20).size == 20
    with pytest.raises(ValueError):
        select_top_k_diverse(scores, pts, 21)
    with pytest.raises(ValueError):
        select_top_k_diverse(scores[:5], pts, 3)
    with pytest.raises(ValueError):
        select_top_k_diverse(scores, pts, 3, oversample=0)


def test_greedy_score_diversity_trades_off_with_lambda():
    g = np.linspace(0.0, 1.0, 10)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    scores = -np.linalg.norm(pts - np.array([1.0, 1.0]), axis=1)

    exploit = greedy_score_diversity(scores, pts, 5, lam=0.0)
    explore = greedy_score_diversity(scores, pts, 5, lam=8.0)
    assert exploit.tolist() == select_top_k(scores, 5).tolist()
    assert min_pairwise_distance(pts, explore) > min_pairwise_distance(pts, exploit)
    assert len(set(explore.tolist())) == 5


def test_min_pairwise_distance_degenerate():
    assert min_pairwise_distance(np.zeros((1, 2))) == float("inf")
    assert min_pairwise_distance(np.array([[0.0, 0.0], [3.0, 4.0]])) == pytest.approx(5.0)


# ==========================================================================
# end-to-end selection on a real pool
# ==========================================================================


def test_select_cases_end_to_end(tmp_path):
    pool = build_pool(
        ["0012", "2412", "4415", "23012", "23112"],
        (2e6, 4e6, 6e6),
        (-4.0, 0.0, 4.0, 8.0, 12.0),
        exclude=[("0012", 4e6, 0.0)],
    )
    assert len(pool) == 5 * 3 * 5 - 1

    rng = np.random.default_rng(0)
    n = len(pool)
    sigma_cd = np.abs(rng.normal(1e-3, 4e-4, n))
    fsc = np.abs(rng.normal(2e-2, 1e-2, n))
    l_sym = np.abs(rng.normal(0.3, 0.2, n))
    scores = acquisition_score(sigma_cd, fsc, l_sym)

    cases = select_cases(pool, scores, 8)
    assert len(cases) == 8
    assert [c["rank"] for c in cases] == list(range(8))
    assert len({c["key"] for c in cases}) == 8
    assert cases[0]["pool_index"] == int(rank_pool(scores)[0])
    assert all("score" in c and "params" in c for c in cases)

    # picks are spread in design space
    x = design_matrix(pool)
    xn, _ = normalize_design_space(x, "minmax")
    idx = [c["pool_index"] for c in cases]
    assert min_pairwise_distance(xn, idx) > min_pairwise_distance(xn, select_top_k(scores, 8))

    # the three strategies the spec compares
    rand = select_cases(pool, random_scores(n, seed=3), 8)
    var = select_cases(pool, variance_only_scores(sigma_cd), 8)
    assert len(rand) == len(var) == 8

    path = write_case_list(cases, tmp_path / "cases_to_run.json", strategy="acquisition")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["strategy"] == "acquisition"
    assert payload["n_cases"] == 8
    assert payload["cases"][0]["code"] == cases[0]["code"]


def test_select_cases_without_diversity_is_plain_top_k():
    pool = build_pool(["0012", "2412"], (2e6, 4e6), (0.0, 5.0))
    scores = np.arange(len(pool), dtype=float)
    cases = select_cases(pool, scores, 3, diverse=False)
    assert [c["pool_index"] for c in cases] == select_top_k(scores, 3).tolist()
    with pytest.raises(ValueError):
        select_cases(pool, scores[:2], 2)


def test_selected_cases_can_be_turned_into_model_batches():
    """The selection output must round-trip straight into model inference."""
    pool = build_pool(["0012", "23012"], (3e6,), (0.0, 6.0))
    scores = np.arange(len(pool), dtype=float)
    cases = select_cases(pool, scores, 3)
    batches = [entry_to_batch(pool[c["pool_index"]], n_points=120) for c in cases]
    batch = collate_batches(batches)
    assert batch["surf_pos"].shape == (360, 2)
    assert batch["cond"].shape == (3, 2)
    assert np.array_equal(np.unique(batch["batch_idx"]), np.arange(3))
    # normals are still outward after collation, per-sim
    for i in range(3):
        m = batch["batch_idx"] == i
        p = batch["surf_pos"][m].astype(np.float64)
        nn = batch["surf_normal"][m].astype(np.float64)
        assert np.all(np.einsum("ij,ij->i", nn, p - p.mean(axis=0)) > 0)
