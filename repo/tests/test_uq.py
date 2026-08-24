"""CPU tests for src/uq (ensembles, split conformal, coverage diagnostics).

Synthetic data only -- no dataset, no checkpoints, no torch required.
"""

import json
import math

import numpy as np
import pytest

from src.uq.conformal import (
    CalibratedIntervals,
    SplitConformalCoefficients,
    SplitConformalField,
    aggregate_point_scores,
    conformal_quantile,
    conformal_quantile_index,
    nonconformity_scores,
    split_conformal,
)
from src.uq.coverage import (
    DEFAULT_LEVELS,
    UQ_SCHEMA_VERSION,
    coverage_with_ci,
    empirical_coverage,
    expected_calibration_error,
    field_coverage,
    make_uq_record,
    make_uq_report,
    reliability_curve,
    sweep_levels_coefficients,
    sweep_levels_field,
    validate_uq_report,
    width_stats,
    wilson_interval,
    write_uq_report,
)
from src.uq.ensembles import (
    EnsemblePredictor,
    EnsembleStats,
    ensemble_mean_std,
    propagate,
    stack_members,
)

# ==========================================================================
# synthetic heteroscedastic generator
# ==========================================================================


def make_heteroscedastic(n, rng, *, sigma_scale=1.0, mu_bias=0.0):
    """Ground truth y = mu(x) + sigma(x) * eps, plus a *miscalibrated* surrogate.

    The "ensemble" reports ``mu_hat = mu + bias`` and ``sigma_hat = sigma_scale
    * sigma``, i.e. it is biased and its spread is off by a constant factor.
    Split conformal must repair the marginal coverage regardless.
    """
    x = rng.uniform(0.0, 1.0, size=n)
    mu = np.sin(3.0 * x)
    sigma = 0.05 + 0.5 * x  # heteroscedastic: 10x spread across the domain
    y = mu + sigma * rng.standard_normal(n)
    return mu + mu_bias, sigma * sigma_scale, y


def combined_coverage_sd(nominal, n_cal, m_test):
    """Std of measured coverage from BOTH randomness sources.

    Split-conformal coverage is itself random: marginally it is
    ``Beta(k, n_cal + 1 - k)`` with ``k = ceil((n_cal+1)(1-alpha))``, so it has
    standard deviation ~1/sqrt(n_cal) *before* any test-set noise.  Measuring it
    on ``m_test`` points adds binomial noise.  A test that only used the
    binomial CI would be checking the wrong (too tight) interval.
    """
    alpha = 1.0 - nominal
    k = conformal_quantile_index(n_cal, alpha)
    n1 = n_cal + 1
    beta_mean = k / n1
    beta_var = k * (n1 - k) / (n1 * n1 * (n_cal + 2))
    binom_var = beta_mean * (1.0 - beta_mean) / m_test
    return beta_mean, math.sqrt(beta_var + binom_var)


# ==========================================================================
# quantile index / conformal quantile
# ==========================================================================


@pytest.mark.parametrize(
    "n, alpha, expected",
    [
        (9, 0.10, 9),      # ceil(10 * 0.90) = 9 -> the max of 9 scores
        (10, 0.10, 10),    # ceil(11 * 0.90) = 10 -> the max of 10 scores
        (8, 0.10, 9),      # ceil(9 * 0.90) = 9 > n -> infinite intervals
        (19, 0.10, 18),    # ceil(20 * 0.90) = 18
        (19, 0.05, 19),    # 20 * 0.95 == 19 exactly; must NOT round up to 20
        (100, 0.05, 96),   # ceil(101 * 0.95) = 96
        (200, 0.10, 181),
        (200, 0.20, 161),
        (1, 0.50, 1),      # ceil(2 * 0.5) = 1
        (1, 0.10, 2),      # a single calibration point cannot certify 90%
    ],
)
def test_quantile_index_edge_cases(n, alpha, expected):
    assert conformal_quantile_index(n, alpha) == expected


def test_quantile_index_no_float_creep():
    """(n+1)(1-alpha) must not be inflated by binary-float representation."""
    for n in range(2, 500):
        for alpha in (0.05, 0.1, 0.2, 0.25, 0.5):
            k = conformal_quantile_index(n, alpha)
            exact = math.ceil(round((n + 1) * (1 - alpha), 9))
            assert k == exact, (n, alpha, k, exact)


def test_quantile_index_rejects_bad_alpha():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            conformal_quantile_index(10, bad)
    with pytest.raises(ValueError):
        conformal_quantile_index(0, 0.1)


def test_conformal_quantile_takes_kth_smallest():
    scores = np.array([5.0, 1.0, 3.0, 2.0, 4.0, 9.0, 7.0, 6.0, 8.0])  # n = 9
    # k = 9 -> the maximum
    assert conformal_quantile(scores, 0.10) == 9.0
    # k = ceil(10 * 0.8) = 8 -> the 8th smallest = 8.0
    assert conformal_quantile(scores, 0.20) == 8.0
    # k = ceil(10 * 0.5) = 5 -> the 5th smallest = 5.0
    assert conformal_quantile(scores, 0.50) == 5.0


def test_conformal_quantile_infinite_when_undersized():
    scores = np.arange(8, dtype=float)
    assert conformal_quantile(scores, 0.10) == float("inf")
    ci = SplitConformalCoefficients().fit(
        np.zeros(8), np.ones(8), np.zeros(8), level=0.9
    )
    lo, hi = ci.predict(np.zeros(5), np.ones(5))
    assert np.all(np.isneginf(lo)) and np.all(np.isposinf(hi))
    assert empirical_coverage(np.zeros(5), lo, hi) == 1.0


# ==========================================================================
# nonconformity scores
# ==========================================================================


def test_normalized_and_absolute_scores():
    mu = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 1.0, 0.0])
    sigma = np.array([1.0, 2.0, 4.0])
    s_abs = nonconformity_scores(mu, None, y, score="absolute")
    assert np.allclose(s_abs, [1.0, 0.0, 2.0])
    s_norm = nonconformity_scores(mu, sigma, y, score="normalized", eps=0.0)
    assert np.allclose(s_norm, [1.0, 0.0, 0.5])


def test_score_input_validation():
    with pytest.raises(ValueError):
        nonconformity_scores(np.zeros(3), np.ones(3), np.zeros(4))
    with pytest.raises(ValueError):
        nonconformity_scores(np.zeros(3), None, np.zeros(3), score="normalized")
    with pytest.raises(ValueError):
        nonconformity_scores(np.zeros(3), -np.ones(3), np.zeros(3))
    with pytest.raises(ValueError):
        nonconformity_scores(np.zeros(3), np.ones(3), np.zeros(3), score="bogus")


def test_aggregate_point_scores():
    s = np.arange(1, 11, dtype=float)  # 1..10
    assert aggregate_point_scores(s, aggregation="max") == 10.0
    # method='higher' -> agg_q == 1.0 reduces exactly to max
    assert aggregate_point_scores(s, aggregation="quantile", agg_q=1.0) == 10.0
    assert aggregate_point_scores(s, aggregation="quantile", agg_q=0.9) == 10.0
    assert aggregate_point_scores(s, aggregation="quantile", agg_q=0.5) == 6.0
    with pytest.raises(ValueError):
        aggregate_point_scores(s, aggregation="mean")


# ==========================================================================
# THE coverage test (spec section 5.6)
# ==========================================================================


N_CAL = 200
M_TEST = 2000
N_REPS = 24


@pytest.mark.parametrize("alpha", [0.1, 0.2])
@pytest.mark.parametrize("score", ["normalized", "absolute"])
def test_coefficient_conformal_achieves_nominal_coverage(alpha, score):
    """Empirical coverage sits inside the CI of nominal for n_cal=200, m=2000.

    Checked two ways: every single repetition lands inside a 4-sigma band that
    accounts for calibration-set *and* test-set randomness, and the mean over
    repetitions lands inside the (much tighter) CI of the mean.
    """
    nominal = 1.0 - alpha
    mean_cov, sd = combined_coverage_sd(nominal, N_CAL, M_TEST)
    rng = np.random.default_rng(20260824)

    covs = []
    for _ in range(N_REPS):
        cal_mu, cal_sig, cal_y = make_heteroscedastic(
            N_CAL, rng, sigma_scale=0.4, mu_bias=0.03
        )
        te_mu, te_sig, te_y = make_heteroscedastic(
            M_TEST, rng, sigma_scale=0.4, mu_bias=0.03
        )
        ci = split_conformal(cal_mu, cal_sig, cal_y, level=nominal, score=score)
        lo, hi = ci.predict(te_mu, te_sig)
        covs.append(empirical_coverage(te_y, lo, hi))

    covs = np.asarray(covs)
    # (a) every repetition inside a 4-sigma combined band
    assert np.all(np.abs(covs - mean_cov) < 4.0 * sd), (
        f"{score} alpha={alpha}: coverages {covs.min():.4f}..{covs.max():.4f} "
        f"outside {mean_cov:.4f} +/- {4 * sd:.4f}"
    )
    # (b) the mean is within the CI of the mean
    sem = sd / math.sqrt(N_REPS)
    assert abs(covs.mean() - mean_cov) < 3.5 * sem, (
        f"{score} alpha={alpha}: mean coverage {covs.mean():.4f} vs expected "
        f"{mean_cov:.4f} +/- {3.5 * sem:.4f}"
    )
    # (c) conformal is conservative: the theoretical mean is at least nominal
    assert mean_cov >= nominal


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_single_draw_coverage_inside_binomial_ci(alpha):
    """One calibration draw: nominal must fall inside the test-set Wilson CI.

    The Wilson CI is widened by the calibration-set standard deviation, which
    is the dominant term at n_cal = 200 -- see :func:`combined_coverage_sd`.
    """
    nominal = 1.0 - alpha
    _, sd = combined_coverage_sd(nominal, N_CAL, M_TEST)
    rng = np.random.default_rng(7)
    cal = make_heteroscedastic(N_CAL, rng, sigma_scale=1.7)
    test = make_heteroscedastic(M_TEST, rng, sigma_scale=1.7)
    ci = split_conformal(cal[0], cal[1], cal[2], level=nominal)
    lo, hi = ci.predict(test[0], test[1])
    stats = coverage_with_ci(test[2], lo, hi)
    cal_slack = 2.5 * sd
    assert stats["coverage_ci_lo"] - cal_slack <= nominal <= stats["coverage_ci_hi"] + cal_slack


def test_normalized_score_is_adaptive_absolute_is_not():
    """Normalized intervals scale with sigma; absolute ones are constant-width."""
    rng = np.random.default_rng(11)
    cal = make_heteroscedastic(N_CAL, rng)
    te_mu, te_sig, _ = make_heteroscedastic(500, rng)

    norm = split_conformal(cal[0], cal[1], cal[2], 0.9, score="normalized")
    absol = split_conformal(cal[0], cal[1], cal[2], 0.9, score="absolute")

    lo_n, hi_n = norm.predict(te_mu, te_sig)
    lo_a, hi_a = absol.predict(te_mu, te_sig)
    w_n, w_a = hi_n - lo_n, hi_a - lo_a

    assert np.corrcoef(w_n, te_sig)[0, 1] > 0.99   # tracks sigma
    assert np.ptp(w_a) < 1e-12                     # constant width
    assert width_stats(lo_n, hi_n)["median_width"] > 0


def test_per_target_vs_pooled_coefficients():
    """Two coefficients (CL, CD) on very different scales."""
    rng = np.random.default_rng(3)
    n, m = 300, 1500
    scales = np.array([1.0, 0.02])  # CL ~ O(1), CD ~ O(0.02)

    def draw(k):
        sigma = np.abs(rng.normal(1.0, 0.3, size=(k, 2))) * scales
        mu = rng.normal(0.0, 1.0, size=(k, 2)) * scales
        y = mu + sigma * rng.standard_normal((k, 2))
        return mu, sigma, y

    cal_mu, cal_sig, cal_y = draw(n)
    te_mu, te_sig, te_y = draw(m)

    per = SplitConformalCoefficients(per_target=True).fit(
        cal_mu, cal_sig, cal_y, 0.9, target_names=("cl", "cd")
    )
    assert per.q_hat.shape == (2,)
    lo, hi = per.predict(te_mu, te_sig)
    for j, name in enumerate(("cl", "cd")):
        cov = empirical_coverage(te_y[:, j], lo[:, j], hi[:, j])
        assert 0.85 < cov < 0.95, f"{name} coverage {cov}"

    pooled = SplitConformalCoefficients(per_target=False).fit(cal_mu, cal_sig, cal_y, 0.9)
    assert pooled.q_hat.ndim == 0
    lo_p, hi_p = pooled.predict(te_mu, te_sig)
    # pooled = simultaneous over both coefficients -> joint coverage ~ 0.9,
    # hence each marginal is >= the per-target one
    joint = np.mean(np.all((te_y >= lo_p) & (te_y <= hi_p), axis=1))
    assert joint > 0.85
    assert float(pooled.q_hat) >= float(per.q_hat.min())


def test_one_dimensional_input_gives_scalar_qhat():
    rng = np.random.default_rng(5)
    cal = make_heteroscedastic(100, rng)
    ci = SplitConformalCoefficients().fit(cal[0], cal[1], cal[2], 0.9)
    assert ci.q_hat.ndim == 0
    lo, hi = ci.predict(np.zeros(4), np.ones(4))
    assert lo.shape == (4,) and hi.shape == (4,)


def test_level_must_be_a_probability():
    rng = np.random.default_rng(1)
    cal = make_heteroscedastic(50, rng)
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            SplitConformalCoefficients().fit(cal[0], cal[1], cal[2], bad)


# ==========================================================================
# field granularity
# ==========================================================================


def make_field_sims(n_sim, n_pts, rng, *, sigma_scale=1.0):
    """Per-simulation surface fields with a per-sim difficulty factor.

    The per-sim factor is what breaks within-sim exchangeability and is exactly
    why the field granularity has to be conformalized at the simulation level.
    """
    mus, sigs, ys = [], [], []
    for _ in range(n_sim):
        difficulty = rng.uniform(0.5, 2.5)
        s = np.linspace(0.0, 1.0, n_pts)
        mu = np.cos(4.0 * s)
        sigma = (0.05 + 0.3 * s) * sigma_scale
        y = mu + difficulty * sigma * rng.standard_normal(n_pts)
        mus.append(mu)
        sigs.append(sigma)
        ys.append(y)
    return mus, sigs, ys


@pytest.mark.parametrize("aggregation, agg_q", [("max", 0.9), ("quantile", 0.9)])
@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_field_conformal_sim_level_coverage(aggregation, agg_q, alpha):
    """Sim-level coverage of the field band matches its stated guarantee."""
    nominal = 1.0 - alpha
    rng = np.random.default_rng(202608)
    n_cal, n_test, n_pts = 200, 400, 120
    cal = make_field_sims(n_cal, n_pts, rng)
    test = make_field_sims(n_test, n_pts, rng)

    fitter = SplitConformalField(aggregation=aggregation, agg_q=agg_q)
    ci = fitter.fit(cal[0], cal[1], cal[2], nominal)
    assert ci.granularity == f"field_{aggregation}"
    assert "Not a pointwise guarantee" in ci.guarantee

    res = field_coverage(ci, test[0], test[1], test[2])
    _, sd = combined_coverage_sd(nominal, n_cal, n_test)
    assert abs(res["coverage"] - nominal) < 4.0 * sd, res
    assert 0.0 <= res["point_coverage"] <= 1.0
    # a simultaneous band must cover at least as many points as it does sims
    assert res["point_coverage"] >= res["coverage"] - 1e-9


def test_field_max_band_is_wider_than_quantile_band():
    rng = np.random.default_rng(42)
    cal = make_field_sims(150, 100, rng)
    q_max = SplitConformalField(aggregation="max").fit(cal[0], cal[1], cal[2], 0.9)
    q_q90 = SplitConformalField(aggregation="quantile", agg_q=0.9).fit(
        cal[0], cal[1], cal[2], 0.9
    )
    assert float(q_max.q_hat) > float(q_q90.q_hat)
    assert q_max.agg_q is None and q_q90.agg_q == 0.9


def test_field_accepts_ragged_2d_and_flat_plus_simidx():
    rng = np.random.default_rng(9)
    mus, sigs, ys = make_field_sims(40, 30, rng)
    fitter = SplitConformalField(aggregation="max")
    ci_list = fitter.fit(mus, sigs, ys, 0.9)

    arr = lambda xs: np.stack(xs, axis=0)
    ci_2d = fitter.fit(arr(mus), arr(sigs), arr(ys), 0.9)

    sim_idx = np.repeat(np.arange(40), 30)
    ci_flat = fitter.fit(
        np.concatenate(mus), np.concatenate(sigs), np.concatenate(ys), 0.9, sim_idx=sim_idx
    )
    assert float(ci_list.q_hat) == pytest.approx(float(ci_2d.q_hat))
    assert float(ci_list.q_hat) == pytest.approx(float(ci_flat.q_hat))

    # ragged point counts are fine too
    ragged_mu = [m[: 10 + i] for i, m in enumerate(mus)]
    ragged_sig = [s[: 10 + i] for i, s in enumerate(sigs)]
    ragged_y = [y[: 10 + i] for i, y in enumerate(ys)]
    assert np.isfinite(float(fitter.fit(ragged_mu, ragged_sig, ragged_y, 0.9).q_hat))


def test_field_requires_sigma_for_normalized_score():
    rng = np.random.default_rng(2)
    mus, sigs, ys = make_field_sims(10, 20, rng)
    with pytest.raises(ValueError):
        SplitConformalField(score="normalized").fit(mus, None, ys, 0.9)
    ci = SplitConformalField(score="absolute", aggregation="max").fit(mus, None, ys, 0.9)
    assert np.isfinite(float(ci.q_hat))


def test_calibrated_intervals_roundtrip():
    rng = np.random.default_rng(13)
    cal = make_heteroscedastic(100, rng)
    ci = SplitConformalCoefficients().fit(cal[0], cal[1], cal[2], 0.9)
    back = CalibratedIntervals.from_dict(json.loads(json.dumps(ci.to_dict())))
    assert float(back.q_hat) == pytest.approx(float(ci.q_hat))
    assert back.level == ci.level and back.score == ci.score
    a = ci.predict(cal[0], cal[1])
    b = back.predict(cal[0], cal[1])
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


# ==========================================================================
# coverage diagnostics
# ==========================================================================


def test_empirical_coverage_and_widths():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    lo = np.array([-1.0, 0.5, 5.0, 2.0])
    hi = np.array([1.0, 1.5, 6.0, 4.0])
    assert empirical_coverage(y, lo, hi) == 0.75
    w = width_stats(lo, hi)
    assert w["mean_width"] == pytest.approx(1.5)
    assert w["median_width"] == pytest.approx(1.5)
    assert w["max_width"] == pytest.approx(2.0)


def test_wilson_interval_properties():
    lo, hi = wilson_interval(90, 100, 0.95)
    assert 0.0 < lo < 0.9 < hi < 1.0
    # a wider CI for fewer samples
    lo2, hi2 = wilson_interval(9, 10, 0.95)
    assert (hi2 - lo2) > (hi - lo)
    # degenerate counts stay inside [0, 1]
    lo3, hi3 = wilson_interval(0, 50)
    assert lo3 == 0.0 and 0.0 < hi3 < 1.0
    lo4, hi4 = wilson_interval(50, 50)
    assert hi4 == 1.0 and 0.0 < lo4 < 1.0
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_expected_calibration_error():
    nominal = [0.8, 0.9, 0.95]
    assert expected_calibration_error(nominal, nominal) == 0.0
    ece = expected_calibration_error(nominal, [0.7, 0.8, 0.85])
    assert ece == pytest.approx(0.1)
    with pytest.raises(ValueError):
        expected_calibration_error([0.9], [0.9, 0.8])


def test_reliability_curve_is_sorted():
    rows = [
        {"nominal": 0.9, "coverage": 0.91},
        {"nominal": 0.8, "coverage": 0.79},
        {"nominal": 0.95, "coverage": 0.96},
    ]
    rel = reliability_curve(rows)
    assert rel["nominal"] == [0.8, 0.9, 0.95]
    assert rel["empirical"] == [0.79, 0.91, 0.96]


def test_level_sweep_is_monotone_in_width_and_coverage():
    rng = np.random.default_rng(17)
    cal = make_heteroscedastic(400, rng)
    test = make_heteroscedastic(3000, rng)
    rows = sweep_levels_coefficients(
        cal[0], cal[1], cal[2], test[0], test[1], test[2], levels=DEFAULT_LEVELS
    )
    assert [r["nominal"] for r in rows] == list(DEFAULT_LEVELS)
    widths = [r["mean_width"] for r in rows]
    covs = [r["coverage"] for r in rows]
    assert widths == sorted(widths), "higher nominal level must not shrink the interval"
    assert covs == sorted(covs)
    for r in rows:
        assert r["coverage_ci_lo"] <= r["coverage"] <= r["coverage_ci_hi"]
        assert r["n"] == 3000


def test_field_level_sweep():
    rng = np.random.default_rng(23)
    cal = make_field_sims(120, 60, rng)
    test = make_field_sims(150, 60, rng)
    rows = sweep_levels_field(
        cal[0], cal[1], cal[2], test[0], test[1], test[2],
        levels=DEFAULT_LEVELS, aggregation="quantile", agg_q=0.9,
    )
    assert len(rows) == 3
    assert all("point_coverage" in r for r in rows)
    assert [r["mean_width"] for r in rows] == sorted(r["mean_width"] for r in rows)


# ==========================================================================
# results/uq schema
# ==========================================================================


def test_uq_record_and_report_roundtrip(tmp_path):
    rng = np.random.default_rng(31)
    cal = make_heteroscedastic(300, rng)
    test = make_heteroscedastic(1200, rng)
    rows = sweep_levels_coefficients(cal[0], cal[1], cal[2], test[0], test[1], test[2])
    rec = make_uq_record("cd", "coefficient", rows, score="normalized", n_cal=300)
    assert rec["n_test"] == 1200
    assert 0.0 <= rec["ece"] < 0.2
    assert len(rec["levels"]) == 3

    report = make_uq_report(
        "gnn_full_K5", "full", [rec], model="gnn", seeds=[0, 1, 2, 3, 4], k=5
    )
    assert report["schema_version"] == UQ_SCHEMA_VERSION
    path = write_uq_report(report, tmp_path)
    assert path.name == "gnn_full_K5.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    validate_uq_report(loaded)
    assert loaded["k"] == 5


def test_uq_report_validation_rejects_malformed():
    rows = [
        {
            "nominal": 0.9, "alpha": 0.1, "q_hat": 1.6, "coverage": 0.9,
            "coverage_ci_lo": 0.88, "coverage_ci_hi": 0.92,
            "mean_width": 0.5, "median_width": 0.5, "n": 100,
        }
    ]
    rec = make_uq_record("cd", "coefficient", rows)
    good = make_uq_report("e1", "full", [rec])

    bad = dict(good, schema_version="nope")
    with pytest.raises(ValueError):
        validate_uq_report(bad)

    with pytest.raises(ValueError):
        validate_uq_report(dict(good, records=[]))

    broken = json.loads(json.dumps(good))
    broken["records"][0]["granularity"] = "pointwise"
    with pytest.raises(ValueError):
        validate_uq_report(broken)

    broken2 = json.loads(json.dumps(good))
    broken2["records"][0]["levels"][0]["coverage"] = 1.4
    with pytest.raises(ValueError):
        validate_uq_report(broken2)

    broken3 = json.loads(json.dumps(good))
    del broken3["records"][0]["reliability"]
    with pytest.raises(ValueError):
        validate_uq_report(broken3)


# ==========================================================================
# ensembles
# ==========================================================================


def test_ensemble_mean_std_uses_ddof_one():
    members = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    mean, std = ensemble_mean_std(members)
    assert np.allclose(mean, [3.0, 4.0])
    assert np.allclose(std, [2.0, 2.0])  # sample std of {1,3,5} = 2
    _, std0 = ensemble_mean_std(members, ddof=0)
    assert np.allclose(std0, np.sqrt(8.0 / 3.0))


def test_single_member_has_zero_spread_not_nan():
    mean, std = ensemble_mean_std([np.array([1.0, 2.0])])
    assert np.allclose(mean, [1.0, 2.0])
    assert np.allclose(std, 0.0)
    assert np.all(np.isfinite(std))


def test_stack_members_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        stack_members([np.zeros(3), np.zeros(4)])
    with pytest.raises(ValueError):
        stack_members([])


def test_ensemble_predictor_from_arrays():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(5, 40))          # K=5 members, 40 surface points
    coef = rng.normal(size=(5, 3, 2))     # K=5, B=3 sims, (CL, CD)
    ens = EnsemblePredictor.from_arrays({"p": p, "coef_head": coef})
    assert ens.k == 5
    stats = ens.predict()
    assert isinstance(stats["p"], EnsembleStats)
    assert stats["p"].mean.shape == (40,)     # per-point
    assert stats["coef_head"].std.shape == (3, 2)  # per-sim, per-coefficient
    assert np.allclose(stats["p"].mean, p.mean(axis=0))
    assert np.allclose(stats["coef_head"].std, coef.std(axis=0, ddof=1))
    flat = ens.predict_mean_std()
    assert set(flat["p"]) == {"mean", "std"}


def test_ensemble_predictor_wraps_callables():
    class FakeModel:
        def __init__(self, offset):
            self.offset = offset
            self.eval_called = False

        def eval(self):
            self.eval_called = True

        def __call__(self, batch):
            n = batch["surf_pos"].shape[0]
            b = batch["cond"].shape[0]
            return {
                "p": np.full(n, self.offset),
                "tau": np.full((n, 2), self.offset),
                "coef_head": np.full((b, 2), self.offset),
            }

    models = [FakeModel(o) for o in (1.0, 2.0, 3.0)]
    batch = {
        "surf_pos": np.zeros((10, 2)),
        "surf_normal": np.zeros((10, 2)),
        "surf_ds": np.ones(10),
        "cond": np.ones((2, 2)),
        "batch_idx": np.array([0] * 5 + [1] * 5),
    }
    ens = EnsemblePredictor(models)
    stats = ens.predict(batch)
    assert np.allclose(stats["p"].mean, 2.0)
    assert np.allclose(stats["p"].std, 1.0)
    assert stats["coef_head"].mean.shape == (2, 2)
    assert all(m.eval_called for m in models)


def test_force_propagation_integrates_then_averages():
    """Cbar_D = mean_k C_D[u_k], not C_D[mean_k u_k] -- checked on a nonlinear probe."""
    n, b = 6, 2
    batch = {
        "surf_pos": np.zeros((n, 2)),
        "surf_normal": np.tile(np.array([1.0, 0.0]), (n, 1)),
        "surf_ds": np.ones(n),
        "cond": np.tile(np.array([1.0, 0.0]), (b, 1)),
        "batch_idx": np.array([0, 0, 0, 1, 1, 1]),
    }
    members = np.array([1.0, 2.0, 6.0])  # constant p per member
    ens = EnsemblePredictor.from_arrays(
        {
            "p": np.stack([np.full(n, v) for v in members]),
            "tau": np.zeros((3, n, 2)),
        }
    )

    def fake_integrate(p, tau, normal, ds, cond, batch_idx=None, **kw):
        """Linear drag, plus a deliberately nonlinear 'cd_sq' probe."""
        b_ = cond.shape[0]
        out_cd = np.zeros(b_)
        for i in range(b_):
            m = batch_idx == i
            out_cd[i] = np.sum(p[m] * normal[m, 0] * ds[m])
        return {"cd": out_cd, "cd_sq": out_cd**2}

    res = ens.propagate_forces(batch, fake_integrate)
    # each sim has 3 points with ds=1, n_x=1 -> cd = 3 * p
    assert np.allclose(res["cd"].members[:, 0], 3.0 * members)
    assert np.allclose(res["cd"].mean, 3.0 * members.mean())
    assert np.allclose(res["cd"].std, (3.0 * members).std(ddof=1))
    # integrate-then-average differs from average-then-integrate for cd_sq
    assert not np.isclose(res["cd_sq"].mean[0], (3.0 * members.mean()) ** 2)
    assert np.allclose(res["cd_sq"].mean, np.mean((3.0 * members) ** 2))


def test_force_propagation_requires_geometry():
    ens = EnsemblePredictor.from_arrays({"p": np.zeros((2, 4)), "tau": np.zeros((2, 4, 2))})
    with pytest.raises(KeyError):
        ens.propagate_forces({"surf_pos": np.zeros((4, 2))}, lambda *a, **k: {"cd": np.zeros(1)})


def test_propagate_helper():
    members = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    st = propagate(members, lambda m: m.sum())
    assert np.allclose(st.members, [3.0, 7.0])
    assert st.mean == pytest.approx(5.0)


def test_ensemble_accepts_torch_tensors():
    torch = pytest.importorskip("torch")
    members = [torch.ones(4), torch.ones(4) * 3.0]
    mean, std = ensemble_mean_std(members)
    assert np.allclose(mean, 2.0)
    assert np.allclose(std, np.sqrt(2.0))


def test_conformal_accepts_torch_tensors():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    mu, sig, y = make_heteroscedastic(60, rng)
    ci_np = SplitConformalCoefficients().fit(mu, sig, y, 0.9)
    ci_pt = SplitConformalCoefficients().fit(
        torch.from_numpy(mu), torch.from_numpy(sig), torch.from_numpy(y), 0.9
    )
    assert float(ci_np.q_hat) == pytest.approx(float(ci_pt.q_hat))


def test_uq_report_also_satisfies_eval_schema():
    """One payload must validate against BOTH uq-1 and src/eval/schema.py."""
    schema = pytest.importorskip("src.eval.schema")
    rng = np.random.default_rng(77)
    cal = make_heteroscedastic(200, rng)
    test = make_heteroscedastic(800, rng)
    coef_rows = sweep_levels_coefficients(cal[0], cal[1], cal[2], test[0], test[1], test[2])

    fcal = make_field_sims(80, 40, rng)
    ftest = make_field_sims(80, 40, rng)
    field_rows = sweep_levels_field(
        fcal[0], fcal[1], fcal[2], ftest[0], ftest[1], ftest[2], aggregation="max"
    )

    report = make_uq_report(
        "gnn_full_K5",
        "full",
        [
            make_uq_record("cd", "coefficient", coef_rows, n_cal=200),
            make_uq_record("p", "field_max", field_rows, n_cal=80),
        ],
        model="gnn",
        seeds=[0, 1, 2, 3, 4],
    )
    assert schema.validate_uq(report) == []
    schema.assert_valid_uq(report)

    # the flat view A4 reads must agree with our records
    assert report["levels"] == [0.8, 0.9, 0.95]
    assert set(report["coef"]) == {"cd"}
    assert set(report["field"]) == {"p"}
    assert report["coef"]["cd"]["0.9"]["coverage"] == pytest.approx(
        report["records"][0]["levels"][1]["coverage"]
    )


def test_build_model_from_config_uses_the_models_factory():
    """The checkpoint loader must match A3's build_model(name, config) signature."""
    models = pytest.importorskip("src.models")
    from src.uq.ensembles import build_model_from_config

    cfg = {"model": {"name": "gnn", "hidden_dim": 16, "layers": 1}}
    model = build_model_from_config(cfg)
    assert isinstance(model, models.MODEL_REGISTRY["gnn"])

    # aliases resolve through A3's registry too
    assert isinstance(build_model_from_config({"model": {"name": "m1"}}),
                      models.MODEL_REGISTRY["gnn"])

    with pytest.raises((KeyError, ValueError)):
        build_model_from_config({"model": {"name": "no_such_model"}})
    with pytest.raises(KeyError):
        build_model_from_config({"train": {"lr": 1e-3}})


def test_load_ensemble_checkpoints_roundtrip(tmp_path):
    """K checkpoints of one architecture load into a working EnsemblePredictor."""
    torch = pytest.importorskip("torch")
    models = pytest.importorskip("src.models")
    from src.uq.ensembles import load_ensemble_checkpoints

    cfg = {"model": {"name": "gnn", "hidden_dim": 16, "layers": 1}}
    paths = []
    for seed in range(3):
        torch.manual_seed(seed)
        m = models.build_model("gnn", {"hidden_dim": 16, "layers": 1})
        d = tmp_path / f"gnn_full_s{seed}"
        d.mkdir()
        p = d / "best.pt"
        torch.save({"model": m.state_dict(), "epoch": 1}, p)
        paths.append(p)

    ens = load_ensemble_checkpoints(paths, cfg)
    assert ens.k == 3
    assert ens.member_ids == ["gnn_full_s0", "gnn_full_s1", "gnn_full_s2"]

    with pytest.raises(FileNotFoundError):
        load_ensemble_checkpoints([tmp_path / "nope.pt"], cfg)
    with pytest.raises(ValueError):
        load_ensemble_checkpoints([], cfg)


def test_end_to_end_pool_ensemble_forces_acquisition():
    """Pool geometry -> K models -> ensemble -> force integration -> sigma_CD.

    Exercises the real A2 integrator and A3 models against the A5 pool/ensemble
    plumbing, which is the exact path the active-learning loop takes on
    unlabelled proposed geometries (spec section 5.8).
    """
    torch = pytest.importorskip("torch")
    models_pkg = pytest.importorskip("src.models")
    fi = pytest.importorskip("src.physics.force_integration")

    from src.active.acquisition import acquisition_score
    from src.active.pool import build_pool, collate_batches, entry_to_batch

    pool = build_pool(["0012", "2412", "23012"], (3e6,), (0.0, 5.0))
    batch = collate_batches(
        [entry_to_batch(e, n_points=128, as_torch=True) for e in pool], as_torch=True
    )
    assert batch["surf_pos"].shape == (6 * 128, 2)
    assert batch["cond"].shape == (6, 2)

    members = []
    for seed in range(3):
        torch.manual_seed(seed)
        members.append(models_pkg.build_model("gnn", {"hidden_dim": 16, "layers": 2}))
    ens = EnsemblePredictor(members)

    stats = ens.predict(batch)
    assert stats["p"].mean.shape == (6 * 128,)        # per surface point
    assert stats["coef_head"].mean.shape == (6, 2)    # per sim, (CL, CD)

    forces = ens.propagate_forces(batch, fi.integrate_forces)
    assert {"cl", "cd"} <= set(forces)
    assert forces["cd"].mean.shape == (6,)
    assert forces["cd"].members.shape == (3, 6)
    assert np.all(forces["cd"].std >= 0.0)
    assert np.all(np.isfinite(forces["cd"].mean))

    # sigma_CD feeds the acquisition score directly
    sigma_cd = forces["cd"].std
    fsc = np.abs(forces["cd"].mean - stats["coef_head"].mean[:, 1])
    scores = acquisition_score(sigma_cd, fsc)
    assert scores.shape == (6,)
    assert np.all(np.isfinite(scores))
