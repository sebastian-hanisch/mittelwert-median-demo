"""Orakel James-Stein und Theorie: das exakte Risiko gegen `scipy.stats.ncx2`-Quadratur, die Varianz des Medians gegen `scipy.integrate`, Lognormal-Quantile gegen `scipy.stats.lognorm`,
Expektilen gegen `scipy.optimize` mit Quadratur. Dazu Handinstanzen und Zweig-Tests."""

import math

import numpy as np
import pytest

import mtw_estimators as E
import mtw_stein as J
import mtw_theory as T
from mtw_scenario import Stream, draw, warehouse_observations

scipy_stats = pytest.importorskip("scipy.stats")
integrate = pytest.importorskip("scipy.integrate")
optimize = pytest.importorskip("scipy.optimize")


# ---- James-Stein -----------------------------------------------------------------------------------------------------------------------------

def test_exact_risk_equals_noncentral_chi_square_quadrature():
    for d in (3, 5, 10, 50):
        for lam in (0.25, 4.0, 25.0, 100.0):
            val, _ = integrate.quad(lambda t: scipy_stats.ncx2.pdf(t, d, lam) / t, 0, np.inf, limit=400)
            assert J.exact_js_risk(d, lam) == pytest.approx(d - (d - 2) ** 2 * val, abs=1e-6)


def test_exact_risk_at_origin_and_limit():
    assert all(J.exact_js_risk(d, 0.0) == 2.0 for d in (3, 5, 10, 50))
    assert J.exact_js_risk(10, 1e6) == pytest.approx(10.0, abs=1e-3)           # weit weg vom Nullpunkt schrumpft nichts mehr
    with pytest.raises(ValueError):
        J.exact_js_risk(2, 1.0)


def test_simulated_risk_matches_the_exact_formula():
    for d, norm in ((3, 0.0), (5, 2.0), (10, 3.0), (10, 8.0)):
        mu, X = warehouse_observations(d, norm, "eine", 60_000, Stream(7))
        simulated = J.risk_summary(mu, X)["js"]
        assert simulated == pytest.approx(J.exact_js_risk(d, norm * norm), rel=0.03), (d, norm)
        assert J.risk_summary(mu, X)["einzel"] == pytest.approx(d, rel=0.03)


def test_hand_instances():
    out = J.james_stein(np.array([[3.0, 0.0, 0.0]]))
    assert out[0].tolist() == pytest.approx([8 / 3, 0.0, 0.0])                       # Faktor 1 − 1/9
    assert J.james_stein_factor(np.array([[3.0, 0.0, 0.0]]))[0] == pytest.approx(8 / 9)
    assert J.positive_part(np.array([[0.5, 0.0, 0.0]]))[0].tolist() == [0.0, 0.0, 0.0]
    assert J.james_stein(np.array([[0.5, 0.0, 0.0]]))[0].tolist() == pytest.approx([-1.5, 0.0, 0.0])    # Faktor 1 − 1/0,25 = −3
    with pytest.raises(ValueError):
        J.james_stein(np.array([[1.0, 2.0]]))


def test_branch_positive_part_differs_from_james_stein_at_the_origin_and_never_loses():
    mu, X = warehouse_observations(10, 0.0, "eine", 100_000, Stream(3))
    r = J.risk_summary(mu, X)
    assert r["pp"] < 0.7 * r["js"]                                                    # 0,126·10 gegen 0,2·10
    for norm in (0.0, 2.0, 5.0):
        mu, X = warehouse_observations(10, norm, "eine", 100_000, Stream(4))
        r = J.risk_summary(mu, X)
        assert r["pp"] <= r["js"] + 0.02 and r["js"] < r["einzel"]


def test_risk_depends_on_the_norm_not_on_the_direction():
    a = J.risk_summary(*warehouse_observations(10, 5.0, "eine", 100_000, Stream(5)))["js"]
    b = J.risk_summary(*warehouse_observations(10, 5.0, "gleich", 100_000, Stream(6)))["js"]
    assert a == pytest.approx(b, rel=0.02) and a == pytest.approx(J.exact_js_risk(10, 25.0), rel=0.02)


def test_single_coordinate_risk_exceeds_the_single_mean_for_large_d():
    mu, X = warehouse_observations(50, 8.0, "eine", 50_000, Stream(8))
    r = J.risk_summary(mu, X)
    assert r["koordinate1_pp"] > 5.0 and r["pp"] / 50 < 0.7                           # das Ganze gewinnt, die erste Koordinate verliert mehr als das Fünffache


def test_shrinkage_toward_the_common_mean():
    X = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 2.0, 0.0, 2.0]])
    out = J.js_toward_mean(X)
    assert out[0].tolist() == [1.0, 1.0, 1.0, 1.0]                                    # alle gleich: nichts zu schrumpfen
    assert out[1].tolist() == pytest.approx([0.25, 1.75, 0.25, 1.75])                # X̄ = 1, ‖X − X̄‖² = 4, d − 3 = 1: Faktor 1 − 1/4 = 0,75, also 1 ± 0,75
    with pytest.raises(ValueError):
        J.js_toward_mean(np.ones((2, 3)))


# ---- Varianz des Medians -------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [5, 21, 101, 301])
def test_variance_of_median_odd_n_equals_scipy_quadrature(n):
    k = (n - 1) // 2
    c = math.exp(math.lgamma(n + 1) - 2 * math.lgamma(k + 1))
    dens = lambda x: c * scipy_stats.norm.pdf(x) * scipy_stats.norm.cdf(x) ** k * scipy_stats.norm.sf(x) ** k
    v, _ = integrate.quad(lambda x: x * x * dens(x), -12, 12, epsabs=1e-13, epsrel=1e-12, limit=400)
    assert T.var_median_normal(n)[0] == pytest.approx(v, rel=1e-8)
    assert T.var_median_normal(n)[1] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("n", [4, 6, 10])
def test_variance_of_median_even_n_equals_double_integral(n):
    """Median = Mittel der beiden mittleren Ordnungsstatistiken; Varianz aus der gemeinsamen Dichte mit scipy.integrate.dblquad."""
    k = n // 2
    c = math.exp(math.lgamma(n + 1) - 2 * math.lgamma(k))
    phi, Phi = scipy_stats.norm.pdf, scipy_stats.norm.cdf
    joint = lambda y, x: c * Phi(x) ** (k - 1) * phi(x) * phi(y) * (1 - Phi(y)) ** (k - 1)
    v, _ = integrate.dblquad(lambda y, x: 0.25 * (x + y) ** 2 * joint(y, x), -9, 9, lambda x: x, lambda x: 9, epsabs=1e-10, epsrel=1e-9)
    assert T.var_median_normal(n)[0] == pytest.approx(v, rel=1e-6)


def test_efficiency_hand_values_sawtooth_and_limit():
    assert T.efficiency_normal(1) == pytest.approx(1.0) and T.efficiency_normal(2) == pytest.approx(1.0)       # n = 2: Median = Mittel
    assert T.efficiency_normal(3) == pytest.approx(0.7429, abs=5e-5)
    assert T.efficiency_normal(4) > T.efficiency_normal(3) and T.efficiency_normal(4) > T.efficiency_normal(5)    # gerades n liegt über beiden Nachbarn: Sägezahn
    assert T.efficiency_normal(100) > T.efficiency_normal(101) > T.efficiency_normal(1001) > T.EFFICIENCY_LIMIT
    assert T.efficiency_normal(1001) == pytest.approx(T.EFFICIENCY_LIMIT, abs=5e-4)


def test_efficiency_equals_a_simulation():
    for n in (4, 5, 10, 21):
        X = Stream(n).normal(40_000 * n).reshape(40_000, n)
        sim = X.mean(axis=1).var() / E.median_rows(X).var()
        assert sim == pytest.approx(T.efficiency_normal(n), rel=0.04), n


# ---- Lognormal -------------------------------------------------------------------------------------------------------------------------------

def test_lognormal_quantile_and_cdf_equal_scipy():
    for sigma in (0.5, 1.0, 1.5):
        for tau in (0.5, 0.8, 0.95, 0.99):
            assert T.lognormal_quantile(sigma, tau) == pytest.approx(scipy_stats.lognorm.ppf(tau, sigma), rel=1e-9)
        assert T.lognormal_cdf(2.0, sigma) == pytest.approx(scipy_stats.lognorm.cdf(2.0, sigma), abs=1e-12)
    assert T.lognormal_cdf(0.0, 1.0) == 0.0 and T.norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-6)


def test_lognormal_and_normal_expectile_equal_numerical_minimization():
    for sigma, tau in ((0.5, 0.8), (1.0, 0.8), (0.5, 0.5), (1.0, 0.95), (1.5, 0.6)):
        dist = scipy_stats.lognorm(sigma)

        def loss(e):
            up = integrate.quad(lambda x: (x - e) ** 2 * dist.pdf(x), e, np.inf, limit=300)[0]
            down = integrate.quad(lambda x: (e - x) ** 2 * dist.pdf(x), 0, e, limit=300)[0]
            return tau * up + (1 - tau) * down

        r = optimize.minimize_scalar(loss, bounds=(0.05, 20), method="bounded", options={"xatol": 1e-10})
        assert T.lognormal_expectile(sigma, tau) == pytest.approx(r.x, rel=1e-5)
    for tau in (0.5, 0.8, 0.95):
        def nloss(e):
            up = integrate.quad(lambda x: (x - e) ** 2 * scipy_stats.norm.pdf(x), e, np.inf)[0]
            down = integrate.quad(lambda x: (e - x) ** 2 * scipy_stats.norm.pdf(x), -np.inf, e)[0]
            return tau * up + (1 - tau) * down
        r = optimize.minimize_scalar(nloss, bounds=(-5, 5), method="bounded", options={"xatol": 1e-10})
        assert T.normal_expectile(tau) == pytest.approx(r.x, abs=1e-5)
    assert T.normal_expectile(0.5) == pytest.approx(0.0, abs=1e-12)


def test_expectile_of_a_large_sample_is_the_exact_expectile():
    x = draw("Lognormal", Stream(9), 1_000_000, sigma=0.5)
    assert E.expectile(x, 0.8) == pytest.approx(T.lognormal_expectile(0.5, 0.8), rel=3e-3)
    assert E.quantile_interval(x, 0.8)[0] == pytest.approx(T.lognormal_quantile(0.5, 0.8), rel=3e-3)


def test_gap_formulas_and_hand_value():
    assert T.gap_mean_median(0.0) == 1.0 and T.gap_mean_median(1.0) == pytest.approx(math.exp(0.5))
    assert T.median_miss_share(0.4) == pytest.approx(1 - math.exp(-0.08)) == pytest.approx(0.0769, abs=5e-5)
    assert T.share_below_mean(0.5) == pytest.approx(scipy_stats.norm.cdf(0.25))
    assert T.poisson_cdf(0.7, 0) == pytest.approx(math.exp(-0.7)) == pytest.approx(0.4966, abs=5e-5)
    assert T.INFLUENCE_PLATEAU == pytest.approx(1.2533, abs=5e-5) and T.EFFICIENCY_LIMIT == pytest.approx(0.6366, abs=5e-5)


def test_branch_quantile_and_expectile_differ_for_lognormal_one():
    gaps = [T.lognormal_expectile(1.0, 0.8) - T.lognormal_quantile(1.0, 0.8), T.lognormal_expectile(0.5, 0.5) - 1.0]
    assert gaps[0] > 0.3 and gaps[1] > 0.1                      # Lognormal(0; 1) bei τ = 0,8: Expektile über dem Quantil; bei τ = ½: Expektile (= Mittelwert) über dem Median
    assert T.lognormal_expectile(0.5, 0.5) == pytest.approx(T.gap_mean_median(0.5), rel=1e-9)   # τ = ½: Expektile = Mittelwert


def test_risk_summary_equals_the_direct_formulas():
    mu, X = warehouse_observations(5, 1.5, "eine", 2000, Stream(21))
    r = J.risk_summary(mu, X)
    assert r["einzel"] == pytest.approx(((X - mu) ** 2).sum(axis=1).mean())
    assert r["pp"] == pytest.approx(((J.positive_part(X) - mu) ** 2).sum(axis=1).mean())
    assert r["js"] == pytest.approx(((J.james_stein(X) - mu) ** 2).sum(axis=1).mean())
    assert r["koordinate1_pp"] == pytest.approx(((J.positive_part(X)[:, 0] - mu[0]) ** 2).mean())
    assert r["gemeinsames_mittel"] == pytest.approx(((J.js_toward_mean(X) - mu) ** 2).sum(axis=1).mean())


def test_zero_observation_has_factor_zero_and_variance_needs_n_at_least_one():
    assert J.james_stein_factor(np.zeros((2, 4))).tolist() == [0.0, 0.0]
    with pytest.raises(ValueError):
        T.var_median_normal(0)


def test_dimension_limits_of_the_estimators():
    with pytest.raises(ValueError):
        J.positive_part(np.ones((2, 2)))
    assert "gemeinsames_mittel" in J.risk_summary(np.zeros(4), np.ones((3, 4))) and "gemeinsames_mittel" not in J.risk_summary(np.zeros(3), np.ones((3, 3)))


def test_lognormal_cdf_below_one_equals_scipy():
    for x in (0.2, 0.9, 1.0, 3.5):
        assert T.lognormal_cdf(x, 0.7) == pytest.approx(scipy_stats.lognorm.cdf(x, 0.7), abs=1e-12)
    assert T.lognormal_cdf(-1.0, 0.7) == 0.0


def test_risk_of_the_first_warehouse_for_james_stein_equals_the_direct_formula():
    mu, X = warehouse_observations(6, 2.0, "eine", 3000, Stream(22))
    r = J.risk_summary(mu, X)
    assert r["koordinate1_js"] == pytest.approx(((J.james_stein(X)[:, 0] - mu[0]) ** 2).mean())
