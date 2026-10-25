"""Orakel Mitten (ORACLE-PLAYBOOK): Median, Quantil, Expektile, Schwerpunkt, Einfluss und Durchschlag gegen `numpy.median`, `numpy.quantile(method="inverted_cdf")`, `scipy.optimize`
und die volle Aufzählung. Dazu Handinstanzen (conftest.py) und Zweig-Tests (Nullspalten-Regel)."""

import itertools
import math

import numpy as np
import pytest

import mtw_estimators as E

optimize = pytest.importorskip("scipy.optimize")


def random_case(rng, ties):
    n = int(rng.integers(1, 41))
    return rng.integers(0, 6, n).astype(float) if ties else rng.normal(0, 1, n)


# ---- Median ----------------------------------------------------------------------------------------------------------------------------------

def test_median_equals_numpy_including_ties_and_even_n():
    rng = np.random.default_rng(99)
    for i in range(1500):
        x = random_case(rng, i % 2)
        assert E.median(x) == float(np.median(x))
    X = rng.normal(0, 1, (200, 17))
    assert np.array_equal(E.median_rows(X), np.median(X, axis=1))
    X = rng.integers(0, 4, (200, 16)).astype(float)
    assert np.array_equal(E.median_rows(X), np.median(X, axis=1))


def test_median_hand_instances(one_to_four):
    assert E.median([3.0, 1.0, 2.0]) == 2.0
    assert E.median_interval(one_to_four) == (2.0, 3.0) and E.median(one_to_four) == 2.5
    assert E.median_interval([3.0, 1.0, 2.0]) == (2.0, 2.0)
    assert [E.sad(one_to_four, c) for c in (2.0, 2.5, 3.0)] == [4.0, 4.0, 4.0]


def test_median_minimizes_sum_of_absolute_deviations_against_scipy():
    rng = np.random.default_rng(7)
    for i in range(300):
        x = random_case(rng, i % 2)
        if len(x) < 2:
            continue
        r = optimize.minimize_scalar(lambda c: E.sad(x, c), bounds=(x.min() - 1e-9, x.max() + 1e-9), method="bounded", options={"xatol": 1e-10})
        assert E.sad(x, E.median(x)) <= r.fun + 1e-9
        lo, hi = E.median_interval(x)
        assert max(E.sad(x, lo), E.sad(x, hi), E.sad(x, 0.5 * (lo + hi))) == pytest.approx(E.sad(x, E.median(x)), abs=1e-12)
        grid = np.concatenate([x, np.linspace(x.min(), x.max(), 101)])
        assert E.sad(x, E.median(x)) <= min(E.sad(x, c) for c in grid) + 1e-12


# ---- Quantil und Pinball --------------------------------------------------------------------------------------------------------------------

def test_quantile_interval_lower_end_equals_numpy_inverted_cdf():
    rng = np.random.default_rng(5)
    for i in range(1500):
        x = rng.integers(0, 10, int(rng.integers(2, 60))).astype(float) if i % 2 else rng.normal(0, 1, int(rng.integers(2, 60)))
        tau = float(rng.choice([0.1, 0.25, 0.5, 0.8, 0.9, 0.95, 0.99]))
        lo, hi = E.quantile_interval(x, tau)
        assert lo == float(np.quantile(x, tau, method="inverted_cdf"))
        assert E.pinball(x, lo, tau) == pytest.approx(E.pinball(x, hi, tau), abs=1e-10)


def test_quantile_interval_hand_instance_and_minimum_over_a_grid():
    x = [1.0, 2.0, 3.0, 4.0, 10.0]
    assert E.quantile_interval(x, 0.8) == (4.0, 10.0)                                       # n·τ = 4 ist ganzzahlig
    assert E.pinball(x, 4.0, 0.8) == pytest.approx(0.2 * 3 + 0.2 * 2 + 0.2 * 1 + 0.8 * 6) == pytest.approx(6.0)
    assert E.pinball(x, 10.0, 0.8) == pytest.approx(0.2 * 9 + 0.2 * 8 + 0.2 * 7 + 0.2 * 6) == pytest.approx(6.0)
    assert E.quantile_interval(x, 0.5) == (3.0, 3.0) and E.quantile_interval(x, 0.9) == (10.0, 10.0)
    rng = np.random.default_rng(31)
    for _ in range(300):
        y = rng.normal(0, 1, int(rng.integers(7, 41)))
        tau = float(rng.choice([0.1, 0.25, 0.5, 0.8, 0.95]))
        q = E.quantile_interval(y, tau)[0]
        grid = np.concatenate([y, np.linspace(y.min(), y.max(), 401)])
        assert E.pinball(y, q, tau) <= min(E.pinball(y, c, tau) for c in grid) + 1e-12


def test_quantile_and_expectile_reject_bad_levels():
    for fn in (E.quantile_interval, E.expectile):
        for tau in (0.0, 1.0, -0.2, 1.5):
            with pytest.raises(ValueError):
                fn([1.0, 2.0], tau)


# ---- Expektile ------------------------------------------------------------------------------------------------------------------------------

def test_expectile_minimizes_the_asymmetric_squares_against_scipy():
    rng = np.random.default_rng(13)
    for _ in range(400):
        x = rng.lognormal(0, 1, int(rng.integers(3, 80)))
        tau = float(rng.choice([0.05, 0.3, 0.5, 0.8, 0.95]))
        e = E.expectile(x, tau)
        r = optimize.minimize_scalar(lambda c: E.expectile_loss(x, c, tau), bounds=(x.min(), x.max()), method="bounded", options={"xatol": 1e-12})
        assert abs(e - r.x) / (x.max() - x.min()) < 1e-7 and E.expectile_loss(x, e, tau) <= r.fun + 1e-9
        # Bedingung erster Ordnung: τ Σ(x − e)₊ = (1 − τ) Σ(e − x)₊
        assert tau * np.maximum(x - e, 0).sum() == pytest.approx((1 - tau) * np.maximum(e - x, 0).sum(), rel=1e-9, abs=1e-9)


def test_expectile_hand_instances():
    assert E.expectile([0.0, 4.0], 0.8) == pytest.approx(3.2)                              # 0,8 · (4 − e) = 0,2 · e
    assert E.expectile([1.0, 2.0, 4.0], 0.5) == pytest.approx(7 / 3)                       # τ = ½ ist das Mittel
    assert E.expectile([5.0], 0.9) == 5.0


def test_expectile_rows_equal_row_by_row():
    rng = np.random.default_rng(2)
    X = rng.lognormal(0, 1, (50, 30))
    for tau in (0.2, 0.8):
        assert E.expectile_rows(X, tau) == pytest.approx([E.expectile(row, tau) for row in X], rel=1e-12)


def test_quantile_and_expectile_differ_on_a_skewed_sample():
    x = np.random.default_rng(1).lognormal(0, 1, 2000)
    q, e = E.quantile_interval(x, 0.8)[0], E.expectile(x, 0.8)
    assert abs(q - e) / (np.quantile(x, 0.75) - np.quantile(x, 0.25)) > 0.2


# ---- Schwerpunkt, Huygens, verkleinertes Mittel ---------------------------------------------------------------------------------------------

def test_centroid_hand_instance_and_huygens(triangle):
    c = E.centroid(triangle)
    assert c.tolist() == pytest.approx([4 / 3, 1.0])
    assert E.ssq(triangle, c) == pytest.approx(50 / 3)
    assert E.ssq(triangle, [0.0, 0.0]) == pytest.approx(25.0)
    within, between = E.huygens(triangle, [0.0, 0.0])
    assert within == pytest.approx(50 / 3) and between == pytest.approx(3 * (16 / 9 + 1)) and within + between == pytest.approx(25.0)


def test_centroid_is_the_minimizer_and_equivariant_against_scipy():
    rng = np.random.default_rng(7)
    P = rng.normal(0, 1, (60, 2))
    w = rng.uniform(0.5, 3, 60)
    for weights in (None, w):
        c = E.centroid(P, weights)
        r = optimize.minimize(lambda z: E.ssq(P, z, weights), c + 1.0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000})
        assert np.abs(r.x - c).max() < 1e-5
    A, b = np.array([[2.0, 1.5], [-0.3, 0.8]]), np.array([3.0, -2.0])
    assert E.centroid(P @ A.T + b) == pytest.approx(A @ E.centroid(P) + b, abs=1e-12)
    cc = np.array([0.7, -1.3])
    within, between = E.huygens(P, cc)
    assert E.ssq(P, cc) == pytest.approx(within + between, rel=1e-12)


def test_shrunk_mse_equals_exact_enumeration():
    from fractions import Fraction
    mu, n, c = Fraction(4, 5), 3, Fraction(3, 5)                       # Zweipunktverteilung μ ± 1, je ½; Mittel von 3 Werten
    mse = sum(Fraction(math.comb(n, k), 2 ** n) * (c * (mu + Fraction(2 * k - n, n)) - mu) ** 2 for k in range(n + 1))
    assert E.shrunk_mse(0.6, 0.8, 1.0, 3) == pytest.approx(0.2224) == pytest.approx(float(mse))


# ---- Einfluss und Verschiebung ---------------------------------------------------------------------------------------------------------------

def test_sensitivity_curve_hand_instance_and_mean_is_linear():
    assert E.sensitivity_curve([1.0, 2.0, 3.0], 10.0, "Mittel") == pytest.approx(8.0)           # 4·(4 − 2) = z − x̄
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 99)
    for t in (0.5, 5.0, 1e6):
        assert E.sensitivity_curve(base, t, "Mittel") == pytest.approx(t - base.mean(), rel=1e-9, abs=1e-6)
    plateau = [E.sensitivity_curve(base, t, "Median") for t in (1.0, 10.0, 1e6)]
    assert plateau[0] == plateau[1] == plateau[2]                                                # ab t = 1 liegt t rechts vom Mittelfeld: konstant, beschränkt
    assert E.sensitivity_curve(base, E.median(base), "Median") == 0.0 != plateau[0]                  # t genau in der Mitte der Basis: der Median bleibt stehen


def test_sensitivity_curve_median_branch_depends_on_t_near_the_centre():
    base = np.array([-2.0, -1.0, 1.0, 2.0])
    assert E.sensitivity_curve(base, 0.5, "Median") != E.sensitivity_curve(base, 5.0, "Median")
    with pytest.raises(KeyError):
        E.sensitivity_curve(base, 1.0, "Modus")


def test_shift_under_contamination_hand_instance():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert E.shift_under_contamination(x, [0, 1], 1000.0, "Median") == 2.0                      # (3, 4, 5, 1000, 1000) hat den Median 5
    assert E.shift_under_contamination(x, [0, 1, 2], 1000.0, "Median") == 997.0                 # (4, 5, 1000, 1000, 1000): 1000 − 3
    assert E.shift_under_contamination(x, [0], 11.0, "Mittel") == pytest.approx(2.0)            # (11 − 1)/5
    assert x == [1.0, 2.0, 3.0, 4.0, 5.0]                                                       # die Eingabe bleibt unverändert


# ---- Durchschlagspunkt ------------------------------------------------------------------------------------------------------------------------

def test_breakdown_hand_instance():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert E.breakdown_k(x) == 3 == math.ceil(5 / 2)
    assert E.max_shift_below_breakdown(x, 2, 1e6) == E.max_shift_below_breakdown(x, 2, 1e3) == 2.0   # bis k = 2 beschränkt: Median 5 gegen 3
    assert E.max_shift_below_breakdown(x, 3, 1e6) > 9e5


def test_breakdown_equals_ceil_half_and_boundedness_property():
    """Unabhängige Kennzeichnung: unterhalb des Durchschlags hängt die größte Verschiebung nicht von M ab, am Durchschlag wächst sie mit M."""
    rng = np.random.default_rng(4)
    for n in range(3, 9):
        for _ in range(3):
            x = rng.normal(0, 1, n)
            k = E.breakdown_k(x)
            assert k == math.ceil(n / 2)
            assert E.max_shift_below_breakdown(x, k - 1, 1e6) == pytest.approx(E.max_shift_below_breakdown(x, k - 1, 1e3), abs=1e-9)
            assert E.max_shift_below_breakdown(x, k, 1e6) > 0.4e6
            y = np.array(x)
            y[0] = 1e6
            assert abs(np.mean(y) - np.mean(x)) > 1e3                                         # beim Mittel genügt ein Wert


def test_breakdown_enumeration_matches_a_second_enumeration():
    x = np.array([0.3, -1.2, 2.0, 0.9, -0.4, 1.1, -2.2])
    best = 0.0
    for idx in itertools.combinations(range(7), 3):
        for signs in itertools.product((1, -1), repeat=3):
            y = x.copy()
            y[list(idx)] = np.array(signs) * 1e6
            best = max(best, abs(float(np.median(y)) - float(np.median(x))))
    assert E.max_shift_below_breakdown(x, 3, 1e6) == pytest.approx(best)
    assert E.breakdown_k(x) == 4


def test_breakdown_sign_patterns_follow_the_documented_limit_n_9():
    """Bis n = 9 werden alle Vorzeichenmuster aufgezählt, ab n = 10 nur +M (Mutationsprüfung: Grenze 9 inklusive)."""
    x9 = np.array([0.3, -1.2, 2.0, 0.9, -0.4, 1.1, -2.2, 0.1, 0.7])
    assert E.max_shift_below_breakdown(x9, 2, 1e6) == E.max_shift_below_breakdown(x9, 2, 1e6, both_signs=True)
    x10 = np.append(x9, -0.9)
    assert E.max_shift_below_breakdown(x10, 2, 1e6) == E.max_shift_below_breakdown(x10, 2, 1e6, both_signs=False)


def test_neumaier_branch_with_growing_summands_is_exact_with_wide_rest():
    """Wächst der Summand über die Summe hinaus (|s| < |v|), greift der andere Zweig der Neumaier-Rechnung."""
    import mtw_sums as M
    x = np.array([1.0, 1e3, 1e6, 1e9, 1e12, 3.0, 1e-3], dtype=np.float32)
    s, c = M.neumaier_sum_wide_rest(x, np.float32)
    exact = M.exact_sum(x.astype(np.float64))
    assert abs(float(s) + c - float(exact)) <= 2e-7 * float(exact)
