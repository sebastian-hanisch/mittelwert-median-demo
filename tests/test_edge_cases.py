"""Randfälle der Kernroutinen (DEMO-PLAYBOOK §12, Mutationstests der Linie Mehrkriterien-Ranking): n = 1, n = 2, Gleichstand, inklusive Grenzen, entartete Eingaben."""

from fractions import Fraction

import numpy as np
import pytest

import mtw_constants as C
import mtw_estimators as E
import mtw_evaluation as V
import mtw_stein as J
import mtw_sums as M
import mtw_theory as T
from mtw_scenario import Stream, draw, make_series, rounded, with_miscount


def test_exactly_two_values_everywhere():
    """n = 2: der kleinste Fall, in dem es etwas zu vergleichen gibt (Zweige `n < 2` und `n − 1` dürfen nicht greifen)."""
    assert E.median([5.0, 1.0]) == 3.0 and E.median_interval([5.0, 1.0]) == (1.0, 5.0)
    assert E.quantile_interval([5.0, 1.0], 0.5) == (1.0, 5.0) and E.quantile_interval([5.0, 1.0], 0.75) == (5.0, 5.0) and E.quantile_interval([5.0, 1.0], 0.25) == (1.0, 1.0)
    assert E.expectile([0.0, 4.0], 0.8) == pytest.approx(3.2)
    assert float(M.welford_mean(np.array([1.0, 2.0]), np.float64)[0]) == 1.5 and M.exact_mean(np.array([1.0, 2.0])) == Fraction(3, 2)
    assert T.efficiency_normal(2) == pytest.approx(1.0)
    assert E.breakdown_k([0.0, 1.0]) == 1 and E.breakdown_k([0.0, 1.0, 2.0]) == 2


def test_single_value():
    assert E.median([7.0]) == 7.0 and E.quantile_interval([7.0], 0.9) == (7.0, 7.0) and E.expectile([7.0], 0.1) == 7.0
    assert float(M.shifted_mean(np.array([7.0]), np.float32)) == 7.0 and float(M.kahan_sum(np.array([7.0]), np.float32)[0]) == 7.0
    assert make_series(1, 5.0, "Normal", "aufsteigend", 1, "float32").shape == (1,)
    assert E.centroid(np.array([[2.0, 3.0]])).tolist() == [2.0, 3.0]


def test_all_values_equal():
    x = np.full(10, 3.5)
    assert E.median(x) == 3.5 and E.median_interval(x) == (3.5, 3.5) and E.expectile(x, 0.7) == pytest.approx(3.5, rel=1e-12) and E.quantile_interval(x, 0.3) == (3.5, 3.5)
    assert E.sad(x, 3.5) == 0.0 and E.pinball(x, 3.5, 0.8) == 0.0 and E.expectile_loss(x, 3.5, 0.8) == 0.0
    assert M.exact_mean(x) == Fraction(7, 2) and M.floor_of_format(Fraction(7, 2), np.float32) == 0.0


def test_bounds_are_inclusive_for_quantile_levels():
    """n·τ ganzzahlig: das Intervall reicht bis zum nächsten Wert; τ nahe 0 und 1 nehmen die Ränder; τ = 0 und τ = 1 sind keine Stufen."""
    x = [1.0, 2.0, 3.0, 4.0]
    assert E.quantile_interval(x, 0.5) == (2.0, 3.0) and E.quantile_interval(x, 0.75) == (3.0, 4.0) and E.quantile_interval(x, 0.25) == (1.0, 2.0)
    assert E.quantile_interval(x, 0.01) == (1.0, 1.0) and E.quantile_interval(x, 0.99) == (4.0, 4.0)
    for tau in (0.0, 1.0):
        with pytest.raises(ValueError):
            E.quantile_interval(x, tau)


def test_empty_inputs_raise_where_there_is_no_answer():
    for fn in (E.median, E.median_interval):
        with pytest.raises(ValueError):
            fn([])
    with pytest.raises(ValueError):
        E.expectile([], 0.5)
    with pytest.raises(ValueError):
        E.centroid(np.zeros((0, 2)))


def test_ties_in_the_median_interval_and_quantile():
    assert E.median_interval([2.0, 2.0, 2.0, 2.0]) == (2.0, 2.0) and E.median([1.0, 2.0, 2.0, 3.0]) == 2.0
    assert E.quantile_interval([1.0, 2.0, 2.0, 2.0, 9.0], 0.6) == (2.0, 2.0)


def test_stein_with_minimal_dimension_and_zero_observation():
    X = np.array([[1.0, 1.0, 1.0]])
    assert J.positive_part(X)[0].tolist() == pytest.approx([2 / 3] * 3)                  # ‖X‖² = 3, Faktor 1 − 1/3
    zero = np.zeros((1, 3))
    assert J.james_stein(zero)[0].tolist() == [0.0, 0.0, 0.0] and J.js_toward_mean(np.zeros((1, 4)))[0].tolist() == [0.0] * 4
    assert J.exact_js_risk(3, 0.0) == 2.0


def test_distributions_with_one_value_and_extreme_parameters():
    for dist in ("Normal", "Laplace", "Logistisch", "Gleichverteilung", "t(3)", "t(5)", "Exponential", "Lognormal", "Pareto(2,5)", "Poisson"):
        v = draw(dist, Stream(1), 1, sigma=C.SIGMA_MAX, lam=C.LAM_MAX)
        assert v.shape == (1,) and np.isfinite(v).all(), dist
        w = draw(dist, Stream(1), 3, sigma=C.SIGMA_MIN, lam=C.LAM_MIN)
        assert np.isfinite(w).all(), dist
    assert rounded(np.array([0.5]), 2.0).tolist() == [0.0]
    assert with_miscount(np.zeros((1, 2)), 0.0)[-1].tolist() == [0.0, 0.0]


@pytest.mark.parametrize("out", [0, 49])
def test_sample_analysis_at_the_smallest_and_largest_settings(out):
    for dist in ("Normal", "Poisson", "Lognormal"):
        a = V.sample_analysis(V.Settings(dist=dist, wn=5, out=out, h=2.0, tau=C.TAU_MAX, seed=3))
        assert len(a.x) == 5 and np.isfinite([a.mean, a.median, a.expectile]).all() and a.quantile[0] <= a.quantile[1]
        b = V.sample_analysis(V.Settings(dist=dist, wn=10_000, out=out, tau=C.TAU_MIN, sigma=C.SIGMA_MAX, lam=C.LAM_MAX, outdist=1000, seed=3))
        assert b.quantile[0] - 1e-12 <= b.median <= b.quantile[1] + 1e-12            # τ = ½: das Quantil-Intervall enthält den Median


def test_rechnen_at_the_smallest_sizes_and_extreme_ratios():
    for fmt in C.FORMATS:
        for ratio in (C.RATIO_OPTIONS[0], C.RATIO_OPTIONS[-1]):
            r = V.rechnen_analysis(V.Settings(fmt=fmt, n=1000, ratio=ratio, rdist="Lognormal", order="absteigend"))
            assert set(r.abs_errors) >= {"naiv", "Kahan", "Untergrenze"} and all(np.isfinite(v) for v in r.abs_errors.values())


def test_plane_with_zero_and_maximal_miscount():
    for d in (C.MISCOUNT_MIN, C.MISCOUNT_MAX):
        p = V.plane_analysis(V.Settings(miscount=d))
        assert p.centroid_shift == pytest.approx(d / (C.PLANE_N + 1), abs=0.25)     # der neue Kunde zieht den Schwerpunkt um etwa Abstand/(n + 1)
        assert p.within >= 0 and p.between >= 0


def test_quantile_interval_rejects_an_empty_sample_with_a_value_error():
    """Wächter für leere Eingaben (Mutationsprüfung: ohne ihn käme ein IndexError aus dem Sortierfeld)."""
    with pytest.raises(ValueError):
        E.quantile_interval([], 0.5)


def test_quantile_interval_level_so_close_to_one_that_n_tau_rounds_to_n():
    """n·τ liegt innerhalb der Toleranz 10⁻⁹ von n (k = n): das Intervall ist der größte Wert, kein Zugriff hinter dem Ende. Handrechnung: n = 1, τ = 1 − 10⁻¹², k = 1 − 10⁻¹² → 1."""
    assert E.quantile_interval([5.0], 1 - 1e-12) == (5.0, 5.0)
    assert E.quantile_interval([1.0, 2.0, 3.0], 1 - 1e-12) == (3.0, 3.0)
