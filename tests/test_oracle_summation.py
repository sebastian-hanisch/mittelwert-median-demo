"""Orakel Summation (ORACLE-PLAYBOOK): die Verfahren aus mtw_sums.py gegen unabhängige Rechenwege.

* fortlaufende Summe, Kahan, Neumaier, Welford: eine im Test geschriebene Schleife mit expliziten Skalaren, Bit-Gleichheit (float32 und float64, beide Richtungen);
* exaktes Mittel: Bruchsumme `fractions.Fraction`, dazu `math.fsum` und die Fehlerschranke 2ε·Σ|x| für Kahan in float64;
* Handinstanzen aus conftest.py; Zweig-Tests (Nullspalten-Regel): jede Variante weicht auf mindestens einer Instanz von der Nachbarvariante ab."""

import math
from fractions import Fraction

import numpy as np
import pytest

import mtw_sums as M
from mtw_scenario import make_series

F32, F64 = np.float32, np.float64


# ---- Orakel: eigene Skalarschleifen (nicht aus mtw_sums) ------------------------------------------------------------------------------------

def ref_naive(values, dt):
    s = dt(0)
    for v in values:
        s = dt(s + v)
    return s


def ref_kahan(values, dt):
    s, c = dt(0), dt(0)
    for v in values:
        y = dt(v - c)
        t = dt(s + y)
        c = dt(dt(t - s) - y)
        s = t
    return s, c


def ref_neumaier(values, dt):
    s, c = dt(0), dt(0)
    for v in values:
        t = dt(s + v)
        c = dt(c + (dt(dt(s - t) + v) if abs(s) >= abs(v) else dt(dt(v - t) + s)))
        s = t
    return s, c


def ref_welford(values, dt):
    m = dt(0)
    for k, v in enumerate(values, start=1):
        m = dt(m + dt(dt(v - m) / dt(k)))
    return m


def datasets():
    rng = np.random.default_rng(1)       # Orakel-Daten dürfen aus numpy stammen: Gegenprobe, nicht Teil der Demo
    return {"N(1000,1)": 1000.0 + rng.standard_normal(20_000), "N(0,1)": rng.standard_normal(20_000), "lognormal": rng.lognormal(0, 1, 20_000)}


@pytest.mark.parametrize("fmt", ["float32", "float64"])
def test_loops_equal_independent_scalar_loops_bit_for_bit(fmt):
    dt = F32 if fmt == "float32" else F64
    for name, raw in datasets().items():
        x = raw.astype(dt)
        for xs in (x, x[::-1].copy()):
            ref_s, ref_c = ref_kahan(xs.tolist(), dt)
            s, c = M.kahan_sum(xs, dt)
            assert (float(s), float(c)) == (float(ref_s), float(ref_c)), (name, "Kahan")
            ns, nc = M.neumaier_sum(xs, dt)
            rs, rc = ref_neumaier(xs.tolist(), dt)
            assert (float(ns), float(nc)) == (float(rs), float(rc)), (name, "Neumaier")
            assert float(M.naive_sum(xs, dt)) == float(ref_naive(xs.tolist(), dt)), (name, "naiv")
            m, k = M.welford_mean(xs, dt)
            assert float(m) == float(ref_welford(xs.tolist(), dt)) and k == len(xs), (name, "Welford")


def test_state_continuation_equals_one_pass():
    x = make_series(5000, 1000.0, "Normal", "zufällig", 3, "float32")
    whole = M.kahan_sum(x, F32)
    part = M.kahan_sum(x[2000:], F32, M.kahan_sum(x[:2000], F32))
    assert (float(whole[0]), float(whole[1])) == (float(part[0]), float(part[1]))
    nw = M.neumaier_sum(x, F32)
    npart = M.neumaier_sum(x[2000:], F32, M.neumaier_sum(x[:2000], F32))
    assert (float(nw[0]), float(nw[1])) == (float(npart[0]), float(npart[1]))
    wm = M.welford_mean(x, F32)
    wp = M.welford_mean(x[2000:], F32, M.welford_mean(x[:2000], F32))
    assert float(wm[0]) == float(wp[0]) and wm[1] == wp[1] == 5000


def test_naive_prefix_is_a_sequential_sum():
    x = make_series(3000, 1000.0, "Normal", "zufällig", 8, "float32")
    pre = M.naive_prefix(x, F32)
    s = F32(0)
    for i in (0, 1, 2, 10, 500, 2999):
        s = ref_naive(x[: i + 1].tolist(), F32)
        assert float(pre[i]) == float(s)


# ---- exaktes Mittel -------------------------------------------------------------------------------------------------------------------------

def test_exact_mean_equals_fraction_sum():
    rng = np.random.default_rng(3)
    for dt in (F32, F64):
        x = (1000.0 + rng.standard_normal(5000)).astype(dt).astype(F64)
        ref = sum((Fraction(float(v)) for v in x), Fraction(0)) / len(x)
        assert M.exact_mean(x) == ref


def test_exact_sum_survives_catastrophic_cancellation():
    x = np.array([1e16, 1.0, -1e16, 1.0, 1e-8, -2.0])
    assert M.exact_sum(x) == sum((Fraction(float(v)) for v in x), Fraction(0)) == Fraction(1e-8)
    big = np.concatenate([np.full(1000, 1e300), np.full(1000, -1e300), [3.0, 1e-300]])
    assert M.exact_sum(big) == Fraction(3.0) + Fraction(1e-300)


def test_exact_mean_needs_data_and_errors_are_signed():
    with pytest.raises(ValueError):
        M.exact_mean(np.array([]))
    assert M.mean_error(2.5, Fraction(2)) == 0.5 and M.mean_error(1.5, Fraction(2)) == -0.5
    assert M.sum_error_as_mean(11.0, 4, Fraction(2)) == pytest.approx(0.75)         # 11/4 − 2


def test_floor_is_the_rounding_of_the_exact_result():
    exact = Fraction(1, 3)
    assert M.floor_of_format(exact, F64) == float(Fraction(float(1 / 3)) - exact)
    assert abs(M.floor_of_format(exact, F32)) <= float(exact) * 2.0 ** -24


# ---- Handinstanzen --------------------------------------------------------------------------------------------------------------------------

def test_hand_instance_float32_two_to_the_24(tiny_f32):
    assert float(M.naive_sum(tiny_f32, F32)) == 2.0 ** 24                       # 2²⁴ + 1 wird auf 2²⁴ gerundet, zweimal
    s, c = M.kahan_sum(tiny_f32, F32)
    assert float(s) == 2.0 ** 24 + 2 and float(c) == 0.0
    s, c = M.neumaier_sum(tiny_f32, F32)
    assert float(s) == 2.0 ** 24 and float(c) == 2.0 and float(s) + float(c) == 2.0 ** 24 + 2
    assert M.exact_sum(tiny_f32.astype(F64)) == 2 ** 24 + 2


def test_hand_instance_float64_tenths(tenths):
    assert float(M.naive_sum(tenths, F64)) == 0.9999999999999999
    assert math.fsum(tenths) == 1.0
    assert float(M.kahan_sum(tenths, F64)[0]) == 1.0
    assert float(M.pairwise_sum(tenths, F64)) == pytest.approx(1.0, abs=2e-16)


def test_hand_instance_welford(one_to_four):
    assert [float(M.welford_mean(one_to_four[:k], F64)[0]) for k in (1, 2, 3, 4)] == [1.0, 1.5, 2.0, 2.5]


def test_welford_is_exponential_smoothing_with_alpha_one_over_k():
    x = make_series(10_000, 50.0, "Normal", "zufällig", 8, "float64")
    smooth = 0.0
    for k, v in enumerate(x.tolist(), start=1):
        a = 1.0 / k
        smooth = (1 - a) * smooth + a * v
    assert float(M.welford_mean(x, F64)[0]) == pytest.approx(smooth, abs=1e-12) and smooth == pytest.approx(math.fsum(x) / len(x), abs=1e-12)


# ---- Schranken --------------------------------------------------------------------------------------------------------------------------------

def test_kahan_float64_stays_within_two_eps_times_sum_of_abs():
    rng = np.random.default_rng(5)
    eps = np.finfo(np.float64).eps / 2
    for _ in range(6):
        x = rng.standard_normal(20_000) * 10.0 ** rng.integers(-3, 4, 20_000)
        s, _ = M.kahan_sum(x, F64)
        assert abs(Fraction(float(s)) - M.exact_sum(x)) <= Fraction(float(2 * eps * np.abs(x).sum()))


def test_pairwise_matches_independent_recursion_and_handles_odd_lengths():
    def tree(v, dt):
        if len(v) == 1:
            return v[0]
        h = (len(v) + 1) // 2 if len(v) % 2 == 0 and False else len(v) // 2
        return dt(tree(v[:h], dt) + tree(v[h:], dt))
    x = make_series(64, 1000.0, "Normal", "zufällig", 3, "float32")
    # Binärbaum über Nachbarpaare ist bei Zweierpotenzen die rekursive Halbierung
    assert float(M.pairwise_sum(x, F32)) == float(tree(list(x), F32))
    assert float(M.pairwise_sum(np.array([1.0, 2.0, 3.0]), F64)) == 6.0 and float(M.pairwise_sum(np.array([5.0]), F64)) == 5.0
    assert float(M.pairwise_sum(np.array([]), F32)) == 0.0


def test_shifted_mean_hand_values():
    assert float(M.shifted_mean(np.array([10.0, 11.0, 12.0, 13.0]), F64)) == 11.5
    assert float(M.shifted_mean(np.array([7.0]), F32)) == 7.0
    with pytest.raises(ValueError):
        M.shifted_mean(np.array([]), F64)


# ---- Zweig-Tests (Nullspalten-Regel) --------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def million():
    x = make_series(1_000_000, 1000.0, "Normal", "zufällig", 5, "float32")
    return x, M.exact_mean(x.astype(F64))


def test_branch_every_method_differs_from_its_neighbour(million):
    x, exact = million
    n = len(x)
    s, c = M.kahan_sum(x, F32)
    ns, nc = M.neumaier_sum(x, F32)
    errors = {
        "naiv": M.sum_error_as_mean(M.naive_sum(x, F32), n, exact),
        "pairwise": M.sum_error_as_mean(M.pairwise_sum(x, F32), n, exact),
        "kahan": M.sum_error_as_mean(s, n, exact),
        "kahan_rest": M.sum_error_as_mean(float(s) - float(c), n, exact),
        "neumaier": M.sum_error_as_mean(F32(float(ns) + float(nc)), n, exact),
        "welford": M.mean_error(M.welford_mean(x, F32)[0], exact),
    }
    assert errors["naiv"] != errors["pairwise"] and abs(errors["naiv"]) > 1000 * abs(errors["pairwise"])
    assert errors["kahan_rest"] != errors["kahan"]                                   # der Rest c ist nicht null
    assert errors["neumaier"] != errors["kahan"]
    assert errors["welford"] != errors["pairwise"]


def test_kahan_with_rest_can_be_exactly_zero_which_is_a_finding_not_a_dead_branch():
    """Bei n = 10⁴ in float32 hält Kahan die Summe vollständig: s − c ist die exakte Summe. Ohne den Rest (nur s) fehlt bis zu einer halben Rastbreite."""
    exact_hits, without_rest_differs = 0, 0
    for seed in range(8):
        x = make_series(10_000, 1000.0, "Normal", "zufällig", seed, "float32")
        total = M.exact_sum(x.astype(F64))
        s, c = M.kahan_sum(x, F32)
        exact_hits += Fraction(float(s)) - Fraction(float(c)) == total
        without_rest_differs += Fraction(float(s)) != total
    assert exact_hits == 8 and without_rest_differs >= 6


def test_branch_neumaier_with_wide_rest_is_exact_where_float32_rest_fails(million):
    x, exact = million
    total = exact * len(x)
    ns, nc = M.neumaier_sum(x, F32)
    ws, wc = M.neumaier_sum_wide_rest(x, F32)
    narrow = abs(Fraction(float(ns)) + Fraction(float(nc)) - total)
    wide = abs(Fraction(float(ws)) + Fraction(wc) - total)
    assert narrow > 10 and wide < Fraction(1, 10)           # die Rest-Summe in float32 trägt den Fehler (an dieser Datenform gemessen)
    assert M.rest_trace(x[:20_000], F32, "neumaier")[2] > M.rest_trace(x[:20_000], F32, "kahan")[2] * 10


def test_rest_trace_agrees_with_the_summation_functions():
    x = make_series(3000, 1000.0, "Normal", "zufällig", 2, "float32")
    for method, fn in (("kahan", M.kahan_sum), ("neumaier", M.neumaier_sum), ("neumaier_wide", M.neumaier_sum_wide_rest)):
        s, c, biggest = M.rest_trace(x, F32, method)
        s2, c2 = fn(x, F32)
        assert float(s) == float(s2) and float(c) == float(c2) and biggest >= abs(float(c))
    with pytest.raises(ValueError):
        M.rest_trace(x, F32, "unbekannt")


def test_float64_accumulator_branch_differs_from_float32_mean():
    x = make_series(100_000, 1e6, "Normal", "zufällig", 5, "float32")
    exact = M.exact_mean(x.astype(F64))
    e32 = abs(M.mean_error(np.mean(x), exact))
    e64 = abs(M.mean_error(M.mean_with_float64_accumulator(x, F32), exact))
    assert e32 > 10 * e64 and e64 <= abs(M.floor_of_format(exact, F32)) + 1e-12


def test_edge_cases_empty_single_pair():
    for fn in (M.naive_sum, M.pairwise_sum):
        assert float(fn(np.array([]), F32)) == 0.0
    assert float(M.kahan_sum(np.array([3.0]), F64)[0]) == 3.0 and float(M.welford_mean(np.array([4.0, 6.0]), F64)[0]) == 5.0
    assert M.exact_mean(np.array([2.0, 3.0])) == Fraction(5, 2)


def test_neumaier_else_branch_with_growing_summands_is_exact_in_all_variants():
    """|s| < |v| schon bei den ersten Summanden: der Rest bleibt null, die Summe ist exakt 1,5 (zweiter Zweig der Neumaier-Rechnung, auch in der Diagnose)."""
    x = np.array([1.0, 10.0, 100.0, 1000.0, -1110.0, 0.5], dtype=np.float32)
    s, c = M.neumaier_sum(x, F32)
    assert float(s) + float(c) == 1.5
    ws, wc = M.neumaier_sum_wide_rest(x, F32)
    assert float(ws) + wc == 1.5
    for method in ("neumaier", "neumaier_wide"):
        ts, tc, biggest = M.rest_trace(x, F32, method)
        assert float(ts) + float(tc) == 1.5 and biggest >= 0.0


def test_neumaier_variants_agree_on_a_series_with_rounding_in_both_branches():
    """[0,1; 10⁸; −10⁸; 0,2] in float32: im zweiten Schritt (|s| < |v|) geht 0,1 verloren und muss im Rest landen; Ergebnis s + c gleich der exakten Summe der gespeicherten Werte."""
    x = np.array([0.1, 1e8, -1e8, 0.2], dtype=np.float32)
    exact = float(M.exact_sum(x.astype(np.float64)))
    s, c = M.neumaier_sum(x, F32)
    assert float(s) + float(c) == pytest.approx(exact, abs=1e-6)
    ws, wc = M.neumaier_sum_wide_rest(x, F32)
    assert float(ws) + wc == pytest.approx(exact, abs=1e-6)
    for method in ("neumaier", "neumaier_wide"):
        ts, tc, biggest = M.rest_trace(x, F32, method)
        assert float(ts) + float(tc) == pytest.approx(exact, abs=1e-6) and biggest == pytest.approx(0.1, abs=1e-6)
