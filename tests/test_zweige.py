"""Zweig-Tests (Nullspalten-Regel, DEMO-PLAYBOOK §0): jede Variante weicht auf mindestens einer Zelle von der Nachbarvariante ab, und keine Spalte der vorgerechneten
Tabellen besteht nur aus Nullen, außer dort, wo die Null ein dokumentierter Befund ist (Kahan mit Rest bei kleinem n)."""

import numpy as np
import pytest

import mtw_estimators as E
import mtw_experiments as X
import mtw_results as R
from mtw_scenario import DISTRIBUTIONS, Stream, draw


def test_no_method_column_of_the_error_tables_is_all_zero():
    for fmt in ("float32", "float64"):
        for name, per_n in R.e1(fmt)["methods"].items():
            assert any(c["mean"] > 0 for c in per_n.values()), (fmt, name)
        for ratio, row in R.e2_ratio(fmt)["ratios"].items():
            assert any(m["mean"] > 0 for m in row["methods"].values()), (fmt, ratio)
        for dist, orders in R.e2_order(fmt)["cells"].items():
            for order, methods in orders.items():
                for name, cell in methods.items():
                    assert cell["mean"] > 0 or name in ("Kahan mit Rest", "Neumaier mit Rest"), (fmt, dist, order, name)


def test_zero_errors_are_documented_findings():
    """Kahan mit Rest (s − c) hält die Summe bei float32 bis n = 10⁵ vollständig (Fehler genau 0) und danach fast: ein Befund, kein toter Zweig; ohne den Rest ist der Fehler nicht null."""
    for n in (1000, 10_000, 100_000):
        assert R.e1_cell("float32", "Kahan mit Rest", n)["mean"] == 0.0 and R.e1_cell("float32", "Kahan", n)["mean"] > 0.0
    assert 0.0 < R.e1_cell("float32", "Kahan mit Rest", 1_000_000)["mean"] < 1e-6 * R.e1_cell("float32", "naiv", 1_000_000)["mean"]


def test_branches_differ_in_the_stored_tables():
    n = 1_000_000
    naive, pairwise = R.e1_cell("float32", "naiv", n), R.e1_cell("float32", "Binärbaum", n)
    assert naive["mean"] > 1000 * pairwise["mean"]
    assert R.e1_cell("float32", "Kahan mit Rest", n)["per_seed"] != R.e1_cell("float32", "Kahan", n)["per_seed"]
    assert R.e1_cell("float32", "Neumaier", n)["per_seed"] != R.e1_cell("float32", "Kahan", n)["per_seed"]
    assert R.e1_cell("float32", "Welford", n)["per_seed"] != pairwise["per_seed"]
    assert R.e1_cell("float64", "Welford", n)["mean"] != R.e1_cell("float64", "Binärbaum", n)["mean"]
    assert R.e1_cell("float32", "verschoben", n)["per_seed"] != R.e1_cell("float32", "naiv", n)["per_seed"]
    for ratio in (100.0, 1e4):
        assert R.e2_ratio_cell("float32", ratio, "naiv")["mean"] != R.e2_ratio_cell("float32", 1.0, "naiv")["mean"]
    assert R.e2_order_cell("float32", "Normal", "aufsteigend", "naiv")["mean"] != R.e2_order_cell("float32", "Normal", "absteigend", "naiv")["mean"]
    assert R.e2_order_cell("float32", "Lognormal", "zufällig", "naiv")["mean"] != R.e2_order_cell("float32", "Normal", "zufällig", "naiv")["mean"]


def test_stein_positive_part_differs_from_james_stein_at_the_origin_but_not_far_away():
    near, far = R.stein_cell("eine", 10, 0.0), R.stein_cell("eine", 10, 20.0)
    assert near["pp"]["mean"] < 0.7 * near["js"]["mean"] and far["pp"]["mean"] == pytest.approx(far["js"]["mean"], abs=0.01)
    assert R.stein_cell("eine", 10, 5.0)["js"]["mean"] == pytest.approx(R.stein_cell("gleich", 10, 5.0)["js"]["mean"], abs=0.01)


def test_median_branches_even_odd_ties_rounding_and_contamination():
    assert R.efficiency(4) != R.efficiency(5) and R.efficiency(100) > R.efficiency(101)                      # gerades und ungerades n
    assert R.poisson_cell(2.0, 10)["non_integer"]["mean"] > 0.25 and R.poisson_cell(2.0, 11)["non_integer"]["mean"] == 0.0     # Ties mit geradem n gegen ungerades n
    assert R.interval_cell("Normal", 10)["share_interval"]["mean"] == 1.0 > R.interval_cell("Poisson", 10)["share_interval"]["mean"]
    assert R.rounding_row(0.0)["rmse_median"]["mean"] < R.rounding_row(1.0)["rmse_median"]["mean"]               # Rundung ändert den Median
    assert R.contamination_row(100.0, 0.49)["median"]["mean"] < 3.0 < R.contamination_row(100.0, 0.5)["median"]["mean"]   # ab einem halben springt der Median
    assert R.sensitivity(101, 0.25)["median"]["mean"] < R.sensitivity(101, 0.5)["median"]["mean"] - 0.03      # die Median-Spalte ist ab t = 0,5 konstant, davor nicht
    assert R.sensitivity(101, 0.5)["median"]["mean"] == pytest.approx(R.sensitivity(101, 1e6)["median"]["mean"], abs=1e-3)


def test_quantile_and_expectile_differ_and_cross():
    assert abs(R.gap_cell("Lognormal(0; 1)", 0.8)["gap"]["mean"]) > 0.3
    signs = {np.sign(R.gap_cell("Exponential", t)["gap"]["mean"]) for t in X.GAP_TAUS}
    assert signs == {-1.0, 1.0}                                                                               # Kreuzung: die Lücke wechselt mit τ das Vorzeichen
    assert abs(R.gap_cell("Normal", 0.5)["gap"]["mean"]) < 0.01                                               # τ = ½: Quantil = Median, Expektile = Mittel, gleich bei Symmetrie


def test_every_distribution_differs_from_normal():
    base = draw("Normal", Stream(5), 20_000)
    for dist in DISTRIBUTIONS:
        if dist == "Normal":
            continue
        x = draw(dist, Stream(5), 20_000, sigma=0.8, lam=0.7)
        assert abs(E.median(x) - E.median(base)) > 0.03 or abs(x.std() - base.std()) > 0.05 or abs(((x - x.mean()) ** 3).mean()) > 0.2, dist


def test_each_stein_dimension_and_norm_cell_differs():
    risks = {(d, nm): R.stein_cell("eine", d, nm)["js"]["mean"] for d in X.STEIN_DIMS for nm in X.STEIN_NORMS}
    for d in X.STEIN_DIMS:
        values = [risks[(d, nm)] for nm in X.STEIN_NORMS]
        assert all(a <= b + 0.01 for a, b in zip(values, values[1:]))                                        # das Verhältnis wächst mit ‖μ‖
        assert values[0] < 0.7 and values[-1] < 1.001
