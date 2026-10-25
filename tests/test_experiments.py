"""Messreihen-Funktionen (mtw_experiments.py) an kleinen Größen: Struktur, Handwerte, Zusammenhang mit den Einheiten. Die vorgerechneten Zahlen prüft tests/test_results.py."""

import math

import numpy as np
import pytest

import mtw_experiments as X
import mtw_theory as theory
from mtw_scenario import make_series


def test_measure_methods_hand_instance(tiny_f32):
    res = X.measure_methods(tiny_f32, "float32", [3])
    assert set(res) == {"floor_sum", "floor_mean"} | set(X.METHODS)
    assert res["naiv"][3] == pytest.approx(-2 / 3)               # Summe 2²⁴ statt 2²⁴ + 2, durch 3
    assert res["Kahan"][3] == 0.0 and res["Kahan mit Rest"][3] == 0.0
    assert res["Neumaier"][3] == 0.0 and res["Neumaier mit Rest"][3] == 0.0


def test_measure_methods_checkpoints_equal_separate_runs():
    x = make_series(6000, 1000.0, "Normal", "zufällig", 5, "float32")
    both = X.measure_methods(x, "float32", [2000, 6000])
    alone = X.measure_methods(x[:2000], "float32", [2000])
    for name in X.METHODS:
        if name == "numpy.sum":
            assert both[name][2000] == pytest.approx(alone[name][2000], abs=1e-7)
        else:
            assert both[name][2000] == alone[name][2000], name
    assert both["floor_sum"][2000] == alone["floor_sum"][2000]


def test_measure_methods_float64_is_much_smaller_than_float32():
    x = make_series(50_000, 1000.0, "Normal", "zufällig", 5, "float32")
    x64 = make_series(50_000, 1000.0, "Normal", "zufällig", 5, "float64")
    a, b = X.measure_methods(x, "float32", [50_000]), X.measure_methods(x64, "float64", [50_000])
    assert abs(b["naiv"][50_000]) < 1e-9 < abs(a["floor_sum"][50_000]) * 1e5 and abs(a["floor_mean"][50_000]) > 1e-9


def test_e1_structure_and_reproducibility():
    r = X.run_e1("float32", checks=(1000, 5000), seeds=(11, 12))
    assert r["checks"] == [1000, 5000] and r["fmt"] == "float32"
    assert set(r["methods"]) == {"floor_sum", "floor_mean"} | set(X.METHODS)
    cell = r["methods"]["naiv"]["5000"]
    assert len(cell["per_seed"]) == 2 and cell["mean"] == pytest.approx(np.mean(np.abs(cell["per_seed"]))) and cell["std"] == pytest.approx(np.std(np.abs(cell["per_seed"])))
    assert X.run_e1("float32", checks=(1000, 5000), seeds=(11, 12)) == r


def test_e1_mechanism_on_one_million_values():
    r = X.run_e1_mechanism(seeds=(11,), n=1_000_000)
    c = r["cells"][0]
    assert 530_000 < c["first_index"] < 550_000 and c["share_1024"] > 0.999 and abs(c["predicted_mean"] - c["measured_mean"]) < 1e-2
    assert c["increment_min"] > 0 and c["increment_max"] < 4096


def test_e1_diagnosis_small():
    r = X.run_e1_diagnosis(checks=(2000, 20_000))
    last = r["rows"][-1]
    assert last["n"] == 20_000 and last["neumaier"]["max_rest"] > 10 * last["kahan"]["max_rest"] and abs(last["neumaier_wide"]["sum_error"]) < abs(last["neumaier"]["sum_error"]) + 1e-12


def test_e1_neumaier_share_counts_add_up():
    r = X.run_e1_neumaier_share(seeds=(101, 102, 103), checks=(100_000,))
    row = r["rows"][0]
    assert row["series"] == 3 and row["worse"] + row["equal"] + row["better"] <= 3 and row["median_kahan"] > 0


def test_e2_ratio_order_and_accumulator_small():
    r = X.run_e2_ratio("float32", seeds=(11,), n=2000)
    assert set(r["ratios"]) == {f"{v:g}" for v in X.E2_RATIOS["float32"]} and r["se"] == pytest.approx(1 / math.sqrt(2000))
    assert r["ratios"]["0"]["methods"]["naiv"]["mean"] < 1.0 < r["ratios"][f"{1e7:g}"]["methods"]["naiv"]["mean"]
    o = X.run_e2_order("float64", seeds=(11,), n=2000)
    assert set(o["cells"]) == {"Normal", "Lognormal"} and set(o["cells"]["Normal"]) == {"zufällig", "aufsteigend", "absteigend"}
    a = X.run_e2_accumulator(seeds=(11,), grid=((1e6, (2000,)),))
    cell = a["cells"][0]
    assert cell["mu"] == 1e6 and cell["n"] == 2000 and cell["float64_accumulator_rounded"]["mean"] <= cell["float32"]["mean"] + 1e-12


def test_sensitivity_cell_small():
    c = X.sensitivity_cell(21, ts=(0.5, 5.0, 1e6), runs=4000, seed=11)
    assert c["5"]["mean_dev"] < 1e-6 and c["1e+06"]["mean"] == pytest.approx(1e6, abs=0.1)
    assert c["5"]["median"] == c["1e+06"]["median"] and 0.9 < c["5"]["median"] < 1.8


def test_contamination_cell_small():
    c = X.contamination_cell(0.2, 100.0, 500, 11)
    assert c["k"] == 20 and c["mean"] == pytest.approx(20.0, abs=0.05) and c["median"] < 1.0
    assert X.contamination_cell(0.0, 100.0, 50, 11) == {"k": 0, "mean": 0.0, "median": 0.0}
    assert X.contamination_cell(0.6, 100.0, 200, 11)["median"] == pytest.approx(100.0, abs=0.01)         # ab 55 % liegt der Median auf den Ausreißern


def test_bias_variance_cell_follows_the_formula():
    for theta, c in ((0.0, 0.9), (2.0, 0.8), (4.0, 0.5)):
        mu = theta / math.sqrt(10)
        formula = (c * c / 10 + (1 - c) ** 2 * mu * mu) * 10
        assert X.bias_variance_cell(theta, c, 100_000, 11) == pytest.approx(formula, rel=0.03)


def test_breakdown_cell_is_ceil_half():
    for n in (3, 4, 5, 6, 7):
        c = X.breakdown_cell(n, 11)
        assert c["k"] == math.ceil(n / 2) and c["mean_k"] == 1 and c["max_shift_below"] > 0.0


def test_efficiency_and_stein_cells():
    e = X.run_e3_efficiency()
    assert e["odd"]["3"] == pytest.approx(0.7429, abs=5e-5) and e["even"]["2"] == pytest.approx(1.0) and e["limit"] == pytest.approx(2 / math.pi)
    c = X.stein_cell(10, 0.0, "eine", 11, 20_000)
    assert c["js"] == pytest.approx(0.2, abs=0.02) and c["pp"] < c["js"] and c["einzel"] == pytest.approx(1.0, abs=0.03) and "gemeinsames_mittel" in c
    assert "gemeinsames_mittel" not in X.stein_cell(3, 1.0, "gleich", 11, 1000)
    assert X.zero_share(10, runs=40_000) == pytest.approx(0.372, abs=0.01)


def test_e6_cells():
    t = X.target_errors(1.0, 200, 400, 11)
    assert t["median_mittelwert"]["typisch"] == pytest.approx(theory.median_miss_share(1.0), abs=0.05) and t["mittel_median"]["q95"] > t["mittel_median"]["typisch"]
    s = X.shape_cell(1.0, 11, count=100_000)
    assert s["below_mean"] == pytest.approx(theory.share_below_mean(1.0), abs=0.01) and 0.3 < s["top10_share"] < 0.5
    d = X.delivery_cell(0.8, 11)
    assert d["plan_median_missing"] == pytest.approx(theory.median_miss_share(0.8), abs=0.01) and d["schaetzung_median"]["zu_klein"] > 0.9 > d["schaetzung_mittel"]["zu_klein"] > 0.3


def test_e7_cells():
    p = X.poisson_median_cell(0.7, 100, 2000, 11)
    assert p["is0"] + p["is_half"] + p["is1"] + p["ge15"] == pytest.approx(1.0, abs=1e-12) and p["non_integer"] >= p["is_half"] > 0.0
    assert X.poisson_median_cell(2.0, 11, 500, 11)["non_integer"] == 0.0                                  # ungerades n: der Median ist ein Beobachtungswert
    i = X.interval_cell("Normal", 4, 500, 11)
    assert i["share_interval"] == 1.0 and 0.3 < i["mean_width"] < 0.9
    assert X.interval_cell("Normal", 5, 100, 11) == {"share_interval": 0.0, "mean_width": 0.0}
    r = X.rounding_cell(1.0, 500, 11)
    assert r["rmse_median"] > r["rmse_mean"] and r["median_zero"] > 0.8
    assert X.rounding_cell(0.0, 500, 11)["median_zero"] == 0.0


def test_e8_gap_cell_has_the_expected_sign():
    g = X.gap_cell("Normal", 1.0, 11, count=50_000)
    assert g["0.8"]["gap"] == pytest.approx(-0.217, abs=0.02) and g["0.5"]["gap"] == pytest.approx(0.0, abs=0.02)
    assert 0.0 < g["0.8"]["F_e"] < g["0.99"]["F_e"] < 1.0


def test_source_hash_is_independent_of_line_endings(tmp_path, monkeypatch):
    for name in X.SOURCES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((X.ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))
    expected = X.source_hash()
    monkeypatch.setattr(X, "ROOT", tmp_path)
    assert X.source_hash() == expected


def test_parts_cover_the_documented_experiments():
    assert set(X.PARTS) == {"e1_f32", "e1_f64", "e1_mech", "e1_diag", "e1_share", "e2_ratio_f32", "e2_ratio_f64", "e2_order_f32", "e2_order_f64", "e2_accu", "e3", "e5", "e6", "e7", "e8"}
    assert X.SEEDS == (11, 12, 13, 14) and "tools/make_results.py" in X.SOURCES
