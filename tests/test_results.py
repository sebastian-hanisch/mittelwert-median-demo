"""Die vorgerechneten Messreihen (data/mtw_results.json): vollständig, aktuell gegenüber den Quelldateien, in sich stimmig und von den Live-Funktionen reproduzierbar."""

import json

import pytest

import mtw_experiments as X
import mtw_results as R


def test_results_file_is_current_with_the_generator_sources():
    """Schlägt fehl, sobald sich mtw_rng, mtw_scenario, mtw_sums, mtw_estimators, mtw_stein, mtw_theory, mtw_experiments oder tools/make_results.py ändern:
    dann alle Teile mit `python tools/make_results.py --part <name>` neu erzeugen und mit `--merge` zusammenfügen."""
    assert R.meta()["erzeuger_hash"] == X.source_hash() and R.is_current()
    assert R.meta()["quellen"] == list(X.SOURCES) and R.meta()["seeds"] == list(X.SEEDS)


def test_results_file_is_valid_utf8_json_with_all_parts():
    text = R.PATH.read_text(encoding="utf-8")
    data = json.loads(text)
    assert set(data) == {"meta"} | set(X.PARTS)
    assert "Lognormal(0; 1,5)" in text and "zufällig" in text                      # Umlaute stehen im Klartext (kein ASCII-Escape)


def test_e1_has_all_methods_and_checkpoints():
    for fmt in ("float32", "float64"):
        e = R.e1(fmt)
        assert e["checks"] == list(X.E1_CHECKS) and set(e["methods"]) == {"floor_sum", "floor_mean"} | set(X.METHODS)
        for name, per_n in e["methods"].items():
            assert list(per_n) == [str(n) for n in X.E1_CHECKS]
            assert all(len(c["per_seed"]) == len(X.SEEDS) for c in per_n.values()), name
    assert R.e1_cell("float32", "naiv", 10_000_000)["mean"] > 20.0 and R.e1_cell("float64", "naiv", 10_000_000)["mean"] < 1e-8


def test_e1_mechanism_diagnosis_and_share_parts():
    assert len(R.e1_mechanism()["cells"]) == 4 and R.e1_mechanism()["n"] == 10_000_000
    assert [r["n"] for r in R.e1_diagnosis()["rows"]] == [10_000, 100_000, 1_000_000]
    assert [r["n"] for r in R.load()["e1_share"]["rows"]] == list(X.NEUMAIER_CHECKS) and R.e1_neumaier_share(2_000_000)["series"] == len(X.NEUMAIER_SEEDS)


def test_e2_parts_are_complete():
    for fmt in ("float32", "float64"):
        r = R.e2_ratio(fmt)
        assert list(r["ratios"]) == [f"{v:g}" for v in X.E2_RATIOS[fmt]] and r["n"] == X.E2_N
        o = R.e2_order(fmt)
        assert set(o["cells"]) == {"Normal", "Lognormal"} and all(set(v) == {"zufällig", "aufsteigend", "absteigend"} for v in o["cells"].values())
    assert [(c["mu"], c["n"]) for c in R.load()["e2_accu"]["cells"]] == [(mu, n) for mu, ns in X.E2_ACCU for n in ns]
    assert R.e2_ratio_cell("float32", 100.0, "naiv")["mean"] > 1.0 and R.e2_order_cell("float32", "Normal", "aufsteigend", "Welford")["mean"] > 100


def test_e3_e5_e6_e7_e8_have_their_rows():
    assert [r["n"] for r in R.e3()["breakdown"]["rows"]] == list(X.BREAKDOWN_NS) and all(set(r["k"]) == {r["ceil_half"]} for r in R.e3()["breakdown"]["rows"])
    assert set(R.e3()["sensitivity"]["n_totals"]) == {"100", "101"} and R.sensitivity(101, 1e6)["mean"]["mean"] == pytest.approx(1e6, abs=0.1)
    assert R.contamination_row(100.0, 0.5)["median"]["mean"] > 50 and R.bias_variance_row(2.0)["cells"]["0,9"]["sim"]["mean"] == pytest.approx(0.85, abs=0.01)
    assert R.efficiency(3) == pytest.approx(0.7429, abs=5e-5) and R.efficiency(4) > R.efficiency(5)
    assert len(R.load()["e5"]["cells"]) == 2 * len(X.STEIN_DIMS) * len(X.STEIN_NORMS) and R.stein_cell("eine", 10, 0.0)["js_exact"] == pytest.approx(0.2)
    assert [float(k) for k in R.load()["e6"]["shapes"]] == list(X.SIGMAS) and R.shape_row(1.0)["gap"] == pytest.approx(1.6487, abs=1e-4)
    assert R.delivery_row(0.8)["exact_missing"] == pytest.approx(0.2739, abs=1e-4) and R.target_row(2.0)["miss_limit"] == pytest.approx(0.8647, abs=1e-4)
    assert set(R.load()["e7"]["poisson"]) == {f"{lam:g}|{n}" for lam in X.TIE_LAMBDAS for n in (10, 11, 100, 101)} and R.rounding_row(1.0)["median_zero"]["mean"] > 0.9
    assert R.lognormal_exact()["rows"][3]["tau"] == 0.8 and R.lognormal_exact()["rows"][3]["e"] == pytest.approx(1.4870, abs=1e-4)
    assert R.gap_labels()[0] == "Normal" and R.gap_cell("Normal", 0.8)["gap"]["mean"] == pytest.approx(-0.217, abs=0.002)


def test_stored_cells_are_reproduced_by_the_live_functions():
    """Das Nachmessen in der App liefert dieselben Zahlen wie die vorgerechnete Zelle (gleiche Seeds)."""
    stored = R.stein_cell("eine", 10, 3.0)
    live = X.stein_cell(10, 3.0, "eine", X.SEEDS[0])
    assert live["pp"] == pytest.approx(stored["pp"]["per_seed"][0], rel=1e-12) and live["koordinate1_js"] == pytest.approx(stored["koordinate1_js"]["per_seed"][0], rel=1e-12)
    r = X.run_e1("float32", checks=(1000, 10_000), seeds=(X.SEEDS[1],))
    for name in ("naiv", "Kahan", "Neumaier", "Welford", "floor_sum"):
        assert r["methods"][name]["10000"]["per_seed"][0] == R.e1_cell("float32", name, 10_000)["per_seed"][1], name
    assert X.rounding_cell(1.0, 40_000, X.SEEDS[2])["rmse_median"] == pytest.approx(R.rounding_row(1.0)["rmse_median"]["per_seed"][2], rel=1e-12)
    assert X.breakdown_cell(6, X.SEEDS[3])["max_shift_below"] == pytest.approx(R.breakdown_row(6)["max_shift_below"]["per_seed"][3], rel=1e-12)
    g = X.gap_cell("Normal", 1.0, X.SEEDS[0])
    assert g["0.8"]["gap"] == pytest.approx(R.gap_cell("Normal", 0.8)["gap"]["per_seed"][0], rel=1e-12)
    assert X.contamination_cell(0.4, 100.0, 4000, X.SEEDS[1])["median"] == pytest.approx(R.contamination_row(100.0, 0.4)["median"]["per_seed"][1], rel=1e-12)
    assert X.delivery_cell(0.8, X.SEEDS[0])["plan_median_missing"] == pytest.approx(R.delivery_row(0.8)["plan_median_missing"]["per_seed"][0], rel=1e-12)


def test_per_seed_values_agree_with_the_summaries():
    import numpy as np
    for cell in (R.e1_cell("float32", "naiv", 1_000_000), R.e1_cell("float64", "Kahan", 100_000)):
        assert cell["mean"] == pytest.approx(np.mean(np.abs(cell["per_seed"]))) and cell["std"] == pytest.approx(np.std(np.abs(cell["per_seed"])))
    c = R.stein_cell("eine", 5, 2.0)["js"]
    assert c["mean"] == pytest.approx(np.mean(c["per_seed"])) and len(c["per_seed"]) == len(X.SEEDS)
