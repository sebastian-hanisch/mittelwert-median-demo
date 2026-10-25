"""Figuren: Spuren, feste Achsen, Randfälle (Nullfehler auf logarithmischer Achse, unsichere Zellen)."""

import numpy as np
import pytest

import mtw_constants as C
import mtw_evaluation as V
import mtw_experiments as X
import mtw_results as R
import mtw_visualization as W


def _locked(fig):
    return all(a.fixedrange for a in (fig.layout.xaxis, fig.layout.yaxis))


def test_method_bars_colour_by_threshold_and_clip_zero_errors():
    errors = {"naiv": 8.9, "Kahan": 1.1e-5, "Kahan mit Rest": 0.0, "Untergrenze": 1.1e-5}
    fig = W.build_method_bars(errors, 1_000_000)
    bar = fig.data[0]
    assert list(bar.y) == ["Kahan mit Rest", "Kahan", "naiv"] and min(bar.x) >= W.LOG_FLOOR
    assert list(bar.marker.color) == [W.C.COLOR_GOOD, W.C.COLOR_GOOD, W.C.COLOR_BAD]            # Kahan genau an der Untergrenze: grün; naiv: rot
    assert bar.x[2] == pytest.approx(8.9 * 1000)                                                # in Standardfehlern (σ/√n = 10⁻³)
    assert fig.layout.xaxis.type == "log" and _locked(fig)
    assert any(a.text == "0" for a in fig.layout.annotations)                                   # die Null wird beschriftet, nicht verschluckt


def test_method_bars_for_the_live_default_has_one_bar_per_method():
    rr = V.rechnen_analysis(V.Settings(n=10_000))
    fig = W.build_method_bars(rr.abs_errors, 10_000)
    assert len(fig.data[0].y) == len(X.METHODS)


def test_error_curve_and_ratio_chart_use_log_axes_and_stored_rows():
    e = R.e1("float32")
    fig = W.build_error_curve(e["methods"], e["checks"], ("naiv", "Kahan", "Neumaier"), "float32")
    names = [t.name for t in fig.data]
    assert names == ["naiv", "Kahan", "Neumaier", "Untergrenze (Summe im Format)", "Standardfehler σ/√n"] and fig.layout.xaxis.type == fig.layout.yaxis.type == "log" and _locked(fig)
    assert min(fig.data[1].y) >= W.LOG_FLOOR
    r = R.e2_ratio("float64")
    fig = W.build_ratio_chart(r["ratios"], ("naiv", "Kahan"), "float64")
    assert len(fig.data) == 2 and len(fig.data[0].x) == len(X.E2_RATIOS["float64"]) and min(fig.data[0].x) == 0.5 and _locked(fig)


def test_sample_chart_has_four_lines_with_labels():
    x = np.random.default_rng(1).normal(0, 1, 200)
    fig = W.build_sample_chart(x, 0.1, 0.0, 0.8, 0.7, 0.8)
    assert len(fig.layout.shapes) == 4 and sorted(a.text for a in fig.layout.annotations) == ["Expektile 0,80", "Median", "Mittel", "Quantil 0,80"] and _locked(fig)


def test_loss_chart_has_a_minimum_marker_in_each_panel_at_the_given_centre():
    x = np.random.default_rng(2).lognormal(0, 0.5, 300)
    mean, med, q, e = float(x.mean()), float(np.median(x)), float(np.quantile(x, 0.8)), 1.3
    fig = W.build_loss_chart(x, 0.8, mean, med, q, e)
    markers = [t for t in fig.data if t.mode == "markers"]
    assert len(fig.data) == 8 and [m.x[0] for m in markers] == [mean, med, q, e]
    curve = fig.data[0]
    assert 0 <= markers[0].y[0] <= min(curve.y)                    # die Raute liegt im Minimum der Quadratsumme, unter jedem Rasterpunkt der Kurve


def test_plane_chart_marks_miscount_and_centres():
    pr = V.plane_analysis(V.Settings(miscount=20.0))
    fig = W.build_plane_chart(pr)
    names = [t.name for t in fig.data if t.name]
    assert names == ["Kunden", "Fehlbuchung", "Schwerpunkt ohne Fehlbuchung", "Schwerpunkt mit Fehlbuchung", "Median je Achse ohne", "Median je Achse mit"]
    assert fig.layout.xaxis.scaleanchor == "y" and fig.layout.xaxis.range is None and _locked(fig)
    assert fig.data[1].x[0] == pytest.approx(20 / np.sqrt(2))


def test_stein_charts_use_exact_line_and_simulated_points():
    cells = {(d, nm): {"pp": R.stein_cell("eine", d, nm)["pp"]["mean"], "js_exact": R.stein_cell("eine", d, nm)["js_exact"], "koordinate1_pp": R.stein_cell("eine", d, nm)["koordinate1_pp"]["mean"]}
             for d in X.STEIN_DIMS for nm in X.STEIN_NORMS}
    fig = W.build_stein_chart(cells, X.STEIN_DIMS, X.STEIN_NORMS)
    assert len(fig.data) == 2 * len(X.STEIN_DIMS) and fig.data[0].mode == "lines" and fig.data[1].mode == "markers" and _locked(fig)
    assert fig.data[0].y[0] == pytest.approx(2 / 3)                                              # d = 3, ‖μ‖ = 0
    fig = W.build_coordinate_chart(cells, X.STEIN_DIMS, X.STEIN_NORMS)
    assert len(fig.data) == len(X.STEIN_DIMS) and fig.layout.yaxis.type == "log"


def test_e3_charts():
    ts = list(X.SC_TS)
    fig = W.build_influence_chart(ts, [R.sensitivity(101, t)["median"]["mean"] for t in ts], [R.sensitivity(101, t)["mean"]["mean"] for t in ts])
    assert [t.name for t in fig.data] == ["Median", "Mittel"] and fig.layout.xaxis.type == "log" and fig.data[0].y[-1] == fig.data[0].y[-2]
    rows = [r for r in R.e3()["contamination"]["rows"] if r["M"] == 100.0 and r["eps"] > 0]
    assert len(W.build_contamination_chart(rows).data) == 2
    eff = R.e3()["efficiency"]
    fig = W.build_efficiency_chart(eff["odd"], eff["even"], eff["limit"])
    assert [t.name for t in fig.data] == ["ungerades n", "gerades n"] and list(fig.data[0].x) == sorted(int(k) for k in eff["odd"])


def test_target_gap_and_rounding_charts():
    assert len(W.build_target_chart(R.load()["e6"]["targets"]["rows"]).data) == 4
    gaps = R.load()["e8"]["gaps"]
    labels = R.gap_labels()
    fig = W.build_gap_chart(gaps, X.GAP_TAUS, labels)
    assert len(fig.data) == len(labels)
    unusable = [(lb, t) for lb in labels for t in X.GAP_TAUS if not gaps[lb][f"{t:g}"]["usable"]]
    lb, t = unusable[0]
    assert t not in fig.data[labels.index(lb)].x                                                  # unsichere Zellen werden ausgelassen
    assert len(W.build_rounding_chart(R.load()["e7"]["rounding"]["rows"]).data) == 2


def test_lock_axes_and_base_layout_apply_everywhere():
    for fig in (W.build_method_bars({"naiv": 1.0, "Untergrenze": 1.0}, 100), W.build_rounding_chart(R.load()["e7"]["rounding"]["rows"])):
        assert _locked(fig) and fig.layout.height and fig.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert C.COLOR_BAD != C.COLOR_GOOD
