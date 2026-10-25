"""Worst-Case am Slider-Maximum (DEMO-PLAYBOOK §1c): fängt Hänger und unerwartetes Aufblähen, prüft keine Geschwindigkeit (nur ein großzügiges Netz)."""

import time

import mtw_constants as C
import mtw_evaluation as V


def test_largest_series_in_both_formats_finishes_within_a_generous_limit():
    for fmt in C.FORMATS:
        t0 = time.perf_counter()
        r = V.rechnen_analysis(V.Settings(fmt=fmt, n=C.LIVE_N_MAX, ratio=C.RATIO_OPTIONS[-1], order="absteigend", rdist="Lognormal"))
        assert time.perf_counter() - t0 < 180 and all(v == v for v in r.abs_errors.values())


def test_largest_sample_and_stein_run_finish_within_a_generous_limit():
    t0 = time.perf_counter()
    V.sample_analysis(V.Settings(dist="t(5)", wn=max(C.WN_OPTIONS), out=C.OUT_MAX, h=C.H_OPTIONS[-1]))
    import mtw_experiments as X
    X.stein_cell(max(C.STEIN_D_OPTIONS), C.STEIN_NORM_MAX, "eine", 1, C.STEIN_LIVE_RUNS)
    assert time.perf_counter() - t0 < 60
