"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, und was jedes Preset zeigt (Abstimmung vor den Hilfetexten, DEMO-PLAYBOOK §0)."""

import math
from dataclasses import fields

import pytest

import mtw_constants as C
import mtw_evaluation as V
import mtw_experiments as X
import mtw_presets as P


def settings_of(name):
    return V.Settings(**C.PRESETS[name])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert len(C.PRESET_HELP[name]) > 60 and "folgt" not in C.PRESET_HELP[name]


def test_preset_values_are_inside_the_allowed_values():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.choices is not None:
                assert preset[key] in spec.choices, (name, key)
            else:
                assert spec.lo <= preset[key] <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    assert V.Settings(**next(iter(C.PRESETS.values()))) == V.Settings()
    assert [f.name for f in fields(V.Settings)] == list(P.PRESET_KEYS)


def test_setting_specs_cover_every_preset_key_and_defaults_match_the_base():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == C._BASE[key]


def test_settings_are_keyword_only():
    with pytest.raises(TypeError):
        V.Settings("float32")                                    # eingefügte Felder dürfen nie still Werte verschieben


def test_live_limits_come_from_the_constants():
    assert C.LIVE_N_MAX == max(C.N_OPTIONS) and C.LIVE_N_MAX <= 5_000_000
    assert max(C.WN_OPTIONS) <= 10_000 and C.STEIN_LIVE_RUNS * max(C.STEIN_D_OPTIONS) <= 5_000_000


# ---- was die Rechnen-Presets zeigen --------------------------------------------------------------------------------------------------------

def test_preset_standard_naive_sum_is_far_off_and_kahan_is_not():
    rr = V.rechnen_analysis(settings_of(next(iter(C.PRESETS))))
    assert rr.abs_errors["naiv"] > 1000 * rr.abs_errors["Kahan"] and rr.abs_errors["naiv"] * math.sqrt(rr.settings.n) > 1000      # tausende Standardfehler
    assert rr.abs_errors["Kahan"] <= 4 * rr.abs_errors["Untergrenze"] and rr.abs_errors["Binärbaum"] <= 4 * rr.abs_errors["Untergrenze"]


def test_preset_five_million_naive_error_is_in_the_tens():
    rr = V.rechnen_analysis(settings_of("Fünf Millionen Zeitstempel"))
    assert rr.settings.n == 5_000_000 and 15.0 < rr.abs_errors["naiv"] < 30.0 and rr.abs_errors["Kahan"] < 1e-3


def test_preset_neumaier_loses_to_kahan_on_this_series():
    rr = V.rechnen_analysis(settings_of("Neumaier verliert"))
    assert rr.abs_errors["Neumaier"] > 10 * rr.abs_errors["Kahan"] and rr.abs_errors["Kahan"] <= 4 * rr.abs_errors["Untergrenze"]


def test_preset_sorted_ruins_welford():
    rr = V.rechnen_analysis(settings_of("Sortiert (Welford)"))
    assert rr.settings.order == "aufsteigend" and rr.abs_errors["Welford"] / rr.standard_error > 500 and rr.abs_errors["Kahan"] / rr.standard_error < 0.1


def test_preset_quantisation_is_limited_by_the_format_for_every_method():
    rr = V.rechnen_analysis(settings_of("Quantisierung"))
    assert rr.abs_errors["Untergrenze"] / rr.standard_error > 2.0                               # die Untergrenze selbst liegt über zwei Standardfehlern
    assert rr.abs_errors["Kahan"] == rr.abs_errors["Untergrenze"] and rr.distinct < 500


# ---- was die Wählen-Presets zeigen ----------------------------------------------------------------------------------------------------------

def test_preset_plan_times_median_leaves_out_a_quarter():
    a = V.sample_analysis(settings_of("Plan mal Median"))
    assert 0.2 < V.plan_missing(a.x) < 0.35 and a.mean > 1.2 * a.median


def test_preset_near_tie_median_sits_below_the_population_median():
    a = V.sample_analysis(settings_of("Fast-Gleichstand"))
    assert a.median in (0.0, 0.5) and a.x.min() == 0 and set(a.x) <= set(range(10))


def test_preset_ten_percent_outliers_moves_the_mean_not_the_median():
    a = V.sample_analysis(settings_of("Zehn Prozent Ausreißer"))
    assert a.outliers == 10 and a.mean > 9.0 and abs(a.median) < 0.5


def test_preset_rounded_data_median_stays_on_the_raster():
    a = V.sample_analysis(settings_of("Gerundete Daten"))
    assert a.median == 0.0 and set(a.x) <= {float(k) for k in range(-4, 6)} and 0.1 < a.mean < 0.6


def test_preset_stock_quantile_is_well_above_the_mean():
    a = V.sample_analysis(settings_of("Mittel gegen Quantil im Bestand"))
    assert a.quantile[0] - a.mean > 1.0 and a.truth is not None and abs(a.truth["expectile"] - a.expectile) < 0.5


def test_stein_presets_show_the_total_gain_and_the_single_loss():
    s10 = settings_of("Stein, d = 10")
    c = X.stein_cell(s10.stein_d, s10.stein_norm, "eine", s10.seed, C.STEIN_LIVE_RUNS)
    assert c["js"] == pytest.approx(0.2, abs=0.02)
    s50 = settings_of("Stein, eine Koordinate")
    c = X.stein_cell(s50.stein_d, s50.stein_norm, "eine", s50.seed, C.STEIN_LIVE_RUNS)
    assert c["koordinate1_pp"] > 5.0 and c["pp"] < 0.7
