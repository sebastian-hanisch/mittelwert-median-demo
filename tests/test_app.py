"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Button einzeln, fährt jeden Regler an seine Grenzen
(einzeln und alle zusammen) und prüft, dass kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe
ohne eindeutiges key= können zufällig identischen Inhalt rendern und kollidieren) und `StreamlitAPIException` bei Reglern mit berechneten Grenzen."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import mtw_constants as C
import mtw_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _widget(at, key):
    for art in ("slider", "select_slider", "selectbox", "number_input", "radio"):
        try:
            return getattr(at.sidebar if key in P.SETTING_SPECS else at, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def test_app_loads_without_exception_and_shows_both_tabs():
    at = _fresh_app()
    assert [t.label for t in at.tabs][:2] == ["🧮 Rechnen", "🎯 Wählen"]
    assert len(at.metric) >= 8 and len(at.dataframe) > 10


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    assert any("Nachmessen" in b for b in labels) and any("PDF" in b for b in labels) and any("Durchschlag" in b for b in labels)
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_live_buttons_show_their_result():
    at = _fresh_app()
    for key, marker in (("e5_live", "Nachgemessen (d = "), ("e6_live", "Nachgemessen (σ = "), ("e3_live", "kleinste Zahl ersetzter Werte")):
        at.button(key=key).click().run(timeout=TIMEOUT)
        assert not at.exception, key
        assert any(marker in i.value for i in at.info), key
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception and at.get("download_button")


@pytest.mark.parametrize("state_key", [k for k, s in P.SETTING_SPECS.items() if s.choices is None and k != "seed_input"])
def test_every_slider_at_its_min_and_max_does_not_raise(state_key):
    for edge in ("min", "max"):
        at = _fresh_app()
        if state_key == "sigma_slider":
            _widget(at, "dist_select").set_value("Lognormal")
        if state_key == "lam_slider":
            _widget(at, "dist_select").set_value("Poisson").run(timeout=TIMEOUT)
        slider = _widget(at, state_key)
        value = slider.min if edge == "min" else slider.max
        slider.set_value(value).run(timeout=TIMEOUT)
        assert not at.exception, (state_key, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("state_key", [k for k, s in P.SETTING_SPECS.items() if s.choices is not None])
def test_every_choice_regler_with_every_option_does_not_raise(state_key):
    options = P.SETTING_SPECS[state_key].choices
    for option in (options[0], options[-1]) if len(options) > 4 else options:
        at = _fresh_app()
        if state_key == "outdist_select":                                  # gesperrt, solange es keine Ausreißer gibt: erst Ausreißer wählen
            _widget(at, "out_slider").set_value(10).run(timeout=TIMEOUT)
        _widget(at, state_key).set_value(option).run(timeout=TIMEOUT)
        assert not at.exception, (state_key, option, [str(e) for e in at.exception])


def test_all_minima_and_all_maxima_together():
    for edge in (0, -1):
        at = _fresh_app()
        # erst die Regler setzen, die andere freischalten (Verteilung, Ausreißeranteil), dann die übrigen; gesperrte Regler bleiben, wie sie sind
        for key in ("dist_select", "out_slider"):
            spec = P.SETTING_SPECS[key]
            _widget(at, key).set_value(spec.choices[edge] if spec.choices is not None else (spec.lo if edge == 0 else spec.hi))
        at.run(timeout=TIMEOUT)
        for key, spec in P.SETTING_SPECS.items():
            if key in ("seed_input", "dist_select", "out_slider"):
                continue
            w = _widget(at, key)
            if w.disabled:
                continue
            w.set_value(spec.choices[edge] if spec.choices is not None else (spec.lo if edge == 0 else spec.hi))
        at.run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])


def test_seed_extremes_and_random_button():
    at = _fresh_app()
    for v in (0, C.SEED_MAX):
        at.number_input(key="seed_input").set_value(v).run(timeout=TIMEOUT)
        assert not at.exception
    before = at.number_input(key="seed_input").value
    at.sidebar.button[-1].click().run(timeout=TIMEOUT)
    assert not at.exception and at.number_input(key="seed_input").value != before


def test_permalink_sets_and_limits_values():
    at = AppTest.from_file(APP_PATH)
    at.query_params["n"] = "5000000"
    at.query_params["fmt"] = "float64"
    at.query_params["tau"] = "7"            # über der Grenze: wird begrenzt
    at.query_params["ratio"] = "123"        # kein erlaubter Wert: wird verworfen
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.select_slider(key="n_select").value == 5_000_000 and at.selectbox(key="fmt_select").value == "float64"
    assert at.slider(key="tau_slider").value == C.TAU_MAX and at.select_slider(key="ratio_select").value == C.DEFAULT_RATIO
