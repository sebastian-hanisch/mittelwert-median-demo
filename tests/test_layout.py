"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt
sicher, dass wirklich JEDER Regler der Seitenleiste dort steht: sonst bliebe das Preset trotz Änderung markiert."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import mtw_constants as C
import mtw_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_der_seitenleiste_steht_in_setting_specs():
    at = _app()
    schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    fehlt = schluessel - set(P.SETTING_SPECS)
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Preset nicht auf"
    assert set(P.SETTING_SPECS) <= schluessel, "ein Eintrag in SETTING_SPECS hat keinen Regler"


def _anderer_wert(regler, state_key):
    """Ein Wert, der vom aktuellen abweicht und im Wertebereich des Reglers liegt (Auswahlregler: eine andere Option aus SETTING_SPECS, Schieberegler: ein Rand)."""
    choices = P.SETTING_SPECS[state_key].choices
    if choices is not None:
        return next(c for c in reversed(choices) if c != regler.value)
    return regler.max if regler.value != regler.max else regler.min


# Regler, die nur bei bestimmten Einstellungen wirken (sonst gesperrt): das Preset, von dem aus ihre Änderung geprüft wird
START_PRESET = {"lam_slider": "Fast-Gleichstand", "outdist_select": "Zehn Prozent Ausreißer"}


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    start = START_PRESET.get(state_key, next(iter(C.PRESETS)))
    at.sidebar.button(key=f"preset_{start}").click().run()
    assert _typen(at)[start] == "primary"
    regler = _regler(at, state_key)
    assert not regler.disabled, f"{state_key} ist gesperrt; Start-Preset anpassen"
    regler.set_value(_anderer_wert(regler, state_key)).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert _typen(at)[start] != "primary"                       # das Start-Preset ist nicht mehr hervorgehoben (ein anderes darf es sein, wenn der neue Wert genau dessen Einstellung trifft)
    # erneutes Anklicken markiert wieder
    at.sidebar.button(key=f"preset_{start}").click().run()
    assert _typen(at)[start] == "primary"
