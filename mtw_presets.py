"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import mtw_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None      # bei Auswahlreglern: die erlaubten Werte


SETTING_SPECS = {
    "fmt_select": SettingSpec("fmt", str, C._BASE["fmt"], choices=C.FORMATS),
    "n_select": SettingSpec("n", int, C.DEFAULT_N, choices=C.N_OPTIONS),
    "ratio_select": SettingSpec("ratio", int, C.DEFAULT_RATIO, choices=C.RATIO_OPTIONS),
    "order_select": SettingSpec("order", str, C._BASE["order"], choices=C.ORDERS),
    "rdist_select": SettingSpec("rdist", str, C._BASE["rdist"], choices=C.SERIES_DISTRIBUTIONS),
    "dist_select": SettingSpec("dist", str, C._BASE["dist"], choices=C.DISTRIBUTIONS),
    "sigma_slider": SettingSpec("sigma", float, C.DEFAULT_SIGMA, C.SIGMA_MIN, C.SIGMA_MAX),
    "lam_slider": SettingSpec("lam", float, C.DEFAULT_LAM, C.LAM_MIN, C.LAM_MAX),
    "wn_select": SettingSpec("wn", int, C.DEFAULT_WN, choices=C.WN_OPTIONS),
    "out_slider": SettingSpec("out", int, C.DEFAULT_OUT, C.OUT_MIN, C.OUT_MAX),
    "outdist_select": SettingSpec("outd", int, C.DEFAULT_OUTDIST, choices=C.OUTDIST_OPTIONS),
    "h_select": SettingSpec("h", float, C._BASE["h"], choices=C.H_OPTIONS),
    "shift_slider": SettingSpec("shift", float, C.DEFAULT_SHIFT, C.SHIFT_MIN, C.SHIFT_MAX),
    "tau_slider": SettingSpec("tau", float, C.DEFAULT_TAU, C.TAU_MIN, C.TAU_MAX),
    "miscount_slider": SettingSpec("mis", float, C.DEFAULT_MISCOUNT, C.MISCOUNT_MIN, C.MISCOUNT_MAX),
    "stein_d_select": SettingSpec("sd", int, C.DEFAULT_STEIN_D, choices=C.STEIN_D_OPTIONS),
    "stein_norm_slider": SettingSpec("sn", float, C.DEFAULT_STEIN_NORM, C.STEIN_NORM_MIN, C.STEIN_NORM_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {
    "fmt": "fmt_select", "n": "n_select", "ratio": "ratio_select", "order": "order_select", "rdist": "rdist_select", "dist": "dist_select", "sigma": "sigma_slider",
    "lam": "lam_slider", "wn": "wn_select", "out": "out_slider", "outdist": "outdist_select", "h": "h_select", "shift": "shift_slider", "tau": "tau_slider",
    "miscount": "miscount_slider", "stein_d": "stein_d_select", "stein_norm": "stein_norm_slider", "seed": "seed_input",
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf erlaubte Werte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.choices is not None:
                if value not in spec.choices:
                    continue
            else:
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
