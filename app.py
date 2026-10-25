"""Mittelwert und Median: Welche Zahl will ich? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Wurzel der Reihe „Mittelwerte und Lagemaße“. Lauffähig mit: streamlit run app.py
"""

import math
import time

import streamlit as st

import mtw_constants as C
import mtw_evaluation as V
import mtw_experiments as X
import mtw_results as R
import mtw_stein as stein
import mtw_theory as theory
from mtw_layout import dunkles_design, kopf_css
from mtw_pdf_export import build_pdf
from mtw_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from mtw_visualization import (
    build_contamination_chart,
    build_coordinate_chart,
    build_efficiency_chart,
    build_error_curve,
    build_gap_chart,
    build_influence_chart,
    build_loss_chart,
    build_method_bars,
    build_plane_chart,
    build_ratio_chart,
    build_rounding_chart,
    build_sample_chart,
    build_stein_chart,
    build_target_chart,
)

st.set_page_config(page_title="Mittelwert und Median – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _rechnen(settings):
    return V.rechnen_analysis(settings)


@st.cache_data(show_spinner=False)
def _sample(settings):
    return V.sample_analysis(settings)


@st.cache_data(show_spinner=False)
def _plane(settings):
    return V.plane_analysis(settings)


@st.cache_data(show_spinner=False)
def _stein_live(d, norm, mode, seed, runs):
    return X.stein_cell(d, norm, mode, seed, runs)


@st.cache_data(show_spinner=False)
def _targets_live(sigma, n, runs, seed):
    return X.target_errors(sigma, n, runs, seed)


@st.cache_data(show_spinner=False)
def _breakdown_live(n, seed):
    return X.breakdown_cell(n, seed)


def _verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


def _thousands(v):
    return f"{v:,}".replace(",", ".")


def _signed(v):
    return "0" if v == 0 else ("+" if v > 0 else "−") + V.sci(abs(v))


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in mtw_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📊 Mittelwert und Median: Welche Zahl will ich?")
st.markdown(
    """
**Wurzel der Reihe „Mittelwerte und Lagemaße“.** Mittel und Median sind **Minimierer**: das Mittel der Summe der Abstände zum Quadrat, der Median der Summe der Beträge, Quantil und Expektile ihre schiefen Verwandten.
Die Demo hat zwei Reiter. **🧮 Rechnen:** Wie genau liefert der Rechner das Mittel von bis zu fünf Millionen Zeitstempeln oder Koordinaten, wenn die Summe fortlaufend in float32 entsteht?
**🎯 Wählen:** Welche Zahl will ich überhaupt (Summe planen oder typischen Fall nennen), was kostet es, eine andere zu schätzen, und wann zieht die Schrumpfung nach James und Stein das Gesamtrisiko nach unten, einzelne Lager aber nach oben?
Abgrenzung: [kmeans-demo](https://github.com/sebastian-hanisch/kmeans-demo) (Mittelwert gegen Medoid bei Ausreißern), [exponential-smoothing-demo](https://github.com/sebastian-hanisch/exponential-smoothing-demo) (festes Gedächtnis statt Mittel aller),
`rangintervalle-demo` (Schrumpfung bei Bradley-Terry-Stärken), [forecast-combination-demo](https://github.com/sebastian-hanisch/forecast-combination-demo) (Median als Kombinationsregel),
[verweilzeit-demo](https://github.com/sebastian-hanisch/verweilzeit-demo) (Median oder Quantil?), [newsvendor-evpi-demo](https://github.com/sebastian-hanisch/newsvendor-evpi-demo) und [forecast-inventory-demo](https://github.com/sebastian-hanisch/forecast-inventory-demo) (Bestand als Quantil).

Weiter unten: die Experimente **🔬 E1, E2** (Reiter Rechnen) und **E5, E6, E8** sowie die kurzen Tabs **E3, E7** (Reiter Wählen), **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("Was Mittel, Median, Quantil und Expektile minimieren", expanded=True):
    st.markdown(
        r"""
Gegeben sind Werte $x_1, \dots, x_n$. Jede der vier „Mitten“ ist der Punkt $c$, der eine **Verlustsumme** möglichst klein macht:

| Mitte | Verlust je Wert | Eigenschaft |
|---|---|---|
| **Mittel** (Schwerpunkt) | $(x_i - c)^2$ | jeder Wert zieht linear mit seinem Abstand, ein Ausreißer beliebig stark |
| **Median** | $\lvert x_i - c\rvert$ | jeder Wert zieht gleich stark, egal wie weit er liegt |
| **Quantil** zur Stufe $\tau$ | $\tau\,(x_i - c)_+ + (1-\tau)\,(c - x_i)_+$ (Pinball) | unterschiedlich starker Zug nach oben und unten |
| **Expektile** zur Stufe $\tau$ | $\tau\,(x_i - c)_+^2 + (1-\tau)\,(c - x_i)_+^2$ | wie das Mittel, aber nach oben und unten verschieden gewichtet |

Bei $\tau = \tfrac12$ ist das Quantil der Median und die Expektile das Mittel. Für Punkte in der Ebene ist das Mittel der **Schwerpunkt**, und die Quadratsumme zerfällt nach Huygens in die Streuung um den Schwerpunkt plus $n$ mal das Abstandsquadrat von $c$ zum Schwerpunkt.
Was ein einzelner Wert $t$ am Ergebnis ändert, beschreibt die **Einflusskurve**: beim Mittel wächst sie linear mit $t$, beim Median bleibt sie beschränkt.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🧮 Rechnen")
    fmt = st.selectbox("Zahlenformat", C.FORMATS, key="fmt_select", help="float32 hat 24 Bit Mantisse (etwa 7 Dezimalstellen), float64 53 Bit. Gespeichert und summiert wird in diesem Format.")
    n = st.select_slider("Zahl der Werte n", C.N_OPTIONS, key="n_select", format_func=_thousands, help="Länge der Reihe (Zeitstempel in Sekunden oder Koordinaten in Metern). Die Rechenzeit wächst linear mit n.")
    ratio = st.select_slider("Verhältnis μ/σ", C.RATIO_OPTIONS, key="ratio_select", format_func=_thousands, help="Mittel geteilt durch Streuung der Werte; die Streuung ist 1. Große Werte machen die Summe groß gegen die Schwankung.")
    order = st.selectbox("Reihenfolge der Werte", C.ORDERS, key="order_select", help="Dieselben Werte zufällig, aufsteigend oder absteigend sortiert.")
    rdist = st.selectbox("Verteilung der Reihe", C.SERIES_DISTRIBUTIONS, key="rdist_select", help="Normal, oder schief (lognormal(0; 1), auf Mittel μ und Streuung 1 gebracht).")
    st.header("📊 Stichprobe")
    dist = st.selectbox("Verteilung", C.DISTRIBUTIONS, key="dist_select", help="Lognormal wird als Lieferzeit in Tagen gelesen (Median 4 Tage). Die übrigen Verteilungen stehen in Standardform.")
    sigma = st.slider("Schiefe σ (Lognormal)", *bounds("sigma_slider"), key="sigma_slider", step=0.1, disabled=dist != "Lognormal", help="Nur bei Lognormal: je größer σ, desto weiter liegt der Mittelwert über dem Median (Faktor e^(σ²/2)).")
    lam = st.slider("Rate λ (Poisson)", *bounds("lam_slider"), key="lam_slider", step=0.1, disabled=dist != "Poisson", help="Nur bei Poisson: Erwartungswert der ganzzahligen Werte.")
    wn = st.select_slider("Stichprobengröße", C.WN_OPTIONS, key="wn_select", format_func=_thousands, help="Zahl der Werte der Stichprobe.")
    out = st.slider("Ausreißeranteil (%)", *bounds("out_slider"), key="out_slider", help="So viele Prozent der Werte werden durch den Ausreißerabstand ersetzt. Ab 50 % kippt der Median (E3).")
    outdist = st.select_slider("Ausreißerabstand", C.OUTDIST_OPTIONS, key="outdist_select", disabled=out == 0, help="Der Wert, durch den die Ausreißer ersetzt werden; nur wirksam bei einem Ausreißeranteil über 0.")
    h = st.select_slider("Raster der Rundung h", C.H_OPTIONS, key="h_select", format_func=lambda v: "keine" if v == 0 else str(v).replace(".", ","), help="Alle Werte werden auf ein Vielfaches von h gerundet (E7).")
    shift = st.slider("Lage verschieben", *bounds("shift_slider"), key="shift_slider", step=0.1, help="Addiert diesen Wert zu allen Werten (nach dem Ziehen, vor Ausreißern und Rundung).")
    tau = st.slider("Stufe τ für Quantil und Expektile", *bounds("tau_slider"), key="tau_slider", step=0.01, help="Bei τ = 0,5 sind Quantil gleich Median und Expektile gleich Mittel.")
    st.header("🏭 Kunden und Lager")
    miscount = st.slider("Abstand der Fehlbuchung (km)", *bounds("miscount_slider"), key="miscount_slider", step=1.0, help=f"Ein {C.PLANE_N + 1}. Kunde wird im Abstand von dort auf der Winkelhalbierenden gebucht (Schwerpunkt der Kunden).")
    stein_d = st.select_slider("Zahl der Lager d (James-Stein)", C.STEIN_D_OPTIONS, key="stein_d_select", help="Zahl der Lager mit je einem Nachfragemittel; Streuung bekannt (1).")
    stein_norm = st.slider("Abstand ‖μ‖ der wahren Mittel", *bounds("stein_norm_slider"), key="stein_norm_slider", step=0.5, help="Abstand der wahren Mittel vom Nullpunkt (alles auf das erste Lager gelegt).")
    st.header("🎲 Zufall")
    st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

settings = V.Settings(fmt=fmt, n=int(n), ratio=int(ratio), order=order, rdist=rdist, dist=dist, sigma=float(sigma), lam=float(lam), wn=int(wn), out=int(out), outdist=int(outdist), h=float(h),
                      shift=float(shift), tau=float(tau), miscount=float(miscount), stein_d=int(stein_d), stein_norm=float(stein_norm), seed=int(st.session_state["seed_input"]))
sync_query_params({k: st.session_state[k] for k in ("fmt_select", "n_select", "ratio_select", "order_select", "rdist_select", "dist_select", "sigma_slider", "lam_slider", "wn_select", "out_slider",
                                                    "outdist_select", "h_select", "shift_slider", "tau_slider", "miscount_slider", "stein_d_select", "stein_norm_slider", "seed_input")})

tab_rechnen, tab_waehlen = st.tabs(["🧮 Rechnen", "🎯 Wählen"])

# =============================================================================================================================================
# Reiter Rechnen
# =============================================================================================================================================
with tab_rechnen:
    with st.spinner(f"Rechne {_thousands(settings.n)} Werte mit allen Verfahren …"):
        rr = _rechnen(settings)
    st.markdown("## 🎯 Ergebnis")
    st.caption("Gemessen wird gegen das **exakte Mittel der gespeicherten Werte** (Bruchrechnung), nicht gegen den Parameter μ des Generators: Es geht um den Fehler der Rechnung, nicht der Stichprobe.")
    m1, m2, m2b, m3, m4 = st.columns(5)
    m1.metric("Fehler der fortlaufenden Summe", V.sci(rr.abs_errors["naiv"]), delta=f"{V.num(rr.abs_errors['naiv'] / rr.standard_error, 2)} Standardfehler", delta_color="inverse" if rr.abs_errors["naiv"] > 4 * rr.abs_errors["Untergrenze"] else "off")
    m2.metric("Fehler von Kahan", V.sci(rr.abs_errors["Kahan"]), delta=f"{V.de(rr.abs_errors['Kahan'] / rr.standard_error, 3)} Standardfehler", delta_color="off")
    m2b.metric("Fehler von Neumaier", V.sci(rr.abs_errors["Neumaier"]), delta=f"{V.de(rr.abs_errors['Neumaier'] / rr.standard_error, 3)} Standardfehler", delta_color="inverse" if rr.abs_errors["Neumaier"] > 2 * rr.abs_errors["Kahan"] else "off")
    m3.metric("Untergrenze der Summe im Format", V.sci(rr.abs_errors["Untergrenze"]), delta="Referenz", delta_color="off")
    m4.metric("Verschiedene Werte (erste 100.000)", _thousands(rr.distinct), delta=f"Rechenzeit {V.de(rr.seconds, 1)} s", delta_color="off")
    st.plotly_chart(build_method_bars(rr.abs_errors, settings.n), width="stretch", key="bars_chart")
    st.caption("Balken: Betrag des Fehlers des Mittels in Standardfehlern σ/√n (log). Rot: mehr als das Vierfache der Untergrenze der Summe (gestrichelt) – der Fehler liegt dann nicht am Zahlenformat allein, sondern am Verfahren. „Mit Rest“ addiert den Rest in float64; Welford und „verschoben“ liefern das Mittel selbst im Format (Untergrenze in der Tabelle).")
    names = [k for k in rr.abs_errors if k != "Untergrenze"] + ["Untergrenze"]
    table_rows = [{"Verfahren": k if k != "Untergrenze" else "Untergrenze (Summe im Format)", "Fehler (mit Vorzeichen)": _signed(rr.signed_errors[k]), "Betrag in Standardfehlern": V.de(rr.abs_errors[k] / rr.standard_error, 4),
                   "Vielfaches der Untergrenze": "–" if rr.abs_errors["Untergrenze"] == 0 or k == "Untergrenze" else V.de(rr.abs_errors[k] / rr.abs_errors["Untergrenze"], 1)} for k in names]
    table_rows.append({"Verfahren": "Untergrenze (Mittel im Format)", "Fehler (mit Vorzeichen)": V.sci(rr.floor_mean), "Betrag in Standardfehlern": V.de(rr.floor_mean / rr.standard_error, 4), "Vielfaches der Untergrenze": "–"})
    st.dataframe(table_rows, hide_index=True, width="stretch")

    st.markdown("## ❓ Kernfrage: Wie genau liefert der Rechner das Mittel?")
    _verdict(V.verdict_rechnen(rr.abs_errors, settings.n))

    with st.expander("🧰 Die Summationsverfahren erklärt"):
        st.markdown(
            r"""
| Verfahren | Was es tut | Was schiefgehen kann |
|---|---|---|
| **naiv (fortlaufend)** | $s \leftarrow s + x_i$, jede Zwischensumme auf das Zahlenformat gerundet | Ist $s$ groß gegen $x_i$, geht von $x_i$ die untere Hälfte der Stellen verloren; die Verluste addieren sich |
| **numpy.sum** | verhält sich in der Messung wie eine paarweise Summe (der Fehler wächst kaum mit n) | Der genaue Aufbau hängt von der numpy-Version ab und ist hier nicht der Dokumentation entnommen; die Zahlen gelten als Band |
| **Binärbaum** | erst Nachbarn, dann Nachbarn der Teilsummen | Teilsummen bleiben ähnlich groß wie ihre Summanden: kleiner Fehler, aber mehr Speicher |
| **verschoben** | $x_1$ abziehen, die Reste fortlaufend summieren, $x_1$ addieren | Hilft nur, wenn $x_1$ typisch ist; ein untypischer erster Wert macht es schlechter (E2) |
| **Kahan** | ein Rest $c$ merkt sich, was beim Addieren verloren ging, und geht in den nächsten Summanden ein | Ergebnis ist $s$; „mit Rest“ gilt $s - c$ |
| **Neumaier** | wie Kahan, der Rest geht aber erst am Ende ein: Ergebnis $s + c$ | Der Rest $c$ wird selbst im Zahlenformat summiert (E1) |
| **Welford** | $m \leftarrow m + (x_k - m)/k$, nur der Mittelwertteil | Für Datenströme gebaut, nicht für die Genauigkeit des Mittels |
            """
        )

    st.markdown("---")
    st.caption(f"Die Messreihen der Experimente sind vorgerechnet (vier feste Zufalls-Seeds je Zelle, Stand {R.meta()['stand']}); Beträge sind Mittel über die Seeds, die Streuung zwischen den Seeds steht in Klammern.")

    # E1 -----------------------------------------------------------------------------------------------------------------------------------
    st.subheader("🔬 E1: Wie wächst der Fehler mit n?")
    st.markdown("Werte $N(1000;\\,1)$ in zufälliger Reihenfolge, die Summen entstehen in einem Durchgang (Kontrollpunkte bei $10^3$ bis $10^7$). Jedes Verfahren wird gegen das exakte Mittel der gespeicherten Werte gemessen.")
    e1_methods = ("naiv", "numpy.sum", "Binärbaum", "Kahan", "Neumaier", "Welford", "verschoben")
    c1, c2 = st.columns(2)
    for col, f_ in ((c1, "float32"), (c2, "float64")):
        col.plotly_chart(build_error_curve(R.e1(f_)["methods"], R.e1(f_)["checks"], e1_methods, f_), width="stretch", key=f"e1_{f_}")
    fmt_e1 = st.radio("Tabelle für das Zahlenformat", C.FORMATS, horizontal=True, key="e1_table_fmt")
    cks = R.e1(fmt_e1)["checks"]
    st.dataframe([{"Verfahren": m_, **{f"n = {_thousands(c)}": f"{V.sci(R.e1_cell(fmt_e1, m_, c)['mean'])} ({V.sci(R.e1_cell(fmt_e1, m_, c)['std'], 1)})" for c in cks}}
                  for m_ in ("floor_sum", "floor_mean") + X.METHODS], hide_index=True, width="stretch")
    st.caption("Betrag des Fehlers des Mittels, Mittel über vier Seeds (Streuung zwischen den Seeds). „floor_sum“ = Untergrenze der Summe im Format, „floor_mean“ = Untergrenze des Mittels im Format (für Welford und „verschoben“). Zwischen numpy.sum, Binärbaum und der Untergrenze liegt der Unterschied im Rauschen der Seeds.")
    nmax = cks[-1]
    _verdict(V.verdict_e1_naive(R.e1_cell("float32", "naiv", nmax), R.e1_cell("float64", "naiv", nmax), R.e1_cell("float32", "floor_sum", nmax), nmax))
    _verdict(V.verdict_e1_mechanism(R.e1_mechanism()["cells"]))
    _verdict(V.verdict_e1_kahan(R.e1_cell("float32", "Kahan", nmax), R.e1_cell("float32", "floor_sum", nmax), R.e1_cell("float32", "Binärbaum", nmax), R.e1_cell("float32", "numpy.sum", nmax), nmax))
    diag_row = R.e1_diagnosis()["rows"][-1]
    diag = {"n": diag_row["n"], "kahan_max_rest": diag_row["kahan"]["max_rest"], "neumaier_max_rest": diag_row["neumaier"]["max_rest"], "neumaier_sum_error": diag_row["neumaier"]["sum_error"],
            "neumaier_wide_sum_error": diag_row["neumaier_wide"]["sum_error"]}
    _verdict(V.verdict_e1_neumaier(R.e1_cell("float32", "Neumaier", nmax), R.e1_cell("float32", "Kahan", nmax), nmax, diag))
    _verdict(V.verdict_e1_neumaier_share(R.e1_neumaier_share(2_000_000)))
    _verdict(V.verdict_e1_welford(R.e1_cell("float32", "Welford", nmax), R.e1_cell("float32", "Binärbaum", nmax), R.e1_cell("float64", "Welford", nmax), R.e1_cell("float64", "naiv", nmax), nmax))

    # E2 -----------------------------------------------------------------------------------------------------------------------------------
    st.subheader("🔬 E2: Auslöschung und Reihenfolge")
    st.markdown(f"**Auslöschung.** $n = {_thousands(X.E2_N)}$ Werte mit Mittel $\\mu$ und Streuung 1; je größer $\\mu/\\sigma$, desto mehr Stellen der Summe gehen an den Mittelwert, und desto weniger bleibt für die Schwankung. Fehler in Standardfehlern $\\sigma/\\sqrt{{n}}$ ({V.sci(1 / math.sqrt(X.E2_N), 1)}).")
    c1, c2 = st.columns(2)
    for col, f_ in ((c1, "float32"), (c2, "float64")):
        col.plotly_chart(build_ratio_chart(R.e2_ratio(f_)["ratios"], ("naiv", "numpy.sum", "Kahan", "Neumaier", "Welford", "verschoben"), f_), width="stretch", key=f"e2r_{f_}")
    fmt_e2 = st.radio("Tabelle für das Zahlenformat", C.FORMATS, horizontal=True, key="e2_table_fmt")
    ratios = R.e2_ratio(fmt_e2)["ratios"]
    st.dataframe([{"Verfahren": m_, **{f"μ/σ = {k}": V.de(ratios[k]["methods"][m_]["mean"], 3) for k in ratios}} for m_ in ("floor_sum",) + X.METHODS], hide_index=True, width="stretch")
    st.caption("Betrag des Fehlers in Standardfehlern, Mittel über vier Seeds. „floor_sum“ = Untergrenze der Summe im Zahlenformat. Bei μ/σ ≥ 10⁶ (float32) sind die Daten selbst grob gerundet: wenige verschiedene Werte unter den ersten 100.000.")
    _verdict(V.verdict_e2_ratio(R.e2_ratio_cell("float32", 100.0, "naiv"), R.e2_ratio_cell("float64", 100.0, "naiv"), R.e2_ratio_cell("float64", 1e10, "naiv"), R.e2_ratio_cell("float32", 1e4, "Kahan"),
                                R.e2_ratio_cell("float32", 1e6, "Kahan"), R.e2_ratio_cell("float32", 1e6, "floor_sum")))
    st.markdown("**Gegenmittel: breiterer Akkumulator.** Daten in float32, aber `numpy.mean` mit float64-Akkumulator (danach auf float32 gerundet).")
    st.dataframe([{"μ": _thousands(int(c["mu"])), "n": _thousands(c["n"]), "numpy.mean in float32": V.de(c["float32"]["mean"], 2), "mit float64-Akkumulator": V.de(c["float64_accumulator_rounded"]["mean"], 2),
                   "Untergrenze": V.de(c["floor"]["mean"], 2)} for c in R.load()["e2_accu"]["cells"]], hide_index=True, width="stretch")
    st.caption("Betrag des Fehlers in Standardfehlern 1/√n, Mittel über vier Seeds.")
    _verdict(V.verdict_e2_accumulator(R.e2_accumulator(1e6, 1_000_000)))
    st.markdown(f"**Reihenfolge.** Dieselben $n = {_thousands(X.E2_N)}$ Werte zufällig, aufsteigend und absteigend, Mittel $1000$, Streuung 1; Normal und schief (lognormal(0; 1), auf Mittel 1000 und Streuung 1 gebracht).")
    fmt_o = st.radio("Zahlenformat der Reihenfolge-Tabelle", C.FORMATS, horizontal=True, key="e2_order_fmt")
    cells_o = R.e2_order(fmt_o)["cells"]
    rows_o = []
    for m_ in ("floor_sum",) + X.METHODS:
        row = {"Verfahren": m_}
        for dist_ in cells_o:
            for order_ in cells_o[dist_]:
                row[f"{dist_}, {order_}"] = V.sci(cells_o[dist_][order_][m_]["mean"], 2)
        rows_o.append(row)
    st.dataframe(rows_o, hide_index=True, width="stretch")
    st.caption("Betrag des Fehlers in Standardfehlern, Mittel über vier Seeds.")
    _verdict(V.verdict_e2_order({o: R.e2_order_cell("float32", "Normal", o, "naiv")["mean"] for o in cells_o["Normal"]}, R.e2_order_cell("float32", "Normal", "aufsteigend", "Welford"),
                                R.e2_order_cell("float32", "Lognormal", "absteigend", "verschoben"), R.e2_order_cell("float32", "Normal", "zufällig", "Kahan")))

# =============================================================================================================================================
# Reiter Wählen
# =============================================================================================================================================
with tab_waehlen:
    sa = _sample(settings)
    st.markdown("## 🎯 Ergebnis")
    st.caption("Eine Stichprobe, vier Mitten. Lognormal ist hier eine Lieferzeit in Tagen mit Median 4 (frei gewähltes Szenario, keine Messung); die übrigen Verteilungen stehen in Standardform.")
    q_lo, q_hi = sa.quantile
    q_text = V.de(q_lo, 3) if q_lo == q_hi else f"{V.de(q_lo, 3)} bis {V.de(q_hi, 3)}"
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Mittel", V.de(sa.mean, 3), delta="Quadratsumme", delta_color="off")
    w2.metric("Median", V.de(sa.median, 3), delta=f"{V.de(sa.median - sa.mean, 3)} gegen das Mittel", delta_color="off")
    w3.metric(f"Quantil τ = {V.de(settings.tau, 2)}", q_text, delta="Pinball-Verlust" + (" (Intervall)" if q_lo != q_hi else ""), delta_color="off")
    w4.metric(f"Expektile τ = {V.de(settings.tau, 2)}", V.de(sa.expectile, 3), delta="asymmetrische Quadrate", delta_color="off")
    st.plotly_chart(build_sample_chart(sa.x, sa.mean, sa.median, q_lo, sa.expectile, settings.tau), width="stretch", key="sample_chart")
    if sa.truth is not None:
        st.caption(f"Exakt für Lognormal mit σ = {V.de(settings.sigma, 1)} (Median 4 Tage): Mittelwert {V.de(sa.truth['mean'], 3)}, Median {V.de(sa.truth['median'], 3)}, Quantil {V.de(sa.truth['quantile'], 3)}, Expektile {V.de(sa.truth['expectile'], 3)}. "
                   "Die Stichprobe weicht davon durch den Zufall ab.")
    if sa.outliers:
        st.caption(f"{sa.outliers} Werte wurden durch {settings.outdist} ersetzt.")
    with st.expander("Die vier Verlustkurven"):
        st.plotly_chart(build_loss_chart(sa.x, settings.tau, sa.mean, sa.median, q_lo, sa.expectile), width="stretch", key="loss_chart")
        st.caption("Jede Kurve zeigt die Verlustsumme über dem Kandidaten c; die Raute markiert das Minimum. Mittel und Expektile liegen in glatten Mulden, Median und Quantil in Knicken (bei vielen Werten sieht man sie kaum).")
    _verdict(V.verdict_sample(sa.mean, sa.median, sa.iqr, settings.wn, V.plan_missing(sa.x)))

    pl = _plane(settings)
    st.markdown("### Der Schwerpunkt der Kunden")
    st.plotly_chart(build_plane_chart(pl), width="stretch", key="plane_chart")
    st.caption(f"{C.PLANE_N} normalverteilte Kunden (Einheit 1 km) und ein falsch gebuchter Kunde. Schwerpunkt ohne Fehlbuchung ({V.de(pl.centroid[0], 3)}; {V.de(pl.centroid[1], 3)}), mit ({V.de(pl.centroid_error[0], 3)}; {V.de(pl.centroid_error[1], 3)}). "
               f"Nach Huygens setzt sich die Quadratsumme um den alten Schwerpunkt aus {V.de(pl.within, 2)} (Streuung um den neuen Schwerpunkt) und {V.de(pl.between, 2)} (Abstand der beiden) zusammen.")
    _verdict(V.verdict_plane(pl.centroid_shift, pl.median_shift, settings.miscount, C.PLANE_N))

    if st.button("📄 PDF erstellen", key="pdf_make"):
        verdicts = [V.verdict_rechnen(rr.abs_errors, settings.n), V.verdict_sample(sa.mean, sa.median, sa.iqr, settings.wn, V.plan_missing(sa.x))]
        st.session_state["pdf"] = (repr(settings), build_pdf(rr, sa, verdicts))
    if st.session_state.get("pdf", (None, None))[0] == repr(settings):
        st.download_button("⬇️ PDF herunterladen", data=st.session_state["pdf"][1], file_name="mittelwert-median.pdf", mime="application/pdf", key="pdf_download")

    st.markdown("## ❓ Kernfrage: Welche Zahl will ich?")
    st.markdown("Summe planen heißt **Mittel mal Anzahl**, den typischen Fall nennen heißt **Median**, einen Bestand für 80 % der Nachfrage festlegen heißt **Quantil**. Die Experimente unten messen, was passiert, wenn man die falsche nimmt.")

    with st.expander("🧰 Mittel, Median, Quantil und Expektile erklärt"):
        st.markdown(
            r"""
| Größe | Beantwortet die Frage | Rechenweg | Was schiefgehen kann |
|---|---|---|---|
| **Mittel** | Wie viel ist es insgesamt, geteilt durch die Anzahl? | Summe durch $n$ | Ein Ausreißer zieht beliebig weit; bei schiefen Verteilungen liegt es über dem typischen Fall |
| **Median** | Was ist ein typischer Fall? | mittlerer Wert, bei geradem $n$ die Mitte der beiden mittleren | Bei Rundung und Gleichständen springt er (E7); als Schätzer der Summe ist er verzerrt (E6) |
| **Quantil** $\tau$ | Welchen Wert unterschreiten $\tau$ der Fälle? | sortieren, $\lceil n\tau \rceil$-ten Wert wählen; bei ganzzahligem $n\tau$ ist jeder Wert im Intervall zulässig | bei schweren Rändern sehr unsicher |
| **Expektile** $\tau$ | Wo halten sich der Zug nach oben und nach unten die Waage? | gewichtetes Mittel mit Gewicht $\tau$ oben, $1-\tau$ unten, bis nichts mehr wechselt | liegt bei gleicher Stufe nur zufällig nah am Quantil (E8) |
            """
        )

    st.markdown("---")
    st.caption(f"Die Messreihen der Experimente sind vorgerechnet (vier feste Zufalls-Seeds je Zelle, Stand {R.meta()['stand']}); die Knöpfe „Nachmessen“ rechnen mit den Reglern der Seitenleiste neu.")

    # E5 -----------------------------------------------------------------------------------------------------------------------------------
    st.subheader("🔬 E5: James-Stein – das Ganze gewinnt, einzelne Lager verlieren")
    st.markdown("Ein Händler schätzt für $d$ Lager je ein Nachfragemittel aus einer Beobachtung mit bekannter Streuung 1. Das **Einzelmittel** nimmt jede Beobachtung selbst. **James-Stein** schrumpft alle Beobachtungen gemeinsam auf den Nullpunkt zu. Risiko = mittlere Summe der quadratischen Fehler über alle Lager; "
                f"{_thousands(X.STEIN_RUNS)} Läufe je Seed.")
    dims, norms = X.STEIN_DIMS, X.STEIN_NORMS
    cells5 = {(d, nm): {"pp": R.stein_cell("eine", d, nm)["pp"]["mean"], "js_exact": R.stein_cell("eine", d, nm)["js_exact"], "koordinate1_pp": R.stein_cell("eine", d, nm)["koordinate1_pp"]["mean"]} for d in dims for nm in norms}
    ca, cb = st.columns(2)
    ca.plotly_chart(build_stein_chart(cells5, dims, norms), width="stretch", key="e5_chart")
    cb.plotly_chart(build_coordinate_chart(cells5, dims, norms), width="stretch", key="e5_coord")
    st.dataframe([{"‖μ‖": nm, **{f"d = {d}": V.de(R.stein_cell("eine", d, nm)["js"]["mean"], 3) for d in dims}} for nm in norms], hide_index=True, width="stretch")
    st.caption("Risiko James-Stein / Risiko des Einzelmittels (ohne Positive Part), Mittel über vier Seeds. Bei ‖μ‖ = 0 gilt 2/d.")
    c10 = R.stein_cell("eine", 10, 0.0)
    _verdict(V.verdict_e5(c10["js"]["mean"], 10, 0.0, c10["js_exact"], c10["pp"]["mean"]))
    best = max(((R.stein_cell("eine", 50, nm)["koordinate1_pp"]["mean"], nm) for nm in norms))
    _verdict(V.verdict_e5_coordinate(best[0], 50, best[1]))
    if st.button(f"Nachmessen mit d = {settings.stein_d}, ‖μ‖ = {V.de(settings.stein_norm, 1)} ({_thousands(C.STEIN_LIVE_RUNS)} Läufe, Zufalls-Seed der Seitenleiste)", key="e5_live"):
        with st.spinner("Rechne James-Stein …"):
            t1 = time.perf_counter()
            res5 = _stein_live(settings.stein_d, settings.stein_norm, "eine", settings.seed, C.STEIN_LIVE_RUNS)
            st.session_state["e5_result"] = (settings.stein_d, settings.stein_norm, res5, time.perf_counter() - t1)
    r5 = st.session_state.get("e5_result")
    if r5 is not None:
        exact5 = stein.exact_js_risk(r5[0], r5[1] ** 2) / r5[0]
        st.info(f"Nachgemessen (d = {r5[0]}, ‖μ‖ = {V.de(r5[1], 1)}, {V.de(r5[3], 1)} s): James-Stein / Einzelmittel = **{V.de(r5[2]['js'], 3)}** (exakt {V.de(exact5, 3)}), "
                f"Positive Part {V.de(r5[2]['pp'], 3)}; Risiko des ersten Lagers allein (Einzelmittel 1): {V.de(r5[2]['koordinate1_pp'], 2)}.")

    # E6 -----------------------------------------------------------------------------------------------------------------------------------
    st.subheader("🔬 E6: Welche Größe will ich?")
    st.markdown("Lieferzeiten sind rechtsschief; hier lognormal $(0;\\,\\sigma)$ mit Median 1. Der **Mittelwert** liegt um den Faktor $e^{\\sigma^2/2}$ über dem **Median**. Wer die Summe plant und den Median nimmt, schätzt eine andere Größe, und dieser Fehler schwindet nicht mit wachsendem $n$.")
    st.dataframe([{"σ": s_["sigma"], "Mittelwert / Median": V.de(s_["gap"], 3), "Anteil unter dem Mittelwert (exakt)": V.pct(s_["below_mean_exact"]) + " %", "Anteil unter dem Mittelwert (simuliert)": V.pct(s_["below_mean"]["mean"]) + " %",
                   "Summenanteil der größten 10 %": V.pct(s_["top10_share"]["mean"]) + " %"} for s_ in R.load()["e6"]["shapes"].values()], hide_index=True, width="stretch")
    sh1, tg1 = R.shape_row(1.0), R.target_row(1.0)
    _verdict(V.verdict_e6_gap(1.0, sh1["gap"], sh1["below_mean_exact"], tg1["miss_limit"], sh1["top10_share"]["mean"]))
    st.plotly_chart(build_target_chart(R.load()["e6"]["targets"]["rows"]), width="stretch", key="e6_targets")
    st.caption(f"Typischer relativer Fehler (Median über {_thousands(R.load()['e6']['targets']['runs'])} Stichproben je Seed) von Mittel und Median gegen beide Ziele, n = {R.load()['e6']['targets']['n']}. Der Median trifft den Mittelwert nicht, auch bei beliebig vielen Daten.")
    _verdict(V.verdict_e6_targets(1.0, tg1["mittel_mittelwert"]["typisch"]["mean"], tg1["median_mittelwert"]["typisch"]["mean"], tg1["mittel_median"]["typisch"]["mean"], tg1["median_median"]["typisch"]["mean"], R.load()["e6"]["targets"]["n"]))
    dl = R.load()["e6"]["delivery"]
    st.markdown(f"**Lieferzeit-Szenario (frei gewählt, keine Messung).** Aufträge brauchen lognormal verteilt im Median {dl['median']:g} Tage, ein Monat hat {_thousands(dl['orders'])} Aufträge. Plan A: „{_thousands(dl['orders'])} × Median“, Plan B: „{_thousands(dl['orders'])} × Mittelwert“.")
    st.dataframe([{"σ": r_["sigma"], "Mittelwert (Tage)": V.de(r_["mean"], 2), "Plan A lässt aus": V.pct(r_["plan_median_missing"]["mean"]) + " %", "Plan B weicht ab (Betrag)": V.pct(r_["plan_mean_deviation"]["mean"]) + " %",
                   "Aufträge höchstens so lang wie der Mittelwert": V.pct(r_["orders_below_mean"]["mean"]) + " %", f"{dl['sample']} Aufträge: „× Median“ zu klein": V.pct(r_["schaetzung_median"]["zu_klein"]["mean"]) + " %"} for r_ in dl["rows"].values()],
                 hide_index=True, width="stretch")
    d08 = R.delivery_row(0.8)
    _verdict(V.verdict_e6_delivery(0.8, d08["plan_median_missing"]["mean"], d08["schaetzung_median"]["typisch"]["mean"], d08["schaetzung_mittel"]["typisch"]["mean"], d08["schaetzung_median"]["zu_klein"]["mean"], dl["orders"], dl["sample"]))
    live_runs6 = int(max(50, min(2000, 4_000_000 // settings.wn)))
    if st.button(f"Nachmessen mit σ = {V.de(settings.sigma, 1)} und n = {_thousands(settings.wn)} ({live_runs6} Stichproben)", key="e6_live"):
        with st.spinner("Rechne Fehler gegen beide Ziele …"):
            st.session_state["e6_result"] = (settings.sigma, settings.wn, live_runs6, _targets_live(settings.sigma, settings.wn, live_runs6, settings.seed))
    r6 = st.session_state.get("e6_result")
    if r6 is not None:
        t6 = r6[3]
        st.info(f"Nachgemessen (σ = {V.de(r6[0], 1)}, n = {_thousands(r6[1])}, {r6[2]} Stichproben): typischer Fehler Mittel → Mittelwert {V.pct(t6['mittel_mittelwert']['typisch'])} %, Median → Mittelwert {V.pct(t6['median_mittelwert']['typisch'])} %, "
                f"Mittel → Median {V.pct(t6['mittel_median']['typisch'])} %, Median → Median {V.pct(t6['median_median']['typisch'])} %. Grenzwert für Median → Mittelwert: {V.pct(theory.median_miss_share(r6[0]))} %.")

    # E8 -----------------------------------------------------------------------------------------------------------------------------------
    st.subheader("🔬 E8: Quantil gegen Expektile")
    st.markdown("Die Expektile zur Stufe $\\tau$ ist das asymmetrische Gegenstück des Mittels, das Quantil das des Medians. Man liest sie oft als „fast dasselbe“. **Die Lücke** (Expektile minus Quantil, in Quartilsabständen) hat je nach Verteilung und Stufe beide Vorzeichen.")
    ex = R.lognormal_exact()
    st.dataframe([{"τ": r_["tau"], "Quantil": V.de(r_["q"], 4), "Expektile": V.de(r_["e"], 4), "Expektile / Quantil": V.de(r_["ratio"], 3), "Anteil unter der Expektile": V.pct(r_["F_e"]) + " %"} for r_ in ex["rows"]], hide_index=True, width="stretch")
    st.caption(f"Exakt für Lognormal(0; {V.de(ex['sigma'], 1)}) (Teilerwartungen, Bisektion). Median {V.de(ex['median'], 1)}, Mittelwert {V.de(ex['mean'], 4)}; ein Bestand in Höhe des Mittelwerts deckt {V.pct(ex['mean_cdf'])} % der Nachfrage.")
    r80 = [r_ for r_ in ex["rows"] if r_["tau"] == 0.8][0]
    _verdict(V.verdict_e8_cover(ex["sigma"], 0.8, ex["mean_cdf"], r80["F_e"], r80["q"], r80["e"], ex["mean"]))
    labels = R.gap_labels()
    st.plotly_chart(build_gap_chart(R.load()["e8"]["gaps"], X.GAP_TAUS, labels), width="stretch", key="e8_chart")
    st.dataframe([{"Verteilung": lb, **{f"τ = {t:g}": (V.de(R.gap_cell(lb, t)["gap"]["mean"], 3) if R.gap_cell(lb, t)["usable"] else "unsicher") for t in X.GAP_TAUS}} for lb in labels], hide_index=True, width="stretch")
    st.caption(f"Lücke (Expektile − Quantil) / Quartilsabstand, Stichproben mit {_thousands(R.load()['e8']['count'])} Werten je Seed, Mittel über vier Seeds. „unsicher“: die Streuung zwischen den Seeds erreicht die halbe Lücke.")
    for lb in ("Normal", "Lognormal(0; 1,5)"):
        c_ = R.gap_cell(lb, 0.8)
        _verdict(V.verdict_e8_gap(lb, 0.8, c_["gap"]["mean"], c_["gap"]["std"], c_["usable"]))
    sig_l, tau_l = settings.sigma, settings.tau
    st.markdown(f"**Exakt für Ihre Regler** (Lognormal(0; {V.de(sig_l, 1)}), τ = {V.de(tau_l, 2)}):")
    ql, el, ml = theory.lognormal_quantile(sig_l, tau_l), theory.lognormal_expectile(sig_l, tau_l), theory.gap_mean_median(sig_l)
    e1c, e2c, e3c, e4c = st.columns(4)
    e1c.metric("Quantil", V.de(ql, 4))
    e2c.metric("Expektile", V.de(el, 4), delta=f"{V.de(el - ql, 4)} gegen das Quantil", delta_color="off")
    e3c.metric("Mittelwert", V.de(ml, 4), delta=f"deckt {V.pct(theory.share_below_mean(sig_l))} %", delta_color="off")
    e4c.metric("Expektile deckt", f"{V.pct(theory.lognormal_cdf(el, sig_l))} %", delta=f"gewünscht {V.pct(tau_l, 0)} %", delta_color="off")

    st.markdown("---")
    tab_e3, tab_e7 = st.tabs(["🔬 E3: Einfluss, Durchschlag, Bias-Varianz", "🔬 E7: Ties und Rundung"])

    # E3 -----------------------------------------------------------------------------------------------------------------------------------
    with tab_e3:
        st.markdown("**Einflusskurve.** $SC(t) = n\\,(T(x_1, \\dots, x_{n-1}, t) - T(x_1, \\dots, x_{n-1}))$ bei insgesamt $n$ Werten (hier so definiert), Basis $N(0;\\,1)$, " f"{_thousands(R.e3()['sensitivity']['runs'])} Stichproben je Seed.")
        n_tot = st.radio("Zahl der Werte insgesamt", (101, 100), horizontal=True, key="e3_ntot", help="Bei ungeradem n ist der Median ein Beobachtungswert, bei geradem das Mittel der mittleren zwei.")
        ts_ = [t for t in X.SC_TS]
        st.plotly_chart(build_influence_chart(ts_, [R.sensitivity(n_tot, t)["median"]["mean"] for t in ts_], [R.sensitivity(n_tot, t)["mean"]["mean"] for t in ts_]), width="stretch", key="e3_influence")
        _verdict(V.verdict_e3_influence(R.sensitivity(n_tot, 1e6)["median"]["mean"], R.sensitivity(n_tot, 1e6)["mean"]["mean"], theory.INFLUENCE_PLATEAU, n_tot))
        st.markdown("**Durchschlagspunkt** durch Aufzählung: alle Teilmengen von $k$ Werten, jeder ersetzte Wert durch $+M$ oder $-M$ (für $n \\le 9$ alle Vorzeichenmuster, darüber nur $+M$); „beliebig weit“ heißt: die größte Verschiebung wächst von $M = 10^3$ auf $10^6$ um mehr als 1.")
        rows3 = R.e3()["breakdown"]["rows"]
        st.dataframe([{"n": r_["n"], "k* (Median, vier Seeds)": ", ".join(str(k) for k in r_["k"]), "⌈n/2⌉": r_["ceil_half"], "k* (Mittel)": ", ".join(str(k) for k in r_["mean_k"]),
                       "größte Verschiebung bei k* − 1": V.de(r_["max_shift_below"]["mean"], 2)} for r_ in rows3], hide_index=True, width="stretch")
        _verdict(V.verdict_e3_breakdown([r_["k"][0] for r_ in rows3], [r_["n"] for r_ in rows3]))
        n_b = st.slider("n für den eigenen Durchschlag", 3, 12, 7, key="e3_n", help="Zählt alle Fälle durch (3^n bis 531.441 Ersetzungen) und braucht bei n = 12 einige Sekunden.")
        if st.button(f"Durchschlag zählen (n = {n_b}, Zufalls-Seed der Seitenleiste)", key="e3_live"):
            with st.spinner("Zähle alle Ersetzungen …"):
                t1 = time.perf_counter()
                st.session_state["e3_result"] = (n_b, settings.seed, _breakdown_live(n_b, settings.seed), time.perf_counter() - t1)
        r3 = st.session_state.get("e3_result")
        if r3 is not None:
            st.info(f"n = {r3[0]}: kleinste Zahl ersetzter Werte, die den Median beliebig weit schiebt: **k = {r3[2]['k']}** (⌈n/2⌉ = {math.ceil(r3[0] / 2)}); größte Verschiebung davor {V.de(r3[2]['max_shift_below'], 2)} ({V.de(r3[3], 1)} s).")
        st.markdown("**Verschiebung über dem Ausreißeranteil** ($n = 100$, $N(0;\\,1)$, Ausreißer bei 100).")
        crows = [r_ for r_ in R.e3()["contamination"]["rows"] if r_["M"] == 100.0 and r_["eps"] > 0]
        st.plotly_chart(build_contamination_chart(crows), width="stretch", key="e3_contamination")
        cm = lambda e: R.contamination_row(100.0, e)
        _verdict(V.verdict_e3_contamination(cm(0.1)["median"]["mean"], cm(0.4)["median"]["mean"], cm(0.49)["median"]["mean"], cm(0.5)["median"]["mean"], cm(0.4)["mean"]["mean"]))
        st.markdown("**Bias und Varianz.** Das um den Faktor $c$ verkleinerte Mittel $c\\,\\bar x$ ($n = 10$, $\\sigma = 1$) hat den MSE $c^2\\sigma^2/n + (1-c)^2\\mu^2$; gemessen relativ zum MSE des Mittels ($1/n$), mit $\\theta = \\mu\\sqrt{n}/\\sigma$.")
        st.dataframe([{"θ": r_["theta"], **{f"c = {k_}": f"{V.de(c_['sim']['mean'], 3)} (Formel {V.de(c_['formula'], 3)})" for k_, c_ in r_["cells"].items() if k_ != "c*"}, "bestes c*": V.de(r_["cstar"], 3)} for r_ in R.e3()["bias_variance"]["rows"]],
                     hide_index=True, width="stretch")
        b2 = R.bias_variance_row(2.0)
        _verdict(V.verdict_e3_bias(2.0, b2["cells"]["0,9"]["sim"]["mean"], b2["cells"]["0,9"]["formula"], b2["cstar"]))
        st.markdown("**Effizienz des Medians bei Normalverteilung** (exakte Quadratur): Varianz des Mittels geteilt durch Varianz des Medians.")
        eff = R.e3()["efficiency"]
        st.plotly_chart(build_efficiency_chart(eff["odd"], eff["even"], eff["limit"]), width="stretch", key="e3_efficiency")
        st.caption(f"Die Werte fallen von {V.de(eff['odd']['3'], 4)} (n = 3) auf {V.de(eff['odd']['1001'], 4)} (n = 1001); der Grenzwert ist 2/π = {V.de(eff['limit'], 4)}. Gerades n liegt höher als die Nachbarn (n = 4: {V.de(eff['even']['4'], 4)}). "
                   "Der Median kostet bei Normalverteilung also bis zu etwa ein Drittel der Genauigkeit; bei anderen Verteilungen (Laplace, t, Gleichverteilung) kann es umgekehrt sein: Stück 4 der Reihe.")

    # E7 -----------------------------------------------------------------------------------------------------------------------------------
    with tab_e7:
        st.markdown("**Gerades n bei ganzen Zahlen.** Der Median ist die Mitte der beiden mittleren Werte; bei ganzzahligen Daten ist sie oft nicht ganzzahlig.")
        st.dataframe([{"λ (Poisson)": lam_, **{f"n = {n_}: Median nicht ganzzahlig": V.pct(R.poisson_cell(lam_, n_)["non_integer"]["mean"]) + " %" for n_ in (10, 100)}} for lam_ in X.TIE_LAMBDAS], hide_index=True, width="stretch")
        st.markdown("**Intervall der Minimierer.** Bei geradem $n$ minimiert jeder Punkt zwischen den beiden mittleren Werten die Betragssumme; bei stetigen Daten ist das ein echtes Intervall, bei ganzen Zahlen oft nur ein Punkt.")
        st.dataframe([{"Verteilung": "Normal" if d_ == "Normal" else "Poisson(2)", **{f"n = {n_}": f"{V.pct(R.interval_cell(d_, n_)['share_interval']['mean'])} % echtes Intervall, mittlere Breite {V.de(R.interval_cell(d_, n_)['mean_width']['mean'], 3)}" for n_ in (4, 10, 100)}}
                      for d_ in ("Normal", "Poisson")], hide_index=True, width="stretch")
        st.markdown("**Fast-Gleichstand.** Poisson(0,7): $F(0)$ liegt knapp unter $\\tfrac12$, der Populationsmedian ist 1.")
        st.dataframe([{"n": _thousands(int(n_)), "Läufe je Seed": _thousands(c_["runs"]), "Median 0": V.pct(c_["is0"]["mean"]) + " %", "Median 0,5": V.pct(c_["is_half"]["mean"]) + " %", "Median 1": V.pct(c_["is1"]["mean"]) + " %"} for n_, c_ in R.load()["e7"]["near_tie"].items()],
                     hide_index=True, width="stretch")
        nt = R.near_tie(100)
        _verdict(V.verdict_e7_ties(nt["is0"]["mean"], nt["is_half"]["mean"], nt["is1"]["mean"], 100, R.load()["e7"]["f0"]))
        rnd = R.load()["e7"]["rounding"]
        st.markdown(f"**Gerundete Daten.** $N({V.de(rnd['mu'], 1)};\\,1)$ auf das Raster $h$ gerundet, $n = {rnd['n']}$, {_thousands(rnd['runs'])} Läufe je Seed.")
        st.plotly_chart(build_rounding_chart(R.load()["e7"]["rounding"]["rows"]), width="stretch", key="e7_rounding")
        rd = R.rounding_row(1.0)
        _verdict(V.verdict_e7_rounding(1.0, rd["rmse_mean"]["mean"], rd["rmse_median"]["mean"], rd["median_zero"]["mean"], R.load()["e7"]["rounding"]["mu"]))
        st.caption("Im Preset „Gerundete Daten“ lässt sich das an einer Stichprobe ansehen. Ab h = 1 Streuung kippt die Varianz des Medians gegen 0 und das Effizienzverhältnis wird sinnlos; es steht deshalb nicht hier.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Simulierte Reihen und Stichproben** | Alle Fehler, Anteile und Risiken stammen aus den Verteilungen und Reglern dieser Demo; echte Zeitstempel und Kundenkoordinaten können anders aussehen. | – |
| **Fehler gegen das exakte Mittel der gespeicherten Werte** | Gemessen wird der Rechenfehler, nicht der Stichprobenfehler gegen den Parameter μ; die Daten sind schon im Zahlenformat gerundet. | – |
| **IEEE 754 mit numpy-float32 und float64** | Die Schleifen sind bitgenau reproduzierbar; `numpy.sum` und die Binärstelle von `log`/`cos` aus numpy können je Version und Rechner abweichen, deshalb werden sie nur als Band gelesen. | – |
| **Neumaier-Befund an einer Datenform** | Dass Neumaier in float32 schlechter als Kahan ist, ist an Werten um 1 000 mit Streuung 1 gemessen; die Ursache (der Rest wird selbst in float32 summiert) ist an einer Reihe diagnostiziert, nicht allgemein bewiesen. | – |
| **Nur das Mittel, keine Varianz** | Die Rechenfehler der Varianz (Welford-Varianz, Σx² − n·x̄²) sind nicht untersucht; Welford wird nur als Mittelwertverfahren gemessen. | – |
| **Kleine Zahlenformate** | float16 und bfloat16 kommen nicht vor. | – |
| **James-Stein nur bei bekannter Streuung** | Ist die Streuung unbekannt, ändert sich der Schätzer; hier ist sie 1. Gemessen sind Normalbeobachtungen mit der Schrumpfung zum Nullpunkt und zum gemeinsamen Mittel. | Schrumpfung bei Bradley-Terry-Stärken: `rangintervalle-demo` |
| **Lieferzeit-Szenario frei gewählt** | Median 4 Tage, 1 000 Aufträge im Monat, lognormal: ein Beispiel, keine Messung an echten Aufträgen. | – |
| **Kosten gleich Verlustform** | Der Bestand wird nur als Quantil bzw. Expektile besprochen; Lager- und Fehlmengenkosten, Bestellrhythmen und Termine fehlen. | `newsvendor-evpi-demo`, `forecast-inventory-demo` |
| **Durchschlag nur aufgezählt** | Der Durchschlagspunkt ist für n bis 12 durch Aufzählung bestimmt (für n ≥ 10 nur mit +M), nicht für beliebiges n bewiesen. | – |
| **Ausreißer in einer Richtung** | Die Verschiebung unter Verschmutzung ist für Ausreißer bei +M gemessen. | Trimmen, Winsorisieren, Huber: Stück 4 der Reihe |
| **Abstandsmaß Quadrat und Betrag** | Andere Exponenten p der Abstandsumme bilden eine ganze Familie von Mitten. | Stück 3 der Reihe (Lp-Mitten) |
| **Mitte je Achse in der Ebene** | Der Median je Achse ist nicht der geometrische Median (Weber-Punkt). | Stück 5 der Reihe (`weiszfeld-demo`) |
| **Streuung zwischen den Seeds** | Jede Zelle nutzt vier feste Seeds; die angegebene Streuung ist die zwischen diesen Seeds, kein Konfidenzintervall. | – |
    """
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* Werte $x_1, \dots, x_n$ (Zeitstempel, Koordinaten, Lieferzeiten). *Gesucht:* eine Mitte $c$ bzw. die Genauigkeit, mit der der Rechner sie liefert.

| Symbol | Bedeutung |
|---|---|
| $x_i$ | die Werte, $i = 1, \dots, n$ |
| $\bar x$ | Mittel, $\bar x = \frac1n \sum_i x_i$ |
| $\tau$ | Stufe von Quantil und Expektile, $0 < \tau < 1$ |
| $\mu,\ \sigma$ | Mittel und Streuung der Verteilung, $\theta = \mu\sqrt n/\sigma$ |
| $\varepsilon$ | Rundungseinheit eines Zahlenformats ($2^{-24}$ bei float32, $2^{-53}$ bei float64) |
| $d,\ \lVert\mu\rVert$ | Zahl der Lager und Abstand der wahren Mittel vom Nullpunkt |

**Mitten als Minimierer.**

$$\bar x = \arg\min_c \sum_{i} (x_i - c)^2, \qquad \operatorname{med}(x) \in \arg\min_c \sum_{i} \lvert x_i - c\rvert.$$

**Quantil und Expektile.** Mit $\rho_\tau(u) = \tau u$ für $u \ge 0$ und $(\tau - 1)u$ sonst:

$$q_\tau \in \arg\min_c \sum_i \rho_\tau(x_i - c), \qquad e_\tau = \arg\min_c \sum_i \big(\tau\,(x_i - c)_+^2 + (1-\tau)\,(c - x_i)_+^2\big).$$

Die Expektile löst $\tau \sum_i (x_i - e)_+ = (1-\tau) \sum_i (e - x_i)_+$; bei $\tau = \tfrac12$ ist sie das Mittel.

**Huygens.** Für Punkte $x_i$ der Ebene mit Schwerpunkt $\bar x$ und jeden Punkt $c$:

$$\sum_i \lVert x_i - c\rVert^2 = \sum_i \lVert x_i - \bar x\rVert^2 + n\,\lVert c - \bar x\rVert^2.$$

**Einflusskurve** bei $n$ Werten:

$$SC_n(t) = n\,\big(T(x_1, \dots, x_{n-1}, t) - T(x_1, \dots, x_{n-1})\big).$$

Beim Mittel ist $SC_n(t) = t - \bar x_{n-1}$; beim Median bei Normalverteilung strebt sie gegen $\sqrt{\pi/2}$ für $n \to \infty$ und $\lvert t\rvert$ groß.

**Verkleinertes Mittel.** Für $c\,\bar x$ gilt

$$\operatorname{MSE}(c\,\bar x) = \frac{c^2\sigma^2}{n} + (1-c)^2\mu^2,$$

es schlägt das Mittel für $\theta^2 < \frac{1+c}{1-c}$; das beste $c = \mu^2/(\mu^2 + \sigma^2/n)$ enthält das unbekannte $\mu$.

**Effizienz des Medians** bei Normalverteilung: $\operatorname{Var}(\bar x)/\operatorname{Var}(\operatorname{med}) \to 2/\pi$ für $n \to \infty$; für endliches $n$ steht die exakte Varianz der mittleren Ordnungsstatistik im Code (Quadratur).

**Lognormal.** Für $X = e^{\sigma Z}$ ist der Median 1 und der Mittelwert $e^{\sigma^2/2}$; der Median als Schätzer des Mittelwerts lässt den Anteil

$$1 - e^{-\sigma^2/2}$$

aus, unabhängig von $n$.

**Summation.** Die fortlaufende Summe ist $s_k = \operatorname{fl}(s_{k-1} + x_k)$, wobei $\operatorname{fl}$ auf das Zahlenformat rundet (relativer Fehler höchstens $\varepsilon$ je Addition). Kahan-Summation führt einen Rest mit, Neumaier addiert ihn erst am Ende. Wie der Fehler mit $n$ wächst, ist hier nicht bewiesen, die Demo misst ihn.

**James-Stein.** Für $X \sim N(\mu, I_d)$ mit $d \ge 3$ hat

$$\hat\mu_{JS} = \Big(1 - \frac{d-2}{\lVert X\rVert^2}\Big) X$$

das Risiko $d - (d-2)^2\,\mathbb{E}\big[1/\chi^2_d(\lVert\mu\rVert^2)\big]$, bei $\mu = 0$ gleich 2, also $2/d$ des Risikos $d$ des Einzelmittels. Als Poisson-Mischung zentraler $\chi^2$-Verteilungen ist $\mathbb{E}[1/\chi^2_{d+2K}] = 1/(d+2K-2)$ mit $K \sim \mathrm{Poisson}(\lVert\mu\rVert^2/2)$.

**Literatur.** Zu Kahan, Neumaier, Welford, der Fehleranalyse der paarweisen Summe, James und Stein, Efron und Morris sowie Newey und Powell (Expektilen) wurden für diese Demo **keine Originalarbeiten eingesehen**; die Verfahren sind nach ihrer Beschreibung umgesetzt und gegen unabhängige Rechenwege geprüft (`fractions`, `math.fsum`, `scipy`, Aufzählung).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
