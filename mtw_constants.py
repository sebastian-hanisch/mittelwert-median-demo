"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Einheiten: Reihe im Reiter „Rechnen“ in Sekunden (Zeitstempel) bzw. Metern (Koordinaten) mit Streuung 1; Stichprobe im Reiter „Wählen“ in Einheiten der
Verteilung, bei Lognormal Lieferzeit in Tagen (Median 4 Tage); Kunden in der Ebene in Kilometern."""

from mtw_scenario import DISTRIBUTIONS, ORDERS, SERIES_DISTRIBUTIONS

RESULTS_FILE = "mtw_results.json"

# --- Regler: Rechnen ------------------------------------------------------------------------------------------------------------------------
FORMATS = ("float32", "float64")
N_OPTIONS = (1_000, 10_000, 100_000, 1_000_000, 2_000_000, 5_000_000)   # Laufzeit linear in n (alle neun Varianten zusammen), Speicher O(n): README „Rechengrenze“
RATIO_OPTIONS = (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000)   # μ/σ
DEFAULT_N, DEFAULT_RATIO = 1_000_000, 1_000
LIVE_N_MAX = N_OPTIONS[-1]

# --- Regler: Wählen -------------------------------------------------------------------------------------------------------------------------
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA = 0.1, 2.0, 0.8        # Schiefe der Lognormal-Verteilung
LAM_MIN, LAM_MAX, DEFAULT_LAM = 0.3, 10.0, 0.7             # Poisson
WN_OPTIONS = (5, 10, 20, 50, 100, 200, 500, 1_000, 5_000, 10_000)
DEFAULT_WN = 200
OUT_MIN, OUT_MAX, DEFAULT_OUT = 0, 49, 0                    # Ausreißeranteil in Prozent (ab 50 % kippt der Median: eigener Befund in E3)
OUTDIST_OPTIONS = (5, 10, 20, 50, 100, 1_000)
DEFAULT_OUTDIST = 50
H_OPTIONS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)                 # Raster der Rundung, 0 = keine
SHIFT_MIN, SHIFT_MAX, DEFAULT_SHIFT = -2.0, 2.0, 0.0
TAU_MIN, TAU_MAX, DEFAULT_TAU = 0.5, 0.99, 0.8
LOGNORMAL_MEDIAN = 4.0                                       # Lieferzeit-Szenario: Median 4 Tage (frei gewählt, keine Messung)
HIST_BINS = 40
STRIP_MAX_POINTS = 400

# --- Regler: Schätzen -----------------------------------------------------------------------------------------------------------------------
MISCOUNT_MIN, MISCOUNT_MAX, DEFAULT_MISCOUNT = 0.0, 50.0, 10.0   # Abstand des Fehlbuchungs-Kunden in km
PLANE_N = 20                                                      # Kunden in der Ebene
STEIN_D_OPTIONS = (3, 5, 10, 50)
DEFAULT_STEIN_D = 5
STEIN_NORM_MIN, STEIN_NORM_MAX, DEFAULT_STEIN_NORM = 0.0, 20.0, 2.0
STEIN_LIVE_RUNS = 50_000
SEED_MAX, DEFAULT_SEED = 999999, 7

# --- Farben -------------------------------------------------------------------------------------------------------------------------------
COLOR_MEAN = "#7f7f7f"
COLOR_MEDIAN = "#2F6B65"
COLOR_QUANTILE = "#4c78a8"
COLOR_EXPECTILE = "#d62728"
COLOR_FLOOR = "#b8860b"
COLOR_POINT = "rgba(80,80,80,0.45)"
COLOR_BAD = "#d62728"
COLOR_GOOD = "#2F6B65"

# --- Presets (Werte für jeden Regler) -----------------------------------------------------------------------------------------------------
_BASE = {
    "fmt": "float32", "n": DEFAULT_N, "ratio": DEFAULT_RATIO, "order": "zufällig", "rdist": "Normal",
    "dist": "Lognormal", "sigma": DEFAULT_SIGMA, "lam": DEFAULT_LAM, "wn": DEFAULT_WN, "out": DEFAULT_OUT, "outdist": DEFAULT_OUTDIST, "h": 0.0, "shift": DEFAULT_SHIFT, "tau": DEFAULT_TAU,
    "miscount": DEFAULT_MISCOUNT, "stein_d": DEFAULT_STEIN_D, "stein_norm": DEFAULT_STEIN_NORM, "seed": DEFAULT_SEED,
}


def _preset(**changes):
    unknown = set(changes) - set(_BASE)
    assert not unknown, unknown
    return {**_BASE, **changes}


PRESETS = {
    "Standard (float32, μ/σ = 1 000, n = 10⁶)": _preset(),
    "Fünf Millionen Zeitstempel": _preset(n=5_000_000),
    "Neumaier verliert": _preset(n=2_000_000, seed=13),
    "Sortiert (Welford)": _preset(order="aufsteigend"),
    "Quantisierung": _preset(ratio=1_000_000),
    "Plan mal Median": _preset(dist="Lognormal", sigma=0.8, wn=1_000),
    "Fast-Gleichstand": _preset(dist="Poisson", lam=0.7, wn=100, seed=8),
    "Zehn Prozent Ausreißer": _preset(dist="Normal", wn=100, out=10, outdist=100),
    "Gerundete Daten": _preset(dist="Normal", shift=0.3, h=1.0, wn=100, seed=9),
    "Mittel gegen Quantil im Bestand": _preset(dist="Lognormal", sigma=0.5, tau=0.8, wn=1_000),
    "Stein, d = 10": _preset(stein_d=10, stein_norm=0.0),
    "Stein, eine Koordinate": _preset(stein_d=50, stein_norm=8.0),
}
PRESET_HELP = {
    "Standard (float32, μ/σ = 1 000, n = 10⁶)": "Eine Million Zeitstempel in Sekunden, Mittel 1 000 und Streuung 1, gespeichert und summiert in float32. Die fortlaufende Summe liegt schon bei dieser Größe weit daneben (Kennzahlen und Tabelle zeigen es), Kahan und die paarweise Summe nicht. Wechseln Sie das Zahlenformat auf float64, und der Unterschied verschwindet.",
    "Fünf Millionen Zeitstempel": "Fünf Millionen Werte in float32: Der Fehler der fortlaufenden Summe wächst mit n weiter, Kahan bleibt an der Untergrenze des Formats. Die Rechnung braucht einige Sekunden.",
    "Neumaier verliert": "Zwei Millionen Werte in float32, eine Reihe, in der Neumaier um ein Mehrfaches ungenauer ist als Kahan, obwohl es als dessen Verbesserung gilt. Nicht jede Reihe zeigt das (E1 zählt es an zwanzig weiteren Reihen): Wechseln Sie den Zufalls-Seed. Der Befund gilt für float32 und Werte um 1 000 mit Streuung 1.",
    "Sortiert (Welford)": "Dieselben Werte aufsteigend sortiert. Das laufende Mittel nach Welford wird auf dieser Anordnung ruinös, Kahan nicht; die fortlaufende Summe ändert sich nur um einen Bruchteil.",
    "Quantisierung": "Mittel eine Million, Streuung 1: In float32 liegen die Werte auf einem Raster von 0,0625, viele sind gleich. Schon die Summe lässt sich nicht genauer darstellen als die Untergrenze, kein Verfahren kommt darunter. Der Ausweg ist float64 oder ein breiterer Akkumulator (E2).",
    "Plan mal Median": "Lieferzeiten in Tagen, lognormal mit Median 4 (frei gewähltes Szenario), 1 000 Aufträge. Der Plan „Anzahl × Median“ lässt einen großen Teil der wahren Summe aus; „Anzahl × Mittel“ trifft sie. Der Median beantwortet die Frage nach dem typischen Auftrag, nicht die nach der Summe.",
    "Fast-Gleichstand": "Poisson mit λ = 0,7 und 100 Werten: F(0) liegt knapp unter ½, der Median der Verteilung ist 1. Die Stichprobe entscheidet zufällig zwischen 0, 0,5 und 1; „Neu würfeln“ zeigt es (E7).",
    "Zehn Prozent Ausreißer": "Normalverteilte Werte, ein Zehntel davon durch 100 ersetzt: Das Mittel wandert mit den Ausreißern, der Median bleibt bei den übrigen Werten (E3).",
    "Gerundete Daten": "Normalverteilt mit Lage 0,3, auf ganze Zahlen gerundet, 100 Werte. Der Median bleibt auf der 0 hängen, das Mittel nicht (E7).",
    "Mittel gegen Quantil im Bestand": "Nachfrage lognormal mit Schiefe 0,5. Ein Bestand in Höhe des Mittels deckt nur einen Teil der Nachfrage; Quantil und Expektile der Stufe 0,8 liegen deutlich darüber. Bei dieser Schiefe liegen beide nah beieinander, bei anderen Verteilungen und Stufen nicht (E8).",
    "Stein, d = 10": "Zehn Lager, alle wahren Mittel null: Das Risiko von James-Stein ist 2/d des Risikos der Einzelmittel. Der Knopf „Nachmessen“ in E5 rechnet es mit Ihrem Zufalls-Seed nach.",
    "Stein, eine Koordinate": "Fünfzig Lager, eines davon weit vom Rest entfernt: Das Gesamtrisiko sinkt, das Risiko dieses einen Lagers steigt auf ein Vielfaches (E5).",
}
