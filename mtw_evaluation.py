"""Einstellungen, Analysen der Live-Ansichten und Verdikte. Verdikte sind reine Funktionen (Art, Text) auf Zahlen oder Zeilen der Messreihen; die App ruft st.<Art>(Text),
`tests/test_verdicts.py` prüft Schwellen und jede Zahl im Text an handgesetzten Werten."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, fields

import numpy as np

import mtw_constants as C
import mtw_estimators as est
import mtw_experiments as X
import mtw_theory as theory
from mtw_scenario import Stream, cell_seed, draw, make_plane, make_series, rounded, with_miscount


def de(x: float, nd: int = 1) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x: float, nd: int = 1) -> str:
    """Anteil (0 bis 1) als Prozentzahl mit Dezimalkomma, ohne Prozentzeichen."""
    return de(100.0 * x, nd)


def sci(x: float, digits: int = 2) -> str:
    """3,6·10⁻⁴ als Text (Null als „0“)."""
    if x == 0:
        return "0"
    mantissa, exp = f"{x:.{digits - 1}e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{mantissa.replace('.', ',')}·10{str(int(exp)).translate(sup)}"


def num(x: float, nd: int = 2) -> str:
    """Zahl mit Tausenderpunkt und Dezimalkomma."""
    s = f"{x:,.{nd}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


@dataclass(frozen=True, kw_only=True)
class Settings:
    fmt: str = C._BASE["fmt"]
    n: int = C._BASE["n"]
    ratio: int = C._BASE["ratio"]
    order: str = C._BASE["order"]
    rdist: str = C._BASE["rdist"]
    dist: str = C._BASE["dist"]
    sigma: float = C._BASE["sigma"]
    lam: float = C._BASE["lam"]
    wn: int = C._BASE["wn"]
    out: int = C._BASE["out"]
    outdist: int = C._BASE["outdist"]
    h: float = C._BASE["h"]
    shift: float = C._BASE["shift"]
    tau: float = C._BASE["tau"]
    miscount: float = C._BASE["miscount"]
    stein_d: int = C._BASE["stein_d"]
    stein_norm: float = C._BASE["stein_norm"]
    seed: int = C._BASE["seed"]


assert [f.name for f in fields(Settings)] == list(C._BASE), "Settings und Presets müssen dieselben Regler in derselben Reihenfolge führen"


# --- Reiter „Rechnen“: eine Reihe, alle Verfahren ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RechnenResult:
    settings: Settings
    abs_errors: dict          # Verfahren → Betrag des Fehlers des Mittels; dazu "Untergrenze" (die exakte SUMME auf das Zahlenformat gerundet, dann exakt durch n geteilt)
    signed_errors: dict
    floor_mean: float         # Untergrenze für Verfahren, die das Mittel selbst im Zahlenformat liefern (Welford, verschoben): das exakte Mittel auf das Format gerundet
    distinct: int             # verschiedene Werte unter den ersten 100 000 gespeicherten
    seconds: float

    @property
    def standard_error(self) -> float:
        return 1.0 / math.sqrt(self.settings.n)         # σ = 1 bei jeder Reihe


def rechnen_analysis(settings: Settings) -> RechnenResult:
    """Erzeugt die Reihe und misst den Fehler aller Verfahren gegen das exakte Mittel der gespeicherten Werte."""
    t0 = time.perf_counter()
    x = make_series(settings.n, float(settings.ratio), settings.rdist, settings.order, settings.seed, settings.fmt)
    res = X.measure_methods(x, settings.fmt, [settings.n])
    signed = {name: per_n[settings.n] for name, per_n in res.items() if name != "floor_mean"}
    signed["Untergrenze"] = signed.pop("floor_sum")
    return RechnenResult(settings, {k: abs(v) for k, v in signed.items()}, signed, abs(res["floor_mean"][settings.n]), int(len(np.unique(x[:100_000]))), time.perf_counter() - t0)


def verdict_rechnen(errors: dict, n: int) -> tuple[str, str]:
    """Verdikt der Hauptansicht „Rechnen“ aus den Beträgen der Fehler (Schlüssel naiv, Kahan, Untergrenze) bei n Werten und der Streuung 1."""
    se = 1.0 / math.sqrt(n)
    naive, kahan, floor = errors["naiv"], errors["Kahan"], errors["Untergrenze"]
    if naive <= 4.0 * floor:
        return "success", (f"Hier reicht die fortlaufende Summe: Ihr Fehler {sci(naive)} liegt höchstens beim Vierfachen der Untergrenze des Zahlenformats ({sci(floor)}). "
                           "Die Verfahren sind bei dieser Einstellung kaum zu unterscheiden.")
    if naive < se:
        return "info", (f"Die fortlaufende Summe verfehlt das exakte Mittel um {sci(naive)}, das {de(naive / floor, 0)}-Fache der Untergrenze ({sci(floor)}), aber weniger als einen Standardfehler "
                        f"({sci(se)}); Kahan: {sci(kahan)}. Gegen die Streuung der Daten ist der Fehler klein, gegen das Zahlenformat nicht.")
    return "warning", (f"Die fortlaufende Summe verfehlt das exakte Mittel um {sci(naive)}: {num(naive / se, 1)} Standardfehler ({sci(se)}), das Mittel ist also schon im Rechner falsch, bevor die Daten ins Spiel kommen. "
                       f"Kahan: {sci(kahan)}, Untergrenze des Formats: {sci(floor)}.")


# --- Reiter „Wählen“: eine Stichprobe, vier Mitten ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleResult:
    settings: Settings
    x: np.ndarray
    mean: float
    median: float
    quantile: tuple[float, float]
    expectile: float
    iqr: float
    outliers: int
    truth: dict | None        # exakte Werte der Lognormal-Verteilung (Mittelwert, Median, Quantil, Expektile), sonst None


def sample_values(s: Settings) -> tuple[np.ndarray, int]:
    """Stichprobe der gewählten Verteilung (Lognormal als Lieferzeit in Tagen mit Median 4), verschoben, mit Ausreißern (die ersten k Werte werden durch den Ausreißerabstand
    ersetzt) und auf das Raster h gerundet. Gibt (Werte, Zahl der Ausreißer) zurück."""
    x = draw(s.dist, Stream(cell_seed(s.seed, "w", s.dist)), s.wn, sigma=s.sigma, lam=s.lam)
    if s.dist == "Lognormal":
        x = x * C.LOGNORMAL_MEDIAN
    x = x + s.shift
    k = int(round(s.out * s.wn / 100.0))
    if k:
        x = x.copy()
        x[:k] = float(s.outdist)
    return rounded(x, s.h), k


def sample_analysis(s: Settings) -> SampleResult:
    x, k = sample_values(s)
    xs = np.sort(x)
    iqr = float(xs[int(0.75 * len(xs))] - xs[int(0.25 * len(xs))]) if len(xs) > 3 else 0.0
    truth = None
    if s.dist == "Lognormal" and s.shift == 0.0 and s.h == 0.0 and k == 0:
        m = C.LOGNORMAL_MEDIAN
        truth = {"mean": m * theory.gap_mean_median(s.sigma), "median": m, "quantile": m * theory.lognormal_quantile(s.sigma, s.tau), "expectile": m * theory.lognormal_expectile(s.sigma, s.tau)}
    return SampleResult(s, x, float(x.mean()), est.median(x), est.quantile_interval(x, s.tau), est.expectile(x, s.tau), iqr, k, truth)


def plan_missing(x) -> float | None:
    """Anteil der wahren Summe, den der Plan „Anzahl × Median“ auslässt, (Σx − n·Median)/Σx; None, wenn die Summe nicht positiv ist oder Werte negativ sind."""
    a = np.asarray(x, dtype=np.float64)
    total = float(a.sum())
    if total <= 0 or (a < 0).any():
        return None
    return (total - len(a) * est.median(a)) / total


def verdict_sample(mean: float, median: float, iqr: float, n: int, plan: float | None) -> tuple[str, str]:
    """Verdikt der Hauptansicht „Wählen“: wie weit liegen Mittel und Median auseinander (in Quartilsabständen), und was lässt der Plan „Anzahl × Median“ aus?"""
    if iqr <= 0:
        return "info", f"Mittel {de(mean, 2)} und Median {de(median, 2)}: Der Quartilsabstand ist null, die Werte liegen auf wenigen Stufen (siehe E7)."
    gap = (mean - median) / iqr
    text = f"Mittel {de(mean, 2)}, Median {de(median, 2)}: Der Abstand beträgt {de(abs(gap), 2)} Quartilsabstände ({de(iqr, 2)})."
    if plan is not None:
        text += (f" Der Plan „{n} × Median“ lässt {pct(plan)} % der wahren Summe aus." if plan >= 0 else f" Der Plan „{n} × Median“ überschätzt die wahre Summe um {pct(-plan)} %.")
    if abs(gap) < 0.1:
        return "success", text + " Mittel und Median liegen hier praktisch zusammen; die Frage, welche Größe gemeint ist, ändert die Antwort kaum."
    if abs(gap) < 0.5:
        return "info", text + " Die beiden Größen unterscheiden sich spürbar: Wer „das Typische“ meint, nimmt den Median, wer die Summe plant, das Mittel."
    return "warning", text + " Die beiden Größen beantworten verschiedene Fragen: Median = typischer Fall, Mittel × Anzahl = Summe."


# --- Ebene: Schwerpunkt und Fehlbuchung -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PlaneResult:
    points: np.ndarray
    with_error: np.ndarray
    centroid: np.ndarray
    centroid_error: np.ndarray
    axis_median: np.ndarray
    axis_median_error: np.ndarray
    centroid_shift: float
    median_shift: float
    within: float
    between: float


def plane_analysis(s: Settings) -> PlaneResult:
    p = make_plane(C.PLANE_N, cell_seed(s.seed, "plane"))
    q = with_miscount(p, s.miscount)
    c0, c1 = est.centroid(p), est.centroid(q)
    m0, m1 = np.array([est.median(p[:, 0]), est.median(p[:, 1])]), np.array([est.median(q[:, 0]), est.median(q[:, 1])])
    within, between = est.huygens(q, c0)
    return PlaneResult(p, q, c0, c1, m0, m1, float(np.linalg.norm(c1 - c0)), float(np.linalg.norm(m1 - m0)), within, between)


def verdict_plane(centroid_shift: float, median_shift: float, miscount: float, n: int) -> tuple[str, str]:
    """Verdikt Schwerpunkt: ein falsch gebuchter Kunde im Abstand `miscount` verschiebt den Schwerpunkt um etwa Abstand/(n + 1), die Mitte je Achse (Median) kaum."""
    text = (f"Ein falsch gebuchter Kunde im Abstand {de(miscount, 1)} km zieht den Schwerpunkt um {de(centroid_shift, 2)} km, der Median je Achse bewegt sich um {de(median_shift, 2)} km. "
            f"Der Schwerpunkt wandert mit dem Abstand (etwa Abstand geteilt durch {n + 1}).")
    if miscount == 0:
        return "info", "Ohne Fehlbuchung (Abstand 0) liegt der zusätzliche Kunde im Ursprung: Der Schwerpunkt bewegt sich kaum, und die Frage nach dem Einfluss stellt sich nicht."
    return ("warning" if centroid_shift > 1.0 else "info"), text


# --- E1 Fehler über n ---------------------------------------------------------------------------------------------------------------------

def verdict_e1_naive(naive32: dict, naive64: dict, floor32: dict, n: int) -> tuple[str, str]:
    """E1: Fehler der fortlaufenden Summe in float32 und float64 bei n Werten, Mittel über die Seeds (Zellen {"mean", "std"})."""
    kind = "warning" if naive32["mean"] > 100.0 * floor32["mean"] else "info"
    return kind, (f"float32, n = {num(n, 0)}: Die fortlaufende Summe verfehlt das exakte Mittel im Mittel über vier Zufalls-Seeds um {sci(naive32['mean'])} "
                  f"(Streuung zwischen den Seeds {sci(naive32['std'])}), das sind {num(naive32['mean'] * math.sqrt(n), 1)} Standardfehler; die Untergrenze der Summe im Format liegt bei {sci(floor32['mean'])}. "
                  f"Dieselbe Rechnung in float64: {sci(naive64['mean'])}.")


def verdict_e1_mechanism(cells: list[dict]) -> tuple[str, str]:
    """E1: Mechanismus der fortlaufenden float32-Summe (Zellen je Seed: first_index, share_1024, predicted_mean, measured_mean)."""
    first = [c["first_index"] for c in cells]
    share = min(c["share_1024"] for c in cells)
    worst = max(abs(c["predicted_mean"] - c["measured_mean"]) for c in cells)
    return "info", (f"Ursache: Die Summe erreicht 2²⁹ beim Wert Nr. {num(min(first), 0)} bis {num(max(first), 0)}. Von da an beträgt der Abstand der float32-Zahlen 64, "
                    f"und bei Werten N(1 000; 1) wird jeder Summand auf 1 024 gerundet: {"In allen übrigen Schritten" if share >= 1.0 else "In mindestens " + pct(share, 5) + " % der übrigen Schritte"} wächst die Summe um genau 1 024. Das Rauschen der Daten verschwindet aus der Summe. "
                    f"Vorhersage (Summe bei 2²⁹ plus 1 024 je Schritt) und Messung unterscheiden sich im Endmittel um höchstens {sci(worst)}.")


def verdict_e1_kahan(kahan: dict, floor: dict, pairwise: dict, numpy_sum: dict, n: int) -> tuple[str, str]:
    """E1: Kahan gegen die Untergrenze der Summe und gegen die paarweise Summe bei n Werten in float32."""
    hit = kahan["mean"] <= 1.5 * floor["mean"]
    text = (f"Kahan: {sci(kahan['mean'])} bei n = {num(n, 0)} (Untergrenze der Summe {sci(floor['mean'])}); Binärbaum {sci(pairwise['mean'])}, numpy.sum {sci(numpy_sum['mean'])}. "
            f"Gegen die Streuung der Daten (Standardfehler {sci(1.0 / math.sqrt(n))}) sind alle drei unauffällig.")
    return ("success" if hit else "info"), (text + (" Kahan erreicht die Untergrenze des Formats." if hit else " Kahan liegt über der Untergrenze des Formats."))


def verdict_e1_neumaier(neumaier: dict, kahan: dict, n: int, diag: dict | None) -> tuple[str, str]:
    """E1: Neumaier gegen Kahan (float32, N(1000; 1), n Werte) und die Diagnose (Fehler der Summe mit dem Rest in float32 und in float64)."""
    worse = neumaier["mean"] > 2.0 * kahan["mean"]
    text = (f"Neumaier: {sci(neumaier['mean'])} bei n = {num(n, 0)}, Kahan: {sci(kahan['mean'])}. "
            + ("Neumaier ist an dieser Datenform (float32, Werte um 1 000 mit Streuung 1) schlechter als Kahan. " if worse else "Neumaier ist hier nicht schlechter als Kahan. "))
    if diag is not None:
        text += (f"Diagnose an einer Reihe mit n = {num(diag['n'], 0)}: Der Rest c wächst bis {sci(diag['neumaier_max_rest'])} (bei Kahan {sci(diag['kahan_max_rest'])}); "
                 f"Fehler der Summe {sci(abs(diag['neumaier_sum_error']))}, mit dem Rest in float64 {sci(abs(diag['neumaier_wide_sum_error']))}. "
                 "Der Befund gilt für diese Datenform und dieses Zahlenformat, nicht allgemein.")
    return ("warning" if worse else "info"), text


def verdict_e1_neumaier_share(row: dict) -> tuple[str, str]:
    """E1: Neumaier gegen Kahan an `series` weiteren Reihen (float32, N(1000; 1)) bei n Werten: wie oft mehr als doppelt so ungenau, wie oft gleich, wie oft besser."""
    kind = "warning" if row["worse"] > row["series"] / 2 else "info"
    return kind, (f"Bei n = {num(row['n'], 0)} ist Neumaier in {row['worse']} von {row['series']} weiteren Reihen mehr als doppelt so ungenau wie Kahan, in {row['equal']} gleich genau und in {row['better']} genauer "
                  f"(mittlerer Fehler {sci(row['median_neumaier'])} gegen {sci(row['median_kahan'])}, Median über die Reihen). Gemessen für float32 und Werte um 1 000 mit Streuung 1.")


def verdict_e1_welford(welford32: dict, pairwise32: dict, welford64: dict, naive64: dict, n: int) -> tuple[str, str]:
    """E1: Welford ist kein Genauigkeitsverfahren fürs Mittel (float32 gegen paarweise Summe, float64 gegen die fortlaufende Summe)."""
    worse = welford32["mean"] > 5.0 * pairwise32["mean"]
    return ("warning" if worse else "info"), (f"Welford (nur der Mittelwertteil): {sci(welford32['mean'])} in float32 gegen {sci(pairwise32['mean'])} bei der paarweisen Summe (n = {num(n, 0)}); "
                                              f"in float64 {sci(welford64['mean'])} gegen {sci(naive64['mean'])} bei der fortlaufenden Summe. Der Vorteil des Verfahrens liegt beim Speicher und beim Datenstrom, nicht bei der Genauigkeit des Mittels.")


# --- E2 Auslöschung und Reihenfolge -------------------------------------------------------------------------------------------------------

def verdict_e2_ratio(naive32_100: dict, naive64_100: dict, naive64_1e10: dict, kahan32_1e4: dict, kahan32_1e6: dict, floor32_1e6: dict) -> tuple[str, str]:
    """E2: Auslöschung (n = 10⁶, Zellen in Standardfehlern): fortlaufende Summe bei μ/σ = 100 (float32, float64) und 10¹⁰ (float64), Kahan in float32 bei 10⁴ und 10⁶,
    dazu die Untergrenze der Summe im Format bei μ/σ = 10⁶ (float32)."""
    return "warning", (f"Bei μ/σ = 100 verfehlt die fortlaufende Summe in float32 das Mittel um {de(naive32_100['mean'], 2)} Standardfehler, in float64 um {sci(naive64_100['mean'])}; "
                       f"float64 erreicht {de(naive64_1e10['mean'], 2)} Standardfehler erst bei μ/σ = 10¹⁰. In float32 hält Kahan bei μ/σ = 10⁴ noch {de(kahan32_1e4['mean'], 2)} Standardfehler, bei 10⁶ aber nur noch {de(kahan32_1e6['mean'], 1)}: "
                       f"Ab dort begrenzt schon das Format der Summe jedes Verfahren (Untergrenze {de(floor32_1e6['mean'], 1)} Standardfehler).")


def verdict_e2_accumulator(cell: dict) -> tuple[str, str]:
    """E2: `numpy.mean` in float32 gegen den float64-Akkumulator (Zelle aus e2_accu, in Standardfehlern)."""
    return "success", (f"Bei μ = {num(cell['mu'], 0)} und n = {num(cell['n'], 0)} verfehlt `numpy.mean` in float32 das exakte Mittel um {de(cell['float32']['mean'], 1)} Standardfehler, mit float64-Akkumulator "
                       f"(danach auf float32 gerundet) um {de(cell['float64_accumulator_rounded']['mean'], 2)}; die Untergrenze des Formats liegt bei {de(cell['floor']['mean'], 2)}. "
                       "Der billigste Ausweg ist ein breiterer Akkumulator, nicht ein anderes Verfahren.")


def verdict_e2_order(naive: dict, welford_asc: dict, shifted_desc_lognormal: dict, kahan: dict) -> tuple[str, str]:
    """E2: Reihenfolge (float32, n = 10⁶, Zellen in Standardfehlern): naive Summe zufällig/aufsteigend/absteigend (Schlüssel der Anordnung), Welford aufsteigend, verschobene
    Summe bei absteigender lognormaler Reihe, Kahan als Bezug."""
    lo, hi = min(naive.values()), max(naive.values())
    return "info", (f"Die fortlaufende Summe verfehlt das Mittel je nach Reihenfolge um {num(lo, 0)} bis {num(hi, 0)} Standardfehler: Die Anordnung ändert die Größe um {pct((hi - lo) / hi, 0)} %, nicht das Wesen. "
                    f"Ruinös wird sie bei Welford (aufsteigend sortiert: {num(welford_asc['mean'], 0)} Standardfehler) und bei der Verschiebung um einen untypischen ersten Wert "
                    f"(lognormale Reihe, absteigend: {num(shifted_desc_lognormal['mean'], 0)}). Kahan bleibt bei {de(kahan['mean'], 3)}.")


# --- E3 Einfluss, Durchschlag, Bias-Varianz -----------------------------------------------------------------------------------------------

def verdict_e3_influence(median_plateau: float, mean_at_1e6: float, limit: float, n_total: int) -> tuple[str, str]:
    """E3: Einflusskurve bei insgesamt n Werten: Plateau des Medians gegen die Gerade des Mittels."""
    return "info", (f"Einflusskurve (n = {n_total}): Der Median erreicht ein Plateau bei {de(median_plateau, 3)} (Grenzwert {de(limit, 3)}), egal wie weit der Punkt wandert; "
                    f"das Mittel folgt ihm linear ({num(mean_at_1e6, 0)} bei t = 1 000 000).")


def verdict_e3_breakdown(ks: list[int], ns: list[int]) -> tuple[str, str]:
    """E3: Durchschlagspunkt durch Aufzählung: kleinstes k je n gegen ⌈n/2⌉."""
    ok = all(k == math.ceil(n / 2) for k, n in zip(ks, ns))
    return ("success" if ok else "warning"), (f"Durchschlagspunkt des Medians per Aufzählung für n = {ns[0]} bis {ns[-1]}: " + ", ".join(f"n = {n}: k = {k}" for n, k in zip(ns, ks))
                                              + (f" – stets ⌈n/2⌉; das Mittel schlägt schon bei k = 1 durch." if ok else " – weicht von ⌈n/2⌉ ab."))


def verdict_e3_contamination(shift10: float, shift40: float, shift49: float, shift50: float, mean40: float) -> tuple[str, str]:
    """E3: Verschiebung des Medians (n = 100, Ausreißer bei 100) bei 10, 40, 49 und 50 % Ausreißern, dazu die des Mittels bei 40 %."""
    return "info", (f"Median verschoben um {de(shift10, 3)} (10 %), {de(shift40, 3)} (40 %), {de(shift49, 3)} (49 %); bei 50 % springt er auf {de(shift50, 1)}. "
                    f"Das Mittel wandert bei 40 % um {de(mean40, 1)}.")


def verdict_e3_bias(theta: float, mse_c09: float, formula_c09: float, cstar: float) -> tuple[str, str]:
    """E3: ein um 10 % verkleinertes Mittel (c = 0,9) bei θ = μ√n/σ gegen das Mittel (MSE 1)."""
    better = mse_c09 < 1.0
    return ("success" if better else "info"), (f"θ = {de(theta, 1)}: Das um 10 % verkleinerte Mittel hat den MSE {de(mse_c09, 3)} (Formel {de(formula_c09, 3)}) gegen 1 beim Mittel – "
                                              + ("es gewinnt. " if better else "es verliert. ") + f"Das beste c wäre {de(cstar, 2)} und enthält das unbekannte μ.")


# --- E5 James-Stein -----------------------------------------------------------------------------------------------------------------------

def verdict_e5(ratio_d: float, d: int, norm: float, exact: float, pp: float) -> tuple[str, str]:
    """E5: Risiko des James-Stein-Schätzers geteilt durch das des Einzelmittels bei d Lagern und ‖μ‖ = norm (simuliert), exakt, Positive Part."""
    two_over_d = 2.0 / d
    kind = "success" if ratio_d < 0.95 else "info"
    text = f"d = {d}, ‖μ‖ = {de(norm, 1)}: Risiko James-Stein / Einzelmittel = {de(ratio_d, 3)} (exakt {de(exact, 3)}), Positive Part {de(pp, 3)}."
    if norm == 0:
        text += f" Bei ‖μ‖ = 0 ist das Verhältnis 2/d = {de(two_over_d, 3)}."
    return kind, text


def verdict_e5_coordinate(max_risk: float, d: int, at_norm: float) -> tuple[str, str]:
    """E5: größtes Risiko der ERSTEN Koordinate allein (Einzelmittel: 1) für James-Stein mit Positive Part bei d Lagern."""
    kind = "warning" if max_risk > 1.2 else "info"
    return kind, (f"Das Gesamtrisiko sinkt, aber das Risiko des ersten Lagers allein steigt bei d = {d} auf das {de(max_risk, 2)}-Fache des Einzelmittels (bei ‖μ‖ = {de(at_norm, 0)}): "
                  "Die Schrumpfung gewinnt im Ganzen und verliert beim einzelnen Lager, dessen Mittel weit vom Rest liegt.")


# --- E6 Welche Größe will ich? -----------------------------------------------------------------------------------------------------------

def verdict_e6_gap(sigma: float, gap: float, below: float, miss_median: float, top10: float) -> tuple[str, str]:
    """E6: Lognormal(0; σ): Mittelwert/Median, Anteil der Werte unter dem Mittelwert, ausgelassener Anteil beim Median als Schätzer des Mittelwerts, Summenanteil der größten 10 %."""
    return "info", (f"σ = {de(sigma, 1)}: Der Mittelwert liegt das {de(gap, 2)}-Fache über dem Median; {pct(below)} % der Werte liegen unter dem Mittelwert, und die größten 10 % tragen {pct(top10)} % der Summe. "
                    f"Der Median als Schätzer des Mittelwerts lässt {pct(miss_median)} % aus (der Populationsmedian; der Stichproben-Median erreicht ihn für wachsendes n).")


def verdict_e6_targets(sigma: float, mean_to_mean: float, median_to_mean: float, mean_to_median: float, median_to_median: float, n: int) -> tuple[str, str]:
    """E6: typischer relativer Fehler (n Werte) von Mittel und Median gegen beide Ziele."""
    return "warning", (f"σ = {de(sigma, 1)}, n = {n}: Das Mittel trifft den Mittelwert mit {pct(mean_to_mean)} % und verfehlt den Median um {pct(mean_to_median)} %; "
                       f"der Median trifft den Median mit {pct(median_to_median)} % und verfehlt den Mittelwert um {pct(median_to_mean)} %. Beide sind gut, nur für verschiedene Fragen.")


def verdict_e6_delivery(sigma: float, missing: float, small_median: float, small_mean: float, too_small_median: float, orders: int, sample: int) -> tuple[str, str]:
    """E6: Lieferzeit-Szenario (frei gewählt, Median 4 Tage): ausgelassener Anteil der Monatssumme beim Plan „Aufträge × Median“; Schätzung der Monatssumme aus `sample` Aufträgen."""
    return "warning", (f"σ = {de(sigma, 1)}: Der Plan „{num(orders, 0)} × Median“ lässt {pct(missing)} % der Monatssumme aus. Aus {sample} Aufträgen geschätzt ist „{num(orders, 0)} × Stichproben-Median“ "
                       f"in {pct(too_small_median)} % der Fälle zu klein (typischer Fehler {pct(small_median)} % gegen {pct(small_mean)} % beim Mittel). Das Szenario ist frei gewählt, keine Messung.")


# --- E7 Ties und Rundung ------------------------------------------------------------------------------------------------------------------

def verdict_e7_ties(is0: float, is_half: float, is1: float, n: int, f0: float) -> tuple[str, str]:
    """E7: Poisson(0,7), n Werte: Anteil der Läufe mit Median 0, 0,5 und 1; F(0) = f0 liegt knapp unter ½."""
    return "warning", (f"Poisson(0,7) hat F(0) = {de(f0, 4)}, knapp unter ½, und der Populationsmedian ist 1. Bei n = {n} ist der Stichprobenmedian in {pct(is0)} % der Läufe 0, in {pct(is1)} % 1 "
                       f"und in {pct(is_half)} % 0,5: Ein Fast-Gleichstand lässt ihn zwischen den Stufen springen.")


def verdict_e7_rounding(h: float, rmse_mean: float, rmse_median: float, zero_share: float, mu: float) -> tuple[str, str]:
    """E7: Normal(μ; 1) auf das Raster h gerundet, n = 101: RMSE von Mittel und Median gegen μ und Anteil der Läufe mit Median genau 0."""
    worse = rmse_median > 1.5 * rmse_mean
    return ("warning" if worse else "info"), (f"Raster h = {de(h, 2)}: RMSE gegen μ = {de(mu, 1)} beim Mittel {de(rmse_mean, 3)}, beim Median {de(rmse_median, 3)}; "
                                              f"der Median steht in {pct(zero_share)} % der Läufe genau auf 0.")


# --- E8 Quantil gegen Expektile -----------------------------------------------------------------------------------------------------------

def verdict_e8_gap(label: str, tau: float, gap: float, std: float, usable: bool) -> tuple[str, str]:
    """E8: Lücke (Expektile − Quantil) in Quartilsabständen bei Stufe τ für eine Verteilung."""
    if not usable:
        return "info", f"{label}, τ = {de(tau, 2)}: Lücke {de(gap, 3)} Quartilsabstände, Streuung zwischen den Seeds {de(std, 3)} – zu unsicher, um sie zu deuten."
    where = "über" if gap > 0 else "unter"
    return "info", f"{label}, τ = {de(tau, 2)}: Die Expektile liegt {de(abs(gap), 3)} Quartilsabstände {where} dem Quantil gleicher Stufe."


def verdict_e8_cover(sigma: float, tau: float, mean_cover: float, expectile_cover: float, quantile: float, expectile: float, mean: float) -> tuple[str, str]:
    """E8: Lognormal(0; σ), Stufe τ: Quantil, Expektile und Mittelwert; Anteil der Nachfrage, den ein Bestand in Höhe des Mittelwerts bzw. der Expektile deckt."""
    return "warning", (f"Lognormal(0; {de(sigma, 1)}), τ = {de(tau, 1)}: Quantil {de(quantile, 3)}, Expektile {de(expectile, 3)}, Mittelwert {de(mean, 3)}. Ein Bestand in Höhe des Mittelwerts deckt "
                       f"{pct(mean_cover, 0)} % der Nachfrage, in Höhe der Expektile {pct(expectile_cover, 0)} %, gewünscht war {pct(tau, 0)} %. Das Muster „Mittel-Bestand deckt zu wenig“ ist in "
                       "`forecast-inventory-demo` schon belegt; hier steht nur die Zahl für die Expektile.")
