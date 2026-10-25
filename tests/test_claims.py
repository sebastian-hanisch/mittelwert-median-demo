"""Jede Zahl der README wird hier aus der vorgerechneten Messreihe (data/mtw_results.json) gebildet und in der README gesucht (DEMO-PLAYBOOK §4).

Die Datei selbst ist durch tests/test_results.py gegen die Quelldateien abgesichert, und tests/test_results.py rechnet Zellen live nach. Nicht geprüft werden die Einmalmessung der Laufzeit
(„Rechengrenze“) und die Ausgabe des Mutationswerkzeugs („Tests“); beide stehen in der README als solche."""

import math
import re
from pathlib import Path

import pytest

import mtw_experiments as X
import mtw_results as R
import mtw_theory as T

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
N7 = 10_000_000


def de(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x, nd=1):
    return de(100 * x, nd)


def th(x, nd=0):
    """Deutsche Zahl mit Punkt als Tausendertrenner und Komma als Dezimalzeichen."""
    s = f"{x:,.{nd}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def sci(x, digits=2):
    """3,8·10⁻¹⁰ als Text."""
    if x == 0:
        return "0"
    mantissa, exp = f"{x:.{digits - 1}e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{mantissa.replace('.', ',')}·10{str(int(exp)).translate(sup)}"


def signed(x, nd=3):
    return ("+" if x > 0 else "−" if x < 0 else "") + de(abs(x), nd)


def has(text):
    assert text in README, text


def c(fmt, method, n):
    return R.e1_cell(fmt, method, n)


# ---- H1 bis H8: Rechnen ---------------------------------------------------------------------------------------------------------------------

def test_h1_naive_float32_sum_and_its_mechanism():
    naive = c("float32", "naiv", N7)
    assert all(e > 0 for e in naive["per_seed"])                                                     # das Mittel liegt zu hoch
    has(f"verfehlt das Mittel um {de(naive['mean'], 2)}** (vier Zufalls-Seeds, Streuung zwischen den Seeds {sci(naive['std'], 1)})")
    has(f"das sind {th(naive['mean'] * math.sqrt(N7))} Standardfehler")
    cells = R.e1_mechanism()["cells"]
    has(f"Nr. {th(min(x['first_index'] for x in cells))} bis {th(max(x['first_index'] for x in cells))}")
    has(f"in mindestens {pct(min(x['share_1024'] for x in cells), 5)} % der übrigen Schritte Zuwachs genau 1 024")
    has(f"unterscheiden sich um höchstens {sci(max(abs(x['predicted_mean'] - x['measured_mean']) for x in cells))}")
    assert all(x["increment_max"] == 1024.0 for x in cells) and all(530_000 < x["first_index"] < 550_000 for x in cells)
    has(f"Fehler {de(naive['mean'], 2)}; die Summe erreicht 2²⁹ bei Wert Nr. {th(min(x['first_index'] for x in cells))} bis {th(max(x['first_index'] for x in cells))}")


def test_h2_pairwise_sums_stay_below_a_third_of_a_standard_error():
    se = 1 / math.sqrt(N7)
    b, s = c("float32", "Binärbaum", N7), c("float32", "numpy.sum", N7)
    assert b["mean"] < se / 3 and s["mean"] < se / 3
    has(f"Binärbaum {sci(b['mean'])} = {de(b['mean'] / se, 2)} Standardfehler, numpy.sum {sci(s['mean'])} = {de(s['mean'] / se, 2)} Standardfehler (Standardfehler {sci(se)})")


def test_h3_kahan_equals_the_floor_everywhere():
    for fmt in ("float32", "float64"):
        for n in X.E1_CHECKS:
            assert c(fmt, "Kahan", n)["mean"] == c(fmt, "floor_sum", n)["mean"]
    for fmt in ("float32", "float64"):
        for key, ratio in R.e2_ratio(fmt)["ratios"].items():
            k, f = ratio["methods"]["Kahan"]["mean"], ratio["methods"]["floor_sum"]["mean"]
            if key == "0":
                assert k <= 3.01 * f and k < 1e-7 and f < 1e-7                                          # bei μ = 0 ist die Summe selbst fast null: kein Gleichstand
            else:
                assert k == f, (fmt, key)
        for dist, orders in R.e2_order(fmt)["cells"].items():
            for order, methods in orders.items():
                assert methods["Kahan"]["mean"] == methods["floor_sum"]["mean"], (fmt, dist, order)
    has(f"({sci(c('float32', 'Kahan', N7)['mean'])} bei 10⁷)")
    has("Kahan = Untergrenze der Summe in allen fünf Größen, in beiden Formaten, in allen sechs Reihenfolge-Zellen und bei jedem μ/σ ab 1 (n = 10⁶); "
        "bei μ = 0 liegt Kahan höchstens das Dreifache über der Untergrenze, beide unter 10⁻⁷ Standardfehlern")


def test_h4_neumaier_is_worse_than_kahan_in_float32():
    n6, n7 = c("float32", "Neumaier", 1_000_000), c("float32", "Neumaier", N7)
    k6, k7 = c("float32", "Kahan", 1_000_000), c("float32", "Kahan", N7)
    assert n6["mean"] > 2 * k6["mean"] and n7["mean"] > 10 * k7["mean"]
    has(f"{sci(n7['mean'])} gegen {sci(k7['mean'])} bei 10⁷ Werten")
    has(f"n = 10⁶: {sci(n6['mean'])} gegen {sci(k6['mean'])}; n = 10⁷: {sci(n7['mean'])} ({de(n7['mean'] * math.sqrt(N7), 1)} Standardfehler)")
    d = R.e1_diagnosis()["rows"][-1]
    assert d["neumaier_wide"]["sum_error"] == 0.0 and abs(d["neumaier"]["sum_error"]) > 10 and d["neumaier"]["max_rest"] > 1e6 > 100 * d["kahan"]["max_rest"]
    has(f"(n = {th(d['n'])}): größter Rest {sci(d['neumaier']['max_rest'])} (Kahan {de(d['kahan']['max_rest'], 0)}), Fehler der Summe {de(abs(d['neumaier']['sum_error']), 2)}, mit dem Rest in float64 0")
    for n, label in ((1_000_000, "(n = 10⁶)"), (2_000_000, "(2·10⁶)"), (5_000_000, "(5·10⁶)")):
        r = R.e1_neumaier_share(n)
        assert r["better"] == 0 and r["worse"] + r["equal"] <= r["series"] == 20
        has(f"{r['worse']} von {r['series']} {label}")
    has(f"in {R.e1_neumaier_share(2_000_000)['worse']} von 20 weiteren Reihen mit zwei Millionen Werten mehr als doppelt so ungenau")
    has(f"(bei n = 10⁶ in {R.e1_neumaier_share(1_000_000)['equal']} von 20 Reihen gleich)")
    has(f"liegt Neumaier schon bei {de(R.e2_ratio_cell('float32', 1e4, 'Neumaier')['mean'], 2)} Standardfehlern gegen {de(R.e2_ratio_cell('float32', 1e4, 'Kahan')['mean'], 2)} bei Kahan")
    assert R.e2_ratio_cell("float32", 1e4, "Neumaier")["mean"] > R.e2_ratio_cell("float32", 1e4, "Kahan")["mean"]


def test_h5_welford():
    se = 1 / math.sqrt(N7)
    w32, w64 = c("float32", "Welford", N7), c("float64", "Welford", N7)
    has(f"{sci(w32['mean'])} = {de(w32['mean'] / se, 1)} Standardfehler gegen {de(c('float32', 'Binärbaum', N7)['mean'] / se, 2)} beim Binärbaum")
    has(f"aufsteigend sortiert (n = 10⁶) {th(R.e2_order_cell('float32', 'Normal', 'aufsteigend', 'Welford')['mean'])} Standardfehler")
    has(f"float64: {sci(w64['mean'])} gegen {sci(c('float64', 'naiv', N7)['mean'])} bei der fortlaufenden Summe")
    assert w32["mean"] > 10 * c("float32", "Binärbaum", N7)["mean"] and w64["mean"] > 1e-11


def test_h6_h7_cancellation_and_the_wide_accumulator():
    has(f"float32 {de(R.e2_ratio_cell('float32', 100.0, 'naiv')['mean'], 2)} Standardfehler, float64 {sci(R.e2_ratio_cell('float64', 100.0, 'naiv')['mean'])}")
    has(f"float64 erreicht {de(R.e2_ratio_cell('float64', 1e10, 'naiv')['mean'], 2)} Standardfehler bei μ/σ = 10¹⁰")
    has(f"Untergrenze der Summe {de(R.e2_ratio_cell('float32', 1e6, 'floor_sum')['mean'], 2)} Standardfehler bei 10⁶ und {de(R.e2_ratio_cell('float32', 1e7, 'floor_sum')['mean'], 1)} bei 10⁷")
    a = R.e2_accumulator(1e6, 1_000_000)
    has(f"{de(a['float32']['mean'], 1)} Standardfehler, mit float64-Akkumulator {de(a['float64_accumulator_rounded']['mean'], 2)} (Untergrenze {de(a['floor']['mean'], 2)})")
    assert a["float32"]["mean"] > 30 * a["float64_accumulator_rounded"]["mean"]
    # Kahan hält die Untergrenze auch bei großem μ/σ; die Daten sind dort grob gerundet
    assert R.e2_ratio("float32")["ratios"]["1e+06"]["distinct_first_1e5"] < 200 and R.e2_ratio("float32")["ratios"]["1e+07"]["distinct_first_1e5"] < 20


def test_h8_order():
    o = {k: R.e2_order_cell("float32", "Normal", k, "naiv")["mean"] for k in ("zufällig", "aufsteigend", "absteigend")}
    has(f"naiv (n = 10⁶, Standardfehler): {th(o['zufällig'])} zufällig, {th(o['aufsteigend'])} aufsteigend, {th(o['absteigend'])} absteigend")
    assert abs(o["aufsteigend"] - o["zufällig"]) / o["zufällig"] < 0.15 and abs(o["absteigend"] - o["zufällig"]) / o["zufällig"] < 0.15
    has("**Ja**, mit etwa ±13 %")
    assert round(100 * abs(o["aufsteigend"] - o["zufällig"]) / o["zufällig"]) == 13 and round(100 * abs(o["absteigend"] - o["zufällig"]) / o["zufällig"]) == 13
    has(f"Verschiebung um x₁ bei absteigender schiefer Reihe {th(R.e2_order_cell('float32', 'Lognormal', 'absteigend', 'verschoben')['mean'])}")
    has(f"Kahan {de(R.e2_order_cell('float32', 'Normal', 'zufällig', 'Kahan')['mean'], 3)} in allen drei Anordnungen")
    assert len({R.e2_order_cell("float32", "Normal", k, "Kahan")["mean"] for k in o}) == 1


# ---- H9 bis H12, H14: Einfluss, Durchschlag, Effizienz, Bias-Varianz ----------------------------------------------------------------------------

def test_h9_h10_influence_and_contamination():
    s101, s100 = R.sensitivity(101, 1e6)["median"], R.sensitivity(100, 1e6)["median"]
    has(f"Median-Plateau {de(s101['mean'], 3)} (n = 101, Streuung zwischen den Seeds {de(s101['std'], 3)}) und {de(s100['mean'], 3)} (n = 100), Grenzwert {de(T.INFLUENCE_PLATEAU, 3)}; bei t = 0,25 erst {de(R.sensitivity(101, 0.25)['median']['mean'], 3)}")
    above101, above100 = s101["mean"] - T.INFLUENCE_PLATEAU, s100["mean"] - T.INFLUENCE_PLATEAU
    has(f"die Plateaus {de(s101['mean'], 3)} (n = 101) und {de(s100['mean'], 3)} (n = 100) liegen {de(above101, 3)} bis {de(above100, 3)} über dem Grenzwert {de(T.INFLUENCE_PLATEAU, 3)}")
    assert 0.005 < above101 < above100 < 0.03 and above101 > 3 * s101["std"] / 2 and R.sensitivity(101, 1e6)["mean_dev"] < 1e-6            # größer als die Streuung zwischen den Seeds
    assert R.sensitivity(101, 1e6)["mean"]["mean"] == pytest.approx(1e6, abs=0.1)
    has(f"{th(R.e3()['sensitivity']['runs'])} Stichproben je Seed")
    shifts = [de(R.contamination_row(100.0, e)["median"]["mean"], 3) for e in (0.05, 0.1, 0.2, 0.4, 0.49)]
    has("Median-Verschiebung bei 5 / 10 / 20 / 40 / 49 %: " + " / ".join(shifts))
    has(f"bei 50 % {de(R.contamination_row(100.0, 0.5)['median']['mean'], 2)}, ab 55 % {de(R.contamination_row(100.0, 0.55)['median']['mean'], 1)}; das Mittel bei 40 % um {de(R.contamination_row(100.0, 0.4)['mean']['mean'], 1)}")
    for e in (0.05, 0.1, 0.2, 0.4):
        assert R.contamination_row(100.0, e)["mean"]["mean"] == pytest.approx(100.0 * e, abs=0.01)                   # das Mittel wandert um ε·M


def test_h11_breakdown():
    rows = R.e3()["breakdown"]["rows"]
    assert all(set(r["k"]) == {math.ceil(r["n"] / 2)} and set(r["mean_k"]) == {1} for r in rows)
    has(f"größte Verschiebung davor im Mittel über die Seeds {de(min(r['max_shift_below']['mean'] for r in rows), 2)} bis {de(max(r['max_shift_below']['mean'] for r in rows), 2)} Streuungen "
        f"(je Seed {de(min(min(r['max_shift_below']['per_seed']) for r in rows), 2)} bis {de(max(max(r['max_shift_below']['per_seed']) for r in rows), 2)})")
    has("hier im Mittel über die Seeds")
    has("Aufzählung für n = 3 bis 12 in allen vier Seeds ⌈n/2⌉, beim Mittel 1")


def test_h12_efficiency():
    has("n = 3 / 5 / 9 / 21 / 101 / 1 001: " + " / ".join(de(R.efficiency(n), 4) for n in (3, 5, 9, 21, 101, 1001)))
    has("gerades n = 4 / 10 / 100: " + " / ".join(de(R.efficiency(n), 4) for n in (4, 10, 100)) + f"; Grenzwert 2/π = {de(T.EFFICIENCY_LIMIT, 4)}")
    assert R.efficiency(4) > R.efficiency(3) and R.efficiency(100) > R.efficiency(101)


def test_h14_bias_variance():
    cells = [R.bias_variance_row(t)["cells"]["0,9"] for t in (0.0, 1.0, 2.0, 4.0)]
    has("c = 0,9 bei θ = 0 / 1 / 2 / 4: " + " / ".join(de(x["formula"], 2) for x in cells))
    for x in cells:
        assert x["sim"]["mean"] == pytest.approx(x["formula"], abs=0.003) and x["formula"] < 1
    has(f"gewinnt für θ unter {de(math.sqrt(1.9 / 0.1), 2)}")
    has(f"c = 0,8 bei θ = 4: {de(R.bias_variance_row(4.0)['cells']['0,8']['formula'], 2)}")
    assert R.bias_variance_row(4.0)["cells"]["0,8"]["sim"]["mean"] == pytest.approx(R.bias_variance_row(4.0)["cells"]["0,8"]["formula"], abs=0.003)


# ---- H15, H16: James-Stein ----------------------------------------------------------------------------------------------------------------

def test_h15_h16_james_stein():
    dims = X.STEIN_DIMS
    has("d = 3 / 5 / 10 / 50 bei ‖μ‖ = 0: " + " / ".join(de(R.stein_cell("eine", d, 0.0)["js"]["mean"], 3) for d in dims) + " (2/d = " + " / ".join(de(2 / d, 3) for d in dims) + ")")
    has("Positive Part " + " / ".join(de(R.stein_cell("eine", d, 0.0)["pp"]["mean"], 3) for d in dims))
    has("bei ‖μ‖ = 20 noch " + " / ".join(de(R.stein_cell("eine", d, 20.0)["js"]["mean"], 3) for d in dims))
    for d in dims:
        assert R.stein_cell("eine", d, 0.0)["js"]["mean"] == pytest.approx(2 / d, abs=0.012) and R.stein_cell("eine", d, 0.0)["js_exact"] == pytest.approx(2 / d)
    biggest = max(R.stein_cell(m, d, nm)["js"]["mean"] for m in ("eine", "gleich") for d in dims for nm in X.STEIN_NORMS)
    has(f"die Simulation liegt höchstens bei {de(biggest, 4)}")
    assert biggest < 1.002
    coords = {d: max((R.stein_cell("eine", d, nm)["koordinate1_pp"]["mean"], nm) for nm in X.STEIN_NORMS) for d in dims}
    has(f"größtes Risiko des ersten Lagers (Einzelmittel 1): {de(coords[3][0], 2)} (d = 3) / {de(coords[5][0], 2)} (5) / {de(coords[10][0], 2)} (10) bei ‖μ‖ = 3 und {de(coords[50][0], 2)} (d = 50, ‖μ‖ = 8)")
    assert [coords[d][1] for d in dims] == [3.0, 3.0, 3.0, 8.0]
    has("in " + " / ".join(pct(R.stein_zero_share(d)) for d in dims) + " % der Läufe auf 0")
    has(f"bis auf das {de(coords[50][0], 2)}-Fache** (d = 50, ‖μ‖ = 8)")
    assert R.stein_cell("eine", 10, 5.0)["js"]["mean"] == pytest.approx(R.stein_cell("gleich", 10, 5.0)["js"]["mean"], abs=0.01)         # hängt nur von ‖μ‖ ab


# ---- H17 bis H22: Median, Ties, Rundung, Quantil und Expektile ---------------------------------------------------------------------------------------

def test_h17_h18_target_and_delivery():
    has("Mittelwert/Median " + " / ".join(de(R.shape_row(s)["gap"], 2) for s in (0.5, 1.0, 1.5, 2.0)) + " (σ = 0,5 / 1 / 1,5 / 2)")
    has(f"bei n = 200: {de(R.target_row(1.0)['median_mittelwert']['typisch']['mean'], 3)} (σ = 1; Grenzwert {de(R.target_row(1.0)['miss_limit'], 4)}) und {de(R.target_row(2.0)['median_mittelwert']['typisch']['mean'], 3)} (σ = 2; {de(R.target_row(2.0)['miss_limit'], 4)})")
    for s in (0.5, 1.0, 1.5, 2.0):
        assert R.shape_row(s)["gap"] == pytest.approx(math.exp(s * s / 2)) and R.target_row(s)["median_mittelwert"]["typisch"]["mean"] == pytest.approx(R.target_row(s)["miss_limit"], abs=0.02)
    has(f"{de(R.target_row(2.0)['mittel_mittelwert']['q95']['mean'], 3)} (Streuung zwischen den Seeds {de(R.target_row(2.0)['mittel_mittelwert']['q95']['std'], 3)})")
    rows = [R.delivery_row(s) for s in (0.4, 0.8, 1.2)]
    has("σ = 0,4 / 0,8 / 1,2: " + " / ".join(pct(r["plan_median_missing"]["mean"]) for r in rows) + " % (exakt " + " / ".join(pct(r["exact_missing"]) for r in rows) + " %)")
    has("im gewählten Lieferzeit-Szenario " + " / ".join(pct(r["plan_median_missing"]["mean"]) for r in rows) + " % der Monatssumme aus (σ = 0,4 / 0,8 / 1,2)")
    has("„× Mittelwert“ weicht im Mittel um " + " / ".join(pct(r["plan_mean_deviation"]["mean"]) for r in rows) + " % ab")
    has("in " + " / ".join(pct(r["schaetzung_median"]["zu_klein"]["mean"]) for r in rows) + " % der Fälle zu klein")
    for r in rows:
        assert r["plan_median_missing"]["mean"] == pytest.approx(r["exact_missing"], abs=0.005)


def test_h17_median_error_does_not_shrink_with_n():
    """„Bei jedem n“: Der typische Fehler des Medians gegen den Mittelwert liegt bei n = 2 000 (Live-Funktion) wie bei n = 200 beim Grenzwert."""
    big = X.target_errors(1.0, 2000, 400, X.SEEDS[0])
    assert big["median_mittelwert"]["typisch"] == pytest.approx(T.median_miss_share(1.0), abs=0.03)
    assert R.target_row(1.0)["median_mittelwert"]["typisch"]["mean"] == pytest.approx(T.median_miss_share(1.0), abs=0.03)


def test_h19_ties():
    has("Poisson λ = 0,7 / 2 / 10: bei n = 10 in " + " / ".join(pct(R.poisson_cell(l, 10)["non_integer"]["mean"]) for l in (0.7, 2.0, 10.0)) + " % nicht ganzzahlig, bei n = 100 in "
        + " / ".join(pct(R.poisson_cell(l, 100)["non_integer"]["mean"]) for l in (0.7, 2.0, 10.0)) + " %")
    nt = R.near_tie(100)
    has(f"Poisson(0,7), n = 100: Median 0 in {pct(nt['is0']['mean'])} %, 0,5 in {pct(nt['is_half']['mean'])} %, 1 in {pct(nt['is1']['mean'])} % (F(0) = {de(R.load()['e7']['f0'], 4)})")
    assert R.load()["e7"]["f0"] == pytest.approx(T.poisson_cdf(0.7, 0)) and R.load()["e7"]["f0"] < 0.5
    has(f"n = 1 000: {pct(R.near_tie(1000)['is0']['mean'])} % bei 0; n = 10 000: {pct(R.near_tie(10000)['is0']['mean'])} %")
    assert R.near_tie(10000)["is0"]["mean"] < R.near_tie(1000)["is0"]["mean"] < R.near_tie(100)["is0"]["mean"] < 0.5
    for l in (0.7, 2.0, 10.0):
        assert R.poisson_cell(l, 11)["non_integer"]["mean"] == 0.0 and R.poisson_cell(l, 101)["non_integer"]["mean"] == 0.0


def test_h20_rounding():
    r1, r0 = R.rounding_row(1.0), R.rounding_row(0.0)
    has(f"RMSE des Medians {de(r1['rmse_median']['mean'], 3)} gegen {de(r1['rmse_mean']['mean'], 3)} beim Mittel (ungerundet {de(r0['rmse_median']['mean'], 3)} gegen {de(r0['rmse_mean']['mean'], 3)}); der Median steht in {pct(r1['median_zero']['mean'])} % der Läufe auf 0")
    assert r1["rmse_median"]["mean"] > 3 * r1["rmse_mean"]["mean"] and R.load()["e7"]["rounding"]["mu"] == 0.3


def test_h21_h22_quantile_and_expectile():
    has("Lücke (Expektile − Quantil) / Quartilsabstand bei τ = 0,8: " + ", ".join(f"{name} {signed(R.gap_cell(lb, 0.8)['gap']['mean'])}" for name, lb in (
        ("Normal", "Normal"), ("Exponential", "Exponential"), ("Lognormal(0; 1)", "Lognormal(0; 1)"), ("Lognormal(0; 1,5)", "Lognormal(0; 1,5)"), ("Pareto(2,5)", "Pareto(2,5)"))))
    has(f"Exponential: {signed(R.gap_cell('Exponential', 0.5)['gap']['mean'])} bei τ = 0,5, {signed(R.gap_cell('Exponential', 0.99)['gap']['mean'])} bei τ = 0,99")
    has(f"Bei τ = 0,8 reicht die Lücke von {signed(R.gap_cell('Normal', 0.8)['gap']['mean'])} (Normal) bis {signed(R.gap_cell('Lognormal(0; 1,5)', 0.8)['gap']['mean'])} (Lognormal(0; 1,5)) Quartilsabstände")
    c15 = R.gap_cell("Lognormal(0; 1,5)", 0.99)
    assert not c15["usable"]
    has(f"({signed(c15['gap']['mean'])} ± {de(c15['gap']['std'], 3)})")
    ex = R.lognormal_exact()
    r80 = [r for r in ex["rows"] if r["tau"] == 0.8][0]
    has(f"Mittelwert {de(ex['mean'], 4)} deckt {pct(ex['mean_cdf'])} %; Quantil {de(r80['q'], 4)}, Expektile {de(r80['e'], 4)} (deckt {pct(r80['F_e'])} %)")
    assert ex["mean_cdf"] == pytest.approx(T.share_below_mean(0.5)) and r80["q"] == pytest.approx(T.lognormal_quantile(0.5, 0.8)) and r80["e"] == pytest.approx(T.lognormal_expectile(0.5, 0.8))
    assert round(100 * ex["mean_cdf"]) == 60
    signs = {math.copysign(1, R.gap_cell("Exponential", t)["gap"]["mean"]) for t in X.GAP_TAUS}
    assert signs == {1.0, -1.0}


# ---- die Tabellen der README -----------------------------------------------------------------------------------------------------------------

def test_e1_tables():
    names = (("floor_sum", "Untergrenze der Summe"), ("floor_mean", "Untergrenze des Mittels"), ("naiv", "naiv (fortlaufend)"), ("numpy.sum", "numpy.sum"), ("Binärbaum", "Binärbaum"), ("verschoben", "verschoben"),
             ("Kahan", "Kahan"), ("Kahan mit Rest", "Kahan mit Rest"), ("Neumaier", "Neumaier"), ("Welford", "Welford"))
    for fmt in ("float32", "float64"):
        for m, label in names:
            has(f"| {label} | " + " | ".join(sci(c(fmt, m, n)["mean"]) for n in X.E1_CHECKS) + " |")
    assert c("float32", "Kahan mit Rest", 100_000)["mean"] == 0 and c("float32", "Kahan", 100_000)["mean"] > 0
    has("Kahan mit Rest hält die Summe in float32 bis n = 10⁵ vollständig (Fehler 0), danach fast")
    for n in (1000, 10_000, 100_000):
        assert c("float32", "Kahan mit Rest", n)["mean"] == 0.0


def test_e2_table():
    ratios = R.e2_ratio("float32")["ratios"]
    for m, label in (("floor_sum", "Untergrenze der Summe"), ("naiv", "naiv"), ("numpy.sum", "numpy.sum"), ("Kahan", "Kahan"), ("Neumaier", "Neumaier"), ("Welford", "Welford"), ("verschoben", "verschoben")):
        has(f"| {label} | " + " | ".join(sci(ratios[k]["methods"][m]["mean"]) for k in ratios) + " |")
    assert list(ratios) == [f"{v:g}" for v in (0.0, 1.0, 1e2, 1e4, 1e6, 1e7)]


def test_e3_e5_e6_e8_tables():
    eps = (0.05, 0.1, 0.2, 0.4, 0.49, 0.5, 0.55)
    has("| Verschiebung des Mittels | " + " | ".join(de(R.contamination_row(100.0, e)["mean"]["mean"], 1) for e in eps) + " |")
    has("| Verschiebung des Medians | " + " | ".join(de(R.contamination_row(100.0, e)["median"]["mean"], 3 if e < 0.5 else 2) for e in eps) + " |")
    for nm in (0.0, 2.0, 5.0, 8.0, 20.0):
        has(f"| {de(nm, 0)} | " + " | ".join(f"{de(R.stein_cell('eine', d, nm)['js']['mean'], 3)} ({de(R.stein_cell('eine', d, nm)['pp']['mean'], 3)})" for d in X.STEIN_DIMS) + " |")
    for s in (0.5, 1.0, 1.5, 2.0):
        sh, tg = R.shape_row(s), R.target_row(s)
        has(f"| {de(s, 1)} | {de(sh['gap'], 2)} | {pct(sh['below_mean']['mean'])} % | {pct(sh['top10_share']['mean'])} % | {pct(tg['mittel_mittelwert']['typisch']['mean'])} % | {pct(tg['median_mittelwert']['typisch']['mean'])} % |")
        assert sh["below_mean"]["mean"] == pytest.approx(sh["below_mean_exact"], abs=0.002)
    for lb in R.gap_labels():
        cells = [signed(R.gap_cell(lb, t)["gap"]["mean"]) if R.gap_cell(lb, t)["usable"] else "unsicher" for t in X.GAP_TAUS]
        has(f"| {lb} | " + " | ".join(cells) + " |")


def test_the_readme_has_no_unresolved_marks_and_ends_with_the_footer():
    assert "TODO:" not in README and "{{" not in README and "PLATZHALTER" not in README
    assert README.rstrip().endswith("[Interaktive Demos](https://sebastianhanisch.net/demos.html).")
    assert not re.search(r"\b(geplant|kommt noch|noch nicht gebaut)\b", README)
