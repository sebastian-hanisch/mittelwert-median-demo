"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zahlen oder Zellen ({"mean", "std"}), nicht Seeds."""

import math

import mtw_evaluation as ev


def cell(mean, std=0.0):
    return {"mean": mean, "std": std}


# ---- Formatierung ---------------------------------------------------------------------------------------------------------------------------

def test_number_formats_use_german_conventions():
    assert ev.de(1.2345, 2) == "1,23" and ev.pct(0.1234) == "12,3" and ev.pct(0.5, 0) == "50"
    assert ev.sci(3.6e-4) == "3,6·10⁻⁴" and ev.sci(8.9) == "8,9·10⁰" and ev.sci(1.02249e3, 3) == "1,02·10³" and ev.sci(0.0) == "0"
    assert ev.num(1234567.891, 2) == "1.234.567,89" and ev.num(5, 0) == "5"


# ---- Rechnen ---------------------------------------------------------------------------------------------------------------------------------

def test_verdict_rechnen_three_branches_and_thresholds():
    n = 10_000                                                                    # Standardfehler 0,01
    kind, text = ev.verdict_rechnen({"naiv": 4e-6, "Kahan": 1e-6, "Untergrenze": 1e-6}, n)
    assert kind == "success" and "4,0·10⁻⁶" in text and "1,0·10⁻⁶" in text      # genau das Vierfache der Untergrenze zählt noch als „reicht“
    kind, text = ev.verdict_rechnen({"naiv": 4.1e-6, "Kahan": 1e-6, "Untergrenze": 1e-6}, n)
    assert kind == "info" and "das 4-Fache der Untergrenze" in text and "1,0·10⁻²" in text
    kind, text = ev.verdict_rechnen({"naiv": 0.05, "Kahan": 1e-6, "Untergrenze": 1e-6}, n)
    assert kind == "warning" and "5,0 Standardfehler" in text and "5,0·10⁻²" in text and "Kahan: 1,0·10⁻⁶" in text
    assert ev.verdict_rechnen({"naiv": 0.01, "Kahan": 1e-6, "Untergrenze": 1e-6}, n)[0] == "warning"      # genau ein Standardfehler ist nicht mehr „weniger als einer“


def test_verdict_e1_naive_names_both_formats_and_the_threshold():
    kind, text = ev.verdict_e1_naive(cell(22.49, 4e-4), cell(4.4e-11), cell(1.8e-5), 10_000_000)
    assert kind == "warning" and "n = 10.000.000" in text and "2,2·10¹" in text and "4,0·10⁻⁴" in text and "71.119,6 Standardfehler" in text and "4,4·10⁻¹¹" in text
    assert ev.verdict_e1_naive(cell(1.8e-3), cell(1e-11), cell(1.8e-5), 1_000)[0] == "info"           # genau das Hundertfache der Untergrenze: noch info
    assert ev.verdict_e1_naive(cell(1.9e-3), cell(1e-11), cell(1.8e-5), 1_000)[0] == "warning"


def test_verdict_e1_mechanism_names_index_range_share_and_prediction_gap():
    cells = [{"first_index": 539_034, "share_1024": 1.0, "predicted_mean": 1022.4896, "measured_mean": 1022.4895},
             {"first_index": 539_038, "share_1024": 1.0, "predicted_mean": 1022.4900, "measured_mean": 1022.4899}]
    kind, text = ev.verdict_e1_mechanism(cells)
    assert kind == "info" and "Nr. 539.034 bis 539.038" in text and "In allen übrigen Schritten" in text and "1,0·10⁻⁴" in text and "2²⁹" in text
    cells[1]["share_1024"] = 0.99
    assert "In mindestens 99,00000 % der übrigen Schritte" in ev.verdict_e1_mechanism(cells)[1]


def test_verdict_e1_kahan_threshold_is_one_and_a_half_times_the_floor():
    args = (cell(3.0e-5), cell(3.0e-5), cell(3.1e-5), cell(3.2e-5), 10_000_000)
    kind, text = ev.verdict_e1_kahan(*args)
    assert kind == "success" and "Kahan: 3,0·10⁻⁵" in text and "Binärbaum 3,1·10⁻⁵" in text and "numpy.sum 3,2·10⁻⁵" in text and "3,2·10⁻⁴" in text and "erreicht die Untergrenze" in text
    assert ev.verdict_e1_kahan(cell(4.5e-5), cell(3.0e-5), cell(3.1e-5), cell(3.2e-5), 10_000_000)[0] == "success"          # genau das 1,5-Fache
    kind, text = ev.verdict_e1_kahan(cell(4.6e-5), cell(3.0e-5), cell(3.1e-5), cell(3.2e-5), 10_000_000)
    assert kind == "info" and "liegt über der Untergrenze" in text


def test_verdict_e1_neumaier_threshold_and_diagnosis_numbers():
    diag = {"n": 1_000_000, "kahan_max_rest": 32.0, "neumaier_max_rest": 8.9e6, "neumaier_sum_error": 123.0, "neumaier_wide_sum_error": 0.0}
    kind, text = ev.verdict_e1_neumaier(cell(1.3e-3), cell(3.0e-5), 10_000_000, diag)
    assert kind == "warning" and "schlechter als Kahan" in text and "1,3·10⁻³" in text and "n = 1.000.000" in text and "8,9·10⁶" in text and "3,2·10¹" in text and "1,2·10²" in text and "float64 0" in text
    assert ev.verdict_e1_neumaier(cell(6.0e-5), cell(3.0e-5), 10, None)[0] == "info"                     # genau das Doppelte ist noch nicht „schlechter“
    assert ev.verdict_e1_neumaier(cell(6.1e-5), cell(3.0e-5), 10, None)[0] == "warning"


def test_verdict_e1_neumaier_share_counts():
    row = {"n": 2_000_000, "series": 20, "worse": 19, "equal": 0, "better": 1, "median_kahan": 1.4e-5, "median_neumaier": 1.8e-4, "max_ratio": 99.0}
    kind, text = ev.verdict_e1_neumaier_share(row)
    assert kind == "warning" and "n = 2.000.000" in text and "19 von 20" in text and "in 0 gleich genau und in 1 genauer" in text and "1,8·10⁻⁴ gegen 1,4·10⁻⁵" in text
    assert ev.verdict_e1_neumaier_share({**row, "worse": 10})[0] == "info"                                  # genau die Hälfte ist nicht „mehr als die Hälfte“


def test_verdict_e1_welford_threshold_is_five_times_pairwise():
    kind, text = ev.verdict_e1_welford(cell(3.4e-3), cell(3.0e-5), cell(4.9e-11), cell(4.4e-11), 10_000_000)
    assert kind == "warning" and "3,4·10⁻³ in float32 gegen 3,0·10⁻⁵" in text and "4,9·10⁻¹¹ gegen 4,4·10⁻¹¹" in text
    assert ev.verdict_e1_welford(cell(1.5e-4), cell(3.0e-5), cell(1.0), cell(1.0), 10)[0] == "info"        # genau das Fünffache
    assert ev.verdict_e1_welford(cell(1.6e-4), cell(3.0e-5), cell(1.0), cell(1.0), 10)[0] == "warning"


def test_verdict_e2_ratio_text():
    kind, text = ev.verdict_e2_ratio(cell(1.93), cell(2.5e-9), cell(0.21), cell(0.25), cell(3.9), cell(3.9))
    assert kind == "warning" and "1,93 Standardfehler" in text and "2,5·10⁻⁹" in text and "0,21 Standardfehler erst bei μ/σ = 10¹⁰" in text and "0,25 Standardfehler, bei 10⁶ aber nur noch 3,9" in text and "Untergrenze 3,9 Standardfehler" in text


def test_verdict_e2_accumulator_and_order_text():
    c = {"mu": 1e6, "n": 1_000_000, "float32": cell(62.2), "float64_accumulator_rounded": cell(0.67), "floor": cell(0.67)}
    kind, text = ev.verdict_e2_accumulator(c)
    assert kind == "success" and "μ = 1.000.000 und n = 1.000.000" in text and "62,2 Standardfehler" in text and "um 0,67" in text and "bei 0,67" in text
    kind, text = ev.verdict_e2_order({"zufällig": 8900.0, "aufsteigend": 7700.0, "absteigend": 10100.0}, cell(2620.0), cell(386.0), cell(0.0134))
    assert kind == "info" and "7.700 bis 10.100 Standardfehler" in text and "24 %" in text and "2.620 Standardfehler" in text and "386" in text and "0,013" in text


# ---- E3 ---------------------------------------------------------------------------------------------------------------------------------

def test_verdict_e3_texts():
    kind, text = ev.verdict_e3_influence(1.267, 1e6, math.sqrt(math.pi / 2), 101)
    assert kind == "info" and "n = 101" in text and "1,267" in text and "Grenzwert 1,253" in text and "1.000.000 bei t = 1 000 000" in text
    kind, text = ev.verdict_e3_breakdown([2, 2, 3], [3, 4, 5])
    assert kind == "success" and "n = 3: k = 2, n = 4: k = 2, n = 5: k = 3" in text and "⌈n/2⌉" in text
    assert ev.verdict_e3_breakdown([2, 3, 3], [3, 4, 5])[0] == "warning"                                  # n = 4 braucht 2, nicht 3
    kind, text = ev.verdict_e3_contamination(0.139, 0.962, 2.059, 51.126, 40.0)
    assert kind == "info" and "0,139 (10 %)" in text and "0,962 (40 %)" in text and "2,059 (49 %)" in text and "51,1" in text and "um 40,0" in text
    kind, text = ev.verdict_e3_bias(2.0, 0.85, 0.85, 0.8)
    assert kind == "success" and "θ = 2,0" in text and "0,850 (Formel 0,850)" in text and "gewinnt" in text and "0,80" in text
    kind, text = ev.verdict_e3_bias(5.0, 1.0, 1.0, 0.96)
    assert kind == "info" and "verliert" in text                                                           # genau 1: kein Gewinn


# ---- E5, E6, E7, E8 ------------------------------------------------------------------------------------------------------------------------

def test_verdict_e5_threshold_and_zero_case():
    kind, text = ev.verdict_e5(0.2, 10, 0.0, 0.2, 0.125)
    assert kind == "success" and "d = 10, ‖μ‖ = 0,0" in text and "0,200 (exakt 0,200)" in text and "Positive Part 0,125" in text and "2/d = 0,200" in text
    kind, text = ev.verdict_e5(0.95, 10, 20.0, 0.95, 0.95)
    assert kind == "info" and "2/d" not in text
    assert ev.verdict_e5(0.949, 10, 5.0, 0.949, 0.9)[0] == "success"


def test_verdict_e5_coordinate_threshold():
    kind, text = ev.verdict_e5_coordinate(12.76, 50, 8.0)
    assert kind == "warning" and "d = 50" in text and "12,76-Fache" in text and "‖μ‖ = 8" in text
    assert ev.verdict_e5_coordinate(1.2, 3, 3.0)[0] == "info" and ev.verdict_e5_coordinate(1.21, 3, 3.0)[0] == "warning"


def test_verdict_e6_texts():
    kind, text = ev.verdict_e6_gap(1.0, 1.649, 0.691, 0.393, 0.389)
    assert kind == "info" and "σ = 1,0" in text and "1,65-Fache" in text and "69,1 %" in text and "38,9 %" in text and "39,3 % aus" in text
    kind, text = ev.verdict_e6_targets(1.0, 0.062, 0.393, 0.641, 0.059, 200)
    assert kind == "warning" and "n = 200" in text and "6,2 %" in text and "64,1 %" in text and "5,9 %" in text and "39,3 %" in text
    kind, text = ev.verdict_e6_delivery(0.8, 0.273, 0.274, 0.09, 0.987, 1000, 50)
    assert kind == "warning" and "„1.000 × Median“ lässt 27,3 %" in text and "Aus 50 Aufträgen" in text
    assert "98,7 %" in text and "27,4 % gegen 9,0 %" in text and "frei gewählt" in text


def test_verdict_e7_texts_and_threshold():
    kind, text = ev.verdict_e7_ties(0.436, 0.079, 0.485, 100, 0.4966)
    assert kind == "warning" and "F(0) = 0,4966" in text and "n = 100" in text and "43,6 %" in text and "48,5 %" in text and "7,9 %" in text
    kind, text = ev.verdict_e7_rounding(1.0, 0.103, 0.334, 0.946, 0.3)
    assert kind == "warning" and "h = 1,00" in text and "0,103" in text and "0,334" in text and "94,6 %" in text and "μ = 0,3" in text
    assert ev.verdict_e7_rounding(0.25, 0.1, 0.15, 0.08, 0.3)[0] == "info"                                  # genau das 1,5-Fache
    assert ev.verdict_e7_rounding(0.25, 0.1, 0.151, 0.08, 0.3)[0] == "warning"


def test_verdict_e8_texts_and_unusable_cells():
    kind, text = ev.verdict_e8_gap("Normal", 0.8, -0.217, 0.001, True)
    assert kind == "info" and "Normal, τ = 0,80" in text and "0,217 Quartilsabstände unter dem Quantil" in text
    assert "über dem Quantil" in ev.verdict_e8_gap("Lognormal(0; 1,5)", 0.8, 1.259, 0.012, True)[1]
    kind, text = ev.verdict_e8_gap("Lognormal(0; 1,5)", 0.99, -0.026, 0.083, False)
    assert kind == "info" and "zu unsicher" in text and "0,083" in text
    kind, text = ev.verdict_e8_cover(0.5, 0.8, 0.5987, 0.7863, 1.5232, 1.487, 1.1331)
    assert kind == "warning" and "Lognormal(0; 0,5), τ = 0,8" in text and "Quantil 1,523" in text and "Expektile 1,487" in text and "Mittelwert 1,133" in text and "deckt 60 % der Nachfrage" in text and "Expektile 79 %" in text and "gewünscht war 80 %" in text


# ---- Wählen: Stichprobe und Ebene --------------------------------------------------------------------------------------------------------------

def test_verdict_sample_thresholds_and_plan_sentence():
    kind, text = ev.verdict_sample(10.0, 9.95, 1.0, 100, None)
    assert kind == "success" and "Abstand beträgt 0,05 Quartilsabstände" in text and "Plan" not in text
    assert ev.verdict_sample(10.5, 10.0, 5.0, 100, None)[0] == "info"                                        # genau 0,1 Quartilsabstände
    kind, text = ev.verdict_sample(10.49, 10.0, 1.0, 100, 0.273)
    assert kind == "info" and "„100 × Median“ lässt 27,3 % der wahren Summe aus" in text
    kind, text = ev.verdict_sample(10.5, 10.0, 1.0, 100, -0.4)
    assert kind == "warning" and "überschätzt die wahre Summe um 40,0 %" in text
    assert ev.verdict_sample(9.0, 10.0, 1.0, 100, None)[0] == "warning"                                       # Betrag des Abstands zählt
    kind, text = ev.verdict_sample(0.7, 1.0, 0.0, 100, None)
    assert kind == "info" and "Quartilsabstand ist null" in text


def test_plan_missing_by_hand():
    assert ev.plan_missing([1.0, 1.0, 4.0]) == (6.0 - 3 * 1.0) / 6.0
    assert ev.plan_missing([0.0, 0.0, 0.0]) is None and ev.plan_missing([-1.0, 2.0, 3.0]) is None
    assert ev.plan_missing([2.0, 2.0]) == 0.0


def test_verdict_plane_thresholds():
    kind, text = ev.verdict_plane(0.5, 0.03, 10.0, 20)
    assert kind == "info" and "Abstand 10,0 km" in text and "0,50 km" in text and "0,03 km" in text and "geteilt durch 21" in text
    assert ev.verdict_plane(1.01, 0.0, 22.0, 20)[0] == "warning" and ev.verdict_plane(1.0, 0.0, 21.0, 20)[0] == "info"
    kind, text = ev.verdict_plane(0.0, 0.0, 0.0, 20)
    assert kind == "info" and "Ohne Fehlbuchung" in text
