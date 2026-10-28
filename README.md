# Mittelwert und Median: Welche Zahl will ich? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-mittelwert-median-demo.streamlit.app/)** -->

Wurzel der Reihe **Mittelwerte und Lagemaße** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Kanten im Abhängigkeitsgraphen: Wurzel → Stück 4 (Trimmen, Winsorisieren, Huber: „ein Punkt zieht das Mittel beliebig weit, der Median verliert Effizienz“), Wurzel → Stück 3 (Lp-Mitten: dieselbe Frage mit anderem Strafmaß), Wurzel → Stück 7 (Raten und Gewichte: das Mittel passt nicht zu Quotienten). Die Wurzel trägt zwei Fragen in zwei Reitern: **🧮 Rechnen** (wie genau liefert der Rechner das Mittel?) und **🎯 Wählen** (welche Zahl will ich, und was kostet es, eine andere zu schätzen?).

Vehikel: **Rechnen:** bis zu fünf Millionen Zeitstempel (Sekunden) oder Kundenkoordinaten (Meter), erzeugt als μ + σ·Z in float32 oder float64; gemessen wird gegen das **exakte Mittel der gespeicherten Werte** (Bruchrechnung), nicht gegen μ. **Wählen:** Lieferzeiten in Tagen (lognormal, Median 4 Tage; ein frei gewähltes Szenario, keine Messung) und weitere zehn Verteilungen im Reiter; 20 Kunden in der Ebene mit einem falsch gebuchten Kunden; d Lager mit je einem Nachfragemittel für James-Stein; Lager- und Fehlmengenkosten nur als Quantil bzw. Expektile des Bestands.

Abgrenzung: `kmeans-demo` (Mittelwert gegen Medoid bei Ausreißern), `exponential-smoothing-demo` (festes Gedächtnis statt Mittel aller Werte; das laufende Mittel nach Welford ist die Glättung mit α = 1/k), `rangintervalle-demo` (Schrumpfung bei Bradley-Terry-Stärken; die James-Stein-Formel steht dort unter „Bewusst nicht umgesetzt“), `forecast-combination-demo` (Median als Kombinationsregel), `gradient-boosting-demo` (absoluter Verlust), `verweilzeit-demo` („Median oder Quantil?“), `newsvendor-evpi-demo` und `forecast-inventory-demo` (Bestand als Quantil, „Bestand = Prognose“ deckt zu wenig). Im Abgleich der READMEs zum Planungsstand (2026-10-24) fand sich keine Demo, die Summationsverfahren, Zahlenformate als Thema oder Expektilen behandelt.

## Warum dieses Problem

Der Lehrbuchkern (das Mittel minimiert die Quadratsumme, der Median die Betragssumme) ist nur der Einstieg. Fünf gemessene Befunde tragen die Demo:

1. **Die fortlaufende float32-Summe von 10⁷ Werten um 1 000 verfehlt das Mittel um 22,49** (vier Zufalls-Seeds, Streuung zwischen den Seeds 2·10⁻⁴), das sind 71.118 Standardfehler. Ursache: Die Summe erreicht 2²⁹ beim Wert Nr. 539.039 bis 539.040 (die Nummern der vier Seeds; je Reihe liegt sie um wenige Stellen anders); von da an ist der Abstand der float32-Zahlen 64, und bei Werten N(1 000; 1) wird jeder Summand auf 1 024 gerundet. **Kahan erreicht in allen Zellen außer μ = 0 die Untergrenze des Zahlenformats** (3,0·10⁻⁵ bei 10⁷, Mittel über die vier Seeds); bei μ = 0 liegt er höchstens beim Dreifachen darüber.
2. **Neumaier, die übliche Verbesserung von Kahan, ist in float32 schlechter als Kahan:** 1,3·10⁻³ gegen 3,0·10⁻⁵ bei 10⁷ Werten (Mittel über vier Seeds; die einzelnen Reihen streuen stark), und in 13 / 19 / 18 von 20 weiteren Reihen (n = 10⁶ / 2·10⁶ / 5·10⁶) mehr als doppelt so ungenau, in keiner genauer. Der Befund ist für float32 und Werte um 1 000 mit Streuung 1 gemessen; er ist nicht auf Werte um 1 000 beschränkt: Bei μ/σ = 10⁴ (n = 10⁶) liegt Neumaier bei 0,47 Standardfehlern gegen 0,25 bei Kahan, bei μ/σ ≤ 100 sind beide gleichauf.
3. **James-Stein senkt das Gesamtrisiko bei ‖μ‖ = 0 auf 2/d, vervielfacht aber das Risiko eines einzelnen Lagers bis auf das 12,76-Fache** (d = 50, ‖μ‖ = 8; größter Wert im Raster der Normen 0 bis 20, ein Lager trägt das ganze μ; zwischen den Rasterpunkten kann er etwas höher liegen).
4. **Der Median zielt bei Schiefe auf eine andere Größe, nicht nur mit anderer Streuung:** Der Plan „Aufträge × Median“ lässt im gewählten Lieferzeit-Szenario 7,7 / 27,3 / 51,1 % der Monatssumme aus (σ = 0,4 / 0,8 / 1,2).
5. **Quantil und Expektile gleicher Stufe liegen nur an Kreuzungen oder bei Stufe 0,5 und symmetrischer Verteilung nah beieinander:** Bei τ = 0,5 ist die Lücke bei Normal, Gleichverteilung und t(3) von null nicht zu unterscheiden; schon bei τ = 0,8 liegt sie bei −0,217 (Normal) und −0,266 (Gleichverteilung), und über alle Verteilungen reicht sie bei τ = 0,8 von −0,266 (Gleichverteilung) bis +1,259 (Lognormal(0; 1,5)) Quartilsabstände.

Kernfrage: *Welche Zahl will ich, wie genau rechnet der Rechner sie, und was kostet es, eine andere zu schätzen?*

## Befunde und Korrekturen gegenüber dem Plan

**Umfang (Nutzerentscheidung 2026-10-24):** Kern sind E1, E2 (Reiter Rechnen) sowie E5, E6, E8 (Reiter Wählen); E3 und E7 sind kurze Tabs. E4 (Preis des Medians nach Verteilung) und E9 (Kosten der falschen Wahl) gibt es nicht; damit entfallen die Hypothesen H13 und H23. Die Nummern der Experimente und Hypothesen bleiben die der beiden Einzelpläne (`mittelwerte-planung/STUECK-1-mittelwert-median.md`). Alle Zahlen wurden im Bau mit dem **eigenen SplitMix64-Strom** neu gemessen; die Wegwerf-Vormessungen der Pläne (numpy-Zufall) liefern keine README-Zahl. Vier feste Zufalls-Seeds je Zelle, Mittel mit Streuung zwischen den Seeds.

| | Hypothese | Ergebnis | Zahl (diese Demo) |
|---|---|---|---|
| H1 | Die fortlaufende float32-Summe von 10⁷ Werten N(1 000; 1) verfehlt das Mittel um 22,49; ab Summe 2²⁹ wächst sie um genau 1 024 je Schritt | **Ja** | Fehler 22,49; die Summe erreicht 2²⁹ bei Wert Nr. 539.039 bis 539.040; in mindestens 99,99998 % der übrigen Schritte Zuwachs genau 1 024; Vorhersage und Messung des Endmittels unterscheiden sich um höchstens 3,8·10⁻⁵ |
| H2 | Paarweise Summe bleibt bei 10⁷ unter einem Drittel Standardfehler | **Ja** (`numpy.sum` nur als Band: es hängt von der numpy-Version ab) | Binärbaum 3,0·10⁻⁵ = 0,10 Standardfehler, numpy.sum 3,2·10⁻⁵ = 0,10 Standardfehler (Standardfehler 3,2·10⁻⁴) |
| H3 | Kahan erreicht in float32 die Untergrenze bei jedem n und jeder Reihenfolge | **Ja** | Kahan = Untergrenze der Summe in allen fünf Größen, in beiden Formaten, in allen sechs Reihenfolge-Zellen und bei jedem μ/σ ab 1 (n = 10⁶); bei μ = 0 liegt Kahan höchstens das Dreifache über der Untergrenze, beide unter 10⁻⁷ Standardfehlern |
| H4 | Neumaier ist in float32 mit N(1 000; 1) schlechter als Kahan, die Ursache ist der Rest | **Ja, gemessen an float32 und dieser Datenform; ähnlich bei μ/σ = 10⁴** | n = 10⁶: 7,1·10⁻⁵ gegen 1,8·10⁻⁵; n = 10⁷: 1,3·10⁻³ (4,1 Standardfehler). Diagnose an einer Reihe (n = 1.000.000): größter Rest 8,9·10⁶ (Kahan 32), Fehler der Summe 104,39, mit dem Rest in float64 0. Gegen 20 weitere Reihen: Neumaier mehr als doppelt so ungenau wie Kahan in 13 von 20 (n = 10⁶), 19 von 20 (2·10⁶), 18 von 20 (5·10⁶); nie genauer |
| H5 | Welford ist kein Genauigkeitsverfahren fürs Mittel | **Ja** (Varianzteil nicht untersucht) | float32, n = 10⁷: 3,4·10⁻³ = 10,9 Standardfehler gegen 0,10 beim Binärbaum; aufsteigend sortiert (n = 10⁶) 2.623 Standardfehler; float64: 4,9·10⁻¹¹ gegen 4,4·10⁻¹¹ bei der fortlaufenden Summe |
| H6 | Auslöschung: float32 naiv ist bei μ/σ = 100 (n = 10⁶) schon falsch, float64 erst bei 10¹⁰ | **Ja** | float32 1,93 Standardfehler, float64 2,5·10⁻⁹; float64 erreicht 0,21 Standardfehler bei μ/σ = 10¹⁰ (Mittel über vier Seeds; die einzelnen Reihen streuen stark, das Mittel von 1,93 sagt nicht, dass jede Reihe über einem Standardfehler liegt) |
| H7 | Ab μ/σ = 10⁶ begrenzt das Format der Summe jedes Verfahren; Ausweg: float64-Akkumulator | **Ja** | Untergrenze der Summe 3,88 Standardfehler bei 10⁶ und 172,3 bei 10⁷; `numpy.mean` in float32 bei μ = 10⁶, n = 10⁶: 62,2 Standardfehler, mit float64-Akkumulator 0,67 (Untergrenze 0,67) |
| H8 | Die Reihenfolge ändert den naiven Fehler um etwa ±15 %; ruinös nur bei Welford und bei einer Verschiebung um einen untypischen ersten Wert | **Ja**, mit etwa ±13 % | Werte N(1 000; 1), naiv (n = 10⁶, Standardfehler): 8.898 zufällig, 7.715 aufsteigend, 10.064 absteigend; Welford aufsteigend 2.623; Verschiebung um x₁ bei absteigender schiefer Reihe 392; Kahan 0,009 in allen drei Anordnungen |
| H9 | Einflusskurve: Mittel linear und unbeschränkt, Median beschränkt (Plateau) | **Ja** | Median-Plateau 1,267 (n = 101, Streuung zwischen den Seeds 0,004) und 1,271 (n = 100), Grenzwert 1,253; bei t = 0,25 erst 1,208 |
| H10 | Ausreißer verschieben das Mittel um ε·M, den Median kaum; ab 50 % springt er | **Ja** | Median-Verschiebung bei 5 / 10 / 20 / 40 / 49 %: 0,065 / 0,139 / 0,318 / 0,962 / 2,059; bei 50 % 51,13, ab 55 % 100,0; das Mittel bei 40 % um 40,0 |
| H11 | Durchschlagspunkt des Medians ⌈n/2⌉, des Mittels 1; davor ist die Verschiebung beschränkt | **Ja** | Aufzählung für n = 3 bis 12 in allen vier Seeds ⌈n/2⌉, beim Mittel 1; größte Verschiebung davor im Mittel über die Seeds 1,09 bis 2,01 Streuungen (je Seed 0,38 bis 3,06) |
| H12 | Effizienz des Medians bei Normalverteilung fällt von 0,743 auf 0,637; gerades n liegt höher | **Ja** (exakte Quadratur) | n = 3 / 5 / 9 / 21 / 101 / 1 001: 0,7429 / 0,6973 / 0,6689 / 0,6500 / 0,6393 / 0,6369; gerades n = 4 / 10 / 100: 0,8384 / 0,7229 / 0,6457; Grenzwert 2/π = 0,6366 |
| H14 | Ein um 10 % verkleinertes Mittel hat bei n = 10 und θ ≤ 4 den kleineren MSE; das beste c enthält das unbekannte μ | **Ja** | MSE des Mittels = 1; c = 0,9 bei θ = 0 / 1 / 2 / 4: 0,81 / 0,82 / 0,85 / 0,97 (Formel; Simulation auf 0,003 gleich); gewinnt für θ unter 4,36; c = 0,8 bei θ = 4: 1,28 |
| H15 | James-Stein / Einzelmittel = 2/d bei ‖μ‖ = 0, nähert sich 1, nie darüber | **Ja** (die Simulation liegt höchstens bei 1,0003: Rauschen, exakt ist das Risiko nie höher) | d = 3 / 5 / 10 / 50 bei ‖μ‖ = 0: 0,674 / 0,403 / 0,200 / 0,040 (2/d = 0,667 / 0,400 / 0,200 / 0,040); Positive Part 0,532 / 0,280 / 0,125 / 0,022; bei ‖μ‖ = 20 noch 1,000 / 0,993 / 0,985 / 0,897 |
| H16 | Das Gesamtrisiko sinkt, das Risiko einzelner Lager steigt bis auf ein Vielfaches | **Ja** | größtes Risiko des ersten Lagers (Einzelmittel 1) mit Positive Part: 1,21 (d = 3) / 1,67 (5) / 2,73 (10) bei ‖μ‖ = 3 und 12,76 (d = 50, ‖μ‖ = 8); beim reinen James-Stein-Schätzer 1,22 / 1,71 / 2,83 und 12,76; der Positive-Part-Faktor steht bei ‖μ‖ = 0 in 19,8 / 29,9 / 37,2 / 44,7 % der Läufe auf 0 |
| H17 | Mittelwert/Median = e^(σ²/2); der Median als Schätzer des Mittelwerts lässt den Anteil 1 − e^(−σ²/2) aus (der Populationsmedian; der Stichproben-Median erreicht ihn für wachsendes n) | **Ja** | Mittelwert/Median 1,13 / 1,65 / 3,08 / 7,39 (σ = 0,5 / 1 / 1,5 / 2); typischer Fehler des Medians gegen den Mittelwert bei n = 200: 0,393 (σ = 1; Grenzwert 0,3935) und 0,865 (σ = 2; 0,8647) |
| H18 | Der Plan „Aufträge × Median“ lässt 7,7 / 27 / 51 % der Monatssumme aus | **Ja** (Szenario frei gewählt) | σ = 0,4 / 0,8 / 1,2: 7,7 / 27,3 / 51,1 % (exakt 7,7 / 27,4 / 51,3 %); „× Mittelwert“ weicht im Mittel um 1,0 / 2,4 / 4,5 % ab; aus 50 Aufträgen ist „× Stichproben-Median“ in 86,5 / 98,7 / 100,0 % der Fälle zu klein |
| H19 | Ties: Der Median ganzer Zahlen ist bei geradem n oft nicht ganzzahlig; ein Fast-Gleichstand lässt ihn springen | **Ja** | Poisson λ = 0,7 / 2 / 10: bei n = 10 in 25,6 / 31,9 / 44,9 % nicht ganzzahlig, bei n = 100 in 7,9 / 1,3 / 7,5 %; Poisson(0,7), n = 100: Median 0 in 43,6 %, 0,5 in 7,9 %, 1 in 48,5 % (F(0) = 0,4966); n = 1 000: 40,6 % bei 0; n = 10 000: 23,6 % |
| H20 | Gerundete Daten: Der Median bleibt am Raster hängen | **Ja** | Normal(0,3; 1) auf h = 1 gerundet, n = 101: RMSE des Medians 0,334 gegen 0,103 beim Mittel (ungerundet 0,124 gegen 0,099); der Median steht in 94,6 % der Läufe auf 0 |
| H21 | Quantil und Expektile gleicher Stufe liegen nur an Kreuzungen oder bei Stufe 0,5 und symmetrischer Verteilung nah beieinander | **Ja** | Lücke (Expektile − Quantil) / Quartilsabstand bei τ = 0,8: Normal −0,217, Exponential −0,005, Lognormal(0; 1) +0,324, Lognormal(0; 1,5) +1,259, Pareto(2,5) +0,574; Exponential: +0,279 bei τ = 0,5, −0,897 bei τ = 0,99 |
| H22 | Ein Mittel-Bestand deckt bei kritischem Quotienten 0,8 nur 60 % (Lognormal(0; 0,5)) | **Ja**, nur als Verweis auf `forecast-inventory-demo` | Mittelwert 1,1331 deckt 59,9 %; Quantil 1,5232, Expektile 1,4870 (deckt 78,6 %) |

H13 (Effizienz nach Verteilung, Cauchy) und H23 (Kosten der falschen Wahl) entfallen mit E4 und E9.

### Korrekturen gegenüber dem Plan

- **Einflussplateau des Medians (offener Punkt des Plans):** Die beiden Quellpläne nannten 1,305 (Basis 99 Werte plus ein Punkt) und 1,2596 (Basis 100 plus ein Punkt). Beide waren Simulationswerte mit wenigen Stichproben. Festgelegt ist hier **SC(t) = n·(T(x₁, …, x_{n−1}, t) − T(x₁, …, x_{n−1}))** bei insgesamt n Werten und 20.000 Stichproben je Seed; die Plateaus 1,267 (n = 101) und 1,271 (n = 100) liegen 0,014 bis 0,018 über dem Grenzwert 1,253; das ist mehr als die Streuung zwischen den Seeds, also ein Effekt des endlichen n (nicht weiter untersucht). Der Wert 1,305 liegt weit über beiden Messungen und ist vermutlich Rauschen der kleinen Stichprobe (2 000 Datensätze je Seed); erklärt ist er nicht.
- **Median-Verschiebung bei Ausreißern** gilt das Maß des zweiten Quellplans (mittlere Verschiebung |T − T_sauber| gegen die unverschmutzte Stichprobe); der erste Quellplan enthielt das Rauschen der Daten mit.
- **Neumaier:** Der Plan erwartete Neumaier mindestens so gut wie Kahan. Gemessen ist das Gegenteil für float32 und Werte um 1 000, und es gilt nicht für jede Reihe (bei n = 10⁶ in 5 von 20 Reihen gleich). Bei μ/σ = 10⁴ (n = 10⁶) liegt Neumaier schon bei 0,47 Standardfehlern gegen 0,25 bei Kahan; die Vormessung des Plans hatte dort noch Gleichstand.
- **Paarweise Summe:** Der Plan nannte 1,7·10⁻⁵ (eine Instanz); über vier Seeds liegen Binärbaum und `numpy.sum` bei 3,0·10⁻⁵ und 3,2·10⁻⁵, die Untergrenze bei 3,0·10⁻⁵.
- **Q95 des Mittels bei σ = 2** (n = 200): Der Plan nannte 0,755 (zu wenige Läufe). Mit 4 × 5 000 Läufen: 0,720 (Streuung zwischen den Seeds 0,015).
- **Durchschlag:** Die größte Verschiebung vor dem Durchschlag lag im Plan bei 0,73 bis 1,75 Streuungen; hier im Mittel über die Seeds 1,09 bis 2,01 (je Seed 0,38 bis 3,06). Die Spanne hängt von der Stichprobe ab, ein fester Wert ist es nicht.
- **Abweichungen der Umsetzung:** (1) Die schiefe Reihe im Reiter Rechnen ist lognormal(0; 1), auf Mittel μ und Streuung 1 gebracht (nicht die rohe lognormal(0; 1) der Vormessung); die Zahlen der Reihenfolge-Tabelle für diese Reihe sind daher nicht die des Plans. (2) Der Regler „Verfahren“ des Plans entfällt: Es werden immer alle neun Varianten gerechnet und verglichen. (3) Live reicht der Reiter Rechnen bis n = 5·10⁶; die Reihe bis 10⁷ ist vorgerechnet. (4) Ein zusätzliches Preset „Zehn Prozent Ausreißer“ und die Regler „Lage verschieben“ und „Abstand der Fehlbuchung“. (5) Die Mitte der Kunden in der Ebene ist der Schwerpunkt gegen den **Median je Achse**; der geometrische Median ist Stück 5 der Reihe. (6) Der Expander „Die vier Verlustkurven“ und die Karte mit Fehlbuchung stehen zusätzlich zu den Experimenten.

## Modell

- **Reihe (Reiter Rechnen).** n Werte μ + σ·Z mit σ = 1 (Normal) bzw. lognormal(0; 1) auf Mittel μ und Streuung 1 gebracht, im Zahlenformat gespeichert (float32 oder float64), dann zufällig, aufsteigend oder absteigend angeordnet. Der Zufall kommt aus einem eigenen SplitMix64-Strom (`mtw_rng.py` ist die wörtliche Kopie aus `reduktionskette-demo/red_scenario.py`, geprüft von `tests/test_copies.py`); Normalwerte nach Box-Muller. Ein SplitMix64-Wert hängt nur vom Seed und von seiner Nummer ab, deshalb wird ein ganzer Block mit numpy auf einmal berechnet und Wert für Wert gegen die Klasse geprüft.
- **Verfahren.** fortlaufende Summe, `numpy.sum`, Binärbaum (paarweise), verschobene Summe (x₁ abziehen), Kahan (Ergebnis s, „mit Rest“ s − c), Neumaier (Ergebnis s + c, Rest im Zahlenformat, „mit Rest“ in float64), Welford (nur der Mittelwertteil). Die Schleifen sind IEEE-genau und bitgleich zu einer im Test geschriebenen Skalarschleife.
- **Fehler.** Ergebnis minus **exaktes Mittel der gespeicherten Werte** (`math.fsum` mit Rest, als Bruch, geprüft gegen `fractions.Fraction`). Summen werden exakt durch n geteilt, damit nur das Verfahren gemessen wird. Die **Untergrenze** ist das exakte Ergebnis (Summe bzw. Mittel) auf das Zahlenformat gerundet: Besser kann ein Verfahren, das sein Ergebnis im Format ausgibt, nicht werden.
- **Stichprobe (Reiter Wählen).** Zehn Verteilungen in Standardform (Lognormal als Lieferzeit mit Median 4 Tagen); Lage, Ausreißer (die ersten k Werte werden durch den Ausreißerabstand ersetzt) und Rundung auf ein Raster h. Mitten: Mittel, Median, Quantil (Pinball-Verlust, bei ganzzahligem n·τ ein Intervall), Expektile (asymmetrisch gewichtete Quadratsumme).
- **James-Stein.** d Lager mit Beobachtung X ~ N(μ, I_d), Streuung 1 bekannt; Einzelmittel gegen (1 − (d − 2)/‖X‖²)·X, Positive Part und Schrumpfung zum gemeinsamen Mittel; das exakte Risiko als Poisson-Mischung (`scipy.stats.ncx2`-Quadratur als Gegenprobe).
- **Theorie.** Varianz des Medians bei Normalverteilung durch Quadratur der Dichte der mittleren Ordnungsstatistik (gerades n über die gemeinsame Dichte zweier benachbarter Ordnungsstatistiken), Lognormal-Quantile und -Expektilen aus Teilerwartungen über Φ. Mathematik und Annahmentabelle stehen in den Expandern **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung** der App. Kahan, Neumaier, Welford, James und Stein, Efron und Morris sowie Newey und Powell wurden **nicht eingesehen**; die Verfahren sind nach ihrer Beschreibung umgesetzt und gegen unabhängige Rechenwege geprüft.

## Methodik

| Baustein | Datei |
|---|---|
| Summationsverfahren, exaktes Mittel, Untergrenze, Rest-Diagnose | `mtw_sums.py` |
| Schwerpunkt, Median, Quantil, Expektile, Einflusskurve, Durchschlag | `mtw_estimators.py` |
| James-Stein: Schätzer, exaktes Risiko | `mtw_stein.py` |
| Varianz des Medians, Lognormal-Quantile und -Expektilen, Lücken | `mtw_theory.py` |
| Generator (SplitMix64-Kopie, Block, Box-Muller), Verteilungen, Reihen, Ebene, Lager | `mtw_rng.py`, `mtw_scenario.py` |
| Messreihen E1 bis E8 (Funktionen für die Live-Knöpfe und die Vorrechnung) | `mtw_experiments.py`, `tools/make_results.py` → `data/mtw_results.json` |

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stammen aus `data/mtw_results.json` und sind in `tests/test_claims.py` nachgerechnet.

**E1 Fehler des Mittels über n** (Werte N(1 000; 1) in zufälliger Reihenfolge, Betrag des Fehlers, Mittel über vier Seeds)

float32:

| Verfahren | n = 10³ | 10⁴ | 10⁵ | 10⁶ | 10⁷ |
|---|---|---|---|---|---|
| Untergrenze der Summe | 1,6·10⁻⁵ | 1,3·10⁻⁵ | 2,8·10⁻⁵ | 1,8·10⁻⁵ | 3,0·10⁻⁵ |
| Untergrenze des Mittels | 8,6·10⁻⁶ | 2,1·10⁻⁵ | 6,5·10⁻⁶ | 1,4·10⁻⁵ | 1,8·10⁻⁵ |
| naiv (fortlaufend) | 2,6·10⁻⁴ | 9,3·10⁻⁴ | 2,1·10⁻³ | 8,9·10⁰ | 2,2·10¹ |
| numpy.sum | 2,5·10⁻⁵ | 1,3·10⁻⁵ | 2,8·10⁻⁵ | 1,8·10⁻⁵ | 3,2·10⁻⁵ |
| Binärbaum | 2,5·10⁻⁵ | 1,3·10⁻⁵ | 2,8·10⁻⁵ | 1,8·10⁻⁵ | 3,0·10⁻⁵ |
| verschoben | 8,6·10⁻⁶ | 2,1·10⁻⁵ | 6,5·10⁻⁶ | 1,4·10⁻⁵ | 1,8·10⁻⁵ |
| Kahan | 1,6·10⁻⁵ | 1,3·10⁻⁵ | 2,8·10⁻⁵ | 1,8·10⁻⁵ | 3,0·10⁻⁵ |
| Kahan mit Rest | 0 | 0 | 0 | 6,6·10⁻⁹ | 7,1·10⁻⁹ |
| Neumaier | 1,6·10⁻⁵ | 1,3·10⁻⁵ | 2,8·10⁻⁵ | 7,1·10⁻⁵ | 1,3·10⁻³ |
| Welford | 1,4·10⁻⁴ | 8,8·10⁻⁴ | 1,6·10⁻³ | 2,9·10⁻³ | 3,4·10⁻³ |

float64:

| Verfahren | n = 10³ | 10⁴ | 10⁵ | 10⁶ | 10⁷ |
|---|---|---|---|---|---|
| Untergrenze der Summe | 1,5·10⁻¹⁴ | 3,7·10⁻¹⁴ | 4,5·10⁻¹⁴ | 2,3·10⁻¹⁴ | 3,9·10⁻¹⁴ |
| Untergrenze des Mittels | 4,2·10⁻¹⁴ | 2,5·10⁻¹⁴ | 2,4·10⁻¹⁴ | 1,1·10⁻¹⁴ | 2,5·10⁻¹⁴ |
| naiv (fortlaufend) | 5,7·10⁻¹³ | 1,4·10⁻¹² | 6,0·10⁻¹² | 1,4·10⁻¹¹ | 4,4·10⁻¹¹ |
| numpy.sum | 1,5·10⁻¹⁴ | 3,7·10⁻¹⁴ | 4,8·10⁻¹⁴ | 5,0·10⁻¹⁴ | 3,9·10⁻¹⁴ |
| Binärbaum | 1,5·10⁻¹⁴ | 3,8·10⁻¹⁴ | 4,5·10⁻¹⁴ | 2,3·10⁻¹⁴ | 9,2·10⁻¹⁴ |
| verschoben | 4,2·10⁻¹⁴ | 2,5·10⁻¹⁴ | 2,4·10⁻¹⁴ | 1,1·10⁻¹⁴ | 9,9·10⁻¹⁴ |
| Kahan | 1,5·10⁻¹⁴ | 3,7·10⁻¹⁴ | 4,5·10⁻¹⁴ | 2,3·10⁻¹⁴ | 3,9·10⁻¹⁴ |
| Kahan mit Rest | 1,5·10⁻¹⁴ | 3,7·10⁻¹⁴ | 4,5·10⁻¹⁴ | 2,3·10⁻¹⁴ | 3,9·10⁻¹⁴ |
| Neumaier | 1,5·10⁻¹⁴ | 3,7·10⁻¹⁴ | 4,5·10⁻¹⁴ | 2,3·10⁻¹⁴ | 3,9·10⁻¹⁴ |
| Welford | 4,2·10⁻¹³ | 3,0·10⁻¹² | 4,6·10⁻¹² | 2,4·10⁻¹¹ | 4,9·10⁻¹¹ |

Die Ursache der float32-Spalte bei 10⁶ und 10⁷ steht oben (Befund 1). Kahan mit Rest hält die Summe in float32 bis n = 10⁵ vollständig (Fehler 0), danach fast. Bei float64 liegen Kahan, Kahan mit Rest und Neumaier in allen fünf Größen bei der Untergrenze; `numpy.sum` (5,0·10⁻¹⁴ gegen 2,3·10⁻¹⁴ bei 10⁶), Binärbaum (9,2·10⁻¹⁴ gegen 3,9·10⁻¹⁴ bei 10⁷) und verschoben (9,9·10⁻¹⁴ gegen 2,5·10⁻¹⁴ bei 10⁷) liegen in einzelnen Spalten darüber, die fortlaufende Summe und Welford in allen.

**E2 Auslöschung** (n = 10⁶, float32, Fehler in Standardfehlern, ein Standardfehler = 10⁻³ bei n = 10⁶, Mittel über vier Seeds; bei μ/σ = 10⁶ und 10⁷ sind die Daten selbst grob gerundet)

| Verfahren | μ/σ = 0 | 1 | 100 | 1·10⁴ | 1·10⁶ | 1·10⁷ |
|---|---|---|---|---|---|---|
| Untergrenze der Summe | 4,1·10⁻⁸ | 1,5·10⁻⁵ | 1,8·10⁻³ | 2,5·10⁻¹ | 3,9·10⁰ | 1,7·10² |
| naiv | 1,3·10⁻⁵ | 4,7·10⁻³ | 1,9·10⁰ | 1,3·10⁵ | 3,6·10⁶ | 2,9·10⁷ |
| numpy.sum | 1,5·10⁻⁷ | 3,3·10⁻⁵ | 5,1·10⁻³ | 2,5·10⁻¹ | 6,9·10¹ | 8,8·10² |
| Kahan | 4,2·10⁻⁸ | 1,5·10⁻⁵ | 1,8·10⁻³ | 2,5·10⁻¹ | 3,9·10⁰ | 1,7·10² |
| Neumaier | 4,1·10⁻⁸ | 1,5·10⁻⁵ | 1,8·10⁻³ | 4,7·10⁻¹ | 2,1·10⁴ | 2,1·10⁵ |
| Welford | 1,1·10⁻⁵ | 9,5·10⁻³ | 7,6·10⁻¹ | 1,1·10¹ | 7,8·10¹ | 5,0·10² |
| verschoben | 2,0·10⁻² | 9,5·10⁻³ | 1,2·10⁻² | 2,4·10⁻¹ | 1,3·10⁰ | 1,2·10⁰ |

**E3 Verschiebung unter Verschmutzung** (n = 100, N(0; 1), Ausreißer bei 100, 4 × 4 000 Läufe)

| Ausreißeranteil | 5 % | 10 % | 20 % | 40 % | 49 % | 50 % | 55 % |
|---|---|---|---|---|---|---|---|
| Verschiebung des Mittels | 5,0 | 10,0 | 20,0 | 40,0 | 49,0 | 50,0 | 55,0 |
| Verschiebung des Medians | 0,065 | 0,139 | 0,318 | 0,962 | 2,059 | 51,13 | 100,00 |

**E5 James-Stein** (Risiko James-Stein / Risiko des Einzelmittels, Positive Part in Klammern; 4 × 50 000 Läufe, wahre Mittel alle auf ein Lager gelegt)

| ‖μ‖ | d = 3 | d = 5 | d = 10 | d = 50 |
|---|---|---|---|---|
| 0 | 0,674 (0,532) | 0,403 (0,280) | 0,200 (0,125) | 0,040 (0,022) |
| 2 | 0,890 (0,859) | 0,692 (0,649) | 0,438 (0,398) | 0,111 (0,097) |
| 5 | 0,987 (0,987) | 0,932 (0,932) | 0,796 (0,796) | 0,363 (0,362) |
| 8 | 0,992 (0,992) | 0,970 (0,970) | 0,910 (0,910) | 0,584 (0,584) |
| 20 | 1,000 (1,000) | 0,993 (0,993) | 0,985 (0,985) | 0,897 (0,897) |

**E6 Welche Größe will ich?** (Lognormal(0; σ); Anteil unter dem Mittelwert und Summenanteil der größten 10 %: Stichprobe mit 500.000 Werten je Seed; Fehler: n = 200, 4 × 5 000 Stichproben, typischer relativer Fehler)

| σ | Mittelwert / Median | Anteil unter dem Mittelwert | Summenanteil der größten 10 % | Fehler des Mittels → Mittelwert | Fehler des Medians → Mittelwert |
|---|---|---|---|---|---|
| 0,5 | 1,13 | 59,8 % | 21,7 % | 2,5 % | 11,7 % |
| 1,0 | 1,65 | 69,2 % | 38,9 % | 6,2 % | 39,3 % |
| 1,5 | 3,08 | 77,4 % | 58,7 % | 12,2 % | 67,5 % |
| 2,0 | 7,39 | 84,1 % | 76,3 % | 22,2 % | 86,5 % |

**E8 Quantil gegen Expektile** (Lücke (Expektile − Quantil) / Quartilsabstand; Stichproben mit 10⁶ Werten je Seed; „unsicher“: Streuung zwischen den Seeds mindestens die halbe Lücke)

| Verteilung | τ = 0,5 | 0,8 | 0,9 | 0,95 | 0,99 |
|---|---|---|---|---|---|
| Normal | unsicher | −0,217 | −0,312 | −0,375 | −0,452 |
| Gleichverteilung | unsicher | −0,266 | −0,300 | −0,273 | −0,163 |
| Exponential | +0,279 | −0,005 | −0,239 | −0,459 | −0,897 |
| Lognormal(0; 0,5) | +0,193 | −0,052 | −0,236 | −0,412 | −0,798 |
| Lognormal(0; 1) | +0,446 | +0,324 | +0,112 | −0,175 | −1,146 |
| Lognormal(0; 1,5) | +0,872 | +1,259 | +1,343 | +1,240 | unsicher |
| t(3) | unsicher | −0,123 | −0,207 | −0,302 | −0,604 |
| Pareto(2,5) | +0,562 | +0,574 | +0,477 | +0,306 | −0,386 |

## Ehrliche Grenzen

- **Simulierte Reihen und Stichproben.** Alle Fehler, Anteile und Risiken gelten für die Verteilungen und Regler dieser Demo; echte Zeitstempel und Kundenkoordinaten können anders aussehen.
- **Neumaier an einer Datenform.** Dass Neumaier in float32 schlechter als Kahan ist, ist für Werte um 1 000 mit Streuung 1 gemessen; die Ursache (der Rest wird selbst im Zahlenformat summiert) ist an einer Reihe diagnostiziert, nicht allgemein bewiesen. Die Literatur zur Fehlerschranke wurde nicht gelesen.
- **`numpy.sum` und die letzte Binärstelle von `log` und `cos`** (Box-Muller) können je numpy-Version und Plattform abweichen; deshalb stehen Zahlen mit zwei bis drei gültigen Stellen, und Tests lesen `numpy.sum` nur als Band. Die eigenen Schleifen sind bitgenau.
- **Nur das Mittel, keine Varianz:** Welford wird nur als Mittelwertverfahren gemessen; float16 und bfloat16 kommen nicht vor.
- **James-Stein nur bei bekannter Streuung** (Normalbeobachtungen, Schrumpfung zum Nullpunkt und zum gemeinsamen Mittel).
- **Lieferzeit-Szenario frei gewählt** (Median 4 Tage, 1 000 Aufträge im Monat, lognormal): ein Beispiel, keine Messung an Aufträgen.
- **Durchschlag nur aufgezählt** (n bis 12, ab n = 10 nur mit +M), Ausreißer nur in einer Richtung.
- **Vier Seeds je Zelle.** Die angegebene Streuung ist die zwischen diesen Seeds, kein Konfidenzintervall.
- **Rundungs-Effizienz** ab h = 0,75 Streuungen ist nicht ausgewiesen (die Varianz des Medians geht gegen null und das Verhältnis wird sinnlos).
- **Zelle Lognormal(0; 1,5), τ = 0,99** der Lücken-Tabelle ist unsicher (−0,026 ± 0,083) und wird nicht gedeutet.
- **Die Laufzeit** ist als Größenordnung gemessen (siehe „Rechengrenze“), nicht als Test festgeschrieben.
- **Mit echtem Mausklick nicht geprüft:** Die App wurde im Browser durchgeklickt (Presets, beide Reiter, die Knöpfe „Nachmessen“), nicht Zelle für Zelle.

## Rechengrenze

Live: Reiter Rechnen bis n = 5·10⁶ (alle neun Varianten; die Rechenzeit wächst linear mit n und steht nach jedem Lauf in der App), Stichprobe bis n = 10 000, James-Stein 50 000 Läufe mit d ≤ 50, Durchschlag-Aufzählung bis n = 12. Einmalmessung bei 5·10⁶ Werten am Entwicklungsrechner: 4,3 s in float32 und 2,3 s in float64; auf anderen Rechnern wird es anders sein. Vorgerechnet (`data/mtw_results.json`, Prüfsumme der Quelldateien): Fehler über n bis 10⁷ in beiden Formaten, μ/σ-Raster, Reihenfolgen, Neumaier-Diagnose und -Zählung, James-Stein-Raster, Effizienz bei Normalverteilung nach n, σ-Raster der Zielgröße, Ties, Rundung, Quantil-Expektile-Tabellen.

## Tests

- **Handrechnungen** (`tests/conftest.py`): float32 [2²⁴, 1, 1] (naiv 2²⁴, Kahan und Neumaier 2²⁴ + 2), zehnmal 0,1 in float64, Welford [1, 2, 3, 4], Punkte (0, 0), (4, 0), (0, 3), Einflusskurve von (1, 2, 3) mit z = 10, James-Stein d = 3, MSE von c·x̄ bei n = 3, Median und Minimierer-Intervall, Durchschlag bei n = 5, Pinball bei τ = 0,8, Expektile von (0, 4).
- **Orakel** (`tests/test_oracle_summation.py`, `test_oracle_median_quantil.py`, `test_oracle_stein_theorie.py`): Summationsverfahren gegen im Test geschriebene Skalarschleifen (Bit-Gleichheit, float32 und float64, beide Richtungen), `math.fsum` und `fractions.Fraction`; Median gegen `numpy.median`, Quantil gegen `numpy.quantile(method="inverted_cdf")`, Expektile und Schwerpunkt gegen `scipy.optimize`, Durchschlag gegen eine zweite Aufzählung; exaktes James-Stein-Risiko gegen `scipy.stats.ncx2`-Quadratur; Varianz des Medians gegen `scipy.integrate.quad` (ungerades n) und `dblquad` (gerades n); Lognormal-Quantile gegen `scipy.stats.lognorm`; Verteilungen gegen die Verteilungsfunktionen aus `scipy.stats`.
- **Zweig-Tests** (`tests/test_zweige.py` und die „Branch“-Tests in den Orakel-Dateien): jede Variante weicht auf mindestens einer Zelle von der Nachbarvariante ab; keine Spalte der vorgerechneten Tabellen besteht nur aus Nullen. Kahan mit Rest gleich null ist ein Befund (E1), kein toter Zweig; ohne den Rest ist der Fehler nicht null.
- **Messreihen** (`tests/test_experiments.py`, `tests/test_results.py`): Struktur, Reproduzierbarkeit, vorgerechnete Datei aktuell gegen die Quelldateien (Prüfsumme, Zeilenenden normalisiert), gespeicherte Zellen von den Live-Funktionen reproduziert.
- **Presets** (`tests/test_presets.py`): vollständig, Werte im erlaubten Bereich, und was jedes Preset zeigt, an seinen Zahlen.
- **Mutationsprüfung** (`oracle_mutation.py` aus dem Website-Repo, inkrementell, Zustand in `tests/mutation_state.json`): `mtw_sums.py`, `mtw_estimators.py`, `mtw_stein.py` und `mtw_theory.py`, 372 Stellen, in mehreren Läufen mit je 80 bis 200 gezogenen Stellen (Werkzeugausgabe, nicht als Test festgeschrieben). Im ersten Lauf überlebten 9 von 80 Mutanten (Rate 89 %); die Lücken (Vorzeichenmuster ab n = 10 beim Durchschlag, quadrierte Fehler im James-Stein-Risiko, der zweite Zweig der Neumaier-Rechnung, Dimensionsgrenzen, Randwerte der Lognormal-Verteilungsfunktion) sind mit Tests geschlossen. Gesichert sind jetzt 370 Stellen (323 gefangen, 47 als äquivalent festgehalten: Schleifenkappen der Bisektionen und Iterationen, Gleichstände ohne Wirkung, Konstanten, die die Quadratur durch ihre Normierung aufhebt). Die 2 Zeitüberschreitungen in `pairwise_sum` (`while len(v) > 1` zu `>= 1` bzw. `> 0`: Endlosschleife, weil die Länge bei 1 nicht mehr sinkt) fängt ein Test mit Zeitgrenze (`tests/test_termination.py`: eigener Prozess, 20 s); am Mutanten `>= 1` in einer Kopie nachgeprüft, der Test schlägt dort nach 20 s fehl. Im Zustand des Mutationswerkzeugs stehen sie nicht, weil das Werkzeug Zeitüberschreitungen nicht einträgt. Nicht gezogen wurden Messreihen, Auswertung, Darstellung und die kopierte `mtw_rng.py`. „Mutation sauber“ steht hier bewusst nicht.
- **Aussagen** (`tests/test_claims.py`): jede Zahl dieser README.
- **Oberfläche:** AppTest (jeder Knopf, jeder Regler am Minimum und Maximum, jede Option der Auswahlregler, Extremkombinationen, Permalink), Layout, Footer, PDF, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App mit zwei Reitern |
| `mtw_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets mit Hilfetexten |
| `mtw_sums.py`, `mtw_estimators.py`, `mtw_stein.py`, `mtw_theory.py` | Kern: Summation, Mitten, James-Stein, Theorie |
| `mtw_scenario.py`, `mtw_rng.py` | Zufallsströme, Verteilungen, Reihen, Ebene, Lager; SplitMix64 (Kopie) |
| `mtw_experiments.py` | Messreihen-Funktionen, Prüfsumme der Quellen |
| `mtw_results.py` | Laden der vorgerechneten Messreihen |
| `mtw_evaluation.py` | Einstellungen, Live-Analysen, Verdikte |
| `mtw_visualization.py`, `mtw_pdf_export.py`, `mtw_presets.py`, `mtw_layout.py` | Figuren, PDF, Permalink und Presets, fester Kopf |
| `tools/make_results.py` | erzeugt `data/mtw_results.json` |
| `data/mtw_results.json` | die vorgerechneten Messreihen mit Prüfsumme |
| `tests/` | Tests (siehe oben) |
| `requirements.txt`, `requirements-dev.txt`, `.github/workflows/tests.yml` | Laufzeit (kein scipy), Test und Gegenproben (scipy nur dort), CI |

## Bewusst nicht umgesetzt

- **Trimmen, Winsorisieren, Hodges–Lehmann, Huber** und die Kurve Effizienz gegen Durchschlag als Familie (Stück 4 der Reihe); die Effizienz des Medians steht hier nur bei Normalverteilung.
- **Die Lp-Familie der Mitten** (Stück 3), **der geometrische Median in der Ebene** (Stück 5), **Median-of-Means** (Stück 6), **Raten und Gewichte** (Stück 7), **die Mitte auf dem Kreis** (Stück 8).
- **Newsvendor mit Kosten:** Der Bestand steht nur als Quantil bzw. Expektile; Kosten der falschen Wahl und Bestellrhythmen gehören zu `newsvendor-evpi-demo` und `forecast-inventory-demo`.
- **Varianz-Rechnung** (Welford-Varianz, Σx² − n·x̄²), **float16 und bfloat16**, **James-Stein bei unbekannter Streuung**.
- **Effizienz des Medians für andere Verteilungen als Normal** (Laplace, t, Gleichverteilung) und der Fall Cauchy.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die Streamlit-Cloud-Einstellung (Python-Version) ist erst beim Deploy festzulegen; entwickelt wird mit Python 3.14, die CI läuft mit 3.12. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die Messreihen werden mit `python tools/make_results.py --part <name>` (jeder Teil einzeln, parallel möglich; Namen: e1_f32, e1_f64, e1_mech, e1_diag, e1_share, e2_ratio_f32, e2_ratio_f64, e2_order_f32, e2_order_f64, e2_accu, e3, e5, e6, e7, e8) und `python tools/make_results.py --merge` neu erzeugt; `tests/test_results.py` meldet, wenn die Datei hinter den Quelldateien zurückliegt.

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
