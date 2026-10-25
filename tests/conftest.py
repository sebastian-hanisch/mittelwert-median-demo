"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Die Erwartungswerte stehen im Test als Rechnung.

Handrechnungen:
* float32 [2²⁴, 1, 1]: 2²⁴ + 1 liegt genau zwischen 2²⁴ und 2²⁴ + 2 (Abstand der float32-Zahlen dort 2) und wird auf die gerade Mantisse 2²⁴ gerundet, ebenso beim zweiten Mal:
  die fortlaufende Summe bleibt bei 2²⁴. Kahan: erster Schritt s = 2²⁴, Rest c = (t − s) − y = 0 − 1 = −1; zweiter Schritt y = 1 − (−1) = 2, t = 2²⁴ + 2 exakt, c = 0.
  Neumaier: c wird 1, dann 2; s + c = 2²⁴ + 2.
* float64, zehnmal 0,1: die fortlaufende Summe ist 0,9999999999999999, Kahan und `math.fsum` geben 1,0.
* Welford [1, 2, 3, 4]: Mittel nach 1, 2, 3, 4 Werten: 1; 1 + (2 − 1)/2 = 1,5; 1,5 + (3 − 1,5)/3 = 2; 2 + (4 − 2)/4 = 2,5.
* Punkte (0, 0), (4, 0), (0, 3): Schwerpunkt (4/3, 1); Σ‖x − c‖² = (16/9 + 1) + ((8/3)² + 1) + (16/9 + 4) = 50/3; im Nullpunkt 0 + 16 + 9 = 25 = 50/3 + 3·(16/9 + 1) = 50/3 + 25/3.
* (1, 2, 3) plus z = 10: Mittel der drei = 2, mit z = 16/4 = 4, SC = 4·(4 − 2) = 8 = z − x̄ = 10 − 2.
* James-Stein d = 3, X = (3, 0, 0): ‖X‖² = 9, Faktor 1 − 1/9 = 8/9, Ergebnis (8/3, 0, 0). X = (0,5, 0, 0): Faktor 1 − 1/0,25 = −3, Positive Part: (0, 0, 0).
* Mittel c · x̄ bei n = 3, μ = 0,8, σ = 1, c = 0,6: 0,36/3 + 0,16 · 0,64 = 0,12 + 0,1024 = 0,2224.
* (3, 1, 2): Median 2. (1, 2, 3, 4): Minimierer-Intervall [2; 3], Σ|x − c| = 1,5 + 0,5 + 0,5 + 1,5 = 4 bei c = 2,5 und bei 2 und 3 ebenfalls 4.
* n = 5 (1 bis 5): zwei Werte (die kleinsten) durch 1000 ersetzt: (3, 4, 5, 1000, 1000), Median 5; drei Werte: (4, 5, 1000, 1000, 1000), Median 1000: Durchschlag bei k = 3 = ⌈5/2⌉.
* (1, 2, 3, 4, 10) mit τ = 0,8: n·τ = 4 ganzzahlig, Intervall [4; 10]; Pinball bei c = 4: 0,6 + 0,4 + 0,2 + 0 + 4,8 = 6; bei c = 10: 1,8 + 1,6 + 1,4 + 1,2 + 0 = 6.
* Expektile von (0, 4) bei τ = 0,8: 0,8 · (4 − e) = 0,2 · e, also e = 3,2; bei τ = 0,5 das Mittel 2.
* Lognormal σ = 0,4: der Plan „Anzahl × Median“ lässt den Anteil 1 − e^(−0,08) = 0,0769 aus.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tiny_f32():
    """[2²⁴, 1, 1] in float32."""
    return np.array([2.0 ** 24, 1.0, 1.0], dtype=np.float32)


@pytest.fixture
def tenths():
    """Zehnmal 0,1 in float64."""
    return np.array([0.1] * 10, dtype=np.float64)


@pytest.fixture
def one_to_four():
    return np.array([1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def triangle():
    """Punkte (0, 0), (4, 0), (0, 3)."""
    return np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
