"""Abbruch der paarweisen Summe (Mutationsprüfung): `pairwise_sum` muss für jede Länge terminieren.

Die Schleife `while len(v) > 1` in `mtw_sums.pairwise_sum` halbiert die Länge. Wird die Bedingung zu `>= 1` oder `> 0` abgewandelt, sinkt die Länge bei 1 nicht
mehr, und die Funktion läuft endlos. Ein solcher Lauf darf den Testlauf nicht blockieren: Die Funktion läuft deshalb in einem eigenen Prozess mit Zeitgrenze, den
`subprocess.run` bei Überschreitung beendet (nur der eigene Kindprozess, nie per Namensmuster)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIMIT_SECONDS = 20

SCRIPT = (
    "import numpy as np, mtw_sums as M\n"
    "out = []\n"
    "for n in (1, 2, 3, 7, 8, 100):\n"
    "    x = np.arange(1, n + 1, dtype=np.float64)\n"
    "    out.append(float(M.pairwise_sum(x, np.float64)))\n"
    "print(out)\n"
)


def test_pairwise_sum_terminates_for_every_length_and_gives_the_exact_sums():
    try:
        done = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=LIMIT_SECONDS)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"pairwise_sum hat nach {LIMIT_SECONDS} s nicht geendet (Endlosschleife)")
    assert done.returncode == 0, done.stderr
    # Summen 1..n = n(n+1)/2: 1, 3, 6, 28, 36, 5050
    assert done.stdout.strip() == "[1.0, 3.0, 6.0, 28.0, 36.0, 5050.0]"
