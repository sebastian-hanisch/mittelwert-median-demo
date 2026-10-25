"""Summationsverfahren für das Mittel, jedes mit einem Zahlenformat (float32 oder float64), und die Referenz „exaktes Mittel der gespeicherten Werte“.

Jede Einheit ist eine Regel für sich (DEMO-PLAYBOOK §1b): fortlaufende Summe, paarweise Summe, Kahan, Neumaier, Welford, verschobene Summe. Die Schleifen rechnen
Wert für Wert im Zahlenformat `dt` (numpy-float32-Skalare bzw. Python-float = IEEE double) und sind deshalb bitgenau reproduzierbar; nur `numpy_sum` hängt von der
numpy-Version ab. Verfahren mit Zustand (Kahan, Neumaier, Welford) lassen sich mit `state` fortsetzen, damit eine Reihe in Abschnitten durchlaufen werden kann.
Fehler werden immer gegen das EXAKTE Mittel der gespeicherten Werte gemessen (`exact_mean`), nicht gegen den Parameter μ des Generators."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np

F32, F64 = np.float32, np.float64
CHUNK = 200_000      # float64-Werte werden abschnittsweise in Python-floats umgewandelt: begrenzt den Speicher bei 10⁷ Werten


def _zero(dt):
    """Null im Zahlenformat; für float64 ein Python-float (gleiche IEEE-Rechnung, aber schneller als numpy-Skalare)."""
    return 0.0 if dt is F64 else F32(0)


def _scalars(x, dt):
    """Die Werte zum Durchlaufen: float64 als Python-floats, float32 als numpy-float32-Skalare."""
    x = np.asarray(x)
    if dt is not F64:
        return x.astype(F32, copy=False)
    return itertools.chain.from_iterable(x[i:i + CHUNK].tolist() for i in range(0, len(x), CHUNK))


# --- Referenz --------------------------------------------------------------------------------------------------------------------------------

def exact_sum(values) -> Fraction:
    """Exakte Summe (als Bruch) von float64-Werten (jeder float32-Wert ist als float64 exakt). `math.fsum` rundet die exakte Summe einmal; der Rest davon wird
    wieder mit fsum summiert, bis nichts mehr übrig ist (in der Regel zwei bis drei Durchgänge)."""
    a = np.asarray(values, dtype=np.float64)
    parts: list[float] = []
    for _ in range(12):
        rest = math.fsum(np.concatenate([a, -np.array(parts)])) if parts else math.fsum(a)
        if rest == 0.0:
            break
        parts.append(rest)
    else:
        raise ArithmeticError("exact_sum: Rest verschwindet nicht")
    return sum((Fraction(p) for p in parts), Fraction(0))


def exact_mean(values) -> Fraction:
    """Exaktes Mittel (Bruch) der gespeicherten Werte."""
    n = len(values)
    if n == 0:
        raise ValueError("leere Reihe")
    return exact_sum(values) / n


def mean_error(m_hat, exact: Fraction) -> float:
    """Fehler (Ergebnis minus exaktes Mittel), vorzeichenbehaftet; das Ergebnis geht ohne weitere Rundung in den Vergleich ein."""
    return float(Fraction(float(m_hat)) - exact)


def sum_error_as_mean(total, n: int, exact: Fraction) -> float:
    """Fehler des Mittels, wenn die Summe `total` EXAKT durch n geteilt wird (gemessen wird nur das Summationsverfahren, nicht die Division)."""
    return float(Fraction(float(total)) / n - exact)


def floor_of_format(exact: Fraction, dt) -> float:
    """Untergrenze: das exakte Ergebnis (Summe bzw. Mittel) auf das Zahlenformat gerundet, abzüglich des Ergebnisses selbst. Besser als dieser Fehler kann ein
    Verfahren, das sein Ergebnis im Zahlenformat ausgibt, nicht werden."""
    return float(Fraction(float(dt(float(exact)))) - exact)


# --- Verfahren --------------------------------------------------------------------------------------------------------------------------------

def naive_prefix(x, dt) -> np.ndarray:
    """Fortlaufende Summe im Zahlenformat: alle Zwischensummen (Wert k = Summe der ersten k + 1 Werte)."""
    return np.cumsum(np.asarray(x, dtype=dt), dtype=dt)


def naive_sum(x, dt):
    """Fortlaufende Summe: s ← s + x_i, jede Zwischensumme auf das Zahlenformat gerundet."""
    return naive_prefix(x, dt)[-1] if len(x) else _zero(dt)


def numpy_sum(x, dt):
    """`numpy.sum` mit dem Zahlenformat als Akkumulator (summiert paarweise in Blöcken; Aufbau hängt von der numpy-Version ab)."""
    return np.sum(np.asarray(x, dtype=dt), dtype=dt)


def pairwise_sum(x, dt):
    """Vollständiger Binärbaum: erst Nachbarn, dann Nachbarn der Teilsummen, … (bei ungerader Länge wandert der letzte Wert unverändert eine Stufe höher)."""
    v = np.asarray(x, dtype=dt).copy()
    if len(v) == 0:
        return _zero(dt)
    while len(v) > 1:
        m = len(v) // 2
        w = v[0:2 * m:2] + v[1:2 * m:2]
        if len(v) % 2:
            w = np.concatenate([w, v[-1:]])
        v = w
    return v[0]


def kahan_sum(x, dt, state=None):
    """Kahan-Summation: ein Rest c merkt sich, was beim Addieren verloren ging, und geht in den nächsten Summanden ein. Gibt (s, c) zurück; das Ergebnis ist s,
    „mit Rest“ gilt s − c. `state` setzt eine begonnene Summe fort."""
    s, c = (_zero(dt), _zero(dt)) if state is None else state
    for v in _scalars(x, dt):
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    return s, c


def neumaier_sum(x, dt, state=None):
    """Neumaier-Summation (verbessertes Kahan): der Rest c sammelt, was beim Addieren verloren ging, ohne selbst in den nächsten Summanden einzugehen; das Ergebnis ist s + c.
    Der Rest wird im Zahlenformat `dt` geführt. Gibt (s, c) zurück."""
    s, c = (_zero(dt), _zero(dt)) if state is None else state
    for v in _scalars(x, dt):
        t = s + v
        if abs(s) >= abs(v):
            c = c + ((s - t) + v)
        else:
            c = c + ((v - t) + s)
        s = t
    return s, c


def neumaier_sum_wide_rest(x, dt, state=None):
    """Wie `neumaier_sum`, aber der Rest c wird in float64 geführt (nur die Summe s im Zahlenformat `dt`): die Gegenprobe zur Frage, ob der Rest selbst zum Problem wird."""
    s, c = (_zero(dt), 0.0) if state is None else state
    for v in _scalars(x, dt):
        t = s + v
        if abs(s) >= abs(v):
            c = c + float((s - t) + v)
        else:
            c = c + float((v - t) + s)
        s = t
    return s, c


def welford_mean(x, dt, state=None):
    """Laufendes Mittel nach Welford (nur der Mittelwertteil): m ← m + (x_k − m) / k. Gibt (m, k) zurück; `state` = (m, k) setzt fort."""
    m, k = (_zero(dt), 0) if state is None else state
    cast = float if dt is F64 else dt
    for v in _scalars(x, dt):
        k += 1
        m = m + (v - m) / cast(k)
    return m, k


def shifted_mean(x, dt):
    """Verschobene Summe: x_1 abziehen, die Reste fortlaufend summieren und x_1 + Summe / n als Mittel zurückgeben (alles im Zahlenformat)."""
    v = np.asarray(x, dtype=dt)
    if len(v) == 0:
        raise ValueError("leere Reihe")
    first = v[0]
    total = naive_sum(v - first, dt)
    return dt(first + dt(total / dt(len(v))))


# --- Diagnose: wächst der Rest? ----------------------------------------------------------------------------------------------------------

def rest_trace(x, dt, method: str):
    """Größter Betrag des Rests c während der Summation und Endzustand (s, c) für `method` ∈ {"kahan", "neumaier", "neumaier_wide"}."""
    s, c = (_zero(dt), _zero(dt))
    if method == "neumaier_wide":
        c = 0.0
    biggest = 0.0
    for v in _scalars(x, dt):
        if method == "kahan":
            y = v - c
            t = s + y
            c = (t - s) - y
        elif method == "neumaier":
            t = s + v
            c = c + ((s - t) + v) if abs(s) >= abs(v) else c + ((v - t) + s)
        elif method == "neumaier_wide":
            t = s + v
            c = c + float((s - t) + v) if abs(s) >= abs(v) else c + float((v - t) + s)
        else:
            raise ValueError(method)
        s = t
        a = abs(float(c))
        if a > biggest:
            biggest = a
    return s, c, biggest


def mean_with_float64_accumulator(x, out_dt):
    """Mittel von Daten im Zahlenformat `out_dt`, aber mit float64-Akkumulator (`numpy.mean(dtype=float64)`), danach auf `out_dt` gerundet."""
    return out_dt(np.mean(np.asarray(x, dtype=out_dt), dtype=np.float64))
