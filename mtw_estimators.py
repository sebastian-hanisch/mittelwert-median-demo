"""Die „Mitten“ als Minimierer: Schwerpunkt (Quadratsumme), Median (Betragssumme), Quantil (Pinball-Verlust), Expektile (asymmetrische Quadratsumme), dazu Einfluss,
Verschiebung bei Ausreißern, Durchschlagspunkt und das verkleinerte Mittel. Nur numpy/Standardbibliothek, jede Regel eine eigene Funktion (DEMO-PLAYBOOK §1b)."""

from __future__ import annotations

import itertools
import math

import numpy as np


# --- Quadratsumme: Mittel und Schwerpunkt -----------------------------------------------------------------------------------------------------

def centroid(points, weights=None) -> np.ndarray:
    """Gewichteter Schwerpunkt der Zeilen von `points` (ohne Gewichte: alle gleich). Er minimiert Σ w_i ‖x_i − c‖²."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) == 0:
        raise ValueError("keine Punkte")
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=np.float64)
    return (w[:, None] * p).sum(axis=0) / w.sum()


def ssq(points, c, weights=None) -> float:
    """Σ w_i ‖x_i − c‖²: die Quadratsumme, die der Schwerpunkt minimiert."""
    p = np.asarray(points, dtype=np.float64)
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=np.float64)
    return float((w * ((p - np.asarray(c, dtype=np.float64)) ** 2).sum(axis=1)).sum())


def huygens(points, c) -> tuple[float, float]:
    """Zerlegung nach Huygens (Steiner): Σ‖x_i − c‖² = Σ‖x_i − x̄‖² + n‖c − x̄‖². Gibt (Streuung um den Schwerpunkt, Zuschlag für den Abstand von c zum Schwerpunkt) zurück."""
    p = np.asarray(points, dtype=np.float64)
    xbar = centroid(p)
    within = float(((p - xbar) ** 2).sum())
    between = float(len(p) * ((np.asarray(c, dtype=np.float64) - xbar) ** 2).sum())
    return within, between


def shrunk_mse(c: float, mu: float, sigma: float, n: int) -> float:
    """Mittlerer quadratischer Fehler des verkleinerten Mittels c · x̄ (n Werte, Mittel μ, Streuung σ): c²σ²/n + (1 − c)²μ² (Varianz plus Verzerrung zum Quadrat)."""
    return c * c * sigma * sigma / n + (1.0 - c) ** 2 * mu * mu


# --- Betragssumme: Median --------------------------------------------------------------------------------------------------------------------

def median(x) -> float:
    """Median: mittlerer Wert, bei geradem n der Mittelwert der beiden mittleren (die Mitte des Minimierer-Intervalls)."""
    s = np.sort(np.asarray(x, dtype=np.float64))
    n = len(s)
    if n == 0:
        raise ValueError("leere Stichprobe")
    return float(s[n // 2]) if n % 2 else float(0.5 * (s[n // 2 - 1] + s[n // 2]))


def median_rows(X) -> np.ndarray:
    """Median je Zeile von X (Form (Zeilen, n)), gleiche Regel wie `median`."""
    s = np.sort(np.asarray(X, dtype=np.float64), axis=1)
    n = s.shape[1]
    return s[:, n // 2] if n % 2 else 0.5 * (s[:, n // 2 - 1] + s[:, n // 2])


def median_interval(x) -> tuple[float, float]:
    """Menge der Minimierer von Σ|x_i − c|: der mittlere Wert (ungerades n) bzw. das Intervall zwischen den beiden mittleren Werten (gerades n)."""
    s = np.sort(np.asarray(x, dtype=np.float64))
    n = len(s)
    if n == 0:
        raise ValueError("leere Stichprobe")
    if n % 2:
        return float(s[n // 2]), float(s[n // 2])
    return float(s[n // 2 - 1]), float(s[n // 2])


def sad(x, c: float) -> float:
    """Σ|x_i − c|: die Betragssumme, die der Median minimiert."""
    return float(np.abs(np.asarray(x, dtype=np.float64) - c).sum())


# --- Quantil und Expektile -------------------------------------------------------------------------------------------------------------------

def pinball(x, c: float, tau: float) -> float:
    """Pinball-Verlust Σ ρ_τ(x_i − c) mit ρ_τ(u) = τu für u ≥ 0 und (τ − 1)u sonst: ihn minimiert das Quantil zur Stufe τ."""
    u = np.asarray(x, dtype=np.float64) - c
    return float(np.where(u >= 0, tau * u, (tau - 1.0) * u).sum())


def quantile_interval(x, tau: float) -> tuple[float, float]:
    """Menge der Minimierer des Pinball-Verlusts: x_(⌈nτ⌉), und bei ganzzahligem nτ das Intervall [x_(nτ), x_(nτ+1)] (für τ = 1/2 und gerades n der Median-Fall)."""
    if not 0.0 < tau < 1.0:
        raise ValueError("τ muss zwischen 0 und 1 liegen")
    s = np.sort(np.asarray(x, dtype=np.float64))
    n = len(s)
    if n == 0:
        raise ValueError("leere Stichprobe")
    k = n * tau
    if abs(k - round(k)) < 1e-9:
        k = int(round(k))
        if k >= n:
            return float(s[-1]), float(s[-1])
        return float(s[k - 1]), float(s[k])
    j = int(math.ceil(k))
    return float(s[j - 1]), float(s[j - 1])


def expectile(x, tau: float, maxit: int = 200) -> float:
    """Expektile zur Stufe τ: Lösung von τ Σ(x_i − e)₊ = (1 − τ) Σ(e − x_i)₊, d. h. Minimierer der asymmetrisch gewichteten Quadratsumme (Gewicht τ oberhalb von e,
    1 − τ unterhalb). Fixpunktiteration (gewichtetes Mittel); endlich, weil die Zuordnung der Werte zu den Gewichten nur endlich oft wechselt."""
    if not 0.0 < tau < 1.0:
        raise ValueError("τ muss zwischen 0 und 1 liegen")
    a = np.asarray(x, dtype=np.float64)
    if len(a) == 0:
        raise ValueError("leere Stichprobe")
    e = float(a.mean())
    for _ in range(maxit):
        w = np.where(a > e, tau, 1.0 - tau)
        e_new = float((w * a).sum() / w.sum())
        if e_new == e:
            break
        e = e_new
    return e


def expectile_rows(X, tau: float, maxit: int = 200) -> np.ndarray:
    """Expektile je Zeile von X, gleiche Regel wie `expectile`."""
    a = np.asarray(X, dtype=np.float64)
    e = a.mean(axis=1)
    for _ in range(maxit):
        w = np.where(a > e[:, None], tau, 1.0 - tau)
        e_new = (w * a).sum(axis=1) / w.sum(axis=1)
        if np.array_equal(e_new, e):
            break
        e = e_new
    return e


def expectile_loss(x, c: float, tau: float) -> float:
    """Asymmetrisch gewichtete Quadratsumme Σ w(x_i − c)² (w = τ für x_i > c, sonst 1 − τ): die Expektile minimiert sie."""
    a = np.asarray(x, dtype=np.float64)
    return float((np.where(a > c, tau, 1.0 - tau) * (a - c) ** 2).sum())


# --- Einfluss, Verschiebung, Durchschlag -----------------------------------------------------------------------------------------------------

def sensitivity_curve(base, t: float, stat: str) -> float:
    """Stichproben-Einflusskurve SC(t) = n · (T(x_1, …, x_{n−1}, t) − T(x_1, …, x_{n−1})) mit n = Zahl der Werte einschließlich t (hier so definiert).
    `stat` ∈ {"Mittel", "Median"}. Beim Mittel ist SC(t) = t − x̄_{n−1}: linear und unbeschränkt."""
    b = np.asarray(base, dtype=np.float64)
    f = {"Mittel": lambda v: float(np.mean(v)), "Median": median}[stat]
    return (len(b) + 1) * (f(np.append(b, t)) - f(b))


def contaminate(x, idx, value: float) -> np.ndarray:
    """Kopie von x, in der die Werte an den Positionen `idx` durch `value` ersetzt sind."""
    y = np.array(x, dtype=np.float64)
    y[list(idx)] = value
    return y


def shift_under_contamination(x, idx, value: float, stat: str) -> float:
    """Betrag der Verschiebung |T(verschmutzt) − T(sauber)|, wenn die Werte an den Positionen `idx` durch `value` ersetzt werden."""
    f = {"Mittel": lambda v: float(np.mean(v)), "Median": median}[stat]
    return abs(f(contaminate(x, idx, value)) - f(x))


def max_shift_below_breakdown(x, k: int, big: float = 1e6, both_signs: bool | None = None) -> float:
    """Größte Verschiebung des Medians, die sich mit k ersetzten Werten (±M, M = big) erreichen lässt: Aufzählung aller k-Teilmengen der Positionen.
    Vorzeichenmuster werden für n ≤ 9 vollständig aufgezählt, darüber nur +M."""
    a = np.asarray(x, dtype=np.float64)
    n = len(a)
    signs_all = (n <= 9) if both_signs is None else both_signs
    m0 = median(a)
    worst = 0.0
    for idx in itertools.combinations(range(n), k):
        patterns = itertools.product((1.0, -1.0), repeat=k) if signs_all else [(1.0,) * k]
        for sg in patterns:
            y = a.copy()
            for i, s in zip(idx, sg):
                y[i] = s * big
            worst = max(worst, abs(median(y) - m0))
    return worst


def breakdown_k(x, big=(1e3, 1e6), both_signs: bool | None = None) -> int | None:
    """Durchschlagspunkt des Medians durch Aufzählung: die kleinste Zahl k von Werten, die sich so ersetzen lassen, dass die größte Verschiebung beim Übergang von
    M = big[0] zu M = big[1] noch um mehr als 1 wächst („beliebig weit“). Rückgabe None, wenn kein k ≤ n genügt (beim Median kommt das nicht vor)."""
    n = len(np.asarray(x))
    for k in range(1, n + 1):
        low = max_shift_below_breakdown(x, k, big[0], both_signs)
        high = max_shift_below_breakdown(x, k, big[1], both_signs)
        if high - low > 1.0:
            return k
    return None
