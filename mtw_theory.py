"""Geschlossene Formeln und Quadratur: exakte Varianz des Medians bei Normalverteilung, Lognormal-Quantile und -Expektilen, Lücke zwischen Mittelwert und Median.

Nur numpy/Standardbibliothek (die Normalverteilung über `math.erf`); scipy dient nur als Gegenprobe in den Tests."""

from __future__ import annotations

import math

import numpy as np

SQRT2 = math.sqrt(2.0)
EFFICIENCY_LIMIT = 2.0 / math.pi            # Var(Mittel)/Var(Median) bei Normalverteilung für n → ∞
INFLUENCE_PLATEAU = math.sqrt(math.pi / 2)  # Grenzwert der Einflusskurve des Medians bei Normalverteilung: 1/(2 f(0))

_erf = np.vectorize(math.erf, otypes=[float])


def norm_cdf(x):
    """Verteilungsfunktion der Standardnormalverteilung (Skalar oder Feld)."""
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=np.float64) / SQRT2))


def norm_pdf(x):
    """Dichte der Standardnormalverteilung."""
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_ppf(p: float) -> float:
    """Umkehrfunktion der Standardnormalverteilung (Bisektion)."""
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(norm_cdf(mid)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- Varianz des Medians bei Normalverteilung --------------------------------------------------------------------------------------------

def _grid(half_width: float, points: int):
    x = np.linspace(-half_width, half_width, points)
    F = norm_cdf(x)
    return x, F, norm_pdf(x)


def var_median_normal(n: int, half_width: float = 12.0, points: int = 60001) -> tuple[float, float]:
    """Exakte Varianz des Medians von n Standardnormalwerten durch Quadratur; gibt (Varianz, Masse der Dichte als Kontrolle, soll 1 sein) zurück.

    Ungerades n = 2k + 1: Dichte der mittleren Ordnungsstatistik c · f · F^k · (1 − F)^k. Gerades n = 2k: Median = (X_(k) + X_(k+1))/2; E[M²] = (E X_(k)² + E X_(k)X_(k+1))/2
    (E X_(k)² = E X_(k+1)² aus der Symmetrie), E X_(k)X_(k+1) aus der gemeinsamen Dichte zweier benachbarter Ordnungsstatistiken, die sich in A(x)·B(y) aufspaltet,
    mit einer laufenden Summe über x < y."""
    if n < 1:
        raise ValueError("n >= 1")
    x, F, f = _grid(half_width, points)
    tiny = 1e-300
    logF, log1F = np.log(np.clip(F, tiny, 1.0)), np.log(np.clip(1.0 - F, tiny, 1.0))
    logf = np.log(f + tiny)
    if n % 2:
        k = (n - 1) // 2
        dens = np.exp(math.lgamma(n + 1) - 2 * math.lgamma(k + 1) + logf + k * logF + k * log1F)
        mass = float(np.trapezoid(dens, x))
        return float(np.trapezoid(dens * x * x, x)) / mass, mass
    k = n // 2
    # E X_(k)^2 aus der Dichte der k-ten Ordnungsstatistik
    d1 = np.exp(math.lgamma(n + 1) - math.lgamma(k) - math.lgamma(n - k + 1) + logf + (k - 1) * logF + (n - k) * log1F)
    mass1 = float(np.trapezoid(d1, x))
    ex2 = float(np.trapezoid(d1 * x * x, x)) / mass1
    # E X_(k) X_(k+1): gemeinsame Dichte = n!/((k-1)!(k-1)!) · A(x) · B(y), x < y
    const = math.lgamma(n + 1) - 2 * math.lgamma(k)
    A = np.exp(logf + (k - 1) * logF) * x
    B = np.exp(logf + (k - 1) * log1F) * x
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (A[1:] + A[:-1]) * np.diff(x))])
    exy = math.exp(const) * float(np.trapezoid(B * cum, x))
    A0, B0 = np.exp(logf + (k - 1) * logF), np.exp(logf + (k - 1) * log1F)
    cum0 = np.concatenate([[0.0], np.cumsum(0.5 * (A0[1:] + A0[:-1]) * np.diff(x))])
    mass2 = math.exp(const) * float(np.trapezoid(B0 * cum0, x))
    return 0.5 * (ex2 + exy / mass2), mass1


def efficiency_normal(n: int) -> float:
    """Var(Mittel)/Var(Median) bei Normalverteilung, exakt: (1/n)/Var(Median). Bei n = 2 gleich 1 (Median = Mittel); gerades n liegt über dem Nachbarn n + 1."""
    var, _ = var_median_normal(n)
    return (1.0 / n) / var


# --- Lognormal ---------------------------------------------------------------------------------------------------------------------------

def gap_mean_median(sigma: float) -> float:
    """Mittelwert / Median der Lognormal(0; σ): e^(σ²/2)."""
    return math.exp(sigma * sigma / 2.0)


def median_miss_share(sigma: float) -> float:
    """Anteil, den der Median am Mittelwert der Lognormal(0; σ) auslässt: 1 − e^(−σ²/2) (Verzerrung des Medians als Schätzer des Mittelwerts, unabhängig von n)."""
    return 1.0 - math.exp(-sigma * sigma / 2.0)


def share_below_mean(sigma: float) -> float:
    """Anteil der Lognormal(0; σ)-Verteilung unterhalb des Mittelwerts: Φ(σ/2)."""
    return float(norm_cdf(sigma / 2.0))


def lognormal_cdf(x: float, sigma: float) -> float:
    """Verteilungsfunktion der Lognormal(0; σ)."""
    return float(norm_cdf(math.log(x) / sigma)) if x > 0 else 0.0


def lognormal_quantile(sigma: float, tau: float) -> float:
    """Quantil der Lognormal(0; σ) zur Stufe τ: e^(σ Φ⁻¹(τ))."""
    return math.exp(sigma * norm_ppf(tau))


def lognormal_expectile(sigma: float, tau: float) -> float:
    """Expektile der Lognormal(0; σ): Lösung von τ E(X − e)₊ = (1 − τ) E(e − X)₊ mit den Teilerwartungen über Φ; Bisektion (die linke Seite minus rechte ist in e fallend)."""
    mean = math.exp(sigma * sigma / 2.0)

    def g(e: float) -> float:
        d2 = -math.log(e) / sigma
        d1 = d2 + sigma
        up = mean * float(norm_cdf(d1)) - e * float(norm_cdf(d2))
        down = e * (1.0 - float(norm_cdf(d2))) - mean * (1.0 - float(norm_cdf(d1)))
        return tau * up - (1.0 - tau) * down

    lo, hi = 1e-9, 1e9
    for _ in range(300):
        mid = math.sqrt(lo * hi)
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def normal_expectile(tau: float) -> float:
    """Expektile der Standardnormalverteilung: τ(φ(e) − e(1 − Φ(e))) = (1 − τ)(eΦ(e) + φ(e)); Bisektion."""
    def g(e: float) -> float:
        p, c = float(norm_pdf(e)), float(norm_cdf(e))
        return tau * (p - e * (1.0 - c)) - (1.0 - tau) * (e * c + p)

    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def poisson_cdf(lam: float, k: int) -> float:
    """Verteilungsfunktion der Poisson(λ) an der ganzen Zahl k."""
    return sum(math.exp(-lam) * lam ** j / math.factorial(j) for j in range(k + 1))
