"""James-Stein-Schätzer für d Mittel bei bekannter Streuung 1 (je Lager eine Beobachtung X_j ~ N(μ_j, 1)): Einzelmittel gegen Schrumpfung zum Nullpunkt bzw. zum gemeinsamen Mittel.

Risiko = mittlere Summe der quadratischen Fehler E‖Schätzer − μ‖². Das Einzelmittel X hat das Risiko d. Nur numpy/Standardbibliothek."""

from __future__ import annotations

import math

import numpy as np


def james_stein_factor(X) -> np.ndarray:
    """Schrumpfungsfaktor 1 − (d − 2)/‖X‖² je Zeile von X (Form (Läufe, d)); wird negativ, wenn ‖X‖² < d − 2. Für X = 0 (Wahrscheinlichkeit null) gilt der Faktor 0."""
    a = np.asarray(X, dtype=np.float64)
    d = a.shape[1]
    ss = (a ** 2).sum(axis=1)
    with np.errstate(divide="ignore"):
        return np.where(ss > 0, 1.0 - (d - 2) / ss, 0.0)


def james_stein(X) -> np.ndarray:
    """James-Stein-Schätzer zum Nullpunkt: (1 − (d − 2)/‖X‖²) · X, je Zeile (d ≥ 3)."""
    a = np.asarray(X, dtype=np.float64)
    if a.shape[1] < 3:
        raise ValueError("James-Stein braucht d >= 3")
    return james_stein_factor(a)[:, None] * a


def positive_part(X) -> np.ndarray:
    """Positive-Part-Variante: der Faktor wird nie unter null gelassen, max(1 − (d − 2)/‖X‖², 0) · X."""
    a = np.asarray(X, dtype=np.float64)
    if a.shape[1] < 3:
        raise ValueError("James-Stein braucht d >= 3")
    return np.maximum(james_stein_factor(a), 0.0)[:, None] * a


def js_toward_mean(X) -> np.ndarray:
    """Positive-Part-Schrumpfung zum gemeinsamen Mittel X̄: X̄ + max(1 − (d − 3)/‖X − X̄‖², 0) · (X − X̄), je Zeile (d ≥ 4)."""
    a = np.asarray(X, dtype=np.float64)
    d = a.shape[1]
    if d < 4:
        raise ValueError("Schrumpfung zum gemeinsamen Mittel braucht d >= 4")
    xbar = a.mean(axis=1, keepdims=True)
    dev = a - xbar
    ss = (dev ** 2).sum(axis=1)
    with np.errstate(divide="ignore"):
        factor = np.where(ss > 0, np.maximum(1.0 - (d - 3) / ss, 0.0), 0.0)
    return xbar + factor[:, None] * dev


def exact_js_risk(d: int, lam: float) -> float:
    """Exaktes Risiko des James-Stein-Schätzers (ohne Positive Part) bei ‖μ‖² = lam: d − (d − 2)² · E[1/χ²_d(lam)]. ‖X‖² ist nichtzentral χ² mit d Freiheitsgraden;
    als Poisson-Mischung zentraler χ²-Verteilungen gilt E[1/χ²_{d+2K}] = 1/(d + 2K − 2) mit K ~ Poisson(lam/2)."""
    if d < 3:
        raise ValueError("d >= 3")
    if lam == 0.0:
        return 2.0
    half = lam / 2.0
    kmax = int(half + 12 * math.sqrt(half + 1) + 50)
    total = 0.0
    for k in range(kmax):
        log_p = -half + k * math.log(half) - math.lgamma(k + 1)
        total += math.exp(log_p) / (d + 2 * k - 2)
    return d - (d - 2) ** 2 * total


def risk_summary(mu, X) -> dict:
    """Risiken aus einem Block von Läufen X (Form (Läufe, d)) zur wahren Lage mu: Einzelmittel, James-Stein, Positive Part, jeweils als Summe über alle Lager,
    dazu das Risiko des ersten Lagers allein (Einzelmittel: 1) für James-Stein und Positive Part."""
    a = np.asarray(X, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    js, pp = james_stein(a), positive_part(a)
    out = {
        "einzel": float(((a - mu) ** 2).sum(axis=1).mean()),
        "js": float(((js - mu) ** 2).sum(axis=1).mean()),
        "pp": float(((pp - mu) ** 2).sum(axis=1).mean()),
        "koordinate1_js": float(((js[:, 0] - mu[0]) ** 2).mean()),
        "koordinate1_pp": float(((pp[:, 0] - mu[0]) ** 2).mean()),
    }
    if a.shape[1] >= 4:
        out["gemeinsames_mittel"] = float(((js_toward_mean(a) - mu) ** 2).sum(axis=1).mean())
    return out
