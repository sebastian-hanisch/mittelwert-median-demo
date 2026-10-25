"""Szenario: Zufallsströme, Verteilungen, Messreihen für die Summation, Kunden in der Ebene, Lager-Mittel.

Alle Zufallszahlen stammen aus dem SplitMix64-Strom von `mtw_rng.py` (wörtliche Kopie); nichts aus `numpy.random`, damit jede Reihe auf jeder
Plattform und in jeder Python-Version gleich ist. SplitMix64 ist ein Zähler-Generator: der k-te Wert des Stroms hängt nur vom Seed und von k ab.
Deshalb lässt sich ein ganzer Block auf einmal mit numpy berechnen (`mix_words`); `tests/test_scenario.py` prüft ihn Wert für Wert gegen die
Python-Klasse `SplitMix64`. Normalwerte nach Box-Muller aus dem Strom; log, cos und sin kommen aus numpy, die letzte Binärstelle kann je Plattform abweichen.
Einheiten: Zeitstempel in Sekunden, Koordinaten in Metern bzw. Kilometern, Lieferzeiten in Tagen."""

from __future__ import annotations

import math
import zlib

import numpy as np

from mtw_rng import MASK, SplitMix64  # noqa: F401  (SplitMix64: Gegenprobe in den Tests)

GAMMA = 0x9E3779B97F4A7C15
_C1 = 0xBF58476D1CE4E5B9
_C2 = 0x94D049BB133111EB
TWO_M53 = 2.0 ** -53

DISTRIBUTIONS = ("Normal", "Laplace", "Logistisch", "Gleichverteilung", "t(3)", "t(5)", "Exponential", "Lognormal", "Pareto(2,5)", "Poisson")
ORDERS = ("zufällig", "aufsteigend", "absteigend")
SERIES_DISTRIBUTIONS = ("Normal", "Lognormal")
FORMATS = {"float32": np.float32, "float64": np.float64}
LOGNORMAL_SERIES_SIGMA = 1.0     # Form der schiefen Reihe im Reiter „Rechnen“: lognormal(0; 1), auf Mittel μ und Streuung σ gebracht


def mix_words(seed: int, start: int, count: int) -> np.ndarray:
    """Die Werte Nr. start … start + count − 1 (ab 0 gezählt) des SplitMix64-Stroms zum Seed, als uint64-Feld."""
    k = np.arange(start + 1, start + 1 + count, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = k * np.uint64(GAMMA) + np.uint64(seed & MASK)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(_C1)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(_C2)
        return z ^ (z >> np.uint64(31))


def cell_seed(base: int, *tags) -> int:
    """Stabiler Seed einer Zelle aus dem Basis-Seed und beliebig vielen Marken (Zahlen oder Texte)."""
    value = int(base)
    for tag in tags:
        part = zlib.crc32(tag.encode("utf-8")) if isinstance(tag, str) else int(round(float(tag) * 1000))
        value = (value * 1_000_003 + part) % (1 << 61)
    return value


class Stream:
    """Zufallsstrom mit Position: jeder Aufruf zieht die nächsten Werte des SplitMix64-Stroms zum Seed (als Block)."""

    def __init__(self, seed: int):
        self.seed = int(seed) & MASK
        self.pos = 0

    def words(self, count: int) -> np.ndarray:
        w = mix_words(self.seed, self.pos, count)
        self.pos += count
        return w

    def uniform(self, count: int) -> np.ndarray:
        """Gleichverteilt in [0, 1), 53 Bit."""
        return (self.words(count) >> np.uint64(11)).astype(np.float64) * TWO_M53

    def uniform_open(self, count: int) -> np.ndarray:
        """Gleichverteilt in (0, 1), nie 0 oder 1 (für Logarithmen und Umkehrfunktionen)."""
        return ((self.words(count) >> np.uint64(11)).astype(np.float64) + 0.5) * TWO_M53

    def normal(self, count: int) -> np.ndarray:
        """Standardnormal nach Box-Muller: aus zwei Strom-Werten (u1, u2) die zwei Werte r·cos(2π u2), r·sin(2π u2), r = √(−2 ln u1). Bei ungerader Zahl entfällt der letzte."""
        pairs = (count + 1) // 2
        w = self.words(2 * pairs)
        u1 = ((w[0::2] >> np.uint64(11)).astype(np.float64) + 0.5) * TWO_M53
        u2 = ((w[1::2] >> np.uint64(11)).astype(np.float64) + 0.5) * TWO_M53
        r = np.sqrt(-2.0 * np.log(u1))
        out = np.empty(2 * pairs)
        out[0::2] = r * np.cos(2.0 * math.pi * u2)
        out[1::2] = r * np.sin(2.0 * math.pi * u2)
        return out[:count]


def poisson_table(lam: float) -> np.ndarray:
    """Verteilungsfunktion F(0), F(1), … der Poisson-Verteilung bis weit hinter den Rand; letzter Eintrag exakt 1."""
    kmax = int(lam + 12.0 * math.sqrt(lam) + 40)
    pmf = np.empty(kmax + 1)
    pmf[0] = math.exp(-lam)
    for k in range(1, kmax + 1):
        pmf[k] = pmf[k - 1] * lam / k
    cdf = np.cumsum(pmf)
    cdf[-1] = 1.0
    return cdf


def draw(dist: str, stream: Stream, count: int, sigma: float = 1.0, lam: float = 1.0) -> np.ndarray:
    """`count` Zufallswerte der Verteilung in Standardform: Normal(0; 1), Laplace(0; 1), Logistisch(0; 1), Gleichverteilung(−1; 1), t(3), t(5), Exponential(1),
    Lognormal(0; sigma) mit Median 1, Pareto(2,5) ab 1 (Skalenparameter 1, Formparameter 2,5), Poisson(lam)."""
    if dist == "Normal":
        return stream.normal(count)
    if dist == "Laplace":
        d = stream.uniform_open(count) - 0.5
        return -np.sign(d) * np.log(1.0 - 2.0 * np.abs(d))
    if dist == "Logistisch":
        u = stream.uniform_open(count)
        return np.log(u / (1.0 - u))
    if dist == "Gleichverteilung":
        return 2.0 * stream.uniform_open(count) - 1.0
    if dist in ("t(3)", "t(5)"):
        nu = 3 if dist == "t(3)" else 5
        z = stream.normal(count * (nu + 1)).reshape(count, nu + 1)
        return z[:, 0] / np.sqrt((z[:, 1:] ** 2).sum(axis=1) / nu)
    if dist == "Exponential":
        return -np.log(stream.uniform_open(count))
    if dist == "Lognormal":
        return np.exp(sigma * stream.normal(count))
    if dist == "Pareto(2,5)":
        return stream.uniform_open(count) ** (-1.0 / 2.5)
    if dist == "Poisson":
        return np.searchsorted(poisson_table(lam), stream.uniform(count), side="right").astype(np.float64)
    raise ValueError(f"unbekannte Verteilung: {dist}")


def rounded(x, h: float):
    """Auf das Raster h gerundet (Wert in der Mitte zweier Rasterpunkte wird aufgerundet); h = 0 lässt die Werte unverändert."""
    x = np.asarray(x, dtype=np.float64)
    if h <= 0:
        return x
    return np.floor(x / h + 0.5) * h


# --- Reihe für den Reiter „Rechnen“ -------------------------------------------------------------------------------------------------------

def make_series(n: int, ratio: float, dist: str, order: str, seed: int, fmt: str) -> np.ndarray:
    """n Werte mit Mittel `ratio` und Streuung 1 (also μ/σ = ratio), gespeichert im Zahlenformat `fmt`, dann nach `order` angeordnet.

    Normal: μ + Z. Lognormal: lognormal(0; 1), auf Mittel μ und Streuung 1 gebracht (positiv schief). Die Rundung auf das Zahlenformat geschieht VOR dem Ordnen
    und gehört zu den Daten: Fehler werden später gegen das exakte Mittel der gespeicherten Werte gemessen, nicht gegen μ."""
    if dist not in SERIES_DISTRIBUTIONS:
        raise ValueError(f"unbekannte Verteilung der Reihe: {dist}")
    if order not in ORDERS:
        raise ValueError(f"unbekannte Reihenfolge: {order}")
    stream = Stream(seed)
    if dist == "Normal":
        raw = ratio + stream.normal(n)
    else:
        s = LOGNORMAL_SERIES_SIGMA
        mean, sd = math.exp(s * s / 2.0), math.sqrt((math.exp(s * s) - 1.0) * math.exp(s * s))
        raw = ratio + (np.exp(s * stream.normal(n)) - mean) / sd
    x = raw.astype(FORMATS[fmt])
    if order == "aufsteigend":
        x = np.sort(x)
    elif order == "absteigend":
        x = np.sort(x)[::-1].copy()
    return x


# --- Kunden in der Ebene, Fehlbuchung ----------------------------------------------------------------------------------------------------

def make_plane(n: int, seed: int) -> np.ndarray:
    """n Kunden, normalverteilt um den Ursprung (Einheit 1 km), Form (n, 2)."""
    return Stream(seed).normal(2 * n).reshape(n, 2)


def with_miscount(points: np.ndarray, distance: float) -> np.ndarray:
    """Hängt einen Fehlbuchungs-Kunden an, der `distance` vom Ursprung entfernt auf der Winkelhalbierenden liegt."""
    extra = np.array([[distance / math.sqrt(2.0), distance / math.sqrt(2.0)]])
    return np.vstack([points, extra])


# --- Lager-Mittel für James-Stein ---------------------------------------------------------------------------------------------------------

def warehouse_truth(d: int, norm: float, mode: str) -> np.ndarray:
    """Wahre Nachfragemittel von d Lagern mit Abstand `norm` vom Nullpunkt: alles auf einem Lager („eine“) oder gleichmäßig verteilt („gleich“)."""
    if mode == "eine":
        mu = np.zeros(d)
        mu[0] = norm
        return mu
    if mode == "gleich":
        return np.full(d, norm / math.sqrt(d))
    raise ValueError(f"unbekannte Verteilung der Mittel: {mode}")


def warehouse_observations(d: int, norm: float, mode: str, runs: int, stream: Stream) -> tuple[np.ndarray, np.ndarray]:
    """(mu, X): `runs` Beobachtungen X = mu + Z mit bekannter Streuung 1 je Lager, Form (runs, d)."""
    mu = warehouse_truth(d, norm, mode)
    return mu, mu + stream.normal(runs * d).reshape(runs, d)
