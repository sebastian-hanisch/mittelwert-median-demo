"""Messreihen der Experimente E1 bis E8: reine Funktionen über feste Zufalls-Seeds. Sie erzeugen sowohl die vorgerechneten Tabellen (tools/make_results.py → data/mtw_results.json)
als auch die Live-Knöpfe der App. Jede Zelle nutzt vier feste Zufalls-Seeds (SEEDS); angegeben wird der Mittelwert mit der Standardabweichung über die Seeds.

Zufall: `Stream(cell_seed(seed, Marken …))` aus mtw_scenario.py (SplitMix64); ein Seed gehört zu einer Zelle, nie zu einer Instanz allein. Die Raster der Messreihen
(Größen, Verhältnisse, Stufen) stehen hier, damit die Prüfsumme der Quelldateien (`source_hash`) sie mit abdeckt."""

from __future__ import annotations

import hashlib
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

import mtw_estimators as est
import mtw_stein as stein
import mtw_sums as sums
import mtw_theory as theory
from mtw_scenario import DISTRIBUTIONS, FORMATS, ORDERS, SERIES_DISTRIBUTIONS, Stream, cell_seed, draw, make_series, rounded, warehouse_observations  # noqa: F401

ROOT = Path(__file__).resolve().parent
SOURCES = ("mtw_rng.py", "mtw_scenario.py", "mtw_sums.py", "mtw_estimators.py", "mtw_stein.py", "mtw_theory.py", "mtw_experiments.py", "tools/make_results.py")

SEEDS = (11, 12, 13, 14)
STAND = "2026-10-25"

METHODS = ("naiv", "numpy.sum", "Binärbaum", "verschoben", "Kahan", "Kahan mit Rest", "Neumaier", "Neumaier mit Rest", "Welford")
E1_CHECKS = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
E1_RATIO = 1000.0
E2_N = 1_000_000
NEUMAIER_SEEDS = tuple(range(101, 121))      # zwanzig weitere Reihen für die Frage „wie oft ist Neumaier schlechter als Kahan?“
NEUMAIER_CHECKS = (1_000_000, 2_000_000, 5_000_000)
E2_RATIOS = {"float32": (0.0, 1.0, 1e2, 1e4, 1e6, 1e7), "float64": (0.0, 1.0, 1e2, 1e4, 1e6, 1e8, 1e10)}
E2_ACCU = ((1000.0, (1_000, 10_000, 100_000, 1_000_000, 10_000_000)), (1e4, (1_000, 10_000, 100_000, 1_000_000)), (1e6, (100_000, 1_000_000)))
STEIN_DIMS = (3, 5, 10, 50)
STEIN_NORMS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
STEIN_RUNS = 50_000
SC_TS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0, 1e6)
SC_RUNS = 20_000
BREAKDOWN_NS = tuple(range(3, 13))
CONTAMINATION_EPS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.48, 0.49, 0.50, 0.55, 0.60)
CONTAMINATION_M = (10.0, 100.0, 1000.0)
THETAS = (0.0, 0.5, 1.0, 2.0, 4.0)
EFFICIENCY_ODD = (3, 5, 7, 9, 11, 21, 51, 101, 301, 1001)
EFFICIENCY_EVEN = (2, 4, 6, 8, 10, 20, 100, 1000)
SIGMAS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
TARGET_SIGMAS = (0.5, 1.0, 1.5, 2.0)
TARGET_N, TARGET_RUNS = 200, 5000
DELIVERY_MEDIAN = 4.0
DELIVERY_SIGMAS = (0.4, 0.8, 1.2)
DELIVERY_ORDERS, DELIVERY_MONTHS, DELIVERY_SAMPLE = 1000, 2000, 50
TIE_LAMBDAS = (0.7, 2.0, 10.0)
ROUND_STEPS = (0.0, 0.25, 0.5, 1.0, 2.0)
ROUND_MU = 0.3
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
GAP_TAUS = (0.5, 0.8, 0.9, 0.95, 0.99)
GAP_N = 1_000_000
GAP_DISTRIBUTIONS = (("Normal", "Normal", 1.0), ("Gleichverteilung", "Gleichverteilung", 1.0), ("Exponential", "Exponential", 1.0), ("Lognormal(0; 0,5)", "Lognormal", 0.5),
                     ("Lognormal(0; 1)", "Lognormal", 1.0), ("Lognormal(0; 1,5)", "Lognormal", 1.5), ("t(3)", "t(3)", 1.0), ("Pareto(2,5)", "Pareto(2,5)", 1.0))


def source_hash() -> str:
    """Prüfsumme der Quelldateien der Messreihen (Zeilenenden normalisiert): stimmt sie nicht mit der in der JSON-Datei überein, ist die Datei veraltet."""
    h = hashlib.sha256()
    for name in SOURCES:
        text = (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        h.update(name.encode("utf-8") + b"\0" + text.encode("utf-8") + b"\0")
    return h.hexdigest()[:16]


def mean_std(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std())


def summary(values) -> dict:
    m, s = mean_std(values)
    return {"mean": m, "std": s, "per_seed": [float(v) for v in values]}


# ---------------------------------------------------------------- E1/E2: Rechnen

def measure_methods(x, fmt: str, checks) -> dict:
    """Fehler des Mittels (Ergebnis minus exaktes Mittel der gespeicherten Werte, vorzeichenbehaftet) für alle Verfahren und jede Länge in `checks` (aufsteigend, Präfixe
    derselben Reihe x). Summen werden exakt durch n geteilt (gemessen wird das Summationsverfahren); verschoben und Welford liefern das Mittel im Zahlenformat.
    Dazu die Untergrenze („floor_sum“, „floor_mean“): das exakte Ergebnis auf das Zahlenformat gerundet."""
    dt = FORMATS[fmt]
    x = np.asarray(x, dtype=dt)
    x64 = x.astype(np.float64)
    prefix = sums.naive_prefix(x, dt)
    out = {name: {} for name in ("floor_sum", "floor_mean") + METHODS}
    kahan = neumaier = welford = None
    prev = 0
    for n in sorted(checks):
        seg = x[prev:n]
        kahan = sums.kahan_sum(seg, dt, kahan)
        neumaier = sums.neumaier_sum(seg, dt, neumaier)
        welford = sums.welford_mean(seg, dt, welford)
        prev = n
        exact = sums.exact_mean(x64[:n])
        out["floor_sum"][n] = sums.floor_of_format(exact * n, dt) / n
        out["floor_mean"][n] = sums.floor_of_format(exact, dt)
        out["naiv"][n] = sums.sum_error_as_mean(prefix[n - 1], n, exact)
        out["numpy.sum"][n] = sums.sum_error_as_mean(sums.numpy_sum(x[:n], dt), n, exact)
        out["Binärbaum"][n] = sums.sum_error_as_mean(sums.pairwise_sum(x[:n], dt), n, exact)
        out["verschoben"][n] = sums.mean_error(sums.shifted_mean(x[:n], dt), exact)
        out["Kahan"][n] = sums.sum_error_as_mean(kahan[0], n, exact)
        out["Kahan mit Rest"][n] = sums.sum_error_as_mean(float(kahan[0]) - float(kahan[1]), n, exact)
        out["Neumaier"][n] = sums.sum_error_as_mean(dt(float(neumaier[0]) + float(neumaier[1])), n, exact)
        out["Neumaier mit Rest"][n] = sums.sum_error_as_mean(float(neumaier[0]) + float(neumaier[1]), n, exact)
        out["Welford"][n] = sums.mean_error(welford[0], exact)
    return out


def _abs_summary(per_seed_errors, scale: float = 1.0) -> dict:
    """Betrag des Fehlers: Mittel und Std über die Seeds (in Einheiten von `scale`) und die vorzeichenbehafteten Fehler je Seed."""
    a = np.abs(np.asarray(per_seed_errors, dtype=float))
    return {"mean": float(a.mean() / scale), "std": float(a.std() / scale), "mean_abs_value": float(a.mean()), "per_seed": [float(v) for v in per_seed_errors]}


def run_e1(fmt: str, checks=E1_CHECKS, seeds=SEEDS) -> dict:
    """E1: Fehler des Mittels über n (Werte N(1000; 1), zufällige Reihenfolge), alle Verfahren, je Seed eine Reihe mit Präfixen."""
    rows: dict = {}
    for seed in seeds:
        x = make_series(max(checks), E1_RATIO, "Normal", "zufällig", cell_seed(seed, "e1", fmt), fmt)
        res = measure_methods(x, fmt, checks)
        for name, per_n in res.items():
            for n, e in per_n.items():
                rows.setdefault(name, {}).setdefault(str(n), []).append(e)
    return {"fmt": fmt, "checks": list(checks), "ratio": E1_RATIO,
            "methods": {name: {n: _abs_summary(v, scale=1.0) for n, v in per_n.items()} for name, per_n in rows.items()}}


def run_e1_mechanism(seeds=SEEDS, n: int = 10_000_000) -> dict:
    """E1 Mechanismus: ab welchem Wert die fortlaufende float32-Summe 2²⁹ erreicht, wie oft danach der Zuwachs genau 1024 ist, vorhergesagtes gegen gemessenes Endmittel."""
    cells = []
    for seed in seeds:
        x = make_series(n, E1_RATIO, "Normal", "zufällig", cell_seed(seed, "e1", "float32"), "float32")
        cs = sums.naive_prefix(x, np.float32)
        inc = np.diff(cs.astype(np.float64), prepend=0.0)
        n0 = int(np.argmax(cs >= np.float32(2.0 ** 29)))
        after = inc[n0 + 1:]
        predicted = (float(cs[n0]) + 1024.0 * (n - 1 - n0)) / n
        cells.append({"seed": seed, "first_index": n0 + 1, "share_1024": float((after == 1024.0).mean()), "predicted_mean": predicted, "measured_mean": float(cs[-1]) / n,
                      "increment_min": float(after.min()), "increment_max": float(after.max())})
    return {"n": n, "cells": cells}


def run_e1_diagnosis(seed: int = SEEDS[0], checks=(10_000, 100_000, 1_000_000)) -> dict:
    """E1 Diagnose: wächst der Rest c, und liegt der Neumaier-Fehler daran, dass c selbst in float32 summiert wird? Eine Reihe (float32, N(1000; 1)), Fehler der SUMME."""
    x = make_series(max(checks), E1_RATIO, "Normal", "zufällig", cell_seed(seed, "e1", "float32"), "float32")
    x64 = x.astype(np.float64)
    rows = []
    for n in checks:
        total = sums.exact_sum(x64[:n])
        cell = {"n": n}
        for key, method in (("kahan", "kahan"), ("neumaier", "neumaier"), ("neumaier_wide", "neumaier_wide")):
            s, c, biggest = sums.rest_trace(x[:n], np.float32, method)
            result = float(s) - float(c) if method == "kahan" else float(s) + float(c)
            cell[key] = {"max_rest": biggest, "sum_error": float(Fraction(result) - total)}
        rows.append(cell)
    return {"seed": seed, "rows": rows}


def run_e1_neumaier_share(seeds=NEUMAIER_SEEDS, checks=NEUMAIER_CHECKS) -> dict:
    """E1: In wie vielen von `seeds` Reihen (float32, N(1000; 1)) ist Neumaier bei n Werten schlechter als Kahan (Fehler mehr als doppelt so groß), in wie vielen gleich gut?
    Fehler der Summe, exakt durch n geteilt; Kahan = Summe s, Neumaier = s + c im Zahlenformat."""
    errors = {n: [] for n in checks}
    for seed in seeds:
        x = make_series(max(checks), E1_RATIO, "Normal", "zufällig", cell_seed(seed, "e1share"), "float32")
        x64 = x.astype(np.float64)
        kahan = neumaier = None
        prev = 0
        for n in sorted(checks):
            seg = x[prev:n]
            kahan = sums.kahan_sum(seg, np.float32, kahan)
            neumaier = sums.neumaier_sum(seg, np.float32, neumaier)
            prev = n
            exact = sums.exact_mean(x64[:n])
            errors[n].append((abs(sums.sum_error_as_mean(kahan[0], n, exact)), abs(sums.sum_error_as_mean(np.float32(float(neumaier[0]) + float(neumaier[1])), n, exact))))
    rows = []
    for n in sorted(checks):
        k = np.array([e[0] for e in errors[n]])
        m = np.array([e[1] for e in errors[n]])
        rows.append({"n": n, "series": len(seeds), "worse": int((m > 2.0 * k).sum()), "equal": int((m == k).sum()), "better": int((m < k).sum()),
                     "median_kahan": float(np.median(k)), "median_neumaier": float(np.median(m)), "max_ratio": float((m / np.maximum(k, 1e-300)).max())})
    return {"rows": rows}


def run_e2_ratio(fmt: str, seeds=SEEDS, n: int = E2_N) -> dict:
    """E2 Auslöschung: Fehler in Standardfehlern σ/√n bei wachsendem Verhältnis μ/σ (n = 10⁶, Normalwerte)."""
    se = 1.0 / math.sqrt(n)
    ratios = {}
    for ratio in E2_RATIOS[fmt]:
        rows: dict = {}
        distinct = []
        for seed in seeds:
            x = make_series(n, ratio, "Normal", "zufällig", cell_seed(seed, "e2", fmt, ratio), fmt)
            distinct.append(len(np.unique(x[:100_000])))
            for name, per_n in measure_methods(x, fmt, [n]).items():
                rows.setdefault(name, []).append(per_n[n])
        ratios[f"{ratio:g}"] = {"ratio": ratio, "distinct_first_1e5": float(np.mean(distinct)),
                                "methods": {name: _abs_summary(v, scale=se) | {"mean_abs_value": float(np.abs(v).mean())} for name, v in rows.items()}}
    return {"fmt": fmt, "n": n, "se": se, "ratios": ratios}


def run_e2_order(fmt: str, seeds=SEEDS, n: int = E2_N) -> dict:
    """E2 Reihenfolge: dieselben Werte zufällig, aufsteigend, absteigend; Normal (μ/σ = 1000) und schiefe lognormale Reihe (Mittel 1000, Streuung 1)."""
    se = 1.0 / math.sqrt(n)
    out = {}
    for dist in SERIES_DISTRIBUTIONS:
        out[dist] = {}
        for order in ORDERS:
            rows: dict = {}
            for seed in seeds:
                x = make_series(n, E1_RATIO, dist, order, cell_seed(seed, "e2o", fmt, dist), fmt)
                for name, per_n in measure_methods(x, fmt, [n]).items():
                    rows.setdefault(name, []).append(per_n[n])
            out[dist][order] = {name: _abs_summary(v, scale=se) | {"mean_abs_value": float(np.abs(v).mean())} for name, v in rows.items()}
    return {"fmt": fmt, "n": n, "se": se, "cells": out}


def run_e2_accumulator(seeds=SEEDS, grid=E2_ACCU) -> dict:
    """E2 Gegenmittel: `numpy.mean` in float32 gegen float64-Akkumulator (danach auf float32 gerundet) und gegen die Untergrenze, in Standardfehlern 1/√n."""
    cells = []
    for mu, ns in grid:
        for n in ns:
            e32, e64, e64r, floor = [], [], [], []
            for seed in seeds:
                x = make_series(n, mu, "Normal", "zufällig", cell_seed(seed, "e2a", mu), "float32")
                exact = sums.exact_mean(x.astype(np.float64))
                e32.append(sums.mean_error(np.mean(x), exact))
                e64.append(sums.mean_error(np.mean(x, dtype=np.float64), exact))
                e64r.append(sums.mean_error(sums.mean_with_float64_accumulator(x, np.float32), exact))
                floor.append(sums.floor_of_format(exact, np.float32))
            se = 1.0 / math.sqrt(n)
            cells.append({"mu": mu, "n": n, "se": se, "float32": _abs_summary(e32, se), "float64_accumulator": _abs_summary(e64, se),
                          "float64_accumulator_rounded": _abs_summary(e64r, se), "floor": _abs_summary(floor, se)})
    return {"cells": cells}


# ---------------------------------------------------------------- E3: Einfluss, Durchschlag, Bias-Varianz

def sensitivity_cell(n_total: int, ts=SC_TS, runs: int = SC_RUNS, seed: int = SEEDS[0]) -> dict:
    """Einflusskurve SC(t) = n·(T(x_1…x_{n−1}, t) − T(x_1…x_{n−1})) bei insgesamt n Werten, Basis N(0; 1), Mittel über `runs` Stichproben."""
    stream = Stream(cell_seed(seed, "e3sc", n_total))
    base = stream.normal(runs * (n_total - 1)).reshape(runs, n_total - 1)
    m0, a0 = est.median_rows(base), base.mean(axis=1)
    out = {}
    for t in ts:
        Y = np.concatenate([base, np.full((runs, 1), t)], axis=1)
        out[f"{t:g}"] = {"t": t, "median": float((n_total * (est.median_rows(Y) - m0)).mean()), "mean": float((n_total * (Y.mean(axis=1) - a0)).mean()),
                         "mean_dev": float(np.abs(n_total * (Y.mean(axis=1) - a0) - (t - a0)).max())}
    return out


def run_e3_sensitivity() -> dict:
    out = {}
    for n_total in (100, 101):
        per_seed = [sensitivity_cell(n_total, seed=s) for s in SEEDS]
        out[str(n_total)] = {key: {"t": per_seed[0][key]["t"], "median": summary([c[key]["median"] for c in per_seed]), "mean": summary([c[key]["mean"] for c in per_seed]),
                                   "mean_dev": max(c[key]["mean_dev"] for c in per_seed)} for key in per_seed[0]}
    return {"runs": SC_RUNS, "n_totals": out}


def breakdown_cell(n: int, seed: int) -> dict:
    """Durchschlagspunkt des Medians einer N(0; 1)-Stichprobe der Größe n durch Aufzählung, größte Verschiebung davor (M = 10⁶), Durchschlag des Mittels (ein Wert genügt)."""
    x = Stream(cell_seed(seed, "e3b", n)).normal(n)
    k = est.breakdown_k(x)
    below = est.max_shift_below_breakdown(x, k - 1, 1e6) if k and k > 1 else 0.0
    y = x.copy()
    y[0] = 1e6
    mean_k = 1 if abs(y.mean() - x.mean()) > 1e3 else None
    return {"n": n, "seed": seed, "k": k, "max_shift_below": below, "mean_k": mean_k}


def run_e3_breakdown() -> dict:
    rows = []
    for n in BREAKDOWN_NS:
        cells = [breakdown_cell(n, s) for s in SEEDS]
        rows.append({"n": n, "k": [c["k"] for c in cells], "ceil_half": math.ceil(n / 2), "mean_k": [c["mean_k"] for c in cells],
                     "max_shift_below": summary([c["max_shift_below"] for c in cells])})
    return {"rows": rows}


def contamination_cell(eps: float, M: float, runs: int, seed: int, n: int = 100) -> dict:
    """Mittlere Verschiebung |T(verschmutzt) − T(sauber)| von Mittel und Median, wenn round(eps·n) zufällig gewählte Werte einer N(0; 1)-Stichprobe durch M ersetzt werden."""
    stream = Stream(cell_seed(seed, "e3c", eps, M))
    X = stream.normal(runs * n).reshape(runs, n)
    k = int(round(eps * n))
    m0, a0 = est.median_rows(X), X.mean(axis=1)
    Y = X.copy()
    if k:
        idx = np.argsort(stream.uniform(runs * n).reshape(runs, n), axis=1)[:, :k]
        np.put_along_axis(Y, idx, M, axis=1)
    return {"k": k, "mean": float(np.abs(Y.mean(axis=1) - a0).mean()), "median": float(np.abs(est.median_rows(Y) - m0).mean())}


def run_e3_contamination(runs: int = 4000) -> dict:
    rows = []
    for M in CONTAMINATION_M:
        for eps in CONTAMINATION_EPS:
            cells = [contamination_cell(eps, M, runs, s) for s in SEEDS]
            rows.append({"M": M, "eps": eps, "k": cells[0]["k"], "mean": summary([c["mean"] for c in cells]), "median": summary([c["median"] for c in cells])})
    return {"n": 100, "runs": runs, "rows": rows}


def bias_variance_cell(theta: float, c: float, runs: int, seed: int, n: int = 10) -> float:
    """MSE des verkleinerten Mittels c·x̄ (n Werte, σ = 1, μ = θ/√n), geteilt durch den MSE des Mittels 1/n, aus `runs` Läufen."""
    mu = theta / math.sqrt(n)
    xbar = mu + Stream(cell_seed(seed, "e3bv", theta, c)).normal(runs) / math.sqrt(n)
    return float(((c * xbar - mu) ** 2).mean() * n)


def run_e3_bias_variance(runs: int = 200_000, n: int = 10) -> dict:
    rows = []
    for theta in THETAS:
        mu = theta / math.sqrt(n)
        cstar = mu * mu / (mu * mu + 1.0 / n)
        cells = {}
        for label, c in (("0,9", 0.9), ("0,8", 0.8), ("0,5", 0.5), ("c*", cstar)):
            formula = est.shrunk_mse(c, mu, 1.0, n) * n
            cells[label] = {"c": c, "formula": formula, "sim": summary([bias_variance_cell(theta, c, runs, s, n) for s in SEEDS])}
        rows.append({"theta": theta, "mu": mu, "cstar": cstar, "cells": cells})
    return {"n": n, "runs": runs, "rows": rows}


def run_e3_efficiency() -> dict:
    """Var(Mittel)/Var(Median) bei Normalverteilung, exakt (Quadratur): ungerades n, gerades n (Sägezahn), Grenzwert 2/π."""
    return {"odd": {str(n): theory.efficiency_normal(n) for n in EFFICIENCY_ODD}, "even": {str(n): theory.efficiency_normal(n) for n in EFFICIENCY_EVEN},
            "limit": theory.EFFICIENCY_LIMIT, "plateau": theory.INFLUENCE_PLATEAU}


def run_e3() -> dict:
    return {"sensitivity": run_e3_sensitivity(), "breakdown": run_e3_breakdown(), "contamination": run_e3_contamination(), "bias_variance": run_e3_bias_variance(),
            "efficiency": run_e3_efficiency()}


# ---------------------------------------------------------------- E5: James-Stein

def stein_cell(d: int, norm: float, mode: str, seed: int, runs: int = STEIN_RUNS) -> dict:
    """Risiken je Lager (Summe geteilt durch d) für Einzelmittel, James-Stein, Positive Part und die Schrumpfung zum gemeinsamen Mittel, dazu das Risiko des ersten Lagers allein."""
    stream = Stream(cell_seed(seed, "e5", d, mode, norm))
    mu, X = warehouse_observations(d, norm, mode, runs, stream)
    r = stein.risk_summary(mu, X)
    out = {"einzel": r["einzel"] / d, "js": r["js"] / d, "pp": r["pp"] / d, "koordinate1_js": r["koordinate1_js"], "koordinate1_pp": r["koordinate1_pp"]}
    if "gemeinsames_mittel" in r:
        out["gemeinsames_mittel"] = r["gemeinsames_mittel"] / d
    return out


def zero_share(d: int, runs: int = 200_000, seed: int = 99) -> float:
    """Anteil der Läufe bei μ = 0, in denen der Positive-Part-Faktor auf null steht (alle Mittel auf den Nullpunkt gezogen)."""
    X = Stream(cell_seed(seed, "e5z", d)).normal(runs * d).reshape(runs, d)
    return float((stein.james_stein_factor(X) < 0).mean())


def run_e5() -> dict:
    cells = {}
    for mode in ("eine", "gleich"):
        for d in STEIN_DIMS:
            for norm in STEIN_NORMS:
                per_seed = [stein_cell(d, norm, mode, s) for s in SEEDS]
                entry = {key: summary([c[key] for c in per_seed]) for key in per_seed[0]}
                entry["js_exact"] = stein.exact_js_risk(d, norm * norm) / d
                cells[f"{mode}|{d}|{norm:g}"] = entry
    return {"runs": STEIN_RUNS, "cells": cells, "zero_share": {str(d): zero_share(d) for d in STEIN_DIMS}}


# ---------------------------------------------------------------- E6: Welche Größe will ich?

def target_errors(sigma: float, n: int, runs: int, seed: int) -> dict:
    """n Werte aus Lognormal(0; σ) (Median 1, Mittelwert e^(σ²/2)): relativer Fehler |T − Ziel|/Ziel von Mittel und Median gegen Mittelwert und Median, `runs` Stichproben.
    Gibt je Kombination den typischen (Median) und den 95-%-Fehler zurück."""
    X = np.exp(sigma * Stream(cell_seed(seed, "e6t", sigma, n)).normal(runs * n)).reshape(runs, n)
    mean_t, median_t = theory.gap_mean_median(sigma), 1.0
    a, d = X.mean(axis=1), est.median_rows(X)
    out = {}
    for key, T, target in (("mittel_mittelwert", a, mean_t), ("median_mittelwert", d, mean_t), ("mittel_median", a, median_t), ("median_median", d, median_t)):
        e = np.abs(T - target) / target
        out[key] = {"typisch": float(np.median(e)), "q95": float(np.quantile(e, 0.95))}
    return out


def shape_cell(sigma: float, seed: int, count: int = 500_000) -> dict:
    """Anteil unterhalb des Mittelwerts und Anteil der Gesamtsumme in den größten 10 % einer Lognormal(0; σ)-Stichprobe."""
    x = np.exp(sigma * Stream(cell_seed(seed, "e6s", sigma)).normal(count))
    xs = np.sort(x)
    return {"below_mean": float((x < theory.gap_mean_median(sigma)).mean()), "top10_share": float(xs[-count // 10:].sum() / xs.sum())}


def delivery_cell(sigma: float, seed: int) -> dict:
    """Lieferzeit-Szenario (Median 4 Tage, lognormal): Monate mit 1000 Aufträgen. Anteil der wahren Monatssumme, den der Plan „1000 × Median“ auslässt; mittlere Abweichung des
    Plans „1000 × Mittelwert“; Anteil der Aufträge höchstens so lang wie der Mittelwert; Schätzung der Monatssumme aus 50 Aufträgen mit Mittel und mit Median."""
    stream = Stream(cell_seed(seed, "e6d", sigma))
    mean_t = DELIVERY_MEDIAN * theory.gap_mean_median(sigma)
    months = DELIVERY_MONTHS
    X = DELIVERY_MEDIAN * np.exp(sigma * stream.normal(months * DELIVERY_ORDERS)).reshape(months, DELIVERY_ORDERS)
    total = X.sum(axis=1)
    small = DELIVERY_MEDIAN * np.exp(sigma * stream.normal(3000 * DELIVERY_SAMPLE)).reshape(3000, DELIVERY_SAMPLE)
    total2 = (DELIVERY_MEDIAN * np.exp(sigma * stream.normal(3000 * DELIVERY_ORDERS)).reshape(3000, DELIVERY_ORDERS)).sum(axis=1)
    out = {"plan_median_missing": float(((total - DELIVERY_ORDERS * DELIVERY_MEDIAN) / total).mean()),
           "plan_mean_deviation": float(np.abs((total - DELIVERY_ORDERS * mean_t) / total).mean()),
           "orders_below_mean": float((X <= mean_t).mean())}
    for key, T in (("mittel", DELIVERY_ORDERS * small.mean(axis=1)), ("median", DELIVERY_ORDERS * est.median_rows(small))):
        e = np.abs(T - total2) / total2
        out[f"schaetzung_{key}"] = {"typisch": float(np.median(e)), "q95": float(np.quantile(e, 0.95)), "zu_klein": float((T < total2).mean())}
    return out


def run_e6() -> dict:
    shapes = {}
    for sigma in SIGMAS:
        cells = [shape_cell(sigma, s) for s in SEEDS]
        shapes[f"{sigma:g}"] = {"sigma": sigma, "gap": theory.gap_mean_median(sigma), "skew": (math.exp(sigma ** 2) + 2) * math.sqrt(math.exp(sigma ** 2) - 1),
                                "below_mean_exact": theory.share_below_mean(sigma), "below_mean": summary([c["below_mean"] for c in cells]),
                                "top10_share": summary([c["top10_share"] for c in cells])}
    targets = {}
    for sigma in TARGET_SIGMAS:
        cells = [target_errors(sigma, TARGET_N, TARGET_RUNS, s) for s in SEEDS]
        targets[f"{sigma:g}"] = {"sigma": sigma, "miss_limit": theory.median_miss_share(sigma), "mean_gap_limit": theory.gap_mean_median(sigma) - 1.0,
                                 **{key: {"typisch": summary([c[key]["typisch"] for c in cells]), "q95": summary([c[key]["q95"] for c in cells])} for key in cells[0]}}
    delivery = {}
    for sigma in DELIVERY_SIGMAS:
        cells = [delivery_cell(sigma, s) for s in SEEDS]
        delivery[f"{sigma:g}"] = {
            "sigma": sigma, "mean": DELIVERY_MEDIAN * theory.gap_mean_median(sigma), "exact_missing": theory.median_miss_share(sigma),
            "plan_median_missing": summary([c["plan_median_missing"] for c in cells]), "plan_mean_deviation": summary([c["plan_mean_deviation"] for c in cells]),
            "orders_below_mean": summary([c["orders_below_mean"] for c in cells]),
            **{key: {k: summary([c[key][k] for c in cells]) for k in ("typisch", "q95", "zu_klein")} for key in ("schaetzung_mittel", "schaetzung_median")}}
    return {"shapes": shapes, "targets": {"n": TARGET_N, "runs": TARGET_RUNS, "rows": targets}, "delivery": {"median": DELIVERY_MEDIAN, "orders": DELIVERY_ORDERS, "months": DELIVERY_MONTHS,
                                                                                                          "sample": DELIVERY_SAMPLE, "rows": delivery}}


# ---------------------------------------------------------------- E7: Ties und Rundung

def poisson_median_cell(lam: float, n: int, runs: int, seed: int) -> dict:
    """Poisson(λ)-Stichproben der Größe n: Anteil mit nicht ganzzahligem Median (Mitte der beiden mittleren), mit Median 0, 0,5, 1 und ≥ 1,5, mittlerer unterer/oberer Median."""
    X = np.sort(draw("Poisson", Stream(cell_seed(seed, "e7p", lam, n)), runs * n, lam=lam).reshape(runs, n), axis=1)
    lo, hi = (X[:, n // 2], X[:, n // 2]) if n % 2 else (X[:, n // 2 - 1], X[:, n // 2])
    m = 0.5 * (lo + hi)
    return {"non_integer": float((m != np.round(m)).mean()), "is0": float((m == 0).mean()), "is_half": float((m == 0.5).mean()), "is1": float((m == 1).mean()),
            "ge15": float((m >= 1.5).mean()), "mean_lower": float(lo.mean()), "mean_upper": float(hi.mean()), "mean_mid": float(m.mean())}


def interval_cell(dist: str, n: int, runs: int, seed: int) -> dict:
    """Anteil der Läufe, in denen der Minimierer von Σ|x − c| ein echtes Intervall ist (gerades n), und mittlere Breite."""
    X = np.sort(draw(dist, Stream(cell_seed(seed, "e7i", dist, n)), runs * n, lam=2.0).reshape(runs, n), axis=1)
    w = X[:, n // 2] - X[:, n // 2 - 1] if n % 2 == 0 else np.zeros(runs)
    return {"share_interval": float((w > 0).mean()), "mean_width": float(w.mean())}


def rounding_cell(h: float, runs: int, seed: int, n: int = 101, mu: float = ROUND_MU) -> dict:
    """Normal(μ; 1) auf das Raster h gerundet, n Werte: RMSE von Mittel und Median gegen μ und Anteil der Läufe, in denen der Median genau 0 ist."""
    X = rounded(mu + Stream(cell_seed(seed, "e7r", h, mu)).normal(runs * n).reshape(runs, n), h)
    med = est.median_rows(X)
    return {"rmse_mean": float(np.sqrt(((X.mean(axis=1) - mu) ** 2).mean())), "rmse_median": float(np.sqrt(((med - mu) ** 2).mean())), "median_zero": float((med == 0).mean())}


def run_e7() -> dict:
    poisson = {}
    for lam in TIE_LAMBDAS:
        for n in (10, 11, 100, 101):
            cells = [poisson_median_cell(lam, n, 20_000, s) for s in SEEDS]
            poisson[f"{lam:g}|{n}"] = {k: summary([c[k] for c in cells]) for k in cells[0]}
    near_tie = {}
    for n, runs in ((11, 20_000), (100, 20_000), (101, 20_000), (1000, 5000), (10000, 1000)):
        cells = [poisson_median_cell(0.7, n, runs, s) for s in SEEDS]
        near_tie[str(n)] = {"runs": runs, **{k: summary([c[k] for c in cells]) for k in ("is0", "is_half", "is1", "ge15")}}
    intervals = {}
    for dist in ("Normal", "Poisson"):
        for n in (4, 10, 100):
            cells = [interval_cell(dist, n, 20_000, s) for s in SEEDS]
            intervals[f"{dist}|{n}"] = {k: summary([c[k] for c in cells]) for k in cells[0]}
    rounding = {}
    for h in ROUND_STEPS:
        cells = [rounding_cell(h, 40_000, s) for s in SEEDS]
        rounding[f"{h:g}"] = {"h": h, **{k: summary([c[k] for c in cells]) for k in cells[0]}}
    return {"poisson": poisson, "near_tie": near_tie, "intervals": intervals, "rounding": {"n": 101, "mu": ROUND_MU, "runs": 40_000, "rows": rounding},
            "f0": theory.poisson_cdf(0.7, 0)}


# ---------------------------------------------------------------- E8: Quantil gegen Expektile

def gap_cell(dist: str, sigma: float, seed: int, count: int = GAP_N) -> dict:
    """Quantil und Expektile gleicher Stufe aus einer großen Stichprobe (näherungsweise die Verteilung): je Stufe (q, e, F(e), Lücke (e − q)/Quartilsabstand)."""
    x = draw(dist, Stream(cell_seed(seed, "e8", dist, sigma)), count, sigma=sigma)
    xs = np.sort(x)
    iqr = float(xs[int(0.75 * count)] - xs[int(0.25 * count)])
    out = {}
    for tau in GAP_TAUS:
        q = float(xs[int(math.ceil(tau * count)) - 1])
        e = est.expectile(x, tau)
        out[f"{tau:g}"] = {"q": q, "e": e, "F_e": float(np.searchsorted(xs, e) / count), "gap": (e - q) / iqr}
    return out


def run_e8() -> dict:
    exact = []
    for tau in TAUS:
        q, e = theory.lognormal_quantile(0.5, tau), theory.lognormal_expectile(0.5, tau)
        exact.append({"tau": tau, "q": q, "e": e, "ratio": e / q, "F_e": theory.lognormal_cdf(e, 0.5)})
    normal = [{"tau": tau, "e": theory.normal_expectile(tau), "F_e": float(theory.norm_cdf(theory.normal_expectile(tau)))} for tau in TAUS]
    gaps = {}
    for label, dist, sigma in GAP_DISTRIBUTIONS:
        per_seed = [gap_cell(dist, sigma, s) for s in SEEDS]
        entry = {}
        for tau in GAP_TAUS:
            key = f"{tau:g}"
            g = summary([c[key]["gap"] for c in per_seed])
            entry[key] = {"q": float(np.mean([c[key]["q"] for c in per_seed])), "e": float(np.mean([c[key]["e"] for c in per_seed])),
                          "F_e": float(np.mean([c[key]["F_e"] for c in per_seed])), "gap": g, "usable": bool(2.0 * g["std"] < abs(g["mean"]))}
        gaps[label] = entry
    return {"lognormal_exact": {"sigma": 0.5, "median": 1.0, "mean": theory.gap_mean_median(0.5), "mean_cdf": theory.share_below_mean(0.5), "rows": exact},
            "normal_exact": normal, "gaps": gaps, "count": GAP_N}


PARTS = {
    "e1_f32": lambda: run_e1("float32"), "e1_f64": lambda: run_e1("float64"), "e1_mech": run_e1_mechanism, "e1_diag": run_e1_diagnosis, "e1_share": run_e1_neumaier_share,
    "e2_ratio_f32": lambda: run_e2_ratio("float32"), "e2_ratio_f64": lambda: run_e2_ratio("float64"),
    "e2_order_f32": lambda: run_e2_order("float32"), "e2_order_f64": lambda: run_e2_order("float64"), "e2_accu": run_e2_accumulator,
    "e3": run_e3, "e5": run_e5, "e6": run_e6, "e7": run_e7, "e8": run_e8,
}
