"""Vorgerechnete Messreihen (data/mtw_results.json, erzeugt von tools/make_results.py aus mtw_experiments.py): Laden und Abfragen.

Die Datei trägt die Prüfsumme der Quelldateien; `is_current()` vergleicht sie mit dem heutigen Stand (tests/test_results.py)."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import mtw_constants as C

PATH = Path(os.environ.get("MTW_RESULTS_PATH") or Path(__file__).resolve().parent / "data" / C.RESULTS_FILE)

SHORT = {"float32": "f32", "float64": "f64"}


@functools.lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def meta() -> dict:
    return load()["meta"]


def is_current() -> bool:
    import mtw_experiments as X
    return meta()["erzeuger_hash"] == X.source_hash()


# --- Rechnen --------------------------------------------------------------------------------------------------------------------------------

def e1(fmt: str) -> dict:
    return load()[f"e1_{SHORT[fmt]}"]


def e1_cell(fmt: str, method: str, n: int) -> dict:
    """Betrag des Fehlers des Mittels: {"mean", "std", "per_seed", …} für ein Verfahren (oder "floor_sum", "floor_mean") bei n Werten."""
    return e1(fmt)["methods"][method][str(n)]


def e1_mechanism() -> dict:
    return load()["e1_mech"]


def e1_diagnosis() -> dict:
    return load()["e1_diag"]


def e1_neumaier_share(n: int) -> dict:
    (row,) = [r for r in load()["e1_share"]["rows"] if r["n"] == n]
    return row


def e2_ratio(fmt: str) -> dict:
    return load()[f"e2_ratio_{SHORT[fmt]}"]


def e2_ratio_cell(fmt: str, ratio: float, method: str) -> dict:
    return e2_ratio(fmt)["ratios"][f"{ratio:g}"]["methods"][method]


def e2_order(fmt: str) -> dict:
    return load()[f"e2_order_{SHORT[fmt]}"]


def e2_order_cell(fmt: str, dist: str, order: str, method: str) -> dict:
    return e2_order(fmt)["cells"][dist][order][method]


def e2_accumulator(mu: float, n: int) -> dict:
    (row,) = [c for c in load()["e2_accu"]["cells"] if c["mu"] == mu and c["n"] == n]
    return row


# --- Wählen ---------------------------------------------------------------------------------------------------------------------------------

def e3() -> dict:
    return load()["e3"]


def sensitivity(n_total: int, t: float) -> dict:
    return e3()["sensitivity"]["n_totals"][str(n_total)][f"{t:g}"]


def breakdown_row(n: int) -> dict:
    (row,) = [r for r in e3()["breakdown"]["rows"] if r["n"] == n]
    return row


def contamination_row(M: float, eps: float) -> dict:
    (row,) = [r for r in e3()["contamination"]["rows"] if r["M"] == M and r["eps"] == eps]
    return row


def bias_variance_row(theta: float) -> dict:
    (row,) = [r for r in e3()["bias_variance"]["rows"] if r["theta"] == theta]
    return row


def efficiency(n: int) -> float:
    eff = e3()["efficiency"]
    return eff["odd"][str(n)] if str(n) in eff["odd"] else eff["even"][str(n)]


def stein_cell(mode: str, d: int, norm: float) -> dict:
    return load()["e5"]["cells"][f"{mode}|{d}|{norm:g}"]


def stein_zero_share(d: int) -> float:
    return load()["e5"]["zero_share"][str(d)]


def shape_row(sigma: float) -> dict:
    return load()["e6"]["shapes"][f"{sigma:g}"]


def target_row(sigma: float) -> dict:
    return load()["e6"]["targets"]["rows"][f"{sigma:g}"]


def delivery_row(sigma: float) -> dict:
    return load()["e6"]["delivery"]["rows"][f"{sigma:g}"]


def poisson_cell(lam: float, n: int) -> dict:
    return load()["e7"]["poisson"][f"{lam:g}|{n}"]


def near_tie(n: int) -> dict:
    return load()["e7"]["near_tie"][str(n)]


def interval_cell(dist: str, n: int) -> dict:
    return load()["e7"]["intervals"][f"{dist}|{n}"]


def rounding_row(h: float) -> dict:
    return load()["e7"]["rounding"]["rows"][f"{h:g}"]


def lognormal_exact() -> dict:
    return load()["e8"]["lognormal_exact"]


def normal_expectiles() -> list[dict]:
    return load()["e8"]["normal_exact"]


def gap_cell(label: str, tau: float) -> dict:
    return load()["e8"]["gaps"][label][f"{tau:g}"]


def gap_labels() -> list[str]:
    return list(load()["e8"]["gaps"])
