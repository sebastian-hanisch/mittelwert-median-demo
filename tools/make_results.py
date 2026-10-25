"""Erzeugt data/mtw_results.json (die vorgerechneten Messreihen der Experimente E1 bis E8) aus mtw_experiments.py.

    python tools/make_results.py --part e5          # eine Messreihe nach data/_parts/e5.json (läuft allein, mehrere Teile können parallel laufen)
    python tools/make_results.py --merge            # alle Teile zu data/mtw_results.json zusammenfügen, mit Prüfsumme der Quelldateien

Teile: e1_f32, e1_f64 (Fehler über n), e1_mech (Mechanismus), e1_diag (Neumaier-Diagnose), e2_ratio_f32/f64 (Auslöschung), e2_order_f32/f64 (Reihenfolge),
e2_accu (float64-Akkumulator), e3 (Einfluss, Durchschlag, Bias-Varianz, Effizienz), e5 (James-Stein), e6 (Zielgröße), e7 (Ties und Rundung), e8 (Quantil, Expektile).
Die Prüfsumme (`erzeuger_hash`) deckt mtw_rng, mtw_scenario, mtw_sums, mtw_estimators, mtw_stein, mtw_theory, mtw_experiments und dieses Skript ab;
tests/test_results.py schlägt fehl, wenn sich eine dieser Dateien nach dem Erzeugen geändert hat."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mtw_constants as C  # noqa: E402
import mtw_experiments as X  # noqa: E402

PARTS_DIR = ROOT / "data" / "_parts"


def run_part(name: str) -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = X.PARTS[name]()
    out = PARTS_DIR / f"{name}.json"
    out.write_text(json.dumps({"hash": X.source_hash(), "seconds": round(time.perf_counter() - t0, 1), "data": result}, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"{name}: fertig in {time.perf_counter() - t0:.0f} s -> {out}", flush=True)


def merge() -> None:
    parts, hashes = {}, set()
    for name in X.PARTS:
        raw = json.loads((PARTS_DIR / f"{name}.json").read_text(encoding="utf-8"))
        parts[name] = raw["data"]
        hashes.add(raw["hash"])
    if hashes != {X.source_hash()}:
        sys.exit(f"Teile sind mit anderem Quellstand erzeugt ({sorted(hashes)} gegen {X.source_hash()}); Teile neu erzeugen")
    meta = {"stand": X.STAND, "erzeuger_hash": X.source_hash(), "seeds": list(X.SEEDS), "quellen": list(X.SOURCES)}
    (ROOT / "data" / C.RESULTS_FILE).write_text(json.dumps({"meta": meta, **parts}, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print("data/" + C.RESULTS_FILE + " geschrieben", X.source_hash())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=sorted(X.PARTS))
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.part:
        run_part(args.part)
    elif args.merge:
        merge()
    else:
        ap.error("--part oder --merge angeben")
