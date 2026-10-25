"""Der kopierte Baustein (mtw_rng.py, Entscheidung der Reihe: wörtlich kopieren, Quelle im Kopf der Datei): Vergleich mit dem Original, solange es im Portfolio-Ordner liegt."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {"mtw_rng.py": ("reduktionskette-demo", "red_scenario.py")}


@pytest.mark.parametrize("copy", list(SOURCES))
def test_copy_is_verbatim_apart_from_the_header(copy):
    folder, original = SOURCES[copy]
    source = ROOT.parent / folder / original
    if not source.exists():
        pytest.skip("Original liegt nicht im Portfolio-Ordner")
    mine = (ROOT / copy).read_text(encoding="utf-8").splitlines()
    assert mine[0].startswith(f"# Wörtliche Kopie von {folder}/{original}")
    assert mine[1:] == source.read_text(encoding="utf-8").splitlines(), "die Kopie weicht vom Original ab: Änderungen gehören ins Original, danach neu kopieren"


def test_copied_generator_has_the_documented_stream():
    """Die Kopie läuft für sich: SplitMix64 mit Seed 0 liefert die bekannten ersten Werte der Konvention (Vigna)."""
    from mtw_rng import SplitMix64
    rng = SplitMix64(0)
    assert [rng.next() for _ in range(3)] == [16294208416658607535, 7960286522194355700, 487617019471545679]
    assert SplitMix64(7).below(10) == SplitMix64(7).below(10) < 10
