# Wörtliche Kopie von reduktionskette-demo/red_scenario.py (letzter Commit der Datei: ba24648, Arbeitsbaum sauber am 2026-10-25); keine Änderung außer dieser Zeile. Änderungen gehören ins Original, nicht hierher.
"""Szenario: Zufallsformeln in 3-KNF und Zufallsgraphen, mit eigenem Zufallsgenerator (SplitMix64), damit alle Instanzen auf jeder Plattform und in jeder Python-Version gleich sind.

Eine Formel ist eine Liste von Klauseln, eine Klausel ein Tripel von Literalen: +v steht für „Variable v“, -v für „nicht v“ (v = 1 bis n)."""

MASK = (1 << 64) - 1


class SplitMix64:
    """Zufallsgenerator nach der SplitMix64-Konvention (Vigna); gleicher Seed, gleicher Strom."""

    def __init__(self, seed: int):
        self.state = seed & MASK

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def below(self, k: int) -> int:
        """Gleichverteilte ganze Zahl in [0, k) (bis auf den vernachlässigbaren Rest der Modulo-Bildung)."""
        return self.next() % k

    def coin(self) -> bool:
        return bool(self.next() >> 63)


def random_clause(n: int, rng: SplitMix64) -> tuple[int, int, int]:
    """Drei verschiedene Variablen aus 1..n, jede mit zufälligem Vorzeichen. Zieht aus dem Strom immer gleich viele Zahlen (drei Positionen, drei Vorzeichen)."""
    variables = list(range(1, n + 1))
    for i in range(3):
        j = i + rng.below(n - i)
        variables[i], variables[j] = variables[j], variables[i]
    signs = [rng.coin() for _ in range(3)]
    return tuple(v if s else -v for v, s in zip(variables[:3], signs))


def random_formula(n: int, m: int, seed: int) -> list[tuple[int, int, int]]:
    """m Klauseln über n >= 3 Variablen, deterministisch aus (n, m, seed). Bei gleichem n und Seed ist die Formel mit m Klauseln ein Anfang der Formel mit m + 1 Klauseln:
    wächst m, kommen Klauseln hinzu, die früheren bleiben (das wachsende Beispiel der Messreihen)."""
    if n < 3:
        raise ValueError("3-KNF braucht mindestens drei Variablen")
    rng = SplitMix64(seed)
    return [random_clause(n, rng) for _ in range(m)]


def random_graph(n: int, edges: int, seed: int) -> tuple[frozenset, ...]:
    """Zufallsgraph mit genau `edges` verschiedenen Kanten auf n Knoten (Adjazenz als Tupel von frozensets), deterministisch aus (n, edges, seed)."""
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    if not 0 <= edges <= len(pairs):
        raise ValueError(f"{edges} Kanten passen nicht in einen Graphen mit {n} Knoten")
    rng = SplitMix64(seed)
    for i in range(edges):
        j = i + rng.below(len(pairs) - i)
        pairs[i], pairs[j] = pairs[j], pairs[i]
    adj = [set() for _ in range(n)]
    for a, b in pairs[:edges]:
        adj[a].add(b)
        adj[b].add(a)
    return tuple(frozenset(s) for s in adj)


def sparse_random_graph(n: int, per_mille: int, seed: int) -> tuple[frozenset, ...]:
    """Graph, in dem jede der n(n-1)/2 möglichen Kanten mit Wahrscheinlichkeit per_mille / 1000 vorkommt (für die Bipartit-Messreihe)."""
    rng = SplitMix64(seed)
    adj = [set() for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            if rng.below(1000) < per_mille:
                adj[a].add(b)
                adj[b].add(a)
    return tuple(frozenset(s) for s in adj)


def literal_name(lit: int) -> str:
    """Anzeigename eines Literals: x1, ¬x2."""
    return f"x{lit}" if lit > 0 else f"¬x{-lit}"
