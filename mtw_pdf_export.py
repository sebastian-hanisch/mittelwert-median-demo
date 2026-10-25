"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Fehler der Summationsverfahren, die vier Mitten der Stichprobe und die Verdikte. Erst nach Klick erzeugt (DEMO-PLAYBOOK §1c)."""

from __future__ import annotations

import re

from fpdf import FPDF

import mtw_evaluation as V

_REPLACE = {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "√": "Wurzel", "‖": "|", "Σ": "Summe", "°": " Grad", "·": "*", "−": "-", "„": '"', "“": '"', "ε": "eps",
            "τ": "tau", "μ": "mu", "σ": "sigma", "→": "->", "×": "x", "½": "1/2", "∞": "inf", "θ": "theta"}
_SUPERSCRIPT = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-"}


def _latin(text: str) -> str:
    """Kernschriften kennen nur Latin-1: Sonderzeichen ersetzen, Markdown-Sterne entfernen."""
    text = re.sub("[" + "".join(_SUPERSCRIPT) + "]+", lambda m: "^" + "".join(_SUPERSCRIPT[c] for c in m.group()), text)      # Hochzahlen: 10⁻³ → 10^-3
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.replace("**", "").replace("`", "").encode("latin-1", "replace").decode("latin-1")


def build_pdf(rechnen: V.RechnenResult, sample: V.SampleResult, verdicts: list[tuple[str, str]]) -> bytes:
    s = rechnen.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Mittelwert und Median: Welche Zahl will ich?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(f"Rechnen: {s.fmt}, n = {V.num(s.n, 0)}, mu/sigma = {V.num(s.ratio, 0)}, Verteilung {s.rdist}, Reihenfolge {s.order}, Zufalls-Seed {s.seed}."))
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Fehler des Mittels je Verfahren (Betrag, gegen das exakte Mittel der gespeicherten Werte)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for name, value in rechnen.abs_errors.items():
        pdf.cell(0, 5.5, _latin(f"{name}: {V.sci(value)}  ({V.de(value / rechnen.standard_error, 3)} Standardfehler)"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    w = sample.settings
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Stichprobe und ihre Mitten", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(f"Verteilung {w.dist}, n = {V.num(w.wn, 0)}, Ausreisser {w.out} % bei {w.outdist}, Raster h = {w.h:g}, Lage {w.shift:g}, tau = {w.tau:g}."))
    lo, hi = sample.quantile
    for name, value in (("Mittel", sample.mean), ("Median", sample.median), (f"Quantil {w.tau:g}", lo), (f"Expektile {w.tau:g}", sample.expectile)):
        pdf.cell(0, 5.5, _latin(f"{name}: {V.de(value, 3)}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    for _, text in verdicts:
        pdf.multi_cell(0, 5, _latin(text))
        pdf.ln(1)
    return bytes(pdf.output())
