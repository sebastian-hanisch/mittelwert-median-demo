"""PDF-Export: erzeugt gültige PDF-Bytes mit Umlauten und Sonderzeichen für jedes Preset."""

import mtw_constants as C
import mtw_evaluation as V
import mtw_pdf_export as P


def _pdf(settings):
    rr = V.rechnen_analysis(settings)
    sa = V.sample_analysis(settings)
    verdicts = [V.verdict_rechnen(rr.abs_errors, settings.n), V.verdict_sample(sa.mean, sa.median, sa.iqr, settings.wn, V.plan_missing(sa.x))]
    return P.build_pdf(rr, sa, verdicts)


def test_pdf_for_every_cheap_preset_is_a_pdf():
    for name, preset in C.PRESETS.items():
        s = V.Settings(**{**preset, "n": min(preset["n"], 100_000)})
        data = _pdf(s)
        assert data[:5] == b"%PDF-" and len(data) > 1500, name


def test_pdf_with_a_long_text():
    s = V.Settings(n=1000)
    rr, sa = V.rechnen_analysis(s), V.sample_analysis(s)
    assert P.build_pdf(rr, sa, [V.verdict_rechnen(rr.abs_errors, 1000)] * 20)[:5] == b"%PDF-"


def test_latin_replaces_characters_the_core_fonts_do_not_know():
    assert P._latin("Zugkraft ≤ Gewicht – ‖x‖ · 45° **fett**") == "Zugkraft <= Gewicht - |x| * 45 Grad fett"
    assert P._latin("Größe ä ö ü ß") == "Größe ä ö ü ß"
    assert P._latin("μ/σ = 10⁻³, τ → ½ × θ") == "mu/sigma = 10^-3, tau -> 1/2 x theta"
    assert P._latin("2²⁹ und 10¹¹ und ‖x‖²") == "2^29 und 10^11 und |x|^2"
