"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.
Beschriftungen als Annotationen (kein Text-Trace), Werte mit Null werden auf eine Untergrenze gesetzt, damit logarithmische Achsen sie zeigen."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import mtw_constants as C
import mtw_estimators as est

LOG_FLOOR = 1e-13          # Null-Fehler werden auf diese Höhe gesetzt (und als „0“ beschriftet)
PALETTE = ("#4c78a8", "#f58518", "#54a24b", "#b279a2", "#9d755d", "#d62728", "#7f7f7f", "#2F6B65", "#b8860b")


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25) if legend else dict(), showlegend=legend, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _pos(v: float) -> float:
    return max(v, LOG_FLOOR)


# --- Rechnen ------------------------------------------------------------------------------------------------------------------------------

def build_method_bars(errors: dict, n: int, title: str = "") -> go.Figure:
    """Betrag des Fehlers je Verfahren in Standardfehlern σ/√n (logarithmisch); Linien bei einem Standardfehler und bei der Untergrenze des Formats."""
    se = 1.0 / math.sqrt(n)
    names = [k for k in errors if k != "Untergrenze"]
    vals = [_pos(errors[k] / se) for k in names]
    colors = [C.COLOR_BAD if errors[k] > 4 * errors["Untergrenze"] + 1e-300 else C.COLOR_GOOD for k in names]
    fig = go.Figure(go.Bar(y=names[::-1], x=vals[::-1], orientation="h", marker_color=colors[::-1], hovertemplate="%{y}: %{x:.3g} Standardfehler<extra></extra>"))
    fig.add_vline(x=1.0, line=dict(color="#555555", dash="dot"))
    fig.add_vline(x=_pos(errors["Untergrenze"] / se), line=dict(color=C.COLOR_FLOOR, dash="dash"))
    fig.add_annotation(x=math.log10(1.0), y=1.0, yref="paper", yshift=26, text="1 Standardfehler", showarrow=False, font=dict(size=11), bgcolor="rgba(255,255,255,0.85)")
    fig.add_annotation(x=math.log10(_pos(errors["Untergrenze"] / se)), y=1.0, yref="paper", yshift=10, text="Untergrenze der Summe im Format", showarrow=False, font=dict(size=11, color=C.COLOR_FLOOR),
                       bgcolor="rgba(255,255,255,0.85)")
    for k, v in zip(names[::-1], vals[::-1]):
        if errors[k] == 0:
            fig.add_annotation(x=math.log10(v), y=k, text="0", showarrow=False, xanchor="left", font=dict(size=11))
    fig.update_xaxes(type="log", title_text="Betrag des Fehlers in Standardfehlern (σ/√n)")
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=60, b=10))
    return _base(fig, height=40 * len(names) + 130, legend=False).update_layout(margin=dict(l=10, r=10, t=60, b=10))


def build_error_curve(rows: dict, checks, methods, title: str = "") -> go.Figure:
    """Fehler über n (log-log): `rows` ist das Feld "methods" eines E1-Teils (Verfahren → n → Zelle); gestrichelt die Untergrenze der Summe, gepunktet der Standardfehler."""
    fig = go.Figure()
    for i, m in enumerate(methods):
        fig.add_trace(go.Scatter(x=list(checks), y=[_pos(rows[m][str(n)]["mean"]) for n in checks], mode="lines+markers", name=m, line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                                 hovertemplate=m + ", n = %{x}: %{y:.3g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=list(checks), y=[_pos(rows["floor_sum"][str(n)]["mean"]) for n in checks], mode="lines", name="Untergrenze (Summe im Format)", line=dict(color=C.COLOR_FLOOR, dash="dash")))
    fig.add_trace(go.Scatter(x=list(checks), y=[1.0 / math.sqrt(n) for n in checks], mode="lines", name="Standardfehler σ/√n", line=dict(color="#555555", dash="dot")))
    fig.update_xaxes(type="log", title_text="Zahl der Werte n")
    fig.update_yaxes(type="log", title_text="Betrag des Fehlers des Mittels")
    fig.update_layout(title=title)
    return _base(fig, height=380)


def build_ratio_chart(ratios: dict, methods, title: str = "") -> go.Figure:
    """Auslöschung: Fehler in Standardfehlern über μ/σ (log-log); `ratios` ist das Feld "ratios" eines E2-Teils."""
    keys = list(ratios)
    xs = [max(ratios[k]["ratio"], 0.5) for k in keys]            # μ/σ = 0 liegt links am Rand
    fig = go.Figure()
    for i, m in enumerate(methods):
        fig.add_trace(go.Scatter(x=xs, y=[_pos(ratios[k]["methods"][m]["mean"]) for k in keys], mode="lines+markers", name=m, line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
    fig.add_hline(y=1.0, line=dict(color="#555555", dash="dot"))
    fig.update_xaxes(type="log", title_text="Verhältnis μ/σ (Punkt ganz links: μ = 0)")
    fig.update_yaxes(type="log", title_text="Betrag des Fehlers in Standardfehlern")
    fig.update_layout(title=title)
    return _base(fig, height=380)


# --- Wählen ------------------------------------------------------------------------------------------------------------------------------

def build_sample_chart(x, mean: float, median: float, quantile: float, expectile: float, tau: float, truth: dict | None = None) -> go.Figure:
    """Histogramm der Stichprobe mit senkrechten Linien für Mittel, Median, Quantil und Expektile (Beschriftung als Annotation)."""
    x = np.asarray(x, dtype=np.float64)
    fig = go.Figure(go.Histogram(x=x, nbinsx=C.HIST_BINS, marker_color=C.COLOR_POINT, name="Stichprobe", hovertemplate="%{x}: %{y}<extra></extra>"))
    lines = [("Mittel", mean, C.COLOR_MEAN), ("Median", median, C.COLOR_MEDIAN), (f"Quantil {tau:.2f}".replace(".", ","), quantile, C.COLOR_QUANTILE), (f"Expektile {tau:.2f}".replace(".", ","), expectile, C.COLOR_EXPECTILE)]
    for i, (name, value, color) in enumerate(lines):
        fig.add_vline(x=value, line=dict(color=color, width=2, dash="solid" if i < 2 else "dash"))
        fig.add_annotation(x=value, y=1.0 - 0.09 * i, yref="paper", text=name, showarrow=False, xanchor="left", font=dict(size=11, color=color), bgcolor="rgba(255,255,255,0.8)")
    fig.update_xaxes(title_text="Wert")
    fig.update_yaxes(title_text="Anzahl")
    return _base(fig, height=340, legend=False)


def build_loss_chart(x, tau: float, mean: float, median: float, quantile: float, expectile: float) -> go.Figure:
    """Vier Verlustkurven über c, jeweils bis auf das Minimum verschoben: Σ(x − c)² (Mittel), Σ|x − c| (Median), Pinball (Quantil), asymmetrische Quadratsumme (Expektile)."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(np.quantile(x, 0.01)), float(np.quantile(x, 0.99))
    pad = 0.15 * (hi - lo) if hi > lo else 1.0
    grid = np.linspace(lo - pad, hi + pad, 160)
    sub = x if len(x) <= 400 else x[:: len(x) // 400]
    specs = (("Quadratsumme → Mittel", lambda c: float(((sub - c) ** 2).sum()), mean, C.COLOR_MEAN), ("Betragssumme → Median", lambda c: est.sad(sub, c), median, C.COLOR_MEDIAN),
             ("Pinball-Verlust → Quantil", lambda c: est.pinball(sub, c, tau), quantile, C.COLOR_QUANTILE), ("asymmetrische Quadrate → Expektile", lambda c: est.expectile_loss(sub, c, tau), expectile, C.COLOR_EXPECTILE))
    fig = make_subplots(rows=2, cols=2, subplot_titles=[s[0] for s in specs], vertical_spacing=0.18)
    for i, (name, f, best, color) in enumerate(specs):
        r, c = divmod(i, 2)
        y = np.array([f(v) for v in grid])
        fig.add_trace(go.Scatter(x=grid, y=y, mode="lines", line=dict(color=color, width=2), showlegend=False, hovertemplate="c = %{x:.3g}<extra></extra>"), row=r + 1, col=c + 1)
        fig.add_trace(go.Scatter(x=[best], y=[f(best)], mode="markers", marker=dict(color=color, size=9, symbol="diamond"), showlegend=False, hoverinfo="skip"), row=r + 1, col=c + 1)
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_plane_chart(pr) -> go.Figure:
    """Karte der Kunden mit Schwerpunkt vor und nach der Fehlbuchung und Median je Achse; der Fehlbuchungs-Kunde ist rot."""
    p, q = pr.points, pr.with_error
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="markers", name="Kunden", marker=dict(color=C.COLOR_POINT, size=8)))
    fig.add_trace(go.Scatter(x=[q[-1, 0]], y=[q[-1, 1]], mode="markers", name="Fehlbuchung", marker=dict(color=C.COLOR_BAD, size=11, symbol="x")))
    for name, c, color, symbol in (("Schwerpunkt ohne Fehlbuchung", pr.centroid, C.COLOR_MEAN, "circle-open"), ("Schwerpunkt mit Fehlbuchung", pr.centroid_error, C.COLOR_MEAN, "star"),
                                   ("Median je Achse ohne", pr.axis_median, C.COLOR_MEDIAN, "circle-open"), ("Median je Achse mit", pr.axis_median_error, C.COLOR_MEDIAN, "star")):
        fig.add_trace(go.Scatter(x=[c[0]], y=[c[1]], mode="markers", name=name, marker=dict(color=color, size=13, symbol=symbol, line=dict(width=2, color=color))))
    extent = float(max(np.abs(q).max(), 3.0))
    fig.add_trace(go.Scatter(x=[-extent, extent], y=[-extent, extent], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))   # erzwingt gleichen Rand
    fig.update_xaxes(title_text="x (km)", scaleanchor="y", constrain="domain")
    fig.update_yaxes(title_text="y (km)", autorange=True)
    return _base(fig, height=420)


# --- Experimente --------------------------------------------------------------------------------------------------------------------------

def build_stein_chart(cells: dict, dims, norms) -> go.Figure:
    """Risiko James-Stein / Einzelmittel über ‖μ‖ je Dimension: Punkte simuliert (Positive Part), Linie exakt (ohne Positive Part); cells[(d, norm)] mit "pp", "js_exact"."""
    fig = go.Figure()
    for i, d in enumerate(dims):
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(x=list(norms), y=[cells[(d, nm)]["js_exact"] for nm in norms], mode="lines", name=f"d = {d} (exakt, James-Stein)", line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(x=list(norms), y=[cells[(d, nm)]["pp"] for nm in norms], mode="markers", name=f"d = {d} (Positive Part, simuliert)", marker=dict(color=color, size=7, symbol="diamond")))
    fig.add_hline(y=1.0, line=dict(color="#555555", dash="dot"))
    fig.update_xaxes(title_text="Abstand ‖μ‖ der wahren Mittel vom Nullpunkt")
    fig.update_yaxes(title_text="Risiko / Risiko des Einzelmittels", range=[0, 1.1])
    return _base(fig, height=380)


def build_coordinate_chart(cells: dict, dims, norms) -> go.Figure:
    """Risiko des ersten Lagers allein (Einzelmittel: 1) über ‖μ‖ je Dimension (Positive Part)."""
    fig = go.Figure()
    for i, d in enumerate(dims):
        fig.add_trace(go.Scatter(x=list(norms), y=[cells[(d, nm)]["koordinate1_pp"] for nm in norms], mode="lines+markers", name=f"d = {d}", line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
    fig.add_hline(y=1.0, line=dict(color="#555555", dash="dot"))
    fig.update_xaxes(title_text="Abstand ‖μ‖")
    fig.update_yaxes(title_text="Risiko des ersten Lagers / Risiko des Einzelmittels", type="log")
    return _base(fig, height=360)


def build_influence_chart(ts, median, mean, title: str = "") -> go.Figure:
    """Einflusskurve: SC(t) für Median (beschränkt) und Mittel (linear), t logarithmisch."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(ts), y=list(median), mode="lines+markers", name="Median", line=dict(color=C.COLOR_MEDIAN, width=3)))
    fig.add_trace(go.Scatter(x=list(ts), y=list(mean), mode="lines+markers", name="Mittel", line=dict(color=C.COLOR_MEAN, width=3)))
    fig.update_xaxes(type="log", title_text="Wert t des hinzugefügten Punktes")
    fig.update_yaxes(type="log", title_text="Einfluss SC(t)")
    fig.update_layout(title=title)
    return _base(fig, height=340)


def build_contamination_chart(rows) -> go.Figure:
    """Verschiebung von Mittel und Median über dem Ausreißeranteil (n = 100, Ausreißer bei 100); Zeilen aus E3."""
    eps = [100 * r["eps"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eps, y=[r["mean"]["mean"] for r in rows], mode="lines+markers", name="Mittel", line=dict(color=C.COLOR_MEAN, width=3)))
    fig.add_trace(go.Scatter(x=eps, y=[r["median"]["mean"] for r in rows], mode="lines+markers", name="Median", line=dict(color=C.COLOR_MEDIAN, width=3)))
    fig.update_xaxes(title_text="Anteil der Ausreißer (%)")
    fig.update_yaxes(title_text="mittlere Verschiebung", type="log")
    return _base(fig, height=340)


def build_efficiency_chart(odd: dict, even: dict, limit: float) -> go.Figure:
    """Var(Mittel)/Var(Median) bei Normalverteilung über n (exakt): gerades n liegt über dem Nachbarn (Sägezahn); gestrichelt 2/π."""
    fig = go.Figure()
    for name, d, color in (("ungerades n", odd, C.COLOR_MEDIAN), ("gerades n", even, C.COLOR_EXPECTILE)):
        ns = sorted(int(k) for k in d)
        fig.add_trace(go.Scatter(x=ns, y=[d[str(n)] for n in ns], mode="lines+markers", name=name, line=dict(color=color, width=2)))
    fig.add_hline(y=limit, line=dict(color="#555555", dash="dash"))
    fig.update_xaxes(type="log", title_text="Zahl der Werte n")
    fig.update_yaxes(title_text="Varianz des Mittels / Varianz des Medians", range=[0.55, 1.05])
    return _base(fig, height=340)


def build_target_chart(rows: dict) -> go.Figure:
    """E6: typischer relativer Fehler von Mittel und Median gegen beide Ziele über σ."""
    sigmas = sorted(rows, key=float)
    fig = go.Figure()
    for i, (key, name) in enumerate((("mittel_mittelwert", "Mittel → Mittelwert"), ("median_mittelwert", "Median → Mittelwert"), ("mittel_median", "Mittel → Median"), ("median_median", "Median → Median"))):
        fig.add_trace(go.Scatter(x=[float(s) for s in sigmas], y=[rows[s][key]["typisch"]["mean"] for s in sigmas], mode="lines+markers", name=name, line=dict(color=PALETTE[i], width=2)))
    fig.update_xaxes(title_text="Schiefe σ der Lognormal-Verteilung")
    fig.update_yaxes(title_text="typischer relativer Fehler", type="log")
    return _base(fig, height=360)


def build_gap_chart(gaps: dict, taus, labels) -> go.Figure:
    """E8: Lücke (Expektile − Quantil) in Quartilsabständen über die Stufe τ je Verteilung; unsichere Zellen (usable = False) werden ausgelassen."""
    fig = go.Figure()
    for i, label in enumerate(labels):
        pts = [(t, gaps[label][f"{t:g}"]) for t in taus if gaps[label][f"{t:g}"]["usable"]]
        fig.add_trace(go.Scatter(x=[t for t, _ in pts], y=[c["gap"]["mean"] for _, c in pts], mode="lines+markers", name=label, line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
    fig.add_hline(y=0.0, line=dict(color="#555555", dash="dot"))
    fig.update_xaxes(title_text="Stufe τ")
    fig.update_yaxes(title_text="(Expektile − Quantil) / Quartilsabstand")
    return _base(fig, height=400)


def build_rounding_chart(rows: dict) -> go.Figure:
    """E7: RMSE von Mittel und Median gegen μ über das Raster h der Rundung."""
    hs = sorted(rows, key=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[float(h) for h in hs], y=[rows[h]["rmse_mean"]["mean"] for h in hs], mode="lines+markers", name="Mittel", line=dict(color=C.COLOR_MEAN, width=3)))
    fig.add_trace(go.Scatter(x=[float(h) for h in hs], y=[rows[h]["rmse_median"]["mean"] for h in hs], mode="lines+markers", name="Median", line=dict(color=C.COLOR_MEDIAN, width=3)))
    fig.update_xaxes(title_text="Raster h der Rundung (in Streuungen)")
    fig.update_yaxes(title_text="RMSE gegen μ")
    return _base(fig, height=320)
