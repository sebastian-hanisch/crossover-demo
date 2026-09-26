"""Plotly-Abbildungen: Weg von der Näherung zur Ecke (n = 2), Indikator und Crash-Basis, Pivots und Fehler über die Genauigkeit der Quelle, Gesamtkosten gegen den Simplex, Klee-Minty-Würfel, Warmstart,
Zustände der Crash-Basis. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen; bei gleichem Achsenmaßstab (scaleanchor) gibt es keine expliziten Bereiche."""

import numpy as np
import plotly.graph_objects as go

import xov_constants as C
import xov_evaluation as ev
import xov_scenario as S

TEAL, ORANGE, RED, BLUE, GREY, PURPLE = "#2F6B65", "#e8a13a", "#d62728", "#1f4e9c", "#8a8f98", "#7b3fbf"
SOURCE_COLORS = {"ipm": PURPLE, "pdlp": ORANGE}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _box(inst):
    """Obere Schranken je Dienst aus den <=-Zeilen mit nichtnegativen Koeffizienten; ohne Schranke das 1.5-Fache der größten."""
    A, b, _c = inst.arrays()
    U = np.full(inst.n, np.inf)
    for i, s in enumerate(inst.senses):
        if s != S.LE or np.any(A[i] < 0):
            continue
        for j in range(inst.n):
            if A[i, j] > 0:
                U[j] = min(U[j], b[i] / A[i, j])
    finite = U[np.isfinite(U)]
    fill = 1.5 * (finite.max() if len(finite) else 10.0)
    return np.where(np.isfinite(U), U, fill)


def feasible_polygon(inst):
    """Zulässiges Vieleck einer Instanz mit zwei Variablen (Kasten [0, U], an <=- und >=-Zeilen abgeschnitten; Gleichungen bleiben unberücksichtigt)."""
    U = _box(inst)
    poly = [np.array([0.0, 0.0]), np.array([U[0], 0.0]), np.array([U[0], U[1]]), np.array([0.0, U[1]])]
    A, b, _c = inst.arrays()
    for i, s in enumerate(inst.senses):
        if s == S.EQ:
            continue
        a, beta = (A[i], b[i]) if s == S.LE else (-A[i], -b[i])
        out = []
        for k, p in enumerate(poly):
            q = poly[(k + 1) % len(poly)]
            sp, sq = float(a @ p) - beta, float(a @ q) - beta
            if sp <= 0:
                out.append(p)
            if (sp < 0 < sq) or (sq < 0 < sp):
                out.append(p + (q - p) * (sp / (sp - sq)))
        poly = out
        if not poly:
            break
    return poly


def build_path(a, k):
    """Zwei Variablen: zulässige Menge, Näherung der Quelle, Crash-Ecke (hohl, wenn sie unzulässig ist), Ecken der ersten k Aufräum-Pivots und Optimum des Simplex."""
    inst = a.inst
    poly = feasible_polygon(inst)
    fig = go.Figure()
    if poly:
        px, py = [p[0] for p in poly], [p[1] for p in poly]
        fig.add_trace(go.Scatter(x=px + [px[0]], y=py + [py[0]], fill="toself", fillcolor="rgba(47,107,101,0.18)", line=dict(color=TEAL, width=2), name="zulässige Menge", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[a.x[0]], y=[a.x[1]], mode="markers", marker=dict(size=14, color=SOURCE_COLORS[a.source], symbol="circle"), name=f"Näherung ({C.SOURCE_SHORT[a.source]})", hoverinfo="skip"))
    verts = ev.vertex_path(a)[: k + 1]
    if verts:
        xs, ys = [v[0] for v in verts], [v[1] for v in verts]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=BLUE, width=2, dash="dot"), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs[1:], y=ys[1:], mode="markers", marker=dict(size=10, color=[BLUE if v[2] else "rgba(0,0,0,0)" for v in verts[1:]], line=dict(color=BLUE, width=2)), name="Ecken nach den Pivots", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[xs[0]], y=[ys[0]], mode="markers", marker=dict(size=12, color=BLUE if verts[0][2] else "rgba(0,0,0,0)", symbol="diamond", line=dict(color=BLUE, width=2)), name="Crash-Ecke", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[xs[-1]], y=[ys[-1]], mode="markers", marker=dict(size=16, color=RED, symbol="x"), name="aktuelle Ecke", hoverinfo="skip"))
    if a.simplex.status == "optimal":
        fig.add_trace(go.Scatter(x=[a.simplex.x[0]], y=[a.simplex.x[1]], mode="markers", marker=dict(size=14, color=BLUE, symbol="star"), name="Optimum (Simplex)", hoverinfo="skip"))
    fig.update_xaxes(title_text="Dienst 1", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="Dienst 2")
    return _base(fig, 460, legend_y=-0.3)


def var_labels(a):
    """Namen der Standardform-Variablen: Dienste x1..xn, danach Schlupf s / Überschuss e je Zeile."""
    labels = [f"x{j + 1}" for j in range(a.inst.n)]
    for i, s in enumerate(a.inst.senses):
        if s != S.EQ:
            labels.append(f"{'s' if s == S.LE else 'e'}{i + 1}")
    return labels


def build_indicator(a):
    """Indikator je Variable in Prioritätsreihenfolge; grün: Crash-Basis, blau umrandet: Spalten der optimalen Basis nach dem Aufräumen, grau: Nichtbasis."""
    xov = a.xov
    order = list(xov.order)
    xi = np.array(xov.indicator)[order]
    labels = var_labels(a)
    labels += [f"v{j + 1}" for j in range(len(labels), len(order))]
    crash, final = set(xov.crash_basis), set(xov.basis)
    fig = go.Figure(go.Bar(x=list(range(len(order))), y=xi, marker_color=[TEAL if j in crash else GREY for j in order], marker_line_color=[BLUE if j in final else "rgba(0,0,0,0)" for j in order],
                           marker_line_width=[3 if j in final else 0 for j in order], text=[labels[j] for j in order] if len(order) <= 30 else None, textposition="outside", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=TEAL, name="Crash-Basis"))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=GREY, name="Nichtbasis"))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color="rgba(0,0,0,0)", marker_line_color=BLUE, marker_line_width=3, name="optimale Basis"))
    fig.update_xaxes(title_text="Variablen nach fallendem Indikator", showticklabels=False)
    fig.update_yaxes(title_text="Indikator x̂ / (x̂ + ŝ)", range=[0, 1.15])
    return _base(fig, 340, legend_y=-0.3)


def build_eps(rows):
    """Aufräum-Pivots (Median) über die Genauigkeit der Quelle, beide Quellen."""
    fig = go.Figure()
    for source in C.SOURCES:
        sel = [r for r in rows if r["source"] == source]
        fig.add_trace(go.Bar(x=[f"10^-{r['k']}" for r in sel], y=[r["pivots"] for r in sel], marker_color=SOURCE_COLORS[source], name=C.SOURCE_SHORT[source], text=[f"{r['zero']}/{r['runs']} ohne Pivot" for r in sel], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Genauigkeit der Quelle (Beschriftung: Läufe ohne Pivot)", type="category")
    fig.update_yaxes(title_text="Aufräum-Pivots (Median)", rangemode="tozero")
    return _base(fig, 320, legend_y=-0.4)


def build_errors(rows):
    """Relativer Zielwertfehler der Quelle gegen den des Crossover (log-y)."""
    fig = go.Figure()
    for source in C.SOURCES:
        sel = [r for r in rows if r["source"] == source]
        fig.add_trace(go.Scatter(x=[r["k"] for r in sel], y=[max(r["err_src"], 1e-17) for r in sel], mode="lines+markers", line=dict(color=SOURCE_COLORS[source], width=3), name=f"{C.SOURCE_SHORT[source]} (Quelle)"))
    fig.add_trace(go.Scatter(x=[r["k"] for r in rows if r["source"] == "ipm"], y=[max(r["err_xov"], 1e-17) for r in rows if r["source"] == "ipm"], mode="lines+markers", line=dict(color=TEAL, width=3, dash="dash"), name="nach dem Crossover"))
    fig.update_yaxes(type="log", title_text="relativer Zielwertfehler")
    fig.update_xaxes(title_text="Genauigkeit der Quelle 10^-k", dtick=1)
    return _base(fig, 300, legend_y=-0.4)


def build_size(sweep):
    rows = sweep["rows"]
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[max(r["flops_simplex"], 1) for r in rows], mode="lines+markers", line=dict(color=TEAL, width=3), name="Simplex allein"))
    fig.add_trace(go.Scatter(x=ns, y=[r["flops_ipm"] for r in rows], mode="lines+markers", line=dict(color=PURPLE, width=3), name="Innere Punkte + Crossover"))
    fig.add_trace(go.Scatter(x=ns, y=[r["flops_pdlp"] for r in rows], mode="lines+markers", line=dict(color=ORANGE, width=3), name="PDLP + Crossover"))
    fig.update_yaxes(type="log", title_text="Operationen (Modell)")
    fig.update_xaxes(title_text="Größe n (Zufall, m = n)", type="log")
    return _base(fig, 340, legend_y=-0.4)


def build_cube(sweep):
    rows = sweep["rows"]
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[r["pivots_simplex"] for r in rows], mode="lines+markers", line=dict(color=TEAL, width=3), name="Simplex von Null (Pivots)"))
    for source in C.SOURCES:
        fig.add_trace(go.Scatter(x=ns, y=[max(r[f"pivots_{source}"], 0.5) for r in rows], mode="lines+markers", line=dict(color=SOURCE_COLORS[source], width=3), name=f"{C.SOURCE_SHORT[source]} + Crossover (Aufräum-Pivots)"))
    fig.update_yaxes(type="log", title_text="Pivots")
    fig.update_xaxes(title_text="Größe n des Würfels", dtick=2)
    return _base(fig, 320, legend_y=-0.4)


def build_warm(rows):
    """Neuoptimierung nach einer Änderung der rechten Seite: Iterationen der Inneren Punkte und Pivots, kalt gegen warm (log-y)."""
    ps = [f"{r['p']:.0%}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ps, y=[r["simplex_cold"] for r in rows], mode="lines+markers", line=dict(color=TEAL, width=3), name="Simplex von Null (Pivots)"))
    fig.add_trace(go.Scatter(x=ps, y=[max(r["dual_warm"], 0.5) for r in rows], mode="lines+markers", line=dict(color=TEAL, width=3, dash="dash"), name="dualer Simplex ab alter Basis (Pivots)"))
    fig.add_trace(go.Scatter(x=ps, y=[r["ipm_cold"] for r in rows], mode="lines+markers", line=dict(color=PURPLE, width=3), name="Innere Punkte kalt (Iterationen)"))
    fig.add_trace(go.Scatter(x=ps, y=[r["ipm_warm_small"] for r in rows], mode="lines+markers", line=dict(color=PURPLE, width=2, dash="dash"), name="Innere Punkte warm, Verschiebung 10^-3"))
    fig.add_trace(go.Scatter(x=ps, y=[r["ipm_warm_big"] for r in rows], mode="lines+markers", line=dict(color=PURPLE, width=2, dash="dot"), name="Innere Punkte warm, Verschiebung 1"))
    fig.update_yaxes(type="log", title_text="Iterationen bzw. Pivots")
    fig.update_xaxes(title_text="Änderung der rechten Seite", type="category")
    return _base(fig, 360, legend_y=-0.55)


def build_cases(rows):
    """Zustand der Crash-Basis je Instanzart (Anzahl Läufe)."""
    labels = {"optimal": "schon optimal", "primal": "nur primal zulässig", "dual": "nur dual zulässig", "beides unzulässig": "beides unzulässig (Kostenverschiebung)"}
    colors = {"optimal": TEAL, "primal": ORANGE, "dual": BLUE, "beides unzulässig": RED}
    fig = go.Figure()
    for case in ("optimal", "primal", "dual", "beides unzulässig"):
        fig.add_trace(go.Bar(x=[r["kind"] for r in rows], y=[r["cases"].get(case, 0) for r in rows], marker_color=colors[case], name=labels[case]))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="Instanzart", type="category")
    fig.update_yaxes(title_text="Läufe")
    return _base(fig, 320, legend_y=-0.45)
