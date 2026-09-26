"""Crossover – aus dem Inneren zurück zur Ecke - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Elftes und letztes Stück der Lineare-Programmierung-Reihe der "Konzepte"-Reihe: Innere Punkte und PDLP liefern eine Näherung im Inneren, keine Basis und keine exakten Duale. Crossover erkennt aus der Näherung
eine Basis und räumt mit wenigen Simplex-Pivots zu einer exakten Ecke auf. Die Demo zeigt den Weg, zählt die Pivots, misst die Gesamtkosten gegen den Simplex allein und den Warmstart.

Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import xov_constants as C
import xov_evaluation as ev
import xov_scenario as S
from xov_evaluation import Settings, analyse
from xov_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    store_from_widget,
    sync_query_params,
)
from xov_visualization import (
    build_cases,
    build_cube,
    build_eps,
    build_errors,
    build_indicator,
    build_path,
    build_size,
    build_warm,
    var_labels,
)

st.set_page_config(page_title="Crossover – Sebastian Hanisch", layout="wide")


def num(x, digits=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{0.0 if abs(x) < 5e-13 else x:.{digits}f}"


def big(x):
    return f"{x:,.0f}".replace(",", ".")


def sci(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1e}"


CASE_TEXT = {"optimal": "schon optimal", "primal": "nur primal zulässig", "dual": "nur dual zulässig", "beides unzulässig": "beides unzulässig (Kostenverschiebung)"}

st.title("🔁 Crossover – aus dem Inneren zurück zur Ecke")
st.markdown(
    """
**Elftes und letztes Stück der Lineare-Programmierung-Reihe.** Innere Punkte (Stück 8) und PDLP (Stück 10) enden im **Inneren**: eine Näherung, keine Ecke, keine Basis, keine exakten Duale, kein Warmstart. Wer Schattenpreise, Ranging
oder eine Neuoptimierung braucht, braucht die Ecke zurück. **Crossover** (Megiddo 1991, Bixby & Saltzman 1994) erkennt aus der Näherung eine **Basis** (der Indikator x gegen s sagt, welche Variablen positiv bleiben) und räumt mit wenigen
**Simplex-Pivots** zur exakten Ecke auf. Vier Fragen, alle gemessen: **(1) Der Weg** - was passiert von der Näherung bis zur Ecke? **(2) Die Basis** - wie gut wird sie erkannt, wie viele Pivots bleiben? **(3) Das Ergebnis** - wie genau, wie viele
Nichtnullen? **(4) Wann welches Verfahren** - Simplex allein, Innere Punkte oder PDLP mit Crossover, und der Warmstart. **(5) Grenzen** - Plateau, Entartung, Zustände der Crash-Basis.
"""
)
st.caption("Konvergenzpunkt der Reihe: [Innere Punkte](https://github.com/sebastian-hanisch/innere-punkte-demo) und [PDLP](https://github.com/sebastian-hanisch/pdlp-demo) liefern die Näherung, der Simplex ([Tableau](https://github.com/sebastian-hanisch/tableau-simplex-demo), [dual](https://github.com/sebastian-hanisch/dualer-simplex-demo)) das Aufräumen.")

with st.expander("So funktioniert Crossover", expanded=True):
    st.markdown(
        """
1. **Standardform:** min c·x unter M x = b, x ≥ 0 mit unabhängigen Zeilen; die Quelle liefert eine Näherung x und Duale y, daraus s = c − Mᵀy (die reduzierten Kosten).
2. **Indikator:** im Optimum ist für jede Variable entweder x_j positiv (Basis) oder s_j positiv (Nichtbasis), fast nie beides. Der Indikator x̂ / (x̂ + ŝ) (mit spaltennormierten Größen, damit die Einheiten nicht stören) ordnet die Variablen: nahe 1 = Basiskandidat.
3. **Crash-Basis:** nach fallendem Indikator werden greedy m linear unabhängige Spalten gewählt (Gram-Schmidt); abhängige Spalten werden übersprungen.
4. **Aufräumen:** an der Crash-Basis prüft man x_B = B⁻¹b und die reduzierten Kosten. Beides zulässig: fertig, **0 Pivots**. Nur primal zulässig: primaler Simplex. Nur dual zulässig: dualer Simplex. Beides unzulässig: **Kostenverschiebung** macht die
   Basis dual zulässig, der duale Simplex holt die primale Zulässigkeit, dann kommen die Kosten zurück und der primale Simplex beendet es.
5. **Ergebnis:** eine Ecke mit höchstens m Nichtnullen, exakte Duale, die Basis: alles, was der Simplex liefert, aber ab einer Näherung statt von Null.
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:4], preset_names[4:8], preset_names[8:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

ss = st.session_state
with st.sidebar:
    st.header("⚙️ Einstellungen")
    kind = st.selectbox("Instanz", options=list(S.KINDS), format_func=lambda v: S.KIND_LABELS[v], key="kind_select",
                        help="Lehrbuchbeispiel, Zentrum und entartete Ecke sind fest; Zufall, Mischung (mit ≥ und =) und Plateau (eine ganze optimale Fläche) sind regelbar; der Klee-Minty-Würfel hat nur die Größe n. Unzulässig und Unbeschränkt haben keine Ecke: kein Crossover.")
    cube = kind in S.CUBE_KINDS
    random_kind = kind in ev.RANDOM_KINDS
    if random_kind:
        m = st.slider("Ressourcen m", *bounds("m_slider"), value=int(ss["m_slider"]), key="m_widget", on_change=store_from_widget, args=("m_slider",), help="Zahl der Bedingungen.")
    else:
        m = C.DEFAULT_M
    if random_kind or cube:
        cap_n = S.CUBE_MAX if cube else C.N_MAX
        n = st.slider("Dienste n", C.N_MIN, cap_n, value=min(int(ss["n_slider"]), cap_n), key="n_widget", on_change=store_from_widget, args=("n_slider",), help="Zahl der Variablen (beim Würfel auch die Zahl der Ressourcen).")
    else:
        n = C.DEFAULT_N
    if random_kind:
        density_i = st.select_slider("Dichte der Matrix", options=list(range(len(C.DENSITIES))), value=int(ss["density_select"]), format_func=C.density_label, key="density_widget", on_change=store_from_widget,
                                     args=("density_select",), help="Anteil der Nichtnullen.")
        seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), value=int(ss["seed_input"]), key="seed_widget", step=1, on_change=store_from_widget, args=("seed_input",))
        st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    else:
        density_i, seed = C.DEFAULT_DENSITY_I, C.DEFAULT_SEED
    source = st.radio("Quelle der Näherung", options=list(C.SOURCES), format_func=lambda v: C.SOURCE_LABELS[v], key="source_select", help="Beide liefern eine Näherung im Inneren; Crossover macht daraus eine Ecke.")
    eps_i = st.select_slider("Genauigkeit der Quelle ε", options=list(range(len(C.EPS_EXPS))), format_func=C.eps_label, key="eps_select", help="Relative Residuen und Lücke, bei denen die Quelle stoppt: je gröber, desto mehr Pivots im Aufräumen.")
    scale_i = st.select_slider("Skalierung der Spalten", options=list(range(len(C.SCALE_EXPS))), format_func=C.scale_label, key="scale_select",
                               help="Die Dienste werden mit Faktoren zwischen 1 und 10^k umgerechnet (Optimalwert gleich, Zahlen sehr verschieden groß): ein Test für Indikator und Crash-Basis.")
    if int(ss["xov_step"]) == 4:
        change_i = st.select_slider("Änderung der rechten Seite", options=list(range(len(C.CHANGES))), value=int(ss["change_select"]), format_func=C.change_label, key="change_widget", on_change=store_from_widget,
                                    args=("change_select",), help="Warmstart-Vergleich: jede rechte Seite b_i wird mit 1 + p·sin(i + 1) multipliziert.")
    else:
        change_i = int(ss["change_select"])

sync_query_params({"kind_select": kind, "m_slider": int(ss["m_slider"]), "n_slider": int(ss["n_slider"]), "seed_input": int(ss["seed_input"]), "density_select": int(ss["density_select"]), "source_select": source,
                   "eps_select": int(eps_i), "scale_select": int(scale_i), "change_select": int(change_i), "xov_step": int(ss["xov_step"])})

settings = Settings(kind, int(m), int(n), int(seed), int(density_i), source, int(eps_i), int(scale_i), int(change_i))
with st.spinner("Rechne..."):
    a = analyse(settings)
xov, inst = a.xov, a.inst

# --- In Aktion ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Von der Näherung zur Ecke")
step = st.select_slider("Schritt", options=list(C.STEPS), key="xov_step", format_func=lambda s: C.STEPS[s])

if xov is None:
    st.info(f"Die Quelle ({C.SOURCE_SHORT[source]}) liefert hier keine Näherung: {a.src.note or a.src.status}. Ohne Näherung gibt es nichts zu erkennen; bei einem unzulässigen oder unbeschränkten LP gibt es keine Ecke. Der Simplex meldet: {a.simplex.status}.")
elif xov.status != "optimal":
    st.warning(f"Das Aufräumen endet mit dem Status **{xov.status}**: {xov.note}. Der Simplex meldet: {a.simplex.status}.")
else:
    st.success(f"✅ Ecke gefunden: Optimum **{num(-xov.obj)}** nach **{xov.pivots}** Aufräum-Pivots ({CASE_TEXT[xov.case]}); der Simplex von Null braucht {a.simplex.pivots} Pivots. Zielwertfehler der Quelle {sci(a.src_error)} → nach dem Crossover {sci(a.xov_error)}.")

if xov is not None and xov.status == "optimal":
    labels = var_labels(a)
    labels += [f"v{j + 1}" for j in range(len(labels), a.std[0].shape[1])]
if step == 1 and xov is not None:
    total = xov.pivots
    if inst.n == 2 and a.x is not None:
        if total > 0:
            ss["pivot_k"] = min(max(0, int(ss.get("pivot_k", total))), total)
            k = st.slider("Aufräum-Pivot", 0, total, key="pivot_k", help="0 = Crash-Ecke; danach Pivot für Pivot bis zur optimalen Ecke.")
        else:
            k = 0
            st.markdown("Die erkannte Basis ist schon optimal: **0 Pivots**, die Crash-Ecke ist die Lösung.")
        st.plotly_chart(build_path(a, k), width="stretch", key=f"s1_path_{k}")
        st.caption("Der Kreis ist die Näherung der Quelle (im Inneren oder knapp daneben), die Raute die Crash-Ecke, die aus der erkannten Basis folgt (hohl, wenn sie nicht zulässig ist), die Kreise die Ecken nach den Pivots, das Kreuz die aktuelle Ecke, der Stern das Optimum des Simplex.")
    else:
        st.markdown("Ab drei Diensten gibt es kein Bild des Weges; die Tabelle zeigt die Pivots des Aufräumens.")
    if xov.path and xov.status == "optimal":
        st.dataframe(pd.DataFrame([{"Pivot": i + 1, "Art": "dual" if mv.kind == "dual" else "primal", "tritt ein": labels[mv.entering], "verlässt": labels[mv.leaving], "Zielwert (Minimierung)": num(mv.obj, 3),
                                    "Summe negativer x_B": sci(mv.infeasibility)} for i, mv in enumerate(xov.path)]), hide_index=True, width="stretch")
        if xov.shifted:
            st.caption("Zielwerte der dualen Pivots gehören zu den verschobenen Kosten (die Basis war weder primal noch dual zulässig); nach dem letzten dualen Pivot kommen die ursprünglichen Kosten zurück.")
elif step == 2 and xov is not None:
    if xov.status == "optimal":
        st.plotly_chart(build_indicator(a), width="stretch", key="s2_ind")
        st.caption(f"Variablen nach fallendem Indikator: grün die gewählte Crash-Basis ({len(xov.crash_basis)} Spalten), blau umrandet die Spalten der optimalen Basis nach dem Aufräumen. Wo Grün und Blau übereinstimmen, wurde die Basis richtig erkannt. "
                   f"{xov.skipped} Spalte(n) mit hoher Priorität wurden wegen linearer Abhängigkeit übersprungen.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Crash-Basis", {"optimal": "optimal", "primal": "primal", "dual": "dual", "beides unzulässig": "beides"}[xov.case], delta=f"{CASE_TEXT[xov.case]}; {xov.pivots} Pivots", delta_color="off")
        both = len(set(xov.crash_basis) & set(xov.basis))
        c2.metric("Richtig erkannt", f"{both} von {len(xov.basis)}", delta="Spalten der optimalen Basis", delta_color="off")
        c3.metric("Träger der exakten Ecke", num(a.support_recall * 100, 0) + " %" if not math.isnan(a.support_recall) else "-", delta="in der Crash-Basis (nicht entartet)", delta_color="off")
    tok = (settings.kind, settings.m, settings.n, settings.seed, settings.density_i, settings.scale_i)
    if st.button("Pivots über die Genauigkeit der Quelle berechnen", key="eps_start"):
        ss["eps_done"] = tok
    if ss.get("eps_done") == tok:
        with st.spinner("Rechne..."):
            rows = ev.eps_sweep(settings)
        st.plotly_chart(build_eps(rows), width="stretch", key="s2_eps")
        st.dataframe(pd.DataFrame([{"Quelle": C.SOURCE_SHORT[r["source"]], "ε": f"10^-{r['k']}", "Pivots (Median)": f"{r['pivots']:.0f}", "größter Wert": f"{r['max_pivots']:.0f}", "ohne Pivot": f"{r['zero']} von {r['runs']}",
                                     "schon optimal / primal / dual / beides": " / ".join(str(r["cases"].get(k, 0)) for k in ("optimal", "primal", "dual", "beides unzulässig"))} for r in rows]), hide_index=True, width="stretch")
        st.caption("Zufall: Median über fünf feste Instanzen (sonst die gewählte Instanz). Je genauer die Quelle, desto öfter ist die erkannte Basis schon optimal.")
elif step == 3 and xov is not None:
    if xov.status == "optimal":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Zielwertfehler", sci(a.xov_error), delta=f"Quelle: {sci(a.src_error)}", delta_color="off")
        c2.metric("Nichtnullen", str(a.xov_nnz), delta=f"Quelle: {a.src_nnz}; Ecke ≤ m = {a.m_rows}", delta_color="off")
        c3.metric("Pivots", str(xov.pivots), delta=f"Simplex von Null: {a.simplex.pivots}", delta_color="off")
        c4.metric("Duale", "exakt", delta="c·x = b·y bis 1e-9", delta_color="off")
        st.markdown(f"Die Näherung der Quelle hat **{a.src_nnz}** Nichtnullen, die Ecke nach dem Crossover **{a.xov_nnz}** (eine Basislösung hat höchstens m = {a.m_rows}). Der Zielwertfehler sinkt von {sci(a.src_error)} auf {sci(a.xov_error)}: "
                    "das ist die Rundungsgrenze, weil die Ecke aus einem linearen Gleichungssystem folgt und nicht mehr aus Iterationen.")
        y_txt = ", ".join(num(v, 3) for v in xov.y[:8]) + (" …" if len(xov.y) > 8 else "")
        st.caption(f"Exakte Duale (Schattenpreise) der Standardform: ({y_txt}).")
    tok = (settings.kind, settings.m, settings.n, settings.seed, settings.density_i, settings.scale_i)
    if st.button("Fehler über die Genauigkeit der Quelle berechnen", key="err_start"):
        ss["err_done"] = tok
    if ss.get("err_done") == tok:
        with st.spinner("Rechne..."):
            rows = ev.eps_sweep(settings)
        st.plotly_chart(build_errors(rows), width="stretch", key="s3_err")
        st.caption("Logarithmische Achse. Die Quellen fallen mit ε; der Crossover liegt bei jeder Genauigkeit an der Rundungsgrenze (Werte unter 10^-17 sind bei 10^-17 gezeichnet), sofern die Basis erkannt wird.")
elif step == 4 and xov is not None:
    fs, tot_src, tot_x = a.flops_simplex, a.flops_src, a.flops_xov
    st.markdown("**Simplex allein gegen Näherung + Crossover auf dieser Instanz** (Operationsmodell, kein Wandzeit-Vergleich):")
    st.dataframe(pd.DataFrame([{"Weg": "Simplex von Null (dichtes Tableau)", "Pivots / Iterationen": big(a.simplex.pivots), "Operationen (Modell)": big(fs)},
                               {"Weg": f"{C.SOURCE_SHORT[source]} + Crossover", "Pivots / Iterationen": f"{big(getattr(a.src, 'iterations', 0))} Iterationen + {xov.pivots} Pivots", "Operationen (Modell)": big(tot_src + tot_x)}]),
                 hide_index=True, width="stretch")
    if fs and xov.status == "optimal":
        ratio = (tot_src + tot_x) / fs
        st.markdown(f"Näherung + Crossover brauchen hier das **{ratio:.1f}-Fache** der Operationen des Simplex allein" + (" (der Simplex ist billiger)." if ratio > 1 else " (das Innere ist billiger)."))
    st.caption("Modell: Quelle wie in Stück 8 (Innere Punkte, dicht) und Stück 10 (PDLP, nur Nichtnullen); Crossover: Crash-Basis 2m²N, Zerlegung 2m³/3, Tableau 2m²(N+1), je Pivot 2(m+1)(N+1); Simplex: 2(m+1)(Spalten+1) je Pivot (dicht).")
    w = ev.warm_instance(settings)
    st.markdown(f"**Warmstart auf dieser Instanz:** die rechte Seite ändert sich um {C.change_label(settings.change_i)} (Regler links).")
    if w is None:
        st.markdown("Kein Warmstart-Vergleich möglich (kein Crossover-Ergebnis oder die Standardform hat Zeilen entfernt).")
    else:
        rows = [{"Verfahren": "Simplex von Null", "Aufwand": f"{w['simplex_cold']} Pivots"}, {"Verfahren": "dualer Simplex ab der Crossover-Basis", "Aufwand": f"{w['dual_warm']} Pivots"}]
        if "ipm_cold" in w:
            rows += [{"Verfahren": "Innere Punkte kalt / warm", "Aufwand": f"{w['ipm_cold']} / {num(w['ipm_warm'], 0)} Iterationen"}]
        if "pdlp_cold" in w:
            rows += [{"Verfahren": "PDLP kalt / warm", "Aufwand": f"{w['pdlp_cold']} / {w['pdlp_warm']} Iterationen"}]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    tok_s = (settings.density_i, settings.eps_i)
    if st.button("Gesamtkosten über die Größe berechnen", key="size_start"):
        ss["size_done"] = tok_s
    if ss.get("size_done") == tok_s:
        with st.spinner("Rechne..."):
            sweep = ev.size_sweep(settings)
        st.plotly_chart(build_size(sweep), width="stretch", key="s4_size")
        st.dataframe(pd.DataFrame([{"n": r["n"], "Simplex": big(r["flops_simplex"]), "Innere Punkte + Crossover": big(r["flops_ipm"]), "PDLP + Crossover": big(r["flops_pdlp"]), "Aufräum-Pivots (IP / PDLP)": f"{r['pivots_ipm']:.0f} / {r['pivots_pdlp']:.0f}"}
                                    for r in sweep["rows"]]), hide_index=True, width="stretch")
        st.caption(f"Zufallsinstanzen m = n, Dichte {settings.density:.0%}, ε = {settings.eps:g}, Median über drei feste Instanzen. "
                   + ("Ab n = " + str(sweep["cross_ipm"] or sweep["cross_pdlp"]) + " braucht eine der Kombinationen weniger als der Simplex allein. " if (sweep["cross_ipm"] or sweep["cross_pdlp"]) else "Der Simplex allein ist in diesem Bereich immer billiger. ")
                   + "Auf Zufallsinstanzen braucht der Simplex nur wenige Pivots; Crossover lohnt hier nur, wenn man Basis und Duale nach einer Näherung braucht.")
    tok_c = (settings.eps_i,)
    if st.button("Auf dem Würfel vergleichen", key="cube_start"):
        ss["cube_done"] = tok_c
    if ss.get("cube_done") == tok_c:
        with st.spinner("Rechne..."):
            sweep = ev.cube_sweep(settings)
        st.plotly_chart(build_cube(sweep), width="stretch", key="s4_cube")
        st.dataframe(pd.DataFrame([{"n": r["n"], "Simplex (Pivots)": big(r["pivots_simplex"]), "IP + Crossover (Aufräum-Pivots)": f"{r['pivots_ipm']:.0f}", "PDLP + Crossover (Aufräum-Pivots)": f"{r['pivots_pdlp']:.0f}",
                                    "Operationen Simplex": big(r["flops_simplex"]), "Operationen IP + Crossover": big(r["flops_ipm"])} for r in sweep["rows"]]), hide_index=True, width="stretch")
        st.caption(f"Klee-Minty-Würfel, Quelle mit ε = {settings.eps:g}. Ab n = {sweep['cross_ipm']} ist Näherung + Crossover im Modell billiger als der Simplex (2^n − 1 Pivots); die Aufräum-Pivots hängen stark an ε.")
    if st.button("Warmstart über alle Änderungen vergleichen", key="warm_start"):
        ss["warm_done"] = (settings.density_i,)
    if ss.get("warm_done") == (settings.density_i,):
        with st.spinner("Rechne..."):
            rows = ev.warm_sweep(settings)
        st.plotly_chart(build_warm(rows), width="stretch", key="s4_warm")
        st.dataframe(pd.DataFrame([{"Änderung": f"{r['p']:.0%}", "Simplex kalt": f"{r['simplex_cold']:.0f}", "dualer Simplex warm": f"{r['dual_warm']:.0f}", "IP kalt": f"{r['ipm_cold']:.0f}", "IP warm (10^-3)": f"{r['ipm_warm_small']:.0f}",
                                     "IP warm (1)": f"{r['ipm_warm_big']:.0f}", "PDLP kalt": f"{r['pdlp_cold']:.0f}", "PDLP warm": f"{r['pdlp_warm']:.0f}"} for r in rows]), hide_index=True, width="stretch")
        st.caption("Zufall 20 × 20, Median über fünf feste Instanzen. Der duale Simplex ab der alten Basis braucht nur wenige Pivots; die Inneren Punkte profitieren vom Warmstart nur bei kleinen Änderungen, und nur mit passender Verschiebung ins Innere; "
                   "PDLP spart wenig.")
elif step == 5:
    tok_d = (settings.source, settings.eps_i, settings.scale_i, settings.m, settings.n)
    st.markdown("**Wo wird die Basis falsch erkannt?** Zustände der Crash-Basis und Pivots über die Instanzarten (🔬 auf Abruf; gewählte Quelle und Genauigkeit, Zufall: Median über fünf Instanzen):")
    if st.button("Instanzarten vergleichen", key="deg_start"):
        ss["deg_done"] = tok_d
    if ss.get("deg_done") == tok_d:
        with st.spinner("Rechne..."):
            rows = ev.degeneracy(settings)
        st.plotly_chart(build_cases(rows), width="stretch", key="s5_cases")
        st.dataframe(pd.DataFrame([{"Instanzart": S.KIND_LABELS[r["kind"]].split(" (")[0], "Läufe": r["runs"], "Pivots (Median)": f"{r['pivots']:.0f}", "größter Wert": f"{r['max_pivots']:.0f}", "Nichtnullen Quelle": f"{r['nnz_src']:.0f}",
                                    "Nichtnullen Ecke": f"{r['nnz_xov']:.0f}", "m": f"{r['m']:.0f}"} for r in rows]), hide_index=True, width="stretch")
        st.caption("Auf Zufalls- und Mischinstanzen ist die Basis meist schon optimal. Auf dem Plateau liegt die Näherung im Inneren einer ganzen optimalen Fläche: der Indikator kann keine Ecke bevorzugen, das Aufräumen braucht viele duale Pivots. "
                   "Beim Klee-Minty-Würfel und entarteten Ecken bleiben je nach Genauigkeit Pivots übrig.")
    else:
        st.markdown("Das Aufräumen ist immer **richtig** (jede Basis führt am Ende zur optimalen Ecke; Tests belegen es auch bei Unsinn als Näherung), nur die **Zahl der Pivots** hängt an der Qualität der Basis-Erkennung.")

st.markdown("---")
st.markdown("## ⚙️ Der gewählte Fall")
m1, m2, m3, m4 = st.columns(4)
if xov is None:
    m1.metric("Quelle", C.SOURCE_SHORT[source], delta=a.src.status, delta_color="off")
    m2.metric("Aufräum-Pivots", "-", delta="kein Crossover", delta_color="off")
    m3.metric("Simplex von Null", str(a.simplex.pivots), delta=a.simplex.status, delta_color="off")
    m4.metric("Ergebnis", "-", delta="keine Ecke", delta_color="off")
else:
    m1.metric("Quelle", C.SOURCE_SHORT[source], delta=f"{getattr(a.src, 'iterations', 0)} Iterationen, ε = {settings.eps:g}", delta_color="off")
    m2.metric("Aufräum-Pivots", str(xov.pivots), delta=CASE_TEXT.get(xov.case, xov.case), delta_color="off")
    m3.metric("Simplex von Null", str(a.simplex.pivots), delta=a.simplex.status, delta_color="off")
    m4.metric("Ergebnis", num(-xov.obj) if xov.status == "optimal" else "-", delta=("Referenz " + num(a.ref_obj)) if math.isfinite(a.ref_obj) else "keine Referenz", delta_color="off")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Crossover macht den Simplex überflüssig.** | Auf Zufallsinstanzen braucht der Simplex nur wenige Pivots und ist im Operationsmodell viel billiger als Näherung + Crossover; Crossover lohnt, wenn man eine **Basis** braucht (Duale, Ranging, Warmstart) und die Näherung schon da ist, oder auf Instanzen, auf denen der Simplex schlecht läuft (Klee-Minty-Würfel). | Echte Großinstanzen |
| **Die Basis wird immer richtig erkannt.** | Bei grober Quelle, auf Plateaus (ganze optimale Fläche) und entarteten Ecken ist der Indikator mehrdeutig; dann bleiben Pivots. Das Ergebnis bleibt richtig, nur der Aufwand steigt. | Push-Phasen nach Bixby & Saltzman |
| **Aufräumen ist billig.** | Es ist ein dichtes Tableau: Zerlegung der Basis und Tableau kosten 2m²N-artig, unabhängig von der Zahl der Pivots. Echte Löser nutzen dünne LU-Zerlegungen mit Updates. | Dünne Lineare Algebra |
| **Die Näherung ist immer brauchbar.** | Die Quelle muss überhaupt eine Näherung liefern; bei unzulässigen oder unbeschränkten LPs gibt es keine Ecke und die Demo bricht mit dem Status der Quelle ab. Bei extrem schlechter Skalierung leidet der Indikator (er ist spaltennormiert, aber nicht vollständig skaleninvariant). | Präsolve und Skalierung (Stück 9) |
| **Warmstart gelingt bei jedem Verfahren.** | Der duale Simplex ab der alten Basis braucht nach einer Änderung nur wenige Pivots; Innere Punkte und PDLP starten mit dem alten Punkt nur bei kleinen Änderungen und passender Verschiebung besser als kalt. | Neuoptimierung (Stück 6) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Standardform:** $\min c^\top x$ unter $Mx = b$, $x \ge 0$; dual $\max b^\top y$ unter $M^\top y + s = c$, $s \ge 0$. **Indikator** mit Spaltennormen $d_j = \|M_j\|$: $\xi_j = \hat x_j/(\hat x_j + \hat s_j)$, $\hat x_j = x_j d_j$, $\hat s_j = s_j/d_j$ (invariant
unter $x \to x/t$, $M_j \to M_j t$). **Crash-Basis:** Spalten nach fallendem $\xi_j$, greedy unabhängig (Gram-Schmidt, relative Toleranz $10^{-8}$). **Zustand der Basis $B$:** $x_B = B^{-1}b$, $y = B^{-\top}c_B$, $r = c - M^\top y$;
optimal, wenn $x_B \ge 0$ und $r \ge 0$. **Kostenverschiebung:** $c_j \leftarrow c_j + \max(0, -r_j)$ für Nichtbasisvariablen macht $B$ dual zulässig; dualer Simplex bis $x_B \ge 0$, dann die ursprünglichen Kosten zurück und primaler Simplex.
**Warmstart:** alte Endbasis, neue rechte Seite: dual zulässig, primal unzulässig, also dualer Simplex.

**Literatur.** Megiddo, N. (1991). *On finding primal- and dual-optimal bases.* ORSA Journal on Computing 3(1), 63-65. Bixby, R. E., & Saltzman, M. J. (1994). *Recovering an optimal LP basis from an interior point solution.* Operations Research Letters 15(4), 169-178.
Applegate, D. u. a. (2021). *Practical large-scale linear programming using primal-dual hybrid gradient.* Advances in Neural Information Processing Systems 34 (PDLP als Quelle).

Implementiert in `xov_basis.py` (Indikator, Crash-Basis, Tableau, primaler und dualer Simplex, Crossover), `xov_ipm.py` und `xov_pdlp.py` (Quellen), `xov_algorithm.py` (Simplex von Null als Vergleich), `xov_evaluation.py`, `xov_scenario.py`.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
