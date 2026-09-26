"""Presets: gültige Werte und jede Zahl der Hilfetexte gegen die echten Auswertungsfunktionen (Iterationszahlen, Pivotzahlen bei Entartung und Kreuzungspunkte nur als Bänder: Windows und Linux können abweichen)."""

import pytest

import xov_constants as C
import xov_evaluation as ev
from xov_presets import PRESET_KEYS, SETTING_SPECS


def _settings(name, **over):
    p = {**C.PRESETS[name], **over}
    return ev.Settings(p["kind"], p["m"], p["n"], p["seed"], p["density"], p["source"], p["eps"], p["scale"], p["change"])


def _has(name, *values):
    for v in values:
        assert v in C.PRESET_HELP[name], (name, v)


def test_every_preset_has_valid_values_and_a_help_text():
    assert list(C.PRESETS) == list(C.PRESET_HELP) and len(C.PRESETS) == 12
    for name, p in C.PRESETS.items():
        assert set(p) <= set(PRESET_KEYS) and {"kind", "step", "density", "source", "eps", "scale", "change"} <= set(p), name
        for key, state_key in PRESET_KEYS.items():
            if key in p and state_key in SETTING_SPECS:
                spec = SETTING_SPECS[state_key]
                assert spec.caster(p[key]) == p[key], (name, key)
                if spec.lo is not None:
                    assert spec.lo <= p[key] <= spec.hi, (name, key)
        assert C.PRESET_HELP[name].strip()


def test_help_degenerate_and_centre_presets():
    name = "Entartete Ecke: ein Pivot"
    a = ev.analyse(_settings(name))
    assert a.xov.status == "optimal" and a.xov.case == "primal" and a.xov.pivots == 1 and a.simplex.pivots == 2 and 3e-4 < a.src_error < 2e-3 and a.xov_error < 1e-12 and a.ref_obj == pytest.approx(36.0)
    _has(name, "2 Dienste, 4 Ressourcen", "36 Iterationen", "7.4e-4", "1 primalen Pivot", "2 Pivots", "Optimum 36")
    name = "Zentrum: aus 10^-2 wird exakt"
    a = ev.analyse(_settings(name))
    assert a.xov.pivots == 0 and a.xov.case == "optimal" and a.support_recall == 1.0 and 2e-3 < a.src_error < 1e-2 and a.xov_error < 1e-12 and a.simplex.pivots == 4 and len(a.xov.basis) == 4
    _has(name, "76 Iterationen", "5.3e-3", "0 Pivots", "alle 4 Spalten", "1.6e-16", "4 Pivots")


def test_help_coarse_and_fine_source_presets():
    name = "Grobe Quelle: ein paar Pivots"
    s = _settings(name)
    a = ev.analyse(s)
    assert a.xov.status == "optimal" and a.xov.case == "primal" and 2 <= a.xov.pivots <= 12 and 0.85 <= a.support_recall < 1.0 and a.simplex.pivots == 6
    rows = {(r["source"], r["k"]): r for r in ev.eps_sweep(s)}
    assert 2 <= rows[("pdlp", 2)]["pivots"] <= 10 and rows[("pdlp", 2)]["max_pivots"] >= 10 and rows[("pdlp", 4)]["pivots"] == 0 and rows[("pdlp", 2)]["zero"] == 0
    _has(name, "30 × 30", "92 Iterationen", "5 Pivots", "der Simplex von Null 6", "93 %", "Median bei 5 Pivots", "größter Wert 58", "ε = 10^-4 bei 0")
    name = "Feine Quelle: null Pivots"
    a = ev.analyse(_settings(name))
    assert a.xov.pivots == 0 and a.support_recall == 1.0 and a.src_error < 1e-6 and a.xov_error < 1e-12 and a.simplex.pivots == 6 and a.flops_simplex == 22692
    assert 1.0e6 < a.flops_src < 1.5e6 and 1.5e5 < a.flops_xov < 3.5e5
    _has(name, "30 × 30", "9 Iterationen", "0 Pivots", "alle 30", "1.3e-7", "1.220.400", "235.800", "22.692", "6 Pivots")


def test_help_plateau_and_cube_presets():
    name = "Plateau: aus dem Inneren zur Ecke"
    a = ev.analyse(_settings(name))
    assert a.xov.status == "optimal" and a.xov.case == "dual" and 8 <= a.xov.pivots <= 45 and a.src_nnz >= 18 and a.xov_nnz == 10 and a.m_rows == 10 and a.simplex.pivots == 3 and a.xov_error < 1e-12
    _has(name, "10 × 12", "116 Iterationen", "21 Nichtnullen", "m = 10", "nur dual zulässig", "21 duale Pivots", "10 Nichtnullen", "1.8e-4", "3 Pivots")
    name = "Klee-Minty-Würfel: wenige gegen Tausende Pivots"
    a = ev.analyse(_settings(name))
    assert a.xov.status == "optimal" and a.xov.case == "primal" and 25 <= a.xov.pivots <= 70 and a.simplex.pivots == 2 ** 12 - 1 and a.flops_simplex == 2661750 and a.flops_src + a.flops_xov < a.flops_simplex / 5
    _has(name, "n = 12", "12 Iterationen", "42 primale Pivots", "4.095", "127.296", "42.564", "2.661.750")
    name = "Würfel: hier lohnt sich das Innere"
    s = _settings(name)
    a = ev.analyse(s)
    assert a.xov.status == "optimal" and a.xov.pivots <= 6 and a.simplex.pivots == 2 ** 14 - 1 and a.flops_simplex == 14253210 and a.flops_simplex / (a.flops_src + a.flops_xov) >= 30
    rows = {r["n"]: r for r in ev.cube_sweep(_settings("Würfel: hier lohnt sich das Innere", eps=0))["rows"]}
    assert 150 <= rows[14]["pivots_ipm"] <= 700
    _has(name, "n = 14", "16 Iterationen", "1 Pivot statt 16.383", "257.440", "25.043", "14.253.210", "50-mal", "341 Pivots")


def test_help_size_and_warm_and_scaling_presets():
    name = "Zufall: der Simplex bleibt vorn"
    a = ev.analyse(_settings(name))
    assert a.xov.pivots == 0 and 20 <= a.simplex.pivots <= 35 and 10 < (a.flops_src + a.flops_xov) / a.flops_simplex < 25
    _has(name, "40 × 40", "7 Iterationen", "0 Pivots", "2.170.931", "557.866", "179.334", "27 Pivots", "15.2-Fache")
    name = "Warmstart nach einer Änderung"
    w = ev.warm_instance(_settings(name))
    assert w["dual_status"] == "optimal" and w["dual_warm"] <= 6 and w["simplex_cold"] >= 8 and w["dual_warm"] < w["simplex_cold"] and w["ipm_warm"] <= w["ipm_cold"] + 4 and w["pdlp_warm"] < w["pdlp_cold"]
    _has(name, "20 × 20", "5 %", "3 Pivots", "12", "8 Iterationen kalt und 8 warm", "1.000 kalt und 580 warm")
    name = "Schlechte Skalierung"
    a = ev.analyse(_settings(name))
    assert a.xov.status == "optimal" and a.xov.pivots <= 2 and a.xov_error < 1e-10
    _has(name, "10 × 10", "10^6", "352 Iterationen", "0 Pivots")


def test_help_cases_and_infeasible_presets():
    name = "Zustände der Crash-Basis"
    rows = {r["kind"]: r for r in ev.degeneracy(_settings(name))}
    assert rows["random"]["pivots"] == 0 and rows["random"]["max_pivots"] <= 3 and rows["mixed"]["pivots"] == 0 and rows["degenerate"]["pivots"] == 1 and 25 <= rows["klee_minty"]["pivots"] <= 70
    assert 15 <= rows["plateau"]["pivots"] <= 60 and rows["plateau"]["cases"]["dual"] == 5 and rows["plateau"]["nnz_src"] > rows["plateau"]["m"] >= rows["plateau"]["nnz_xov"]
    _has(name, "12 × 12", "0 Pivots im Median", "größter Wert 1", "entartete Ecke 1", "Klee-Minty-Würfel 42", "Plateau 32", "größter Wert 38", "in allen 5 Läufen nur dual zulässig")
    name = "Unzulässig: kein Crossover"
    a = ev.analyse(_settings(name))
    assert a.src.status == "infeasible" and a.xov is None and 100 <= a.src.iterations <= 400
    _has(name, "192 Iterationen", "Farkas", "keine Ecke")
