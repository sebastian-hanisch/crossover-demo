"""Auswertung: Einstellungen, Analyse gegen Referenz, Ecken-Pfad, Genauigkeitsleiter, Gesamtkosten, Würfel, Warmstart, Zustände der Crash-Basis."""

import math

import numpy as np
import pytest

import xov_basis as X
import xov_constants as C
import xov_evaluation as ev
import xov_scenario as S


def test_settings_clamp_the_index_controls():
    s = ev.Settings(density_i=99, eps_i=-3, scale_i=99, change_i=99)
    assert s.density == 1.0 and s.eps == 1e-2 and s.scale_exp == 8 and s.change == 0.25
    d = ev.Settings()
    assert d.density == 0.5 and d.eps == 1e-2 and d.scale_exp == 0 and d.change == 0.05 and d.source == "pdlp"


def test_analyse_returns_the_exact_vertex_and_compares_with_the_simplex_and_is_cached():
    a = ev.analyse(ev.Settings(eps_i=2))
    assert a is ev.analyse(ev.Settings(eps_i=2))
    assert a.src.status == "optimal" and a.xov.status == "optimal" and a.ref_status == "optimal" and a.xov_error < 1e-12 < a.src_error and a.xov_nnz <= a.m_rows and a.flops_xov > 0 and a.flops_simplex > 0
    for source in C.SOURCES:
        b = ev.analyse(ev.Settings("mixed", source=source))
        assert b.xov.status == "optimal" and b.xov_error < 1e-12 and b.source == source
    for kind in ("infeasible", "unbounded"):
        c = ev.analyse(ev.Settings(kind))
        assert c.xov is None and c.x is None and c.src.status in ("infeasible", "unbounded", "limit", "numerical") and c.simplex.status == kind
    scaled = ev.analyse(ev.Settings(m=10, n=10, scale_i=3))
    assert scaled.xov.status == "optimal" and scaled.xov_error < 1e-9


def test_vertex_path_starts_at_the_crash_vertex_and_ends_at_the_optimum_for_two_variables():
    a = ev.analyse(ev.Settings("degenerate"))
    path = ev.vertex_path(a)
    assert len(path) == a.xov.pivots + 1 and path[-1][2] and (path[-1][0], path[-1][1]) == pytest.approx(tuple(a.simplex.x[:2]))
    assert ev.vertex_path(ev.analyse(ev.Settings("centre"))) == [] and ev.vertex_path(ev.analyse(ev.Settings("infeasible"))) == []
    coarse = ev.analyse(ev.Settings("textbook", eps_i=0))
    assert len(ev.vertex_path(coarse)) == coarse.xov.pivots + 1


def test_family_uses_five_fixed_seeds_for_random_kinds_and_one_instance_otherwise():
    assert len(ev.family(ev.Settings())) == 5 and len(ev.family(ev.Settings("centre"))) == 1 and len(ev.family(ev.Settings("plateau"))) == 5 and len(ev.family(ev.Settings(), kind="klee_minty")) == 1
    assert ev.family(ev.Settings(scale_i=2))[0] != ev.family(ev.Settings())[0]


def test_eps_sweep_rows_and_exactness():
    rows = ev.eps_sweep(ev.Settings())
    assert [(r["source"], r["k"]) for r in rows] == [(s, k) for s in C.SOURCES for k in C.EPS_SWEEP_EXPS] and all(r["runs"] == 5 and r["optimal"] == 5 for r in rows)
    assert all(r["err_xov"] < 1e-12 and sum(r["cases"].values()) == 5 and r["zero"] <= 5 for r in rows)
    assert all(r["err_src"] > r["err_xov"] for r in rows)
    fixed = ev.eps_sweep(ev.Settings("centre"))
    assert all(r["runs"] == 1 for r in fixed)


def test_size_and_cube_and_warm_sweeps_have_the_documented_shape():
    sw = ev.size_sweep(ev.Settings(eps_i=2))
    assert [r["n"] for r in sw["rows"]] == list(C.SIZE_SIZES) and sw["cross_ipm"] is None and all(r["flops_simplex"] > 0 and r["flops_ipm"] > r["flops_simplex"] for r in sw["rows"])
    cube = ev.cube_sweep(ev.Settings(eps_i=3))
    rows = cube["rows"]
    assert [r["n"] for r in rows] == list(C.CUBE_SIZES) and all(r["pivots_simplex"] == 2 ** r["n"] - 1 and r["status_ipm"] == "optimal" and r["status_pdlp"] == "optimal" for r in rows)
    assert rows[-1]["flops_ipm"] < rows[-1]["flops_simplex"] / 20 and cube["cross_ipm"] in (6, 7, 8, 9)
    warm = ev.warm_sweep(ev.Settings())
    assert [r["p"] for r in warm] == list(C.CHANGES) and all(r["dual_warm"] < r["simplex_cold"] and r["ipm_cold"] > 0 for r in warm) and warm[0]["dual_warm"] <= warm[-1]["dual_warm"]


def test_changed_rhs_and_warm_instance_edge_cases():
    b = np.array([10.0, 20.0, 30.0])
    assert ev.changed_rhs(b, 0.0).tolist() == b.tolist() and ev.changed_rhs(b, 0.1)[0] == pytest.approx(10.0 * (1 + 0.1 * math.sin(1.0)))
    assert ev.warm_instance(ev.Settings("infeasible")) is None
    w = ev.warm_instance(ev.Settings())
    assert w["dual_status"] == "optimal" and w["simplex_status"] == "optimal" and w["dual_warm"] <= w["simplex_cold"] and "ipm_cold" in w and "pdlp_cold" in w
    dup = S.Instance(((1.0, 2.0), (1.0, 2.0)), (10.0, 10.0), (3.0, 1.0), (S.EQ, S.EQ), ("a", "b"), ("r1", "r2"), "custom")
    assert ev.std_nnz([0.0, 0.0]) == 0 and ev.std_nnz([1.0, 1e-12, 0.0]) == 1 and dup.m == 2


def test_degeneracy_rows_cover_the_instance_kinds():
    rows = {r["kind"]: r for r in ev.degeneracy(ev.Settings())}
    assert list(rows) == list(C.DEGENERACY_KINDS) and rows["random"]["runs"] == 5 and rows["degenerate"]["runs"] == 1 and all(sum(r["cases"].values()) == r["runs"] for r in rows.values())
    assert rows["plateau"]["pivots"] > rows["random"]["pivots"] and rows["plateau"]["nnz_src"] > rows["plateau"]["nnz_xov"] and X.PIVOT_TOL > 0
