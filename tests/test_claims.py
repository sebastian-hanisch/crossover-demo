"""Jede Zahl aus README und App über die echten Auswertungsfunktionen (Pivot-, Iterations- und Kreuzungspunkt-Zahlen nur als Bänder: Windows und Linux können abweichen)."""

import xov_evaluation as ev
import xov_scenario as S
from xov_evaluation import Settings


def _rows(s):
    return {(r["source"], r["k"]): r for r in ev.eps_sweep(s)}


def test_readme_accuracy_ladder_on_random_and_mixed_instances():
    for s, coarse_max in ((Settings(), 3), (Settings(m=30, n=30), 70), (Settings("mixed"), 3)):
        rows = _rows(s)
        for source in ("ipm", "pdlp"):
            assert rows[(source, 2)]["pivots"] <= 6 and rows[(source, 2)]["max_pivots"] <= coarse_max
            assert all(rows[(source, k)]["pivots"] == 0 and rows[(source, k)]["max_pivots"] <= 1 for k in (3, 4, 6, 8))
            assert all(rows[(source, k)]["optimal"] == 5 and rows[(source, k)]["err_xov"] < 1e-12 for k in (2, 3, 4, 6, 8))
            assert all(rows[(source, k)]["err_src"] > 1e-10 for k in (2, 3, 4))
        assert all(rows[(src, k)]["zero"] == 5 for src in ("ipm", "pdlp") for k in (4, 6, 8)) and rows[("ipm", 2)]["zero"] <= 4 and rows[("pdlp", 2)]["zero"] <= 4
    coarse = _rows(Settings(m=30, n=30))
    assert coarse[("pdlp", 2)]["pivots"] >= 2 and coarse[("pdlp", 2)]["zero"] == 0 and coarse[("ipm", 2)]["pivots"] >= 1 and coarse[("pdlp", 2)]["cases"]["beides unzulässig"] >= 2


def test_readme_the_plateau_needs_pivots_at_every_accuracy():
    rows = _rows(Settings("plateau", 10, 12))
    for source in ("ipm", "pdlp"):
        assert all(rows[(source, k)]["pivots"] >= 2 and rows[(source, k)]["zero"] == 0 and rows[(source, k)]["cases"]["dual"] == 5 and rows[(source, k)]["err_xov"] < 1e-12 for k in (2, 3, 4, 6, 8))
    assert all(rows[("pdlp", k)]["pivots"] >= 4 for k in (2, 3, 4, 6, 8)) and rows[("pdlp", 8)]["pivots"] >= rows[("pdlp", 2)]["pivots"] and rows[("ipm", 8)]["pivots"] <= 5


def test_readme_the_klee_minty_cube_and_the_crossing_point():
    coarse = {r["n"]: r for r in ev.cube_sweep(Settings("klee_minty", 12, 12, eps_i=0))["rows"]}
    assert all(coarse[n]["pivots_ipm"] <= 2 and coarse[n]["pivots_pdlp"] <= 2 for n in (2, 4, 6, 8)) and 5 <= coarse[10]["pivots_ipm"] <= 20 and 20 <= coarse[12]["pivots_ipm"] <= 90
    assert 150 <= coarse[14]["pivots_ipm"] <= 700 and coarse[14]["pivots_pdlp"] >= 60 and all(coarse[n]["pivots_simplex"] == 2 ** n - 1 for n in coarse)
    assert coarse[14]["pivots_ipm"] < coarse[14]["pivots_simplex"] / 20 and coarse[14]["flops_ipm"] < coarse[14]["flops_simplex"] / 20
    fine = ev.cube_sweep(Settings("klee_minty", 12, 12, eps_i=3))
    rows = {r["n"]: r for r in fine["rows"]}
    assert all(rows[n]["pivots_ipm"] <= 2 for n in (8, 10, 12, 14)) and rows[14]["pivots_pdlp"] >= 5
    for sweep in (ev.cube_sweep(Settings("klee_minty", 12, 12, eps_i=0)), fine):
        assert sweep["cross_ipm"] in (6, 7, 8, 9) and sweep["cross_pdlp"] in (7, 8, 9, 10, 11)


def test_readme_the_simplex_alone_stays_ahead_on_random_instances():
    for dens_i in (3, 0):
        sw = ev.size_sweep(Settings(density_i=dens_i, eps_i=2))
        rows = {r["n"]: r for r in sw["rows"]}
        assert sw["cross_ipm"] is None and sw["cross_pdlp"] is None and all(r["pivots_ipm"] <= 3 and r["pivots_pdlp"] <= 3 for r in rows.values())
        assert all(rows[n]["flops_ipm"] > 5 * rows[n]["flops_simplex"] and rows[n]["flops_pdlp"] > 5 * rows[n]["flops_simplex"] for n in rows)
    dense = {r["n"]: r for r in ev.size_sweep(Settings(density_i=3, eps_i=2))["rows"]}
    assert 12 <= dense[8]["flops_ipm"] / dense[8]["flops_simplex"] <= 30 and dense[100]["flops_pdlp"] < dense[100]["flops_ipm"] and dense[8]["flops_pdlp"] > 3 * dense[8]["flops_ipm"]


def test_readme_warm_starts_after_a_change_of_the_right_hand_side():
    rows = {round(r["p"], 2): r for r in ev.warm_sweep(Settings())}
    assert all(r["ipm_cold"] in (7, 8, 9) for r in rows.values())
    assert rows[0.01]["dual_warm"] <= 1 and rows[0.05]["dual_warm"] <= 1 and rows[0.1]["dual_warm"] <= 3 and 2 <= rows[0.25]["dual_warm"] <= 8 and all(8 <= r["simplex_cold"] <= 14 for r in rows.values())
    assert rows[0.01]["ipm_warm_small"] <= 4 and rows[0.05]["ipm_warm_small"] <= 5 and rows[0.25]["ipm_warm_small"] >= 12 and rows[0.25]["ipm_warm_big"] <= rows[0.25]["ipm_cold"] + 3
    assert all(r["ipm_warm_big"] < r["ipm_cold"] for r in (rows[0.01], rows[0.05], rows[0.1])) and rows[0.01]["pdlp_warm"] < rows[0.01]["pdlp_cold"] and rows[0.25]["pdlp_warm"] > 0.85 * rows[0.25]["pdlp_cold"]


def test_readme_a_hundred_fresh_instances_are_always_solved_and_mostly_without_a_pivot():
    counts = {}
    for eps in (1e-2, 1e-4):
        ok = zero = 0
        for sd in range(100):
            out = ev.run_one(S.generate("mixed", 8, 8, 0.5, 1000 + sd), "pdlp", eps)
            cr = out[0]
            ok += cr.status == "optimal"
            zero += cr.pivots == 0
        counts[eps] = (ok, zero)
    assert counts[1e-2][0] == counts[1e-4][0] == 100 and 55 <= counts[1e-2][1] <= 90 and counts[1e-4][1] >= 90
