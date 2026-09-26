"""Korrektheitskette Crossover: Ergebnis gegen HiGHS, Fixpunkt und Unsinnsquellen, Indikator, Crash-Basis, Simplex-Kerne, Kostenverschiebung, Plateau, Warmstart, Buchführung, Sonderfälle, Kopien."""

import itertools

import numpy as np
import pytest
from scipy.optimize import linprog

import xov_algorithm as A
import xov_basis as X
import xov_ipm as IP
import xov_pdlp as PD
import xov_scenario as S


def _std(inst):
    return IP.standard_form(inst)


def _ref(M, b, c):
    return linprog(c, A_eq=M, b_eq=b, bounds=(0, None), method="highs")


def _instances():
    yield S.textbook_instance()
    yield S.centre_instance()
    yield S.degenerate_instance()
    for n in range(2, 11):
        yield S.klee_minty_instance(n)
    for m, n in ((5, 6), (8, 4), (12, 12), (20, 30)):
        for density in (0.2, 0.6, 1.0):
            for seed in range(14):
                yield S.generate("random", m, n, density, seed)
    for m, n in ((6, 6), (10, 10), (16, 20)):
        for density in (0.3, 0.7):
            for seed in range(20):
                yield S.generate("mixed", m, n, density, seed)
    for seed in range(20):
        yield S.generate("plateau", 8, 10, 0.5, seed)


def _check_vertex(M, b, c, cr, ref_obj):
    m = M.shape[0]
    x, y = np.array(cr.x), np.array(cr.y)
    assert cr.status == "optimal" and cr.obj == pytest.approx(ref_obj, rel=1e-7, abs=1e-6)
    assert x.min() >= 0.0 and np.count_nonzero(x) <= m and len(cr.basis) == m and len(set(cr.basis)) == m
    nonbasic = np.ones(len(x), dtype=bool)
    nonbasic[list(cr.basis)] = False
    assert not x[nonbasic].any()                                                       # Ecke: außerhalb der Basis exakt null
    assert np.linalg.norm(M @ x - b) <= 1e-8 * (1 + np.linalg.norm(b))
    assert (c - M.T @ y).min() >= -1e-7 * (1 + np.abs(c).max())                         # dual zulässig
    assert abs(c @ x - b @ y) <= 1e-7 * (1 + abs(c @ x))                                # starke Dualität


# --- 1. Ergebnis gegen HiGHS ---------------------------------------------------------------------------------------------------------------------

def test_crossover_reaches_the_optimal_vertex_of_highs_from_ipm_and_pdlp_on_over_300_instances():
    count = ok = 0
    for inst in _instances():
        std = _std(inst)
        M, b, c = std[0], std[1], std[2]
        ref = _ref(M, b, c)
        assert ref.status == 0
        for src, eps in (("ipm", 1e-2), ("ipm", 1e-6), ("pdlp", 1e-2), ("pdlp", 1e-4)):
            r = IP.ipm(inst, "mehrotra", eps=eps, std=std) if src == "ipm" else PD.pdlp(inst, eps=eps, std=std)
            x, y = (np.array(r.z_full), np.array(r.y)) if src == "ipm" else (np.array(r.x_full), np.array(r.y))
            if not len(x) or r.status not in ("optimal", "limit"):
                continue
            cr = X.crossover(M, b, c, x, y)
            _check_vertex(M, b, c, cr, ref.fun)
            ok += 1
        count += 1
    assert count >= 300 and ok >= 4 * count * 0.97


# --- 2. Fixpunkt und Unsinnsquellen ---------------------------------------------------------------------------------------------------------------

def test_an_exact_or_slightly_perturbed_source_needs_no_pivot_and_a_garbage_source_still_ends_correct():
    rng = np.random.default_rng(0)
    checked = 0
    for seed in range(20):
        inst = S.generate("random", 10, 12, 0.6, seed)
        M, b, c = _std(inst)[:3]
        ref = _ref(M, b, c)
        y_ref = ref.eqlin.marginals
        if np.count_nonzero(ref.x > 1e-9) != M.shape[0]:
            continue                                                                        # entartet: Basis nicht eindeutig
        exact = X.crossover(M, b, c, ref.x, y_ref)
        assert exact.pivots == 0 and exact.case == "optimal" and set(np.nonzero(ref.x > 1e-9)[0]) == set(exact.basis)
        noisy = X.crossover(M, b, c, ref.x + 1e-9 * rng.random(len(ref.x)), y_ref + 1e-9 * rng.random(len(y_ref)))
        assert noisy.pivots == 0 and set(noisy.basis) == set(exact.basis)
        for x0, y0 in ((np.zeros(len(c)), np.zeros(len(b))), (rng.random(len(c)), rng.normal(size=len(b))), (np.ones(len(c)), np.ones(len(b)))):
            _check_vertex(M, b, c, X.crossover(M, b, c, x0, y0), ref.fun)
        checked += 1
    assert checked >= 12


# --- 3. Indikator und Crash-Basis ------------------------------------------------------------------------------------------------------------------

def test_indicator_by_hand_and_it_is_invariant_under_column_scaling():
    M = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0]])
    xi = X.indicator(M, [2.0, 0.0, 0.0], [0.0, 3.0, 0.0])
    assert xi[0] == 1.0 and xi[1] == 0.0 and xi[2] == 0.5
    assert X.priority_order([0.2, 0.9, 0.9, 0.5]) == [1, 2, 3, 0]
    rng = np.random.default_rng(1)
    M = rng.normal(size=(4, 7))
    x, s = rng.random(7), rng.random(7)
    t = 10.0 ** rng.integers(-4, 5, size=7)
    assert np.allclose(X.indicator(M, x, s), X.indicator(M * t, x / t, s * t), atol=1e-12)
    assert np.negative(np.ones(3)).tolist() == [-1, -1, -1] and X.indicator(M, -np.ones(7), -np.ones(7)).tolist() == [0.5] * 7


def test_crash_basis_takes_independent_columns_in_priority_order_and_fills_up():
    M = np.array([[1.0, 2.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    assert X.crash_basis(M, [1, 0, 2, 3]) == ([1, 2], 1)                                    # Spalte 0 ist ein Vielfaches von Spalte 1: übersprungen
    assert X.crash_basis(M, [2, 3, 0, 1]) == ([2, 3], 0)
    for seed in range(30):
        rng = np.random.default_rng(seed)
        M = rng.normal(size=(6, 14))
        order = list(rng.permutation(14))
        basis, skipped = X.crash_basis(M, order)
        assert len(basis) == 6 and skipped == 0 and basis == order[:6] and np.linalg.matrix_rank(M[:, basis]) == 6
    dep = np.hstack([np.eye(3), np.eye(3)])
    basis, skipped = X.crash_basis(dep, [0, 3, 1, 4, 2, 5])
    assert basis == [0, 1, 2] and skipped == 2 and np.linalg.matrix_rank(dep[:, basis]) == 3


# --- 4. Simplex-Kerne -------------------------------------------------------------------------------------------------------------------------------

def test_primal_and_dual_simplex_from_arbitrary_bases_reach_the_optimum_and_the_tableau_stays_consistent():
    inst = S.centre_instance()
    M, b, c = _std(inst)[:3]
    ref = _ref(M, b, c)
    slack = [5, 6, 7, 8]                                                                    # Anfangsbasis: die vier Schlupfvariablen (primal zulässig)
    for k in range(0, 8):
        tab = X.Tableau(M, b, c, slack)
        st, piv = X.primal_simplex(tab, [], 1e-9, 1e-9, max_pivots=k)
        assert tab.check_invariants() and piv <= k and (st == "optimal" or piv == k)
    tab = X.Tableau(M, b, c, slack)
    path = []
    st, piv = X.primal_simplex(tab, path, 1e-9, 1e-9)
    assert st == "optimal" and tab.obj() == pytest.approx(ref.fun) and piv == len(path) and all(p.kind == "primal" for p in path)
    objs = [p.obj for p in path]
    assert all(a >= b_ - 1e-9 for a, b_ in zip(objs, objs[1:]))                              # Minimierung: der Zielwert fällt
    b_new = b * np.array([1.0, 1.6, 0.7, 1.0])
    ref_new = _ref(M, b_new, c)
    cr = X.optimize_from_basis(M, b_new, c, tab.basis)                                       # alte Endbasis, neue rechte Seite: dual zulässig
    assert cr.case in ("dual", "optimal") and cr.status == "optimal" and cr.obj == pytest.approx(ref_new.fun) and all(p.kind == "dual" for p in cr.path) and cr.pivots_primal == 0
    objs = [p.obj for p in cr.path]
    assert all(a <= b_ + 1e-9 for a, b_ in zip(objs, objs[1:]))                              # der duale Simplex steigt zum Optimum


def test_every_basis_of_the_degenerate_fixture_ends_in_the_optimum_with_bland_as_a_safety_net():
    inst = S.degenerate_instance()
    M, b, c = _std(inst)[:3]
    ref = _ref(M, b, c)
    done = 0
    for basis in itertools.combinations(range(M.shape[1]), M.shape[0]):
        if abs(np.linalg.det(M[:, basis])) < 1e-9:
            continue
        cr = X.optimize_from_basis(M, b, c, list(basis))
        assert cr.status == "optimal" and cr.obj == pytest.approx(ref.fun)
        done += 1
    assert done >= 10


def test_cost_shifting_handles_bases_that_are_neither_primal_nor_dual_feasible_and_restores_the_costs():
    cases = {"optimal": 0, "primal": 0, "dual": 0, "beides unzulässig": 0}
    shifted = 0
    for seed in range(40):
        inst = S.generate("random", 6, 7, 0.6, seed)
        M, b, c = _std(inst)[:3]
        ref = _ref(M, b, c)
        rng = np.random.default_rng(seed)
        basis = list(rng.choice(M.shape[1], size=M.shape[0], replace=False))
        if abs(np.linalg.det(M[:, basis])) < 1e-6:
            continue
        cr = X.optimize_from_basis(M, b, c, basis)
        cases[cr.case] += 1
        shifted += cr.shifted
        assert cr.status == "optimal" and cr.obj == pytest.approx(ref.fun, rel=1e-7, abs=1e-6)
        assert (c - M.T @ np.array(cr.y)).min() >= -1e-7 * (1 + np.abs(c).max())              # gegen die ORIGINAL-Kosten dual zulässig
        assert cr.shifted == (cr.case == "beides unzulässig") and (not cr.shifted or (cr.pivots_dual + cr.pivots_primal == cr.pivots))
    assert cases["beides unzulässig"] >= 10 and shifted == cases["beides unzulässig"]


def test_infeasible_and_unbounded_programs_are_reported_by_the_cleanup():
    for inst, want in ((S.infeasible_instance(), "infeasible"), (S.unbounded_instance(), "unbounded")):
        std = _std(inst)
        M, b, c = std[:3]
        assert M is not None
        got = set()
        for basis in itertools.combinations(range(M.shape[1]), M.shape[0]):
            if abs(np.linalg.det(M[:, basis])) < 1e-9:
                continue
            got.add(X.optimize_from_basis(M, b, c, list(basis)).status)
        assert got == {want}


# --- 5. Plateau -------------------------------------------------------------------------------------------------------------------------------------

def test_on_the_plateau_pdlp_stays_inside_the_optimal_face_and_crossover_returns_a_vertex_of_it():
    inst = S.generate("plateau", 10, 12, 0.5, 35)
    std = _std(inst)
    M, b, c = std[:3]
    ref = _ref(M, b, c)
    r = PD.pdlp(inst, eps=1e-4, std=std)
    assert r.status == "optimal" and r.nnz_x > M.shape[0] + 5
    cr = X.crossover(M, b, c, np.array(r.x_full), np.array(r.y))
    _check_vertex(M, b, c, cr, ref.fun)
    assert np.count_nonzero(cr.x) <= M.shape[0] < r.nnz_x and cr.pivots >= 1 and abs(cr.obj - ref.fun) < abs(r.obj * -1 - ref.fun) + 1e-9


# --- 6. Warmstart -----------------------------------------------------------------------------------------------------------------------------------

def test_warm_starts_reach_the_same_optimum_as_a_cold_start():
    inst = S.generate("random", 12, 12, 0.5, 35)
    std = _std(inst)
    M, b, c = std[:3]
    first = X.crossover(M, b, c, *[np.array(v) for v in (IP.ipm(inst, "mehrotra", eps=1e-8, std=std).z_full, IP.ipm(inst, "mehrotra", eps=1e-8, std=std).y)])
    for pct in (0.05, 0.25):
        b2 = b * (1.0 + pct * np.sin(np.arange(len(b)) + 1.0))
        ref = _ref(M, b2, c)
        warm = X.optimize_from_basis(M, b2, c, list(first.basis))
        assert warm.status == "optimal" and warm.obj == pytest.approx(ref.fun, rel=1e-7)
        cold_ipm = IP.ipm(inst, "mehrotra", eps=1e-8, std=(M, b2, c, std[3], ""))
        z0, y0 = np.array(IP.ipm(inst, "mehrotra", eps=1e-8, std=std).z_full), np.array(first.y)
        warm_ipm = IP.ipm(inst, "mehrotra", eps=1e-8, std=(M, b2, c, std[3], ""), start=(z0 + 1e-2, y0, np.maximum(c - M.T @ y0, 0.0) + 1e-2))
        assert cold_ipm.status == "optimal" and warm_ipm.status in ("optimal", "numerical", "suspect", "limit")
        if warm_ipm.status == "optimal":
            assert -np.array(c) @ np.array(warm_ipm.z_full) == pytest.approx(-ref.fun, rel=1e-5)
        cold_pdlp = PD.pdlp(inst, eps=1e-6, std=(M, b2, c, std[3], ""))
        x0 = np.array(PD.pdlp(inst, eps=1e-6, std=std).x_full)
        warm_pdlp = PD.pdlp(inst, eps=1e-6, std=(M, b2, c, std[3], ""), start=(x0, y0))
        assert cold_pdlp.status == warm_pdlp.status == "optimal" and warm_pdlp.obj == pytest.approx(cold_pdlp.obj, rel=1e-4)


# --- 7. Buchführung und Sonderfälle ----------------------------------------------------------------------------------------------------------------

def test_bookkeeping_pivot_path_flops_and_determinism():
    inst = S.klee_minty_instance(10)
    std = _std(inst)
    M, b, c = std[:3]
    r = PD.pdlp(inst, eps=1e-2, std=std)
    a = X.crossover(M, b, c, np.array(r.x_full), np.array(r.y))
    b2 = X.crossover(M, b, c, np.array(r.x_full), np.array(r.y))
    assert a.pivots == len(a.path) > 0 and a.pivots_dual + a.pivots_primal == a.pivots and a.flops == X.setup_flops(*M.shape) + a.pivots * X.pivot_flops(*M.shape)
    assert a.basis == b2.basis and a.x == b2.x and [p.obj for p in a.path] == [p.obj for p in b2.path]
    assert X.setup_flops(3, 5) == int(2 * 9 * 5 + 2 * 27 / 3 + 2 * 9 * 6) and X.pivot_flops(3, 5) == 2 * 4 * 6
    assert A.solve(S.klee_minty_instance(10)).pivots == 2 ** 10 - 1 and 0 < a.pivots < 40


def test_special_cases_single_row_zero_objective_and_dependent_rows():
    one = S.Instance(((2.0, 3.0),), (12.0,), (4.0, 5.0), (S.LE,), ("a", "b"), ("r",), "custom")
    std = _std(one)
    M, b, c = std[:3]
    cr = X.crossover(M, b, c, np.zeros(3), np.zeros(1))
    _check_vertex(M, b, c, cr, _ref(M, b, c).fun)
    zero_c = S.Instance(((1.0, 1.0),), (5.0,), (0.0, 0.0), (S.LE,), ("a", "b"), ("r",), "custom")
    std = _std(zero_c)
    cr = X.crossover(std[0], std[1], std[2], np.ones(3), np.zeros(1))
    assert cr.status == "optimal" and cr.obj == 0.0
    dup = S.Instance(((1.0, 2.0), (1.0, 2.0), (2.0, 4.0)), (10.0, 10.0, 20.0), (3.0, 1.0), (S.EQ, S.EQ, S.EQ), ("a", "b"), ("r1", "r2", "r3"), "custom")
    std = _std(dup)
    assert std[0].shape[0] < 3                                                              # abhängige Zeilen entfernt (Standardform mit Rangprüfung)
    cr = X.crossover(std[0], std[1], std[2], np.ones(std[0].shape[1]), np.zeros(std[0].shape[0]))
    _check_vertex(std[0], std[1], std[2], cr, _ref(std[0], std[1], std[2]).fun)
    sing = X.optimize_from_basis(np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), [0, 1])
    assert sing.status == "numerical" and "singulär" in sing.note


def test_copies_are_faithful_simplex_centre_720_and_the_extended_results():
    assert A.solve(S.centre_instance()).obj == pytest.approx(720.0)
    inst = S.centre_instance()
    std = _std(inst)
    r = IP.ipm(inst, "mehrotra", eps=1e-8, std=std)
    assert r.status == "optimal" and len(r.z_full) == std[0].shape[1] and len(r.s_full) == std[0].shape[1] and 5 <= r.iterations <= 9
    p = PD.pdlp(inst, eps=1e-6, std=std)
    assert p.status == "optimal" and len(p.x_full) == std[0].shape[1]
    warm = PD.pdlp(inst, eps=1e-6, std=std, start=(p.x_full, p.y))
    assert warm.status == "optimal" and warm.iterations <= 8


def test_large_klee_minty_cubes_are_not_reported_unbounded_because_of_a_coarse_pivot_tolerance():
    """Regression: bei n >= 13 waren die Zahlen bis 5^n so groß, dass eine an b skalierte Toleranz Pivotelemente als null verwarf und den Würfel als unbeschränkt meldete."""
    for n in (12, 13, 14):
        inst = S.klee_minty_instance(n)
        std = _std(inst)
        M, b, c = std[:3]
        r = PD.pdlp(inst, eps=1e-2, std=std)
        cr = X.crossover(M, b, c, np.array(r.x_full), np.array(r.y))
        assert cr.status == "optimal" and cr.obj == pytest.approx(-(5.0 ** n), rel=1e-9) and cr.pivots >= 1
