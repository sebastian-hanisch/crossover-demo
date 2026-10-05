"""Orakel für den Crossover auf kleinen Zufalls-LPs mit ganzen Zahlen in Standardform (viele Entartungen, Unzulässige und Unbeschränkte): Ergebnis gegen HiGHS, die Basislösung wird mit eigenen
linearen Algebra-Mitteln nachgeprüft (Stützspalten linear unabhängig, B x_B = b, y = B^-T c_B), die Crash-Basis gegen eine Greedy-Wahl über den Rang der Präfixe, der Indikator von Hand;
dazu der Innere-Punkte-Löser auf Instanzen, deren Zielfunktion im Zeilenraum liegt (Komplementaritätsmaß 0 am Start; früher ZeroDivisionError)."""

import numpy as np
import pytest

import xov_basis as X
import xov_ipm as IP
import xov_pdlp as PD
import xov_scenario as S

linprog = pytest.importorskip("scipy.optimize").linprog


def _random_lp(rng):
    while True:
        m, N = int(rng.integers(1, 6)), 0
        N = int(rng.integers(m, 9))
        M = rng.integers(-2, 4, size=(m, N)).astype(float)
        if np.linalg.matrix_rank(M) == m:
            break
    b = M @ rng.integers(0, 4, size=N).astype(float) if rng.random() < 0.7 else rng.integers(-3, 6, size=m).astype(float)
    return M, b, rng.integers(-2, 5, size=N).astype(float)


def test_crossover_from_garbage_agrees_with_highs_and_returns_a_true_vertex():
    rng = np.random.default_rng(1)
    seen = {"optimal": 0, "infeasible": 0, "unbounded": 0}
    for _ in range(300):
        M, b, c = _random_lp(rng)
        ref = linprog(c, A_eq=M, b_eq=b, bounds=(0, None), method="highs")
        cr = X.crossover(M, b, c, rng.random(M.shape[1]) * rng.choice([0, 1, 10]), rng.normal(size=M.shape[0]) * rng.choice([0, 1]))
        if ref.status == 0:
            seen["optimal"] += 1
            assert cr.status == "optimal" and cr.obj == pytest.approx(ref.fun, rel=1e-7, abs=1e-7)
            x, y, basis = np.array(cr.x), np.array(cr.y), list(cr.basis)
            assert x.min() >= -1e-12 and np.abs(M @ x - b).max() <= 1e-8 * (1 + np.abs(b).max())
            supp = np.nonzero(x > 0)[0]
            assert len(supp) == 0 or np.linalg.matrix_rank(M[:, supp]) == len(supp)              # Ecke: Stützspalten linear unabhängig
            assert np.allclose(M[:, basis] @ x[basis], b, atol=1e-8) and np.allclose(y, np.linalg.solve(M[:, basis].T, c[basis]), atol=1e-8)
            assert (c - M.T @ y).min() >= -1e-7 and abs(b @ y - cr.obj) <= 1e-7 * (1 + abs(cr.obj))
        else:
            seen["infeasible" if linprog(np.zeros(len(c)), A_eq=M, b_eq=b, bounds=(0, None), method="highs").status == 2 else "unbounded"] += 1
            assert cr.status in ("infeasible", "unbounded")
    assert all(v > 20 for v in seen.values())


def test_indicator_and_crash_basis_agree_with_a_prefix_rank_oracle():
    rng = np.random.default_rng(3)
    for _ in range(150):
        M, _b, _c = _random_lp(rng)
        m, N = M.shape
        x, s = rng.random(N) * (rng.random(N) < 0.6), rng.random(N) * (rng.random(N) < 0.6)
        d = np.linalg.norm(M, axis=0)
        d[d == 0] = 1.0
        xh, sh = x * d, s / d
        expected = np.where(xh + sh > 0, xh / np.where(xh + sh > 0, xh + sh, 1.0), 0.5)
        xi = X.indicator(M, x, s)
        assert np.allclose(xi, expected)
        order = X.priority_order(xi)
        assert order == sorted(range(N), key=lambda j: (-expected[j], j))
        chosen = []
        for j in order:
            if len(chosen) < m and np.linalg.matrix_rank(M[:, chosen + [j]]) > len(chosen):
                chosen.append(j)
        assert X.crash_basis(M, order)[0] == chosen


def test_interior_point_start_with_zero_complementarity_does_not_crash():
    """Zielfunktion im Zeilenraum (hier c = 0, ein Dienst): mehrotra_start liefert s = 0, mu = 0. Früher ZeroDivisionError, jetzt ein ehrlicher Abbruch statt Optimum."""
    feasible = S._inst([[-1.0], [-2.0]], [5.0, 0.0], [0.0], [S.LE, S.GE], ["x"], ["r0", "r1"], "custom")
    infeasible = S._inst([[-1.0], [-2.0]], [1.0, 1.0], [2.0], [S.GE, S.EQ], ["x"], ["r0", "r1"], "custom")
    assert linprog([0.0], A_ub=[[-1.0], [2.0]], b_ub=[5.0, 0.0], bounds=(0, None), method="highs").status == 0
    assert linprog([-2.0], A_ub=[[1.0]], b_ub=[-1.0], A_eq=[[-2.0]], b_eq=[1.0], bounds=(0, None), method="highs").status == 2
    for inst in (feasible, infeasible):
        r = IP.ipm(inst, "mehrotra")
        assert r.status != "optimal" or inst is feasible
        PD.pdlp(inst)                                                                           # darf nicht abstürzen
