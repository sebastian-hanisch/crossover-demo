"""Crossover: aus einer Näherung im Inneren (Innere Punkte oder PDLP) eine optimale Ecke, exakte Duale und eine Basis gewinnen.

Standardform: min c^T x unter M x = b, x >= 0 (Zeilen unabhängig). Zwei Stufen:
  1. Basis erkennen: Indikator xi_j = x^_j / (x^_j + s^_j) mit x^_j = x_j ||M_j||, s^_j = s_j / ||M_j|| (s = c - M^T y; die Skalierung macht ihn unabhängig von den Einheiten der Spalten),
     Spalten nach fallendem Indikator; greedy m linear unabhängige Spalten (Gram-Schmidt): die Crash-Basis.
  2. Aufräumen: x_B = B^-1 b und die reduzierten Kosten r = c - M^T y_B an der Crash-Basis entscheiden: optimal (0 Pivots), nur primal zulässig (primaler Simplex), nur dual zulässig (dualer Simplex),
     beides unzulässig: Kostenverschiebung (c_j += max(0, -r_j) macht die Basis dual zulässig), dualer Simplex bis primal zulässig, Kosten zurück, primaler Simplex.
Dichtes Tableau wie in den Stücken 1-10; Pivots und Operationen (Modell) sind das Maß."""

from dataclasses import dataclass, field

import numpy as np

TOL = 1e-9
MAX_PIVOTS = 20_000
STALL_LIMIT = 25                            # so viele Nullschritte in Folge: bis zum nächsten echten Schritt gilt die Bland-Regel
PIVOT_TOL = 1e-9                           # Pivotelemente kleiner als dies (relativ zur größten Zahl der Spalte bzw. Zeile) gelten als null; die Zulässigkeitstoleranz hängt dagegen an der Größe von b
RANK_TOL = 1e-8                             # relative Restlänge einer Spalte nach Gram-Schmidt, ab der sie als abhängig gilt


def column_norms(M):
    d = np.linalg.norm(M, axis=0)
    d[d == 0] = 1.0
    return d


def indicator(M, x, s):
    """Skalenfreier Indikator xi_j = x^_j / (x^_j + s^_j) in [0, 1] (x^ = x ||M_j||, s^ = s / ||M_j||); nahe 1: Basiskandidat, nahe 0: Nichtbasis; 0.5, wenn beide null sind."""
    d = column_norms(M)
    xh, sh = np.maximum(np.asarray(x, dtype=float), 0.0) * d, np.maximum(np.asarray(s, dtype=float), 0.0) / d
    tot = xh + sh
    return np.where(tot > 0, xh / np.where(tot > 0, tot, 1.0), 0.5)


def priority_order(xi):
    """Spalten nach fallendem Indikator; Gleichstand: kleinster Index."""
    return sorted(range(len(xi)), key=lambda j: (-float(xi[j]), j))


def crash_basis(M, order, tol=RANK_TOL):
    """Greedy m linear unabhängige Spalten in Prioritätsreihenfolge (Gram-Schmidt mit relativer Toleranz). Spalten mit kleinem Rest werden übersprungen; fehlen am Ende welche, wird mit der kleineren
    Toleranz 1e-12 und zuletzt mit den übrigen Spalten aufgefüllt. Rückgabe (Basisindizes in Auswahlreihenfolge, Zahl der wegen Abhängigkeit übersprungenen Spalten mit hoher Priorität)."""
    m = M.shape[0]
    d = column_norms(M)
    chosen, Q, skipped = [], [], 0
    for t in (tol, 1e-12):
        for j in order:
            if len(chosen) == m:
                break
            if j in chosen:
                continue
            v = M[:, j] / d[j]
            for _ in range(2):
                for q in Q:
                    v = v - (q @ v) * q
            nv = float(np.linalg.norm(v))
            if nv > t:
                Q.append(v / nv)
                chosen.append(j)
            elif t == tol:
                skipped += 1
    for j in order:
        if len(chosen) == m:
            break
        if j not in chosen:
            chosen.append(j)
    return chosen, skipped


def basis_state(M, b, c, basis, tol=TOL):
    """x_B, Duale y, reduzierte Kosten r und der Zustand der Basis: 'optimal', 'primal' (nur primal zulässig), 'dual' (nur dual zulässig), 'beides unzulässig'; None bei singulärer Basis."""
    try:
        Bm = M[:, basis]
        xb = np.linalg.solve(Bm, b)
        y = np.linalg.solve(Bm.T, c[basis])
    except np.linalg.LinAlgError:
        return None
    r = c - M.T @ y
    tp, td = tol * (1.0 + float(np.max(np.abs(b)))), tol * (1.0 + float(np.max(np.abs(c))))
    pinf = float(max(0.0, -xb.min())) if len(xb) else 0.0
    nonbasic = np.ones(len(c), dtype=bool)
    nonbasic[basis] = False
    dinf = float(max(0.0, -r[nonbasic].min())) if nonbasic.any() else 0.0
    pf, df = pinf <= tp, dinf <= td
    kind = "optimal" if pf and df else ("primal" if pf else ("dual" if df else "beides unzulässig"))
    return {"x_B": xb, "y": y, "r": r, "primal_inf": pinf, "dual_inf": dinf, "state": kind}


def vertex_of(M, b, basis):
    """Basislösung (alle Variablen) zu einer Basis, auch wenn sie primal unzulässig ist; None bei singulärer Basis."""
    try:
        xb = np.linalg.solve(M[:, list(basis)], b)
    except np.linalg.LinAlgError:
        return None
    x = np.zeros(M.shape[1])
    x[list(basis)] = xb
    return x


def setup_flops(m, N):
    """Operationsmodell für Crash-Basis und Tableau: Gram-Schmidt 2 m² N, Zerlegung der Basis 2m³/3 und Tableau B^-1 [M | b] 2 m² (N+1)."""
    return int(2 * m * m * N + 2 * m ** 3 / 3.0 + 2 * m * m * (N + 1))


def pivot_flops(m, N):
    """Ein Pivot des dichten Tableaus (wie in den Stücken 1-10): 2 (m+1) (N+1)."""
    return 2 * (m + 1) * (N + 1)


@dataclass
class Move:
    kind: str                     # "dual" | "primal"
    row: int
    col: int
    leaving: int
    entering: int
    obj: float                    # Zielwert (Minimierung) nach dem Pivot (bei verschobenen Kosten: der verschobene)
    infeasibility: float          # Summe der negativen x_B nach dem Pivot
    basis: tuple = ()             # Basis nach dem Pivot (für die Wiedergabe der Ecken)


class Tableau:
    """Dichtes Tableau T = [B^-1 M | B^-1 b] mit Zielzeile (reduzierte Kosten r, rechts -c_B x_B); wird aus einer Basis frisch gerechnet."""

    def __init__(self, M, b, c, basis):
        self.M, self.b = M, b
        self.m, self.N = M.shape
        Bm = M[:, basis]
        self.T = np.zeros((self.m + 1, self.N + 1))
        self.T[: self.m, : self.N] = np.linalg.solve(Bm, M)
        self.T[: self.m, self.N] = np.linalg.solve(Bm, b)
        self.basis = list(basis)
        self.set_cost(c)

    def set_cost(self, c):
        self.c = np.array(c, dtype=float)
        cb = self.c[self.basis]
        self.T[self.m, : self.N] = self.c - cb @ self.T[: self.m, : self.N]
        self.T[self.m, self.N] = -float(cb @ self.T[: self.m, self.N])

    def x_basic(self):
        return self.T[: self.m, self.N]

    def reduced(self):
        return self.T[self.m, : self.N]

    def obj(self):
        return -float(self.T[self.m, self.N])

    def primal_infeasibility(self):
        return float(np.sum(np.maximum(-self.x_basic(), 0.0)))

    def pivot(self, row, col):
        T = self.T
        T[row] /= T[row, col]
        f = T[:, col].copy()
        f[row] = 0.0
        idx = np.nonzero(f)[0]
        if len(idx):
            T[idx] -= np.outer(f[idx], T[row])
        self.basis[row] = col

    def full_x(self):
        x = np.zeros(self.N)
        x[self.basis] = self.x_basic()
        return x

    def check_invariants(self, tol=1e-7):
        """Basisspalten sind Einheitsvektoren und das Tableau stimmt mit B^-1 [M | b] überein (für die Tests)."""
        Bm = self.M[:, self.basis]
        ref = np.linalg.solve(Bm, np.hstack([self.M, self.b[:, None]]))
        return bool(np.allclose(self.T[: self.m], ref, atol=tol * (1 + np.abs(ref).max())))


def _tols(M, b, c):
    return TOL * (1.0 + float(np.max(np.abs(b)))), TOL * (1.0 + float(np.max(np.abs(c))))


def primal_simplex(tab, path, tol_d, tol_p, max_pivots=MAX_PIVOTS):
    """Primaler Simplex (Minimierung) ab einer primal zulässigen Basis: Dantzig-Regel, nach STALL_LIMIT Nullschritten Bland. Rückgabe (Status, Pivots)."""
    pivots, stall = 0, 0
    while True:
        r = tab.reduced().copy()
        r[tab.basis] = 0.0
        cand = np.nonzero(r < -tol_d)[0]
        if len(cand) == 0:
            return "optimal", pivots
        if pivots >= max_pivots:
            return "limit", pivots
        col = int(cand[0]) if stall >= STALL_LIMIT else int(cand[np.argmin(r[cand])])
        colv = tab.T[: tab.m, col]
        rows = np.nonzero(colv > PIVOT_TOL * max(1.0, float(np.abs(colv).max())))[0]
        if len(rows) == 0:
            return "unbounded", pivots
        ratios = np.maximum(tab.x_basic()[rows], 0.0) / colv[rows]
        best = float(ratios.min())
        tied = rows[ratios <= best + 1e-12 * (1.0 + abs(best))]
        row = int(min(tied, key=lambda i: tab.basis[i]))
        leaving = tab.basis[row]
        stall = stall + 1 if best <= 1e-12 else 0
        tab.pivot(row, col)
        pivots += 1
        path.append(Move("primal", row, col, leaving, col, tab.obj(), tab.primal_infeasibility(), tuple(tab.basis)))


def dual_simplex(tab, path, tol_d, tol_p, max_pivots=MAX_PIVOTS):
    """Dualer Simplex ab einer dual zulässigen Basis (alle reduzierten Kosten >= 0): Zeile mit dem kleinsten x_B < 0, Spalte mit kleinstem r_j / |alpha_ij| über alpha_ij < 0. Rückgabe (Status, Pivots)."""
    pivots, stall = 0, 0
    while True:
        xb = tab.x_basic()
        cand = np.nonzero(xb < -tol_p)[0]
        if len(cand) == 0:
            return "optimal", pivots
        if pivots >= max_pivots:
            return "limit", pivots
        row = int(min(cand, key=lambda i: tab.basis[i])) if stall >= STALL_LIMIT else int(cand[np.argmin(xb[cand])])
        rowv = tab.T[row, : tab.N]
        cols = np.nonzero(rowv < -PIVOT_TOL * max(1.0, float(np.abs(rowv).max())))[0]
        if len(cols) == 0:
            return "infeasible", pivots
        ratios = np.maximum(tab.reduced()[cols], 0.0) / -rowv[cols]
        best = float(ratios.min())
        tied = cols[ratios <= best + 1e-12 * (1.0 + abs(best))]
        col = int(tied.min())
        leaving = tab.basis[row]
        stall = stall + 1 if best <= 1e-12 else 0
        tab.pivot(row, col)
        pivots += 1
        path.append(Move("dual", row, col, leaving, col, tab.obj(), tab.primal_infeasibility(), tuple(tab.basis)))


@dataclass
class CrossoverResult:
    status: str                                       # "optimal" | "infeasible" | "unbounded" | "limit" | "numerical"
    basis: tuple = ()                                 # optimale Basis (Spaltenindizes der Standardform)
    crash_basis: tuple = ()
    order: tuple = ()                                 # Prioritätsreihenfolge der Spalten
    indicator: tuple = ()
    x: tuple = ()                                     # Ecke (alle Variablen der Standardform)
    y: tuple = ()                                     # exakte Duale
    obj: float = float("nan")                         # Zielwert der Minimierung
    case: str = ""                                    # Zustand der Crash-Basis: optimal | primal | dual | beides unzulässig
    pivots_dual: int = 0
    pivots_primal: int = 0
    shifted: bool = False                             # Kostenverschiebung war nötig
    skipped: int = 0                                  # wegen linearer Abhängigkeit übersprungene Spalten mit hoher Priorität
    path: list = field(default_factory=list)
    flops: int = 0
    note: str = ""

    @property
    def pivots(self):
        return self.pivots_dual + self.pivots_primal


def optimize_from_basis(M, b, c, basis, crash=None, order=(), xi=(), skipped=0, max_pivots=MAX_PIVOTS):
    """Optimum ab einer gegebenen Basis: die Fälle (a)-(d) der Beschreibung oben. Das ist zugleich der Warmstart (alte Endbasis, neue rechte Seite: dual zulässig, primal unzulässig)."""
    M, b, c = np.asarray(M, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    m, N = M.shape
    res = CrossoverResult(status="numerical", basis=tuple(basis), crash_basis=tuple(crash if crash is not None else basis), order=tuple(order), indicator=tuple(xi), skipped=skipped)
    with np.errstate(all="ignore"):
        state = basis_state(M, b, c, list(basis))
        if state is None or not np.all(np.isfinite(state["x_B"])):
            res.note = "Basis singulär"
            return res
        res.case = state["state"]
        tol_p, tol_d = _tols(M, b, c)
        tab = Tableau(M, b, c, list(basis))
        status = "optimal"
        if res.case == "primal":
            status, res.pivots_primal = primal_simplex(tab, res.path, tol_d, tol_p, max_pivots)
        elif res.case == "dual":
            status, res.pivots_dual = dual_simplex(tab, res.path, tol_d, tol_p, max_pivots)
        elif res.case == "beides unzulässig":
            shift = np.maximum(-tab.reduced(), 0.0)
            shift[tab.basis] = 0.0
            res.shifted = True
            tab.set_cost(c + shift)
            status, res.pivots_dual = dual_simplex(tab, res.path, tol_d, tol_p, max_pivots)
            if status == "optimal":
                tab.set_cost(c)
                status, res.pivots_primal = primal_simplex(tab, res.path, tol_d, tol_p, max_pivots)
        res.status = status
        res.flops = setup_flops(m, N) + res.pivots * pivot_flops(m, N)
        if status != "optimal":
            res.note = {"infeasible": "das LP ist unzulässig (dualer Simplex ohne Eintrittsspalte)", "unbounded": "das LP ist unbeschränkt (primaler Simplex ohne Austrittszeile)", "limit": "Pivotgrenze erreicht"}[status]
            return res
        fin = basis_state(M, b, c, tab.basis)
        if fin is None or fin["state"] != "optimal":
            res.status, res.note = "numerical", "Endbasis nicht optimal (numerisch)"
            return res
        x = np.zeros(N)
        x[tab.basis] = fin["x_B"]
        x[np.abs(x) < 1e-14] = 0.0
        res.basis, res.x, res.y, res.obj = tuple(tab.basis), tuple(float(v) for v in x), tuple(float(v) for v in fin["y"]), float(c @ x)
    return res


def crossover(M, b, c, x, y, s=None, max_pivots=MAX_PIVOTS):
    """Crossover aus einer Näherung (x, y) des Inneren: Basis erkennen, dann optimieren. s = c - M^T y, wenn nicht gegeben."""
    M, b, c = np.asarray(M, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    s = c - M.T @ y if s is None else np.asarray(s, dtype=float)
    xi = indicator(M, x, s)
    order = priority_order(xi)
    basis, skipped = crash_basis(M, order)
    return optimize_from_basis(M, b, c, basis, crash=basis, order=order, xi=xi, skipped=skipped, max_pivots=max_pivots)
