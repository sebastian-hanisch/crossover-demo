"""Auswertung von Crossover: ein Lauf (Quelle, Crash-Basis, Aufräumen, Simplex von Null), Basis-Erkennung und Pivots über die Genauigkeit, Gesamtkosten im Operationsmodell, Klee-Minty-Würfel, Warmstart,
Zustandsverteilung über Instanzarten."""

import statistics
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import xov_algorithm as A
import xov_basis as X
import xov_constants as C
import xov_ipm as IP
import xov_pdlp as PD
import xov_scenario as S

RANDOM_KINDS = ("random", "mixed", "plateau")
SOURCE_CAP = 100000


@dataclass(frozen=True)
class Settings:
    kind: str = "random"
    m: int = C.DEFAULT_M
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    density_i: int = C.DEFAULT_DENSITY_I
    source: str = "pdlp"
    eps_i: int = C.DEFAULT_EPS_I
    scale_i: int = C.DEFAULT_SCALE_I
    change_i: int = C.DEFAULT_CHANGE_I

    @property
    def density(self):
        return C.DENSITIES[min(max(self.density_i, 0), len(C.DENSITIES) - 1)]

    @property
    def eps(self):
        return 10.0 ** -C.EPS_EXPS[min(max(self.eps_i, 0), len(C.EPS_EXPS) - 1)]

    @property
    def scale_exp(self):
        return C.SCALE_EXPS[min(max(self.scale_i, 0), len(C.SCALE_EXPS) - 1)]

    @property
    def change(self):
        return C.CHANGES[min(max(self.change_i, 0), len(C.CHANGES) - 1)]


def base_instance(s, seed=None, kind=None):
    return S.generate(kind or s.kind, s.m, s.n, s.density, s.seed if seed is None else seed)


def instance_of(s, seed=None, kind=None):
    return S.column_scaled(base_instance(s, seed, kind), s.scale_exp)


def _med(values):
    return statistics.median(values) if values else float("nan")


def simplex_flops(inst, solution):
    """Operationsmodell des dichten Tableau-Simplex von Null: je Pivot 2 (m+1)(Spalten+1)."""
    return solution.pivots * 2 * (inst.m + 1) * (A.standard_form(inst)[2]["ncols"] + 1)


def std_nnz(vec, tol=1e-9):
    vec = np.asarray(vec, dtype=float)
    return int(np.sum(np.abs(vec) > tol * max(1.0, float(np.max(np.abs(vec))) if len(vec) else 1.0)))


def source_point(inst, std, source, eps, start=None):
    """Näherung der Quelle in der Standardform: (x, y, s, Ergebnisobjekt); x ist None, wenn die Quelle nichts liefert (unzulässig, unbeschränkt, gescheitert)."""
    M, b, c = std[:3]
    if source == "ipm":
        r = IP.ipm(inst, "mehrotra", eps=eps, std=std, start=start)
        if r.status in ("optimal", "limit") and r.z_full:
            return np.array(r.z_full), np.array(r.y), np.array(r.s_full), r
        return None, None, None, r
    r = PD.pdlp(inst, eps=eps, max_iter=SOURCE_CAP, std=std, start=start)
    if r.status in ("optimal", "limit") and r.x_full:
        y = np.array(r.y)
        return np.array(r.x_full), y, c - M.T @ y, r
    return None, None, None, r


@dataclass
class Analysis:
    inst: object
    base: object
    std: tuple                       # (M, b, c, n, note) der Standardform mit Rangprüfung; M None: Zeilen widersprechen sich
    source: str
    src: object                      # IPMResult bzw. PDLPResult
    x: object                        # Näherung der Quelle (alle Variablen) oder None
    y: object
    s: object
    xov: object                      # CrossoverResult oder None
    simplex: object                  # Simplex von Null (eigener Löser, Stück 1)
    ref_status: str
    ref_obj: float                   # Optimalwert der Maximierung (Simplex auf der unskalierten Instanz)
    src_error: float                 # relativer Zielwertfehler der Quelle
    xov_error: float                 # relativer Zielwertfehler nach dem Crossover
    src_nnz: int
    xov_nnz: int
    m_rows: int
    flops_src: int
    flops_xov: int
    flops_simplex: int
    support_recall: float            # Anteil der Träger-Spalten der exakten Ecke in der Crash-Basis (nur bei nicht entarteter exakter Ecke, sonst nan)


@lru_cache(maxsize=64)
def analyse(s):
    inst, base = instance_of(s), base_instance(s)
    std = IP.standard_form(inst)
    sol, ref = A.solve(inst), A.solve(base)
    ref_obj = ref.obj if ref.status == "optimal" else float("nan")
    fs = simplex_flops(inst, sol) if sol.status == "optimal" else 0
    if std[0] is None:
        return Analysis(inst, base, std, s.source, IP.IPMResult(status="infeasible", note=std[4]), None, None, None, None, sol, ref.status, ref_obj, float("nan"), float("nan"), 0, 0, 0, 0, 0, fs, float("nan"))
    M, b, c = std[:3]
    x, y, sv, src = source_point(inst, std, s.source, s.eps)
    if x is None:
        return Analysis(inst, base, std, s.source, src, None, None, None, None, sol, ref.status, ref_obj, float("nan"), float("nan"), 0, 0, M.shape[0], getattr(src, "flops", 0), 0, fs, float("nan"))
    cr = X.crossover(M, b, c, x, y, s=sv)
    src_err = abs(-float(c @ x) - ref_obj) / (1.0 + abs(ref_obj)) if ref.status == "optimal" else float("nan")
    xov_err = abs(-cr.obj - ref_obj) / (1.0 + abs(ref_obj)) if cr.status == "optimal" and ref.status == "optimal" else float("nan")
    recall = float("nan")
    if ref.status == "optimal" and cr.status == "optimal":
        xs = np.array(ref.x, dtype=float)
        slack = [inst.b[i] - np.array(inst.A[i]) @ xs if inst.senses[i] == S.LE else np.array(inst.A[i]) @ xs - inst.b[i] for i in range(inst.m) if inst.senses[i] != S.EQ]
        full = np.concatenate([xs, slack]) if slack else xs
        if len(full) == M.shape[1]:
            supp = set(np.nonzero(np.abs(full) > 1e-9 * max(1.0, float(np.max(np.abs(full)))))[0])
            if len(supp) == M.shape[0]:
                recall = len(supp & set(cr.crash_basis)) / len(supp)
    return Analysis(inst, base, std, s.source, src, x, y, sv, cr, sol, ref.status, ref_obj, src_err, xov_err, std_nnz(x, 1e-6), std_nnz(cr.x) if cr.x else 0, M.shape[0], src.flops, cr.flops, fs, recall)


def vertex_path(a):
    """Ecken (Strukturvariablen) der Crash-Basis und nach jedem Pivot, nur für Instanzen mit zwei Variablen: Liste von (x1, x2, primal zulässig)."""
    if a.inst.n != 2 or a.xov is None:
        return []
    M, b = a.std[0], a.std[1]
    out = []
    for basis in [a.xov.crash_basis] + [mv.basis for mv in a.xov.path]:
        v = X.vertex_of(M, b, basis)
        if v is not None:
            out.append((float(v[0]), float(v[1]), bool(v.min() >= -1e-9)))
    return out


def family(s, kind=None):
    """Instanzen für Sweeps: fünf feste Seeds bei Zufalls-, Misch- und Plateau-Instanzen, sonst die eine Instanz."""
    kind = kind or s.kind
    if kind in RANDOM_KINDS:
        return [instance_of(s, sd, kind) for sd in C.SWEEP_SEEDS]
    return [instance_of(s, None, kind)]


def run_one(inst, source, eps):
    """Quelle und Crossover auf einer Instanz: (Crossover-Ergebnis, Quellergebnis, Standardform) oder None, wenn die Quelle nichts liefert."""
    std = IP.standard_form(inst)
    if std[0] is None:
        return None
    x, y, sv, src = source_point(inst, std, source, eps)
    if x is None:
        return None
    return X.crossover(std[0], std[1], std[2], x, y, s=sv), src, std


@lru_cache(maxsize=32)
def eps_sweep(s):
    """Genauigkeit der Quelle 10^-k (beide Quellen): Aufräum-Pivots (Median, größter Wert), Anteil der Läufe ohne Pivot, Zustände der Crash-Basis, Zielwertfehler der Quelle gegen den des Crossover."""
    fam = family(s)
    rows = []
    for source in C.SOURCES:
        for k in C.EPS_SWEEP_EXPS:
            piv, err_s, err_x, ok = [], [], [], 0
            cases = {"optimal": 0, "primal": 0, "dual": 0, "beides unzulässig": 0}
            for inst in fam:
                out = run_one(inst, source, 10.0 ** -k)
                if out is None:
                    continue
                cr, src, std = out
                ref = A.solve(inst)
                ok += cr.status == "optimal"
                piv.append(cr.pivots)
                cases[cr.case] = cases.get(cr.case, 0) + 1
                if ref.status == "optimal":
                    err_s.append(abs(-float(std[2] @ (np.array(src.x_full) if source == "pdlp" else np.array(src.z_full))) - ref.obj) / (1.0 + abs(ref.obj)))
                    if cr.status == "optimal":
                        err_x.append(abs(-cr.obj - ref.obj) / (1.0 + abs(ref.obj)))
            rows.append({"source": source, "k": k, "pivots": _med(piv), "max_pivots": max(piv) if piv else float("nan"), "zero": sum(1 for p in piv if p == 0), "runs": len(piv), "optimal": ok, "cases": cases,
                         "err_src": _med(err_s), "err_xov": _med(err_x)})
    return rows


@lru_cache(maxsize=32)
def size_sweep(s):
    """Gesamtkosten im Operationsmodell über die Größe m = n (Zufall, Dichte aus dem Regler, ε aus dem Regler): Simplex allein, Innere Punkte + Crossover, PDLP + Crossover; Kreuzungspunkte."""
    rows = []
    for n in C.SIZE_SIZES:
        fs, fi, fp, pi, pp = [], [], [], [], []
        for sd in C.SIZE_SEEDS:
            inst = S.generate("random", n, n, s.density, sd)
            sol = A.solve(inst)
            fs.append(simplex_flops(inst, sol))
            for source, fl, pv in (("ipm", fi, pi), ("pdlp", fp, pp)):
                out = run_one(inst, source, s.eps)
                if out is None:
                    continue
                cr, src, _std = out
                fl.append(src.flops + cr.flops)
                pv.append(cr.pivots)
        rows.append({"n": n, "flops_simplex": _med(fs), "flops_ipm": _med(fi), "flops_pdlp": _med(fp), "pivots_ipm": _med(pi), "pivots_pdlp": _med(pp)})
    return {"rows": rows, "cross_ipm": _crossing(rows, "flops_ipm"), "cross_pdlp": _crossing(rows, "flops_pdlp")}


def _crossing(rows, key):
    """Kleinstes n, ab dem die Kombination dauerhaft weniger Operationen braucht als der Simplex allein (None: nie)."""
    for r in rows:
        if all(q[key] < q["flops_simplex"] for q in rows if q["n"] >= r["n"]):
            return r["n"]
    return None


@lru_cache(maxsize=32)
def cube_sweep(s):
    """Klee-Minty-Würfel n = 2..14: Aufräum-Pivots aus beiden Quellen gegen die 2^n − 1 Pivots des Simplex, Gesamtkosten; Kreuzungspunkte."""
    rows = []
    for n in C.CUBE_SIZES:
        inst = S.klee_minty_instance(n)
        sol = A.solve(inst)
        row = {"n": n, "pivots_simplex": sol.pivots, "flops_simplex": simplex_flops(inst, sol)}
        for source in C.SOURCES:
            out = run_one(inst, source, s.eps)
            if out is None:
                row.update({f"pivots_{source}": float("nan"), f"flops_{source}": float("nan")})
                continue
            cr, src, _std = out
            row.update({f"pivots_{source}": cr.pivots, f"flops_{source}": src.flops + cr.flops, f"status_{source}": cr.status})
        rows.append(row)
    return {"rows": rows, "cross_ipm": _crossing(rows, "flops_ipm"), "cross_pdlp": _crossing(rows, "flops_pdlp")}


@lru_cache(maxsize=32)
def warm_sweep(s):
    """Neuoptimierung nach einer Änderung der rechten Seite b_i (1 + p sin(i + 1)), Zufall m = n = 20: Innere Punkte kalt gegen warm (alter Punkt um 10^-3 bzw. 1 ins Innere verschoben), PDLP kalt gegen warm,
    dualer Simplex ab der alten Basis gegen Simplex von Null (Median über fünf Instanzen)."""
    rows = []
    n = C.WARM_SIZE
    for p in C.CHANGES:
        acc = {k: [] for k in ("ipm_cold", "ipm_warm_small", "ipm_warm_big", "pdlp_cold", "pdlp_warm", "dual_warm", "simplex_cold")}
        for sd in C.WARM_SEEDS:
            inst = S.generate("random", n, n, s.density, sd)
            std = IP.standard_form(inst)
            M, b, c, nn, _ = std
            ip = IP.ipm(inst, "mehrotra", eps=1e-8, std=std)
            pd = PD.pdlp(inst, eps=1e-6, std=std)
            cr = X.crossover(M, b, c, np.array(ip.z_full), np.array(ip.y), s=np.array(ip.s_full))
            b2 = b * (1.0 + p * np.sin(np.arange(len(b)) + 1.0))
            std2 = (M, b2, c, nn, "")
            acc["ipm_cold"].append(IP.ipm(inst, "mehrotra", eps=1e-8, std=std2).iterations)
            z0, y0, s0 = np.array(ip.z_full), np.array(ip.y), np.array(ip.s_full)
            for key, dl in (("ipm_warm_small", C.WARM_SHIFTS[0]), ("ipm_warm_big", C.WARM_SHIFTS[1])):
                w = IP.ipm(inst, "mehrotra", eps=1e-8, std=std2, start=(z0 + dl, y0, s0 + dl))
                acc[key].append(w.iterations if w.status == "optimal" else float("nan"))
            acc["pdlp_cold"].append(PD.pdlp(inst, eps=1e-6, std=std2).iterations)
            acc["pdlp_warm"].append(PD.pdlp(inst, eps=1e-6, std=std2, start=(np.array(pd.x_full), np.array(pd.y))).iterations)
            acc["dual_warm"].append(X.optimize_from_basis(M, b2, c, list(cr.basis)).pivots)
            inst2 = S.Instance(inst.A, tuple(float(v) for v in b2), inst.c, inst.senses, inst.names, inst.row_names, "custom")
            acc["simplex_cold"].append(A.solve(inst2).pivots)
        rows.append({"p": p, **{k: _med([v for v in vals if v == v]) for k, vals in acc.items()}})
    return rows


def changed_rhs(b, p):
    """Änderung der rechten Seite b_i (1 + p sin(i + 1)) für den Warmstart-Vergleich."""
    return np.asarray(b, dtype=float) * (1.0 + p * np.sin(np.arange(len(b)) + 1.0))


@lru_cache(maxsize=32)
def warm_instance(s):
    """Neuoptimierung auf der gewählten Instanz nach einer Änderung der rechten Seite um s.change: dualer Simplex ab der Crossover-Basis gegen Simplex von Null (Pivots), Innere Punkte und PDLP kalt gegen warm
    (Iterationen); None, wenn kein Crossover-Ergebnis vorliegt oder die Standardform Zeilen entfernt hat."""
    a = analyse(s)
    if a.xov is None or a.xov.status != "optimal" or a.std[0].shape[0] != a.inst.m:
        return None
    M, b, c, n, _ = a.std
    b2 = changed_rhs(b, s.change)
    std2 = (M, b2, c, n, "")
    inst2 = S.Instance(a.inst.A, tuple(float(v) for v in b2), a.inst.c, a.inst.senses, a.inst.names, a.inst.row_names, "custom")
    ref2 = A.solve(inst2)
    warm = X.optimize_from_basis(M, b2, c, list(a.xov.basis))
    ip0 = IP.ipm(a.inst, "mehrotra", eps=1e-8, std=a.std)
    pd0 = PD.pdlp(a.inst, eps=1e-6, max_iter=SOURCE_CAP, std=a.std)
    out = {"p": s.change, "simplex_status": ref2.status, "simplex_cold": ref2.pivots, "dual_status": warm.status, "dual_warm": warm.pivots, "flops_cold": simplex_flops(inst2, ref2) if ref2.status == "optimal" else 0,
           "flops_warm": warm.flops - X.setup_flops(*M.shape)}
    if ip0.z_full:
        z0, y0, s0 = np.array(ip0.z_full), np.array(ip0.y), np.array(ip0.s_full)
        out["ipm_cold"] = IP.ipm(a.inst, "mehrotra", eps=1e-8, std=std2).iterations
        w = IP.ipm(a.inst, "mehrotra", eps=1e-8, std=std2, start=(z0 + C.WARM_SHIFTS[1], y0, s0 + C.WARM_SHIFTS[1]))
        out["ipm_warm"] = w.iterations if w.status == "optimal" else float("nan")
    if pd0.x_full:
        out["pdlp_cold"] = PD.pdlp(a.inst, eps=1e-6, max_iter=SOURCE_CAP, std=std2).iterations
        out["pdlp_warm"] = PD.pdlp(a.inst, eps=1e-6, max_iter=SOURCE_CAP, std=std2, start=(np.array(pd0.x_full), np.array(pd0.y))).iterations
    return out


@lru_cache(maxsize=32)
def degeneracy(s):
    """Zustände der Crash-Basis und Pivots über die Instanzarten (gewählte Quelle und Genauigkeit)."""
    rows = []
    for kind in C.DEGENERACY_KINDS:
        cases = {"optimal": 0, "primal": 0, "dual": 0, "beides unzulässig": 0}
        piv, nz_src, nz_xov, runs = [], [], [], 0
        ms = []
        for inst in family(s, kind):
            out = run_one(inst, s.source, s.eps)
            if out is None:
                continue
            cr, src, std = out
            runs += 1
            cases[cr.case] = cases.get(cr.case, 0) + 1
            piv.append(cr.pivots)
            x = np.array(src.x_full if s.source == "pdlp" else src.z_full)
            nz_src.append(std_nnz(x, 1e-6))
            nz_xov.append(std_nnz(cr.x) if cr.x else 0)
            ms.append(std[0].shape[0])
        rows.append({"kind": kind, "runs": runs, "pivots": _med(piv), "max_pivots": max(piv) if piv else float("nan"), "cases": cases, "nnz_src": _med(nz_src), "nnz_xov": _med(nz_xov), "m": _med(ms)})
    return rows
