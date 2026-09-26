"""Konstanten der Demo Crossover: Regler-Bereiche, Stufen, feste Instanzen für die Auswertung, Presets."""
M_MIN, M_MAX, DEFAULT_M = 2, 40, 12
N_MIN, N_MAX, DEFAULT_N = 2, 40, 12
DENSITIES = (0.05, 0.1, 0.2, 0.5, 1.0)
DEFAULT_DENSITY_I = 3
SEED_MAX = 999999
DEFAULT_SEED = 35
SOURCES = ("ipm", "pdlp")
SOURCE_LABELS = {"ipm": "Innere Punkte (Mehrotra)", "pdlp": "PDLP"}
SOURCE_SHORT = {"ipm": "Innere Punkte", "pdlp": "PDLP"}
EPS_EXPS = (2, 3, 4, 6, 8)                            # Genauigkeit der Quelle 10^-k
DEFAULT_EPS_I = 0
SCALE_EXPS = (0, 2, 4, 6, 8)                          # Spaltenskalierung 10^k
DEFAULT_SCALE_I = 0
CHANGES = (0.01, 0.05, 0.10, 0.25)                    # Änderung der rechten Seite (Warmstart): b_i (1 + p sin(i + 1))
DEFAULT_CHANGE_I = 1
STEPS = {1: "1 · Vom Innenpunkt zur Ecke", 2: "2 · Basis erkennen", 3: "3 · Was das Aufräumen bringt", 4: "4 · Wann welches Verfahren", 5: "5 · Grenzen"}
SWEEP_SEEDS = tuple(range(100000, 100005))
SIZE_SIZES = (8, 16, 32, 64, 100)
SIZE_SEEDS = SWEEP_SEEDS[:3]
CUBE_SIZES = tuple(range(2, 15))
EPS_SWEEP_EXPS = (2, 3, 4, 6, 8)
WARM_SEEDS = SWEEP_SEEDS
WARM_SIZE = 20
WARM_SHIFTS = (1e-3, 1.0)                             # Verschiebung des alten Punkts ins Innere für den Warmstart der Inneren Punkte
DEGENERACY_KINDS = ("random", "mixed", "degenerate", "plateau", "klee_minty")
_BASE = {"kind": "random", "m": 12, "n": 12, "seed": 35, "density": 3, "source": "pdlp", "eps": 0, "scale": 0, "change": 1, "step": 1}
PRESETS = {
    "Entartete Ecke: ein Pivot": {**_BASE, "kind": "degenerate", "step": 1, "pivot_k": 1},
    "Zentrum: aus 10^-2 wird exakt": {**_BASE, "kind": "centre", "step": 3},
    "Grobe Quelle: ein paar Pivots": {**_BASE, "m": 30, "n": 30, "step": 2},
    "Feine Quelle: null Pivots": {**_BASE, "m": 30, "n": 30, "source": "ipm", "eps": 3, "step": 2},
    "Plateau: aus dem Inneren zur Ecke": {**_BASE, "kind": "plateau", "m": 10, "n": 12, "eps": 2, "step": 3},
    "Klee-Minty-Würfel: wenige gegen Tausende Pivots": {**_BASE, "kind": "klee_minty", "n": 12, "source": "ipm", "step": 3},
    "Zufall: der Simplex bleibt vorn": {**_BASE, "m": 40, "n": 40, "source": "ipm", "eps": 2, "step": 4},
    "Würfel: hier lohnt sich das Innere": {**_BASE, "kind": "klee_minty", "n": 14, "source": "ipm", "eps": 3, "step": 4},
    "Warmstart nach einer Änderung": {**_BASE, "m": 20, "n": 20, "step": 4, "change": 1},
    "Schlechte Skalierung": {**_BASE, "m": 10, "n": 10, "scale": 3, "eps": 2, "step": 2},
    "Zustände der Crash-Basis": {**_BASE, "kind": "mixed", "step": 5},
    "Unzulässig: kein Crossover": {**_BASE, "kind": "infeasible", "step": 1},
}
PRESET_HELP = {
    "Entartete Ecke: ein Pivot": "Entartete Ecke (2 Dienste, 4 Ressourcen), PDLP bei ε = 10^-2 (36 Iterationen, Zielwertfehler 7.4e-4): die erkannte Basis ist nur primal zulässig und braucht 1 primalen Pivot; der Simplex von Null 2 Pivots. Ergebnis exakt, Optimum 36.",
    "Zentrum: aus 10^-2 wird exakt": "Zentrum (5 Dienste, 4 Ressourcen), PDLP bei ε = 10^-2 (76 Iterationen, Zielwertfehler 5.3e-3): die Crash-Basis ist schon optimal (0 Pivots), alle 4 Spalten der optimalen Basis wurden erkannt, der Fehler fällt auf 1.6e-16. Der Simplex von Null braucht 4 Pivots.",
    "Grobe Quelle: ein paar Pivots": "Zufall 30 × 30 (Seed 35), PDLP bei ε = 10^-2 (92 Iterationen): die Crash-Basis ist nur primal zulässig und braucht 5 Pivots (der Simplex von Null 6); 93 % der Träger der exakten Ecke waren in ihr. Über fünf Instanzen liegt der Median bei 5 Pivots (größter Wert 58), bei ε = 10^-4 bei 0.",
    "Feine Quelle: null Pivots": "Zufall 30 × 30 (Seed 35), Innere Punkte bei ε = 10^-6 (9 Iterationen): 0 Pivots, alle 30 Träger der Ecke erkannt, Zielwertfehler 1.3e-7 → 0. Im Operationsmodell kostet die Quelle 1.220.400, das Crossover 235.800 Operationen, der Simplex von Null nur 22.692 (6 Pivots).",
    "Plateau: aus dem Inneren zur Ecke": "Plateau 10 × 12 (Seed 35), PDLP bei ε = 10^-4 (116 Iterationen): die Näherung hat 21 Nichtnullen (m = 10), die Crash-Basis ist nur dual zulässig und braucht 21 duale Pivots zur Ecke mit 10 Nichtnullen; Zielwertfehler 1.8e-4 → 1.4e-16. Der Simplex von Null braucht 3 Pivots.",
    "Klee-Minty-Würfel: wenige gegen Tausende Pivots": "Würfel n = 12, Innere Punkte bei ε = 10^-2 (12 Iterationen): 42 primale Pivots im Aufräumen gegen 4.095 des Simplex von Null; im Modell 127.296 (Quelle) + 42.564 (Crossover) gegen 2.661.750 Operationen.",
    "Zufall: der Simplex bleibt vorn": "Zufall 40 × 40 (Seed 35), Innere Punkte bei ε = 10^-4 (7 Iterationen): 0 Pivots, aber im Modell 2.170.931 (Quelle) + 557.866 (Crossover) gegen 179.334 Operationen des Simplex von Null (27 Pivots): das 15.2-Fache. Klick auf 'Gesamtkosten über die Größe berechnen'.",
    "Würfel: hier lohnt sich das Innere": "Würfel n = 14, Innere Punkte bei ε = 10^-6 (16 Iterationen): 1 Pivot statt 16.383; im Modell 257.440 (Quelle) + 25.043 (Crossover) gegen 14.253.210 Operationen: 50-mal weniger. Bei ε = 10^-2 wären es 341 Pivots (Klick auf 'Auf dem Würfel vergleichen').",
    "Warmstart nach einer Änderung": "Zufall 20 × 20 (Seed 35), rechte Seite um 5 % geändert: der duale Simplex ab der Crossover-Basis braucht 3 Pivots, der Simplex von Null 12; die Inneren Punkte 8 Iterationen kalt und 8 warm, PDLP 1.000 kalt und 580 warm.",
    "Schlechte Skalierung": "Zufall 10 × 10 (Seed 35), Spalten über 10^6 gestreut, PDLP bei ε = 10^-2 (352 Iterationen): die Crash-Basis ist schon optimal, 0 Pivots. Der Indikator ist spaltennormiert und übersteht die Skalierung.",
    "Zustände der Crash-Basis": "Klick auf 'Instanzarten vergleichen' (PDLP, ε = 10^-2, 12 × 12): Zufall und Mischung 0 Pivots im Median (größter Wert 1), entartete Ecke 1, Klee-Minty-Würfel 42, Plateau 32 (größter Wert 38): auf dem Plateau ist die Crash-Basis in allen 5 Läufen nur dual zulässig.",
    "Unzulässig: kein Crossover": "PDLP meldet nach 192 Iterationen den nachgerechneten Farkas-Strahl (Stück 10): das LP ist unzulässig, es gibt keine Ecke und damit nichts zu erkennen; die Demo zeigt den Status der Quelle.",
}


def eps_label(i):
    return f"10^-{EPS_EXPS[i]}"


def scale_label(i):
    return "unskaliert" if SCALE_EXPS[i] == 0 else f"über 10^{SCALE_EXPS[i]}"


def density_label(i):
    return f"{DENSITIES[i]:.0%}"


def change_label(i):
    return f"{CHANGES[i]:.0%}"
