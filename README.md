# Crossover – aus dem Inneren zurück zur Ecke – Streamlit-Demo

Elftes und letztes Stück der **Lineare-Programmierung-Reihe** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Konvergenzpunkt der Reihe: [Innere Punkte](https://github.com/sebastian-hanisch/innere-punkte-demo) (Stück 8) und [PDLP](https://github.com/sebastian-hanisch/pdlp-demo) (Stück 10) enden im **Inneren**: eine Näherung, keine Ecke, keine Basis, keine exakten Duale, kein Warmstart. Wer Schattenpreise, Ranging oder eine Neuoptimierung braucht, braucht die Ecke zurück. **Crossover** (Megiddo 1991, Bixby & Saltzman 1994) erkennt aus der Näherung eine **Basis** (der Indikator x gegen s sagt, welche Variablen positiv bleiben) und räumt mit wenigen **Simplex-Pivots** zur exakten Ecke auf. Die Demo baut das Verfahren, misst, wie gut die Basis erkannt wird, was das Aufräumen kostet, wann Näherung + Crossover den Simplex allein schlägt und wie sich der Warmstart verhält. Fünf Fragen: **(1) Der Weg** von der Näherung zur Ecke, **(2) die Basis** – wie gut wird sie erkannt, **(3) das Ergebnis** – wie genau, **(4) wann welches Verfahren** – Simplex allein, Innere Punkte oder PDLP mit Crossover, Warmstart, **(5) Grenzen** – Plateau, Entartung, Zustände der Crash-Basis.

**Einordnung in die Reihe:** die Reihe hat **elf** Stücke, dies ist das letzte (Details in `lp-planung/PLAN.md` des Portfolio-Ordners):

```
Tableau-Simplex (Wurzel)                                                                  [gebaut: tableau-simplex-demo]
 ├─ Pivotregeln & Entartung ─ Simplex im schlimmsten und im typischen Fall (Klee-Minty)   [gebaut: pivotregeln-demo, klee-minty-demo]
 ├─ Revised Simplex ─ Präsolve, Skalierung & Numerik                                     [gebaut: revised-simplex-demo, praesolve-demo]
 ├─ Dualität & Sensitivität ─ Dualer Simplex & Neuoptimierung                            [gebaut: lp-dualitaet-demo, dualer-simplex-demo]
 ├─ Ellipsoid-Methode (Kontrast: polynomial in der Theorie)                              [gebaut: ellipsoid-demo]
 └─ Innere Punkte ─ PDLP (Verfahren erster Ordnung) ─ Crossover & Simplex gegen Innere Punkte gegen PDLP
      [gebaut: innere-punkte-demo]   →   [gebaut: pdlp-demo]   →   [DIESES STÜCK]
```

Ergebnis in Kürze: **Crossover liefert aus jeder Näherung die exakte Ecke – der Zielwertfehler fällt von 10⁻³ bis 10⁻¹¹ auf 10⁻¹⁶ –, meist ohne einen einzigen Pivot, aber auf Zufallsinstanzen lohnt es sich gegenüber dem Simplex allein nie.** Ab ε = 10⁻³ der Quelle (Innere Punkte oder PDLP) braucht das Aufräumen auf Zufalls- und Mischinstanzen höchstens 1 Pivot, ab 10⁻⁴ ist die erkannte Basis auf allen schon optimal (0 Pivots); bei ε = 10⁻² bleiben 0 bis 5 Pivots im Median (30 × 30 mit PDLP im Median 5, größter Wert 58). Auf einer optimalen **Fläche** (Plateau) kann der Indikator keine Ecke bevorzugen: dort braucht das Aufräumen bei jeder Genauigkeit duale Pivots (3 bis 11 im Median). Im Operationsmodell ist der Simplex allein auf Zufallsinstanzen 10- bis 200-mal billiger als Näherung + Crossover; auf dem **Klee-Minty-Würfel** gewinnt das Innere ab n = 7 (n = 14: 1 Pivot statt 16.383, 50-mal weniger Operationen). Beim **Warmstart** nach einer Änderung der rechten Seite braucht der duale Simplex ab der Crossover-Basis 0 bis 4 Pivots (von Null: 9 bis 11); die Inneren Punkte profitieren nur bei kleinen Änderungen und passender Verschiebung (8 → 3 Iterationen bei 1 bis 5 %, bei 25 % mit kleiner Verschiebung schlechter: 21), PDLP spart wenig.

| Frage | Ergebnis (Auslastungsplanung als Standard-LP max c·x; Zufallsinstanzen mit Dichte 50 %, 5 feste Instanzen Seeds 100000–100004, Größe je Zeile genannt; Referenz: HiGHS bzw. eigener Tableau-Simplex; das Verfahren ist deterministisch, kein Zufall) |
|---|---|
| **Stimmt das Verfahren?** | ✅ Crossover liefert auf 320 Instanzen (Zufall, Mischung mit ≥ und =, Zentrum, Lehrbuch, entartet, Plateau, Klee-Minty-Würfel n ≤ 10) aus Innere Punkte (ε = 10⁻², 10⁻⁶) und PDLP (10⁻², 10⁻⁴) den Optimalwert von HiGHS, eine **Ecke** (außerhalb der Basis exakt null, höchstens m Nichtnullen, Ax = b bis 10⁻⁸), ein dual zulässiges y und **c·x = b·y bis 10⁻⁷**. Es endet auch bei **Unsinn als Näherung** (x = 0, y = 0, Zufallsvektoren) im richtigen Optimum; eine exakte oder um 10⁻⁹ gestörte Quelle braucht 0 Pivots und liefert dieselbe Basis |
| **Genauigkeitsleiter** | Zufall 12 × 12, Aufräum-Pivots (Median / größter Wert / Läufe ohne Pivot) bei ε = 10⁻² der Quelle: Innere Punkte **1 / 1 / 2 von 5**, PDLP **0 / 1 / 3 von 5**; ab 10⁻³: **0 / 0 / 5 von 5** bei beiden. Mischung 12 × 12: dasselbe. **Zufall 30 × 30:** Innere Punkte bei 10⁻² 2 / 5 / 1 von 5, PDLP **5 / 58 / 0 von 5** (4 von 5 Basen weder primal noch dual zulässig: Kostenverschiebung), PDLP bei 10⁻³ 0 / 1 / 3 von 5, ab 10⁻⁴ 0 / 0 / 5 von 5. 100 frische Mischinstanzen (8 × 8, PDLP): **alle 100 gelöst**, ohne Pivot 73 bei 10⁻², **97 bei 10⁻⁴** |
| **Genauigkeit des Ergebnisses** | Relativer Zielwertfehler der Quelle bei ε = 10⁻² / 10⁻³ / 10⁻⁴ / 10⁻⁶ / 10⁻⁸ (PDLP, Zufall 12 × 12): 3.0·10⁻³ / 2.5·10⁻⁴ / 5.1·10⁻⁵ / 4.2·10⁻⁷ / 5.8·10⁻⁹; **nach dem Crossover überall ≈ 10⁻¹⁶** (Rundungsgrenze), Duale exakt. Nichtnullen: Zentrum 4 → 4; Plateau 10 × 12 (PDLP, 10⁻⁴): **21 → 10** (Ecke: höchstens m) |
| **Plateau (optimale Fläche)** | Plateau 10 × 12 (Zielfunktion parallel zur Gesamtkapazität), Aufräum-Pivots im Median: Innere Punkte **3** (größter Wert 20) bei jeder Genauigkeit, PDLP **4 / 9 / 7 / 7 / 11** bei ε = 10⁻² / 10⁻³ / 10⁻⁴ / 10⁻⁶ / 10⁻⁸; nie 0 Pivots, die Crash-Basis ist in allen 5 Läufen nur **dual** zulässig; Ergebnis exakt (Fehler unter 10⁻¹²) |
| **Klee-Minty-Würfel** | Aufräum-Pivots aus der Quelle mit ε = 10⁻² bei n = 8 / 10 / 12 / 14: Innere Punkte **0 / 10 / 42 / 341**, PDLP **0 / 10 / 42 / 170**, Simplex von Null **255 / 1.023 / 4.095 / 16.383**. Bei ε = 10⁻⁴ IP 0 / 0 / 1 / 10, PDLP 0 / 5 / 21 / 170; bei ε = 10⁻⁶ IP 0 / 0 / 0 / 1, PDLP 0 / 0 / 5 / 21. **Kreuzungspunkt** (Näherung + Crossover braucht dauerhaft weniger Operationen als der Simplex): Innere Punkte **n = 7**, PDLP n = 7 bis 9. n = 14, IP bei 10⁻⁶: 257.440 + 25.043 gegen 14.253.210 Operationen (**50-mal weniger**) |
| **Gesamtkosten auf Zufall** | Modell, ε = 10⁻⁴, Median über 3 Instanzen, Näherung + Crossover im Verhältnis zum Simplex von Null: Dichte 50 %, n = 8 / 16 / 32 / 64 / 100: Innere Punkte **18.7 / 22.1 / 26.5 / 27.0 / 12.3**, PDLP **118.7 / 40.8 / 39.7 / 23.5 / 9.9**; Dichte 5 %: Innere Punkte 24.9 / 52 / 47.5 / 89.7 / 206, PDLP 50.4 / 55.5 / 30.9 / 38.3 / 59.5. **Kein Kreuzungspunkt**; Aufräum-Pivots höchstens 3. Zufall 40 × 40 (IP, 10⁻⁴): 2.170.931 + 557.866 gegen 179.334 (15.2-fach) |
| **Warmstart** | Zufall 20 × 20, Median über 5 Instanzen, rechte Seite um 1 / 5 / 10 / 25 % geändert (b_i (1 + p sin(i + 1))): **dualer Simplex ab der Crossover-Basis 0 / 0 / 1 / 4 Pivots**, Simplex von Null 9 / 9 / 9 / 11; Innere Punkte kalt 8 Iterationen, warm (Punkt um 10⁻³ ins Innere verschoben) **3 / 3 / 8 / 21**, warm (um 1 verschoben) 4 / 5 / 7 / 8; PDLP kalt 600 / 504 / 580 / 472, warm 500 / 392 / 520 / 496 |

## Vorab-Hypothesen

| Hypothese (vor der Messung) | Ergebnis |
|---|---|
| Aus Innere Punkte bei ε = 10⁻⁶ folgt meist 0 Pivots | **Bestätigt, und früher als gedacht:** auf Zufall und Mischung ab ε = 10⁻³ (12 × 12: alle fünf Instanzen ohne Pivot; 30 × 30: höchstens 1 Pivot, bei den Inneren Punkten alle fünf ohne Pivot), ab 10⁻⁴ auf allen; bei 10⁻² bleiben 0 bis 5 Pivots im Median |
| PDLP braucht mehr Pivots als Innere Punkte | **Nur teilweise:** auf Zufall und Mischung nicht (bei 12 × 12 ist PDLP bei 10⁻² sogar besser, ab 10⁻³ beide 0); bei 30 × 30 und 10⁻² schon (5 gegen 2, größter Wert 58), auf dem Plateau (4 bis 11 gegen 3) und auf dem Würfel bei 10⁻⁴ (n = 14: 170 gegen 10) deutlich |
| Aus 10⁻⁴ von PDLP wird exakt | **Bestätigt:** Zielwertfehler ≈ 10⁻¹⁶ bei jeder Genauigkeit der Quelle, Duale exakt |
| Auf dem Klee-Minty-Würfel spart Crossover die 2ⁿ − 1 Pivots | **Bestätigt, aber abhängig von ε:** bei 10⁻⁶ 0 bis 1 Pivot bis n = 14 (Innere Punkte), bei 10⁻² wächst die Zahl der Aufräum-Pivots mit n (0 / 10 / 42 / 341 bei n = 8 / 10 / 12 / 14) und liegt trotzdem weit unter 2ⁿ − 1. Kreuzungspunkt n = 7 |
| Auf Zufallsinstanzen lohnt sich Crossover nicht | **Bestätigt:** der Simplex allein ist im Modell 10- bis 200-mal billiger, weil er nur wenige Pivots braucht und die Näherung selbst schon teurer ist; Crossover lohnt, wenn man eine Basis braucht und die Näherung schon da ist |
| Der Warmstart der Inneren Punkte ist kaum besser als kalt | **Teils widerlegt:** bei 1 bis 5 % Änderung 8 → 3 Iterationen (bei 10⁻³ Verschiebung), bei 25 % mit kleiner Verschiebung aber schlechter (21), mit großer gleich (8); PDLP spart bei 1 bis 10 % 10 bis 22 % der Iterationen (500 gegen 600), bei 25 % nichts. Der duale Simplex bleibt weit vorn (0 bis 4 Pivots) |
| Auf entarteten Instanzen scheitert die Basis-Erkennung häufig | **Widerlegt für die Fixture, bestätigt fürs Plateau:** entartete Ecke 1 Pivot; auf dem Plateau nie 0 Pivots (aber immer korrekt); auf Zufall und Mischung 0 bis 1 |

## Was die Demo zeigt

1. **Fünf Schritte** (Schritt-Slider): **Vom Innenpunkt zur Ecke** (2 Variablen: zulässiges Vieleck, Näherung, Crash-Ecke, Ecken nach den Pivots, Slider über die Pivots; sonst Tabelle des Pivotpfads) → **Basis erkennen** (Indikator je Variable in Prioritätsreihenfolge, Crash-Basis gegen optimale Basis, Zustand der Crash-Basis; auf Abruf Pivots über die Genauigkeit beider Quellen) → **Was das Aufräumen bringt** (Zielwertfehler vor und nach, Nichtnullen gegen m, Pivots gegen Simplex von Null, exakte Duale; auf Abruf Fehler über die Genauigkeit) → **Wann welches Verfahren** (Tabelle für die gewählte Instanz, Warmstart auf ihr; auf Abruf Gesamtkosten über Größe, Klee-Minty-Würfel, Warmstart über alle Änderungen) → **Grenzen** (auf Abruf: Zustände der Crash-Basis und Pivots über die Instanzarten).
2. **Instanzen:** Lehrbuch, Zentrum, entartete Ecke, Zufall, Mischung, **Plateau**, Klee-Minty-Würfel, Unzulässig, Unbeschränkt (die letzten beiden: keine Ecke, die Demo zeigt den Status der Quelle).
3. **Regler:** Größe (m, n bis 40), Dichte, **Quelle** (Innere Punkte / PDLP), **Genauigkeit der Quelle** (10⁻² bis 10⁻⁸), Skalierung der Spalten (bis 10⁸), Änderung der rechten Seite (1 bis 25 %, nur in Schritt 4).

Presets (12): Entartete Ecke: ein Pivot, Zentrum: aus 10^-2 wird exakt, Grobe Quelle: ein paar Pivots, Feine Quelle: null Pivots, Plateau: aus dem Inneren zur Ecke, Klee-Minty-Würfel: wenige gegen Tausende Pivots, Zufall: der Simplex bleibt vorn, Würfel: hier lohnt sich das Innere, Warmstart nach einer Änderung, Schlechte Skalierung, Zustände der Crash-Basis, Unzulässig: kein Crossover.

## Messwerte der Presets

| Preset | Einstellungen | Ergebnis |
|---|---|---|
| **Entartete Ecke: ein Pivot** | Fixture (2 × 4), PDLP 10⁻² | 36 Iterationen der Quelle, Fehler 7.4·10⁻⁴ → 0; 1 primaler Pivot (Simplex von Null 2) |
| **Zentrum: aus 10^-2 wird exakt** | Fixture, PDLP 10⁻² | 76 Iterationen, Fehler 5.3·10⁻³ → 1.6·10⁻¹⁶; 0 Pivots, 4 von 4 Basisspalten erkannt (Simplex 4) |
| **Grobe Quelle: ein paar Pivots** | Zufall 30 × 30, PDLP 10⁻² | 5 primale Pivots (Simplex 6), 93 % der Träger der Ecke in der Crash-Basis; Median über 5 Instanzen 5 (größter Wert 58), bei 10⁻⁴ 0 |
| **Feine Quelle: null Pivots** | Zufall 30 × 30, Innere Punkte 10⁻⁶ | 9 Iterationen, 0 Pivots, alle 30 Träger erkannt; Modell: Quelle 1.220.400, Crossover 235.800, Simplex von Null 22.692 (6 Pivots) |
| **Plateau: aus dem Inneren zur Ecke** | Plateau 10 × 12, PDLP 10⁻⁴ | 116 Iterationen, 21 → 10 Nichtnullen, 21 duale Pivots (Simplex 3) |
| **Klee-Minty-Würfel: wenige gegen Tausende Pivots** | n = 12, Innere Punkte 10⁻² | 42 primale Pivots gegen 4.095; Modell 127.296 + 42.564 gegen 2.661.750 |
| **Zufall: der Simplex bleibt vorn** | Zufall 40 × 40, Innere Punkte 10⁻⁴ | 0 Pivots, aber 15.2-fach die Operationen des Simplex (27 Pivots) |
| **Würfel: hier lohnt sich das Innere** | n = 14, Innere Punkte 10⁻⁶ | 1 Pivot statt 16.383; 50-mal weniger Operationen |
| **Warmstart nach einer Änderung** | Zufall 20 × 20, 5 % | dualer Simplex 3 Pivots (von Null 12); Innere Punkte 8 kalt / 8 warm; PDLP 1.000 kalt / 580 warm |
| **Schlechte Skalierung** | Zufall 10 × 10, Spalten über 10⁶, PDLP 10⁻² | 352 Iterationen, 0 Pivots: der Indikator ist spaltennormiert |
| **Zustände der Crash-Basis** | Instanzarten, PDLP 10⁻² | Zufall und Mischung 0 (größter Wert 1), entartet 1, Würfel 42, Plateau 32 (größter Wert 38) |
| **Unzulässig: kein Crossover** | Fixture | Farkas-Strahl der Quelle nach 192 Iterationen: keine Ecke |

## Modell und Verfahren

- **Instanzen und Quellen** (`xov_scenario.py`, `xov_ipm.py`, `xov_pdlp.py`, `xov_algorithm.py`): wortgleiche Kopien aus `pdlp-demo`; erweitert um die volle Standardform-Lösung und den Dualschlupf der Inneren Punkte sowie einen **Warmstart** (alter Punkt) für PDLP. Beide Quellen laufen auf derselben Standardform mit Rangprüfung (`standard_form` der Inneren Punkte).
- **Crossover** (`xov_basis.py`): **Indikator** ξ_j = x̂_j / (x̂_j + ŝ_j) mit x̂_j = x_j‖M_j‖ und ŝ_j = s_j/‖M_j‖ (invariant unter Spaltenskalierung), s = c − Mᵀy; Spalten nach fallendem Indikator, greedy m unabhängige (Gram-Schmidt, relative Toleranz 10⁻⁸): die **Crash-Basis**. Zustand der Basis: x_B = B⁻¹b, y = B⁻ᵀc_B, r = c − Mᵀy. **Aufräumen** mit einem dichten Tableau, frisch aus B gerechnet: beides zulässig 0 Pivots; nur primal zulässig: primaler Simplex (Dantzig, Bland nach 25 Nullschritten); nur dual zulässig: dualer Simplex; beides unzulässig: **Kostenverschiebung** (c_j += max(0, −r_j) für Nichtbasisvariablen), dualer Simplex bis primal zulässig, ursprüngliche Kosten, primaler Simplex. Der **Warmstart** ist derselbe Code ab der alten Endbasis.
- **Operationsmodell:** Crash-Basis 2m²N, Zerlegung der Basis 2m³/3, Tableau 2m²(N+1), je Pivot 2(m+1)(N+1); Quellen und Simplex von Null wie in Stück 8 und 10 (dicht bzw. nur Nichtnullen).
- **Auswertung** (`xov_evaluation.py`): Analyse, Ecken-Pfad, Genauigkeitsleiter, Gesamtkosten, Würfel, Warmstart auf der Instanz und über alle Änderungen, Zustände der Crash-Basis.

## Was nicht funktioniert hat / Grenzen

- **Erste Fassung meldete den Würfel ab n = 13 als unbeschränkt.** Die Toleranz für die Pivotelemente war an die Größe von b gekoppelt (10⁻⁹ · (1 + max |b|)); beim Klee-Minty-Würfel reicht b bis 5ⁿ (6·10⁹), also wurden Pivotelemente unter 6 als null verworfen und der primale Simplex fand "keine Austrittszeile". Aufgefallen ist es erst im Würfel-Sweep (n = 14 schien mit 2 Pivots gelöst, war aber falsch); jetzt sind Pivottoleranz (relativ zur größten Zahl der Spalte) und Zulässigkeitstoleranz (relativ zu b) getrennt, und ein Regressionstest prüft n = 12 bis 14.
- **Nur die Grundform von Crossover.** Basis-Erkennung per Indikator plus Simplex-Aufräumen; keine Push-Phasen nach Bixby & Saltzman (die primale und duale Variablen nacheinander an ihre Grenzen schieben), keine Kreuzungsschranken, keine dünne LU mit Updates: ein dichtes Tableau, dessen Aufbau (2m²N-artig) unabhängig von der Pivotzahl kostet.
- **Der Indikator ist mehrdeutig, wenn die Ecke es ist.** Auf dem Plateau (ganze optimale Fläche) und an entarteten Ecken kann er keine bevorzugen; das Ergebnis bleibt richtig, aber die Pivotzahl steigt. Die Erkennungsquote ("Träger der exakten Ecke in der Crash-Basis") ist nur bei nicht entarteter exakter Ecke definiert.
- **Operationsmodell, keine Wandzeit.** Innere Punkte und Simplex sind dicht modelliert, PDLP nur mit den Nichtnullen: der Vergleich mit dem Simplex allein fällt für die Näherungs-Verfahren ungünstiger aus, als es echte Löser (dünn, mit Präsolve) zeigen würden.
- **Synthetische, gutartige Instanzen.** Der Simplex braucht auf den Zufallsinstanzen wenige Pivots; schwere Instanzen (Netlib, MIPLIB) hat die Demo nicht. Deshalb "lohnt sich Crossover nie" nur für diese Familie.
- **Warmstart nur in der einfachsten Form.** Innere Punkte: alter Punkt um einen festen Betrag ins Innere verschoben (10⁻³ oder 1); bessere Verfahren (Zentrierung, homogene Einbettung) sind nicht gebaut. PDLP: alter Punkt (x, y) als Start.
- **Plattformabhängigkeit.** Pivotzahlen bei Entartung, Iterationszahlen der Quellen und Kreuzungspunkte können unter Windows und Linux um einzelne Werte abweichen; die Tests prüfen dort Bänder. Ganzzahl-Logik (2ⁿ − 1 Pivots des Würfels, höchstens m Nichtnullen) ist exakt.
- **Die LP-Themenseite der Website** (Überblick über alle elf Stücke) ist noch nicht gebaut.

## Verifikation

- `tests/test_algorithm.py`: **Crossover gegen HiGHS auf 320 Instanzen** (Optimalwert, Ecke, Zulässigkeit, starke Dualität) aus vier Quellen (Innere Punkte 10⁻² und 10⁻⁶, PDLP 10⁻² und 10⁻⁴); **Fixpunkt** (exakte und um 10⁻⁹ gestörte Quelle: 0 Pivots, dieselbe Basis) und **Unsinns-Quellen** (Null, Zufall, Einsen: richtiges Optimum); Indikator von Hand und **invariant unter Spaltenskalierung**; Crash-Basis (Prioritätsreihenfolge, Abhängigkeiten, Auffüllen); primaler und dualer Simplex ab beliebigen Basen (Tableau nach jedem Pivot gleich B⁻¹[M | b], Zielwert monoton), **jede Basis der entarteten Fixture** endet im Optimum, **Kostenverschiebung** (Original-Kosten wiederhergestellt), Unzulässig und Unbeschränkt im Aufräumen erkannt; Plateau (PDLP innen, Crossover Ecke), Warmstarts gegen Neulösung, Buchführung (Pivotpfad, Operationsmodell, Determinismus), Sonderfälle (m = 1, c = 0, abhängige Zeilen, singuläre Basis), Regressionstest für den Würfel n = 12 bis 14, Kopien treu.
- `tests/test_scenario.py`, `test_evaluation.py`, `test_presets.py` (jede Zahl der Hilfetexte), `test_claims.py` (jede Zahl aus README und App über die echten `ev.*`-Funktionen; Pivot- und Iterationszahlen nur als Bänder), `test_app.py` (Streamlit-AppTest: Voreinstellung, jedes Preset, jeder Schritt für jede Instanz und Quelle, Pivot-Slider, Regler-Randwerte, Permalink-Grenzen, bedingte Regler, Berechnungen auf Abruf, Footer).
- Für die Prüfung genügt **pytest**; `scipy` dient nur als Gegenprobe (`requirements-dev.txt`), die App braucht nur numpy, pandas, plotly und streamlit.

## Lokal starten

```bash
python -m venv venv && venv/Scripts/activate  # Windows; Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -W error::SyntaxWarning`.

## Literatur

- Megiddo, N. (1991). *On finding primal- and dual-optimal bases.* ORSA Journal on Computing 3(1), 63–65.
- Bixby, R. E., & Saltzman, M. J. (1994). *Recovering an optimal LP basis from an interior point solution.* Operations Research Letters 15(4), 169–178.
- Applegate, D., Díaz, M., Hinder, O., Lu, H., Lubin, M., O'Donoghue, B., & Schudy, W. (2021). *Practical large-scale linear programming using primal-dual hybrid gradient.* Advances in Neural Information Processing Systems 34 (PDLP als Quelle, Stück 10).

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
