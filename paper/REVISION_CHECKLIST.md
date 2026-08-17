# Revision Checklist — TSAS Round 1

Built from a full read of the manuscript and all three reviews. Four parts:
(A) per-review actions, (B) section-by-section edit plan, (C) paper-vs-implementation
inconsistencies found during the deep read (fix regardless of reviews),
(D) figures/tables to produce. Status: ☐ open, ◐ evidence ready (writing pending), ☑ done.

Colour key in `main_revision_1.tex`: `\rone{}`=R1 blue, `\rtwo{}`=R2 green,
`\rthree{}`=R3 purple, `\rall{}`=multiple/editor vermilion, `\rdel{}`=deletion.

## A. Review-driven actions

| # | Point | Action | Evidence | Status |
|---|---|---|---|---|
| A1 | R1.9 / W1 / R3.2 | Rewrite §9 around automatic matching; temper "automatic" claims in abstract §1 §9; state filename metadata explicitly | matcher + MatcherEval suite | ◐ |
| A2 | R1.4 | Port corrected Algorithm 1 + Proposition + proof + bypass note + complexity ¶ (already drafted in this file's §7.3) | FunnelingAblation | ☑ drafted / ☐ colour-mark |
| A3 | R2.W3 / R3.3b | Write degradation subsection: knockout curves fig, by-type, single-door table, cut-vertex targeting | DoorKnockout | ◐ |
| A4 | R1.7 / W7 / R3.5 | Rewrite fidelity subsection around geodesic reference (dual bounds), add overlays fig, drop Euclid-proxy claims | RouteFidelity | ◐ |
| A5 | R3.5 | Rewrite §9 inter-floor weights (vertical travel + door-width scale); update §9.4 "nominal" text | MultiFloor weights, Downstream | ◐ |
| A6 | R1.1 | Density ablation ¶ + corridor-vs-room sampling rationale in §6; downstream-validation subsection (drafted at §10.5) | CorridorDensity, Downstream | ◐ |
| A7 | R1.6 / W2 / R3.1c | Dataset table (all plans + sites), multi-site metric table; reconcile 22-vs-27 rooms + resolution vs conference Table 1 | Metrics + **needs dataset + conference sources** | ☐ |
| A8 | R3.1c | Baseline subsection: skeleton comparison table + framing; decide on published-system comparison | BaselineSkeleton | ◐ |
| A9 | R2.W6 | Ablation subsections: funneling (drafted), pruning keep-terms, pruning cost | FunnelingAblation, PruningAblation | ◐ |
| A10 | R1.8 | Text-accuracy ¶ + failure-consequence analysis | TextAccuracy | ◐ |
| A11 | R1.3 / R1.17 | Label-placement measurement ¶ in §6.3; purge "semantic centroid" | LabelPlacement | ◐ |
| A12 | R2.W4 | Undirected statement + exit-door definition (§3.2); one-way limitation (§11) | — | ☐ |
| A13 | R2.W5 | Label vocabulary table in §5.3; unlabeled-corridor behaviour; fix colour-convention claim (see C1) | LF failure demo | ◐ |
| A14 | R3.4 | Operating-envelope rewrite of §3.1 + §11 | README conventions table | ◐ |
| A15 | R3.1a | Rewrite contribution list around multi-floor + funneling | — | ☐ |
| A16 | R3.1b | Confirm vs conference Algorithm 2; rephrase contribution #3 | **needs conference sources** | ☐ |
| A17 | R1.5 | Dataset figure (main + appendix full-page) | floorplan images | ◐ |
| A18 | R1.2 | Representation statement (§3.1) + pixel-vs-polygon rationale (§6.2) | — | ☐ |
| A19 | R1.10 | Acronym pass (VGG, BiLSTM, CTC, CRAFT, FPN, BIM, SLAM at first use) | — | ☐ |
| A20 | R1.11 | Describe actual OCR-normalization mechanism (verify in `text_interpreter.py` before writing) | code | ☐ |
| A21 | R1.12 | Text-in-wall-mask honesty note (§6.1: 9% ink, harmless there) + glyph removal in registration (§9) | measured | ◐ |
| A22 | R1.13 | Regenerate Figure 8 from a single run; caption states panel-to-panel operation | **needs submission figures** | ☐ |
| A23 | R1.14 | Degree-zero transition explanation + keep-term ablation cite | PruningAblation | ◐ |
| A24 | R1.15 | Move "conservative approach" ¶ to §10 | — | ☐ |
| A25 | R1.16 | Replace Figure 3 box plot with strip/violin | **needs submission figures/data** | ☐ |
| A26 | all | Response letter: fill `[p. XX]` locations + [bracketed placeholders] once edits frozen | — | ☐ |

## B. Section-by-section edit plan (main_revision_1.tex)

- **Abstract**: rewrite multi-floor sentence (auto-matching; remove "spatial alignment
  verification" phrasing); update pruning numbers (see C4). `\rall`
- **§1 Intro**: contribution list rewrite (A15); delta positioning; tempered automation. `\rthree`/`\rall`
- **§2 Related Work**: six [FILL] paragraphs must be written (floorplan parsing,
  indoor mapping, graph representations, text recognition, architectural detection,
  multi-floor/BIM) — required regardless of reviews.
- **§3 Problem**: representation statement (A18); input conventions fix (C1); exit-door
  definition + undirected statement (A12); operating envelope (A14).
- **§5 Text**: acronyms (A19); OCR mechanism (A20).
- **§6 Segmentation**: subnode spacing constant fix (C2); label-placement measurement (A11);
  corridor-vs-room rationale (A6); text-in-mask note (A21).
- **§7 Doors**: funneling already corrected (A2 — colour-mark it); transition-integration
  description fix (C3).
- **§8 Graph**: pruning description fix (C5); numbers update (C4); move ¶ (A24).
- **§9 Multi-floor**: major rewrite — automatic matching subsections (registration,
  assignment-with-rejection, warnings), inter-floor weights (A5), validation reframed as
  verification of machine output; keep manual-mapping subsection as override.
- **§10 Evaluation**: dataset table + figure; text accuracy; door degradation; graph metrics
  (exhaustive layer); funneling/pruning/density ablations; matcher robustness; fidelity;
  downstream tasks (drafted); baseline; multi-floor eval; timing table (uncomment + refresh).
- **§11 Discussion**: operating envelope; one-way doors; registration/mirroring limitations;
  outdoor-routing sparsity (from fidelity gap); repair mechanism as future work.
- **§12 Conclusion**: align with revised contributions.
- **Appendices + Acks + Related Work fills**: complete all [FILL] stubs.

## C. Paper-vs-implementation inconsistencies (fix in revision, colour `\rall`)

| # | Location | Paper says | Implementation reality |
|---|---|---|---|
| C1 | §3.1(2), §6.1 | corridors/outdoors identified by **fill colours** (pink/blue) | semantics are **label-driven** ("hall"/"na"); no colour classification exists |
| C2 | §6.3 | subnode grid spacing "default: 30 pixels" | code uses **60 px** (`spacing_px=60`) |
| C3 | §7.4 | transitions "connect to the nearest corridor node" | actual: synthetic **entry doors** via wall tracing, then corridor linkage |
| C4 | §8.3, abstract | pruning 2136→875 nodes, 6259→1104 edges | current: **2139→878 / 6262→1118** (post bug-fix regeneration) |
| C5 | §8.3 | pruning = "edges not on any shortest path between door nodes" | actual keep-set: family paths + all-pairs main rooms + exits×rooms + transitions×rooms |
| C6 | §9.4 | inter-floor weight "nominal traversal cost" | now metric vertical-travel weights (A5 covers) |
| C7 | §10 (timing table) | pruning 2.0 s, total 77.6 s | verify against current timer logs when refreshing table |

## D. Figures & tables to produce

1. Dataset figure (plans + convention inset) — A17.
2. Pipeline architecture figure (placeholder in §4 — still unmade).
3. Knockout curves (exists: `results/evaluation/DoorKnockout/knockout_curves.png`).
4. Matcher robustness panels + registration-error envelope (exist under `MatcherEval/`).
5. Route-fidelity overlays (exist under `RouteFidelity/overlays/`).
6. Density trade-off plot (data in `CorridorDensity/`; plot to render).
7. Regenerated Figure 8; violin/strip replacement for Figure 3 (need submission assets).
8. Tables: dataset, multi-site metrics, ablation summaries, timing (refresh), baseline comparison.

## Workflow reminders

- Edit **only** `main_revision_1.tex`; run `./make_clean.sh` after each session.
- Wrap every insertion in the requesting reviewer's colour macro; deletions in `\rdel{}`.
- Update `table_of_changes.tex` and the `[p. XX]` markers in `response_to_reviewers.tex` last.
- No LaTeX on this machine — compile checks must run on the author's side.
