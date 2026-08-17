# Evaluation Suite — Methodology and Headline Results

All numbers below were produced on the included demo data (see the README's
*Data Notes* — in particular, the FF↔SF pair duplicates one drawing, and
`GF part 1upE` carries programmatically added labels). Every experiment is
reproducible with the listed command; outputs land under
`results/evaluation/<ExperimentName>/`.

## Metric layer — `evaluate_metrics.py`

Definitions (exhaustive, component-based — never sampled):

- **completeness** — fraction of unordered main-room pairs connected by a path
- **exit reachability** — fraction of main rooms with a path to ≥1 exit door
- **transition reachability** — fraction of main rooms with a path to ≥1 stairs/elevator

Headline: most processed floors are fully connected; three sections are
fragmented as built (`FF part 3upE` 0.513 completeness / 7 components,
`FF part 4up` 0.758, `FF part 6up` 0.716) — the corridor-door failure mode,
made measurable.

## Automatic transition matching — `evaluate_matcher.py`

Synthetically perturbs a real floor by known similarity transforms
(+5 px transition jitter), then measures correspondence accuracy against a
greedy nearest-neighbour baseline, plus registration recovery error
(estimated vs true transform, mean corner displacement).

Headline (8 trials/level): matcher accuracy **1.000 on every sweep** —
translation to 400 px, rotation to 30°, scale 0.7–1.4×, twin-shaft
separations down to 25 px, and shaft-termination (rejection) tests.
Nearest neighbour degrades to **0.75** as twin separation shrinks.
Registration recovery: median **0.5 px** (translation), **4.4 px**
(rotation), **4.9 px** (scale).

Caveats are printed into `summary.txt` (shared canvas, shared node ids,
near-identical context features — the synthetic setting measures
robustness, not real-floor accuracy).

## Funneling ablation — `evaluate_funneling.py`

Three intra-room connectivity strategies on the same floor: dense
subnode-to-door wiring (`full`), the shipped construction (`funneling`),
and subnode-free rooms (`one-per-room`).

Headline (`FF part 1upE`): all variants complete; funneling uses
**167 intra-room edges vs 1,869 dense** (second floor: exactly n edges for
n subnodes, ratio 1.00) at **7.2% route stretch** vs the dense reference;
one-per-room pays 13.5% and loses intra-room spatial fidelity.

## Door knockout — `evaluate_door_knockout.py`

Monte Carlo deletion of detected doors from the pre-pruning graph
(0–50%, 20 draws/level; random-all vs r2c-only vs c2c-only), plus
single-door sensitivity. Deletions apply to detector *output* (no door
ground truth exists for these floors), so curves measure sensitivity
relative to the current detector.

Headline: per-percentage, r2c deletions hurt completeness more (rooms are
single-homed); most c2c doors are redundant through the corridor mesh —
**but sole-bridge c2c doors are individually catastrophic**: deleting one
door (`FF part 2up`, `c2c_door_6`) costs **0.511 completeness and 0.543
exit reachability** on its own. Bridge doors are exactly the graph's cut
vertices, which suggests a principled repair/verification target.

## Corridor density — `evaluate_corridor_density.py`

Full pipeline reruns at corridor grid spacings 80/60/40/30/20 px
(`Main.py --corridor-distance`), comparing size, connectivity, and
room-to-exit route lengths against the 20 px default.

Headline: **60 px suffices** — completeness 1.000, routes within 4.6% of
the default, **57% fewer nodes**. At 80 px the corridor mesh fragments
catastrophically (completeness 0.49, routes +163%). Density buys a safety
margin against narrow corridors; the failure mode is a cliff, not a slope.

## Pruning ablation — `evaluate_pruning.py`

(1) Keep-term ablation: the pruning keep-set is re-derived from the
pre-pruning graph with individual terms disabled. Dropping the transition
term silently removes **all transitions**; dropping the exit term removes
**all exits** — each term is load-bearing.
(2) Actual pre/post comparison: **59% node / 82% edge reduction, +0.00%
route inflation, 4.2× query speedup** (2.10 → 0.50 ms).

## Downstream tasks — `evaluate_downstream.py`

Stairs-forbidden (elevator-only) routing and per-room egress distances on
merged multi-floor graphs.

Headline: FF↔SF — 200/200 sampled cross-floor pairs reachable step-free,
mean detour 2.02× (p90 3.93×). Real GF↔FF pair — 102/102 pairs reachable
via stairs but **0/102 step-free** (no elevator correspondence
established): the analysis surfaces an accessibility gap untyped graphs
cannot express. Egress distances are reported in pixels; the egress
*error* distribution awaits a hand-built ground-truth graph.

## Text accuracy — `evaluate_text_accuracy.py`

Detected labels vs a hand-enumerated list of every label visible on the
drawing (enumeration by visual inspection — verify before quoting).

Headline: `FF part 1upE` — **100% detection (43/43), 95.3% recognition
(41/43)**; both failures truncate a leading digit (`1011`→`101`,
`1022`→`022`) and still yield valid room nodes, so connectivity is
unaffected. `GF part 1upE` — 9/9 and 9/9. A label missed *entirely* leaves
its region without a semantic seed (the unlabeled `FF part 1up` variant
produces zero transitions).

## Label placement — `evaluate_label_placement.py`

Offset between each room's label position (used as the main node) and the
geometric centroid of its flood-filled region, over 172 rooms on 9 floors.

Headline: median offset **0.24 equivalent-radii** (18 px), 92% of labels
inside the room's inscribed circle, worst case 0.98 radii — architectural
labels are empirically central, so using them as main-node positions is
sound.

## Inter-floor edge weights (MultiFloor)

Inter-floor edges now carry vertical-travel weights instead of a nominal
1.0: `factor x storey height (4 m) x px-per-metre`, with the drawing scale
bootstrapped from detected door widths (median short side / 0.9 m; FF
31.1 px/m, GF 32.2 px/m — independently consistent). Stairs factor 2.2,
elevator 1.0; falls back to 1.0 with a warning when no scale is available.
Corrected FF↔SF step-free detour: 2.02× → **1.79× mean**. The same
door-width scale also feeds registration as a prior (`scale_hints`), which
re-adjudicated the ambiguous GF↔FF pair (three weak hypotheses disagree —
flagged for author judgment).

## Route fidelity — `evaluate_route_fidelity.py`

Graph routes vs free-space geodesics (true shortest walkable paths through
non-wall pixels, doorways opened), with two bracketing references: FULL
free space (permissive) and INTERIOR-only (the pipeline's semantic
outside class excluded). Verification overlays rendered per pair.

Headline: `FF part 1upE` — stretch **1.047 mean / 1.093 p90** against both
references. `FF part 2up` — 1.24 (interior) to 2.30 (full): the gap
quantifies sparse outdoor routing. `geodesic/euclid` runs 1.35–1.67 mean
(max 3.3×) — the measured reason straight-line Euclidean was a weak
fidelity reference.

## Skeleton baseline — `evaluate_baseline_skeleton.py`

Structure-only medial-axis routing backbone, granted Tesseract++'s own
room/exit detections, scored on the same pairs against the same geodesic
reference.

Headline: the skeleton needs **3.7–12× more nodes**, is **incomplete**
(0.78–0.83; it fragments at doorway gaps), has worse stretch (1.23/1.38 vs
1.05/1.23), and is 37–94× slower per query — and expresses none of the
typed downstream tasks (accessibility, egress, multi-floor merging).
