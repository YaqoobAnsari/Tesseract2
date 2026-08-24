"""
evaluate_door_knockout.py - Connectivity under door-detector degradation (Task F)

Simulates worse door detectors by deleting doors from the PRE-pruning graph
(where the full corridor/room mesh is intact; connectivity is decided before
pruning, which only sparsifies while preserving connected paths) and
measuring completeness and exit reachability with the metric layer.

Experiments per floor:
  1. Random deletion of detected doors (r2c + r2r + c2c + exit) at
     0..50%, N draws per level, with error bars.
  2. By-type deletion at the same levels: r2c-only vs c2c-only. If the
     paper's causal claim holds, the c2c curve falls off a cliff while the
     r2c curve barely moves (rooms lose only their own access).
  3. Single-door sensitivity: delete each c2c door alone and report the
     completeness drop it alone causes.

Entry doors (synthetic, transition-serving) are never deleted.

Usage:
    python evaluate_door_knockout.py [--floors "FF part 1upE,FF part 2up,FF part 3up"]

Outputs (results/evaluation/DoorKnockout/):
    knockout_<floor>.csv, knockout_curves.png, summary.txt
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import (BASE_PATH, INPUT_IMAGES_DIR, RESULTS_DIR,
                    MULTIFLOOR_RESULTS_DIR, EVAL_RESULTS_DIR)

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.metrics import load_graph_json, compute_metrics

OUT_DIR = os.path.join(EVAL_RESULTS_DIR, "DoorKnockout")

LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
CATEGORIES = {
    'random-all': ('r2c_door_', 'r2r_door_', 'c2c_door_', 'exit_door_'),
    'r2c-only': ('r2c_door_',),
    'c2c-only': ('c2c_door_',),
}


def doors_of(G, prefixes):
    return [n for n in G if str(n).startswith(prefixes)]


def knock(G, remove):
    H = G.copy()
    H.remove_nodes_from(remove)
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--floors', default="FF part 1upE,FF part 2up,FF part 3up")
    ap.add_argument('--draws', type=int, default=20)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    floors = [f.strip() for f in args.floors.split(',')]

    all_rows = {}
    sens_lines = []
    for stem in floors:
        path = os.path.join(RESULTS_DIR, "Json", stem,
                            f"{stem}_pre_pruning.json")
        G = load_graph_json(path)
        base = compute_metrics(G)
        print(f"\n{stem}: baseline completeness "
              f"{base['completeness']:.3f}, exit reach "
              f"{base['exit_reachability']}")

        rows = []
        for cat, prefixes in CATEGORIES.items():
            pool = doors_of(G, prefixes)
            for frac in LEVELS:
                k = int(round(frac * len(pool)))
                n_draws = 1 if k == 0 else args.draws
                for d in range(n_draws):
                    remove = list(rng.choice(pool, k, replace=False)) if k else []
                    m = compute_metrics(knock(G, remove))
                    rows.append({
                        'floor': stem, 'category': cat, 'fraction': frac,
                        'n_removed': k, 'draw': d,
                        'completeness': m['completeness'],
                        'exit_reachability': m['exit_reachability'],
                        'n_components': m['n_components'],
                    })
            got = [r for r in rows if r['category'] == cat]
            by_level = {
                frac: np.mean([r['completeness'] for r in got
                               if r['fraction'] == frac])
                for frac in LEVELS}
            print(f"  {cat:>10} ({len(pool):>2} doors): completeness by level "
                  + " ".join(f"{by_level[f]:.2f}" for f in LEVELS))
        all_rows[stem] = rows

        # Single-door sensitivity for c2c doors.
        for door in doors_of(G, ('c2c_door_',)):
            m = compute_metrics(knock(G, [door]))
            drop = base['completeness'] - m['completeness']
            er = (f"{base['exit_reachability'] - m['exit_reachability']:+.3f}"
                  if base['exit_reachability'] is not None
                  and m['exit_reachability'] is not None else "n/a")
            sens_lines.append(f"  {stem} {door}: completeness "
                              f"{base['completeness']:.3f} -> "
                              f"{m['completeness']:.3f} (drop {drop:.3f}), "
                              f"exit reach change {er}")

        with open(os.path.join(OUT_DIR, f"knockout_{stem}.csv"), 'w',
                  newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)

    # ---- figure: one row per floor, completeness + exit reachability ----
    colors = {'random-all': '#555555', 'r2c-only': '#0072B2',
              'c2c-only': '#D55E00'}
    fig, axes = plt.subplots(len(floors), 2,
                             figsize=(16, 4.6 * len(floors)), squeeze=False)
    handles = None
    for fi, stem in enumerate(floors):
        rows = all_rows[stem]
        for mi, metric in enumerate(['completeness', 'exit_reachability']):
            ax = axes[fi][mi]
            for cat in CATEGORIES:
                got = [r for r in rows if r['category'] == cat
                       and r[metric] is not None]
                means = [np.mean([r[metric] for r in got if r['fraction'] == f])
                         for f in LEVELS]
                stds = [np.std([r[metric] for r in got if r['fraction'] == f])
                        for f in LEVELS]
                ax.errorbar([f * 100 for f in LEVELS], means, yerr=stds,
                            marker='o', markersize=7, linewidth=2, capsize=4,
                            label=cat, color=colors[cat])
            ax.set_title(f"{stem}, {metric.replace('_', ' ')}", fontsize=17)
            ax.set_xlabel('doors deleted (%)', fontsize=15)
            ax.set_ylabel(metric.replace('_', ' '), fontsize=15)
            ax.tick_params(labelsize=13)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(CATEGORIES),
               fontsize=16, framealpha=0.95, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    plot_path = os.path.join(OUT_DIR, "knockout_curves.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')

    with open(os.path.join(OUT_DIR, "summary.txt"), 'w') as f:
        f.write("Door-knockout experiment (pre-pruning graphs, "
                f"{args.draws} draws/level)\n\n")
        for stem in floors:
            rows = all_rows[stem]
            f.write(f"{stem}:\n")
            for cat in CATEGORIES:
                got = [r for r in rows if r['category'] == cat]
                line = "  " + f"{cat:>10}: "
                line += "  ".join(
                    f"{int(fr*100)}%:{np.mean([r['completeness'] for r in got if r['fraction'] == fr]):.3f}"
                    for fr in LEVELS)
                f.write(line + "\n")
            f.write("\n")
        f.write("Single c2c door sensitivity (delete one door alone):\n")
        f.write("\n".join(sens_lines) + "\n")
        f.write("\nNOTE: deletions are applied to the detector's OUTPUT "
                "(no door ground truth exists for these floors), so curves "
                "measure sensitivity relative to the current detector, not "
                "absolute recall. Entry doors (transition-serving, "
                "synthetic) are never deleted.\n")

    print(f"\nSaved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
