"""
evaluate_corridor_density.py - Corridor sampling density ablation (R1)

Reruns the full single-floor pipeline at several corridor grid spacings and
measures what density actually buys: graph size, connectivity, and route
quality (subnode-to-nearest-exit lengths, compared against the densest
setting as reference).

Runs LAST at the default 20px so the on-disk results end in the default
state.

Usage:
    python evaluate_corridor_density.py [--image "FF part 1upE.png"]

Outputs (results/evaluation/CorridorDensity/):
    density_<image>.csv, density_<image>.txt
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import (BASE_PATH, INPUT_IMAGES_DIR, RESULTS_DIR,
                    MULTIFLOOR_RESULTS_DIR, EVAL_RESULTS_DIR)

import argparse
import csv
import json
import math
import os
import subprocess
import sys

import networkx as nx
import numpy as np

OUT_DIR = os.path.join(EVAL_RESULTS_DIR, "CorridorDensity")
SPACINGS = [80, 60, 40, 30, 20]   # densest (default) runs last


def load_post(stem):
    path = os.path.join(RESULTS_DIR, "Json", stem,
                        f"{stem}_post_pruning.json")
    with open(path) as f:
        data = json.load(f)
    G = nx.Graph()
    for nd in data['nodes']:
        G.add_node(nd['id'], **{k: v for k, v in nd.items() if k != 'id'})
    for ed in data['edges']:
        G.add_edge(ed['source'], ed['target'],
                   weight=float(ed.get('weight') or 1.0))
    return G


def route_lengths(G):
    """Room-main-node -> nearest exit door graph distance, per room."""
    exits = [n for n in G if str(n).startswith('exit_door_')]
    rooms = [n for n, d in G.nodes(data=True)
             if d.get('type') == 'room' and '_subnode_' not in str(n)]
    dist = nx.multi_source_dijkstra_path_length(G, exits, weight='weight')
    return {r: dist[r] for r in rooms if r in dist}



def plot_density(rows, stem):
    """Density trade-off figure. Route-error annotations sit at the top of
    the chart, drawn last so they stay above the lines."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: int(r['spacing']))
    sp = [int(r['spacing']) for r in rows]
    nodes = [int(r['nodes']) for r in rows]
    comp = [float(r['completeness']) for r in rows]
    err = [float(r['mean_route_rel_err_vs_ref']) * 100 for r in rows]

    fig, ax1 = plt.subplots(figsize=(8.5, 4.6))
    ax1.plot(sp, comp, marker='o', markersize=8, linewidth=2.2,
             color='#0072B2', label='completeness', zorder=3)
    ax1.set_xlabel('corridor grid spacing (px)', fontsize=15)
    ax1.set_ylabel('pairwise room completeness', color='#0072B2', fontsize=15)
    ax1.tick_params(labelsize=13)
    ax1.set_ylim(0, 1.18)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(sp, nodes, marker='s', markersize=8, linewidth=2.2,
             color='#D55E00', label='graph nodes', zorder=3)
    ax2.set_ylabel('post-pruning graph nodes', color='#D55E00', fontsize=15)
    ax2.tick_params(labelsize=13)
    for s, e in zip(sp, err):
        label = f"{e:.1f}%" if e < 100 else f"{e:.0f}%"
        ax1.annotate(label, (s, 1.10), ha='center', fontsize=13,
                     color='black', zorder=10, annotation_clip=False,
                     bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                               edgecolor='#999999', linewidth=0.6))
    ax1.annotate('route error vs 20 px reference', (min(sp), 1.10),
                 xytext=(-6, 16), textcoords='offset points',
                 fontsize=12, color='black', zorder=10, annotation_clip=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"density_{stem}.png"), dpi=170,
                bbox_inches='tight')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image', default="FF part 1upE.png")
    args = ap.parse_args()
    stem = os.path.splitext(args.image)[0]
    os.makedirs(OUT_DIR, exist_ok=True)

    from utils.metrics import compute_metrics

    results = {}
    for spacing in SPACINGS:
        print(f"\n=== corridor spacing {spacing}px ===")
        r = subprocess.run(
            [sys.executable, os.path.join(BASE_PATH, "Main.py"), args.image,
             "--corridor-distance", str(spacing)],
            cwd=BASE_PATH, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-1500:])
            print(r.stderr[-1500:])
            raise SystemExit(f"pipeline failed at spacing {spacing}")
        G = load_post(stem)
        m = compute_metrics(G)
        routes = route_lengths(G)
        corridor_nodes = sum(1 for n, d in G.nodes(data=True)
                             if d.get('type') == 'corridor')
        results[spacing] = {'metrics': m, 'routes': routes,
                            'corridor_nodes': corridor_nodes}
        print(f"  nodes={m['n_nodes']} (corridor {corridor_nodes}) "
              f"edges={m['n_edges']} completeness={m['completeness']:.3f} "
              f"exit_reach={m['exit_reachability']}")

    ref_routes = results[SPACINGS[-1]]['routes']   # densest = reference
    rows = []
    lines = [f"Corridor density ablation - {args.image}",
             f"Reference spacing: {SPACINGS[-1]}px (default)", "",
             f"{'spacing':>8} {'nodes':>6} {'corridor':>9} {'edges':>6} "
             f"{'complete':>9} {'exit_reach':>10} {'mean route':>11} "
             f"{'route err vs 20px':>18}"]
    for spacing in sorted(results, reverse=True):
        r = results[spacing]
        m = r['metrics']
        common = [k for k in r['routes'] if k in ref_routes]
        errs = [abs(r['routes'][k] - ref_routes[k]) / ref_routes[k]
                for k in common if ref_routes[k] > 0]
        mean_route = np.mean(list(r['routes'].values())) if r['routes'] else float('nan')
        err = np.mean(errs) if errs else 0.0
        rows.append({'spacing': spacing, 'nodes': m['n_nodes'],
                     'corridor_nodes': r['corridor_nodes'],
                     'edges': m['n_edges'],
                     'completeness': m['completeness'],
                     'exit_reachability': m['exit_reachability'],
                     'mean_route_px': mean_route,
                     'mean_route_rel_err_vs_ref': err})
        lines.append(f"{spacing:>8} {m['n_nodes']:>6} {r['corridor_nodes']:>9} "
                     f"{m['n_edges']:>6} {m['completeness']:>9.3f} "
                     f"{str(m['exit_reachability']):>10} {mean_route:>11.1f} "
                     f"{err:>18.3%}")

    txt = os.path.join(OUT_DIR, f"density_{stem}.txt")
    with open(txt, 'w') as f:
        f.write("\n".join(lines) + "\n")
        f.write("\nNOTE: route error is measured on rooms present at both "
                "densities, as |route - route_ref| / route_ref for the "
                "room-to-nearest-exit task.\n")
    with open(os.path.join(OUT_DIR, f"density_{stem}.csv"), 'w',
              newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    plot_density(rows, stem)
    print("\n" + "\n".join(lines))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
