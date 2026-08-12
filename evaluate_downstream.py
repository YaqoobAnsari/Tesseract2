"""
evaluate_downstream.py - Downstream-task validation (Task E)

Two concrete jobs run on a merged multi-floor graph:

1. Accessibility routing: route between rooms on different floors with
   stairs forbidden (elevators only). Reports, per cross-floor room pair
   sample: reachable normally? reachable accessibly? length ratio. This
   exercises typed transition nodes - impossible without them - and flags
   buildings/sections with no step-free route.

2. Egress distances: for every main room on every floor, graph distance to
   the nearest exit door. Reports the distribution and writes per-room
   values. NOTE: this is the measurement half of the egress task; the error
   distribution against a hand-built ground-truth graph still requires that
   ground truth (not yet available for these floors).

Usage:
    python evaluate_downstream.py [--sequence FF_SF] [--pairs 200]

Outputs (Multifloor_Results/Downstream/<sequence>/):
    accessibility.txt / accessibility.csv
    egress.txt / egress.csv
"""

import argparse
import csv
import math
import json
import os

import networkx as nx
import numpy as np

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MULTIFLOOR_DIR = os.path.join(BASE_PATH, "Multifloor_Results")


def load_merged(sequence):
    path = os.path.join(MULTIFLOOR_DIR, "Jsons", sequence,
                        "merged_multi_floor_graph.json")
    with open(path) as f:
        data = json.load(f)
    G = nx.Graph()
    for nd in data['nodes']:
        G.add_node(nd['id'], **{k: v for k, v in nd.items() if k != 'id'})
    for ed in data['edges']:
        G.add_edge(ed['source'], ed['target'],
                   weight=float(ed.get('weight') or 1.0),
                   edge_type=ed.get('edge_type'))
    return G


def original_id(G, n):
    return G.nodes[n].get('original_id') or str(n).split('_', 1)[-1]


def is_stairs(G, n):
    return (G.nodes[n].get('type') == 'floor_transition'
            and original_id(G, n).startswith('stairs'))


def is_elevator(G, n):
    return (G.nodes[n].get('type') == 'floor_transition'
            and original_id(G, n).startswith('elevator'))


def main_rooms_by_floor(G):
    out = {}
    for n, d in G.nodes(data=True):
        if d.get('type') == 'room' and '_subnode_' not in str(n):
            out.setdefault(d.get('floor', '?'), []).append(n)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sequence', default="FF_SF")
    ap.add_argument('--pairs', type=int, default=200)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    G = load_merged(args.sequence)
    out_dir = os.path.join(MULTIFLOOR_DIR, "Downstream", args.sequence)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    stairs = [n for n in G if is_stairs(G, n)]
    elevators = [n for n in G if is_elevator(G, n)]
    inter_edges = [(u, v, d) for u, v, d in G.edges(data=True)
                   if d.get('edge_type') == 'inter_floor']
    rooms = main_rooms_by_floor(G)
    floors = sorted(rooms)

    print(f"Merged graph {args.sequence}: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    print(f"Floors: {floors} | stairs: {len(stairs)} | elevators: "
          f"{len(elevators)} | inter-floor edges: {len(inter_edges)}")
    for u, v, d in inter_edges:
        via = 'stairs' if is_stairs(G, u) else 'elevator'
        print(f"  inter-floor via {via}: {u} <-> {v}")

    # ------------------------------------------------------------------
    # 1. Accessibility routing (stairs forbidden)
    # ------------------------------------------------------------------
    G_acc = G.copy()
    G_acc.remove_nodes_from(stairs)

    if len(floors) < 2:
        raise SystemExit("Need at least 2 floors for accessibility routing")
    fa, fb = floors[0], floors[-1]
    pair_count = min(args.pairs, len(rooms[fa]) * len(rooms[fb]))
    sampled = set()
    while len(sampled) < pair_count:
        a = rooms[fa][rng.integers(len(rooms[fa]))]
        b = rooms[fb][rng.integers(len(rooms[fb]))]
        sampled.add((a, b))

    acc_rows = []
    n_normal = n_acc = 0
    ratios = []
    for a, b in sorted(sampled):
        try:
            d_norm = nx.shortest_path_length(G, a, b, weight='weight')
            n_normal += 1
        except nx.NetworkXNoPath:
            d_norm = None
        try:
            d_acc = nx.shortest_path_length(G_acc, a, b, weight='weight')
            n_acc += 1
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            d_acc = None
        if d_norm and d_acc:
            ratios.append(d_acc / d_norm)
        acc_rows.append({'from': a, 'to': b, 'dist_normal': d_norm,
                         'dist_accessible': d_acc,
                         'ratio': (d_acc / d_norm) if d_norm and d_acc else None})

    acc_lines = [
        f"Accessibility routing - {args.sequence} "
        f"(floor {fa} <-> floor {fb}, {pair_count} room pairs)",
        f"Reachable with all transitions:    {n_normal}/{pair_count}",
        f"Reachable with stairs forbidden:   {n_acc}/{pair_count}",
        f"Step-free detour ratio (accessible/normal): "
        f"mean {np.mean(ratios):.3f}, p90 {np.percentile(ratios, 90):.3f}"
        if ratios else
        "No step-free routes exist between these floors: the merged graph "
        "has no elevator connection - an accessibility gap this analysis "
        "is designed to surface.",
    ]
    ex = next((r for r in acc_rows if r['ratio']), None)
    if ex:
        acc_lines.append(f"Example: {ex['from']} -> {ex['to']}  "
                         f"normal {ex['dist_normal']:.0f}px, "
                         f"accessible {ex['dist_accessible']:.0f}px")
    with open(os.path.join(out_dir, "accessibility.txt"), 'w') as f:
        f.write("\n".join(acc_lines) + "\n")
    with open(os.path.join(out_dir, "accessibility.csv"), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(acc_rows[0].keys()))
        wr.writeheader()
        wr.writerows(acc_rows)
    print("\n" + "\n".join(acc_lines))

    # ------------------------------------------------------------------
    # 2. Egress distances (per main room, nearest exit door)
    # ------------------------------------------------------------------
    exits = [n for n in G if original_id(G, n).startswith('exit_door_')]
    eg_rows = []
    unreachable = 0
    # Multi-source Dijkstra from all exits at once.
    dist = nx.multi_source_dijkstra_path_length(G, exits, weight='weight') \
        if exits else {}
    for fl in floors:
        for r in rooms[fl]:
            d = dist.get(r)
            if d is None:
                unreachable += 1
            eg_rows.append({'floor': fl, 'room': r,
                            'egress_dist_px': d})
    vals = [r['egress_dist_px'] for r in eg_rows
            if r['egress_dist_px'] is not None]
    eg_lines = [
        f"Egress distances - {args.sequence} ({len(exits)} exit doors, "
        f"{len(eg_rows)} main rooms)",
        f"Rooms with an egress path: {len(vals)}/{len(eg_rows)} "
        f"(unreachable: {unreachable})",
        f"Distance to nearest exit (px): mean {np.mean(vals):.0f}, "
        f"median {np.median(vals):.0f}, p90 {np.percentile(vals, 90):.0f}, "
        f"max {np.max(vals):.0f}",
        "",
        "NOTE: distances are in pixels (no metres-per-pixel scale exists in "
        "the pipeline yet) and are measurements, not an evaluation - the "
        "egress ERROR distribution requires a hand-built ground-truth graph "
        "for at least one floor, which does not yet exist for these images.",
    ]
    with open(os.path.join(out_dir, "egress.txt"), 'w') as f:
        f.write("\n".join(eg_lines) + "\n")
    with open(os.path.join(out_dir, "egress.csv"), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['floor', 'room', 'egress_dist_px'])
        wr.writeheader()
        wr.writerows(eg_rows)
    print("\n" + "\n".join(eg_lines))
    print(f"\nSaved to: {out_dir}")


if __name__ == '__main__':
    main()
