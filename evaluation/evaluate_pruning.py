"""
evaluate_pruning.py - Pruning ablation (R2-W6)

Two studies on already-processed floors:

1. Keep-term ablation. The pruning keep-set is a union of four terms
   (re-implemented here from the pre-pruning graph, mirroring
   BuildingGraph.connect_all_rooms): family paths, all-pairs main-room
   paths, main-room-to-exit paths, and main-room-to-transition paths (plus
   never-pruning transition nodes). Each variant drops one term and reports
   what breaks - isolating the contribution of transition preservation and
   exit-path preservation.

2. Pre vs post comparison. Route-length inflation and query-time change
   caused by the actual pipeline pruning.

Usage:
    python evaluate_pruning.py [--image "FF part 1upE.png"]

Output: results/evaluation/PruningAblation/pruning_<image>.txt
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import (BASE_PATH, INPUT_IMAGES_DIR, RESULTS_DIR,
                    MULTIFLOOR_RESULTS_DIR, EVAL_RESULTS_DIR)

import argparse
import math
import os
import time

import networkx as nx
import numpy as np

from utils.metrics import (load_graph_json, compute_metrics, main_rooms,
                           exit_doors, transitions, format_metrics)

OUT_DIR = os.path.join(EVAL_RESULTS_DIR, "PruningAblation")


def is_subnode(G, n):
    d = G.nodes[n]
    return d.get('type') == 'room' and (d.get('is_subnode') or '_subnode_' in str(n))


def keep_sets(G, use_family=True, use_main_pairs=True, use_exits=True,
              use_transitions=True, protect_transitions=True):
    rooms = main_rooms(G)
    exits = exit_doors(G)
    trans = transitions(G)

    keep_nodes, keep_edges = set(), set()

    def add_path(sp):
        keep_nodes.update(sp)
        for u, v in zip(sp[:-1], sp[1:]):
            keep_edges.add((u, v) if u < v else (v, u))

    if use_family:
        for r in rooms:
            for s in G.nodes:
                if is_subnode(G, s) and (
                        G.nodes[s].get('parent_room_id') == r
                        or str(s).split('_subnode_')[0] == r):
                    try:
                        add_path(nx.shortest_path(G, s, r, weight='weight'))
                    except nx.NetworkXNoPath:
                        pass

    if use_main_pairs:
        for i, a in enumerate(rooms):
            for b in rooms[i + 1:]:
                try:
                    add_path(nx.shortest_path(G, a, b, weight='weight'))
                except nx.NetworkXNoPath:
                    pass

    if use_exits:
        for e in exits:
            best_sp, best_cost = None, math.inf
            for r in rooms:
                try:
                    sp = nx.shortest_path(G, r, e, weight='weight')
                    c = sum(G[u][v].get('weight', 1.0)
                            for u, v in zip(sp[:-1], sp[1:]))
                    if c < best_cost:
                        best_cost, best_sp = c, sp
                except nx.NetworkXNoPath:
                    continue
            if best_sp:
                add_path(best_sp)

    if use_transitions:
        for t in trans:
            for r in rooms:
                try:
                    add_path(nx.shortest_path(G, r, t, weight='weight'))
                except nx.NetworkXNoPath:
                    pass

    if protect_transitions:
        keep_nodes.update(trans)

    H = nx.Graph()
    for n in keep_nodes:
        H.add_node(n, **G.nodes[n])
    for u, v in keep_edges:
        H.add_edge(u, v, **G[u][v])
    return H


def mean_exit_route(G):
    exits = [n for n in G if str(n).startswith('exit_door_')]
    rooms = [n for n in main_rooms(G) if n in G]
    if not exits:
        return float('nan'), 0
    dist = nx.multi_source_dijkstra_path_length(G, [e for e in exits if e in G],
                                                weight='weight')
    vals = [dist[r] for r in rooms if r in dist]
    return (np.mean(vals) if vals else float('nan')), len(vals)


def query_time(G, n_queries, rng):
    nodes = list(G.nodes)
    t0 = time.perf_counter()
    done = 0
    for _ in range(n_queries):
        a, b = rng.choice(len(nodes), 2, replace=False)
        try:
            nx.shortest_path_length(G, nodes[a], nodes[b], weight='weight')
            done += 1
        except nx.NetworkXNoPath:
            pass
    return (time.perf_counter() - t0) / max(1, done) * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image', default="FF part 1upE.png")
    ap.add_argument('--queries', type=int, default=100)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    stem = os.path.splitext(args.image)[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    pre = load_graph_json(os.path.join(
        RESULTS_DIR, "Json", stem, f"{stem}_pre_pruning.json"))
    post = load_graph_json(os.path.join(
        RESULTS_DIR, "Json", stem, f"{stem}_post_pruning.json"))

    lines = [f"Pruning ablation - {args.image}", ""]

    # ---- Study 1: keep-term ablation ----
    variants = {
        'full': {},
        'no-transition-term': {'use_transitions': False,
                               'protect_transitions': False},
        'no-exit-term': {'use_exits': False},
        'main-pairs+family only': {'use_exits': False,
                                   'use_transitions': False,
                                   'protect_transitions': False},
    }
    lines.append("Keep-term ablation (re-implemented keep-set on the "
                 "pre-pruning graph):")
    for name, kw in variants.items():
        H = keep_sets(pre, **kw)
        m = compute_metrics(H)
        lines.append(format_metrics(m, name[:18]))
    lines.append("")

    # ---- Study 2: actual pre vs post ----
    lines.append("Actual pipeline pruning, pre vs post:")
    for label, G in [('pre-pruning', pre), ('post-pruning', post)]:
        m = compute_metrics(G)
        route, n_ok = mean_exit_route(G)
        q = query_time(G, args.queries, rng)
        lines.append(format_metrics(m, label))
        lines.append(f"{'':>18}  mean room->exit route {route:.1f}px "
                     f"({n_ok} rooms), mean query {q:.2f}ms")
    pre_route, _ = mean_exit_route(pre)
    post_route, _ = mean_exit_route(post)
    if np.isfinite(pre_route) and np.isfinite(post_route) and pre_route > 0:
        lines.append(f"\nRoute inflation from pruning: "
                     f"{(post_route / pre_route - 1):+.2%}")

    out = os.path.join(OUT_DIR, f"pruning_{stem}.txt")
    with open(out, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
