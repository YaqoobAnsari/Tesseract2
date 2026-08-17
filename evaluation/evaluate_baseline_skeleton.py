"""
evaluate_baseline_skeleton.py - Medial-axis baseline comparison (R3-1c)

A classic structure-only alternative to Tesseract++'s typed graphs: the
free-space skeleton (medial axis) as a routing backbone. Rooms and exit
doors attach to their nearest skeleton point (Tesseract++'s own label/door
detections are granted to the baseline, so the comparison isolates the
ROUTING STRUCTURE, not the semantic extraction - a deliberately generous
baseline).

Both graphs are scored on the same room/exit pairs against the same
interior free-space geodesic reference:

  size          nodes/edges (skeleton after degree-2 chain contraction)
  completeness  fraction of room pairs connected
  exit reach    fraction of rooms reaching an exit
  stretch       route length / interior geodesic (lower = better)
  query time    mean single-pair Dijkstra

What the skeleton cannot provide regardless of numbers: typed rooms,
doors, corridors, or stairs/elevator transitions - i.e. none of the
downstream tasks (accessibility routing, egress semantics, multi-floor
merging) are expressible on it.

Usage:
    python evaluation/evaluate_baseline_skeleton.py [--floors "FF part 1upE,FF part 2up"]

Output: results/evaluation/BaselineSkeleton/report.txt (+ per-floor CSV)
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RESULTS_DIR, INPUT_IMAGES_DIR, EVAL_RESULTS_DIR

import argparse
import csv
import json
import math
import os
import time

import cv2
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

from utils.freespace import (traversable_mask, geodesic_from, read_distance,
                             semantic_outside_mask, DS)
from utils.metrics import load_graph_json, main_rooms, exit_doors

OUT_DIR = os.path.join(EVAL_RESULTS_DIR, "BaselineSkeleton")


def build_skeleton_graph(mask, attach_points):
    """Skeletonize the (downsampled) free space and return a weighted
    networkx graph over skeleton pixels (full-res px weights), with the
    given named points attached to their nearest skeleton pixel."""
    small = mask[::DS, ::DS]
    skel = skeletonize(small)
    ys, xs = np.nonzero(skel)
    pix = set(zip(ys.tolist(), xs.tolist()))

    S = nx.Graph()
    for (y, x) in pix:
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            n = (y + dy, x + dx)
            if n in pix:
                w = math.hypot(dy, dx) * DS
                S.add_edge((y, x), n, weight=w)

    # attach named nodes to nearest skeleton pixel
    if len(ys):
        coords = np.stack([xs, ys], axis=1) * DS   # full-res (x, y)
        for name, (px, py) in attach_points.items():
            d2 = ((coords[:, 0] - px) ** 2 + (coords[:, 1] - py) ** 2)
            k = int(np.argmin(d2))
            S.add_edge(name, (int(ys[k]), int(xs[k])),
                       weight=float(math.sqrt(d2[k])))
    return S, skel


def contracted_size(S):
    """Node/edge count after contracting degree-2 chains (junctions,
    endpoints, and attached named nodes kept)."""
    keep = [n for n in S if S.degree(n) != 2 or isinstance(n, str)]
    keep_set = set(keep)
    edges = 0
    visited = set()
    for n in keep:
        for nbr in S.neighbors(n):
            prev, cur = n, nbr
            key = (n, nbr)
            if key in visited:
                continue
            while cur not in keep_set:
                nxt = [x for x in S.neighbors(cur) if x != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            visited.add((n, nbr))
            if cur in keep_set:
                visited.add((cur, prev))
                edges += 1
    return len(keep), edges // 2 + edges % 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--floors', default="FF part 1upE,FF part 2up")
    ap.add_argument('--room-pairs', type=int, default=30)
    ap.add_argument('--queries', type=int, default=100)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    report = ["Medial-axis (skeleton) baseline vs Tesseract++ typed graph", ""]

    for stem in [s.strip() for s in args.floors.split(',')]:
        G = load_graph_json(os.path.join(RESULTS_DIR, "Json", stem,
                                         f"{stem}_post_pruning.json"))
        img = cv2.imread(os.path.join(INPUT_IMAGES_DIR, f"{stem}.png"))
        doors = [tuple(G.nodes[n]['position']) for n in G if 'door' in str(n)]
        recreated = cv2.imread(os.path.join(
            RESULTS_DIR, "Plots", "connective_plots", stem, "recreated_image.png"))
        pre_path = os.path.join(RESULTS_DIR, "Json", stem, f"{stem}_pre_pruning.json")
        outside_pos = []
        if os.path.exists(pre_path):
            with open(pre_path) as f:
                outside_pos = [tuple(n['position']) for n in json.load(f)['nodes']
                               if n.get('type') == 'outside']
        excl = semantic_outside_mask(recreated, outside_pos)
        mask = traversable_mask(img, doors, interior_only=excl is not None,
                                exclude_mask=excl)

        rooms = main_rooms(G)
        exits = exit_doors(G)
        attach = {n: tuple(G.nodes[n]['position']) for n in rooms + exits}
        S, _ = build_skeleton_graph(mask, attach)
        sk_nodes, sk_edges = contracted_size(S)

        # same pair sample as the fidelity harness
        pairs = [(r, e) for r in rooms for e in exits]
        for _ in range(args.room_pairs):
            a, b = rng.choice(len(rooms), 2, replace=False)
            pairs.append((rooms[a], rooms[b]))

        by_src = {}
        for a, b in pairs:
            by_src.setdefault(a, []).append(b)

        rows = []
        for a, targets in by_src.items():
            cost_map, _ = geodesic_from(mask, attach[a])
            try:
                d_sk_all = nx.single_source_dijkstra_path_length(S, a, weight='weight')
            except nx.NodeNotFound:
                d_sk_all = {}
            for b in targets:
                d_geo = read_distance(cost_map, attach[b])
                d_sk = d_sk_all.get(b, math.inf)
                try:
                    d_ts = nx.shortest_path_length(G, a, b, weight='weight')
                except nx.NetworkXNoPath:
                    d_ts = math.inf
                ok = all(map(math.isfinite, (d_geo, d_sk, d_ts))) and d_geo > 0
                rows.append({'floor': stem, 'src': a, 'dst': b,
                             'stretch_skeleton': d_sk / d_geo if ok else None,
                             'stretch_tesseract': d_ts / d_geo if ok else None})

        # connectivity metrics on the two graphs
        def room_metrics(H):
            comp_of = {}
            for ci, comp in enumerate(nx.connected_components(H)):
                for n in comp:
                    comp_of[n] = ci
            from collections import Counter
            sizes = Counter(comp_of.get(r) for r in rooms if r in comp_of)
            n = len(rooms)
            conn = sum(k * (k - 1) // 2 for k in sizes.values())
            comp = conn / (n * (n - 1) // 2) if n > 1 else 1.0
            ec = {comp_of.get(e) for e in exits if e in comp_of}
            er = (sum(1 for r in rooms if comp_of.get(r) in ec) / n
                  if n and exits else None)
            return comp, er

        comp_s, er_s = room_metrics(S)
        comp_t, er_t = room_metrics(G)

        def qtime(H):
            nodes = [n for n in (rooms + exits) if n in H]
            t0 = time.perf_counter()
            done = 0
            for _ in range(args.queries):
                a, b = rng.choice(len(nodes), 2, replace=False)
                try:
                    nx.shortest_path_length(H, nodes[a], nodes[b], weight='weight')
                    done += 1
                except nx.NetworkXNoPath:
                    pass
            return (time.perf_counter() - t0) / max(1, done) * 1000

        qs, qt = qtime(S), qtime(G)
        sks = [r['stretch_skeleton'] for r in rows if r['stretch_skeleton']]
        tss = [r['stretch_tesseract'] for r in rows if r['stretch_tesseract']]

        report += [
            f"{stem} ({len(sks)} scored pairs):",
            f"  {'':>12} {'nodes':>7} {'edges':>7} {'complete':>9} "
            f"{'exit_reach':>10} {'stretch mean':>13} {'p90':>7} {'query ms':>9}",
            f"  {'skeleton':>12} {sk_nodes:>7} {sk_edges:>7} {comp_s:>9.3f} "
            f"{er_s if er_s is None else round(er_s, 3)!s:>10} "
            f"{np.mean(sks):>13.3f} {np.percentile(sks, 90):>7.3f} {qs:>9.2f}",
            f"  {'tesseract++':>12} {G.number_of_nodes():>7} {G.number_of_edges():>7} "
            f"{comp_t:>9.3f} {er_t if er_t is None else round(er_t, 3)!s:>10} "
            f"{np.mean(tss):>13.3f} {np.percentile(tss, 90):>7.3f} {qt:>9.2f}",
            "",
        ]
        with open(os.path.join(OUT_DIR, f"skeleton_{stem}.csv"), 'w',
                  newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)

    report += [
        "The skeleton is granted Tesseract++'s own room/exit detections, so",
        "this isolates routing structure. Regardless of route quality, the",
        "skeleton has no typed rooms/doors/corridors/transitions: none of",
        "the downstream tasks (accessibility routing, egress semantics,",
        "automatic multi-floor merging) are expressible on it.",
    ]
    with open(os.path.join(OUT_DIR, "report.txt"), 'w') as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"Saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
