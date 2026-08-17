"""
evaluate_route_fidelity.py - Graph routes vs free-space geodesics (R1/R2-W7/R3-5)

Reviewers asked for validation against "manually verified navigable paths"
and questioned the Euclidean-distance fidelity proxy. This harness supplies
an objective reference: the free-space geodesic - the true shortest
walkable path through the drawing's non-wall pixels (doorways opened at
detected door positions). For sampled (room, target) pairs it reports

  stretch     = d_graph / d_geodesic   (how much longer graph routes are
                than the truly shortest walkable path; 1.0 = perfect)
  euclid gap  = d_geodesic / d_euclid  (how far straight-line distance is
                from walkable distance - why Euclidean is a weak reference)

and renders verification overlays (graph route + geodesic drawn on the
floorplan) so an author can visually confirm the reference paths.

Usage:
    python evaluation/evaluate_route_fidelity.py [--floors "FF part 1upE,FF part 2up"]

Outputs (results/evaluation/RouteFidelity/):
    fidelity_<floor>.csv, report.txt, overlays/*.png
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RESULTS_DIR, INPUT_IMAGES_DIR, EVAL_RESULTS_DIR

import argparse
import csv
import json
import math
import os

import cv2
import networkx as nx
import numpy as np

from utils.freespace import (traversable_mask, geodesic_from, read_distance,
                             semantic_outside_mask, DS)
from utils.metrics import load_graph_json, main_rooms, exit_doors

OUT_DIR = os.path.join(EVAL_RESULTS_DIR, "RouteFidelity")


def load_floor(stem):
    G = load_graph_json(os.path.join(RESULTS_DIR, "Json", stem,
                                     f"{stem}_post_pruning.json"))
    img = cv2.imread(os.path.join(INPUT_IMAGES_DIR, f"{stem}.png"))
    return G, img


def pos(G, n):
    return tuple(G.nodes[n]['position'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--floors', default="FF part 1upE,FF part 2up")
    ap.add_argument('--room-pairs', type=int, default=30)
    ap.add_argument('--overlays', type=int, default=6)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    os.makedirs(os.path.join(OUT_DIR, "overlays"), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    report = ["Route fidelity: graph routes vs free-space geodesics", ""]
    all_rows = {}

    for stem in [s.strip() for s in args.floors.split(',')]:
        G, img = load_floor(stem)
        doors = [pos(G, n) for n in G if 'door' in str(n)]
        # Interior-only reference: exclude the pipeline's semantic 'outside'
        # class so geodesics cannot shortcut around the building exterior.
        recreated = cv2.imread(os.path.join(
            RESULTS_DIR, "Plots", "connective_plots", stem, "recreated_image.png"))
        outside_pos = []
        pre_path = os.path.join(RESULTS_DIR, "Json", stem, f"{stem}_pre_pruning.json")
        if os.path.exists(pre_path):
            with open(pre_path) as f:
                pre = json.load(f)
            outside_pos = [tuple(n['position']) for n in pre['nodes']
                           if n.get('type') == 'outside']
        excl = semantic_outside_mask(recreated, outside_pos)
        mask_full = traversable_mask(img, doors)
        mask_int = traversable_mask(img, doors, interior_only=excl is not None,
                                    exclude_mask=excl)
        rooms = main_rooms(G)
        exits = exit_doors(G)

        # target sets: every exit + sampled peer rooms
        pairs = [(r, e) for r in rooms for e in exits]
        for _ in range(args.room_pairs):
            a, b = rng.choice(len(rooms), 2, replace=False)
            pairs.append((rooms[a], rooms[b]))

        # geodesics: one cost map per unique source
        rows = []
        overlay_budget = args.overlays
        by_src = {}
        for a, b in pairs:
            by_src.setdefault(a, []).append(b)
        for a, targets in by_src.items():
            cost_full, mcp = geodesic_from(mask_full, pos(G, a))
            cost_int, _ = geodesic_from(mask_int, pos(G, a))
            for b in targets:
                d_full = read_distance(cost_full, pos(G, b))
                d_int = read_distance(cost_int, pos(G, b))
                try:
                    d_graph = nx.shortest_path_length(G, a, b, weight='weight')
                except nx.NetworkXNoPath:
                    d_graph = math.inf
                d_euc = math.dist(pos(G, a), pos(G, b))
                ok_f = math.isfinite(d_full) and math.isfinite(d_graph) and d_full > 0
                ok_i = math.isfinite(d_int) and math.isfinite(d_graph) and d_int > 0
                rows.append({
                    'floor': stem, 'src': a, 'dst': b,
                    'd_graph_px': d_graph if math.isfinite(d_graph) else None,
                    'd_geo_full_px': d_full if math.isfinite(d_full) else None,
                    'd_geo_interior_px': d_int if math.isfinite(d_int) else None,
                    'd_euclid_px': d_euc,
                    'stretch_full': (d_graph / d_full) if ok_f else None,
                    'stretch_interior': (d_graph / d_int) if ok_i else None,
                    'geo_over_euclid': (d_full / d_euc) if ok_f and d_euc > 0 else None,
                })
                d_geo = d_full  # for the overlay gate below
                ok = ok_f
                # verification overlay
                if ok and overlay_budget > 0 and rng.random() < 0.15:
                    overlay_budget -= 1
                    vis = img.copy()
                    tb = mcp.traceback((int(pos(G, b)[1]) // DS,
                                        int(pos(G, b)[0]) // DS))
                    for (r_, c_) in tb:
                        cv2.circle(vis, (c_ * DS, r_ * DS), 2, (180, 90, 0), -1)
                    sp = nx.shortest_path(G, a, b, weight='weight')
                    pts = np.array([[pos(G, n)] for n in sp], dtype=np.int32)
                    cv2.polylines(vis, [pts], False, (0, 80, 220), 3)
                    for n, col in ((a, (0, 160, 0)), (b, (0, 0, 200))):
                        cv2.circle(vis, tuple(int(v) for v in pos(G, n)), 10, col, -1)
                    fn = os.path.join(OUT_DIR, "overlays",
                                      f"{stem}_{a}_to_{b}.png".replace(" ", "_"))
                    cv2.imwrite(fn, vis)
        all_rows[stem] = rows

        stf = [r['stretch_full'] for r in rows if r['stretch_full']]
        sti = [r['stretch_interior'] for r in rows if r['stretch_interior']]
        ge = [r['geo_over_euclid'] for r in rows if r['geo_over_euclid']]
        report += [
            f"{stem}: {len(rows)} pairs "
            f"({len(stf)} scored vs full free space, {len(sti)} vs interior-only)",
            f"  stretch vs FULL free space (permissive lower bound):     "
            f"mean {np.mean(stf):.3f}, median {np.median(stf):.3f}, "
            f"p90 {np.percentile(stf, 90):.3f}",
            f"  stretch vs INTERIOR-ONLY (conservative reference):       "
            f"mean {np.mean(sti):.3f}, median {np.median(sti):.3f}, "
            f"p90 {np.percentile(sti, 90):.3f}",
            f"  geodesic/euclid: mean {np.mean(ge):.3f}, "
            f"p90 {np.percentile(ge, 90):.3f}, max {np.max(ge):.3f}",
            "",
        ]
        with open(os.path.join(OUT_DIR, f"fidelity_{stem}.csv"), 'w',
                  newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)

    report += [
        "Reading: the two references bracket the truth. The FULL free-space",
        "geodesic is permissive (may shortcut through semantically outdoor",
        "space); the INTERIOR-ONLY geodesic is conservative (excludes the",
        "pipeline's semantic outside class, and cannot score pairs whose",
        "only connection runs outdoors). Graph route fidelity lies between",
        "the two stretch numbers. 'geodesic/euclid' well above 1 shows why",
        "straight-line Euclidean is a weak reference (walls force real",
        "detours). Overlays under overlays/ show geodesic (orange dots) vs",
        "graph route (blue line) for author verification.",
    ]
    with open(os.path.join(OUT_DIR, "report.txt"), 'w') as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"Saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
