"""
evaluate_funneling.py - Ablation for the room-family funneling construction

Builds three variants of an already-processed floor graph and compares them:

  funneling    : the post-pruning graph exactly as the pipeline builds it
                 (main-rooted intra-room tree + closest-node door shortcut)
  full         : every room-family node additionally gets a direct edge to
                 every door its family touches (dense subnode-to-door)
  one-per-room : subnodes removed entirely; rooms are single nodes (doors
                 that lose their only family edge are re-anchored to the
                 main room)

Metrics per variant:
  - graph size (nodes, edges)
  - completeness: fraction of room family nodes that can reach every door
    their family touches
  - realized route length from every subnode position to its nearest exit
    door (for one-per-room: Euclidean hop to the main room + graph path)
  - stretch vs the dense 'full' variant's path length (full is the
    best-paths reference)
  - mean single-query Dijkstra time

Also verifies the linear-edge property behind the funneling proposition:
per family, intra-room edge count vs subnode count.

Usage:
    python evaluate_funneling.py [--image "FF part 1upE.png"]

Outputs (Multifloor_Results/FunnelingAblation/):
    ablation_<image>.txt  (summary table + per-family edge counts)
    ablation_<image>.csv  (per-query rows)
"""

import argparse
import csv
import json
import math
import os
import time

import networkx as nx
import numpy as np

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_PATH, "Results")
OUT_DIR = os.path.join(BASE_PATH, "Multifloor_Results", "FunnelingAblation")

DOOR_PREFIXES = ("r2c_door_", "r2r_door_", "exit_door_", "c2c_door_",
                 "entry_door_", "door_")


def load_graph(image_name):
    stem = os.path.splitext(image_name)[0]
    path = os.path.join(RESULTS_DIR, "Json", stem, f"{stem}_post_pruning.json")
    with open(path) as f:
        data = json.load(f)
    G = nx.Graph()
    for nd in data['nodes']:
        G.add_node(nd['id'], **{k: v for k, v in nd.items() if k != 'id'})
    for ed in data['edges']:
        G.add_edge(ed['source'], ed['target'],
                   weight=float(ed.get('weight') or 1.0))
    return G


def pos(G, n):
    return np.array(G.nodes[n]['position'], dtype=float)


def is_subnode(G, n):
    d = G.nodes[n]
    return d.get('type') == 'room' and (d.get('is_subnode') or '_subnode_' in str(n))


def family_of(G):
    """{main_room_id: [subnode ids]}"""
    fams = {}
    for n in G.nodes:
        if G.nodes[n].get('type') == 'room' and not is_subnode(G, n):
            fams[n] = []
    for n in G.nodes:
        if is_subnode(G, n):
            parent = G.nodes[n].get('parent_room_id') or str(n).split('_subnode_')[0]
            if parent in fams:
                fams[parent].append(n)
    return fams


def family_doors(G, main, subs):
    doors = set()
    for n in [main] + subs:
        for nbr in G.neighbors(n):
            if str(nbr).startswith(DOOR_PREFIXES):
                doors.add(nbr)
    return sorted(doors)


def build_variants(G):
    fams = family_of(G)
    variants = {'funneling': G.copy()}

    full = G.copy()
    for main, subs in fams.items():
        doors = family_doors(G, main, subs)
        for n in [main] + subs:
            for d in doors:
                if not full.has_edge(n, d):
                    w = float(np.linalg.norm(pos(G, n) - pos(G, d)))
                    full.add_edge(n, d, weight=w)
    variants['full'] = full

    one = G.copy()
    for main, subs in fams.items():
        doors = family_doors(G, main, subs)
        one.remove_nodes_from(subs)
        for d in doors:
            if d in one and not any(True for _ in one.neighbors(d)):
                w = float(np.linalg.norm(pos(G, main) - pos(G, d)))
                one.add_edge(main, d, weight=w)
        # Doors that still exist but lost their family-side edge entirely
        for d in doors:
            if d in one and not one.has_edge(main, d) and not any(
                    one.nodes[x].get('type') == 'room'
                    for x in one.neighbors(d)):
                w = float(np.linalg.norm(pos(G, main) - pos(G, d)))
                one.add_edge(main, d, weight=w)
    variants['one-per-room'] = one
    return variants, fams


def completeness(V, fams):
    """Fraction of (family node, family door) pairs that are connected."""
    ok = total = 0
    for main, subs in fams.items():
        nodes = [n for n in [main] + subs if n in V]
        doors = [d for d in family_doors(V, main, [s for s in subs if s in V])
                 if d in V]
        for n in nodes:
            for d in doors:
                total += 1
                if nx.has_path(V, n, d):
                    ok += 1
    return ok / total if total else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image', default="FF part 1upE.png")
    ap.add_argument('--queries', type=int, default=100)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    G = load_graph(args.image)
    variants, fams = build_variants(G)
    exits = [n for n in G if str(n).startswith('exit_door_')]
    subnodes = [n for n in G if is_subnode(G, n)]
    if not exits:
        raise SystemExit("No exit doors in this graph - pick another floor")

    stem = os.path.splitext(args.image)[0]
    lines = [f"Funneling ablation - {args.image}",
             f"families: {len(fams)}  subnodes: {len(subnodes)}  "
             f"exits: {len(exits)}", ""]

    # Linear-edge property evidence (on the as-built graph).
    intra_counts = []
    for main, subs in fams.items():
        fam_set = set([main] + subs)
        e = sum(1 for u, v in G.edges
                if u in fam_set and v in fam_set)
        intra_counts.append((main, len(subs), e))
    total_sub = sum(c[1] for c in intra_counts)
    total_edge = sum(c[2] for c in intra_counts)
    lines.append(f"Intra-family room-room edges: {total_edge} for "
                 f"{total_sub} subnodes across {len(fams)} families "
                 f"(linear-property ratio {total_edge / max(1, total_sub):.2f}; "
                 f"fully-connected would need "
                 f"{sum(((c[1] + 1) * c[1]) // 2 for c in intra_counts)})")
    lines.append("")

    # Reference path lengths come from 'full' (densest = best possible).
    ref_len = {}

    rows = []
    header = (f"{'variant':>14} {'nodes':>7} {'edges':>7} {'complete':>9} "
              f"{'mean route px':>14} {'p90 route px':>13} "
              f"{'stretch':>8} {'query ms':>9}")
    lines.append(header)

    for name in ['full', 'funneling', 'one-per-room']:
        V = variants[name]
        comp = completeness(V, fams)

        route_lens = []
        for s in subnodes:
            targets = [e for e in exits if e in V]
            if name == 'one-per-room':
                parent = G.nodes[s].get('parent_room_id') or str(s).split('_subnode_')[0]
                if parent not in V:
                    continue
                entry_cost = float(np.linalg.norm(pos(G, s) - pos(G, parent)))
                src = parent
            else:
                if s not in V:
                    continue
                entry_cost, src = 0.0, s
            best = math.inf
            try:
                dists = nx.single_source_dijkstra_path_length(V, src,
                                                              weight='weight')
                for e in targets:
                    if e in dists:
                        best = min(best, entry_cost + dists[e])
            except nx.NetworkXError:
                pass
            if math.isfinite(best):
                route_lens.append(best)
                if name == 'full':
                    ref_len[s] = best
                rows.append({'variant': name, 'subnode': s,
                             'route_px': best,
                             'stretch_vs_full': (best / ref_len[s]
                                                 if s in ref_len and ref_len[s] > 0
                                                 else None)})

        stretches = [r['stretch_vs_full'] for r in rows
                     if r['variant'] == name and r['stretch_vs_full']]

        # Query time: random node-pair Dijkstra.
        nodes = list(V.nodes)
        t0 = time.perf_counter()
        done = 0
        for _ in range(args.queries):
            a, b = rng.choice(len(nodes), 2, replace=False)
            try:
                nx.shortest_path_length(V, nodes[a], nodes[b], weight='weight')
                done += 1
            except nx.NetworkXNoPath:
                pass
        q_ms = (time.perf_counter() - t0) / max(1, done) * 1000

        lines.append(f"{name:>14} {V.number_of_nodes():>7} "
                     f"{V.number_of_edges():>7} {comp:>9.3f} "
                     f"{np.mean(route_lens):>14.1f} "
                     f"{np.percentile(route_lens, 90):>13.1f} "
                     f"{np.mean(stretches) if stretches else 1.0:>8.3f} "
                     f"{q_ms:>9.2f}")

    txt_path = os.path.join(OUT_DIR, f"ablation_{stem}.txt")
    with open(txt_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
        f.write("\nPer-family intra-room edges (family, subnodes, edges):\n")
        for main, ns, e in sorted(intra_counts, key=lambda c: -c[1])[:15]:
            f.write(f"  {main:>10}  n={ns:>3}  edges={e:>3}\n")

    csv_path = os.path.join(OUT_DIR, f"ablation_{stem}.csv")
    with open(csv_path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['variant', 'subnode', 'route_px',
                                           'stretch_vs_full'])
        wr.writeheader()
        wr.writerows(rows)

    print("\n".join(lines))
    print(f"\nSaved: {txt_path}\n       {csv_path}")


if __name__ == '__main__':
    main()
