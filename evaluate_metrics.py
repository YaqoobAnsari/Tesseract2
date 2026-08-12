"""
evaluate_metrics.py - Run the metric layer over every processed graph

Reports completeness, exit reachability, transition reachability, and
component structure for all single-floor post-pruning graphs in
Results/Json/ and all merged multi-floor graphs in Multifloor_Results/.

Usage:
    python evaluate_metrics.py

Output: printed table + Multifloor_Results/Metrics/all_graphs.csv
"""

import csv
import glob
import os

from utils.metrics import load_graph_json, compute_metrics, format_metrics

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_PATH, "Multifloor_Results", "Metrics")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    print("== Single-floor graphs (post-pruning) ==")
    for d in sorted(glob.glob(os.path.join(BASE_PATH, "Results", "Json", "*"))):
        stem = os.path.basename(d)
        path = os.path.join(d, f"{stem}_post_pruning.json")
        if not os.path.exists(path):
            continue
        m = compute_metrics(load_graph_json(path))
        m['graph'] = stem
        m['kind'] = 'single-floor'
        rows.append(m)
        print(format_metrics(m, stem[:18]))

    print("\n== Merged multi-floor graphs ==")
    for path in sorted(glob.glob(os.path.join(
            BASE_PATH, "Multifloor_Results", "Jsons", "*",
            "merged_multi_floor_graph.json"))):
        seq = os.path.basename(os.path.dirname(path))
        if seq.endswith('.manual-backup'):
            continue
        m = compute_metrics(load_graph_json(path))
        m['graph'] = seq
        m['kind'] = 'multi-floor'
        rows.append(m)
        print(format_metrics(m, seq[:18]))

    csv_path = os.path.join(OUT_DIR, "all_graphs.csv")
    with open(csv_path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nSaved: {csv_path}")


if __name__ == '__main__':
    main()
