"""
evaluate_metrics.py - Run the metric layer over every processed graph

Reports completeness, exit reachability, transition reachability, and
component structure for all single-floor post-pruning graphs in
results/single_floor/Json/ and all merged multi-floor graphs in results/multi_floor/.

Usage:
    python evaluate_metrics.py

Output: printed table + results/evaluation/Metrics/all_graphs.csv
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import (BASE_PATH, INPUT_IMAGES_DIR, RESULTS_DIR,
                    MULTIFLOOR_RESULTS_DIR, EVAL_RESULTS_DIR)

import csv
import glob
import os

from utils.metrics import load_graph_json, compute_metrics, format_metrics

OUT_DIR = os.path.join(EVAL_RESULTS_DIR, "Metrics")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    print("== Single-floor graphs (post-pruning) ==")
    for d in sorted(glob.glob(os.path.join(RESULTS_DIR, "Json", "*"))):
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
    for path in sorted(glob.glob(os.path.join(MULTIFLOOR_RESULTS_DIR, "Jsons", "*",
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
