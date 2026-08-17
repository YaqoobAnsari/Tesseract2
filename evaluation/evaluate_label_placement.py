"""
evaluate_label_placement.py - Is the label position a reasonable main node? (R1)

R1 asks why the text label's position serves as the room's main node when
"the label could be arbitrarily placed in principle." This measures how
arbitrary it actually is: for every main room, the offset between the label
position (used as the main node) and the geometric centroid of the room's
flood-filled region, absolute (px) and relative to the room's equivalent
radius.

Usage:
    python evaluation/evaluate_label_placement.py

Output: results/evaluation/LabelPlacement/report.txt (+ CSV)
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RESULTS_DIR, EVAL_RESULTS_DIR

import csv
import glob
import json
import math
import os

import numpy as np

OUT_DIR = os.path.join(EVAL_RESULTS_DIR, "LabelPlacement")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for d in sorted(glob.glob(os.path.join(RESULTS_DIR, "Json", "*"))):
        stem = os.path.basename(d)
        path = os.path.join(d, f"{stem}_post_pruning.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for nd in data['nodes']:
            if (nd.get('type') == 'room' and '_subnode_' not in nd['id']
                    and nd.get('room_centroid_xy') and nd.get('room_eq_radius')):
                lx, ly = nd['position']
                cx, cy = nd['room_centroid_xy']
                off = math.hypot(lx - cx, ly - cy)
                rows.append({
                    'floor': stem, 'room': nd['id'],
                    'offset_px': off,
                    'room_eq_radius_px': nd['room_eq_radius'],
                    'offset_over_radius': off / nd['room_eq_radius'],
                    'inside_inradius': off <= nd.get('room_inradius', 0),
                })

    offs = np.array([r['offset_px'] for r in rows])
    rel = np.array([r['offset_over_radius'] for r in rows])
    inside = sum(1 for r in rows if r['inside_inradius'])

    lines = [
        "Label position vs geometric room centroid (main-node placement)",
        f"Rooms measured: {len(rows)} across "
        f"{len({r['floor'] for r in rows})} processed floors",
        "",
        f"Offset (px):            mean {offs.mean():.1f}, median {np.median(offs):.1f}, "
        f"p90 {np.percentile(offs, 90):.1f}, max {offs.max():.1f}",
        f"Offset / eq. radius:    mean {rel.mean():.2f}, median {np.median(rel):.2f}, "
        f"p90 {np.percentile(rel, 90):.2f}, max {rel.max():.2f}",
        f"Label within the room's inscribed circle of the centroid: "
        f"{inside}/{len(rows)} ({inside/len(rows):.0%})",
        "",
        "Reading: offset/radius < 1 means the label sits within one",
        "equivalent-radius of the centroid - i.e. architectural labels are",
        "placed centrally in practice, so using them as main-node positions",
        "is empirically sound. Rooms where the ratio exceeds 1 are the",
        "counter-examples worth inspecting (listed below).",
        "",
        "Worst offsets:",
    ]
    for r in sorted(rows, key=lambda r: -r['offset_over_radius'])[:8]:
        lines.append(f"  {r['floor']:>14} {r['room']:>9}  "
                     f"offset {r['offset_px']:.0f}px  "
                     f"ratio {r['offset_over_radius']:.2f}")

    with open(os.path.join(OUT_DIR, "report.txt"), 'w') as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT_DIR, "label_placement.csv"), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("\n".join(lines))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
