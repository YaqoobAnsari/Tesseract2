"""
evaluate_text_accuracy.py - Text label detection/recognition accuracy (R1)

Compares the interpreter's detected labels against a hand-enumerated ground
truth of every label visible on the drawing.

GROUND TRUTH PROVENANCE: enumerated by visual inspection of the input
images (by Claude, 2026-08-12); it should be verified by an author before
the numbers are quoted in the manuscript.

Metrics:
  detection rate    fraction of ground-truth labels matched by any detected
                    box (by text equality or near-miss)
  recognition rate  fraction of ground-truth labels whose text was read
                    exactly (case-insensitive)

Usage:
    python evaluate_text_accuracy.py

Output: results/evaluation/TextAccuracy/report.txt
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import (BASE_PATH, INPUT_IMAGES_DIR, RESULTS_DIR,
                    MULTIFLOOR_RESULTS_DIR, EVAL_RESULTS_DIR)

import os
import re

OUT_DIR = os.path.join(EVAL_RESULTS_DIR, "TextAccuracy")

# Hand-enumerated visible labels per floor (see provenance note above).
GROUND_TRUTH = {
    "FF part 1upE": (
        [str(n) for n in ([1001] + list(range(1004, 1012))
                          + list(range(1013, 1033))
                          + list(range(1036, 1041)))]
        + ['hall'] * 5 + ['na'] + ['stairs'] * 2 + ['elev']
    ),
    "GF part 1upE": (
        ['0006', '0007', '0008'] + ['hall'] * 2 + ['na'] + ['stairs'] * 2
        + ['elev']
    ),
}


def detected_labels(stem):
    path = os.path.join(RESULTS_DIR, "Plots", "interpreter_detect",
                        stem, "room_labels.txt")
    labels = []
    with open(path) as f:
        for line in f:
            m = re.search(r"Text:\s*(\S+),\s*Confidence:\s*([\d.]+)", line)
            if m:
                labels.append((m.group(1).lower(), float(m.group(2))))
    return labels


def near_miss(gt, det):
    """A detection that clearly corresponds to the GT label but was
    misread (e.g. truncated digits): one is a substring of the other."""
    return gt != det and (gt in det or det in gt) and min(len(gt), len(det)) >= 3


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = ["Text label detection/recognition accuracy",
             "(ground truth hand-enumerated from the drawings - see script "
             "header; verify before quoting)", ""]

    for stem, gt in GROUND_TRUTH.items():
        det = detected_labels(stem)
        det_texts = [t for t, _ in det]
        gt_pool = list(gt)
        exact, miss_read, undetected = 0, [], []
        unmatched_det = list(det_texts)
        for g in gt:
            if g in unmatched_det:
                exact += 1
                unmatched_det.remove(g)
        # near-miss pass for remaining GT labels
        remaining_gt = list(gt)
        for t in det_texts:
            if t in remaining_gt:
                remaining_gt.remove(t)
        for g in list(remaining_gt):
            nm = next((t for t in unmatched_det if near_miss(g, t)), None)
            if nm is not None:
                miss_read.append((g, nm))
                unmatched_det.remove(nm)
                remaining_gt.remove(g)
        undetected = remaining_gt
        spurious = unmatched_det

        n = len(gt)
        detection = (exact + len(miss_read)) / n
        recognition = exact / n
        lines.append(f"{stem}: {n} ground-truth labels, {len(det)} detections")
        lines.append(f"  detection rate:   {detection:.3f} "
                     f"({exact + len(miss_read)}/{n})")
        lines.append(f"  recognition rate: {recognition:.3f} ({exact}/{n})")
        for g, t in miss_read:
            lines.append(f"  misread: '{g}' -> '{t}'")
        for g in undetected:
            lines.append(f"  undetected: '{g}'")
        for t in spurious:
            lines.append(f"  spurious detection: '{t}'")
        conf = [c for _, c in det]
        lines.append(f"  confidence: min {min(conf):.3f}, "
                     f"mean {sum(conf)/len(conf):.3f}")
        lines.append("")

    lines.append("Failure consequence: misread numeric labels still yield "
                 "room nodes (any numeric token is treated as a room "
                 "identifier), so connectivity is preserved and only the "
                 "room NAME is wrong. A label undetected entirely leaves "
                 "its region without a semantic seed: the region is then "
                 "not instantiated as a room/corridor (cf. the unlabeled "
                 "'FF part 1up' variant, which yields zero transitions "
                 "because 'Stairs'/'Elev' labels are absent).")

    out = os.path.join(OUT_DIR, "report.txt")
    with open(out, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
