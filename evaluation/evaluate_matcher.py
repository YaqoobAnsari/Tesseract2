"""
evaluate_matcher.py - Synthetic stress test for automatic transition matching

Creates perturbed copies of a real floor (known similarity transform + jitter
+ optionally deleted shafts), runs utils/transition_matcher.py against them,
and compares with a greedy nearest-neighbour baseline. Because the transform
is known, correctness is exact: the right answer is always id -> same id,
minus deliberately deleted shafts.

Sweeps:
  translation : shift floor 2 by 0..400 px in random directions
  rotation    : rotate floor 2 by 0..30 degrees about the image centre
  scale       : rescale floor 2 by 0.7..1.4
  twin        : inject a synthetic twin stairwell at decreasing separations
                from stairs_1 on BOTH floors (scissor/fire-stair case), under
                a fixed moderate misregistration
  termination : delete each shaft from floor 2 in turn; the matcher must
                leave its floor-1 partner unmatched

Every trial adds Gaussian position jitter (default 5 px) to floor-2
transitions, simulating detection/annotation noise, so exact-coincidence
shortcuts cannot pass.

Usage:
    python evaluate_matcher.py [--image "FF part 1upE.png"] [--trials 8]

Outputs (results/evaluation/MatcherEval/):
    matcher_eval.csv, matcher_eval.png, summary.txt
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
import shutil

import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.transition_matcher import (
    load_floor, match_transitions, transition_kind, DEFAULT_REJECT_COST,
    _apply, _make_transform,
)

OUT_DIR = os.path.join(EVAL_RESULTS_DIR, "MatcherEval")
TMP_DIR = os.path.join(OUT_DIR, "_tmp")


def find_graph_json(image_name):
    stem = os.path.splitext(image_name)[0]
    for suffix in ("post_pruning", "pre_pruning", "final_graph"):
        p = os.path.join(RESULTS_DIR, "Json", stem, f"{stem}_{suffix}.json")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No graph JSON for {image_name} - run Main.py on it first")


def make_perturbed_floor(json_path, image_path, scale, rot_deg, shift,
                         jitter_px, drop_ids, rng, tag):
    """Write a transformed copy of the floor (JSON + warped image) and
    return (json_path, image_path). Transform maps original -> copy."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    th = math.radians(rot_deg)
    a = scale * math.cos(th)
    b = scale * math.sin(th)

    def T(p):
        x, y = p[0] - cx, p[1] - cy
        return (a * x - b * y + cx + shift[0], b * x + a * y + cy + shift[1])

    with open(json_path) as f:
        data = json.load(f)

    drop = set(drop_ids)
    nodes = []
    for nd in data['nodes']:
        if nd['id'] in drop:
            continue
        nd = dict(nd)
        p = T(nd.get('position') or (0.0, 0.0))
        if nd.get('type') == 'floor_transition' and jitter_px > 0:
            p = (p[0] + rng.normal(0, jitter_px), p[1] + rng.normal(0, jitter_px))
        nd['position'] = [p[0], p[1]]
        nodes.append(nd)
    edges = []
    for ed in data['edges']:
        if ed['source'] in drop or ed['target'] in drop:
            continue
        ed = dict(ed)
        if ed.get('weight') is not None:
            ed['weight'] = float(ed['weight']) * scale
        if ed.get('distance') is not None:
            ed['distance'] = float(ed['distance']) * scale
        edges.append(ed)

    M = np.array([[a, -b, cx + shift[0] - a * cx + b * cy],
                  [b,  a, cy + shift[1] - b * cx - a * cy]])
    warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderValue=(255, 255, 255))

    out_json = os.path.join(TMP_DIR, f"{tag}.json")
    out_img = os.path.join(TMP_DIR, f"{tag}.png")
    with open(out_json, 'w') as f:
        json.dump({'nodes': nodes, 'edges': edges}, f)
    cv2.imwrite(out_img, warped)
    return out_json, out_img


def inject_twin(json_path, anchor_id, offset_px, tag):
    """Write a copy of the floor JSON with a synthetic twin transition
    (same kind as anchor, same connectivity) offset_px to the right of it -
    the twin fire-stair / scissor-stair case. Returns the new JSON path."""
    with open(json_path) as f:
        data = json.load(f)
    anchor = next(n for n in data['nodes'] if n['id'] == anchor_id)
    twin_id = f"{transition_kind(anchor_id)}_99"
    twin = dict(anchor)
    twin['id'] = twin_id
    twin['position'] = [anchor['position'][0] + offset_px, anchor['position'][1]]
    data['nodes'].append(twin)
    for ed in list(data['edges']):
        if ed['source'] == anchor_id:
            data['edges'].append({**ed, 'source': twin_id})
        elif ed['target'] == anchor_id:
            data['edges'].append({**ed, 'target': twin_id})
    out = os.path.join(TMP_DIR, f"{tag}_base.json")
    with open(out, 'w') as f:
        json.dump(data, f)
    return out


def nearest_neighbour_baseline(floor1, floor2):
    """Greedy same-kind nearest neighbour on per-image-normalized
    coordinates - the naive approach. Forces a match whenever any same-kind
    candidate exists (no rejection)."""
    w1, h1 = floor1['size']
    w2, h2 = floor2['size']
    t1 = floor1['transitions']
    t2 = floor2['transitions']
    used = set()
    pairs = {}
    # Greedy in globally increasing distance order.
    cands = []
    for i, x in enumerate(t1):
        for j, y in enumerate(t2):
            if x['kind'] != y['kind']:
                continue
            d = math.hypot(x['pos'][0] / w1 - y['pos'][0] / w2,
                           x['pos'][1] / h1 - y['pos'][1] / h2)
            cands.append((d, i, j))
    for d, i, j in sorted(cands):
        if i in pairs or j in used:
            continue
        pairs[i] = j
        used.add(j)
    return {t1[i]['id']: t2[j]['id'] for i, j in pairs.items()}


def score(pred_pairs, floor1_ids, dropped):
    """Accuracy over all floor-1 transitions: correct = paired with its own
    id, or left unmatched when its partner was deleted."""
    correct = 0
    for tid in floor1_ids:
        want = None if tid in dropped else tid
        if pred_pairs.get(tid, None) == want:
            correct += 1
    return correct / len(floor1_ids)


def registration_error(reg, scale, rot_deg, shift, size):
    """Mean displacement (px) between the estimated floor2->floor1 transform
    and the inverse of the true perturbation, over the canvas corners and
    centre."""
    w, h = size
    c = complex(w / 2.0, h / 2.0)
    a_t = scale * np.exp(1j * math.radians(rot_deg))
    b_t = c + complex(*shift) - a_t * c
    T_true_inv = _make_transform(1.0 / a_t, -b_t / a_t, False)
    a_e = reg['scale'] * np.exp(1j * math.radians(reg['rotation_deg']))
    T_est = _make_transform(a_e, complex(*reg['translation_px']),
                            reg.get('mirrored', False))
    pts = np.array([[0, 0], [w, 0], [0, h], [w, h], [w / 2, h / 2]], float)
    return float(np.mean(np.linalg.norm(
        _apply(T_est, pts) - _apply(T_true_inv, pts), axis=1)))


def run_trial(floor1, json_path, image_path, scale, rot, shift, jitter,
              drop_ids, rng, tag, reject_cost):
    pj, pi = make_perturbed_floor(json_path, image_path, scale, rot, shift,
                                  jitter, drop_ids, rng, tag)
    floor2 = load_floor(pj, pi)

    res = match_transitions(floor1, floor2, reject_cost=reject_cost,
                            verbose=False)
    matcher_pairs = {m['floor1_node']: m['floor2_node'] for m in res['matches']}
    nn_pairs = nearest_neighbour_baseline(floor1, floor2)

    ids = [t['id'] for t in floor1['transitions']]
    reg = res['registration']
    reg_err = registration_error(reg, scale, rot, shift, floor1['size'])
    return (score(matcher_pairs, ids, set(drop_ids)),
            score(nn_pairs, ids, set(drop_ids)),
            reg, reg_err)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image', default="FF part 1upE.png")
    ap.add_argument('--trials', type=int, default=8)
    ap.add_argument('--jitter', type=float, default=5.0)
    ap.add_argument('--reject-cost', type=float, default=DEFAULT_REJECT_COST)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    os.makedirs(TMP_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    json_path = find_graph_json(args.image)
    image_path = os.path.join(INPUT_IMAGES_DIR, args.image)
    floor1 = load_floor(json_path, image_path)
    ids = [t['id'] for t in floor1['transitions']]
    print(f"Base floor: {args.image}  transitions: {ids}")
    print(f"Jitter: {args.jitter}px  trials/level: {args.trials}  "
          f"reject cost: {args.reject_cost}")

    rows = []

    sweeps = {
        'translation_px': [0, 50, 100, 150, 200, 300, 400],
        'rotation_deg': [0, 2, 5, 10, 15, 20, 30],
        'scale': [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4],
    }
    for sweep, levels in sweeps.items():
        print(f"\nSweep: {sweep}")
        for level in levels:
            for k in range(args.trials):
                scale, rot, shift = 1.0, 0.0, (0.0, 0.0)
                if sweep == 'translation_px':
                    ang = rng.uniform(0, 2 * math.pi)
                    shift = (level * math.cos(ang), level * math.sin(ang))
                elif sweep == 'rotation_deg':
                    rot = level if k % 2 == 0 else -level
                else:
                    scale = level
                acc_m, acc_nn, reg, reg_err = run_trial(
                    floor1, json_path, image_path, scale, rot, shift,
                    args.jitter, [], rng, f"{sweep}_{level}_{k}",
                    args.reject_cost)
                rows.append({'sweep': sweep, 'level': level, 'trial': k,
                             'matcher_acc': acc_m, 'nn_acc': acc_nn,
                             'registration_residual_px': reg['residual_px'],
                             'reg_error_px': reg_err})
            got = [r for r in rows if r['sweep'] == sweep and r['level'] == level]
            print(f"  {level:>6}: matcher "
                  f"{np.mean([r['matcher_acc'] for r in got]):.3f}  "
                  f"nn {np.mean([r['nn_acc'] for r in got]):.3f}")

    # Twin-shaft test: a synthetic second stairwell near stairs_1 on both
    # floors, under a fixed moderate misregistration (rot 10 deg, shift 100px,
    # scale 1.05). Nearest neighbour confuses the twins once the
    # misregistration displacement rivals their separation; registration +
    # assignment should hold until separation approaches the jitter level.
    print("\nSweep: twin shaft separation (synthetic scissor stair)")
    if any(t['id'] == 'stairs_1' for t in floor1['transitions']):
        for sep in [800, 400, 200, 100, 50, 25]:
            for k in range(args.trials):
                tag = f"twin_{sep}_{k}"
                base_json = inject_twin(json_path, 'stairs_1', sep, tag)
                floor1_twin = load_floor(base_json, image_path)
                ang = rng.uniform(0, 2 * math.pi)
                acc_m, acc_nn, reg, reg_err = run_trial(
                    floor1_twin, base_json, image_path,
                    1.05, 10.0, (100 * math.cos(ang), 100 * math.sin(ang)),
                    args.jitter, [], rng, tag, args.reject_cost)
                rows.append({'sweep': 'twin_separation_px', 'level': sep,
                             'trial': k, 'matcher_acc': acc_m,
                             'nn_acc': acc_nn,
                             'registration_residual_px': reg['residual_px'],
                             'reg_error_px': reg_err})
            got = [r for r in rows if r['sweep'] == 'twin_separation_px'
                   and r['level'] == sep]
            print(f"  {sep:>6}: matcher "
                  f"{np.mean([r['matcher_acc'] for r in got]):.3f}  "
                  f"nn {np.mean([r['nn_acc'] for r in got]):.3f}")
    else:
        print("  (skipped - base floor has no stairs_1)")

    # Termination test: delete each shaft in turn, under a moderate combined
    # perturbation (the realistic case: different export + missing shaft).
    print("\nSweep: shaft termination (delete one transition from floor 2)")
    term_rows = []
    for dropped in ids:
        for k in range(args.trials):
            ang = rng.uniform(0, 2 * math.pi)
            shift = (100 * math.cos(ang), 100 * math.sin(ang))
            acc_m, acc_nn, _, reg_err = run_trial(
                floor1, json_path, image_path,
                rng.uniform(0.9, 1.1), rng.uniform(-10, 10), shift,
                args.jitter, [dropped], rng, f"term_{dropped}_{k}",
                args.reject_cost)
            term_rows.append({'sweep': 'termination', 'level': dropped,
                              'trial': k, 'matcher_acc': acc_m,
                              'nn_acc': acc_nn,
                              'registration_residual_px': None,
                              'reg_error_px': reg_err})
        got = [r for r in term_rows if r['level'] == dropped]
        print(f"  drop {dropped:>12}: matcher "
              f"{np.mean([r['matcher_acc'] for r in got]):.3f}  "
              f"nn {np.mean([r['nn_acc'] for r in got]):.3f}")
    rows.extend(term_rows)

    # ---- outputs ----
    csv_path = os.path.join(OUT_DIR, "matcher_eval.csv")
    with open(csv_path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, sweep in zip(axes, ['translation_px', 'rotation_deg', 'scale',
                                'twin_separation_px', 'termination']):
        got = [r for r in rows if r['sweep'] == sweep]
        levels = sorted({r['level'] for r in got},
                        key=lambda v: (isinstance(v, str), v))
        for method, color, label in [('matcher_acc', '#0072B2',
                                      'Assignment + registration'),
                                     ('nn_acc', '#D55E00',
                                      'Nearest neighbour')]:
            means = [np.mean([r[method] for r in got if r['level'] == lv])
                     for lv in levels]
            stds = [np.std([r[method] for r in got if r['level'] == lv])
                    for lv in levels]
            x = range(len(levels))
            ax.errorbar(x, means, yerr=stds, marker='o', color=color,
                        label=label, capsize=3)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(lv) for lv in levels], rotation=45 if
                           sweep == 'termination' else 0, fontsize=8)
        ax.set_title(sweep.replace('_', ' '))
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('correspondence accuracy')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Transition matching under synthetic perturbation - "
                 f"{args.image} (jitter {args.jitter}px, "
                 f"{args.trials} trials/level)")
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, "matcher_eval.png")
    fig.savefig(plot_path, dpi=150)

    # Registration recovery error (task B deliverable): estimated transform
    # vs known ground-truth perturbation.
    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
    for ax, sweep in zip(axes2, ['translation_px', 'rotation_deg', 'scale']):
        got = [r for r in rows if r['sweep'] == sweep]
        levels = sorted({r['level'] for r in got})
        med = [np.median([r['reg_error_px'] for r in got if r['level'] == lv])
               for lv in levels]
        q1 = [np.percentile([r['reg_error_px'] for r in got if r['level'] == lv], 25)
              for lv in levels]
        q3 = [np.percentile([r['reg_error_px'] for r in got if r['level'] == lv], 75)
              for lv in levels]
        x = range(len(levels))
        ax.plot(x, med, marker='o', color='#0072B2')
        ax.fill_between(x, q1, q3, color='#0072B2', alpha=0.2)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(lv) for lv in levels], fontsize=8)
        ax.set_title(sweep.replace('_', ' '))
        ax.set_ylabel('registration error (px, corner displacement)')
        ax.set_yscale('symlog', linthresh=1.0)
        ax.grid(alpha=0.3)
    fig2.suptitle("Registration recovery error vs known perturbation "
                  f"(jitter {args.jitter}px)")
    fig2.tight_layout()
    reg_plot_path = os.path.join(OUT_DIR, "registration_error.png")
    fig2.savefig(reg_plot_path, dpi=150)

    summary_path = os.path.join(OUT_DIR, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Transition matcher synthetic evaluation - {args.image}\n")
        f.write(f"Transitions: {ids}\n")
        f.write(f"Jitter: {args.jitter}px, trials/level: {args.trials}, "
                f"reject cost: {args.reject_cost}\n\n")
        for sweep in ['translation_px', 'rotation_deg', 'scale',
                      'twin_separation_px', 'termination']:
            got = [r for r in rows if r['sweep'] == sweep]
            errs = [r['reg_error_px'] for r in got
                    if r.get('reg_error_px') is not None]
            f.write(f"{sweep}: matcher "
                    f"{np.mean([r['matcher_acc'] for r in got]):.3f}  "
                    f"nearest-neighbour "
                    f"{np.mean([r['nn_acc'] for r in got]):.3f}  "
                    f"median reg error {np.median(errs):.1f}px\n")
        f.write("\nNOTE: floor 2 is a synthetically perturbed copy of floor 1 "
                "(known transform + jitter + deletions). This measures "
                "robustness of the matcher, not accuracy on genuinely "
                "different floor drawings. Further leakage caveats: canvas "
                "size and node ids are shared between the floors, and "
                "graph-context features are near-identical by construction, "
                "so the context term is not exercised. With only two "
                "same-kind shafts on this floor (1637px apart), the plain "
                "translation/rotation/scale sweeps cannot separate the "
                "matcher from the nearest-neighbour baseline; the twin and "
                "termination sweeps carry the comparison.\n")

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print(f"\nSaved: {csv_path}\n       {plot_path}\n       "
          f"{reg_plot_path}\n       {summary_path}")


if __name__ == '__main__':
    main()
