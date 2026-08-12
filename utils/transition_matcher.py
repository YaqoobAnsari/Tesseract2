"""
transition_matcher.py - Automatic inter-floor transition correspondence

Given two single-floor graphs (Tesseract++ JSON exports) and their source
images, infer which transition nodes (stairs/elevators) on one floor are the
same physical shaft as transitions on the other floor.

The problem is framed as an optimal assignment with a rejection option:

  1. Registration: put both floors in a shared pixel frame. Candidate
     similarity transforms come from four generators - (a) pairs of same-kind
     transition correspondences, (b) their mirrored (conjugate) variants,
     (c) per-image dimension normalization, optionally anchored on a single
     candidate pair, and (d) phase correlation of the wall masks. Hypotheses
     outside a plausibility gate (scale 0.4-2.5, |rotation| <= 60 deg) are
     dropped. Ranking is wall-evidence-first: a symmetric, coverage-gated
     chamfer distance between the (glyph-filtered) wall masks, with a small
     bonus per one-to-one aligned transition pair. The winner is refined by
     least squares on its one-to-one inlier pairs and re-scored.
  2. Cost matrix: each floor-1/floor-2 transition pair is scored on position
     in the shared frame, stairs-vs-elevator kind mismatch, and (lightly
     weighted) graph-context feature distance.
  3. Assignment: the Hungarian solver runs on the cost matrix augmented with
     per-transition "match to nothing" slots; accepted pairs are then
     post-filtered so that a pair is kept if and only if its cost is at most
     reject_cost. Shafts whose best pairing exceeds that price stay
     unmatched (e.g. a stairwell that terminates).

Registration quality (chamfer, coverage, aligned count, mirror flag) is
reported and written into the mapping-file header; treat high-chamfer or
mirrored registrations as needing human review.

Pure file-level API - no imports from Main.py or MultiFloor.py.
"""

import json
import math
import os
from itertools import combinations

import cv2
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

# Wall pixels are anything darker than this (same convention as
# BuildingGraph.add_corridor_nodes in utils/graph.py).
WALL_THRESHOLD = 240
# Longest side of the downscaled masks used for chamfer scoring.
CHAMFER_MAX_DIM = 400
# Connected components smaller than this (downscaled px^2) are treated as
# text glyphs, not walls, and removed before chamfer scoring.
GLYPH_MIN_AREA = 30
# A transform counts a transition pair as aligned when the mapped position
# lands within this fraction of the floor-1 image diagonal.
ALIGN_FRACTION = 0.02
# Hypothesis plausibility gates: floor drawings of one building.
SCALE_RANGE = (0.4, 2.5)
MAX_ROTATION_DEG = 60.0
# Minimum fraction of floor-2 wall ink that must land inside the floor-1
# canvas for a hypothesis to be scoreable.
MIN_COVERAGE = 0.25
# Chamfer-units bonus for each one-to-one aligned transition pair when
# ranking hypotheses (aligning one more shaft is worth this many px of
# wall disagreement).
ALIGN_BONUS_PX = 1.5

DEFAULT_WEIGHTS = {
    'position': 1.0,   # normalized distance in the shared frame
    'kind': 1.0,       # stairs vs elevator mismatch penalty
    'context': 0.05,   # graph-neighbourhood feature distance
}
# A matched pair is accepted iff its cost <= reject_cost.
DEFAULT_REJECT_COST = 0.10


# =============================================================================
# LOADING
# =============================================================================

def transition_kind(node_id):
    """'stairs_3' -> 'stairs', 'elevator_1' -> 'elevator'."""
    return str(node_id).rsplit('_', 1)[0]


def load_floor(json_path, image_path):
    """Load one floor: graph, image size, and transition feature records."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    G = nx.Graph()
    for nd in data.get('nodes', []):
        G.add_node(nd['id'], type=nd.get('type', 'unknown'),
                   position=tuple(nd.get('position') or (0.0, 0.0)))
    for ed in data.get('edges', []):
        if ed['source'] in G and ed['target'] in G:
            G.add_edge(ed['source'], ed['target'],
                       weight=float(ed.get('weight', 1.0)))

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]

    transitions = []
    for t in sorted(n for n, d in G.nodes(data=True)
                    if d.get('type') == 'floor_transition'):
        pos = G.nodes[t]['position']
        if pos is None or any(p is None or not np.isfinite(p) for p in pos):
            print(f"  ⚠ Skipping transition {t}: invalid position {pos}")
            continue
        transitions.append(t)

    records = [{
        'id': t,
        'kind': transition_kind(t),
        'pos': np.array(G.nodes[t]['position'], dtype=float),
        'features': _context_features(G, t, transitions, (w, h)),
    } for t in transitions]

    return {'graph': G, 'image': img, 'size': (w, h),
            'diag': math.hypot(w, h), 'transitions': records,
            'json_path': json_path, 'image_path': image_path}


def _context_features(G, t, transitions, size):
    """Scale-invariant description of a transition's graph neighbourhood.
    Unavailable features are NaN and skipped during comparison."""
    diag = math.hypot(*size)
    pos = np.array(G.nodes[t]['position'], dtype=float)

    # Graph distance to the nearest exit door (fraction of the diagonal).
    exits = [n for n in G if str(n).startswith('exit_door_')]
    exit_dist = float('nan')
    if exits:
        try:
            lengths = nx.single_source_dijkstra_path_length(G, t, weight='weight')
            reached = [lengths[e] for e in exits if e in lengths]
            if reached:
                exit_dist = min(1.0, min(reached) / diag)
        except nx.NetworkXError:
            pass

    # Local corridor density: corridor nodes within 6% of the diagonal.
    radius = 0.06 * diag
    corridor_count = sum(
        1 for n, d in G.nodes(data=True)
        if d.get('type') == 'corridor'
        and np.linalg.norm(np.array(d['position'], dtype=float) - pos) <= radius
    )

    # Separation from the nearest other transition on the same floor.
    others = [np.array(G.nodes[o]['position'], dtype=float)
              for o in transitions if o != t]
    near_trans = float('nan')
    if others:
        near_trans = min(1.0, min(np.linalg.norm(p - pos) for p in others) / diag)

    return np.array([
        exit_dist,
        min(1.0, corridor_count / 50.0),
        min(1.0, G.degree(t) / 5.0),
        near_trans,
    ])


def _context_distance(f1, f2):
    """Mean absolute difference over the feature dims valid on both floors."""
    valid = np.isfinite(f1) & np.isfinite(f2)
    if not valid.any():
        return 0.0
    return float(np.mean(np.abs(f1[valid] - f2[valid])))


# =============================================================================
# TRANSFORMS (floor2 pixel frame -> floor1 pixel frame)
# =============================================================================
# A transform is a dict {'a': complex, 'b': complex, 'mirror': bool} mapping
# z2 -> a*z2 + b, or z2 -> a*conj(z2) + b when mirror is True.

def _make_transform(a, b, mirror=False):
    return {'a': complex(a), 'b': complex(b), 'mirror': bool(mirror)}


def _apply(tr, pts):
    z = np.asarray(pts, dtype=float)
    zc = z[..., 0] + 1j * z[..., 1]
    if tr['mirror']:
        zc = np.conj(zc)
    m = tr['a'] * zc + tr['b']
    return np.stack([m.real, m.imag], axis=-1)


def _invert(tr):
    """Inverse transform (floor1 -> floor2)."""
    a, b = tr['a'], tr['b']
    if not tr['mirror']:
        return _make_transform(1.0 / a, -b / a, False)
    # z1 = a*conj(z2)+b  =>  z2 = conj((z1-b)/a) = conj(1/a)*conj(z1) - conj(b/a)
    return _make_transform(np.conj(1.0 / a), -np.conj(b / a), True)


def _similarity_from_pairs(src_pts, dst_pts, mirror=False):
    """Least-squares similarity transform as a complex map. With mirror,
    fits z -> a*conj(z) + b. Returns a transform dict or None."""
    src = np.asarray(src_pts, dtype=float)
    dst = np.asarray(dst_pts, dtype=float)
    zs = src[:, 0] + 1j * src[:, 1]
    if mirror:
        zs = np.conj(zs)
    ws = dst[:, 0] + 1j * dst[:, 1]
    zc, wc = zs.mean(), ws.mean()
    denom = np.vdot(zs - zc, zs - zc).real
    if denom < 1e-9:
        return None
    a = np.vdot(zs - zc, ws - wc) / denom
    if abs(a) < 1e-6:
        return None
    b = wc - a * zc
    return _make_transform(a, b, mirror)


def _plausible(tr):
    s = abs(tr['a'])
    if not (SCALE_RANGE[0] <= s <= SCALE_RANGE[1]):
        return False
    return abs(math.degrees(np.angle(tr['a']))) <= MAX_ROTATION_DEG


# =============================================================================
# WALL-MASK EVIDENCE
# =============================================================================

def _wall_mask_small(img):
    """Downscaled binary wall mask with text glyphs removed."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = CHAMFER_MAX_DIM / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    mask = (small < WALL_THRESHOLD).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < GLYPH_MIN_AREA:
            mask[labels == i] = 0
    return mask, scale


def _ink_coords(mask, max_samples=4000, seed=12345):
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(float)
    if len(pts) > max_samples:
        idx = np.random.default_rng(seed).choice(len(pts), max_samples,
                                                 replace=False)
        pts = pts[idx]
    return pts


def _sym_chamfer(walls1, walls2, tr):
    """Symmetric, coverage-gated chamfer between wall masks under transform
    tr (full-res floor2 -> full-res floor1). Coordinate-sampled, so raster
    resampling cannot be gamed by extreme scales.

    Returns (chamfer_px_in_floor1_downscaled_units, forward_coverage)."""
    mask1, s1, dt1, pts1 = walls1
    mask2, s2, dt2, pts2 = walls2
    h1, w1 = mask1.shape
    h2, w2 = mask2.shape

    # Forward: floor-2 ink -> floor-1 frame.
    full2 = pts2 / s2
    mapped = _apply(tr, full2) * s1
    inside = ((mapped[:, 0] >= 0) & (mapped[:, 0] < w1)
              & (mapped[:, 1] >= 0) & (mapped[:, 1] < h1))
    coverage = float(inside.mean()) if len(mapped) else 0.0
    if coverage < MIN_COVERAGE:
        return float('inf'), coverage
    mi = mapped[inside]
    fwd = float(dt1[mi[:, 1].astype(int), mi[:, 0].astype(int)].mean())

    # Reverse: floor-1 ink -> floor-2 frame (skip if barely any lands there,
    # which is legitimate for partial-overlap sections).
    inv = _invert(tr)
    full1 = pts1 / s1
    mapped_r = _apply(inv, full1) * s2
    inside_r = ((mapped_r[:, 0] >= 0) & (mapped_r[:, 0] < w2)
                & (mapped_r[:, 1] >= 0) & (mapped_r[:, 1] < h2))
    if inside_r.mean() >= 0.05:
        mr = mapped_r[inside_r]
        rev = float(dt2[mr[:, 1].astype(int), mr[:, 0].astype(int)].mean())
        # Express reverse in floor-1 downscaled units for comparability.
        rev *= abs(tr['a']) * s1 / s2
        chamfer = 0.5 * (fwd + rev)
    else:
        chamfer = fwd
    return chamfer, coverage


def _one_to_one_aligned(t1, t2, tr, align_thresh):
    """Greedy one-to-one alignment of same-kind transitions under tr.
    Returns list of (i, j, dist) pairs with dist <= align_thresh."""
    if not t1 or not t2:
        return []
    mapped = _apply(tr, np.array([b['pos'] for b in t2]))
    cands = []
    for i, a in enumerate(t1):
        for j, b in enumerate(t2):
            if a['kind'] != b['kind']:
                continue
            d = float(np.linalg.norm(mapped[j] - a['pos']))
            if d <= align_thresh:
                cands.append((d, i, j))
    used_i, used_j, pairs = set(), set(), []
    for d, i, j in sorted(cands):
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((i, j, d))
    return pairs


# =============================================================================
# REGISTRATION
# =============================================================================

def _generate_hypotheses(floor1, floor2, walls1, walls2):
    t1, t2 = floor1['transitions'], floor2['transitions']
    candidate_pairs = [(i, j) for i, a in enumerate(t1)
                       for j, b in enumerate(t2) if a['kind'] == b['kind']]
    hyps = []

    # (a) dims: uniform-scale approximation of per-image normalization.
    s_dims = 0.5 * (floor1['size'][0] / floor2['size'][0]
                    + floor1['size'][1] / floor2['size'][1])
    hyps.append(('dims', _make_transform(s_dims, 0.0)))

    # (b) dims anchored on each single candidate pair (fixes crop shifts).
    for i, j in candidate_pairs:
        p1 = t1[i]['pos'][0] + 1j * t1[i]['pos'][1]
        p2 = t2[j]['pos'][0] + 1j * t2[j]['pos'][1]
        hyps.append((f'dims+anchor({t1[i]["id"]}~{t2[j]["id"]})',
                     _make_transform(s_dims, p1 - s_dims * p2)))

    # (c) two-pair similarity fits, plain and mirrored.
    for (i1, j1), (i2, j2) in combinations(candidate_pairs, 2):
        if i1 == i2 or j1 == j2:
            continue
        name = (f'pairs({t1[i1]["id"]}~{t2[j1]["id"]},'
                f'{t1[i2]["id"]}~{t2[j2]["id"]})')
        for mirror in (False, True):
            tr = _similarity_from_pairs(
                [t2[j1]['pos'], t2[j2]['pos']],
                [t1[i1]['pos'], t1[i2]['pos']], mirror=mirror)
            if tr is not None:
                hyps.append((name + ('+mirror' if mirror else ''), tr))

    # (d) phase correlation of the wall masks at dims scale, tried at the
    # four CAD-typical orientations plus a mirrored variant.
    for deg in (0.0, 90.0, 180.0, 270.0):
        a = s_dims * np.exp(1j * math.radians(deg))
        tr_phase = _phase_at(walls1, walls2, floor1, floor2, a, mirror=False)
        if tr_phase is not None:
            hyps.append((f'phase@{int(deg)}', tr_phase))
    tr_phase = _phase_at(walls1, walls2, floor1, floor2,
                         complex(s_dims), mirror=True)
    if tr_phase is not None:
        hyps.append(('phase@0+mirror', tr_phase))

    return [(n, tr) for n, tr in hyps if _plausible(tr)]


def _transform_matrix_grid(tr, s1, s2):
    """2x3 warpAffine matrix taking mask2 grid coords to mask1 grid coords
    for a full-resolution transform tr."""
    a = tr['a'] * (s1 / s2)
    b = tr['b'] * s1
    if not tr['mirror']:
        return np.array([[a.real, -a.imag, b.real],
                         [a.imag,  a.real, b.imag]])
    # z1 = a*conj(z2)+b: x1 = ar*x2 + ai*y2 + br; y1 = ai*x2 - ar*y2 + bi
    return np.array([[a.real,  a.imag, b.real],
                     [a.imag, -a.real, b.imag]])


def _phase_at(walls1, walls2, floor1, floor2, a, mirror=False):
    """Translation estimate via phase correlation, given the linear part
    `a` (scale+rotation, optionally mirrored). Seeds the translation so the
    floor-2 centre maps to the floor-1 centre, warps, and reads the
    residual shift off the correlation peak."""
    mask1, s1 = walls1[0], walls1[1]
    mask2, s2 = walls2[0], walls2[1]
    c1 = complex(floor1['size'][0] / 2.0, floor1['size'][1] / 2.0)
    c2 = complex(floor2['size'][0] / 2.0, floor2['size'][1] / 2.0)
    c2_eff = np.conj(c2) if mirror else c2
    b0 = c1 - a * c2_eff
    tr0 = _make_transform(a, b0, mirror)
    M = _transform_matrix_grid(tr0, s1, s2)
    h1, w1 = mask1.shape
    warped = cv2.warpAffine(mask2, M, (w1, h1),
                            flags=cv2.INTER_NEAREST, borderValue=0)
    if warped.sum() < 50:
        return None
    try:
        (dx, dy), _ = cv2.phaseCorrelate(warped.astype(np.float32),
                                         mask1.astype(np.float32))
    except cv2.error:
        return None
    # (dx, dy): shift aligning the warped mask to mask1, in mask1-grid units.
    return _make_transform(a, b0 + complex(dx / s1, dy / s1), mirror)


def _icp_polish(walls1, walls2, tr, iterations=8, trim=3.0):
    """Iterative closest point on wall ink: repeatedly pair each mapped
    floor-2 wall sample with its nearest floor-1 wall pixel, trim outliers,
    refit the similarity, and iterate. Returns the polished transform (the
    caller decides whether it actually improved the chamfer)."""
    mask1, s1, _, _ = walls1
    _, s2, _, pts2 = walls2
    h1, w1 = mask1.shape

    # Nearest-wall lookup for mask1 via labelled distance transform.
    _, labels = cv2.distanceTransformWithLabels(
        (1 - mask1).astype(np.uint8), cv2.DIST_L2, 5,
        labelType=cv2.DIST_LABEL_PIXEL)
    wall_ys, wall_xs = np.nonzero(mask1)
    if len(wall_xs) == 0:
        return tr
    lut = np.zeros((labels.max() + 1, 2))
    lut[labels[wall_ys, wall_xs]] = np.stack([wall_xs, wall_ys], axis=1)

    full2 = pts2 / s2
    for _ in range(iterations):
        mapped = _apply(tr, full2) * s1
        inside = ((mapped[:, 0] >= 0) & (mapped[:, 0] < w1)
                  & (mapped[:, 1] >= 0) & (mapped[:, 1] < h1))
        if inside.sum() < 20:
            return tr
        mi = mapped[inside]
        nearest = lut[labels[mi[:, 1].astype(int), mi[:, 0].astype(int)]]
        d = np.linalg.norm(mi - nearest, axis=1)
        keep = d <= max(2.0, trim * np.median(d))
        if keep.sum() < 20:
            return tr
        refit = _similarity_from_pairs(full2[inside][keep],
                                       nearest[keep] / s1,
                                       mirror=tr['mirror'])
        if refit is None or not _plausible(refit):
            return tr
        if abs(refit['a'] - tr['a']) < 1e-9 and abs(refit['b'] - tr['b']) < 1e-6:
            return refit
        tr = refit
    return tr


def register_floors(floor1, floor2, verbose=True):
    """Estimate the similarity transform mapping floor-2 pixel coordinates
    into the floor-1 frame. Returns (transform, report)."""
    t1, t2 = floor1['transitions'], floor2['transitions']
    align_thresh = ALIGN_FRACTION * floor1['diag']

    def prep(img):
        mask, s = _wall_mask_small(img)
        dt = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
        return (mask, s, dt, _ink_coords(mask))

    walls1 = prep(floor1['image'])
    walls2 = prep(floor2['image'])

    hyps = _generate_hypotheses(floor1, floor2, walls1, walls2)

    def evaluate(tr):
        chamfer, coverage = _sym_chamfer(walls1, walls2, tr)
        pairs = _one_to_one_aligned(t1, t2, tr, align_thresh)
        score = chamfer - ALIGN_BONUS_PX * len(pairs)
        return chamfer, coverage, pairs, score

    best = None
    for name, tr in hyps:
        chamfer, coverage, pairs, score = evaluate(tr)
        if math.isinf(chamfer):
            continue
        if best is None or score < best['score']:
            best = {'name': name, 'transform': tr, 'chamfer': chamfer,
                    'coverage': coverage, 'pairs': pairs, 'score': score}

    if best is None:
        # Nothing scoreable (e.g. no wall overlap at any plausible
        # transform). Fall back to dims and let rejection do its job.
        s_dims = 0.5 * (floor1['size'][0] / floor2['size'][0]
                        + floor1['size'][1] / floor2['size'][1])
        tr = _make_transform(s_dims, 0.0)
        best = {'name': 'dims (unscored fallback)', 'transform': tr,
                'chamfer': float('inf'), 'coverage': 0.0, 'pairs': [],
                'score': float('inf')}

    # Refine on one-to-one inliers, then RE-SCORE so the report matches the
    # transform actually returned.
    if len(best['pairs']) >= 2:
        src = [t2[j]['pos'] for _, j, _ in best['pairs']]
        dst = [t1[i]['pos'] for i, _, _ in best['pairs']]
        refined = _similarity_from_pairs(src, dst,
                                         mirror=best['transform']['mirror'])
        if refined is not None and _plausible(refined):
            chamfer, coverage, pairs, score = evaluate(refined)
            if not math.isinf(chamfer) and score <= best['score'] + 1e-9:
                best.update({'transform': refined, 'chamfer': chamfer,
                             'coverage': coverage, 'pairs': pairs,
                             'score': score,
                             'name': best['name'] + '+refined'})

    # ICP polish on wall ink; keep only if the combined score improves.
    if not math.isinf(best['chamfer']):
        polished = _icp_polish(walls1, walls2, best['transform'])
        chamfer, coverage, pairs, score = evaluate(polished)
        if not math.isinf(chamfer) and score < best['score'] - 1e-9:
            best.update({'transform': polished, 'chamfer': chamfer,
                         'coverage': coverage, 'pairs': pairs,
                         'score': score, 'name': best['name'] + '+icp'})

    tr = best['transform']
    residual = (float(np.mean([d for _, _, d in best['pairs']]))
                if best['pairs'] else None)
    a = tr['a']
    report = {
        'hypothesis': best['name'],
        'n_hypotheses': len(hyps),
        'scale': abs(a),
        'rotation_deg': math.degrees(np.angle(a)),
        'translation_px': (tr['b'].real, tr['b'].imag),
        'mirrored': tr['mirror'],
        'aligned_transitions': len(best['pairs']),
        'chamfer_px': best['chamfer'],
        'coverage': best['coverage'],
        'residual_px': residual,
    }
    if verbose:
        print(f"  Registration: {report['hypothesis']}  "
              f"scale={report['scale']:.4f}  rot={report['rotation_deg']:.2f}°"
              f"{'  MIRRORED' if report['mirrored'] else ''}  "
              f"aligned={report['aligned_transitions']}/{min(len(t1), len(t2))}  "
              f"chamfer={report['chamfer_px']:.2f}px  "
              f"coverage={report['coverage']:.2f}")
    return tr, report


# =============================================================================
# MATCHING
# =============================================================================

def build_cost_matrix(floor1, floor2, transform, weights=None):
    weights = weights or DEFAULT_WEIGHTS
    t1, t2 = floor1['transitions'], floor2['transitions']
    diag = floor1['diag']
    mapped = _apply(transform, np.array([b['pos'] for b in t2])) if t2 else np.zeros((0, 2))

    C = np.zeros((len(t1), len(t2)))
    for i, a in enumerate(t1):
        for j, b in enumerate(t2):
            pos_term = np.linalg.norm(mapped[j] - a['pos']) / diag
            kind_term = 0.0 if a['kind'] == b['kind'] else 1.0
            ctx_term = _context_distance(a['features'], b['features'])
            C[i, j] = (weights['position'] * pos_term
                       + weights['kind'] * kind_term
                       + weights['context'] * ctx_term)
    return C


def solve_assignment(C, reject_cost=DEFAULT_REJECT_COST):
    """Hungarian assignment over C augmented with one 'match to nothing'
    slot per transition, followed by a post-filter so that a pair is
    accepted iff C[i, j] <= reject_cost. Returns (matches, um1, um2)
    where matches is a list of (i, j, cost)."""
    n1, n2 = C.shape
    BIG = 1e6
    size = n1 + n2
    A = np.full((size, size), BIG)
    A[:n1, :n2] = C
    for i in range(n1):
        A[i, n2 + i] = reject_cost          # floor-1 shaft goes unmatched
    for j in range(n2):
        A[n1 + j, j] = reject_cost          # floor-2 shaft goes unmatched
    A[n1:, n2:] = 0.0                       # dummy-dummy pairings are free

    rows, cols = linear_sum_assignment(A)
    matches, matched1, matched2 = [], set(), set()
    for r, c in zip(rows, cols):
        # Post-filter enforces the documented acceptance rule; without it
        # the augmentation only rejects pairs costing above 2*reject_cost
        # (matching frees BOTH sides' rejection fees).
        if r < n1 and c < n2 and C[r, c] <= reject_cost:
            matches.append((r, c, float(C[r, c])))
            matched1.add(r)
            matched2.add(c)
    um1 = [i for i in range(n1) if i not in matched1]
    um2 = [j for j in range(n2) if j not in matched2]
    return matches, um1, um2


def match_transitions(floor1, floor2, reject_cost=DEFAULT_REJECT_COST,
                      weights=None, verbose=True):
    """Full pipeline on two loaded floors. Returns a result dict."""
    transform, reg_report = register_floors(floor1, floor2, verbose=verbose)
    C = build_cost_matrix(floor1, floor2, transform, weights=weights)
    matches, um1, um2 = solve_assignment(C, reject_cost=reject_cost)

    t1, t2 = floor1['transitions'], floor2['transitions']
    out_matches = []
    for i, j, cost in matches:
        # Confidence: margin between this cost and the cheapest alternative
        # for EITHER endpoint (row and column competitors, or rejection),
        # as a fraction of the reject cost.
        alternatives = ([C[i, k] for k in range(len(t2)) if k != j]
                        + [C[k, j] for k in range(len(t1)) if k != i]
                        + [reject_cost])
        margin = min(alternatives) - cost
        confidence = max(0.0, min(1.0, margin / reject_cost))
        out_matches.append({
            'floor1_node': t1[i]['id'],
            'floor2_node': t2[j]['id'],
            'cost': cost,
            'confidence': confidence,
        })

    result = {
        'matches': out_matches,
        'unmatched_floor1': [t1[i]['id'] for i in um1],
        'unmatched_floor2': [t2[j]['id'] for j in um2],
        'registration': reg_report,
        'cost_matrix': C,
        'reject_cost': reject_cost,
    }
    if verbose:
        for m in out_matches:
            print(f"  Matched: {m['floor1_node']} ↔ {m['floor2_node']}  "
                  f"cost={m['cost']:.4f}  confidence={m['confidence']:.2f}")
        for nid in result['unmatched_floor1']:
            print(f"  Unmatched on floor 1 (no partner above/below): {nid}")
        for nid in result['unmatched_floor2']:
            print(f"  Unmatched on floor 2 (no partner above/below): {nid}")
    return result


# =============================================================================
# OUTPUT
# =============================================================================

def registration_warnings(reg):
    """Human-review warnings derived from the registration report."""
    warns = []
    if reg.get('mirrored'):
        warns.append("registration is MIRRORED - one drawing appears "
                     "reflected relative to the other; verify manually")
    chamfer = reg.get('chamfer_px')
    if chamfer is None or (isinstance(chamfer, float) and
                           (math.isinf(chamfer) or chamfer > 3.0)):
        warns.append(f"weak wall agreement (chamfer={chamfer}); "
                     "registration may be unreliable")
    if reg.get('aligned_transitions', 0) < 2:
        warns.append("fewer than 2 aligned transition pairs support the "
                     "registration")
    return warns


def write_mapping_file(result, floor1_num, image1, floor2_num, image2, out_path):
    """Write matches in the standard mapping-file format, with provenance."""
    reg = result['registration']

    def fmt(v, nd=2):
        return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    lines = [
        "# AUTO-GENERATED transition mapping - utils/transition_matcher.py",
        f"# Floors: {floor1_num} ({image1}) -> {floor2_num} ({image2})",
        f"# Registration: {reg['hypothesis']}, scale={reg['scale']:.4f}, "
        f"rotation={reg['rotation_deg']:.2f} deg, mirrored={reg['mirrored']}",
        f"# Wall chamfer: {fmt(reg['chamfer_px'])} px, coverage "
        f"{fmt(reg['coverage'])}, aligned transitions "
        f"{reg['aligned_transitions']}, residual {fmt(reg['residual_px'])} px",
        f"# Acceptance rule: pair kept iff cost <= {result['reject_cost']}",
    ]
    for w in registration_warnings(reg):
        lines.append(f"# WARNING: {w}")
    lines += ["# Review before use; hand-edit like any mapping file.", ""]
    for m in result['matches']:
        lines.append(f"# cost={m['cost']:.4f} confidence={m['confidence']:.2f}")
        lines.append(f"({floor1_num}, {image1}, {m['floor1_node']}):"
                     f"({floor2_num}, {image2}, {m['floor2_node']})")
        lines.append("")
    for nid in result['unmatched_floor1']:
        lines.append(f"# UNMATCHED on floor {floor1_num}: {nid} "
                     "(best pairing cost exceeded the reject cost)")
    for nid in result['unmatched_floor2']:
        lines.append(f"# UNMATCHED on floor {floor2_num}: {nid} "
                     "(best pairing cost exceeded the reject cost)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def auto_match(json1, image1_path, json2, image2_path,
               reject_cost=DEFAULT_REJECT_COST, weights=None, verbose=True):
    """Convenience wrapper: load both floors from files and match."""
    floor1 = load_floor(json1, image1_path)
    floor2 = load_floor(json2, image2_path)
    if verbose:
        print(f"  Floor 1: {len(floor1['transitions'])} transition(s) "
              f"{[t['id'] for t in floor1['transitions']]}")
        print(f"  Floor 2: {len(floor2['transitions'])} transition(s) "
              f"{[t['id'] for t in floor2['transitions']]}")
    return match_transitions(floor1, floor2, reject_cost=reject_cost,
                             weights=weights, verbose=verbose)
