"""
freespace.py - Free-space (traversable pixel) helpers

Shared by the route-fidelity harness and the skeleton baseline:
  - a traversable mask built from the drawing (non-wall pixels, with door
    positions opened, since door swing-arc strokes can seal a doorway)
  - geodesic distances through free space (the objective reference for
    "true walkable distance" - respects walls, unlike straight-line
    Euclidean)

Distances are exact pixel geodesics computed with skimage's MCP_Geometric
on a downsampled grid (factor DS), rescaled back to full-resolution pixels.
"""

import math

import cv2
import numpy as np
from skimage.graph import MCP_Geometric

WALL_THRESHOLD = 240
DOOR_OPEN_RADIUS = 18   # px disc painted free at each door position
DS = 2                  # downsample factor for the geodesic grid


def traversable_mask(image, door_positions=(), interior_only=False,
                     exclude_mask=None):
    """Boolean mask (full resolution): True = walkable. Non-wall pixels are
    walkable; a disc at every door position is forced walkable so door
    glyph strokes cannot seal a doorway.

    With interior_only and an exclude_mask (e.g. the pipeline's semantic
    'outside' class from semantic_outside_mask), those pixels are removed
    so geodesics cannot shortcut through the building exterior; door discs
    are re-opened afterwards so exit doors stay reachable from inside."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    free = (gray >= WALL_THRESHOLD).astype(np.uint8)
    for (x, y) in door_positions:
        cv2.circle(free, (int(round(x)), int(round(y))), DOOR_OPEN_RADIUS, 1, -1)
    if interior_only and exclude_mask is not None:
        free[exclude_mask] = 0
        for (x, y) in door_positions:
            cv2.circle(free, (int(round(x)), int(round(y))), DOOR_OPEN_RADIUS, 1, -1)
    return free.astype(bool)


def semantic_outside_mask(recreated_image, outside_positions, tol=10):
    """Boolean mask of the pipeline's 'outside' semantic class, recovered
    from the 5-color recreated image by sampling the class color at known
    outside-node positions (pre-pruning graph). Returns None if no color
    can be established."""
    if recreated_image is None or not outside_positions:
        return None
    samples = []
    h, w = recreated_image.shape[:2]
    for (x, y) in outside_positions[:50]:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            samples.append(recreated_image[yi, xi].astype(int))
    if not samples:
        return None
    color = np.median(np.array(samples), axis=0)
    diff = np.abs(recreated_image.astype(int) - color[None, None, :]).sum(axis=2)
    return diff <= tol


def _grid(mask):
    small = mask[::DS, ::DS]
    costs = np.where(small, 1.0, np.inf)
    return costs


def geodesic_from(mask, src_xy):
    """Return (cost_map, mcp) with distances (full-res px) from src to every
    pixel; unreachable = inf."""
    costs = _grid(mask)
    mcp = MCP_Geometric(costs)
    src = (int(src_xy[1]) // DS, int(src_xy[0]) // DS)
    if not np.isfinite(costs[src]):
        # nudge onto the nearest free cell within a small window
        best, r = None, 6
        ys, xs = np.nonzero(np.isfinite(
            costs[max(0, src[0]-r):src[0]+r+1, max(0, src[1]-r):src[1]+r+1]))
        if len(ys):
            best = (max(0, src[0]-r) + ys[0], max(0, src[1]-r) + xs[0])
        if best is None:
            return None, None
        src = best
    cost_map, _ = mcp.find_costs([src])
    return cost_map * DS, mcp


def read_distance(cost_map, dst_xy):
    if cost_map is None:
        return math.inf
    r, c = int(dst_xy[1]) // DS, int(dst_xy[0]) // DS
    r = min(max(r, 0), cost_map.shape[0] - 1)
    c = min(max(c, 0), cost_map.shape[1] - 1)
    # tolerate an endpoint on a wall pixel: best value in a small window
    r0, r1 = max(0, r - 10), min(cost_map.shape[0], r + 11)
    c0, c1 = max(0, c - 10), min(cost_map.shape[1], c + 11)
    return float(np.min(cost_map[r0:r1, c0:c1]))
