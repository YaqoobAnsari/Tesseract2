# utils/floodfill.py
import os
import cv2
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


# ---------------------------- small helpers ---------------------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _is_white(pixel_bgr):
    """Return True if pixel is pure white (BGR)."""
    return (pixel_bgr[0] == 255 and pixel_bgr[1] == 255 and pixel_bgr[2] == 255)


def _is_wall(pixel_bgr):
    """Treat any non-white pixel as a wall/obstacle for legacy flood modes."""
    return not _is_white(pixel_bgr)


def _distance_transform(mask_uint8):
    """
    mask_uint8: 0/1 mask (H, W)
    returns float32 distance-to-boundary (same shape), measured in pixels.
    """
    return cv2.distanceTransform((mask_uint8 * 255).astype(np.uint8), cv2.DIST_L2, 3)


def _grid_sample_from_mask(mask_uint8, spacing_px=30, pad_px=2, jitter_px=0):
    """
    Deterministic, centroid-aligned hex grid sampling inside a binary mask.
    Enforces a 'pad_px' clearance from walls using a distance transform.
    Returns: list[(x, y)] in image coordinates.
    """
    if mask_uint8 is None or mask_uint8.size == 0:
        return []

    H, W = mask_uint8.shape
    dt = _distance_transform((mask_uint8 > 0).astype(np.uint8))

    ys, xs = np.where(mask_uint8 > 0)
    if xs.size == 0:
        return []
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    cx, cy = float(xs.mean()), float(ys.mean())

    s = max(1, int(spacing_px))  # column spacing
    row_step = max(1, int(round(s * np.sqrt(3) / 2)))  # hex vertical step

    # align grid so one node sits near centroid
    oy = int(round(cy)) % row_step
    ox = int(round(cx)) % s

    def _first_aligned(start, period, offset):
        return start + ((period - (start - offset) % period) % period)

    start_y = _first_aligned(y_min, row_step, oy)
    start_x_even = _first_aligned(x_min, s, ox)
    start_x_odd = start_x_even + s // 2

    points = []
    row_idx = 0
    y = start_y
    rng = np.random.default_rng(1337)  # deterministic jitter
    while y <= y_max:
        x0 = start_x_even if (row_idx % 2 == 0) else start_x_odd
        x = x0
        while x <= x_max:
            yy = int(np.clip(y, 0, H - 1))
            xx = int(np.clip(x, 0, W - 1))

            if jitter_px > 0:
                jx = int(rng.integers(-jitter_px, jitter_px + 1))
                jy = int(rng.integers(-jitter_px, jitter_px + 1))
                xx = int(np.clip(xx + jx, 0, W - 1))
                yy = int(np.clip(yy + jy, 0, H - 1))

            if mask_uint8[yy, xx] and dt[yy, xx] >= pad_px:
                points.append((int(xx), int(yy)))
            x += s
        y += row_step
        row_idx += 1

    return points


# ------------------------- legacy fill function -----------------------------

def process_fill_rooms(
    image_path,
    graph,
    results_dir,
    radius_threshold=90,
    node_radius=20,
    fill_mode="smart",  # "smart" or "flood"
    point_radius=10,  # smart fill only
    point_step=10,    # smart fill only
    flood_threshold=30,  # flood fill only
):
    """
    Legacy visualization: display nodes and perform smart/flood filling around nodes.
    """
    floorplan = cv2.imread(image_path)
    if floorplan is None:
        raise FileNotFoundError(f"Floorplan image not found at {image_path}")

    H, W = floorplan.shape[:2]
    image_name_no_ext = os.path.splitext(os.path.basename(image_path))[0]

    # Create directories
    plots_dir = os.path.join(results_dir, "Plots")
    fill_dir = os.path.join(plots_dir, "smart_fill" if fill_mode == "smart" else "flood_fill")
    _ensure_dir(fill_dir)
    fill_img_dir = os.path.join(fill_dir, f"{image_name_no_ext}")
    _ensure_dir(fill_img_dir)

    output_image_path = os.path.join(fill_img_dir, f"{image_name_no_ext}_{fill_mode}fill.png")
    area_file_path = os.path.join(fill_img_dir, f"{fill_mode}_fill_area.txt")

    with open(area_file_path, "w") as area_file:
        area_file.write("Node_ID\tPosition\tFilled_Area\n")

    def is_valid_room_seed(seed_x, seed_y, min_area_threshold=200):
        """
        Test if a seed point is in a valid room area (not text annotation).
        Performs a small test flood fill to check if the region is large enough.
        
        Args:
            seed_x, seed_y: Seed coordinates to test
            min_area_threshold: Minimum area (pixels) required to consider seed valid
            
        Returns:
            True if seed is in a valid room area, False otherwise
        """
        if not (0 <= seed_x < W and 0 <= seed_y < H):
            return False
        
        # Must be white
        if not _is_white(floorplan[seed_y, seed_x]):
            return False
        
        # Perform a small test flood fill to check if this is a large enough region
        test_image = floorplan.copy()
        test_mask = np.zeros((H + 2, W + 2), np.uint8)
        try:
            _, _, _, rect = cv2.floodFill(
                test_image, test_mask, (seed_x, seed_y), (128, 128, 128),
                loDiff=(10, 10, 10), upDiff=(10, 10, 10),
                flags=cv2.FLOODFILL_MASK_ONLY
            )
            # Check the filled area
            filled_area = int(test_mask[1:-1, 1:-1].astype(bool).sum())
            # If filled area is large enough, this is likely a room, not text
            return filled_area >= min_area_threshold
        except Exception:
            return False

    def find_valid_seeds(center_x, center_y, max_radius=150, min_radius=20, step_radius=10, angle_step=30):
        """
        Find valid seed points in concentric circles around the center.
        Skips the center if it's on text/wall, and finds nearby valid room pixels.
        Uses area-based validation to avoid text annotations.
        
        Args:
            center_x, center_y: Center coordinates (node position)
            max_radius: Maximum radius to search (pixels) - increased to avoid text
            min_radius: Minimum radius to start search (pixels) - start further out
            step_radius: Radius increment between circles
            angle_step: Angle step for sampling points on each circle
            
        Returns:
            List of valid (x, y) seed points
        """
        valid_seeds = []
        cx = max(0, min(int(center_x), W - 1))
        cy = max(0, min(int(center_y), H - 1))
        
        # Check center first, but validate it's in a room area (not text)
        if is_valid_room_seed(cx, cy):
            valid_seeds.append((cx, cy))
        
        # Search in concentric circles, starting from min_radius (further from center)
        # This helps avoid text annotations that are typically near the node position
        for radius in range(min_radius, max_radius + 1, step_radius):
            if len(valid_seeds) >= 12:  # Collect more seeds for better coverage
                break
            for angle in range(0, 360, angle_step):
                radian = np.radians(angle)
                px = int(cx + radius * np.cos(radian))
                py = int(cy + radius * np.sin(radian))
                if 0 <= px < W and 0 <= py < H:
                    # Validate that this seed is in a room area, not text
                    if is_valid_room_seed(px, py):
                        if (px, py) not in valid_seeds:
                            valid_seeds.append((px, py))
                            if len(valid_seeds) >= 12:
                                break
            if len(valid_seeds) >= 12:
                break
        
        # If no valid seeds found, try a more aggressive search with lower threshold
        if not valid_seeds:
            for radius in range(min_radius, max_radius + 1, step_radius):
                for angle in range(0, 360, angle_step):
                    radian = np.radians(angle)
                    px = int(cx + radius * np.cos(radian))
                    py = int(cy + radius * np.sin(radian))
                    if 0 <= px < W and 0 <= py < H:
                        if _is_white(floorplan[py, px]):
                            if is_valid_room_seed(px, py, min_area_threshold=50):  # Lower threshold
                                valid_seeds.append((px, py))
                                if len(valid_seeds) >= 4:
                                    break
                if len(valid_seeds) >= 4:
                    break
        
        # Last resort: return center if nothing found (but this should be rare)
        if not valid_seeds:
            valid_seeds = [(cx, cy)]
        
        return valid_seeds

    def fill_node(node_id, node_data, fill_color):
        nonlocal floorplan
        x, y = node_data["position"]
        x = max(0, min(int(x), W - 1))
        y = max(0, min(int(y), H - 1))
        fill_area = 0

        if fill_mode == "smart":
            visited = np.zeros((H, W), dtype=bool)
            # Generate starting points using radius-based search to avoid text annotations
            # Use larger radius to avoid text, and validate seeds are in room areas
            starts = find_valid_seeds(x, y, max_radius=max(150, point_radius * 5), 
                                     min_radius=max(20, point_radius), step_radius=10, angle_step=point_step)
            # Also validate and add points at point_radius distance if they're valid room seeds
            for angle in range(0, 360, point_step):
                radian = np.radians(angle)
                px = int(x + point_radius * np.cos(radian))
                py = int(y + point_radius * np.sin(radian))
                if 0 <= px < W and 0 <= py < H:
                    if is_valid_room_seed(px, py):
                        if (px, py) not in starts:
                            starts.append((px, py))

            # BFS
            for sx, sy in starts:
                if visited[sy, sx]:
                    continue
                q = [(sx, sy)]
                visited[sy, sx] = True
                while q:
                    cx, cy = q.pop(0)
                    if (cx - x) ** 2 + (cy - y) ** 2 > radius_threshold ** 2:
                        continue
                    if _is_wall(floorplan[cy, cx]):
                        continue
                    floorplan[cy, cx] = fill_color
                    fill_area += 1
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < W and 0 <= ny < H and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((nx, ny))

        else:  # "flood"
            # Use radius-based seed finding to avoid text annotations
            # Use larger radius and validate seeds are in room areas
            flood_seeds = find_valid_seeds(x, y, max_radius=max(150, flood_threshold * 5),
                                          min_radius=max(20, flood_threshold), step_radius=10, angle_step=10)
            # Also validate and add original flood_threshold points if they're valid room seeds
            for angle in range(0, 360, 10):
                rad = np.radians(angle)
                fx = int(x + flood_threshold * np.cos(rad))
                fy = int(y + flood_threshold * np.sin(rad))
                if 0 <= fx < W and 0 <= fy < H:
                    if is_valid_room_seed(fx, fy):
                        if (fx, fy) not in flood_seeds:
                            flood_seeds.append((fx, fy))
            
            # Perform flood fill from all valid seeds
            for fx, fy in flood_seeds:
                if 0 <= fx < W and 0 <= fy < H and not _is_wall(floorplan[fy, fx]):
                    mask = np.zeros((H + 2, W + 2), np.uint8)
                    cv2.floodFill(
                        floorplan, mask, (fx, fy), fill_color,
                        loDiff=(10, 10, 10), upDiff=(10, 10, 10),
                    )
                    fill_area += int(mask[1:-1, 1:-1].astype(bool).sum())

        return fill_area

    # outside
    for node_id in graph.node_types.get("outside", []):
        if node_id not in graph.graph.nodes:
            continue
        node_data = graph.graph.nodes[node_id]
        fill_area = fill_node(node_id, node_data, (204, 102, 255))  # purple
        with open(area_file_path, "a") as f:
            f.write(f"{node_id}\t{tuple(node_data['position'])}\t{fill_area}\n")

    # corridor
    for node_id in graph.node_types.get("corridor", []):
        if node_id not in graph.graph.nodes:
            continue
        node_data = graph.graph.nodes[node_id]
        fill_area = fill_node(node_id, node_data, (102, 255, 178))  # mint
        with open(area_file_path, "a") as f:
            f.write(f"{node_id}\t{tuple(node_data['position'])}\t{fill_area}\n")

    # rooms (main only)
    for node_id in graph.node_types.get("room", []):
        if node_id not in graph.graph.nodes:
            continue
        if graph.graph.nodes[node_id].get("is_subnode", False):
            continue
        node_data = graph.graph.nodes[node_id]
        fill_area = fill_node(node_id, node_data, (102, 204, 255))  # light blue
        with open(area_file_path, "a") as f:
            f.write(f"{node_id}\t{tuple(node_data['position'])}\t{fill_area}\n")

    cv2.imwrite(output_image_path, floorplan)
    print(f"{fill_mode.capitalize()}-filled image saved to {output_image_path}")
    print(f"{fill_mode.capitalize()}-filled areas saved to {area_file_path}")
    return output_image_path, area_file_path


# ------------------ segmentation-driven room partition ----------------------

def _mask_from_color_image(seg_bgr, rgb_color, tol=12):
    """
    Build a binary mask for pixels near the given RGB color in an RGB image.
    seg_bgr: image in BGR or RGB? We will assume **RGB** when caller passes RGB tuple.
    """
    # If input is BGR, convert to RGB (we detect by checking channel order heuristically)
    # Safer: let caller pass already-RGB; here we convert BGR->RGB if seems likely.
    if seg_bgr.ndim == 3:
        # Assume input was BGR; convert to RGB for consistent comparison
        seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("seg_bgr must be a 3-channel image")

    target = np.array(rgb_color, dtype=np.int16)
    img = seg_rgb.astype(np.int16)
    lower = np.maximum(target - tol, 0)
    upper = np.minimum(target + tol, 255)
    mask = np.all((img >= lower) & (img <= upper), axis=-1)
    return mask.astype(np.uint8)


def _room_nodes_from_graph(graph):
    """Return list of (room_id, (x,y)) for MAIN rooms (excludes subnodes)."""
    out = []
    for nid, data in graph.graph.nodes(data=True):
        if data.get("type") != "room":
            continue
        if data.get("is_subnode", False) or ("_subnode_" in str(nid)):
            continue
        if "position" in data:
            x, y = data["position"]
            out.append((nid, (int(x), int(y))))
    return out


def _snap_seed_into_mask(seed_xy, mask_uint8, max_radius=50):
    """
    If a seed lies outside the mask, move it to the nearest 1-pixel within max_radius.
    Returns (x,y) or None if no pixel within radius is found.
    """
    H, W = mask_uint8.shape
    x0, y0 = seed_xy
    x0 = int(np.clip(x0, 0, W - 1))
    y0 = int(np.clip(y0, 0, H - 1))
    if mask_uint8[y0, x0]:
        return (x0, y0)

    ys, xs = np.where(mask_uint8 > 0)
    if xs.size == 0:
        return None
    # limit search by radius to avoid snapping across building
    d2 = (xs - x0) ** 2 + (ys - y0) ** 2
    idx = np.argmin(d2)
    if float(d2[idx]) <= float(max_radius ** 2):
        return (int(xs[idx]), int(ys[idx]))
    return None


def _partition_room_mask_by_seeds(room_mask, seeds_xy):
    """
    Multi-source BFS partition of a binary room mask into Voronoi-like regions.
    Returns label_img (H,W) with values in {0..N}, 0=unassigned, i>0 -> i-th seed.
    """
    H, W = room_mask.shape
    label = np.zeros((H, W), dtype=np.int32)
    q = deque()

    # Initialize queue with seeds
    for i, (x, y) in enumerate(seeds_xy, start=1):
        if not (0 <= x < W and 0 <= y < H):
            continue
        if room_mask[y, x] == 0:
            continue
        label[y, x] = i
        q.append((x, y, i))

    # BFS growth constrained to room_mask
    while q:
        x, y, lab = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and room_mask[ny, nx] and label[ny, nx] == 0:
                label[ny, nx] = lab
                q.append((nx, ny, lab))
    return label


def _extract_per_room_masks_from_seg(segmented_map_path, graph,
                                     room_rgb=(255, 204, 102), tol=14, snap_radius=50):
    """
    Use the recreated segmented map (from pixelwise_areas) to generate **per-room** masks
    with geodesic partitioning inside the global room mask.
    """
    seg = cv2.imread(segmented_map_path)
    if seg is None:
        raise FileNotFoundError(f"Segmented map not found: {segmented_map_path}")
    H, W = seg.shape[:2]

    # Build global room mask from the known "room" color
    room_mask = _mask_from_color_image(seg, room_rgb, tol=tol)  # (H,W) uint8 0/1

    # Collect main room nodes and snap their seeds into the room mask
    main_rooms = _room_nodes_from_graph(graph)
    seed_xy = []
    seed_ids = []
    for rid, (x, y) in main_rooms:
        snapped = _snap_seed_into_mask((x, y), room_mask, max_radius=snap_radius)
        if snapped is None:
            # Seed is too far from room region; skip this room for partition (no mask).
            continue
        seed_xy.append(snapped)
        seed_ids.append(rid)

    if not seed_xy:
        return {}  # nothing to do

    # Partition the room mask among seeds (multi-source BFS)
    label_img = _partition_room_mask_by_seeds(room_mask, seed_xy)
    # Map each seed label -> per-room binary mask
    per_room_masks = {}
    for i, rid in enumerate(seed_ids, start=1):
        per_room_masks[rid] = (label_img == i).astype(np.uint8)

    return per_room_masks


# ---- previous flood helpers (fallback when no segmented map is provided) ----

def _build_room_masks_by_flood(image_bgr, main_rooms, point_step=10, flood_threshold=30):
    """
    Legacy: flood-fill around each room seed independently (can leak through doors!).
    Returns dict: room_id -> mask (H x W uint8 0/255).
    Uses radius-based seed finding to avoid text annotations.
    """
    H, W = image_bgr.shape[:2]
    base = image_bgr.copy()
    masks = {}

    def is_valid_room_seed_flood(seed_x, seed_y, min_area_threshold=200):
        """Test if a seed point is in a valid room area (not text annotation)."""
        if not (0 <= seed_x < W and 0 <= seed_y < H):
            return False
        
        if not _is_white(base[seed_y, seed_x]):
            return False
        
        # Perform a small test flood fill to check if this is a large enough region
        test_image = base.copy()
        test_mask = np.zeros((H + 2, W + 2), np.uint8)
        try:
            _, _, _, rect = cv2.floodFill(
                test_image, test_mask, (seed_x, seed_y), (128, 128, 128),
                loDiff=(10, 10, 10), upDiff=(10, 10, 10),
                flags=cv2.FLOODFILL_MASK_ONLY
            )
            filled_area = int(test_mask[1:-1, 1:-1].astype(bool).sum())
            return filled_area >= min_area_threshold
        except Exception:
            return False

    def find_valid_seeds_flood(center_x, center_y, max_radius=150, min_radius=20, step_radius=10, angle_step=30):
        """Helper to find valid seeds avoiding text annotations with area-based validation."""
        valid_seeds = []
        cx = max(0, min(int(center_x), W - 1))
        cy = max(0, min(int(center_y), H - 1))
        
        # Check center first, but validate it's in a room area
        if is_valid_room_seed_flood(cx, cy):
            valid_seeds.append((cx, cy))
        
        # Search in concentric circles, starting further out to avoid text
        for radius in range(min_radius, max_radius + 1, step_radius):
            if len(valid_seeds) >= 12:
                break
            for angle in range(0, 360, angle_step):
                radian = np.radians(angle)
                px = int(cx + radius * np.cos(radian))
                py = int(cy + radius * np.sin(radian))
                if 0 <= px < W and 0 <= py < H:
                    if is_valid_room_seed_flood(px, py):
                        if (px, py) not in valid_seeds:
                            valid_seeds.append((px, py))
                            if len(valid_seeds) >= 12:
                                break
            if len(valid_seeds) >= 12:
                break
        
        # Fallback with lower threshold
        if not valid_seeds:
            for radius in range(min_radius, max_radius + 1, step_radius):
                for angle in range(0, 360, angle_step):
                    radian = np.radians(angle)
                    px = int(cx + radius * np.cos(radian))
                    py = int(cy + radius * np.sin(radian))
                    if 0 <= px < W and 0 <= py < H:
                        if _is_white(base[py, px]):
                            if is_valid_room_seed_flood(px, py, min_area_threshold=50):
                                valid_seeds.append((px, py))
                                if len(valid_seeds) >= 4:
                                    break
                if len(valid_seeds) >= 4:
                    break
        
        # Last resort
        if not valid_seeds:
            valid_seeds = [(cx, cy)]
        
        return valid_seeds

    for room_id, (x, y) in main_rooms:
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))

        visited = np.zeros((H, W), dtype=np.uint8)
        # Use radius-based seed finding with area validation to avoid text annotations
        seeds = find_valid_seeds_flood(x, y, max_radius=max(150, flood_threshold * 5),
                                      min_radius=max(20, flood_threshold), step_radius=10, angle_step=point_step)
        # Also validate and add original flood_threshold points if they're valid room seeds
        for ang in range(0, 360, point_step):
            rad = np.deg2rad(ang)
            sx = int(x + flood_threshold * np.cos(rad))
            sy = int(y + flood_threshold * np.sin(rad))
            if 0 <= sx < W and 0 <= sy < H:
                if is_valid_room_seed_flood(sx, sy):
                    if (sx, sy) not in seeds:
                        seeds.append((sx, sy))

        for sx, sy in seeds:
            if visited[sy, sx] == 255:
                continue
            mask = np.zeros((H + 2, W + 2), np.uint8)
            _, _, _, rect = cv2.floodFill(
                base, mask, (sx, sy), (0, 0, 255),
                loDiff=(10, 10, 10), upDiff=(10, 10, 10)
            )
            y0, x0 = rect[1], rect[0]
            y1, x1 = y0 + rect[3], x0 + rect[2]
            sub = mask[1:-1, 1:-1]
            visited[y0:y1, x0:x1] = np.where(sub[y0:y1, x0:x1] != 0, 255, visited[y0:y1, x0:x1])

        masks[room_id] = visited
    return masks


# ---- main API ---------------------------------------------------------------

def get_room_subnode_candidates(
    image_path,
    graph,
    results_dir=None,
    *,
    segmented_map_path=None,          # <<< NEW: pass thr_img_path here for robust partition
    room_rgb=(255, 204, 102),         # room color in the recreated map (RGB)
    room_color_tol=14,
    fill_mode="flood",                # legacy fallback only
    radius_threshold=90,              # legacy "smart" only
    point_radius=10,                  # legacy "smart" only
    point_step=10,                    # legacy "flood"/"smart" seed spacing
    flood_threshold=30,               # legacy "flood" only
    spacing_px=30,
    wall_pad_px=2,
    jitter_px=0,
    save_overlay=True,
    snap_radius=50,                   # seed snapping to global room mask (px)
):
    """
    Compute candidate pixels to densify room interiors (NO graph mutation).

    Preferred path:
      - Provide `segmented_map_path` (from pixelwise_areas) so rooms are partitioned
        *inside the global room mask only* (no hallway leakage).

    Returns:
        candidates (dict): {room_id: [(x,y), ...]}
        overlay_path (str|None): red-dot overlay path
        room_props (dict): {room_id: {area_px, centroid_xy, eq_radius_px, inradius_px, num_candidates}}
        area_map_path (str|None): path to color-mapped room areas with colorbar
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Floorplan image not found at {image_path}")
    H, W = image_bgr.shape[:2]
    image_name_no_ext = os.path.splitext(os.path.basename(image_path))[0]

    # 1) MAIN room nodes
    main_rooms = _room_nodes_from_graph(graph)
    if not main_rooms:
        return {}, None, {}, None

    # 2) Build per-room masks
    if segmented_map_path:
        per_room_masks = _extract_per_room_masks_from_seg(
            segmented_map_path, graph,
            room_rgb=room_rgb, tol=room_color_tol, snap_radius=snap_radius
        )
    else:
        # Fallback to legacy independent floods (less robust)
        per_room_masks = {}
        masks_legacy = _build_room_masks_by_flood(
            image_bgr=image_bgr,
            main_rooms=main_rooms,
            point_step=point_step,
            flood_threshold=flood_threshold,
        )
        # Convert to 0/1 masks
        for rid, m in masks_legacy.items():
            per_room_masks[rid] = (m > 0).astype(np.uint8)

    # 3) Compute candidates & properties
    candidates = {}
    room_props = {}
    areas = []

    for rid, mask in per_room_masks.items():
        if mask is None or mask.sum() == 0:
            continue

        area_px = int(mask.sum())
        areas.append(area_px)

        ys, xs = np.where(mask > 0)
        centroid = (float(xs.mean()), float(ys.mean())) if xs.size else (None, None)

        eq_radius_px = float(math.sqrt(max(area_px, 0) / math.pi)) if area_px > 0 else 0.0
        dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        inradius_px = float(dt.max()) if area_px > 0 else 0.0

        pts = _grid_sample_from_mask(
            mask_uint8=mask.astype(np.uint8),
            spacing_px=spacing_px,
            pad_px=wall_pad_px,
            jitter_px=jitter_px,
        )
        candidates[rid] = pts

        room_props[rid] = {
            "area_px": area_px,
            "centroid_xy": centroid,
            "eq_radius_px": float(eq_radius_px),
            "inradius_px": float(inradius_px),
            "num_candidates": int(len(pts)),
        }

    # 4) Save overlays
    overlay_path = None
    area_map_path = None
    if results_dir and save_overlay:
        plots_dir = os.path.join(results_dir, "Plots")
        out_dir = os.path.join(plots_dir, "room_subnodes", image_name_no_ext)
        _ensure_dir(out_dir)

        # 4a) candidate dot overlay
        overlay = image_bgr.copy()
        for pts in candidates.values():
            for (x, y) in pts:
                cv2.circle(overlay, (x, y), 3, (0, 0, 255), -1)
        overlay_path = os.path.join(out_dir, f"{image_name_no_ext}_room_subnodes.png")
        cv2.imwrite(overlay_path, overlay)

        # 4b) per-room area color map with colorbar
        areas_arr = np.array(areas, dtype=float) if areas else np.array([0.0])
        vmin = float(areas_arr.min())
        vmax = float(max(areas_arr.max(), vmin + 1.0))

        from matplotlib import cm
        from matplotlib.colors import BoundaryNorm

        n_segments = 8
        boundaries = np.linspace(vmin, vmax, n_segments + 1)
        cmap = cm.get_cmap("tab10", n_segments)
        norm = BoundaryNorm(boundaries, cmap.N, clip=True)

        overlay_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        for rid, mask in per_room_masks.items():
            if rid not in room_props:
                continue
            a = room_props[rid]["area_px"]
            rgba = np.array(cmap(norm(a)))
            color = (rgba[:3] * 255).astype(np.uint8)
            alpha = int(255 * 0.35)
            idx = mask.astype(bool)
            overlay_rgba[idx, 0:3] = color
            overlay_rgba[idx, 3] = alpha

        base_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pad_width = int(W * 0.15)
        padded_rgb = np.pad(base_rgb, ((0, 0), (0, pad_width), (0, 0)), mode="constant", constant_values=255)
        padded_overlay = np.pad(overlay_rgba, ((0, 0), (0, pad_width), (0, 0)), mode="constant")

        fig_h = max(1, H / 100)
        fig_w = max(1, (W + pad_width) / 100)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
        ax.imshow(padded_rgb)
        ax.imshow(padded_overlay, interpolation="nearest")
        ax.set_axis_off()

        for rid, props in room_props.items():
            cx, cy = props["centroid_xy"]
            if cx is None or cy is None:
                continue
            label = f"A={props['area_px']:,} px²\nr_in={props['inradius_px']:.1f}px"
            ax.text(
                cx, cy, label,
                ha="center", va="center", fontsize=11, color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.65)
            )

        try:
            cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
            cbar.ax.tick_params(labelsize=9, length=3, width=0.8)
            cbar.set_label("Per-room area (px²)", fontsize=10)
            plt.tight_layout(pad=0.0)
        except Exception:
            # Some backends complain with tight_layout + new axes; safe to ignore.
            pass

        area_map_path = os.path.join(out_dir, f"{image_name_no_ext}_room_area_colormap.png")
        fig.savefig(area_map_path, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)

        with open(os.path.join(out_dir, f"{image_name_no_ext}_room_properties.json"), "w") as f:
            json.dump(room_props, f, indent=2)

    return candidates, overlay_path, room_props, area_map_path
