import json
import cv2
import networkx as nx
import numpy as np
from pathlib import Path
import os
import heapq
import math
from collections import deque
import matplotlib.pyplot as plt
from PIL import Image
import random


class BuildingGraph:
    def __init__(self, default_floor="UNKNOWN"):
        """
        Initialize the BuildingGraph with an empty graph.

        Args:
            default_floor (str): Default floor id to use for nodes when not provided.
                                 Outdoors will always be stored as floor="NA".
        """
        self.graph = nx.Graph()
        self.node_types = {"room": [], "door": [], "corridor": [], "outside": [], "transition": []}
        self.default_floor = default_floor  # used if caller doesn't pass floor_id
        self.property = {}

    # --------------------------- Utilities: floors -----------------------------

    def set_default_floor(self, floor_id: str):
        """Update the default floor used for new nodes (non-outdoor)."""
        self.default_floor = floor_id

    def _resolve_floor(self, node_type: str, floor_id):
        """
        Decide which floor to store for this node.
        Outdoors are explicitly tagged as 'NA'.
        """
        if node_type == "outside":
            return "NA"
        return self.default_floor if floor_id is None else floor_id

    # --------------------------------------------------------------------------

    def add_node(self, node_id, node_type, position, pixels=None, floor_id=None):
        """
        Add a node to the graph.

        Args:
            node_id (str): Unique identifier for the node.
            node_type (str): Type of the node ('room', 'door', 'corridor', 'outside').
            position (tuple): (x, y) coordinates of the node.
            pixels (list, optional): List of pixels belonging to the node.
            floor_id (str, optional): Floor id for this node (overrides default).
                                      Outdoors will always be saved as 'NA'.
        """
        if node_type not in self.node_types:
            raise ValueError(
                f"Invalid node type: {node_type}. Must be one of {list(self.node_types.keys())}."
            )

        if pixels is None:
            pixels = []

        node_floor = self._resolve_floor(node_type, floor_id)

        # Add node to the graph
        self.graph.add_node(
            node_id,
            type=node_type,
            position=position,
            pixels=pixels,
            floor=node_floor,
        )

        # Update the node_types dictionary
        self.node_types[node_type].append(node_id)

    def _ensure_edge_metrics(self, overwrite_weight_if_one=True):
        """Backfill `distance` and (optionally) replace default weight 1 with distance."""
        import math
        for u, v, ed in self.graph.edges(data=True):
            pos1 = self.graph.nodes[u].get("position")
            pos2 = self.graph.nodes[v].get("position")

            # compute distance if possible
            dist = ed.get("distance")
            if (dist is None) and (pos1 is not None) and (pos2 is not None):
                try:
                    x1, y1 = float(pos1[0]), float(pos1[1])
                    x2, y2 = float(pos2[0]), float(pos2[1])
                    dist = ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5
                    ed["distance"] = float(dist)
                except Exception:
                    pass

            # upgrade weight if it's missing or the legacy default (1)
            if overwrite_weight_if_one:
                if ("weight" not in ed) or (ed["weight"] in (None, 1, 1.0)):
                    if dist is not None:
                        ed["weight"] = float(dist)

    def add_edge(self, node_id_1, node_id_2, weight=None):
        """
        Add an edge and record BOTH `weight` and geometric `distance`.
        If `weight` is None, default to Euclidean distance (or 1.0 if positions missing).
        """
        import math

        if not self.graph.has_node(node_id_1) or not self.graph.has_node(node_id_2):
            print(f"Warning: attempted to add edge between non-existent nodes "
                f"'{node_id_1}' and '{node_id_2}'. Skipping.")
            return

        pos1 = self.graph.nodes[node_id_1].get("position")
        pos2 = self.graph.nodes[node_id_2].get("position")

        distance = None
        if pos1 is not None and pos2 is not None:
            try:
                x1, y1 = float(pos1[0]), float(pos1[1])
                x2, y2 = float(pos2[0]), float(pos2[1])
                distance = ( (x1 - x2)**2 + (y1 - y2)**2 ) ** 0.5
            except Exception:
                distance = None

        if weight is None:
            weight = float(distance) if distance is not None else 1.0

        # write both attrs
        if self.graph.has_edge(node_id_1, node_id_2):
            self.graph[node_id_1][node_id_2]["weight"] = float(weight)
            self.graph[node_id_1][node_id_2]["distance"] = (None if distance is None else float(distance))
        else:
            self.graph.add_edge(
                node_id_1, node_id_2,
                weight=float(weight),
                distance=(None if distance is None else float(distance)),
            )


    def _to_json_safe(self, x):
        """Make NetworkX attrs JSON-safe (handles numpy, tuples, ndarrays)."""
        import numpy as _np
        if isinstance(x, (int, float, str)) or x is None:
            return x
        if isinstance(x, (list, tuple)):
            return [self._to_json_safe(v) for v in x]
        if isinstance(x, dict):
            return {str(k): self._to_json_safe(v) for k, v in x.items()}
        if isinstance(x, _np.generic):        # e.g., np.int64, np.float32
            return x.item()
        if isinstance(x, _np.ndarray):
            return x.tolist()
        return str(x)  # last-resort fallback

    def _json_sanitize(self, obj):
        """
        Recursively convert NumPy scalars/arrays, tuples, sets, etc. into
        JSON-serializable Python types.
        """
        import numpy as np

        # NumPy scalars -> Python scalars
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)

        # NumPy arrays -> lists
        if isinstance(obj, np.ndarray):
            return [self._json_sanitize(x) for x in obj.tolist()]

        # Containers
        if isinstance(obj, (list, tuple, set)):
            return [self._json_sanitize(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): self._json_sanitize(v) for k, v in obj.items()}

        # Leave JSON-friendly primitives (str, int, float, bool, None) as-is
        return obj
        
    def save_to_json(self, path):
        """
        Save graph to JSON, ensuring every edge has `weight` and `distance`,
        and all attributes are JSON-serializable.
        """
        import json

        # Make sure edges have metrics (fills in `distance`, upgrades `weight` if 1)
        if hasattr(self, "_ensure_edge_metrics"):
            self._ensure_edge_metrics(overwrite_weight_if_one=True)

        data = {"nodes": [], "edges": []}

        # ---- Nodes ----
        for n, d in self.graph.nodes(data=True):
            # prefer unified keys but keep all attrs (sanitized)
            node_entry = {
                "id": n,
                "type": d.get("type") or d.get("node_type"),
                "position": d.get("position"),
                "floor": d.get("floor") or d.get("floor_id"),
            }
            # include remaining attributes
            for k, v in d.items():
                if k not in node_entry:
                    node_entry[k] = v

            data["nodes"].append(self._json_sanitize(node_entry))

        # ---- Edges ----
        for u, v, ed in self.graph.edges(data=True):
            # build base edge payload
            edge_entry = {
                "source": u,
                "target": v,
                "weight": ed.get("weight"),
                "distance": ed.get("distance"),
            }
            # include any additional edge attrs
            for k, v_attr in ed.items():
                if k not in edge_entry:
                    edge_entry[k] = v_attr

            data["edges"].append(self._json_sanitize(edge_entry))

        # Write JSON
        with open(path, "w") as f:
            json.dump(self._json_sanitize(data), f, indent=2)



    def plot_on_image(
        self,
        image_path,
        output_path,
        display_labels=True,
        threshold_radius=20,
        highlight_regions=False,
    ):
        """
        Plot the graph on the given image and save it.

        Args:
            image_path (str): Path to the input image.
            output_path (str): Path to save the plotted image.
            display_labels (bool): If True, display text labels for nodes.
            highlight_regions (bool): If True, draw semi-transparent regions around nodes.
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        overlay = image.copy()  # Create an overlay for transparency

        # BGR colors
        colors = {
            # node fills
            "room": (255, 128, 0),        # Bright Orange
            "door": (0, 204, 102),        # Emerald Green
            "corridor": (255, 102, 255),  # Magenta
            "outside": (204, 51, 51),     # Crimson Red
            "transition": (0, 0, 255),    # RED for stairs/elevator (no differentiation)
            "unknown": (128, 128, 128),   # Gray

            # edge strokes
            "room_edge": (255, 165, 0),      # Lighter Orange
            "corridor_edge": (51, 153, 255), # Medium Blue
            "outside_edge": (255, 0, 0),     # Bright Red
            "transition_edge": (0, 0, 180),  # Deep RED for any edge touching transition
        }

        def _node_color(node_id: str, node_type: str):
            """Pick a color for the node; stairs/elevator both RED."""
            t = (node_type or "").lower()
            if t in ("transition", "tranistion"):
                return colors["transition"]
            return colors.get(t, colors["unknown"])

        def _edge_color(type_u: str, type_v: str):
            """Decide edge color; any transition involvement -> deep RED."""
            u = (type_u or "unknown").lower()
            v = (type_v or "unknown").lower()
            if "outside" in (u, v):
                return colors["outside_edge"]
            if "corridor" in (u, v):
                return colors["corridor_edge"]
            if ("transition" in (u, v)) or ("tranistion" in (u, v)):
                return colors["transition_edge"]
            return colors["room_edge"]

        # Step 1: Highlight regions if enabled
        if highlight_regions:
            for node_id, data in self.graph.nodes(data=True):
                if "position" not in data:
                    continue
                x, y = data["position"]
                node_type = data.get("type", "unknown")
                if node_type in {"room", "door", "corridor", "outside", "transition", "tranistion"}:
                    highlight_color = _node_color(node_id, node_type)
                    cv2.circle(overlay, (int(x), int(y)), threshold_radius, highlight_color, -1)

            # Blend the overlay with the original image for transparency
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Step 2: Plot nodes
        for node_id, data in self.graph.nodes(data=True):
            if "position" not in data:
                continue

            x, y = data["position"]
            node_type = data.get("type", "unknown")
            color = _node_color(node_id, node_type)

            # Size tweaks
            t = (node_type or "").lower()
            if t == "corridor":
                radius = 4
            elif t in ("transition", "tranistion"):
                radius = 9  # a touch larger for visibility
            else:
                radius = 8

            cv2.circle(image, (int(x), int(y)), radius, color, -1)

            if display_labels and t != "corridor":
                cv2.putText(
                    image,
                    str(node_id),
                    (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        # Step 3: Plot edges
        for u, v in self.graph.edges():
            if not self.graph.has_node(u) or not self.graph.has_node(v):
                continue
            if "position" not in self.graph.nodes[u] or "position" not in self.graph.nodes[v]:
                continue

            pos_u = self.graph.nodes[u]["position"]
            pos_v = self.graph.nodes[v]["position"]
            type_u = self.graph.nodes[u].get("type", "unknown")
            type_v = self.graph.nodes[v].get("type", "unknown")

            edge_color = _edge_color(type_u, type_v)

            cv2.line(
                image,
                (int(pos_u[0]), int(pos_u[1])),
                (int(pos_v[0]), int(pos_v[1])),
                edge_color,
                2,
            )

        # Save the image
        cv2.imwrite(output_path, image)
        print(f"Graph plotted and saved to {output_path}")



    def add_door_nodes(
        self,
        exit_dbboxes,
        corridor2corridor_dbboxes,
        room2corridor_dbboxes,
        room2room_dbboxes,
        floor_id=None,
    ):
        """
        Adds door nodes to the graph based on bounding boxes, with unique IDs reflecting the door type.

        Args:
            exit_dbboxes (list): List of bounding boxes for exit doors.
            corridor2corridor_dbboxes (list): List of bounding boxes for corridor-to-corridor doors.
            room2corridor_dbboxes (list): List of bounding boxes for room-to-corridor doors.
            room2room_dbboxes (list): List of bounding boxes for room-to-room doors.
            floor_id (str, optional): Floor id to assign to all created door nodes (if provided).
        """
        # Define door types and their corresponding bounding boxes
        door_types = [
            ("exit", exit_dbboxes),
            ("c2c", corridor2corridor_dbboxes),
            ("r2c", room2corridor_dbboxes),
            ("r2r", room2room_dbboxes),
        ]

        # Initialize counters for each door type
        node_counters = {door_type: 1 for door_type, _ in door_types}

        for door_type, dbboxes in door_types:
            centers = []
            for bbox in dbboxes:
                x_center = (bbox[0] + bbox[2]) // 2
                y_center = (bbox[1] + bbox[3]) // 2
                centers.append((x_center, y_center))

            for x, y in centers:
                node_id = f"{door_type}_door_{node_counters[door_type]}"
                self.add_node(node_id, "door", (x, y), floor_id=floor_id)
                node_counters[door_type] += 1

            print(f"{len(centers)}-{door_type} door nodes added!")

    def make_room_door_edges(self, image_path, bboxes):
        """
        Associate door bboxes to MAIN rooms via flood-overlap, then for each MAIN room
        choose ONE anchor door (closest to the main room). Create exactly ONE edge from
        that door to the CLOSEST node in the whole family (main or subnode). Do NOT create
        direct edges to all subnodes; the rest will connect via shortest paths later.

        Returns:
            dict: Mapping from bbox tuple -> list of associated MAIN room ids.
        """
        import math

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")

        H, W = image.shape[:2]
        floodfilled_image = image.copy()

        # ---- helpers ----
        def _is_room(d): return d.get("type") == "room"
        def _is_sub(n, d): return _is_room(d) and (d.get("is_subnode", False) or "_subnode_" in str(n))
        def _parent(n, d):
            if not _is_sub(n, d): return None
            p = d.get("parent_room_id")
            if p: return p
            s = str(n)
            return s.split("_subnode_")[0] if "_subnode_" in s else None
        def _pos(n): return self.graph.nodes[n].get("position")

        # MAIN rooms only for flood association
        main_rooms = [(nid, d["position"]) for nid, d in self.graph.nodes(data=True)
                    if _is_room(d) and not _is_sub(nid, d) and "position" in d]

        door_nodes = [(nid, d["position"]) for nid, d in self.graph.nodes(data=True)
                    if d.get("type") == "door" and "position" in d]
        if not door_nodes:
            print("No door nodes present; skipping room↔door association.")
            return {}

        door_center_to_id = {(int(px), int(py)): did for did, (px, py) in door_nodes}

        # ---- flood per main room to get interior pixels ----
        flooded_pixels = {}  # room_id -> set[(x,y)]
        point_step = 90
        seed_r = 10
        for rid, (x, y) in main_rooms:
            x, y = int(x), int(y)
            x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
            visited = set()
            seed_pts = [(x, y)]
            for ang in range(0, 360, point_step):
                rad = np.radians(ang)
                sx = int(x + seed_r * np.cos(rad))
                sy = int(y + seed_r * np.sin(rad))
                if 0 <= sx < W and 0 <= sy < H:
                    seed_pts.append((sx, sy))
            for sx, sy in seed_pts:
                if (sx, sy) in visited: continue
                mask = np.zeros((H + 2, W + 2), np.uint8)
                _, _, _, rect = cv2.floodFill(
                    floodfilled_image, mask, (sx, sy),
                    (0, 0, 255), loDiff=(10,10,10), upDiff=(10,10,10)
                )
                y0, x0 = max(rect[1], 0), max(rect[0], 0)
                y1, x1 = min(rect[1]+rect[3], H), min(rect[0]+rect[2], W)
                for py in range(y0, y1):
                    for px in range(x0, x1):
                        if mask[py+1, px+1] != 0:
                            visited.add((px, py))
            flooded_pixels[rid] = visited

        # ---- associate bboxes to rooms ----
        bbox_to_room = {}
        room_to_doors = {rid: set() for rid, _ in main_rooms}

        def _doors_in_bbox(x1, y1, x2, y2):
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            did = door_center_to_id.get((int(cx), int(cy)))
            if did: return [did]
            found = []
            for did2, (dx, dy) in door_nodes:
                if x1 <= int(dx) <= x2 and y1 <= int(dy) <= y2:
                    found.append(did2)
            return found

        for (x1, y1, x2, y2) in bboxes:
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(W - 1, x2), min(H - 1, y2)
            if x2c < x1c or y2c < y1c: continue

            bbox_pixels = {(x, y) for x in range(x1c, x2c+1) for y in range(y1c, y2c+1)}
            associated = []
            for rid, pixset in flooded_pixels.items():
                if pixset & bbox_pixels:
                    associated.append(rid)
            if not associated: continue

            bbox_to_room[(x1, y1, x2, y2)] = associated
            dids = _doors_in_bbox(x1c, y1c, x2c, y2c)
            if not dids:
                print(f"Warning: door bbox {(x1,y1,x2,y2)} matched rooms {associated} but no door node found.")
                continue
            for rid in associated: 
                for did in dids:
                    room_to_doors[rid].add(did)

        # ---- for each room: pick ONE anchor door, connect that door to CLOSEST family node only ----
        for rid, _ in main_rooms:
            dids = list(room_to_doors.get(rid, []))
            if not dids:
                continue

            # choose nearest door to main room
            rx, ry = _pos(rid)
            best_door, best_d = None, float("inf")
            for did in dids:
                dx, dy = _pos(did)
                d = math.hypot(rx - dx, ry - dy)
                if d < best_d: best_d, best_door = d, did
            if best_door is None: 
                continue

            # find the CLOSEST node in the whole family to that door (main or any sub)
            family = [rid] + [n for n, d in self.graph.nodes(data=True)
                            if _is_sub(n, d) and _parent(n, d) == rid and "position" in d]
            dx, dy = _pos(best_door)
            nearest_node, nearest_dist = None, float("inf")
            for nid in family:
                sx, sy = _pos(nid)
                d = math.hypot(sx - dx, sy - dy)
                if d < nearest_dist:
                    nearest_dist, nearest_node = d, nid

            # create exactly ONE edge: door <-> nearest family node (if missing)
            if nearest_node is not None and not self.graph.has_edge(nearest_node, best_door):
                self.graph.add_edge(nearest_node, best_door, weight=float(nearest_dist))

            # store anchor for downstream
            self.graph.nodes[rid]["anchor_door"] = best_door

        print("Doors associated: each room has at most one door edge to its closest family node.")
        return bbox_to_room


    def add_corridor_nodes(self, image_path, corridor_pixels, test_img_dir, dest="corridor", distance=20):
        """
        Processes an image to overlay corridor pixels, create a wall mask, buffer the wall mask,
        identify invalid pixels, refine the corridor pixel list, and select pixels based on a grid step.
        Adds selected pixels to the graph and constructs grid-style edges with cross-diagonals.

        Args:
            image_path (str): Path to the input image.
            corridor_pixels (list): List of (y, x) coordinates representing corridor pixels.
            test_img_dir (str): Directory to save the output images.
            distance (int): Minimum distance between selected pixels (grid step size).

        Returns:
            list[(y,x)]: selected pixels
        """
        # Verify the image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")

        # Create the test image directory if it doesn't exist
        os.makedirs(test_img_dir, exist_ok=True)

        # 1. Plot the corridor pixels on the image (corridor pixels are in (y, x) format)
        corridor_overlay_image = image.copy()
        for y, x in corridor_pixels:
            cv2.circle(corridor_overlay_image, (x, y), 1, (0, 255, 0), -1)  # Green dots for corridor pixels
        if dest == "corridor":
            corridor_overlay_path = os.path.join(test_img_dir, "corridor_pixel_overlay.png")
        else:
            corridor_overlay_path = os.path.join(test_img_dir, "outside_pixel_overlay.png")
        cv2.imwrite(corridor_overlay_path, corridor_overlay_image)

        # 2. Threshold the input image at 240 to create a binary wall mask
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, wall_mask = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)

        # Invert the wall mask
        inverted_wall_mask = cv2.bitwise_not(wall_mask)
        if dest == "corridor":
            inverted_wall_mask_path = os.path.join(test_img_dir, "corridor_inverted_wall_mask.png")
        else:
            inverted_wall_mask_path = os.path.join(test_img_dir, "outside_inverted_wall_mask.png")
        cv2.imwrite(inverted_wall_mask_path, inverted_wall_mask)

        # 3. Buffer the inverted wall mask
        buffered_wall_mask = cv2.dilate(inverted_wall_mask, np.ones((15, 15), np.uint8))
        buffered_wall_mask = cv2.bitwise_not(buffered_wall_mask)
        if dest == "corridor":
            buffered_wall_mask_path = os.path.join(test_img_dir, "corridor_buffered_wall_mask.png")
        else:
            buffered_wall_mask_path = os.path.join(test_img_dir, "outside_buffered_wall_mask.png")
        cv2.imwrite(buffered_wall_mask_path, buffered_wall_mask)
        buffered_wall_mask = cv2.bitwise_not(buffered_wall_mask)

        # 4. Identify invalid pixels and refine the corridor pixel list
        buffered_wall_coords = set(zip(*np.where(buffered_wall_mask == 255)))  # Get (y, x) of wall pixels
        corridor_set = set(map(tuple, corridor_pixels))  # Convert each [y, x] to (y, x) before creating a set
        invalid_pixels = corridor_set & buffered_wall_coords  # Intersection of wall pixels and corridor pixels
        refined_corridor_pixels = list(corridor_set - invalid_pixels)  # Remove invalids from corridor pixels

        refined_set = set(refined_corridor_pixels)  # Convert to set for fast lookup
        selected_pixels = []

        # Generate grid points and filter them
        for y in range(0, image.shape[0], distance):
            for x in range(0, image.shape[1], distance):
                if (y, x) in refined_set:  # Keep only points that exist in refined_corridor_pixels
                    selected_pixels.append((y, x))

        # 8. Save refined corridor pixels image
        refined_corridor_image = image.copy()
        for y, x in invalid_pixels:
            cv2.circle(refined_corridor_image, (x, y), 1, (0, 0, 255), -1)  # Red for invalid pixels
        for y, x in refined_corridor_pixels:
            cv2.circle(refined_corridor_image, (x, y), 1, (0, 255, 0), -1)  # Green for valid pixels
        if dest == "corridor":
            refined_corridor_path = os.path.join(test_img_dir, "refined_corridor_pixels.png")
        else:
            refined_corridor_path = os.path.join(test_img_dir, "refined_outside_pixels.png")
        cv2.imwrite(refined_corridor_path, refined_corridor_image)

        # 9. Save selected pixel map image
        selected_pixel_image = image.copy()
        for y, x in selected_pixels:
            cv2.circle(selected_pixel_image, (x, y), 4, (139, 0, 139), -1)  # Dark blue for selected pixels
        if dest == "corridor":
            selected_pixel_map_path = os.path.join(test_img_dir, "selected_corridor_pixel_map.png")
        else:
            selected_pixel_map_path = os.path.join(test_img_dir, "selected_outside_pixel_map.png")
        cv2.imwrite(selected_pixel_map_path, selected_pixel_image)
        return selected_pixels

    def add_corridor_edges(self, selected_pixels, distance=20):
        """
        Adds corridor edges based on the selected pixels and grid distance.

        Args:
            selected_pixels (list): List of (y, x) coordinates representing corridor pixels.
            distance (int): The grid step size for connecting nodes.
        """
        # Step 2: Add edges between corridor nodes
        selected_pixel_positions = {(y, x): f"corridor_connect_{i + 1}" for i, (y, x) in enumerate(selected_pixels)}
        for y, x in selected_pixel_positions.keys():
            node_id = selected_pixel_positions[(y, x)]

            # Define neighbor offsets (horizontal, vertical, and diagonal)
            neighbors = [
                (y + distance, x),  # Down
                (y - distance, x),  # Up
                (y, x + distance),  # Right
                (y, x - distance),  # Left
                (y + distance, x + distance),  # Bottom-right diagonal
                (y - distance, x - distance),  # Top-left diagonal
                (y + distance, x - distance),  # Bottom-left diagonal
                (y - distance, x + distance),  # Top-right diagonal
            ]

            # Add edges if the neighbor exists in the grid AND both nodes already exist
            for ny, nx in neighbors:
                if (ny, nx) in selected_pixel_positions:
                    neighbor_id = selected_pixel_positions[(ny, nx)]
                    if self.graph.has_node(node_id) and self.graph.has_node(neighbor_id):
                        self.graph.add_edge(node_id, neighbor_id)

    def add_outdoor_edges(self, outdoor_pixels, distance=20):
        """
        Adds outdoor edges based on the selected pixels and grid distance.

        Args:
            outdoor_pixels (list): List of (y, x) coordinates representing outdoor pixels.
            distance (int): The grid step size for connecting nodes.
        """
        selected_pixel_positions = {(y, x): f"outside_connect_{i + 1}" for i, (y, x) in enumerate(outdoor_pixels)}
        for y, x in selected_pixel_positions.keys():
            node_id = selected_pixel_positions[(y, x)]

            neighbors = [
                (y + distance, x),  # Down
                (y - distance, x),  # Up
                (y, x + distance),  # Right
                (y, x - distance),  # Left
                (y + distance, x + distance),  # Bottom-right diagonal
                (y - distance, x - distance),  # Top-left diagonal
                (y + distance, x - distance),  # Bottom-left diagonal
                (y - distance, x + distance),  # Top-right diagonal
            ]

            for ny, nx in neighbors:
                if (ny, nx) in selected_pixel_positions:
                    neighbor_id = selected_pixel_positions[(ny, nx)]
                    if self.graph.has_node(node_id) and self.graph.has_node(neighbor_id):
                        self.graph.add_edge(node_id, neighbor_id)

    def connect_hallways(self):
        print("Connecting hallways...")

        # Find all corridor main nodes
        corridor_main_nodes = [
            node for node, data in self.graph.nodes(data=True)
            if data.get('type') == 'corridor' and str(node).startswith("corridor_main")
        ]

        # Find all corridor connect nodes
        corridor_connect_nodes = {
            node for node, data in self.graph.nodes(data=True)
            if data.get('type') == 'corridor' and str(node).startswith("corridor_connect")
        }

        # Find all outside main nodes
        outside_main_nodes = [
            node for node, data in self.graph.nodes(data=True)
            if data.get('type') == 'outside' and str(node).startswith("outside_main")
        ]

        # Find all outside connect nodes
        outside_connect_nodes = {
            node for node, data in self.graph.nodes(data=True)
            if data.get('type') == 'outside' and str(node).startswith("outside_connect")
        }

        # Euclidean distance on node positions
        def euclidean_distance(node_1, node_2):
            pos_1 = self.graph.nodes[node_1].get('position', [0, 0])
            pos_2 = self.graph.nodes[node_2].get('position', [0, 0])
            return math.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)

        # Radius within which we look for corridor_connect or outside_connect nodes
        radius = 200

        # For each main corridor node, connect to up to 4 nearby corridor_connect nodes
        for main_node in corridor_main_nodes:
            nearby = []
            for connect_node in corridor_connect_nodes:
                dist = euclidean_distance(main_node, connect_node)
                if dist <= radius:
                    nearby.append((dist, connect_node))
            if nearby:
                for dist, connect_node in sorted(nearby, key=lambda x: x[0])[:4]:
                    if not self.graph.has_edge(main_node, connect_node):
                        self.add_edge(main_node, connect_node, weight=dist)
            else:
                print(f"No corridor_connect nodes found within radius of {main_node}")

        # For each main outside node, connect to up to 4 nearby outside_connect nodes
        for main_node in outside_main_nodes:
            nearby = []
            for connect_node in outside_connect_nodes:
                dist = euclidean_distance(main_node, connect_node)
                if dist <= radius:
                    nearby.append((dist, connect_node))
            if nearby:
                for dist, connect_node in sorted(nearby, key=lambda x: x[0])[:4]:
                    if not self.graph.has_edge(main_node, connect_node):
                        self.add_edge(main_node, connect_node, weight=dist)
            else:
                print(f"No outside_connect nodes found within radius of {main_node}")

        print(f"Added edges to {len(corridor_main_nodes)} corridor main nodes.")
        print(f"Added edges to {len(outside_main_nodes)} outside main nodes.")


    def connect_doors(self):
        print("\nConnecting doors...")

        # Door nodes by category
        exit_doors = [
            node for node, data in self.graph.nodes(data=True)
            if data.get('type') == 'door' and str(node).startswith("exit_door")
        ]
        c2c_doors = [
            node for node, data in self.graph.nodes(data=True)
            if data.get('type') == 'door' and str(node).startswith("c2c_door")
        ]
        r2c_doors = [
            node for node, data in self.graph.nodes(data=True)
            if data.get('type') == 'door' and str(node).startswith("r2c_door")
        ]

        # Corridor/outside connectivity targets
        outside_connect_nodes = {
            node for node, data in self.graph.nodes(data=True)
            if data.get('type') == 'outside' and str(node).startswith("outside_connect")
        }
        corridor_connect_nodes = {
            node for node, data in self.graph.nodes(data=True)
            if data.get('type') == 'corridor' and str(node).startswith("corridor_connect")
        }
        corridor_main_nodes = {
            node for node, data in self.graph.nodes(data=True)
            if data.get('type') == 'corridor' and str(node).startswith("corridor_main")
        }

        def euclidean_distance(node_1, node_2):
            pos_1 = self.graph.nodes[node_1].get('position', [0, 0])
            pos_2 = self.graph.nodes[node_2].get('position', [0, 0])
            return math.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)

        radius = 100

        # Exit doors: connect to nearest outside_connect and corridor_connect
        for door in exit_doors:
            nearby_out = []
            nearby_cor = []
            for connect_node in outside_connect_nodes:
                dist = euclidean_distance(door, connect_node)
                if dist <= radius:
                    nearby_out.append((dist, connect_node))
            for connect_node in corridor_connect_nodes:
                dist = euclidean_distance(door, connect_node)
                if dist <= radius:
                    nearby_cor.append((dist, connect_node))

            if nearby_out:
                dist, cn = min(nearby_out, key=lambda x: x[0])
                if not self.graph.has_edge(door, cn):
                    self.add_edge(door, cn, weight=dist)
            else:
                print(f"No outside_connect nodes found within radius of {door}")

            if nearby_cor:
                dist, cn = min(nearby_cor, key=lambda x: x[0])
                if not self.graph.has_edge(door, cn):
                    self.add_edge(door, cn, weight=dist)
            else:
                print(f"No corridor_connect nodes found within radius of {door}")

        # c2c doors: connect to up to 4 nearest corridor_connect nodes
        for door in c2c_doors:
            nearby = []
            for connect_node in corridor_connect_nodes:
                dist = euclidean_distance(door, connect_node)
                if dist <= radius:
                    nearby.append((dist, connect_node))
            if nearby:
                for dist, cn in sorted(nearby, key=lambda x: x[0])[:4]:
                    if not self.graph.has_edge(door, cn):
                        self.add_edge(door, cn, weight=dist)
            else:
                print(f"No corridor_connect nodes found within radius of {door}")

        # r2c doors: connect to nearest of corridor_connect or corridor_main
        for door in r2c_doors:
            nearby = []
            for cn in corridor_connect_nodes:
                dist = euclidean_distance(door, cn)
                if dist <= radius:
                    nearby.append((dist, cn))
            for mn in corridor_main_nodes:
                dist = euclidean_distance(door, mn)
                if dist <= radius:
                    nearby.append((dist, mn))
            if nearby:
                dist, tgt = min(nearby, key=lambda x: x[0])
                if not self.graph.has_edge(door, tgt):
                    self.add_edge(door, tgt, weight=dist)
            else:
                print(f"No corridor_connect or corridor_main nodes found within radius of {door}")


    def connect_rooms(self):
        print("\nConnecting rooms...")

        # Helper: is this a room subnode?
        def _is_room_subnode(node_id, data):
            return data.get('type') == 'room' and (data.get('is_subnode', False) or "_subnode_" in str(node_id))

        # Only MAIN room nodes (exclude densified subnodes)
        room_nodes = [
            node for node, data in self.graph.nodes(data=True)
            if data.get('type') == 'room' and not _is_room_subnode(node, data)
        ]

        corridor_main_nodes = {
            node for node, data in self.graph.nodes(data=True)
            if data.get('type') == 'corridor' and str(node).startswith("corridor_main")
        }
        corridor_connect_nodes = {
            node for node, data in self.graph.nodes(data=True)
            if data.get('type') == 'corridor' and str(node).startswith("corridor_connect")
        }

        def euclidean_distance(node_1, node_2):
            pos_1 = self.graph.nodes[node_1].get('position', [0, 0])
            pos_2 = self.graph.nodes[node_2].get('position', [0, 0])
            return math.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)

        radius = 400
        disconnected_room_count = 0

        for room in room_nodes:
            # If this main room has no neighbors, try to tie it to the corridor graph
            if len(list(self.graph.neighbors(room))) == 0:
                disconnected_room_count += 1

                nearby = []
                for cn in corridor_main_nodes:
                    dist = euclidean_distance(room, cn)
                    if dist <= radius:
                        nearby.append((dist, cn))
                for cn in corridor_connect_nodes:
                    dist = euclidean_distance(room, cn)
                    if dist <= radius:
                        nearby.append((dist, cn))

                if nearby:
                    dist, closest = min(nearby, key=lambda x: x[0])
                    if not self.graph.has_edge(room, closest):
                        self.add_edge(room, closest, weight=dist)
                else:
                    print(f"No corridor nodes found within radius of {room}")

        print(f"Total disconnected main room nodes: {disconnected_room_count}")


    def merge_nearby_nodes(self, threshold_room=50, threshold_door=30):
        """
        Merge nodes that are within a certain vicinity threshold, with different thresholds
        for "room" and "door" nodes.
        """
        # Build a list of node IDs and their positions
        node_ids = list(self.graph.nodes)
        positions = {node_id: self.graph.nodes[node_id]["position"] for node_id in node_ids}

        # Separate nodes by type to apply different thresholds
        room_nodes = [node_id for node_id in node_ids if self.graph.nodes[node_id].get("type") == "room"]
        door_nodes = [node_id for node_id in node_ids if self.graph.nodes[node_id].get("type") == "door"]

        # Initialize Union-Find structure for nodes
        parent = {node_id: node_id for node_id in room_nodes + door_nodes}

        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u, v):
            pu, pv = find(u), find(v)
            if pu != pv:
                parent[pv] = pu

        # Merging room nodes
        for i in range(len(room_nodes)):
            node_id_1 = room_nodes[i]
            pos_1 = np.array(positions[node_id_1])
            for j in range(i + 1, len(room_nodes)):
                node_id_2 = room_nodes[j]
                pos_2 = np.array(positions[node_id_2])
                dist = np.linalg.norm(pos_1 - pos_2)
                if dist < threshold_room:
                    union(node_id_1, node_id_2)

        # Merging door nodes
        for i in range(len(door_nodes)):
            node_id_1 = door_nodes[i]
            pos_1 = np.array(positions[node_id_1])
            for j in range(i + 1, len(door_nodes)):
                node_id_2 = door_nodes[j]
                pos_2 = np.array(positions[node_id_2])
                dist = np.linalg.norm(pos_1 - pos_2)
                if dist < threshold_door:
                    union(node_id_1, node_id_2)

        # Group nodes by their representative parent node
        clusters = {}
        for node_id in room_nodes + door_nodes:
            p = find(node_id)
            clusters.setdefault(p, []).append(node_id)

        # Merge nodes in each cluster
        for cluster_nodes in clusters.values():
            if len(cluster_nodes) > 1:
                positions_list = [positions[node_id] for node_id in cluster_nodes]
                avg_position = tuple(map(int, np.mean(positions_list, axis=0)))

                # Keep the first node as the main node
                main_node = cluster_nodes[0]
                self.graph.nodes[main_node]["position"] = avg_position

                # Remove other nodes from the graph and node_types
                for node_id in cluster_nodes[1:]:
                    self.graph.remove_node(node_id)

                    for node_type, node_list in self.node_types.items():
                        if node_id in node_list:
                            node_list.remove(node_id)

                print(f"Merged nodes {cluster_nodes} into {main_node} at {avg_position}")

        return self.graph

    def _ensure_edge_weights(self):
        """
        Ensure every edge has a numeric 'weight' for shortest-path queries.
        If missing, use Euclidean distance between node positions; fall back to 1.0.
        """
        import math
        for u, v, data in self.graph.edges(data=True):
            if "weight" not in data or data["weight"] is None:
                pu = self.graph.nodes[u].get("position")
                pv = self.graph.nodes[v].get("position")
                if pu is not None and pv is not None:
                    w = math.hypot(float(pu[0]) - float(pv[0]), float(pu[1]) - float(pv[1]))
                else:
                    w = 1.0
                data["weight"] = float(w)

    def connect_all_rooms(self, input_path, graph_img_dir):
        """
        PRUNING with guarantees:

        1) For each room family (main + subnodes):
        - Ensure an anchor exists: use main's 'anchor_door' if present; otherwise
        add ONE edge from the nearest corridor (same floor preferred) to the
        closest family node.
        - Keep the union of SHORTEST PATHS (legal, weighted) from every family
        node (main + subs) to that anchor. Only INTRA-FAMILY edges on those
        paths are kept; corridor/door segments used by those paths are kept too.

        2) Global connectivity/pruning:
        - Keep the union of SHORTEST PATHS between every pair of MAIN room nodes.
        - For each EXIT DOOR, keep the shortest path from the closest MAIN room.
        - NEW: For each TRANSITION node (stairs/elevator), keep the union of SHORTEST
        PATHS from EVERY MAIN room to that transition node (all pairs main→transition).

        3) Prune: remove all nodes/edges not on any kept path.
        4) NEW: Transition nodes are never pruned. After pruning, ensure each transition
        is attached to its nearest remaining corridor (fallback connection), if not already.
        """
        import math, random
        import numpy as np
        import networkx as nx
        from PIL import Image
        import matplotlib.pyplot as plt

        # ---------- helpers ----------
        def _is_room(d): return d.get('type') == 'room'
        def _is_sub(n, d): return _is_room(d) and (d.get('is_subnode', False) or "_subnode_" in str(n))
        def _parent(n, d):
            if not _is_sub(n, d): return None
            p = d.get("parent_room_id")
            if p: return p
            s = str(n)
            return s.split("_subnode_")[0] if "_subnode_" in s else None
        def _pos(n): return self.graph.nodes[n].get('position') if n in self.graph else None
        def _dist(p, q):
            if p is None or q is None: return float('inf')
            return ((p[0]-q[0])**2 + (p[1]-q[1])**2) ** 0.5

        # Make sure weighted shortest paths reflect Euclidean-ish lengths
        self._ensure_edge_weights()

        # ---------- collect entities ----------
        main_rooms = [n for n, d in self.graph.nodes(data=True) if _is_room(d) and not _is_sub(n, d)]
        subrooms   = [n for n, d in self.graph.nodes(data=True) if _is_sub(n, d)]
        corridors  = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'corridor']
        exit_doors = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'door' and str(n).startswith("exit_door")]

        # NEW: strictly 'transition' (no legacy 'tranistion')
        transitions = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'transition']

        room_family = {rid: [rid] for rid in main_rooms}
        for n in subrooms:
            pr = _parent(n, self.graph.nodes[n])
            if pr in room_family:
                room_family[pr].append(n)

        print("\nConnecting all rooms (with pruning to shortest paths)...")
        print(
            f"Totals -> nodes: {len(self.graph.nodes)}, "
            f"main rooms: {len(main_rooms)}, subrooms: {len(subrooms)}, "
            f"corridors: {len(corridors)}, exits: {len(exit_doors)}, transitions: {len(transitions)}"
        )

        # ---------- per-family: ensure anchor & build union of shortest-to-anchor ----------
        family_keep_nodes, family_keep_edges = set(), set()
        corridor_fallback_used = 0
        path_segments_to_plot = []

        for rid, fam in room_family.items():
            fam = [n for n in fam if n in self.graph]
            if not fam:
                continue

            # (a) determine/create anchor
            anchor = self.graph.nodes[rid].get("anchor_door")
            if anchor is not None and anchor not in self.graph:
                anchor = None

            if anchor is None:
                # no door known -> attach nearest corridor to the closest family node
                rid_floor = self.graph.nodes[rid].get('floor')
                same_floor = [c for c in corridors if self.graph.nodes[c].get('floor') == rid_floor]
                candidates = same_floor if same_floor else corridors
                if candidates:
                    best_pair, best_d = None, float('inf')
                    for fn in fam:
                        p = _pos(fn)
                        if p is None: continue
                        for cn in candidates:
                            d = _dist(p, _pos(cn))
                            if d < best_d:
                                best_d, best_pair = d, (fn, cn)
                    if best_pair is not None:
                        fn, cn = best_pair
                        if not self.graph.has_edge(fn, cn):
                            self.graph.add_edge(fn, cn, weight=float(best_d))
                        anchor = cn
                        corridor_fallback_used += 1
                    else:
                        continue
                else:
                    continue

            # (b) union of shortest paths from every family node to anchor
            for n in fam:
                if n == anchor:
                    continue
                try:
                    sp = nx.shortest_path(self.graph, source=n, target=anchor, weight='weight')
                except nx.NetworkXNoPath:
                    # micro-fix: stitch nearest family mate then retry once
                    pn = _pos(n)
                    best_mate, best_d = None, float('inf')
                    for m in fam:
                        if m == n: continue
                        d = _dist(pn, _pos(m))
                        if d < best_d:
                            best_d, best_mate = d, m
                    if best_mate is not None and not self.graph.has_edge(n, best_mate):
                        self.graph.add_edge(n, best_mate, weight=float(best_d))
                        try:
                            sp = nx.shortest_path(self.graph, source=n, target=anchor, weight='weight')
                        except nx.NetworkXNoPath:
                            print(f"[{rid}] no path from {n} to anchor after local fix; skipping this node.")
                            continue
                    else:
                        print(f"[{rid}] no path from {n} to anchor; skipping this node.")
                        continue

                family_keep_nodes.update(sp)
                path_segments_to_plot.append(sp)
                for u, v in zip(sp[:-1], sp[1:]):
                    family_keep_edges.add((u, v) if u < v else (v, u))

        if corridor_fallback_used:
            print(f"Corridor fallback used for {corridor_fallback_used} families lacking doors.")

        # ---------- global: shortest paths between EVERY pair of MAIN rooms ----------
        global_keep_nodes, global_keep_edges = set(), set()
        main_pairs_no_path = 0

        for i, a in enumerate(main_rooms):
            for b in main_rooms[i+1:]:
                try:
                    sp = nx.shortest_path(self.graph, source=a, target=b, weight='weight')
                    global_keep_nodes.update(sp)
                    path_segments_to_plot.append(sp)
                    for u, v in zip(sp[:-1], sp[1:]):
                        global_keep_edges.add((u, v) if u < v else (v, u))
                except nx.NetworkXNoPath:
                    main_pairs_no_path += 1

        if main_pairs_no_path:
            print(f"WARNING: {main_pairs_no_path} main-room pairs had no path before pruning (graph may be fragmented).")

        # ---------- exit doors: keep shortest path from closest MAIN room ----------
        exit_keep_nodes, exit_keep_edges = set(), set()
        for ed in exit_doors:
            best_sp, best_len = None, float('inf')
            for rid in main_rooms:
                try:
                    sp = nx.shortest_path(self.graph, source=rid, target=ed, weight='weight')
                    if len(sp) < best_len:
                        best_len, best_sp = len(sp), sp
                except nx.NetworkXNoPath:
                    continue
            if best_sp:
                exit_keep_nodes.update(best_sp)
                path_segments_to_plot.append(best_sp)
                for u, v in zip(best_sp[:-1], best_sp[1:]):
                    exit_keep_edges.add((u, v) if u < v else (v, u))

        # ---------- NEW: transitions - keep SHORTEST PATHS from EVERY MAIN room to EVERY transition ----------
        transition_keep_nodes, transition_keep_edges = set(), set()
        trans_pairs_no_path = 0
        for t in transitions:
            for rid in main_rooms:
                try:
                    sp = nx.shortest_path(self.graph, source=rid, target=t, weight='weight')
                    transition_keep_nodes.update(sp)
                    path_segments_to_plot.append(sp)
                    for u, v in zip(sp[:-1], sp[1:]):
                        transition_keep_edges.add((u, v) if u < v else (v, u))
                except nx.NetworkXNoPath:
                    trans_pairs_no_path += 1
        if trans_pairs_no_path:
            print(f"Note: {trans_pairs_no_path} main→transition pairs had no path before pruning.")

        # ---------- build final KEEP sets & prune ----------
        keep_nodes = (
            family_keep_nodes |
            global_keep_nodes |
            exit_keep_nodes |
            transition_keep_nodes |
            set(transitions)  # NEVER prune transitions
        )

        keep_edges = family_keep_edges | global_keep_edges | exit_keep_edges | transition_keep_edges

        # Ensure endpoints of kept edges are kept
        for u, v in list(keep_edges):
            keep_nodes.add(u); keep_nodes.add(v)

        # Remove nodes not in keep
        nodes_to_remove = set(self.graph.nodes) - keep_nodes
        if nodes_to_remove:
            self.graph.remove_nodes_from(nodes_to_remove)

        # Remove edges not in keep (and re-check endpoints)
        edges_to_remove = []
        for u, v in self.graph.edges():
            e = (u, v) if u < v else (v, u)
            if (u not in keep_nodes) or (v not in keep_nodes) or (e not in keep_edges):
                edges_to_remove.append((u, v))
        if edges_to_remove:
            self.graph.remove_edges_from(edges_to_remove)

        print(f"After pruning -> nodes: {len(self.graph.nodes)}, edges: {len(self.graph.edges)}")

        # ---------- Fallback: attach isolated transitions to nearest remaining corridor ----------
        post_corridors = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'corridor']
        attached = 0
        if post_corridors:
            for tn in transitions:
                if tn not in self.graph:
                    continue
                # If already connected to any corridor, skip
                if any(self.graph.nodes[nbr].get('type') == 'corridor' for nbr in self.graph.neighbors(tn)):
                    continue
                pt = _pos(tn)
                if pt is None:
                    continue
                best_c, best_d = None, float('inf')
                for cn in post_corridors:
                    d = _dist(pt, _pos(cn))
                    if d < best_d:
                        best_d, best_c = d, cn
                if best_c is not None and not math.isinf(best_d):
                    self.graph.add_edge(tn, best_c, weight=float(best_d))
                    path_segments_to_plot.append([tn, best_c])
                    attached += 1
        else:
            print("WARNING: No corridors remain after pruning; transition nodes were kept but not connected.")

        if attached:
            print(f"Transition attachments added post-pruning: {attached}")

        # ---------- visualize kept paths ----------
        try:
            img = Image.open(input_path)
            w, h = img.size
            fig, ax = plt.subplots(figsize=(max(1, w/100), max(1, h/100)), dpi=100)
            ax.imshow(img)
            random.seed(42)
            def _clr(): return (random.random(), random.random(), random.random())
            for sp in path_segments_to_plot:
                coords = []
                for n in sp:
                    if n in self.graph.nodes:
                        p = self.graph.nodes[n].get('position')
                        if p is not None: coords.append(p)
                if len(coords) >= 2:
                    arr = np.array(coords)
                    ax.plot(arr[:, 0], arr[:, 1], color=_clr(), linewidth=2)
            out_path = f"{graph_img_dir}/colored_paths.png"
            plt.axis('off'); plt.savefig(out_path, bbox_inches='tight', pad_inches=0); plt.close()
            print(f"Shortest-path visualization saved: {out_path}")
        except Exception as e:
            print(f"Plotting skipped: {e}")

        return self.graph


    def remove_edges_not_in_web(self, web_nodes):
        # Create a list of edges to remove
        edges_to_remove = [
            (u, v) for u, v in self.graph.edges()
            if u not in web_nodes or v not in web_nodes
        ]
        self.graph.remove_edges_from(edges_to_remove)

    def return_graph_size(self):
        return len(self.graph.nodes)

    def connect_room_family_funnel(self, room_id: str, spacing_px: int = 60, door_selector: str = "nearest") -> int:
        """
        Build a local lattice inside a room (main + subnodes), then keep only
        the intra-room edges that lie on shortest paths to the room's anchor(s).

        Anchors:
        - Doors attached to any family member (preferred).
        - If no doors exist, FALL BACK to the nearest corridor node and use the
            nearest family node to that corridor as the single anchor.

        Returns: number of kept intra-room edges for this family.
        """
        import math
        import networkx as nx

        if room_id not in self.graph:
            return 0

        # -------- collect family (main + subnodes) --------
        def _is_room(d): return d.get("type") == "room"
        def _is_sub(n, d): return _is_room(d) and (d.get("is_subnode", False) or "_subnode_" in str(n))

        family = [room_id]
        for nid, data in self.graph.nodes(data=True):
            if nid == room_id:
                continue
            if not _is_room(data):
                continue
            if _is_sub(nid, data):
                parent = data.get("parent_room_id")
                if parent == room_id or (parent is None and str(nid).startswith(f"{room_id}_subnode_")):
                    family.append(nid)

        # positions
        pos = {}
        for nid in family:
            p = self.graph.nodes[nid].get("position")
            if p is not None:
                pos[nid] = (float(p[0]), float(p[1]))
        family = [nid for nid in family if nid in pos]
        if len(family) <= 1:
            return 0

        # -------- preferred anchors: DOORS attached to ANY family member --------
        ext_to_anchor = []  # list of (external_node_id, anchor_family_node_id)
        for nid in family:
            for nbr in self.graph.neighbors(nid):
                if self.graph.nodes[nbr].get("type") == "door":
                    # choose nearest family node to this door as anchor
                    dp = self.graph.nodes[nbr].get("position")
                    if dp is None:
                        continue
                    dx, dy = float(dp[0]), float(dp[1])
                    anchor = min(family, key=lambda n: math.hypot(pos[n][0] - dx, pos[n][1] - dy))
                    ext_to_anchor.append((nbr, anchor))

        # -------- FALLBACK: nearest CORRIDOR node if no doors --------
        if not ext_to_anchor:
            # choose nearest corridor node to the room family's centroid
            fx = sum(pos[n][0] for n in family) / len(family)
            fy = sum(pos[n][1] for n in family) / len(family)

            best_corr = None
            best_d = float("inf")
            for nid, data in self.graph.nodes(data=True):
                if data.get("type") != "corridor":
                    continue
                cp = data.get("position")
                if cp is None:
                    continue
                d = math.hypot(float(cp[0]) - fx, float(cp[1]) - fy)
                if d < best_d:
                    best_d = d
                    best_corr = nid

            if best_corr is None:
                return 0  # nothing to funnel to

            # anchor is the family node closest to this corridor node
            cpx, cpy = map(float, self.graph.nodes[best_corr]["position"])
            anchor = min(family, key=lambda n: math.hypot(pos[n][0] - cpx, pos[n][1] - cpy))
            ext_to_anchor.append((best_corr, anchor))

        # ensure direct graph edges from each anchor to its external node (door/corridor)
        for ext, anc in ext_to_anchor:
            # add weighted edge if missing
            ep = self.graph.nodes[ext].get("position")
            if ep is None:
                continue
            w = math.hypot(pos[anc][0] - float(ep[0]), pos[anc][1] - float(ep[1]))
            if not self.graph.has_edge(anc, ext):
                self.graph.add_edge(anc, ext, weight=float(w))

        # -------- build local lattice (short edges only) --------
        import numpy as np
        r = max(2.0, float(spacing_px) * 1.25)  # neighbor radius
        temp_edges = set()

        def _eudist(a, b):
            ax, ay = pos[a]; bx, by = pos[b]
            return math.hypot(ax - bx, ay - by)

        # lattice subgraph with only family nodes
        Gf = nx.Graph()
        for n in family:
            Gf.add_node(n)

        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                u, v = family[i], family[j]
                d = _eudist(u, v)
                if d <= r:
                    Gf.add_edge(u, v, weight=d)
                    if not self.graph.has_edge(u, v):
                        self.graph.add_edge(u, v, weight=float(d),
                                            _temp_family_edge=True, _family_owner=room_id)
                    else:
                        ed = self.graph.edges[u, v]
                        ed.setdefault("_temp_family_edge", True)
                        ed["_family_owner"] = room_id
                    temp_edges.add(tuple(sorted((u, v))))

        # -------- compute funnel paths inside the room lattice --------
        keep_edges = set()

        if door_selector == "nearest":
            # precompute SSSP from each anchor node (inside lattice)
            packs = []
            for _, anchor in ext_to_anchor:
                if anchor not in Gf:
                    continue
                dist, paths = nx.single_source_dijkstra(Gf, anchor, weight="weight")
                packs.append((anchor, dist, paths))

            for n in family:
                best_path = None
                best_cost = float("inf")
                for anchor, dist, paths in packs:
                    if n in dist and dist[n] < best_cost:
                        best_cost = dist[n]
                        best_path = paths[n]
                if best_path and len(best_path) > 1:
                    for u, v in zip(best_path[:-1], best_path[1:]):
                        keep_edges.add(tuple(sorted((u, v))))
        else:
            # union to all anchors
            for _, anchor in ext_to_anchor:
                if anchor not in Gf:
                    continue
                dist, paths = nx.single_source_dijkstra(Gf, anchor, weight="weight")
                for n in family:
                    if n in paths and len(paths[n]) > 1:
                        for u, v in zip(paths[n][:-1], paths[n][1:]):
                            keep_edges.add(tuple(sorted((u, v))))

        # -------- prune temporary lattice edges not used by any path --------
        for u, v in list(self.graph.edges()):
            ed = self.graph.edges[u, v]
            if ed.get("_temp_family_edge") and ed.get("_family_owner") == room_id:
                if tuple(sorted((u, v))) not in keep_edges:
                    self.graph.remove_edge(u, v)
                else:
                    ed.pop("_temp_family_edge", None)
                    ed.pop("_family_owner", None)

        kept = len([e for e in keep_edges if e in temp_edges])
        return kept


    def connect_all_families_funnel(self, spacing_px: int = 60, door_selector: str = "nearest") -> int:
        """Run the funnel connector for every MAIN room (excludes subnodes)."""
        total = 0
        for nid, data in self.graph.nodes(data=True):
            if data.get("type") == "room" and not data.get("is_subnode", False) and "_subnode_" not in str(nid):
                total += self.connect_room_family_funnel(nid, spacing_px=spacing_px, door_selector=door_selector)
        return total



    @staticmethod
    def calculate_bbox_centers(bboxes):
        """
        Calculate the centers of bounding boxes from the given list of bounding boxes.

        Args:
            bboxes (list): List of bounding boxes, where each bounding box is represented as a list
                           of 8 coordinates [x1, y1, x2, y2, x3, y3, x4, y4].

        Returns:
            list: List of (x, y) centers for each bounding box.
        """
        bbox_centers = []
        for coordinates in bboxes:
            points = np.array(coordinates).reshape(4, 2)
            center_x = np.mean(points[:, 0])
            center_y = np.mean(points[:, 1])
            bbox_centers.append((center_x, center_y))
        return bbox_centers

    def connect_doors_to_rooms(self):
        """
        Connect doors to their nearest rooms.

        Rules:
        - A door can only connect to one room (nearest room based on distance).
        - A room can have multiple doors.
        """
        door_nodes = [node_id for node_id in self.node_types["door"]]
        room_nodes = [node_id for node_id in self.node_types["room"]]

        if not door_nodes or not room_nodes:
            print("No doors or rooms available to connect.")
            return

        print("\nConnecting doors to rooms...")
        for door_id in door_nodes:
            door_pos = np.array(self.graph.nodes[door_id]["position"])
            nearest_room = None
            min_distance = float("inf")
            for room_id in room_nodes:
                room_pos = np.array(self.graph.nodes[room_id]["position"])
                distance = np.linalg.norm(door_pos - room_pos)
                if distance < min_distance:
                    min_distance = distance
                    nearest_room = room_id

            if nearest_room:
                self.add_edge(door_id, nearest_room)
                print(f"Connected door '{door_id}' to room '{nearest_room}' (distance: {min_distance:.2f})")
