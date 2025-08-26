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
    def __init__(self):
        """
        Initialize the BuildingGraph with an empty graph.
        """
        self.graph = nx.Graph()
        self.node_types = {"room": [], "door": [], "corridor": [], "outside": []}

    def add_node(self, node_id, node_type, position, pixels=None):
        """
        Add a node to the graph.

        Args:
            node_id (str): Unique identifier for the node.
            node_type (str): Type of the node ('room', 'door', 'corridor').
            position (tuple): (x, y) coordinates of the node.
            pixels (list, optional): List of pixels belonging to the node.
        """
        if node_type not in self.node_types:
            raise ValueError(f"Invalid node type: {node_type}. Must be one of {list(self.node_types.keys())}.")
        
        if pixels is None:
            pixels = []  # Default to an empty list if not provided

        # Add node to the graph
        self.graph.add_node(node_id, type=node_type, position=position, pixels=pixels)

        # Update the node_types dictionary
        self.node_types[node_type].append(node_id)

    def add_edge(self, node_id_1, node_id_2, weight=1.0):
        """
        Add an edge between two nodes.

        Args:
            node_id_1 (str): ID of the first node.
            node_id_2 (str): ID of the second node.
            weight (float): Weight of the edge.
        """
        self.graph.add_edge(node_id_1, node_id_2, weight=weight)

    def save_to_json(self, output_path):
        """
        Save the graph to a JSON file.

        Args:
            output_path (str): Path to save the JSON file.
        """
        graph_data = {
            "nodes": [
                {
                    "id": node_id,
                    "type": data["type"],
                    "position": tuple(map(int, data["position"])),  # Convert to Python int
                }
                for node_id, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "weight": 1,  # Ensure weight is a Python float
                }
                for u, v, data in self.graph.edges(data=True)
            ],
        }
        with open(output_path, "w") as f:
            json.dump(graph_data, f, indent=4)

    def plot_on_image(self, image_path, output_path, display_labels=True, threshold_radius=20, highlight_regions=False):
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

        # Define colors for nodes and edges
        colors = {
            "room": (255, 128, 0),         # Bright Orange
            "door": (0, 204, 102),         # Emerald Green
            "corridor": (255, 102, 255),  # Magenta
            "outside": (204, 51, 51),      # Crimson Red
            "unknown": (128, 128, 128),    # Gray for unknown types

            "room_edge": (255, 165, 0),    # Lighter Orange for room-door edges
            "corridor_edge": (51, 153, 255),  # Medium Blue for corridor edges
            "outside_edge": (255, 0, 0),   # Bright Red for outside edges
        }

        # Step 1: Highlight regions if enabled
        if highlight_regions:
            for node_id, data in self.graph.nodes(data=True):
                if "position" not in data:  # Skip nodes without a valid position
                    print(f"Node {node_id} has no position, skipping...")
                    continue

                x, y = data["position"]
                node_type = data.get("type", "unknown")  # Use "unknown" if type is missing
                if node_type in {"room", "door", "corridor", "outside"}:
                    highlight_color = colors.get(node_type, (128, 128, 128))  # Default to gray
                    cv2.circle(overlay, (int(x), int(y)), threshold_radius, highlight_color, -1)

            # Blend the overlay with the original image for transparency
            alpha = 0.3  # Transparency level for highlighting
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Step 2: Plot nodes
        for node_id, data in self.graph.nodes(data=True):
            if "position" not in data:
                print(f"Node {node_id} has no position, skipping...")
                continue

            x, y = data["position"]
            node_type = data.get("type", "unknown")  # Use "unknown" if type is missing
            color = colors.get(node_type, (128, 128, 128))  # Default to gray

            # Adjust the radius based on the node type
            if node_type == "corridor":
                radius = 4  # Smaller radius for corridors
            else:
                radius = 8  # Larger radius for other types

            # Draw the node with the adjusted radius
            cv2.circle(image, (int(x), int(y)), radius, color, -1)

            # Display labels for all node types
            if display_labels:
                if node_type != "corridor":  # Skip labeling corridor nodes
                    cv2.putText(
                        image,
                        node_id,
                        (int(x) + 10, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

        # Step 3: Plot edges
        for u, v in self.graph.edges():
            pos_u = self.graph.nodes[u]["position"]
            pos_v = self.graph.nodes[v]["position"]
            node_type_u = self.graph.nodes[u].get("type", "unknown")
            node_type_v = self.graph.nodes[v].get("type", "unknown")

            # Determine edge color based on node types
            if "outside" in {node_type_u, node_type_v}:
                edge_color = colors["outside_edge"]
            elif "corridor" in {node_type_u, node_type_v}:
                edge_color = colors["corridor_edge"]
            else:
                edge_color = colors["room_edge"]

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

    def add_door_nodes(self, exit_dbboxes, corridor2corridor_dbboxes, room2corridor_dbboxes, room2room_dbboxes):
        """
        Adds door nodes to the graph based on bounding boxes, with unique IDs reflecting the door type.

        Args:
            exit_dbboxes (list): List of bounding boxes for exit doors.
            corridor2corridor_dbboxes (list): List of bounding boxes for corridor-to-corridor doors.
            room2corridor_dbboxes (list): List of bounding boxes for room-to-corridor doors.
            room2room_dbboxes (list): List of bounding boxes for room-to-room doors.
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

            #print(f"\nAdding {door_type} door nodes to the graph...")
            for x, y in centers:
                # Generate a unique node ID for the door node
                node_id = f"{door_type}_door_{node_counters[door_type]}"
                self.add_node(node_id, "door", (x, y))
                node_counters[door_type] += 1

            print(f"{len(centers)}-{door_type} door nodes added!")


    def make_room_door_edges(self, image_path, bboxes):
        """
        Annotates room nodes from a graph on an image, performs floodfill for each room node,
        and associates bounding boxes with room nodes based on floodfilled pixels.

        Args:
            image_path (str): Path to the image.
            bboxes (list): List of bounding boxes, where each bbox is [x1, y1, x2, y2].

        Returns:
            dict: Dictionary mapping bounding boxes to room nodes.
        """
        # Verify that the image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")

        floodfilled_image = image.copy()
        bbox_annotated_image = image.copy()
        room2door_image = image.copy()  # For room-door edges visualization

        # Find all room nodes
        room_nodes = [
            (node_id, data["position"])
            for node_id, data in self.graph.nodes(data=True)
            if data["type"] == "room"
        ]
        #print(f"{(len(room_nodes))}-Room nodes found")
        door_nodes = [
            (node_id, data["position"])
            for node_id, data in self.graph.nodes(data=True)
            if data["type"] == "door"
        ]
        #print(f"{(len(door_nodes))}-Door nodes found")
        # Dictionary to store flooded pixels for each room
        flooded_pixels = {}

        # Annotate and floodfill each room node
        for room_id, position in room_nodes:
            x, y = map(int, position)

            # Flood-fill properties
            point_step = 90  # Angle step for generating seed points
            flood_threshold = 10  # Distance for generating new seeds
            fill_color = tuple(np.random.randint(0, 256, size=3).tolist())
            visited = set()  # Use a set to avoid duplicate pixels
            fill_area = 0

            # Generate initial seed points around the node
            seed_points = [(x, y)]  # Include the node itself
            for angle in range(0, 360, point_step):
                radian = np.radians(angle)
                sx = int(x + flood_threshold * np.cos(radian))
                sy = int(y + flood_threshold * np.sin(radian))

                if 0 <= sx < floodfilled_image.shape[1] and 0 <= sy < floodfilled_image.shape[0]:
                    seed_points.append((sx, sy))

            # Perform flood-filling using seed points
            for sx, sy in seed_points:
                if (sx, sy) in visited:
                    continue

                # Use OpenCV floodFill for each seed
                mask = np.zeros((floodfilled_image.shape[0] + 2, floodfilled_image.shape[1] + 2), np.uint8)
                _, _, _, rect = cv2.floodFill(
                    floodfilled_image,
                    mask,
                    (sx, sy),
                    fill_color,
                    loDiff=(10, 10, 10),
                    upDiff=(10, 10, 10),
                )

                # Add flooded pixels from the rectangle
                for px in range(rect[0], rect[0] + rect[2]):
                    for py in range(rect[1], rect[1] + rect[3]):
                        if (px, py) not in visited:
                            visited.add((px, py))
                            fill_area += 1

            # Store the flooded pixels
            flooded_pixels[room_id] = list(visited)
        # Dictionary to map bounding boxes to room nodes
        bbox_to_room = {}

        # Check bounding boxes
        for idx, (x1, y1, x2, y2) in enumerate(bboxes):
            bbox_pixels = set(
                (x, y)
                for x in range(x1, x2 + 1)
                for y in range(y1, y2 + 1)
            )
            associated_rooms = []  # List to store room IDs connected to this bounding box

            for room_id, room_pixels in flooded_pixels.items():
                if bbox_pixels & set(room_pixels):  # Check for overlap
                    #print(f"BBOX {idx + 1} [{x1}, {y1}, {x2}, {y2}] overlaps with Room Node {room_id}")
                    associated_rooms.append(room_id)  # Add room ID to the list

            if associated_rooms:
                bbox_to_room[(x1, y1, x2, y2)] = associated_rooms  # Store the list of room IDs

        # Draw bounding boxes
        for idx, (x1, y1, x2, y2) in enumerate(bboxes):
            cv2.rectangle(bbox_annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red box
            cv2.putText(bbox_annotated_image, f"BBOX {idx + 1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Compute room-door edges
        for bbox, room_ids in bbox_to_room.items():  # `room_ids` is now a list
            x1, y1, x2, y2 = bbox
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Search for the corresponding door node
            door_found = False
            for door_id, door_position in door_nodes:
                door_x, door_y = map(int, door_position)
                if (center_x, center_y) == (door_x, door_y):
                    door_found = True

                    # Create edges between the door node and all associated room nodes
                    for room_id in room_ids:
                        self.graph.add_edge(room_id, door_id)
                        #print(f"Edge created between Room Node {room_id} and Door Node {door_id}")

                        # Draw the edge on the room2door_image
                        room_x, room_y = map(int, self.graph.nodes[room_id]["position"])
                        cv2.line(room2door_image, (room_x, room_y), (door_x, door_y), (0, 255, 255), 2)  # Yellow line
                    break

            if not door_found:
                print(f"Error: No door node found at center ({center_x}, {center_y}) of BBOX [{x1}, {y1}, {x2}, {y2}]")
        print("All rooms connected to doors..!")

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
            None
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
        if dest=="corridor":
            corridor_overlay_path = os.path.join(test_img_dir, "corridor_pixel_overlay.png")
        else:
            corridor_overlay_path = os.path.join(test_img_dir, "outside_pixel_overlay.png")
        cv2.imwrite(corridor_overlay_path, corridor_overlay_image)

        # 2. Threshold the input image at 240 to create a binary wall mask
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, wall_mask = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)

        # Invert the wall mask
        inverted_wall_mask = cv2.bitwise_not(wall_mask) 
        if dest=="corridor":
            inverted_wall_mask_path = os.path.join(test_img_dir, "corridor_inverted_wall_mask.png")
        else:
            inverted_wall_mask_path = os.path.join(test_img_dir, "outside_inverted_wall_mask.png")
        cv2.imwrite(inverted_wall_mask_path, inverted_wall_mask)

        # 3. Buffer the inverted wall mask
        buffered_wall_mask = cv2.dilate(inverted_wall_mask, np.ones((15, 15), np.uint8))  # Buffer by 5 pixels
        buffered_wall_mask = cv2.bitwise_not(buffered_wall_mask)
        if dest=="corridor":
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

        # Debug print for selected nodes
        #print(f"Selected pixels based on grid step: {len(selected_pixels)}")

        # 8. Save refined corridor pixels image
        refined_corridor_image = image.copy()
        for y, x in invalid_pixels:
            cv2.circle(refined_corridor_image, (x, y), 1, (0, 0, 255), -1)  # Red for invalid pixels
        for y, x in refined_corridor_pixels:
            cv2.circle(refined_corridor_image, (x, y), 1, (0, 255, 0), -1)  # Green for valid pixels
        if dest=="corridor":
            refined_corridor_path = os.path.join(test_img_dir, "refined_corridor_pixels.png")
        else:
            refined_corridor_path = os.path.join(test_img_dir, "refined_outside_pixels.png") 
        cv2.imwrite(refined_corridor_path, refined_corridor_image)

        # 9. Save selected pixel map image
        selected_pixel_image = image.copy()
        for y, x in selected_pixels:
            cv2.circle(selected_pixel_image, (x, y), 4, (139, 0, 139), -1)  # Dark blue for selected pixels
        if dest=="corridor":
            selected_pixel_map_path = os.path.join(test_img_dir, "selected_corridor_pixel_map.png")
        else:
            selected_pixel_map_path = os.path.join(test_img_dir, "selected_outside_pixel_map.png") 
        cv2.imwrite(selected_pixel_map_path, selected_pixel_image)
        return selected_pixels

    def add_corridor_edges(self, selected_pixels, distance=20):
        """
        Adds corridor nodes and edges based on the selected pixels and grid distance.
        
        Args:
            selected_pixels (list): List of (y, x) coordinates representing corridor pixels.
            distance (int): The grid step size for connecting nodes.
        """
        # Step 1: Add corridor nodes to the graph
        # Debugging: Print node counts
        total_nodes = len(self.graph.nodes)
        corridor_nodes = sum(1 for n, d in self.graph.nodes(data=True) if d.get("type") == "corridor")  

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

            # Add edges if the neighbor exists in the grid
            for ny, nx in neighbors:
                if (ny, nx) in selected_pixel_positions:
                    neighbor_id = selected_pixel_positions[(ny, nx)]
                    self.graph.add_edge(node_id, neighbor_id)

    def add_outdoor_edges(self, outdoor_pixels, distance=20):
        """
        Adds corridor nodes and edges based on the selected pixels and grid distance.
        
        Args:
            selected_pixels (list): List of (y, x) coordinates representing corridor pixels.
            distance (int): The grid step size for connecting nodes.
        """
        # Step 1: Add corridor nodes to the graph
        # Debugging: Print node counts
        total_nodes = len(self.graph.nodes)
        corridor_nodes = sum(1 for n, d in self.graph.nodes(data=True) if d.get("type") == "corridor")  

        # Step 2: Add edges between corridor nodes
        selected_pixel_positions = {(y, x): f"outside_connect_{i + 1}" for i, (y, x) in enumerate(outdoor_pixels)}
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

            # Add edges if the neighbor exists in the grid
            for ny, nx in neighbors:
                if (ny, nx) in selected_pixel_positions:
                    neighbor_id = selected_pixel_positions[(ny, nx)]
                    self.graph.add_edge(node_id, neighbor_id)
    
    def connect_hallways(self):
        print("Connecting hallways...")

        # Find all corridor main nodes
        corridor_main_nodes = [node for node, data in self.graph.nodes(data=True) 
                            if data.get('type') == 'corridor' and node.startswith("corridor_main")]

        # Find all corridor connect nodes
        corridor_connect_nodes = {node for node, data in self.graph.nodes(data=True) 
                                if data.get('type') == 'corridor' and node.startswith("corridor_connect")}

        # Find all outside main nodes
        outside_main_nodes = [node for node, data in self.graph.nodes(data=True)
                            if data.get('type') == 'outside' and node.startswith("outside_main")]

        # Find all outside connect nodes
        outside_connect_nodes = {node for node, data in self.graph.nodes(data=True)
                                if data.get('type') == 'outside' and node.startswith("outside_connect")}
 
        # Function to calculate the Euclidean distance between two nodes (based on position)
        def euclidean_distance(node_1, node_2):
            pos_1 = self.graph.nodes[node_1].get('position', [0, 0])
            pos_2 = self.graph.nodes[node_2].get('position', [0, 0])
            return math.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)

        # Radius within which we look for corridor_connect or outside_connect nodes
        radius = 200

        # For each main corridor node, find corridor_connect nodes within the radius
        for main_node in corridor_main_nodes:
            #print(f"Processing main corridor node: {main_node}")

            # List to store corridor_connect nodes within the radius
            nearby_connect_nodes = []

            # Check distance to each corridor_connect node
            for connect_node in corridor_connect_nodes:
                dist = euclidean_distance(main_node, connect_node)
                #print(f"Distance from {main_node} to {connect_node}: {dist} px")

                # If within radius, add to the list
                if dist <= radius:
                    nearby_connect_nodes.append((dist, connect_node))

            # If we found any corridor_connect nodes within the radius
            if nearby_connect_nodes:
                #print(f"Found {len(nearby_connect_nodes)} corridor_connect nodes within radius of {main_node}")

                # Sort the nodes by distance and pick the closest 4
                closest_connect_nodes = sorted(nearby_connect_nodes, key=lambda x: x[0])[:4]

                # Establish edges to the 4 closest corridor connect nodes
                for dist, connect_node in closest_connect_nodes:
                    #print(f"Connecting {main_node} to {connect_node} with distance {dist} px")
                    self.add_edge(main_node, connect_node, weight=dist)

            else:
                print(f"No corridor_connect nodes found within radius of {main_node}")

        # For each main outside node, find outside_connect nodes within the radius
        for main_node in outside_main_nodes:
            #print(f"Processing main outside node: {main_node}")

            # List to store outside_connect nodes within the radius
            nearby_connect_nodes = []

            # Check distance to each outside_connect node
            for connect_node in outside_connect_nodes:
                dist = euclidean_distance(main_node, connect_node)
                #print(f"Distance from {main_node} to {connect_node}: {dist} px")

                # If within radius, add to the list
                if dist <= radius:
                    nearby_connect_nodes.append((dist, connect_node))

            # If we found any outside_connect nodes within the radius
            if nearby_connect_nodes:
                #print(f"Found {len(nearby_connect_nodes)} outside_connect nodes within radius of {main_node}")

                # Sort the nodes by distance and pick the closest 4
                closest_connect_nodes = sorted(nearby_connect_nodes, key=lambda x: x[0])[:4]

                # Establish edges to the 4 closest outside connect nodes
                for dist, connect_node in closest_connect_nodes:
                    #print(f"Connecting {main_node} to {connect_node} with distance {dist} px")
                    self.add_edge(main_node, connect_node, weight=dist)

            else:
                print(f"No outside_connect nodes found within radius of {main_node}")

        print(f"Added Edges to {len(corridor_main_nodes)} main corridor nodes.")
        print(f"Added Edges to {len(outside_main_nodes)} main corridor nodes.") 
    
    def connect_doors(self):
        print("\nConnecting doors...")

        # Find all door nodes (exit_door_COUNTER, c2c_door_COUNTER, r2c_door_COUNTER)
        exit_doors = [node for node, data in self.graph.nodes(data=True)
                    if data.get('type') == 'door' and node.startswith("exit_door")]

        c2c_doors = [node for node, data in self.graph.nodes(data=True)
                    if data.get('type') == 'door' and node.startswith("c2c_door")]

        r2c_doors = [node for node, data in self.graph.nodes(data=True)
                    if data.get('type') == 'door' and node.startswith("r2c_door")]

        #print(f"Found {len(exit_doors)} exit door nodes.")
        #print(f"Found {len(c2c_doors)} c2c door nodes.")
        #print(f"Found {len(r2c_doors)} r2c door nodes.")

        # Find all outside connect nodes, corridor connect nodes, and corridor main nodes
        outside_connect_nodes = {node for node, data in self.graph.nodes(data=True)
                                if data.get('type') == 'outside' and node.startswith("outside_connect")}

        corridor_connect_nodes = {node for node, data in self.graph.nodes(data=True)
                                if data.get('type') == 'corridor' and node.startswith("corridor_connect")}

        corridor_main_nodes = {node for node, data in self.graph.nodes(data=True)
                            if data.get('type') == 'corridor' and node.startswith("corridor_main")}

        #print(f"Found {len(outside_connect_nodes)} outside connect nodes.")
        #print(f"Found {len(corridor_connect_nodes)} corridor connect nodes.")
        #print(f"Found {len(corridor_main_nodes)} corridor main nodes.")

        # Function to calculate the Euclidean distance between two nodes (based on position)
        def euclidean_distance(node_1, node_2):
            pos_1 = self.graph.nodes[node_1].get('position', [0, 0])
            pos_2 = self.graph.nodes[node_2].get('position', [0, 0])
            return math.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)

        # Radius within which we look for connect nodes (100px)
        radius = 100

        # For each exit door, find the closest outside_connect and corridor_connect nodes within the radius
        for door in exit_doors:
            #print(f"Processing exit door node: {door}")

            # List to store outside_connect nodes within the radius
            nearby_outside_connect_nodes = []
            # List to store corridor_connect nodes within the radius
            nearby_corridor_connect_nodes = []

            # Check distance to each outside_connect node
            for connect_node in outside_connect_nodes:
                dist = euclidean_distance(door, connect_node)
                #print(f"Distance from {door} to {connect_node}: {dist} px")

                # If within radius, add to the list
                if dist <= radius:
                    nearby_outside_connect_nodes.append((dist, connect_node))

            # Check distance to each corridor_connect node
            for connect_node in corridor_connect_nodes:
                dist = euclidean_distance(door, connect_node)
                #print(f"Distance from {door} to {connect_node}: {dist} px")

                # If within radius, add to the list
                if dist <= radius:
                    nearby_corridor_connect_nodes.append((dist, connect_node))

            # If we found any outside_connect nodes within the radius
            if nearby_outside_connect_nodes:
                #print(f"Found {len(nearby_outside_connect_nodes)} outside_connect nodes within radius of {door}")

                # Sort the nodes by distance and pick the closest one
                closest_outside_connect_node = min(nearby_outside_connect_nodes, key=lambda x: x[0])

                # Establish an edge to the closest outside_connect node
                #print(f"Connecting {door} to {closest_outside_connect_node[1]} with distance {closest_outside_connect_node[0]} px")
                self.add_edge(door, closest_outside_connect_node[1], weight=closest_outside_connect_node[0])

            else:
                print(f"No outside_connect nodes found within radius of {door}")

            # If we found any corridor_connect nodes within the radius
            if nearby_corridor_connect_nodes:
                #print(f"Found {len(nearby_corridor_connect_nodes)} corridor_connect nodes within radius of {door}")

                # Sort the nodes by distance and pick the closest one
                closest_corridor_connect_node = min(nearby_corridor_connect_nodes, key=lambda x: x[0])

                # Establish an edge to the closest corridor_connect node
                #print(f"Connecting {door} to {closest_corridor_connect_node[1]} with distance {closest_corridor_connect_node[0]} px")
                self.add_edge(door, closest_corridor_connect_node[1], weight=closest_corridor_connect_node[0])

            else:
                print(f"No corridor_connect nodes found within radius of {door}")

        # For each c2c door, find the closest 8 corridor_connect nodes within the radius
        for door in c2c_doors:
            #print(f"Processing c2c door node: {door}")

            # List to store corridor_connect nodes within the radius
            nearby_connect_nodes = []

            # Check distance to each corridor_connect node
            for connect_node in corridor_connect_nodes:
                dist = euclidean_distance(door, connect_node)
                #print(f"Distance from {door} to {connect_node}: {dist} px")

                # If within radius, add to the list
                if dist <= radius:
                    nearby_connect_nodes.append((dist, connect_node))

            # If we found any corridor_connect nodes within the radius
            if nearby_connect_nodes:
                #print(f"Found {len(nearby_connect_nodes)} corridor_connect nodes within radius of {door}")

                # Sort the nodes by distance and pick the closest 8
                closest_connect_nodes = sorted(nearby_connect_nodes, key=lambda x: x[0])[:4]

                # Establish edges to the 8 closest corridor_connect nodes
                for dist, connect_node in closest_connect_nodes:
                    #print(f"Connecting {door} to {connect_node} with distance {dist} px")
                    self.add_edge(door, connect_node, weight=dist)

            else:
                print(f"No corridor_connect nodes found within radius of {door}")

        # For each r2c door, find the closest corridor_connect and corridor_main nodes within the radius
        for door in r2c_doors:
            #print(f"Processing r2c door node: {door}")

            # List to store corridor_connect and corridor_main nodes within the radius
            nearby_connect_nodes = []

            # Check distance to each corridor_connect node
            for connect_node in corridor_connect_nodes:
                dist = euclidean_distance(door, connect_node)
                #print(f"Distance from {door} to {connect_node}: {dist} px")

                # If within radius, add to the list
                if dist <= radius:
                    nearby_connect_nodes.append((dist, connect_node))

            # Check distance to each corridor_main node
            for main_node in corridor_main_nodes:
                dist = euclidean_distance(door, main_node)
                #print(f"Distance from {door} to {main_node}: {dist} px")

                # If within radius, add to the list
                if dist <= radius:
                    nearby_connect_nodes.append((dist, main_node))

            # If we found any corridor_connect or corridor_main nodes within the radius
            if nearby_connect_nodes:
                #print(f"Found {len(nearby_connect_nodes)} corridor_connect or corridor_main nodes within radius of {door}")

                # Sort the nodes by distance and pick the closest one
                closest_connect_node = min(nearby_connect_nodes, key=lambda x: x[0])

                # Establish an edge to the closest corridor_connect or corridor_main node
                #print(f"Connecting {door} to {closest_connect_node[1]} with distance {closest_connect_node[0]} px")
                self.add_edge(door, closest_connect_node[1], weight=closest_connect_node[0])

            else:
                print(f"No corridor_connect or corridor_main nodes found within radius of {door}") 
 
    def connect_rooms(self):
        print("\nConnecting rooms...")

        # Find all room nodes
        room_nodes = [node for node, data in self.graph.nodes(data=True) 
                    if data.get('type') == 'room']

        # Print out total number of room nodes
        #print(f"Found {len(room_nodes)} room nodes.")

        # Find all corridor_main and corridor_connect nodes
        corridor_main_nodes = {node for node, data in self.graph.nodes(data=True)
                            if data.get('type') == 'corridor' and node.startswith("corridor_main")}
        
        corridor_connect_nodes = {node for node, data in self.graph.nodes(data=True)
                                if data.get('type') == 'corridor' and node.startswith("corridor_connect")}

        #print(f"Found {len(corridor_main_nodes)} corridor_main nodes.")
        #print(f"Found {len(corridor_connect_nodes)} corridor_connect nodes.")

        # Function to calculate the Euclidean distance between two nodes (based on position)
        def euclidean_distance(node_1, node_2):
            pos_1 = self.graph.nodes[node_1].get('position', [0, 0])
            pos_2 = self.graph.nodes[node_2].get('position', [0, 0])
            return math.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)

        # Radius within which we look for corridor nodes (400px)
        radius = 400
        disconnected_room_count = 0  # Counter for disconnected rooms

        # Iterate over all room nodes
        for room in room_nodes:
            # Check if the room node is disconnected (no edges)
            if len(list(self.graph.neighbors(room))) == 0:
                disconnected_room_count += 1
                #print(f"     Room node {room} is disconnected. Growing radius...")

                # List to store corridor nodes within the radius
                nearby_corridor_nodes = []

                # Check distance to each corridor_main node
                for connect_node in corridor_main_nodes:
                    dist = euclidean_distance(room, connect_node)
                    if dist <= radius:
                        nearby_corridor_nodes.append((dist, connect_node))

                # Check distance to each corridor_connect node
                for connect_node in corridor_connect_nodes:
                    dist = euclidean_distance(room, connect_node)
                    if dist <= radius:
                        nearby_corridor_nodes.append((dist, connect_node))

                # If we found any corridor nodes within the radius
                if nearby_corridor_nodes:
                    #print(f"Found {len(nearby_corridor_nodes)} corridor nodes within radius of {room}")

                    # Sort the nodes by distance and pick the closest one
                    closest_corridor_node = min(nearby_corridor_nodes, key=lambda x: x[0])

                    # Establish an edge to the closest corridor node
                    #print(f"Connecting {room} to {closest_corridor_node[1]} with distance {closest_corridor_node[0]} px")
                    self.add_edge(room, closest_corridor_node[1], weight=closest_corridor_node[0])

                else:
                    print(f"No corridor nodes found within radius of {room}")

        # Print the number of disconnected room nodes
        print(f"Total disconnected room nodes: {disconnected_room_count}") 

    def merge_nearby_nodes(self, threshold_room=50, threshold_door=30):
        """
        Merge nodes that are within a certain vicinity threshold, with different thresholds for "room" and "door" nodes.

        Args:
            threshold_room (float): Maximum distance for merging "room" nodes (default: 50).
            threshold_door (float): Maximum distance for merging "door" nodes (default: 30).

        Returns:
            networkx.Graph: The modified graph after merging nodes.
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
            # Path compression for efficient find
            if parent[u] != u:
                parent[u] = find(parent[u])  # Path compression
            return parent[u]

        def union(u, v):
            # Union by assigning parent
            pu, pv = find(u), find(v)
            if pu != pv:
                parent[pv] = pu  # Merge set containing v into set containing u

        # Merging room nodes
        for i in range(len(room_nodes)):
            node_id_1 = room_nodes[i]
            pos_1 = np.array(positions[node_id_1])
            for j in range(i + 1, len(room_nodes)):
                node_id_2 = room_nodes[j]
                pos_2 = np.array(positions[node_id_2])
                dist = np.linalg.norm(pos_1 - pos_2)
                if dist < threshold_room:  # Apply room threshold
                    union(node_id_1, node_id_2)

        # Merging door nodes
        for i in range(len(door_nodes)):
            node_id_1 = door_nodes[i]
            pos_1 = np.array(positions[node_id_1])
            for j in range(i + 1, len(door_nodes)):
                node_id_2 = door_nodes[j]
                pos_2 = np.array(positions[node_id_2])
                dist = np.linalg.norm(pos_1 - pos_2)
                if dist < threshold_door:  # Apply door threshold
                    union(node_id_1, node_id_2)

        # Group nodes by their representative parent node
        clusters = {}
        for node_id in room_nodes + door_nodes:
            p = find(node_id)
            clusters.setdefault(p, []).append(node_id)

        # Merge nodes in each cluster
        for cluster_nodes in clusters.values():
            if len(cluster_nodes) > 1:
                # Compute the average position
                positions_list = [positions[node_id] for node_id in cluster_nodes]
                avg_position = tuple(map(int, np.mean(positions_list, axis=0)))

                # Keep the first node as the main node
                main_node = cluster_nodes[0]
                self.graph.nodes[main_node]["position"] = avg_position

                # Remove other nodes from the graph and node_types
                for node_id in cluster_nodes[1:]:
                    # Remove from the graph
                    self.graph.remove_node(node_id)
                    
                    # Remove from the node_types
                    for node_type, node_list in self.node_types.items():
                        if node_id in node_list:
                            node_list.remove(node_id)

                    print(f"Merged nodes {cluster_nodes} into {main_node} at {avg_position}")

        # Return the modified graph
        return self.graph
 
    def connect_all_rooms(self, input_path, graph_img_dir):
        print("\nConnecting all rooms...")

        # Find all room nodes
        room_nodes = [node for node, data in self.graph.nodes(data=True) 
                    if data.get('type') == 'room']
        
        # Find all exit door nodes
        exit_doors = [node for node, data in self.graph.nodes(data=True)
                    if data.get('type') == 'door' and node.startswith("exit_door")]

        # Print total number of nodes in the graph
        print(f"Total number of nodes in the graph: {len(self.graph.nodes)}")

        # If there's only one room node, no need to do anything
        if len(room_nodes) <= 1:
            print("Only one or no room node, no connection needed.")
            return

        # Initialize the "web" which will store the minimal set of nodes that connect the rooms
        web_nodes = set()

        # To store paths for plotting
        paths_to_plot = []

        # Find the shortest path between every pair of room nodes
        for i, room_1 in enumerate(room_nodes):
            for room_2 in room_nodes[i+1:]:
                # Find the shortest path between the two rooms
                try:
                    shortest_path = nx.shortest_path(self.graph, source=room_1, target=room_2)
                    #print(f"Shortest path from {room_1} to {room_2}: {shortest_path}")
                    
                    # Add all nodes from this path to the web
                    web_nodes.update(shortest_path)
                    paths_to_plot.append(shortest_path)  # Store path for plotting
                
                except nx.NetworkXNoPath:
                    print(f"No path found between {room_1} and {room_2}")

        # Ensure that rooms are connected to exit doors
        for room in room_nodes:
            closest_exit = None
            min_distance = float('inf')
            
            # Find the nearest exit door to the room node
            for exit_door in exit_doors:
                try:
                    path = nx.shortest_path(self.graph, source=room, target=exit_door)
                    path_length = len(path)
                    
                    # Update the minimum distance and closest exit door
                    if path_length < min_distance:
                        min_distance = path_length
                        closest_exit = exit_door
                        web_nodes.update(path)  # Add the path to the web
            
                except nx.NetworkXNoPath:
                    print(f"No path found from room {room} to exit door {exit_door}")

        # Now we have the minimal set of nodes in web_nodes
        print(f"Total number of nodes in the web (connecting all rooms and exit doors): {len(web_nodes)}")

        # Remove all nodes and edges that are not in the web
        nodes_to_remove = set(self.graph.nodes) - web_nodes
        self.graph.remove_nodes_from(nodes_to_remove)
        
        # Custom function to remove edges that are not between web_nodes
        self.remove_edges_not_in_web(web_nodes)

        # Print the final number of nodes and edges in the modified graph
        print(f"Graph after modification: {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")
        
        # Now let's plot the paths on the image

        # Load the input image
        img = Image.open(input_path)
        img_width, img_height = img.size

        # Create a figure for plotting
        fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100), dpi=100)
        
        # Plot the image
        ax.imshow(img)

        # Generate random colors for each path
        random.seed(42)  # For reproducibility
        def get_random_color():
            return (random.random(), random.random(), random.random())

        # Plot each path with a random color
        for path in paths_to_plot:
            path_colors = get_random_color()
            
            # Extract positions for each node in the path
            path_positions = [self.graph.nodes[node].get('position') for node in path]
            path_positions = np.array(path_positions)
            
            # Plot the path
            ax.plot(path_positions[:, 0], path_positions[:, 1], color=path_colors, linewidth=2)

        # Save the resulting image 
        output_image_path = f"{graph_img_dir}/colored_paths.png"
        plt.axis('off')  # Turn off the axis
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Shortest paths image saved as {output_image_path}")

        return self.graph 


    def remove_edges_not_in_web(self, web_nodes):
        # Create a list of edges to remove
        edges_to_remove = [
            (u, v) for u, v in self.graph.edges()
            if u not in web_nodes or v not in web_nodes
        ]
        
        # Remove edges
        self.graph.remove_edges_from(edges_to_remove)
        #print(f"Removed {len(edges_to_remove)} edges.")

    def return_graph_size(self):
        return len(self.graph.nodes)

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

        Returns:
            None
        """
        door_nodes = [node_id for node_id in self.node_types["door"]]
        room_nodes = [node_id for node_id in self.node_types["room"]]

        if not door_nodes or not room_nodes:
            print("No doors or rooms available to connect.")
            return

        print("\nConnecting doors to rooms...")
        for door_id in door_nodes:
            # Get the position of the current door
            door_pos = np.array(self.graph.nodes[door_id]["position"])

            # Find the nearest room
            nearest_room = None
            min_distance = float("inf")
            for room_id in room_nodes:
                room_pos = np.array(self.graph.nodes[room_id]["position"])
                distance = np.linalg.norm(door_pos - room_pos)
                if distance < min_distance:
                    min_distance = distance
                    nearest_room = room_id

            # Add edge between the door and its nearest room
            if nearest_room:
                self.add_edge(door_id, nearest_room)
                print(f"Connected door '{door_id}' to room '{nearest_room}' (distance: {min_distance:.2f})")


 