import os
import cv2
import numpy as np
import sys

sys.path.append("./utils")  # Add the utils folder to the Python path
 
 
def process_fill_rooms(
    image_path,
    graph,
    results_dir,
    radius_threshold=90,
    node_radius=20,
    fill_mode="smart",  # "smart" or "flood"
    point_radius=10,  # Radius for generating points around the node (smart fill only)
    point_step=10,  # Step angle for generating points (smart fill only)
    flood_threshold=30,  # Radius for flood fill points (flood fill only)
):
    """
    Display nodes on the floorplan image and perform smart or flood filling around nodes.
    Filling stops at non-white pixels (walls) or when exceeding a threshold radius.

    Args:
        image_path (str): Path to the input floorplan image.
        graph (BuildingGraph): Graph containing nodes with room types and positions.
        results_dir (str): Directory to save the results.
        radius_threshold (int): Maximum radius for the smart fill (default: 90 pixels).
        node_radius (int): Radius for displaying the node centers (default: 20 pixels).
        fill_mode (str): "smart" for BFS-based fill, "flood" for multi-start flood-fill.
        point_radius (int): Radius for generating points around the node for smart fill.
        point_step (int): Angle step in degrees for generating starting points around the node.
        flood_threshold (int): Radius around the node to generate flood-fill start points.

    Returns:
        str: Path to the modified image with nodes and fills.
        str: Path to the file with fill area details.
    """
    floorplan = cv2.imread(image_path)
    if floorplan is None:
        raise FileNotFoundError(f"Floorplan image not found at {image_path}")

    height, width = floorplan.shape[:2]
    image_name_no_ext = os.path.splitext(os.path.basename(image_path))[0]

    # Create directories for saving results
    plots_dir = os.path.join(results_dir, "Plots")
    fill_dir = os.path.join(plots_dir, "smart_fill" if fill_mode == "smart" else "flood_fill")
    os.makedirs(fill_dir, exist_ok=True)

    fill_img_dir = os.path.join(fill_dir, f"{image_name_no_ext}")
    os.makedirs(fill_img_dir, exist_ok=True)

    output_image_path = os.path.join(fill_img_dir, f"{image_name_no_ext}_{fill_mode}fill.png")
    area_file_path = os.path.join(fill_img_dir, f"{fill_mode}_fill_area.txt")

    # Initialize area file
    with open(area_file_path, "w") as area_file:
        area_file.write("Node_ID\tPosition\tFilled_Area\n")

    def is_wall(pixel):
        """Check if a pixel is a wall (non-white)."""
        return not all(pixel == [255, 255, 255])  # Wall if not white
 
    def fill_node(node_id, node_data, fill_color):
        nonlocal floorplan
        x, y = node_data["position"]

        # Ensure node position is within bounds
        x = max(0, min(int(x), width - 1))
        y = max(0, min(int(y), height - 1))

        fill_area = 0
        visited = np.zeros((height, width), dtype=bool)

        if fill_mode == "smart":
            # Generate starting points around the node
            starting_points = [(x, y)]  # Include the node itself
            for angle in range(0, 360, point_step):  # Generate points in a circle
                radian = np.radians(angle)
                px = int(x + point_radius * np.cos(radian))
                py = int(y + point_radius * np.sin(radian))

                if 0 <= px < width and 0 <= py < height and not is_wall(floorplan[py, px]):
                    starting_points.append((px, py))

            # Smart fill using BFS for each starting point
            for sx, sy in starting_points:
                if visited[sy, sx]:
                    continue

                queue = [(sx, sy)]
                visited[sy, sx] = True

                while queue:
                    cx, cy = queue.pop(0)
                    distance = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
                    if distance > radius_threshold:
                        continue

                    if is_wall(floorplan[cy, cx]):  # Stop at walls
                        continue

                    # Fill the current pixel
                    floorplan[cy, cx] = fill_color
                    fill_area += 1

                    # Add neighbors to the queue
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((nx, ny))

        elif fill_mode == "flood":
            # Generate flood-fill starting points around the node
            for angle in range(0, 360, 10):  # Generate points every 10 degrees
                radian = np.radians(angle)
                fx = int(x + flood_threshold * np.cos(radian))
                fy = int(y + flood_threshold * np.sin(radian))

                if 0 <= fx < width and 0 <= fy < height:
                    if not is_wall(floorplan[fy, fx]):
                        # Apply flood fill from this point
                        mask = np.zeros((height + 2, width + 2), np.uint8)
                        _, _, _, rect = cv2.floodFill(
                            floorplan,
                            mask,
                            (fx, fy),
                            fill_color,
                            loDiff=(10, 10, 10),
                            upDiff=(10, 10, 10),
                        )
                        fill_area += (rect[2] - rect[0]) * (rect[3] - rect[1])

        return fill_area

    # Fill outside nodes
    for node_id in graph.node_types["outside"]:
        if node_id not in graph.graph.nodes:
            print(f"Node {node_id} is missing from the graph. Skipping fill.")
            continue

        node_data = graph.graph.nodes[node_id]
        fill_color = (204, 102, 255)  # Purple for outside fill
        fill_area = fill_node(node_id, node_data, fill_color)

        with open(area_file_path, "a") as area_file:
            area_file.write(f"{node_id}\t({node_data['position'][0]}, {node_data['position'][1]})\t{fill_area}\n")

    # Fill corridor nodes
    for node_id in graph.node_types["corridor"]:
        if node_id not in graph.graph.nodes:
            print(f"Node {node_id} is missing from the graph. Skipping fill.")
            continue

        node_data = graph.graph.nodes[node_id]
        fill_color = (102, 255, 178)  # Mint green for corridor fill
        fill_area = fill_node(node_id, node_data, fill_color)

        with open(area_file_path, "a") as area_file:
            area_file.write(f"{node_id}\t({node_data['position'][0]}, {node_data['position'][1]})\t{fill_area}\n")

    # Fill room nodes
    for node_id in graph.node_types["room"]:
        if node_id not in graph.graph.nodes:
            print(f"Node {node_id} is missing from the graph. Skipping fill.")
            continue

        node_data = graph.graph.nodes[node_id]
        fill_color = (102, 204, 255)  # Light blue for room fill
        fill_area = fill_node(node_id, node_data, fill_color)

        with open(area_file_path, "a") as area_file:
            area_file.write(f"{node_id}\t({node_data['position'][0]}, {node_data['position'][1]})\t{fill_area}\n")

        # Draw the node on top to ensure it is not colored over
        #cv2.circle(floorplan, (int(node_data['position'][0]), int(node_data['position'][1])), node_radius, (255, 102, 102), -1)  # Coral red
    # Save the modified image
    cv2.imwrite(output_image_path, floorplan)
    print(f"{fill_mode.capitalize()}-filled image saved to {output_image_path}")
    print(f"{fill_mode.capitalize()}-filled areas saved to {area_file_path}")

    return output_image_path, area_file_path
