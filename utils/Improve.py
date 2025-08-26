import cv2
import numpy as np
import math
import os
import networkx as nx

def pixelwise_areas(flood_output_img_pth, graph, connect_img_dir, print_tag = False):
    # Load the image
    image = cv2.imread(flood_output_img_pth)
    if image is None:
        raise ValueError(f"Image not found at path: {flood_output_img_pth}")

    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get dimensions of the image
    height, width, _ = image.shape
    if print_tag:
        print(f"\nImage dimensions: {width}x{height}")

    # Define color mappings
    colors = {
        "room_pixels": (255, 204, 102),  # #ffcc66
        "outdoor_pixels": (255, 102, 204),  # #ff66cc
        "corridor_pixels": (178, 255, 102),  # Assuming a gray corridor
        "unmarked_pixels": (255, 255, 255),  # White
        "wall_pixels": (0, 0, 0),  # Black
    }

    # Threshold for color matching
    threshold = 10

    # Convert the image into a numpy array for faster processing
    image_array = np.array(image)

    # Create masks for each color with a threshold
    def create_mask(color):
        lower_bound = np.maximum(np.array(color) - threshold, 0)
        upper_bound = np.minimum(np.array(color) + threshold, 255)
        return np.all((image_array >= lower_bound) & (image_array <= upper_bound), axis=-1)

    room_mask = create_mask(colors["room_pixels"])
    outdoor_mask = create_mask(colors["outdoor_pixels"])
    corridor_mask = create_mask(colors["corridor_pixels"])
    unmarked_mask = create_mask(colors["unmarked_pixels"])
    wall_mask = create_mask(colors["wall_pixels"])

    # Find pixel locations for each category
    room_pixels = np.argwhere(room_mask).tolist()
    outdoor_pixels = np.argwhere(outdoor_mask).tolist()
    corridor_pixels = np.argwhere(corridor_mask).tolist()
    unmarked_pixels = np.argwhere(unmarked_mask).tolist()
    wall_pixels = np.argwhere(wall_mask).tolist()

    # Print the lengths of each category
    if print_tag:
        print(f"Number of room pixels: {len(room_pixels)}")
        print(f"Number of outdoor pixels: {len(outdoor_pixels)}")
        print(f"Number of corridor pixels: {len(corridor_pixels)}")
        print(f"Number of unmarked pixels: {len(unmarked_pixels)}")
        print(f"Number of wall pixels: {len(wall_pixels)}")

    # Create a new blank image to recreate the segmented map
    recreated_image = np.zeros_like(image_array)

    # Assign colors to the corresponding pixel locations
    for y, x in room_pixels:
        recreated_image[y, x] = colors["room_pixels"]
    for y, x in outdoor_pixels:
        recreated_image[y, x] = colors["outdoor_pixels"]
    for y, x in corridor_pixels:
        recreated_image[y, x] = colors["corridor_pixels"]
    for y, x in unmarked_pixels:
        recreated_image[y, x] = colors["unmarked_pixels"]
    for y, x in wall_pixels:
        recreated_image[y, x] = colors["wall_pixels"]
 
    # Save the recreated image
    output_path = connect_img_dir + "/recreated_image.png"
    recreated_image_bgr = cv2.cvtColor(recreated_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, recreated_image_bgr)
    print(f"Recreated Thresholded image saved to: {output_path}")

    return room_pixels, outdoor_pixels, corridor_pixels, unmarked_pixels, wall_pixels, output_path

def classify_doors(image_path, door_bboxes, output_dir, print_tag=False):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Check for unique colors in the image
    unique_colors = np.unique(image.reshape(-1, image.shape[2]), axis=0)
    if len(unique_colors) != 5:
        raise ValueError(f"The image must contain exactly 5 unique colors, but it has {len(unique_colors)}.")

    # Copy the image for annotations
    doors_image = image.copy()
    annotated_image = image.copy()

    # Lists to store bounding boxes
    exit_doors_bboxes = []  # Red for exit doors (pink pixels)
    corridor2corridor_doors_bboxes = []  # Green for corridor-to-corridor doors
    room2corridor_doors_bboxes = []  # Blue for room-to-corridor doors
    room2room_doors_bboxes = []  # Purple for room-to-room doors
    wardrobe_doors_bboxes = []  # Yellow for wardrobe doors

    # Define colors
    all_doors_color = (128, 128, 128)  # Gray for all doors in `doors_image`
    red = (255, 0, 0)  # Red for exit doors
    green = (0, 255, 0)  # Green for corridor-to-corridor doors
    blue = (0, 0, 255)  # Blue for room-to-corridor doors
    purple = (128, 0, 128)  # Purple for room-to-room doors
    yellow = (255, 255, 0)  # Yellow for wardrobe doors

    # Track processed doors
    processed_bboxes = set()

    # Loop over bounding boxes
    for bbox in door_bboxes:
        x1, y1, x2, y2 = bbox  # Properly unpack bbox coordinates

        # Ensure bbox is within image dimensions
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        # Extract the region of interest (ROI)
        roi = image[y1:y2, x1:x2]

        # Check for pink (exit doors)
        if np.any(np.all(roi == [255, 102, 204], axis=-1)):
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), red, thickness=2)
            exit_doors_bboxes.append([x1, y1, x2, y2])
            processed_bboxes.add(tuple(bbox))
        # Check for green (corridor doors)
        elif np.any(np.all(roi == [178, 255, 102], axis=-1)):
            if np.any(np.all(roi == [255, 204, 102], axis=-1)):  # Orange indicates room2corridor
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), blue, thickness=2)
                room2corridor_doors_bboxes.append([x1, y1, x2, y2])
            else:
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), green, thickness=2)
                corridor2corridor_doors_bboxes.append([x1, y1, x2, y2])
            processed_bboxes.add(tuple(bbox))

    # Handle remaining doors
    for bbox in door_bboxes:
        if tuple(bbox) not in processed_bboxes:
            x1, y1, x2, y2 = bbox  # Properly unpack bbox coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            roi = image[y1:y2, x1:x2]

            # Check for white to classify remaining doors
            if not np.any(np.all(roi == [255, 255, 255], axis=-1)):
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), purple, thickness=2)
                room2room_doors_bboxes.append([x1, y1, x2, y2])
            else:
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), yellow, thickness=2)
                wardrobe_doors_bboxes.append([x1, y1, x2, y2])

    # Draw all bboxes on doors image
    for bbox in door_bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(doors_image, (x1, y1), (x2, y2), all_doors_color, thickness=2)

    # Convert images back to BGR for saving
    doors_image = cv2.cvtColor(doors_image, cv2.COLOR_RGB2BGR)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the images
    doors_path = os.path.join(output_dir, "original_doors.png")
    annotated_path = os.path.join(output_dir, "annotated_doors.png")
    cv2.imwrite(doors_path, doors_image)
    cv2.imwrite(annotated_path, annotated_image)

    # Validate the total count
    total_doors = len(exit_doors_bboxes) + len(corridor2corridor_doors_bboxes) + len(room2corridor_doors_bboxes) + len(room2room_doors_bboxes) + len(wardrobe_doors_bboxes)
    assert total_doors == len(door_bboxes), "The classified doors do not match the total input doors."
    if print_tag:
        print(f"{len(exit_doors_bboxes)}-exit doors found")
        print(f"{len(corridor2corridor_doors_bboxes)}-corridor-to-corridor doors found")
        print(f"{len(room2corridor_doors_bboxes)}-room-to-corridor doors found")
        print(f"{len(room2room_doors_bboxes)}-room-to-room doors found")
        print(f"{len(wardrobe_doors_bboxes)}-wardrobe doors found")

    print(f"Doors pre-classification saved at {doors_path}")
    print(f"Annotated doors saved at {annotated_path}")
    return exit_doors_bboxes, corridor2corridor_doors_bboxes, room2corridor_doors_bboxes, room2room_doors_bboxes, wardrobe_doors_bboxes
 

def plot_graph_door(image_path, graph, exit_dbboxes, corridor2corridor_dbboxes, room2corridor_dbboxes, room2room_dbboxes, connect_img_dir):
    # Verify if the image path is valid
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Print the contents of the graph
    print("Graph contains:")
    print(f"Number of rooms: {len(graph.node_types['room'])}")
    print(f"Number of doors: {len(graph.node_types['door'])}")
    print(f"Number of corridors: {len(graph.node_types['corridor'])}")
    print(f"Number of outsides: {len(graph.node_types['outside'])}")
 
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Copy the image for annotations
    annotated_image = image_rgb.copy()

    # Define colors for nodes
    room_color = (255, 0, 0)  # Red for rooms
    door_color = (255, 182, 193)  # Pink for doors
    corridor_color = (255, 165, 0)  # Orange for corridors
    outside_color = (157,163,0)
    # Define colors for door bounding boxes
    exit_color = (0, 0, 255)  # Blue for exit doors
    corridor2corridor_color = (0, 255, 0)  # Green for corridor-to-corridor doors
    room2corridor_color = (0, 255, 255)  # Cyan for room-to-corridor doors
    room2room_color = (255, 0, 255)  # Magenta for room-to-room doors

    # Draw the door bounding boxes with respective colors
    for bbox in exit_dbboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), exit_color, thickness=2)

    for bbox in corridor2corridor_dbboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), corridor2corridor_color, thickness=2)

    for bbox in room2corridor_dbboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), room2corridor_color, thickness=2)

    for bbox in room2room_dbboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), room2room_color, thickness=2)

    # Draw graph nodes on the image
    for node_id, node_data in graph.graph.nodes(data=True):
        node_type = node_data['type']
        position = node_data.get('position')

        # Validate position
        if not isinstance(position, tuple) or len(position) != 2:
            print(f"Warning: Node {node_id} has an invalid position: {position}. Skipping.")
            continue

        # Convert position to integer coordinates
        position = tuple(map(int, position))

        # Determine node color
        if node_type == "room":
            node_color = room_color
        elif node_type == "door":
            node_color = door_color
        elif node_type == "corridor":
            node_color = corridor_color
        elif node_type == "outside":
            node_color = outside_color
        else:
            continue  # Ignore any other types of nodes

        # Draw node on the image
        cv2.circle(annotated_image, position, 5, node_color, -1)  # Filled circle

    # Ensure output directory exists
    os.makedirs(connect_img_dir, exist_ok=True)

    # Save the annotated image
    output_path = os.path.join(connect_img_dir, "bbox_with_graph.png")
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, annotated_image)

    print(f"Image with bounding boxes and graph nodes saved to: {output_path}")