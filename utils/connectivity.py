import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle

def create_wall_mask(image_array, tolerance=240, proximity=70):
    black_color = np.array([0, 0, 0])
    lower_bound = np.clip(black_color - tolerance, 0, 255)
    upper_bound = np.clip(black_color + tolerance, 0, 255)
    wall_mask = cv2.inRange(image_array, lower_bound, upper_bound)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * proximity + 1, 2 * proximity + 1))
    expanded_wall_mask = cv2.dilate(wall_mask, kernel)
    return expanded_wall_mask

def create_color_mask(image_array, target_color, tolerance=20):
    # Convert target_color to a numpy array for element-wise operations
    target_color = np.array(target_color)
    
    # Calculate the lower and upper bounds
    lower_bound = np.clip(target_color - tolerance, 0, 255)
    upper_bound = np.clip(target_color + tolerance, 0, 255)
    
    # Create the mask using the color range
    mask = cv2.inRange(image_array, lower_bound, upper_bound)
    
    return mask

def extract_pixels_per_blob(labels, num_labels):
    blobs = []
    for i in range(1, num_labels):  # Skip background (label 0)
        blob_pixels = np.argwhere(labels == i)
        blobs.append([tuple(coord) for coord in blob_pixels])
    return blobs

def clean_and_label_components(mask, wall_mask, area_threshold=1000):
    cleaned_mask = cv2.bitwise_and(mask, cv2.bitwise_not(wall_mask))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask)

    filtered_mask = np.zeros_like(cleaned_mask)
    valid_labels = []
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
            filtered_mask[labels == i] = 255
            valid_labels.append(i)

    num_filtered_labels, filtered_labels, _, _ = cv2.connectedComponentsWithStats(filtered_mask)

    blobs = extract_pixels_per_blob(filtered_labels, num_filtered_labels)

    return filtered_labels, num_filtered_labels - 1, blobs

def process_image(image_path, corridor_color, outdoor_color, tolerance=20, wall_tolerance=240, proximity=5, area_threshold=1000):
    image = Image.open(image_path)
    image_rgb = image.convert("RGB")
    image_array = np.array(image_rgb)

    # Create wall, corridor, and outdoor masks
    wall_mask = create_wall_mask(image_array, wall_tolerance, proximity)
    corridor_mask = create_color_mask(image_array, corridor_color, tolerance)
    outdoor_mask = create_color_mask(image_array, outdoor_color, tolerance)

    # Clean and label components for corridor and outdoor
    _, corridor_count, corridor_pixels = clean_and_label_components(corridor_mask, wall_mask, area_threshold)
    _, outdoor_count, outdoor_pixels = clean_and_label_components(outdoor_mask, wall_mask, area_threshold)

    # Detect the color #FFCC66 (RGB: [255, 204, 102])
    room_color = [255, 204, 102]
    room_mask = create_color_mask(image_array, room_color, tolerance)
    
    # Extract room pixels and convert to list of [y, x] format
    room_pixels = [[[y, x] for y, x in np.argwhere(room_mask == 255)]]
    
    # Return the image array along with the corridor, outdoor, and room pixels 
    return image_array, corridor_pixels, outdoor_pixels, room_pixels
    
def grow_regions(image_array, corridor_pixels, outdoor_pixels, distance=20, connect_img_dir=""):
    """
    Grows regions starting from the top-left corner of each blob, selecting pixels
    that are `distance` pixels apart. Plots the result with red circles on the original image.

    Args:
        image_array (np.ndarray): The original image array.
        corridor_pixels (List[List[Tuple[int, int]]]): List of corridor blobs.
        outdoor_pixels (List[List[Tuple[int, int]]]): List of outdoor blobs.
        distance (int): Distance in pixels for growing regions (default: 20).
        connect_img_dir (str): Directory to save the resulting image.

    Returns:
        np.ndarray: Array of all marked pixel coordinates.
    """
    height, width, _ = image_array.shape 
    print(f"Plot dimensions: {width}x{height}")

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(image_array)

    marked_pixels = []

    def grow_blob(blob, color='red'):
        nonlocal marked_pixels
        visited = set()
        queue = [blob[0]]  # Start from the top-left pixel
        while queue:
            x, y = queue.pop(0)
            if (x, y) in visited:
                continue
            visited.add((x, y))
            marked_pixels.append((x, y))
            # Plot the pixel as a circle
            circle = Circle((y, x), radius=3, color=color, fill=True)
            ax.add_patch(circle)

            # Add neighbors that are `distance` away
            for dx in range(-distance, distance + 1, distance):
                for dy in range(-distance, distance + 1, distance):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in blob and (nx, ny) not in visited:
                        queue.append((nx, ny))

    # Grow and plot for each corridor and outdoor blob
    for i, corridor_blob in enumerate(corridor_pixels, start=1):
        print(f"Processing corridor blob {i}/{len(corridor_pixels)}")
        grow_blob(corridor_blob, color='red')

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')

    if connect_img_dir:
        save_path = f"{connect_img_dir}/grown_regions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Image saved at: {save_path}")

    plt.close(fig)
    return np.array(marked_pixels)

def plot_first_ten_pixels(image_path, marked_pixels, results_dir):
    """
    Plots the first ten marked pixels on the image with their coordinates displayed as text.
    Args:
        image_array (np.ndarray): The original image array.
        marked_pixels (np.ndarray): Array of all marked pixel coordinates.
    """
    image = Image.open(image_path)
    image_rgb = image.convert("RGB")
    image_array = np.array(image_rgb)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_array)

    for i, (x, y) in enumerate(marked_pixels[:60]):
        ax.scatter(y, x, color='blue', s=10)
        if i%3==0:
            ax.text(y + 5, x, f'({x}, {y})', color='red', fontsize=8)

    ax.set_xlim(0, image_array.shape[1])
    ax.set_ylim(image_array.shape[0], 0)
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Height (pixels)")
    ax.set_title("First Ten Marked Pixels")
    save_path = f"{results_dir}/10_checker.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
 

def refine_marked_points(marked_points, door_bboxes):
    """
    Refine the list of marked points by removing those within the expanded bounding boxes.

    Args:
        marked_points (np.ndarray): Array of points in (y, x) format, shape (N, 2).
        door_bboxes (np.ndarray): Array of bounding boxes in [y1, x1, y2, x2] format.

    Returns:
        np.ndarray: Refined array of marked points that are outside the expanded bounding boxes,
                    in the same format as the input (N, 2).
    """
    refined_points = []

    for point in marked_points:
        y, x = point  # Unpack point
        keep_point = True
        
        # Check each bounding box
        for bbox in door_bboxes:
            #y1, x1, y2, x2 = bbox
            x1, y1, x2, y2 = bbox

            # Expand the bounding box by 3 pixels on all sides
            y1_expanded = max(0, y1 - 3)
            x1_expanded = max(0, x1 - 3)
            y2_expanded = y2 + 3
            x2_expanded = x2 + 3

            # Check if the point lies within the expanded bounding box
            if y1_expanded <= y <= y2_expanded and x1_expanded <= x <= x2_expanded:
                keep_point = False
                break  # If the point is inside any bounding box, skip it
        
        # If the point is not inside any bounding box, add it to the refined list
        if keep_point:
            refined_points.append(point)
    
    # Return the refined points as a numpy array in the same format as input
    return np.array(refined_points)

def merge_overlapping_bboxes(door_bboxes):
    """
    Merge overlapping door bounding boxes into larger bounding boxes.
    
    Args:
        door_bboxes (List[List[int]]): List of bounding boxes in the form [xmin, ymin, xmax, ymax].
        
    Returns:
        List[List[int]]: List of merged bounding boxes.
    """
    merged_bboxes = []
    door_bboxes = sorted(door_bboxes, key=lambda bbox: (bbox[0], bbox[1]))  # Sort by top-left corner
    
    while door_bboxes:
        # Take the first bounding box from the list
        current_bbox = door_bboxes.pop(0)
        x1, y1, x2, y2 = current_bbox
        merged = False
        
        # Check for overlap with other bounding boxes
        for i, (bx1, by1, bx2, by2) in enumerate(door_bboxes):
            # Check if the bounding boxes overlap
            if not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2):  # If they overlap
                # Merge the bounding boxes
                new_bbox = [min(x1, bx1), min(y1, by1), max(x2, bx2), max(y2, by2)]
                door_bboxes.pop(i)
                door_bboxes.append(new_bbox)
                merged = True
                break
        
        if not merged:
            merged_bboxes.append(current_bbox)
    
    # If no overlaps, just add the current bbox as is
    if not merged:
        merged_bboxes.append(current_bbox)
        
    return merged_bboxes


def paint_and_overlay_doors(image_path, corridor_pixels, outdoor_pixels, room_pixels, door_bboxes, buffer_size=20, save_path="overlay1.png"):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert to RGB for easier visualization with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 1: Paint corridor, outdoor, and room pixels in the image
    print("Painting corridor pixels (light green)...")
    for pixels in corridor_pixels:
        for y, x in pixels:
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                image_rgb[y, x] = [144, 238, 144]  # Light green

    print("Painting outdoor pixels (light blue)...")
    for pixels in outdoor_pixels:
        for y, x in pixels:
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                image_rgb[y, x] = [173, 216, 230]  # Light blue

    print("Painting room pixels (light orange)...")
    for pixels in room_pixels:
        for y, x in pixels:
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                image_rgb[y, x] = [252, 224, 169]  # Light orange

    # Save the first image with colorized pixels
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=image.shape[1]/plt.gcf().get_size_inches()[0])
    plt.close()

    # Step 2: Draw doors and extended bounding boxes
    print("Drawing doors and extended bounding boxes...")

    # Create a copy of the image for overlay
    overlay_image = image_rgb.copy()

    for bbox in door_bboxes:
        x1, y1, x2, y2 = bbox

        # Create 4 extended bounding boxes
        top_bbox = [x1, y1 - buffer_size, x2, y1]
        bottom_bbox = [x1, y2, x2, y2 + buffer_size]
        left_bbox = [x1 - buffer_size, y1, x1, y2]
        right_bbox = [x2, y1, x2 + buffer_size, y2]

        # Draw the extended bounding boxes in red (1px thick)
        for extended_bbox in [top_bbox, bottom_bbox, left_bbox, right_bbox]:
            ex1, ey1, ex2, ey2 = extended_bbox
            if 0 <= ex1 < overlay_image.shape[1] and 0 <= ey1 < overlay_image.shape[0]:
                cv2.rectangle(overlay_image, (ex1, ey1), (ex2, ey2), (255, 0, 0), 1)  # Red color

        # Draw the original door bbox in blue (2px thick)
        if 0 <= x1 < overlay_image.shape[1] and 0 <= y1 < overlay_image.shape[0]:
            cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Blue color

    # Save the image with doors and extended bounding boxes
    plt.imshow(overlay_image)
    plt.axis('off')
    plt.savefig("overlay_with_doors.png", bbox_inches='tight', pad_inches=0, dpi=image.shape[1]/plt.gcf().get_size_inches()[0])
    plt.close()

    # Step 3: Categorize and plot door bounding boxes based on extended bbox analysis
    print("Categorizing door bounding boxes...")

    door_image = image_rgb.copy()

    for bbox in door_bboxes:
        x1, y1, x2, y2 = bbox

        # Create opposite extended bounding boxes
        top_bbox = [x1, y1 - buffer_size, x2, y1]
        bottom_bbox = [x1, y2, x2, y2 + buffer_size]
        left_bbox = [x1 - buffer_size, y1, x1, y2]
        right_bbox = [x2, y1, x2 + buffer_size, y2]

        # Check pixel content in the extended bounding boxes
        top_pixels = image_rgb[max(0, y1 - buffer_size):y1, x1:x2]
        bottom_pixels = image_rgb[y2:min(image_rgb.shape[0], y2 + buffer_size), x1:x2]
        left_pixels = image_rgb[y1:y2, max(0, x1 - buffer_size):x1]
        right_pixels = image_rgb[y1:y2, x2:min(image_rgb.shape[1], x2 + buffer_size)]

        top_majority_white = np.mean(np.all(top_pixels == [255, 255, 255], axis=-1)) > 0.5
        bottom_majority_orange = np.mean(np.all(bottom_pixels == [252, 224, 169], axis=-1)) > 0.5
        left_majority_white = np.mean(np.all(left_pixels == [255, 255, 255], axis=-1)) > 0.5
        right_majority_orange = np.mean(np.all(right_pixels == [252, 224, 169], axis=-1)) > 0.5

        if (top_majority_white and bottom_majority_orange) or (left_majority_white and right_majority_orange):
            # Wardrobe door: color pink
            cv2.rectangle(door_image, (x1, y1), (x2, y2), (255, 105, 180), 2)  # Pink color
        else:
            # Default: color blue
            cv2.rectangle(door_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Blue color

    # Save the categorized door image
    plt.imshow(door_image)
    plt.axis('off')
    plt.savefig("categorized_doors.png", bbox_inches='tight', pad_inches=0, dpi=image.shape[1]/plt.gcf().get_size_inches()[0])
    plt.close()

    print("Categorized door image saved.")
