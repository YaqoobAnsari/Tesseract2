import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import cv2
import time
import re
import pandas as pd
from PIL import Image


# Add the required paths to the Python path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_PROJECT_ROOT, "Models", "Text_Models"))
sys.path.append(os.path.join(_PROJECT_ROOT, "utils"))
sys.path.append(os.path.join(_PROJECT_ROOT, "Models", "Interpreter"))
sys.path.append(os.path.join(_PROJECT_ROOT, "Models", "Door_Models"))
from text_interpreter import interpret_bboxes, parse_transition_labels
from door_bboxer import*

# Import functions and classes
from text_bboxer import*
from utils.graph import BuildingGraph
from utils.floodfill import* 
from utils.floodfill import get_room_subnode_candidates
from utils.connectivity import*
from utils.Improve import*
from utils.compute_time_eval import*

def check_image_exists(image_name, input_images_dir):
    """
    Check if the given image exists in the input images directory.
    
    Args:
        image_name (str): Name of the image with the extension.
        input_images_dir (str): Path to the input images directory.

    Returns:
        str: Full path to the image if it exists.

    Raises:
        FileNotFoundError: If the image does not exist.
    """
    image_path = Path(input_images_dir) / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Error: The image '{image_name}' does not exist in '{input_images_dir}'.")
    return str(image_path)

def detect_floor_from_filename(image_name):
    """
    Extract floor number from image filename.
    
    Rules:
    - First word after splitting by spaces indicates the floor
    - "FF" → 1 (First Floor, lowest)
    - "SF" → 2 (Second Floor)
    - "TF" → 3 (Third Floor)
    - Numeric (e.g., "4") → 4, 5, etc.
    
    Args:
        image_name (str): Image filename (e.g., "FF part 1upE.png")
        
    Returns:
        int: Floor number (e.g., 1, 2, 3, etc.)
    """
    # Remove extension and split by spaces
    name_no_ext = os.path.splitext(os.path.basename(image_name))[0]
    parts = name_no_ext.split()
    
    if not parts:
        print(f"Warning: Could not extract floor from '{image_name}', defaulting to floor 1")
        return 1
    
    first_word = parts[0].upper()
    
    # Map floor codes to numeric
    floor_mapping = {
        "FF": 1,  # First Floor (lowest)
        "SF": 2,  # Second Floor
        "TF": 3,  # Third Floor
    }
    
    if first_word in floor_mapping:
        floor_num = floor_mapping[first_word]
        print(f"Detected floor from '{image_name}': '{first_word}' → floor {floor_num}")
        return floor_num
    
    # Check if first word is numeric
    if first_word.isdigit():
        floor_num = int(first_word)
        print(f"Detected floor from '{image_name}': numeric '{first_word}' → floor {floor_num}")
        return floor_num
    
    # Fallback: try to extract number from first word
    import re
    match = re.search(r'(\d+)', first_word)
    if match:
        floor_num = int(match.group(1))
        print(f"Detected floor from '{image_name}': extracted '{floor_num}' → floor {floor_num}")
        return floor_num
    
    # Default fallback
    print(f"Warning: Could not determine floor from '{image_name}' (first word: '{first_word}'), defaulting to floor 1")
    return 1

def make_graph(image_name, floor_id=None, progress_callback=None):
    """
    Main function to check image existence, construct paths, run text detection,
    and save graph-related outputs (plot and JSON).

    Args:
        image_name (str): Name of the image with the extension.
        floor_id (int, optional): Floor number (e.g., 1, 2, 3).
                                  If None, will be detected from filename or default to 1.
        progress_callback (callable, optional): Called with (stage_name: str) at each pipeline stage.
    """
    def _report(stage):
        if progress_callback:
            progress_callback(stage)
    # Define base paths
    base_path = os.getcwd()
    input_images_dir = os.path.join(base_path, "Input_Images")
    model_weights_dir = os.path.join(base_path, "Model_weights")
    results_dir = os.path.join(base_path, "Results")

    time_dir = os.path.join(results_dir, "Time&Meta")
    os.makedirs(time_dir, exist_ok=True)

    time_dir_txt = os.path.join(time_dir, "Text files")
    os.makedirs(time_dir_txt, exist_ok=True)

    time_dir_plots = os.path.join(time_dir, "Time Correlation Plots")
    os.makedirs(time_dir_plots, exist_ok=True)
    # Ensure the image exists
    image_path = check_image_exists(image_name, input_images_dir)
    image_name_no_ext = os.path.splitext(os.path.basename(image_path))[0]
    file_size_bytes = os.path.getsize(image_path)

    with Image.open(image_path) as img:
        width, height = img.size


    timer_file = f"{time_dir_txt}/{image_name_no_ext}_timer_info.txt"
    with open(timer_file, "w") as tf:
        tf.write("Timer & Metadata Information\n")
        tf.write("=================\n")
        tf.write(f"File Size: {file_size_bytes / 1024:.2f} KB\n")   
        tf.write(f"Image Dimensions: {width} x {height} pixels\n")

    # Helper to log time
    def log_time(stage, start_time):
        elapsed_time = time.time() - start_time
        with open(timer_file, "a") as tf:
            tf.write(f"{stage}: {elapsed_time:.2f} seconds\n")

    start_total = time.time()

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    # Construct paths
    plots_dir = os.path.join(results_dir, "Plots")
    json_dir = os.path.join(results_dir, "Json")
    
    json_img_dir = os.path.join(json_dir, f"{image_name_no_ext}")
    os.makedirs(json_img_dir, exist_ok=True)

    graph_plot_dir = os.path.join(plots_dir, "graph_plots")
    os.makedirs(graph_plot_dir, exist_ok=True)
    graph_img_dir = os.path.join(graph_plot_dir, f"{image_name_no_ext}")
    os.makedirs(graph_img_dir, exist_ok=True)

    text_detection_dir = os.path.join(plots_dir, "text_detection")
    os.makedirs(text_detection_dir, exist_ok=True)

    connective_plot_dir = os.path.join(plots_dir, "connective_plots")
    os.makedirs(connective_plot_dir, exist_ok=True)
    connect_img_dir = os.path.join(connective_plot_dir, f"{image_name_no_ext}")
    os.makedirs(connect_img_dir, exist_ok=True)

    test_plot_dir = os.path.join(plots_dir, "test_plots")
    os.makedirs(test_plot_dir, exist_ok=True)
    test_img_dir = os.path.join(test_plot_dir, f"{image_name_no_ext}")
    os.makedirs(test_img_dir, exist_ok=True)

    # Call get_Textboxes to perform text detection
    _report("Detecting text regions")
    start_step = time.time()
    text_file_path = get_Textboxes(image_path, model_weights_dir, text_detection_dir)
    log_time("text detection check", start_step)

    _report("Interpreting text labels")
    start_step = time.time()
    print("\nInterpreting bboxes...")
    room_bboxes, hallway_bboxes, outside_bboxes, transition_bboxes, result_file_path = interpret_bboxes(image_path, text_file_path, plots_dir)
    log_time("Interpreting bboxes check", start_step)
     
    # Initialize the graph
    _report("Initializing graph nodes")
    print("\nInitializing Graph")
    # Detect floor from filename if not provided
    if floor_id is None:
        floor_id = detect_floor_from_filename(image_name)
    else:
        print(f"Using provided floor_id: {floor_id}")
    
    # Convert floor_id to string for BuildingGraph (it stores as string internally)
    floor_id_str = str(floor_id)
    graph = BuildingGraph(default_floor=floor_id_str)
    print(f"Graph initialized with default_floor: {floor_id_str} (floor number: {floor_id})")
    
    # Calculate bounding box centers
    start_step = time.time()
    bbox_centers = graph.calculate_bbox_centers(room_bboxes)
    hallway_bbox_centers = graph.calculate_bbox_centers(hallway_bboxes)
    outside_bbox_centers = graph.calculate_bbox_centers(outside_bboxes)
    transition_bbox_centers = graph.calculate_bbox_centers(transition_bboxes)

    # Add nodes to the graph
    for i, (x, y) in enumerate(bbox_centers):
        node_id = f"room_{i + 1}"
        graph.add_node(node_id, node_type="room", position=(x, y))
    print(f"\nAdded {len(bbox_centers)} room nodes to the graph")

    for i, (x, y) in enumerate(hallway_bbox_centers):
        node_id = f"corridor_main_{i + 1}"
        graph.add_node(node_id, node_type="corridor", position=(x, y))
    print(f"Added {len(hallway_bbox_centers)} corridor nodes to the graph")

    for i, (x, y) in enumerate(outside_bbox_centers):
        node_id = f"outside_main_{i + 1}"
        graph.add_node(node_id, node_type="outside", position=(x, y))
    print(f"Added {len(outside_bbox_centers)} outside nodes to the graph")

    transition_label_by_bbox = parse_transition_labels(result_file_path)
    stairs_count = 0
    elevator_count = 0
    
    for i, ((x, y), bbox) in enumerate(zip(transition_bbox_centers, transition_bboxes), start=1):
        key = tuple(bbox)
        label = transition_label_by_bbox.get(key)

        if label == "stairs":
            stairs_count += 1
            node_id = f"stairs_{stairs_count}"
        elif label == "elevator":
            elevator_count += 1
            node_id = f"elevator_{elevator_count}"
        else:
            node_id = f"transition_{i}"

        graph.add_node(
            node_id,
            node_type="transition",
            position=(x, y),
            floor_id=graph.default_floor,  
        )
    print(f"Added {stairs_count} stair nodes and {elevator_count} elevator nodes to the graph\n")
    
    log_time("graph initialization check", start_step)
    json_output_path = os.path.join(json_img_dir, f"{image_name_no_ext}_ini_graph.json")
    graph.save_to_json(json_output_path) 

    graph_plot_output_path = os.path.join(graph_img_dir, f"{image_name_no_ext}_ini_graph.png")
    graph.plot_on_image(image_path, graph_plot_output_path, display_labels=True, threshold_radius = 20, highlight_regions=False) 
    
    graph_plot_output_path = os.path.join(graph_img_dir, f"{image_name_no_ext}_w_thr_graph.png")
    graph.plot_on_image(image_path, graph_plot_output_path, display_labels=True, threshold_radius = 20, highlight_regions=True) 

    start_step = time.time()
    graph.merge_nearby_nodes(threshold_room=50, threshold_door=20)
    log_time("Merging nodes check", start_step)

    json_output_path = os.path.join(json_img_dir, f"{image_name_no_ext}_final_graph.json")
    graph.save_to_json(json_output_path) 

    graph_plot_output_path = os.path.join(graph_img_dir, f"{image_name_no_ext}_thr_graph.png")
    graph.plot_on_image(image_path, graph_plot_output_path, display_labels=True, threshold_radius = 20, highlight_regions=False) 
 
    _report("Flood filling rooms")
    print("\nFloodfilling rooms prior to door detection")
    start_step = time.time()
    #smart_fill_rooms(image_path, graph, results_dir, radius_threshold=50, node_radius=10)
    smart_output_img_pth, smart_area_file_pth = process_fill_rooms(image_path, graph, results_dir, radius_threshold=70, node_radius=10, fill_mode="smart", point_radius=10, point_step=10, flood_threshold=30)
    flood_output_img_pth, flood_area_file_pth = process_fill_rooms(image_path, graph, results_dir, radius_threshold=70, node_radius=10, fill_mode="flood", point_radius=40, point_step=10, flood_threshold=30)
    log_time("Flood Filling check", start_step)

    print("\nFinding Pixelwise areas")
    room_pixels, outdoor_pixels, corridor_pixels, unmarked_pixels, wall_pixels, thr_img_path = pixelwise_areas(flood_output_img_pth, graph, connect_img_dir,print_tag=False)   

    candidates, overlay_path, room_props, area_map_path = get_room_subnode_candidates(
        image_path,
        graph,
        results_dir=results_dir,
        segmented_map_path=thr_img_path,   # <<< new (strongly recommended)
        spacing_px=60, wall_pad_px=10, jitter_px=4,
        fill_mode="flood", point_radius=10, point_step=10, flood_threshold=30,
        radius_threshold=70, save_overlay=True
    )

    # When adding subnodes (unchanged), also copy room_props onto MAIN room node only:
    for room_id, pts in candidates.items():
        floor = graph.graph.nodes[room_id].get("floor", graph.default_floor)
        # attach properties to the main room node
        if room_id in room_props:
            graph.graph.nodes[room_id]["room_area_px"]   = room_props[room_id]["area_px"]
            graph.graph.nodes[room_id]["room_eq_radius"] = room_props[room_id]["eq_radius_px"]
            graph.graph.nodes[room_id]["room_inradius"]  = room_props[room_id]["inradius_px"]
            graph.graph.nodes[room_id]["room_num_subnode_candidates"] = room_props[room_id]["num_candidates"]
            graph.graph.nodes[room_id]["room_centroid_xy"] = room_props[room_id]["centroid_xy"]

        for i, (x, y) in enumerate(pts, start=1):
            sub_id = f"{room_id}_subnode_{i}"
            graph.add_node(sub_id, node_type="room", position=(x, y), floor_id=floor)
            graph.graph.nodes[sub_id]["is_subnode"] = True
            graph.graph.nodes[sub_id]["parent_room_id"] = room_id
        
    _report("Detecting doors")
    print("\nDetecting doors")
    start_step = time.time()
    door_bbox = detect_doors(image_path,threshold=0.9, chunk_size=300, overlap=75, results_dir=plots_dir)
    print("\nRefining doors")
    door_bbox = refine_door_bboxes(image_path, plots_dir, door_threshold= 20, door_bboxes=door_bbox)  
    log_time("Detecting doors check", start_step)

    _report("Classifying doors")
    print("\nClassifying doors")
    start_step = time.time()
    exit_dbboxes, corridor2corridor_dbboxes, room2corridor_dbboxes, room2room_dbboxes, wardrobe_dbboxes = classify_doors(thr_img_path, door_bbox, connect_img_dir, print_tag=False)
    log_time("Classifying doors check", start_step)
    
    _report("Building room-door connectivity")
    print("\nAdding classified door nodes to graph")
    start_step = time.time()
    graph.add_door_nodes(exit_dbboxes, corridor2corridor_dbboxes, room2corridor_dbboxes, room2room_dbboxes)
    print("\nAdding room to door edges to graph")
    graph.make_room_door_edges(image_path, (room2corridor_dbboxes+room2room_dbboxes+exit_dbboxes))   

    _report("Populating corridor network")
    print("\nAdding corridor nodes to graph")
    corridor_distance = 20
    corridor_pixels = graph.add_corridor_nodes(image_path, corridor_pixels, test_img_dir, dest="corridor", distance=corridor_distance)
    print(f"Added {len(corridor_pixels)} corridor nodes to the graph")
    for i, (y, x) in enumerate(corridor_pixels):
        node_id = f"corridor_connect_{i + 1}"
        node_type = "corridor"  
        graph.add_node(node_id, node_type=node_type, position=(x, y))
    graph.add_corridor_edges(corridor_pixels, distance=corridor_distance)
    
    print("\nAdding outdoor nodes to graph")
    outside_distance = 40
    outdoor_pixels = graph.add_corridor_nodes(image_path, outdoor_pixels, test_img_dir, dest="outside", distance=outside_distance)
    print(f"Added {len(outdoor_pixels)} outside nodes to the graph")
    for i, (y, x) in enumerate(outdoor_pixels):
        node_id = f"outside_connect_{i + 1}"
        node_type = "outside"   
        graph.add_node(node_id, node_type=node_type, position=(x, y))
    graph.add_outdoor_edges(outdoor_pixels, distance=outside_distance)
    log_time("Updating graph nodes check", start_step)

    _report("Funneling room paths to doors")
    print("\nFunneling room families to doors (grid lattice -> shortest paths)...")
    kept = graph.connect_all_families_funnel(spacing_px=60, door_selector="nearest")
    print(f"Kept {kept} intra-room edges across all rooms.")

    _report("Creating edges")
    start_step = time.time()
    graph.connect_hallways()
    graph.connect_doors()
    graph.connect_rooms()
    graph.connect_transitions()
    log_time("Edge creation check", start_step)

    graph_plot_output_path = os.path.join(graph_img_dir, f"{image_name_no_ext}_pre_pruning_wothr_connect_graph.png")
    graph.plot_on_image(image_path, graph_plot_output_path, display_labels=False, threshold_radius = 20, highlight_regions=False)
 
    graph_plot_output_path = os.path.join(graph_img_dir, f"{image_name_no_ext}_pre_pruning_wthr_connect_graph.png")
    graph.plot_on_image(image_path, graph_plot_output_path, display_labels=False, threshold_radius = 20, highlight_regions=True) 
 
    json_output_path = os.path.join(json_img_dir, f"{image_name_no_ext}_pre_pruning.json")
    graph.save_to_json(json_output_path) 
      
    json_file_size_kb_bfr = os.path.getsize(json_output_path) / 1024

    tot_graph_nodes = graph.return_graph_size()
     
    _report("Pruning graph")
    start_step = time.time()
    graph.connect_all_rooms(image_path, graph_img_dir)
    log_time("Graph pruning check", start_step)

    mod_graph_nodes = graph.return_graph_size() 

    graph_plot_output_path = os.path.join(graph_img_dir, f"{image_name_no_ext}_post_pruning_wothr_connect_graph.png")
    graph.plot_on_image(image_path, graph_plot_output_path, display_labels=False, threshold_radius = 20, highlight_regions=False)
 
    graph_plot_output_path = os.path.join(graph_img_dir, f"{image_name_no_ext}_post_pruning_wthr_connect_graph.png")
    graph.plot_on_image(image_path, graph_plot_output_path, display_labels=False, threshold_radius = 20, highlight_regions=True) 

    white_image_path = os.path.join(os.path.dirname(image_path), "white_background.png")
    original_image = Image.open(image_path)
    width, height = original_image.size
    
    white_image = Image.new("RGB", (width, height), (255, 255, 255))  
    white_image.save(white_image_path)   

    graph_plot_output_path = os.path.join(graph_img_dir, f"{image_name_no_ext}_post_pruning_blank_connect_graph_1.png")
    graph.plot_on_image(white_image_path, graph_plot_output_path, display_labels=False, threshold_radius=20, highlight_regions=True)
    
    graph_plot_output_path = os.path.join(graph_img_dir, f"{image_name_no_ext}_post_pruning_blank_connect_graph_2.png")
    graph.plot_on_image(white_image_path, graph_plot_output_path, display_labels=False, threshold_radius=20, highlight_regions=False)
 
    graph_plot_output_path = os.path.join(graph_img_dir, f"{image_name_no_ext}_post_pruning_blank_connect_graph_3.png")
    graph.plot_on_image(white_image_path, graph_plot_output_path, display_labels=True, threshold_radius=20, highlight_regions=False)
 
    json_output_path = os.path.join(json_img_dir, f"{image_name_no_ext}_post_pruning.json")
    graph.save_to_json(json_output_path) 

    json_file_size_kb_aft = os.path.getsize(json_output_path) / 1024
    
    total_time = time.time() - start_total
    with open(timer_file, "a") as tf:
        tf.write(f"Total Time: {total_time:.2f} seconds\n")
        tf.write(f"Total graph nodes (before pruning): {tot_graph_nodes}\n")
        tf.write(f"JSON File Size (before pruning): {json_file_size_kb_bfr:.2f} KB\n")
        tf.write(f"Total graph nodes (after pruning): {mod_graph_nodes}\n")
        tf.write(f"JSON File Size (after pruning): {json_file_size_kb_aft:.2f} KB\n")
    
    df_results = analyze_timer_files(time_dir_txt, time_dir_plots)
    
    return graph, image_name_no_ext, floor_id  # floor_id is int


# ============================================================================
# MULTI-FLOOR PROCESSING FUNCTIONS (READY FOR USE)
# ============================================================================

def check_results_exist(image_name, results_dir):
    """
    Check if results directory exists for a given image.
    
    Args:
        image_name (str): Name of the image with extension
        results_dir (str): Path to Results directory
        
    Returns:
        bool: True if results directory exists, False otherwise
    """
    image_name_no_ext = os.path.splitext(os.path.basename(image_name))[0]
    json_dir = os.path.join(results_dir, "Json", image_name_no_ext)
    return os.path.exists(json_dir) and os.path.isdir(json_dir)


def validate_transition_mapping(transition_mapping, input_images_dir, results_dir):
    """
    Validate transition mapping before processing.
    
    Checks:
    1. All images exist in Input_Images directory
    2. All nodes are transition nodes (stairs/elevators)
    3. Floor order constraints (no skipping floors)
    4. One-to-one constraint (one transition on floor X can only connect to one transition on floor Y)
    
    Args:
        transition_mapping (dict): Transition mapping to validate
        input_images_dir (str): Path to Input_Images directory
        results_dir (str): Path to Results directory
        
    Returns:
        tuple: (is_valid, errors, warnings)
            - is_valid (bool): True if mapping is valid
            - errors (list): List of error messages
            - warnings (list): List of warning messages
    """
    errors = []
    warnings = []
    
    if not transition_mapping:
        return True, errors, warnings
    
    # Collect all unique images and floors from mapping
    all_images = set()
    all_floors = set()
    node_references = {}  # Track node references: (floor, image, node_id)
    
    for (src_floor, src_image, src_node_id), targets in transition_mapping.items():
        all_images.add(src_image)
        all_floors.add(src_floor)
        node_references[(src_floor, src_image, src_node_id)] = "source"
        
        for tgt_floor, tgt_image, tgt_node_id in targets:
            all_images.add(tgt_image)
            all_floors.add(tgt_floor)
            node_references[(tgt_floor, tgt_image, tgt_node_id)] = "target"
    
    # Check 1: All images exist
    print("\n" + "="*70)
    print("VALIDATING TRANSITION MAPPING")
    print("="*70)
    print("\n[1/5] Checking image existence...")
    
    missing_images = []
    for image_name in all_images:
        image_path = os.path.join(input_images_dir, image_name)
        if not os.path.exists(image_path):
            missing_images.append(image_name)
            errors.append(f"Image '{image_name}' not found in '{input_images_dir}'")
    
    if missing_images:
        print(f"  ✗ ERROR: {len(missing_images)} image(s) not found:")
        for img in missing_images:
            print(f"    - {img}")
    else:
        print(f"  ✓ All {len(all_images)} images exist")
    
    # Check 2: Floor order constraints
    print("\n[2/5] Checking floor order constraints...")
    sorted_floors = sorted(all_floors)
    
    # Check for direct connections between non-adjacent floors
    floor_connections = {}  # {floor: set of connected floors}
    
    for (src_floor, src_image, src_node_id), targets in transition_mapping.items():
        if src_floor not in floor_connections:
            floor_connections[src_floor] = set()
        
        for tgt_floor, tgt_image, tgt_node_id in targets:
            floor_connections[src_floor].add(tgt_floor)
            
            # Check if floors are adjacent (difference of 1)
            floor_diff = abs(tgt_floor - src_floor)
            if floor_diff > 1:
                warnings.append(
                    f"Non-adjacent floor connection: Floor {src_floor} → Floor {tgt_floor} "
                    f"(difference: {floor_diff}). This may violate building semantics."
                )
                print(f"  ⚠ WARNING: Floor {src_floor} connects directly to Floor {tgt_floor} (non-adjacent)")
    
    if not warnings:
        print(f"  ✓ All floor connections are adjacent")
    
    # Check 3: One-to-one constraint
    print("\n[3/5] Checking one-to-one constraint...")
    # For each floor pair (src, tgt), ensure only one transition connects them
    floor_pair_transitions = {}  # {(src_floor, tgt_floor): [list of transition pairs]}
    
    for (src_floor, src_image, src_node_id), targets in transition_mapping.items():
        for tgt_floor, tgt_image, tgt_node_id in targets:
            pair = (src_floor, tgt_floor)
            if pair not in floor_pair_transitions:
                floor_pair_transitions[pair] = []
            floor_pair_transitions[pair].append((src_node_id, tgt_node_id))
    
    one_to_one_violations = []
    for (src_floor, tgt_floor), transitions in floor_pair_transitions.items():
        if len(transitions) > 1:
            one_to_one_violations.append((src_floor, tgt_floor, transitions))
            errors.append(
                f"One-to-one violation: Floor {src_floor} → Floor {tgt_floor} has {len(transitions)} "
                f"transition connections: {transitions}. Only one transition can connect a floor pair."
            )
    
    if one_to_one_violations:
        print(f"  ✗ ERROR: {len(one_to_one_violations)} one-to-one constraint violation(s):")
        for src_floor, tgt_floor, transitions in one_to_one_violations:
            print(f"    Floor {src_floor} → Floor {tgt_floor}: {len(transitions)} connections")
            for src_node, tgt_node in transitions:
                print(f"      - {src_node} → {tgt_node}")
    else:
        print(f"  ✓ One-to-one constraint satisfied for all floor pairs")
    
    # Check 4: Node type validation (will be done after graphs are loaded)
    print("\n[4/5] Node type validation will be performed after graphs are loaded")
    
    # Check 5: Image name consistency
    print("\n[5/5] Checking image name consistency...")
    # Verify that images match their detected floors
    image_floor_mismatches = []
    for image_name in all_images:
        detected_floor = detect_floor_from_filename(image_name)
        # Find which floor this image is referenced as in the mapping
        referenced_floors = set()
        for (src_floor, src_image, _), targets in transition_mapping.items():
            if src_image == image_name:
                referenced_floors.add(src_floor)
            for tgt_floor, tgt_image, _ in targets:
                if tgt_image == image_name:
                    referenced_floors.add(tgt_floor)
        
        # Check if detected floor matches any referenced floor
        if referenced_floors and detected_floor not in referenced_floors:
            image_floor_mismatches.append((image_name, detected_floor, referenced_floors))
            warnings.append(
                f"Image '{image_name}' detected as Floor {detected_floor}, but referenced as "
                f"Floor(s) {referenced_floors} in mapping"
            )
    
    if image_floor_mismatches:
        print(f"  ⚠ WARNING: {len(image_floor_mismatches)} image-floor mismatch(es):")
        for img, detected, referenced in image_floor_mismatches:
            print(f"    {img}: detected={detected}, referenced={referenced}")
    else:
        print(f"  ✓ All image-floor references are consistent")
    
    is_valid = len(errors) == 0
    
    print(f"\n{'='*70}")
    if is_valid:
        print("✓ VALIDATION PASSED")
        if warnings:
            print(f"  ({len(warnings)} warning(s) - see above)")
    else:
        print("✗ VALIDATION FAILED")
        print(f"  {len(errors)} error(s) found - see above")
    print(f"{'='*70}\n")
    
    return is_valid, errors, warnings


def ensure_floor_graph_exists(image_name, input_images_dir, results_dir):
    """
    Ensure that a floor graph exists for the given image.
    If image exists but graph doesn't, process it first.
    
    Args:
        image_name (str): Name of the image with extension
        input_images_dir (str): Path to Input_Images directory
        results_dir (str): Path to Results directory
        
    Returns:
        tuple: (floor_graph, floor_num, image_path) or (None, None, None) if image doesn't exist
    """
    # Check if image exists
    image_path = os.path.join(input_images_dir, image_name)
    if not os.path.exists(image_path):
        print(f"  ✗ ERROR: Image '{image_name}' not found in '{input_images_dir}'")
        return None, None, None
    
    # Detect floor
    floor_num = detect_floor_from_filename(image_name)
    
    # Check if results exist
    if check_results_exist(image_name, results_dir):
        print(f"  ✓ Results exist for '{image_name}' (Floor {floor_num})")
        # Load existing graph (we'll process it fresh anyway to ensure consistency)
        # For now, we'll always reprocess to ensure graphs are up-to-date
        print(f"  → Processing '{image_name}' to create/update graph...")
        floor_graph, image_name_no_ext, detected_floor = make_graph(image_name, floor_id=floor_num)
        return floor_graph, floor_num, image_path
    else:
        print(f"  → No results found for '{image_name}' (Floor {floor_num})")
        print(f"  → Processing '{image_name}' to create graph...")
        floor_graph, image_name_no_ext, detected_floor = make_graph(image_name, floor_id=floor_num)
        return floor_graph, floor_num, image_path


def normalize_image_coordinates(image_path, position):
    '''
    Normalize coordinates to account for different image scales.
    Returns normalized (x, y) in [0, 1] range.
    
    Args:
        image_path (str): Path to the image
        position (tuple): (x, y) pixel coordinates
        
    Returns:
        tuple: Normalized (x_norm, y_norm) coordinates
    '''
    with Image.open(image_path) as img:
        width, height = img.size
    x_norm = position[0] / width if width > 0 else 0.0
    y_norm = position[1] / height if height > 0 else 0.0
    return (x_norm, y_norm)


def align_transitions_spatially(transition1_pos, transition2_pos, image1_path, image2_path, 
                                tolerance=0.02):
    '''
    Check if two transitions are vertically aligned (same x,y position regardless of scale).
    Uses normalized coordinates for scale-invariant comparison.
    
    Args:
        transition1_pos (tuple): (x, y) position of transition in image1
        transition2_pos (tuple): (x, y) position of transition in image2
        image1_path (str): Path to first image
        image2_path (str): Path to second image
        tolerance (float): Maximum allowed difference in normalized coordinates (default: 0.02 = 2%)
        
    Returns:
        bool: True if transitions are vertically aligned
    '''
    norm1 = normalize_image_coordinates(image1_path, transition1_pos)
    norm2 = normalize_image_coordinates(image2_path, transition2_pos)
    
    dx = abs(norm1[0] - norm2[0])
    dy = abs(norm1[1] - norm2[1])
    
    is_aligned = (dx <= tolerance) and (dy <= tolerance)
    
    if is_aligned:
        print(f"  ✓ Transitions aligned: ({norm1[0]:.4f}, {norm1[1]:.4f}) vs ({norm2[0]:.4f}, {norm2[1]:.4f})")
        print(f"    Difference: dx={dx:.4f}, dy={dy:.4f} (tolerance={tolerance})")
    else:
        print(f"  ✗ Transitions NOT aligned: ({norm1[0]:.4f}, {norm1[1]:.4f}) vs ({norm2[0]:.4f}, {norm2[1]:.4f})")
        print(f"    Difference: dx={dx:.4f}, dy={dy:.4f} (tolerance={tolerance})")
    
    return is_aligned


def merge_floor_graphs(floor_graphs, floor_image_paths):
    '''
    Merge multiple floor graphs into a single graph, preserving floor information.
    Node IDs are prefixed with floor number to avoid conflicts.
    
    Args:
        floor_graphs (dict): {floor_num: BuildingGraph} - Graphs for each floor (floor_num is int)
        floor_image_paths (dict): {floor_num: image_path} - Image paths for each floor
        
    Returns:
        BuildingGraph: Merged graph with all floors
    '''
    print("\n" + "="*70)
    print("MERGING MULTI-FLOOR GRAPHS")
    print("="*70)
    
    merged_graph = BuildingGraph(default_floor="MULTI_FLOOR")
    total_nodes = 0
    total_edges = 0
    
    # Sort floors by floor number for consistent processing
    sorted_floors = sorted(floor_graphs.keys())
    print(f"Processing floors in order: {sorted_floors}")
    
    # Add all nodes from all floors with floor prefix
    for floor_num in sorted_floors:
        floor_graph = floor_graphs[floor_num]
        floor_id_str = str(floor_num)  # Convert to string for node prefix
        
        print(f"\nProcessing floor {floor_num}:")
        floor_nodes = 0
        floor_edges = len(floor_graph.graph.edges())
        
        for node_id, node_data in floor_graph.graph.nodes(data=True):
            # Create unique node ID with floor prefix (e.g., "1_room_1", "2_stairs_1")
            prefixed_node_id = f"{floor_num}_{node_id}"
            
            # Preserve original floor information (store as string in graph)
            merged_graph.add_node(
                prefixed_node_id,
                node_type=node_data['type'],
                position=node_data['position'],
                floor_id=floor_id_str  # Store as string in graph
            )
            
            # Copy all other attributes
            for key, value in node_data.items():
                if key not in ['type', 'position', 'floor']:
                    merged_graph.graph.nodes[prefixed_node_id][key] = value
            
            floor_nodes += 1
        
        # Add edges within this floor (with prefixed node IDs)
        for u, v, edge_data in floor_graph.graph.edges(data=True):
            prefixed_u = f"{floor_num}_{u}"
            prefixed_v = f"{floor_num}_{v}"
            
            if merged_graph.graph.has_node(prefixed_u) and merged_graph.graph.has_node(prefixed_v):
                merged_graph.add_edge(prefixed_u, prefixed_v, weight=edge_data.get('weight'))
                # Copy edge attributes
                for key, value in edge_data.items():
                    if key not in ['weight', 'distance']:
                        merged_graph.graph[prefixed_u][prefixed_v][key] = value
        
        total_nodes += floor_nodes
        total_edges += floor_edges
        print(f"  Added {floor_nodes} nodes and {floor_edges} edges from floor {floor_num}")
    
    print(f"\nMerged graph summary:")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Total edges: {total_edges}")
    print(f"  Floors: {sorted_floors}")
    
    return merged_graph


def connect_transitions_across_floors(merged_graph, floor_graphs, floor_image_paths, 
                                     transition_mapping, spatial_tolerance=0.02):
    '''
    Connect transition nodes across floors based on manual mapping and spatial alignment.
    
    Args:
        merged_graph (BuildingGraph): Merged multi-floor graph
        floor_graphs (dict): {floor_num: BuildingGraph} - Original floor graphs (floor_num is int)
        floor_image_paths (dict): {floor_num: image_path} - Image paths for each floor
        transition_mapping (dict): Manual mapping of transitions
            Format: {
                (source_floor_num, source_image, source_node_id): [
                    (target_floor_num, target_image, target_node_id),
                    ...
                ]
            }
            Example: {
                (1, "FF part 1upE.png", "stairs_1"): [
                    (2, "SF part 1upE.png", "stairs_1"),
                    (3, "TF part 1upE.png", "stairs_1")
                ]
            }
            Note: floor_num is integer (1, 2, 3, etc.)
        spatial_tolerance (float): Tolerance for spatial alignment check (default: 0.02)
        
    Returns:
        int: Number of inter-floor connections created
    '''
    print("\n" + "="*70)
    print("CONNECTING TRANSITIONS ACROSS FLOORS")
    print("="*70)
    
    connections_created = 0
    connections_failed = 0
    
    # Track connections per floor pair to enforce one-to-one constraint
    floor_pair_connections = {}  # {(src_floor, tgt_floor): (src_node_id, tgt_node_id)}
    
    for (src_floor_num, src_image, src_node_id), targets in transition_mapping.items():
        print(f"\nProcessing transition mapping:")
        print(f"  Source: Floor {src_floor_num} / {src_image} / {src_node_id}")
        
        # Get source node from original floor graph
        if src_floor_num not in floor_graphs:
            print(f"  ✗ ERROR: Floor {src_floor_num} not found in floor_graphs")
            print(f"    Available floors: {list(floor_graphs.keys())}")
            connections_failed += len(targets)
            continue
        
        src_graph = floor_graphs[src_floor_num]
        if src_node_id not in src_graph.graph.nodes:
            print(f"  ✗ ERROR: Node '{src_node_id}' not found in floor {src_floor_num} graph")
            # List available transition nodes
            transition_nodes = [n for n in src_graph.graph.nodes() 
                              if src_graph.graph.nodes[n].get('type') == 'transition']
            if transition_nodes:
                print(f"    Available transition nodes: {transition_nodes[:10]}")
            else:
                print(f"    No transition nodes found on this floor")
            connections_failed += len(targets)
            continue
        
        src_node_data = src_graph.graph.nodes[src_node_id]
        src_node_type = src_node_data.get('type')
        
        # Validate node type - must be transition
        if src_node_type != 'transition':
            print(f"  ✗ ERROR: Source node '{src_node_id}' is not a transition node (type: '{src_node_type}')")
            print(f"    Only transition nodes (stairs/elevators) can be connected across floors")
            connections_failed += len(targets)
            continue
        
        src_pos = src_node_data.get('position')
        src_image_path = floor_image_paths.get(src_floor_num)
        
        if src_pos is None:
            print(f"  ✗ ERROR: Source node '{src_node_id}' has no position")
            connections_failed += len(targets)
            continue
        
        if src_image_path is None:
            print(f"  ✗ ERROR: No image path found for floor {src_floor_num}")
            print(f"    Available floors: {list(floor_image_paths.keys())}")
            connections_failed += len(targets)
            continue
        
        src_prefixed_id = f"{src_floor_num}_{src_node_id}"
        
        if src_prefixed_id not in merged_graph.graph.nodes:
            print(f"  ✗ ERROR: Prefixed source node '{src_prefixed_id}' not found in merged graph")
            connections_failed += len(targets)
            continue
        
        # Connect to each target
        for tgt_floor_num, tgt_image, tgt_node_id in targets:
            print(f"  Target: Floor {tgt_floor_num} / {tgt_image} / {tgt_node_id}")
            
            # Check floor order constraint
            floor_diff = abs(tgt_floor_num - src_floor_num)
            if floor_diff == 0:
                print(f"    ✗ ERROR: Cannot connect node to same floor ({src_floor_num})")
                connections_failed += 1
                continue
            
            if floor_diff > 1:
                print(f"    ⚠ WARNING: Non-adjacent floor connection (Floor {src_floor_num} → Floor {tgt_floor_num})")
                print(f"      This violates building semantics - floors should connect sequentially")
                # Continue anyway but warn
            
            # Get target node from original floor graph
            if tgt_floor_num not in floor_graphs:
                print(f"    ✗ ERROR: Floor {tgt_floor_num} not found in floor_graphs")
                print(f"      Available floors: {list(floor_graphs.keys())}")
                connections_failed += 1
                continue
            
            tgt_graph = floor_graphs[tgt_floor_num]
            if tgt_node_id not in tgt_graph.graph.nodes:
                print(f"    ✗ ERROR: Node '{tgt_node_id}' not found in floor {tgt_floor_num} graph")
                # List available transition nodes
                transition_nodes = [n for n in tgt_graph.graph.nodes() 
                                  if tgt_graph.graph.nodes[n].get('type') == 'transition']
                if transition_nodes:
                    print(f"      Available transition nodes: {transition_nodes[:10]}")
                else:
                    print(f"      No transition nodes found on this floor")
                connections_failed += 1
                continue
            
            tgt_node_data = tgt_graph.graph.nodes[tgt_node_id]
            tgt_node_type = tgt_node_data.get('type')
            
            # Validate node type - must be transition
            if tgt_node_type != 'transition':
                print(f"    ✗ ERROR: Target node '{tgt_node_id}' is not a transition node (type: '{tgt_node_type}')")
                print(f"      Only transition nodes (stairs/elevators) can be connected across floors")
                connections_failed += 1
                continue
            
            tgt_pos = tgt_node_data.get('position')
            tgt_image_path = floor_image_paths.get(tgt_floor_num)
            
            if tgt_pos is None:
                print(f"    ✗ ERROR: Target node '{tgt_node_id}' has no position")
                connections_failed += 1
                continue
            
            if tgt_image_path is None:
                print(f"    ✗ ERROR: No image path found for floor {tgt_floor_num}")
                print(f"      Available floors: {list(floor_image_paths.keys())}")
                connections_failed += 1
                continue
            
            tgt_prefixed_id = f"{tgt_floor_num}_{tgt_node_id}"
            
            if tgt_prefixed_id not in merged_graph.graph.nodes:
                print(f"    ✗ ERROR: Prefixed target node '{tgt_prefixed_id}' not found in merged graph")
                connections_failed += 1
                continue
            
            # Enforce one-to-one constraint: check if this floor pair already has a connection
            floor_pair = (src_floor_num, tgt_floor_num)
            if floor_pair in floor_pair_connections:
                existing_src, existing_tgt = floor_pair_connections[floor_pair]
                print(f"    ✗ ERROR: One-to-one constraint violation!")
                print(f"      Floor {src_floor_num} → Floor {tgt_floor_num} already connected via:")
                print(f"        {existing_src} → {existing_tgt}")
                print(f"      Cannot add another connection: {src_node_id} → {tgt_node_id}")
                print(f"      Each floor pair can only have ONE transition connection")
                connections_failed += 1
                continue
            
            # Record this connection
            floor_pair_connections[floor_pair] = (src_node_id, tgt_node_id)
            
            # Check spatial alignment
            print(f"    Checking spatial alignment...")
            is_aligned = align_transitions_spatially(
                src_pos, tgt_pos, src_image_path, tgt_image_path, 
                tolerance=spatial_tolerance
            )
            
            if not is_aligned:
                print(f"    ⚠ WARNING: Transitions are not spatially aligned!")
                print(f"    Proceeding anyway based on manual mapping...")
            
            # Create inter-floor edge
            if merged_graph.graph.has_edge(src_prefixed_id, tgt_prefixed_id):
                print(f"    ⚠ Edge already exists between {src_prefixed_id} and {tgt_prefixed_id}")
            else:
                # Calculate distance (use normalized coordinates for consistency)
                src_norm = normalize_image_coordinates(src_image_path, src_pos)
                tgt_norm = normalize_image_coordinates(tgt_image_path, tgt_pos)
                # Use a fixed weight for inter-floor transitions (e.g., 1000 units)
                # This represents vertical movement between floors
                inter_floor_weight = 1000.0
                
                merged_graph.add_edge(
                    src_prefixed_id, 
                    tgt_prefixed_id, 
                    weight=inter_floor_weight
                )
                # Mark as inter-floor edge
                merged_graph.graph[src_prefixed_id][tgt_prefixed_id]['edge_type'] = 'inter_floor'
                merged_graph.graph[src_prefixed_id][tgt_prefixed_id]['spatial_aligned'] = is_aligned
                
                print(f"    ✓ Created inter-floor connection: {src_prefixed_id} ↔ {tgt_prefixed_id}")
                print(f"      Weight: {inter_floor_weight} (inter-floor transition)")
                connections_created += 1
    
    print(f"\nInter-floor connection summary:")
    print(f"  Connections created: {connections_created}")
    print(f"  Connections failed: {connections_failed}")
    
    return connections_created


def process_multi_floor(image_names, transition_mapping=None, spatial_tolerance=0.02):
    '''
    Process multiple floorplan images and create a unified multi-floor graph.
    
    Args:
        image_names (list): List of image filenames (e.g., ["FF part 1upE.png", "SF part 1upE.png"])
        transition_mapping (dict, optional): Manual mapping of transitions across floors
            Format: {
                (source_floor_num, source_image, source_node_id): [
                    (target_floor_num, target_image, target_node_id),
                    ...
                ]
            }
            Example: {
                (1, "FF part 1upE.png", "stairs_1"): [
                    (2, "SF part 1upE.png", "stairs_1"),
                    (3, "TF part 1upE.png", "stairs_1")
                ]
            }
            Note: floor_num is integer (1, 2, 3, etc.)
        spatial_tolerance (float): Tolerance for spatial alignment check (default: 0.02)
        
    Returns:
        tuple: (merged_graph, floor_graphs, floor_image_paths)
            - merged_graph: Unified BuildingGraph with all floors
            - floor_graphs: Dict {floor_num: BuildingGraph} of individual floor graphs (floor_num is int)
            - floor_image_paths: Dict {floor_num: image_path} mapping floor number to image path
    '''
    print("\n" + "="*70)
    print("MULTI-FLOOR PROCESSING")
    print("="*70)
    
    base_path = os.getcwd()
    input_images_dir = os.path.join(base_path, "Input_Images")
    results_dir = os.path.join(base_path, "Results")
    
    # Step 1: Validate transition mapping if provided
    if transition_mapping:
        is_valid, errors, warnings = validate_transition_mapping(
            transition_mapping, input_images_dir, results_dir
        )
        
        if not is_valid:
            print("\n✗ TRANSITION MAPPING VALIDATION FAILED")
            print("Cannot proceed with invalid mapping. Please fix the errors above.")
            raise ValueError(f"Transition mapping validation failed with {len(errors)} error(s)")
        
        if warnings:
            print(f"\n⚠ {len(warnings)} warning(s) found - proceeding anyway")
        
        # Collect all unique images from mapping
        all_mapping_images = set()
        for (src_floor, src_image, _), targets in transition_mapping.items():
            all_mapping_images.add(src_image)
            for tgt_floor, tgt_image, _ in targets:
                all_mapping_images.add(tgt_image)
        
        # Ensure all images from mapping are in image_names
        missing_from_list = all_mapping_images - set(image_names)
        if missing_from_list:
            print(f"\n⚠ Adding {len(missing_from_list)} image(s) from mapping to processing list:")
            for img in missing_from_list:
                print(f"  - {img}")
            image_names = list(set(image_names) | missing_from_list)
    
    print(f"\nProcessing {len(image_names)} floorplan images...")
    
    start_multi_total = time.time()
    
    floor_graphs = {}
    floor_image_paths = {}
    
    # Step 2: Process each floor (ensure graphs exist)
    for image_name in image_names:
        print(f"\n{'='*70}")
        print(f"Processing image: {image_name}")
        print(f"{'='*70}")
        
        start_floor = time.time()
        
        # Ensure graph exists (will process if needed)
        floor_graph, floor_num, image_path = ensure_floor_graph_exists(
            image_name, input_images_dir, results_dir
        )
        
        if floor_graph is None:
            print(f"  ✗ ERROR: Failed to process '{image_name}' - skipping")
            continue
        
        # Store results (use floor_num as key)
        floor_graphs[floor_num] = floor_graph
        floor_image_paths[floor_num] = image_path
        
        floor_time = time.time() - start_floor
        print(f"\nCompleted processing floor {floor_num} in {floor_time:.2f} seconds")
        print(f"  Nodes: {floor_graph.return_graph_size()}")
        print(f"  Edges: {len(floor_graph.graph.edges())}")
    
    # Merge all floor graphs
    print(f"\n{'='*70}")
    print("MERGING FLOOR GRAPHS")
    print(f"{'='*70}")
    start_merge = time.time()
    
    merged_graph = merge_floor_graphs(floor_graphs, floor_image_paths)
    
    merge_time = time.time() - start_merge
    print(f"Merging completed in {merge_time:.2f} seconds")
    
    # Connect transitions across floors
    if transition_mapping:
        print(f"\n{'='*70}")
        print("CONNECTING TRANSITIONS ACROSS FLOORS")
        print(f"{'='*70}")
        start_connect = time.time()
        
        connections = connect_transitions_across_floors(
            merged_graph, floor_graphs, floor_image_paths,
            transition_mapping, spatial_tolerance=spatial_tolerance
        )
        
        connect_time = time.time() - start_connect
        print(f"Inter-floor connection completed in {connect_time:.2f} seconds")
        print(f"  Created {connections} inter-floor connections")
    else:
        print("\n⚠ No transition mapping provided - skipping inter-floor connections")
        print("  Provide transition_mapping to connect floors via transitions")
    
    # Save merged graph
    json_dir = os.path.join(results_dir, "Json")
    merged_json_dir = os.path.join(json_dir, "MULTI_FLOOR")
    os.makedirs(merged_json_dir, exist_ok=True)
    
    merged_json_path = os.path.join(merged_json_dir, "merged_multi_floor_graph.json")
    merged_graph.save_to_json(merged_json_path)
    
    json_size_kb = os.path.getsize(merged_json_path) / 1024
    print(f"\nMerged graph saved to: {merged_json_path}")
    print(f"  JSON size: {json_size_kb:.2f} KB")
    print(f"  Total nodes: {merged_graph.return_graph_size()}")
    print(f"  Total edges: {len(merged_graph.graph.edges())}")
    
    total_time = time.time() - start_multi_total
    print(f"\n{'='*70}")
    print(f"MULTI-FLOOR PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"  Per-floor processing: {total_time - merge_time - (connect_time if transition_mapping else 0):.2f} seconds")
    print(f"  Merging: {merge_time:.2f} seconds")
    if transition_mapping:
        print(f"  Inter-floor connections: {connect_time:.2f} seconds")
    
    return merged_graph, floor_graphs, floor_image_paths


# Example usage (commented out - ready for use when multi-floor images are available):
"""
# Example: Process 3 floors and connect transitions
image_names = [
    "FF part 1upE.png",  # Floor 1
    "SF part 1upE.png",  # Floor 2
    "TF part 1upE.png"   # Floor 3
]

# Manual transition mapping (floor numbers are integers: 1, 2, 3, etc.)
# Format: (source_floor_num, source_image, source_node_id): [(target_floor_num, target_image, target_node_id), ...]
transition_mapping = {
    # stairs_1 connects Floor 1 → Floor 2 → Floor 3
    (1, "FF part 1upE.png", "stairs_1"): [
        (2, "SF part 1upE.png", "stairs_1"),
        (3, "TF part 1upE.png", "stairs_1")
    ],
    # stairs_2 connects Floor 1 → Floor 2 → Floor 3
    (1, "FF part 1upE.png", "stairs_2"): [
        (2, "SF part 1upE.png", "stairs_2"),
        (3, "TF part 1upE.png", "stairs_2")
    ],
    # elevator_1 connects all floors
    (1, "FF part 1upE.png", "elevator_1"): [
        (2, "SF part 1upE.png", "elevator_1"),
        (3, "TF part 1upE.png", "elevator_1")
    ]
}

# Process multi-floor
merged_graph, floor_graphs, floor_image_paths = process_multi_floor(
    image_names,
    transition_mapping=transition_mapping,
    spatial_tolerance=0.02
)
"""


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run text detection on a given image and generate graph outputs.",
        epilog="Example usage: python main.py image_name.png"
    )
    parser.add_argument(
        "image_name",
        type=str,
        nargs="?",
        help="Name of the image (with extension) in the Input_Images folder."
    )

    # Parse arguments
    args = parser.parse_args()

    # Check if image_name is provided
    if not args.image_name:
        parser.error("The following argument is required: image_name (name of the image with extension)")

    try:
        # Run the main function
        make_graph(args.image_name)
    except FileNotFoundError as e:
        # Print the error and exit
        print(e)
        sys.exit(1)
