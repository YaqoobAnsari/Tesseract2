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
sys.path.append("/data1/yansari/cad2map/Tesseract++/Models/Text_Models")
sys.path.append("./utils")  # Add the utils folder to the Python path
sys.path.append("/data1/yansari/cad2map/Tesseract++/Models/Interpreter")
sys.path.append("/data1/yansari/cad2map/Tesseract++/Models/Door_Models") 
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

def make_graph(image_name):
    """
    Main function to check image existence, construct paths, run text detection, 
    and save graph-related outputs (plot and JSON).

    Args:
        image_name (str): Name of the image with the extension.
    """
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
    start_step = time.time()
    text_file_path = get_Textboxes(image_path, model_weights_dir, text_detection_dir)
    log_time("text detection check", start_step)

    start_step = time.time()
    print("\nInterpreting bboxes...")
    room_bboxes, hallway_bboxes, outside_bboxes, transition_bboxes, result_file_path = interpret_bboxes(image_path, text_file_path, plots_dir)
    log_time("Interpreting bboxes check", start_step)
     
    # Initialize the graph
    print("\nInitializing Graph")
    graph = BuildingGraph(default_floor="Ground_Floor")
    
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
        
    print("\nDetecting doors")
    start_step = time.time()
    door_bbox = detect_doors(image_path,threshold=0.9, chunk_size=300, overlap=75, results_dir=plots_dir)
    print("\nRefining doors")
    door_bbox = refine_door_bboxes(image_path, plots_dir, door_threshold= 20, door_bboxes=door_bbox)  
    log_time("Detecting doors check", start_step)

    print("\nClassifying doors")
    start_step = time.time()
    exit_dbboxes, corridor2corridor_dbboxes, room2corridor_dbboxes, room2room_dbboxes, wardrobe_dbboxes = classify_doors(thr_img_path, door_bbox, connect_img_dir, print_tag=False)
    log_time("Classifying doors check", start_step)
    
    print("\nAdding classified door nodes to graph")
    start_step = time.time()
    graph.add_door_nodes(exit_dbboxes, corridor2corridor_dbboxes, room2corridor_dbboxes, room2room_dbboxes)
    print("\nAdding room to door edges to graph")
    graph.make_room_door_edges(image_path, (room2corridor_dbboxes+room2room_dbboxes+exit_dbboxes))   

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

    print("\nFunneling room families to doors (grid lattice -> shortest paths)...")
    kept = graph.connect_all_families_funnel(spacing_px=60, door_selector="nearest")
    print(f"Kept {kept} intra-room edges across all rooms.")

    start_step = time.time()
    graph.connect_hallways()
    graph.connect_doors()
    graph.connect_rooms()
    log_time("Edge creation check", start_step)

    graph_plot_output_path = os.path.join(graph_img_dir, f"{image_name_no_ext}_pre_pruning_wothr_connect_graph.png")
    graph.plot_on_image(image_path, graph_plot_output_path, display_labels=False, threshold_radius = 20, highlight_regions=False)
 
    graph_plot_output_path = os.path.join(graph_img_dir, f"{image_name_no_ext}_pre_pruning_wthr_connect_graph.png")
    graph.plot_on_image(image_path, graph_plot_output_path, display_labels=False, threshold_radius = 20, highlight_regions=True) 
 
    json_output_path = os.path.join(json_img_dir, f"{image_name_no_ext}_pre_pruning.json")
    graph.save_to_json(json_output_path) 
      
    json_file_size_kb_bfr = os.path.getsize(json_output_path) / 1024

    tot_graph_nodes = graph.return_graph_size()
     
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
