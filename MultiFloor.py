#!/usr/bin/env python3
"""
MultiFloor.py - Multi-Floor Connectivity Module for Tesseract++

This module handles processing multiple floorplan images and connecting them
via transition nodes (stairs/elevators) to create a unified multi-floor graph.

Usage:
    # As standalone script:
    python MultiFloor.py --mapping-file mappings/FF_SF.txt
    python MultiFloor.py --mapping "(1, FF part 1upE.png, stairs_1):(2, SF part 1upE.png, stairs_1)"

    # As imported module:
    from MultiFloor import process_multi_floor
    result = process_multi_floor(mapping_file_path="mappings/FF_SF.txt")

Mapping Format:
    (floor_num, image_name, node_id):(floor_num, image_name, node_id)
    
    Example:
    (1, FF part 1upE.png, stairs_1):(2, SF part 1upE.png, stairs_1)
    (1, FF part 1upE.png, elevator_1):(2, SF part 1upE.png, elevator_1)

Author: Tesseract++ Team
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

# Add required paths
sys.path.insert(0, "/data1/yansari/cad2map/Tesseract++")
sys.path.insert(0, "/data1/yansari/cad2map/Tesseract++/utils")

from utils.graph import BuildingGraph


# =============================================================================
# CONSTANTS
# =============================================================================

BASE_PATH = "/data1/yansari/cad2map/Tesseract++"
INPUT_IMAGES_DIR = os.path.join(BASE_PATH, "Input_Images")
RESULTS_DIR = os.path.join(BASE_PATH, "Results")
MULTIFLOOR_RESULTS_DIR = os.path.join(BASE_PATH, "Multifloor_Results")

# Floor code mappings (supports negative floors for basements)
FLOOR_CODE_MAP = {
    "B2": -2, "B1": -1,  # Basements
    "GF": 0, "G": 0,      # Ground floor
    "FF": 1, "SF": 2, "TF": 3, "FO": 4, "FI": 5,  # Named floors
}

# Reverse mapping for display
FLOOR_NAME_MAP = {v: k for k, v in FLOOR_CODE_MAP.items()}


# =============================================================================
# FLOOR DETECTION
# =============================================================================

def detect_floor_from_filename(image_name):
    """
    Extract floor number from image filename.
    
    Supports:
        - Named floors: FF (1), SF (2), TF (3), B1 (-1), B2 (-2), GF (0)
        - Numeric floors: "4 part 1upE.png" → 4
    
    Args:
        image_name (str): Image filename (e.g., "FF part 1upE.png")
    
    Returns:
        int: Floor number (can be negative for basements)
    """
    name_no_ext = os.path.splitext(os.path.basename(image_name))[0]
    parts = name_no_ext.split()
    
    if not parts:
        print(f"⚠ Warning: Could not extract floor from '{image_name}', defaulting to floor 1")
        return 1
    
    first_word = parts[0].upper()
    
    # Check named floor codes
    if first_word in FLOOR_CODE_MAP:
        floor_num = FLOOR_CODE_MAP[first_word]
        return floor_num
    
    # Check if first word is numeric (including negative)
    try:
        floor_num = int(first_word)
        return floor_num
    except ValueError:
        pass
    
    # Try to extract any number from the filename
    match = re.search(r'(-?\d+)', name_no_ext)
    if match:
        floor_num = int(match.group(1))
        return floor_num
    
    print(f"⚠ Warning: Could not determine floor from '{image_name}', defaulting to floor 1")
    return 1


def get_floor_display_name(floor_num):
    """
    Convert floor number to display name.
    
    Args:
        floor_num (int): Floor number
    
    Returns:
        str: Display name (e.g., "FF" for floor 1)
    """
    if floor_num in FLOOR_NAME_MAP:
        return FLOOR_NAME_MAP[floor_num]
    return str(floor_num)


# =============================================================================
# MAPPING PARSING
# =============================================================================

def parse_mapping_line(line):
    """
    Parse a single mapping line.
    
    Format: (floor_num, image_name, node_id):(floor_num, image_name, node_id)
    
    Args:
        line (str): Single mapping line
    
    Returns:
        tuple: ((src_floor, src_image, src_node), (tgt_floor, tgt_image, tgt_node))
               or None if line is comment/empty/invalid
    """
    line = line.strip()
    
    # Skip empty lines and comments
    if not line or line.startswith('#'):
        return None
    
    # Pattern: (floor, image, node):(floor, image, node)
    # Allow spaces within the tuple but be careful with image names containing spaces
    pattern = r'\(\s*(-?\d+)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)\s*:\s*\(\s*(-?\d+)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)'
    
    match = re.match(pattern, line)
    if not match:
        return None
    
    src_floor = int(match.group(1))
    src_image = match.group(2).strip()
    src_node = match.group(3).strip()
    tgt_floor = int(match.group(4))
    tgt_image = match.group(5).strip()
    tgt_node = match.group(6).strip()
    
    return ((src_floor, src_image, src_node), (tgt_floor, tgt_image, tgt_node))


def parse_mapping_file(mapping_file_path):
    """
    Parse a mapping file into internal dictionary format.
    
    Args:
        mapping_file_path (str): Path to mapping file
    
    Returns:
        dict: Transition mapping in format:
            {
                (source_floor_num, source_image, source_node_id): [
                    (target_floor_num, target_image, target_node_id),
                    ...
                ]
            }
    
    Raises:
        FileNotFoundError: If mapping file doesn't exist
        ValueError: If mapping file has invalid format
    """
    if not os.path.exists(mapping_file_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file_path}")
    
    transition_mapping = {}
    errors = []
    line_num = 0
    
    with open(mapping_file_path, 'r') as f:
        for line in f:
            line_num += 1
            result = parse_mapping_line(line)
            
            if result is None:
                # Skip empty/comment lines silently
                if line.strip() and not line.strip().startswith('#'):
                    errors.append(f"Line {line_num}: Invalid format: {line.strip()}")
                continue
            
            source, target = result
            
            # Add to mapping (grouping targets by source)
            if source not in transition_mapping:
                transition_mapping[source] = []
            transition_mapping[target] = transition_mapping.get(target, [])
            
            # Add bidirectional mapping for connectivity
            transition_mapping[source].append(target)
    
    if errors:
        print("\n⚠ Mapping file format errors:")
        for err in errors:
            print(f"  {err}")
        print("\nExpected format: (floor_num, image_name, node_id):(floor_num, image_name, node_id)")
        raise ValueError(f"Found {len(errors)} format error(s) in mapping file")
    
    return transition_mapping


def parse_inline_mapping(mapping_str):
    """
    Parse inline mapping string (from command line).
    
    Multiple mappings can be separated by semicolons.
    
    Args:
        mapping_str (str): Inline mapping string
    
    Returns:
        dict: Transition mapping dictionary
    """
    transition_mapping = {}
    
    # Split by semicolons for multiple mappings
    lines = mapping_str.split(';')
    
    for line in lines:
        result = parse_mapping_line(line)
        if result:
            source, target = result
            if source not in transition_mapping:
                transition_mapping[source] = []
            transition_mapping[source].append(target)
    
    return transition_mapping


# =============================================================================
# VALIDATION
# =============================================================================

def validate_mapping_semantics(transition_mapping, input_images_dir=INPUT_IMAGES_DIR):
    """
    Validate mapping semantics comprehensively.
    
    Checks:
        1. Floor adjacency (N can only connect to N±1) - ERROR
        2. One-to-one constraint (one transition node per floor pair) - ERROR
        3. Image existence in Input_Images/ - ERROR
        4. Image-floor consistency - WARNING
    
    Args:
        transition_mapping (dict): Parsed transition mapping
        input_images_dir (str): Path to input images directory
    
    Returns:
        tuple: (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    print("\n" + "=" * 70)
    print("MAPPING VALIDATION")
    print("=" * 70)
    
    if not transition_mapping:
        errors.append("Mapping is empty - no transitions defined")
        return False, errors, warnings
    
    # Collect all unique images and floors
    all_images = set()
    all_floors = set()
    
    for (src_floor, src_image, src_node), targets in transition_mapping.items():
        all_images.add(src_image)
        all_floors.add(src_floor)
        for tgt_floor, tgt_image, tgt_node in targets:
            all_images.add(tgt_image)
            all_floors.add(tgt_floor)
    
    print(f"\nMapping summary:")
    print(f"  Unique images: {len(all_images)}")
    print(f"  Floors involved: {sorted(all_floors)}")
    print(f"  Total mappings: {sum(len(t) for t in transition_mapping.values())}")
    
    # Check 1: Image existence
    print("\n[1/4] Checking image existence...")
    missing_images = []
    for img in all_images:
        img_path = os.path.join(input_images_dir, img)
        if not os.path.exists(img_path):
            missing_images.append(img)
            errors.append(f"Image not found: {img}")
    
    if missing_images:
        print(f"  ✗ ERROR: {len(missing_images)} image(s) not found:")
        for img in missing_images:
            print(f"    - {img}")
        # List available images
        available = [f for f in os.listdir(input_images_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if available:
            print(f"  Available images in {input_images_dir}:")
            for img in sorted(available)[:10]:
                print(f"    - {img}")
            if len(available) > 10:
                print(f"    ... and {len(available) - 10} more")
    else:
        print(f"  ✓ All {len(all_images)} images exist")
    
    # Check 2: Floor adjacency (CRITICAL - must be adjacent, including basements)
    print("\n[2/4] Checking floor adjacency...")
    adjacency_violations = []
    
    for (src_floor, src_image, src_node), targets in transition_mapping.items():
        for tgt_floor, tgt_image, tgt_node in targets:
            floor_diff = abs(tgt_floor - src_floor)
            if floor_diff != 1:
                adjacency_violations.append({
                    'src': (src_floor, src_image, src_node),
                    'tgt': (tgt_floor, tgt_image, tgt_node),
                    'diff': floor_diff
                })
                errors.append(
                    f"Floor adjacency violation: Floor {src_floor} → Floor {tgt_floor} "
                    f"(difference: {floor_diff}). Floors must be adjacent (N to N±1)."
                )
    
    if adjacency_violations:
        print(f"  ✗ ERROR: {len(adjacency_violations)} floor adjacency violation(s):")
        for v in adjacency_violations:
            print(f"    Floor {v['src'][0]} ({v['src'][1]}) → Floor {v['tgt'][0]} ({v['tgt'][1]})")
            print(f"      Difference: {v['diff']} floors (must be 1)")
    else:
        print(f"  ✓ All floor connections are adjacent (N to N±1)")
    
    # Check 3: One-to-one constraint per floor pair
    print("\n[3/4] Checking one-to-one constraint...")
    floor_pair_transitions = {}  # {(src_floor, tgt_floor): [(src_node, tgt_node), ...]}
    
    for (src_floor, src_image, src_node), targets in transition_mapping.items():
        for tgt_floor, tgt_image, tgt_node in targets:
            # Normalize pair to always have smaller floor first
            pair = (min(src_floor, tgt_floor), max(src_floor, tgt_floor))
            if pair not in floor_pair_transitions:
                floor_pair_transitions[pair] = []
            floor_pair_transitions[pair].append((src_node, tgt_node, src_image, tgt_image))
    
    one_to_one_violations = []
    for (floor_a, floor_b), transitions in floor_pair_transitions.items():
        # Check for duplicate node connections within the same floor pair
        seen_src_nodes = {}
        for src_node, tgt_node, src_img, tgt_img in transitions:
            key = src_node
            if key in seen_src_nodes:
                one_to_one_violations.append({
                    'floors': (floor_a, floor_b),
                    'src_node': src_node,
                    'connections': [seen_src_nodes[key], (tgt_node, tgt_img)]
                })
            else:
                seen_src_nodes[key] = (tgt_node, tgt_img)
    
    if one_to_one_violations:
        for v in one_to_one_violations:
            errors.append(
                f"One-to-one violation: Node '{v['src_node']}' on floor pair "
                f"{v['floors']} has multiple connections"
            )
        print(f"  ✗ ERROR: {len(one_to_one_violations)} one-to-one constraint violation(s)")
    else:
        print(f"  ✓ One-to-one constraint satisfied")
    
    # Check 4: Image-floor consistency
    print("\n[4/4] Checking image-floor consistency...")
    inconsistencies = []
    for img in all_images:
        detected_floor = detect_floor_from_filename(img)
        # Find floor references for this image
        referenced_floors = set()
        for (src_floor, src_image, _), targets in transition_mapping.items():
            if src_image == img:
                referenced_floors.add(src_floor)
            for tgt_floor, tgt_image, _ in targets:
                if tgt_image == img:
                    referenced_floors.add(tgt_floor)
        
        for ref_floor in referenced_floors:
            if ref_floor != detected_floor:
                inconsistencies.append({
                    'image': img,
                    'detected': detected_floor,
                    'referenced': ref_floor
                })
                warnings.append(
                    f"Image '{img}' detected as floor {detected_floor} but "
                    f"referenced as floor {ref_floor}"
                )
    
    if inconsistencies:
        print(f"  ⚠ WARNING: {len(inconsistencies)} image-floor inconsistency(ies):")
        for inc in inconsistencies:
            print(f"    {inc['image']}: detected={inc['detected']}, referenced={inc['referenced']}")
    else:
        print(f"  ✓ All image-floor references are consistent")
    
    # Summary
    is_valid = len(errors) == 0
    
    print(f"\n{'=' * 70}")
    if is_valid:
        print("✓ MAPPING VALIDATION PASSED")
        if warnings:
            print(f"  ({len(warnings)} warning(s) - see above)")
    else:
        print("✗ MAPPING VALIDATION FAILED")
        print(f"  {len(errors)} error(s), {len(warnings)} warning(s)")
        print("\nErrors (must fix):")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
    print("=" * 70)
    
    return is_valid, errors, warnings


# =============================================================================
# GRAPH MANAGEMENT
# =============================================================================

def check_graph_exists(image_name, results_dir=RESULTS_DIR):
    """
    Check if a graph exists for the given image.
    
    Args:
        image_name (str): Image filename
        results_dir (str): Results directory path
    
    Returns:
        tuple: (exists, json_path) - exists is bool, json_path is the path if exists
    """
    image_name_no_ext = os.path.splitext(image_name)[0]
    json_dir = os.path.join(results_dir, "Json", image_name_no_ext)
    
    # Check for various possible graph files
    possible_files = [
        f"{image_name_no_ext}_final_graph.json",
        f"{image_name_no_ext}_post_pruning.json",
        f"{image_name_no_ext}_pre_pruning.json",
    ]
    
    for fname in possible_files:
        fpath = os.path.join(json_dir, fname)
        if os.path.exists(fpath):
            return True, fpath
    
    return False, None


def load_graph_from_json(json_path):
    """
    Load a BuildingGraph from JSON file.
    
    Args:
        json_path (str): Path to JSON file
    
    Returns:
        BuildingGraph: Loaded graph
    """
    graph = BuildingGraph()
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Add nodes
    for node_data in data.get('nodes', []):
        node_id = node_data['id']
        node_type = node_data.get('type', 'unknown')
        position = tuple(node_data.get('position', [0, 0]))
        pixels = node_data.get('pixels', [])
        floor_id = node_data.get('floor', 'UNKNOWN')
        
        # Ensure node_type is valid
        if node_type not in graph.node_types:
            node_type = 'room'  # Default fallback
        
        graph.graph.add_node(
            node_id,
            type=node_type,
            position=position,
            pixels=pixels,
            floor=floor_id
        )
        graph.node_types[node_type].append(node_id)
    
    # Add edges
    for edge_data in data.get('edges', []):
        source = edge_data['source']
        target = edge_data['target']
        weight = edge_data.get('weight', 1.0)
        distance = edge_data.get('distance')
        
        if source in graph.graph and target in graph.graph:
            graph.graph.add_edge(source, target, weight=weight, distance=distance)
    
    # Set default floor from first non-outside node
    for node_id in graph.graph.nodes():
        floor = graph.graph.nodes[node_id].get('floor', 'UNKNOWN')
        if floor not in ('NA', 'UNKNOWN'):
            graph.default_floor = floor
            break
    
    return graph


def ensure_all_graphs_exist(transition_mapping, input_images_dir=INPUT_IMAGES_DIR, results_dir=RESULTS_DIR):
    """
    Ensure all images in mapping have generated graphs.
    If graph doesn't exist but image exists, generate it.
    
    Args:
        transition_mapping (dict): Transition mapping
        input_images_dir (str): Input images directory
        results_dir (str): Results directory
    
    Returns:
        dict: {image_name: (graph, json_path)} mapping
    
    Raises:
        FileNotFoundError: If image doesn't exist
        RuntimeError: If graph generation fails
    """
    # Collect all unique images
    all_images = set()
    for (src_floor, src_image, src_node), targets in transition_mapping.items():
        all_images.add(src_image)
        for tgt_floor, tgt_image, tgt_node in targets:
            all_images.add(tgt_image)
    
    print(f"\n{'=' * 70}")
    print("ENSURING ALL GRAPHS EXIST")
    print("=" * 70)
    print(f"Checking {len(all_images)} image(s)...")
    
    graphs = {}
    images_to_generate = []
    
    for img in all_images:
        exists, json_path = check_graph_exists(img, results_dir)
        if exists:
            print(f"  ✓ {img}: Graph exists at {json_path}")
            graphs[img] = {'path': json_path, 'needs_generation': False}
        else:
            # Check if image exists
            img_path = os.path.join(input_images_dir, img)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            print(f"  ○ {img}: Graph not found, will generate")
            images_to_generate.append(img)
            graphs[img] = {'path': None, 'needs_generation': True}
    
    # Generate missing graphs
    if images_to_generate:
        print(f"\nGenerating graphs for {len(images_to_generate)} image(s)...")
        
        # Import Main.py's make_graph function
        try:
            from Main import make_graph
        except ImportError:
            raise RuntimeError("Could not import make_graph from Main.py")
        
        for img in images_to_generate:
            print(f"\n{'─' * 50}")
            print(f"Generating graph for: {img}")
            print(f"{'─' * 50}")
            
            try:
                # Generate the graph using Main.py's pipeline
                graph, image_name_no_ext, floor_id = make_graph(img)
                
                # Find the generated JSON
                exists, json_path = check_graph_exists(img, results_dir)
                if exists:
                    graphs[img] = {'path': json_path, 'needs_generation': False}
                    print(f"  ✓ Graph generated: {json_path}")
                else:
                    raise RuntimeError(f"Graph was not saved properly for {img}")
                    
            except Exception as e:
                raise RuntimeError(f"Failed to generate graph for {img}: {e}")
    
    print(f"\n✓ All {len(all_images)} graphs are ready")
    
    # Load all graphs
    loaded_graphs = {}
    for img, info in graphs.items():
        loaded_graphs[img] = load_graph_from_json(info['path'])
        print(f"  Loaded {img}: {loaded_graphs[img].return_graph_size()} nodes")
    
    return loaded_graphs


# =============================================================================
# FLOOR SEQUENCE NAMING
# =============================================================================

def generate_floor_sequence_name(image_names_or_mapping):
    """
    Generate folder name from floor sequence.
    
    Args:
        image_names_or_mapping: List of image names or transition mapping dict
    
    Returns:
        str: Floor sequence name (e.g., "FF_SF", "B1_FF_SF")
    """
    floors = set()
    
    if isinstance(image_names_or_mapping, dict):
        # It's a transition mapping
        for (src_floor, src_image, _), targets in image_names_or_mapping.items():
            floors.add(src_floor)
            for tgt_floor, _, _ in targets:
                floors.add(tgt_floor)
    else:
        # It's a list of image names
        for img in image_names_or_mapping:
            floors.add(detect_floor_from_filename(img))
    
    # Sort floors and generate name
    sorted_floors = sorted(floors)
    name_parts = [get_floor_display_name(f) for f in sorted_floors]
    
    return "_".join(name_parts)


# =============================================================================
# GRAPH MERGING
# =============================================================================

def merge_floor_graphs(floor_graphs, floor_image_map):
    """
    Merge individual floor graphs into a unified multi-floor graph.
    
    Node IDs are prefixed with floor number to ensure uniqueness.
    
    Args:
        floor_graphs (dict): {image_name: BuildingGraph}
        floor_image_map (dict): {image_name: floor_num}
    
    Returns:
        BuildingGraph: Merged multi-floor graph
    """
    print(f"\n{'=' * 70}")
    print("MERGING FLOOR GRAPHS")
    print("=" * 70)
    
    merged_graph = BuildingGraph(default_floor="MULTI")
    
    # Track node ID mappings for edge creation
    node_id_mapping = {}  # {(image, old_id): new_id}
    
    total_nodes = 0
    total_edges = 0
    
    for img, graph in floor_graphs.items():
        floor_num = floor_image_map.get(img, detect_floor_from_filename(img))
        floor_prefix = f"{floor_num}_"
        
        print(f"\n  Processing: {img} (Floor {floor_num})")
        print(f"    Nodes: {graph.return_graph_size()}, Edges: {len(graph.graph.edges())}")
        
        # Add nodes with prefixed IDs
        for node_id in graph.graph.nodes():
            node_data = graph.graph.nodes[node_id]
            new_node_id = f"{floor_prefix}{node_id}"
            
            node_type = node_data.get('type', 'unknown')
            if node_type not in merged_graph.node_types:
                node_type = 'room'
            
            merged_graph.graph.add_node(
                new_node_id,
                type=node_type,
                position=node_data.get('position'),
                pixels=node_data.get('pixels', []),
                floor=str(floor_num),
                original_id=node_id,
                source_image=img
            )
            merged_graph.node_types[node_type].append(new_node_id)
            
            node_id_mapping[(img, node_id)] = new_node_id
            total_nodes += 1
        
        # Add edges with prefixed node IDs
        for u, v, edge_data in graph.graph.edges(data=True):
            new_u = f"{floor_prefix}{u}"
            new_v = f"{floor_prefix}{v}"
            
            merged_graph.graph.add_edge(
                new_u, new_v,
                weight=edge_data.get('weight', 1.0),
                distance=edge_data.get('distance'),
                floor=str(floor_num)
            )
            total_edges += 1
    
    print(f"\n  Merged graph statistics:")
    print(f"    Total nodes: {total_nodes}")
    print(f"    Total edges: {total_edges}")
    
    return merged_graph, node_id_mapping


def connect_transitions_across_floors(merged_graph, node_id_mapping, transition_mapping, floor_image_map):
    """
    Connect transition nodes across floors based on mapping.
    
    Args:
        merged_graph (BuildingGraph): Merged multi-floor graph
        node_id_mapping (dict): {(image, old_id): new_id}
        transition_mapping (dict): Transition mapping
        floor_image_map (dict): {image_name: floor_num}
    
    Returns:
        int: Number of inter-floor connections created
    """
    print(f"\n{'=' * 70}")
    print("CONNECTING TRANSITIONS ACROSS FLOORS")
    print("=" * 70)
    
    connections_created = 0
    connection_details = []
    
    for (src_floor, src_image, src_node), targets in transition_mapping.items():
        # Get the merged node ID for source
        src_merged_id = node_id_mapping.get((src_image, src_node))
        
        if src_merged_id is None:
            print(f"  ⚠ Warning: Source node not found: {src_node} in {src_image}")
            # Try to find it with floor prefix
            src_merged_id = f"{src_floor}_{src_node}"
            if src_merged_id not in merged_graph.graph:
                print(f"    Skipping - node does not exist in merged graph")
                continue
        
        # Verify source is a transition node
        src_type = merged_graph.graph.nodes[src_merged_id].get('type', '')
        if src_type != 'transition':
            print(f"  ⚠ Warning: {src_node} is not a transition node (type: {src_type})")
        
        src_pos = merged_graph.graph.nodes[src_merged_id].get('position')
        
        for tgt_floor, tgt_image, tgt_node in targets:
            # Get the merged node ID for target
            tgt_merged_id = node_id_mapping.get((tgt_image, tgt_node))
            
            if tgt_merged_id is None:
                tgt_merged_id = f"{tgt_floor}_{tgt_node}"
                if tgt_merged_id not in merged_graph.graph:
                    print(f"  ⚠ Warning: Target node not found: {tgt_node} in {tgt_image}")
                    continue
            
            # Verify target is a transition node
            tgt_type = merged_graph.graph.nodes[tgt_merged_id].get('type', '')
            if tgt_type != 'transition':
                print(f"  ⚠ Warning: {tgt_node} is not a transition node (type: {tgt_type})")
            
            tgt_pos = merged_graph.graph.nodes[tgt_merged_id].get('position')
            
            # Calculate weight (could factor in floor difference)
            weight = 1.0  # Base weight for inter-floor connection
            
            # Create the inter-floor edge
            merged_graph.graph.add_edge(
                src_merged_id, tgt_merged_id,
                weight=weight,
                edge_type='inter_floor',
                src_floor=src_floor,
                tgt_floor=tgt_floor
            )
            
            connections_created += 1
            connection_details.append({
                'src': src_merged_id,
                'tgt': tgt_merged_id,
                'floors': (src_floor, tgt_floor)
            })
            
            print(f"  ✓ Connected: {src_merged_id} (Floor {src_floor}) ↔ {tgt_merged_id} (Floor {tgt_floor})")
    
    print(f"\n  Total inter-floor connections: {connections_created}")
    
    return connections_created, connection_details


# =============================================================================
# RESULT SAVING
# =============================================================================

def save_multifloor_results(merged_graph, floor_graphs, floor_sequence_name, 
                           connection_details, timing_info, validation_info,
                           multifloor_results_dir=MULTIFLOOR_RESULTS_DIR):
    """
    Save multi-floor results to organized directory structure.
    
    Args:
        merged_graph (BuildingGraph): Merged multi-floor graph
        floor_graphs (dict): Individual floor graphs
        floor_sequence_name (str): Name like "FF_SF"
        connection_details (list): Details of inter-floor connections
        timing_info (dict): Timing information
        validation_info (dict): Validation results
        multifloor_results_dir (str): Base output directory
    
    Returns:
        dict: Paths to saved files
    """
    print(f"\n{'=' * 70}")
    print("SAVING MULTIFLOOR RESULTS")
    print("=" * 70)
    
    # Create directory structure
    json_dir = os.path.join(multifloor_results_dir, "Jsons", floor_sequence_name)
    plots_dir = os.path.join(multifloor_results_dir, "Plots", floor_sequence_name)
    time_dir = os.path.join(multifloor_results_dir, "Time&Meta", floor_sequence_name)
    
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(time_dir, exist_ok=True)
    
    saved_files = {}
    
    # 1. Save merged graph JSON
    merged_json_path = os.path.join(json_dir, "merged_multi_floor_graph.json")
    merged_graph.save_to_json(merged_json_path)
    saved_files['merged_json'] = merged_json_path
    print(f"  ✓ Merged graph: {merged_json_path}")
    
    # 2. Save individual floor graph backups
    for img, graph in floor_graphs.items():
        floor_num = detect_floor_from_filename(img)
        floor_json_path = os.path.join(json_dir, f"floor_{floor_num}_graph.json")
        graph.save_to_json(floor_json_path)
        saved_files[f'floor_{floor_num}_json'] = floor_json_path
        print(f"  ✓ Floor {floor_num} backup: {floor_json_path}")
    
    # 3. Save timing info
    timing_path = os.path.join(time_dir, "multifloor_timer_info.txt")
    with open(timing_path, 'w') as f:
        f.write(f"Multi-Floor Processing Timing Report\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Floor Sequence: {floor_sequence_name}\n\n")
        
        for key, value in timing_info.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.2f} seconds\n")
            else:
                f.write(f"{key}: {value}\n")
    saved_files['timing'] = timing_path
    print(f"  ✓ Timing info: {timing_path}")
    
    # 4. Save mapping summary
    mapping_path = os.path.join(time_dir, "mapping_summary.txt")
    with open(mapping_path, 'w') as f:
        f.write(f"Multi-Floor Connection Summary\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Total inter-floor connections: {len(connection_details)}\n\n")
        
        f.write("Connections:\n")
        for conn in connection_details:
            f.write(f"  {conn['src']} ↔ {conn['tgt']}\n")
            f.write(f"    Floors: {conn['floors'][0]} → {conn['floors'][1]}\n\n")
    saved_files['mapping_summary'] = mapping_path
    print(f"  ✓ Mapping summary: {mapping_path}")
    
    # 5. Save validation report
    validation_path = os.path.join(time_dir, "validation_report.txt")
    with open(validation_path, 'w') as f:
        f.write(f"Mapping Validation Report\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Status: {'PASSED' if validation_info.get('is_valid', False) else 'FAILED'}\n\n")
        
        if validation_info.get('errors'):
            f.write("Errors:\n")
            for err in validation_info['errors']:
                f.write(f"  - {err}\n")
            f.write("\n")
        
        if validation_info.get('warnings'):
            f.write("Warnings:\n")
            for warn in validation_info['warnings']:
                f.write(f"  - {warn}\n")
    saved_files['validation'] = validation_path
    print(f"  ✓ Validation report: {validation_path}")
    
    return saved_files


# =============================================================================
# PLOTTING
# =============================================================================

def plot_merged_graph(merged_graph, floor_graphs, floor_sequence_name, 
                     connection_details, plots_dir):
    """
    Generate visualization plots for the merged multi-floor graph.
    
    Args:
        merged_graph (BuildingGraph): Merged graph
        floor_graphs (dict): Individual floor graphs
        floor_sequence_name (str): Name like "FF_SF"
        connection_details (list): Inter-floor connection details
        plots_dir (str): Output directory for plots
    """
    print(f"\n{'=' * 70}")
    print("GENERATING PLOTS")
    print("=" * 70)
    
    # Colors matching the existing style from graph.py
    colors = {
        'room': '#FF8000',        # Bright Orange
        'door': '#00CC66',        # Emerald Green
        'corridor': '#FF66FF',    # Magenta
        'outside': '#CC3333',     # Crimson Red
        'transition': '#FF0000',  # RED for stairs/elevator
        'unknown': '#808080',     # Gray
        'inter_floor_edge': '#00FFFF',  # Cyan for inter-floor connections
    }
    
    floor_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot 1: Full merged graph schematic
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Get all positions
    pos = {}
    node_colors = []
    node_sizes = []
    
    floors = set()
    for node_id in merged_graph.graph.nodes():
        node_data = merged_graph.graph.nodes[node_id]
        position = node_data.get('position')
        if position:
            floor = node_data.get('floor', '1')
            try:
                floor_num = int(floor)
            except:
                floor_num = 1
            floors.add(floor_num)
            
            # Offset by floor for 3D-like visualization
            x, y = position
            y_offset = floor_num * 100  # Vertical offset per floor
            pos[node_id] = (x, y + y_offset)
            
            # Color by type
            node_type = node_data.get('type', 'unknown')
            node_colors.append(colors.get(node_type, colors['unknown']))
            
            # Size based on type
            if node_type == 'transition':
                node_sizes.append(100)
            elif node_type in ('room', 'corridor'):
                node_sizes.append(30)
            else:
                node_sizes.append(20)
    
    # Draw regular edges
    regular_edges = [(u, v) for u, v, d in merged_graph.graph.edges(data=True) 
                     if d.get('edge_type') != 'inter_floor']
    inter_floor_edges = [(u, v) for u, v, d in merged_graph.graph.edges(data=True) 
                         if d.get('edge_type') == 'inter_floor']
    
    # Draw nodes
    nodes_to_draw = [n for n in merged_graph.graph.nodes() if n in pos]
    if nodes_to_draw:
        nx.draw_networkx_nodes(merged_graph.graph, pos, 
                               nodelist=nodes_to_draw,
                               node_color=[colors.get(merged_graph.graph.nodes[n].get('type', 'unknown'), 
                                                     colors['unknown']) for n in nodes_to_draw],
                               node_size=[30 for _ in nodes_to_draw],
                               alpha=0.7, ax=ax)
    
    # Draw regular edges
    regular_edges_to_draw = [(u, v) for u, v in regular_edges if u in pos and v in pos]
    if regular_edges_to_draw:
        nx.draw_networkx_edges(merged_graph.graph, pos,
                               edgelist=regular_edges_to_draw,
                               edge_color='gray', alpha=0.3, width=0.5, ax=ax)
    
    # Draw inter-floor edges (highlighted)
    inter_floor_to_draw = [(u, v) for u, v in inter_floor_edges if u in pos and v in pos]
    if inter_floor_to_draw:
        nx.draw_networkx_edges(merged_graph.graph, pos,
                               edgelist=inter_floor_to_draw,
                               edge_color=colors['inter_floor_edge'],
                               width=3, alpha=0.9, style='solid', ax=ax)
    
    # Add floor labels
    sorted_floors = sorted(floors)
    for floor_num in sorted_floors:
        ax.text(0.02, 0.02 + (floor_num - min(sorted_floors)) * 0.1, 
                f"Floor {floor_num}", transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                color=floor_colors[floor_num % len(floor_colors)])
    
    ax.set_title(f"Multi-Floor Graph: {floor_sequence_name}\n"
                 f"Nodes: {len(merged_graph.graph.nodes())}, "
                 f"Edges: {len(merged_graph.graph.edges())}, "
                 f"Inter-floor: {len(inter_floor_edges)}", fontsize=14)
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['room'], 
                   markersize=10, label='Room'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['corridor'], 
                   markersize=10, label='Corridor'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['transition'], 
                   markersize=10, label='Transition'),
        plt.Line2D([0], [0], color=colors['inter_floor_edge'], linewidth=3, 
                   label='Inter-floor Connection'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "merged_graph_visualization.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Merged graph plot: {plot_path}")
    
    # Plot 2: Inter-floor connections only
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Only show transition nodes and their connections
    transition_nodes = [n for n in merged_graph.graph.nodes() 
                       if merged_graph.graph.nodes[n].get('type') == 'transition' and n in pos]
    
    if transition_nodes:
        # Draw transition nodes
        nx.draw_networkx_nodes(merged_graph.graph, pos,
                               nodelist=transition_nodes,
                               node_color=colors['transition'],
                               node_size=200, alpha=0.9, ax=ax)
        
        # Draw labels
        labels = {n: n.split('_', 1)[1] if '_' in n else n for n in transition_nodes}
        nx.draw_networkx_labels(merged_graph.graph, pos,
                                labels=labels, font_size=8, ax=ax)
        
        # Draw inter-floor edges
        if inter_floor_to_draw:
            nx.draw_networkx_edges(merged_graph.graph, pos,
                                   edgelist=inter_floor_to_draw,
                                   edge_color=colors['inter_floor_edge'],
                                   width=4, alpha=0.9, ax=ax)
    
    ax.set_title(f"Inter-Floor Connections: {floor_sequence_name}\n"
                 f"Transitions: {len(transition_nodes)}, "
                 f"Connections: {len(connection_details)}", fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "inter_floor_connections.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Inter-floor connections plot: {plot_path}")
    
    return True


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_multi_floor(mapping_file_path=None, mapping_str=None, spatial_tolerance=0.02):
    """
    Main entry point for multi-floor processing.
    
    Args:
        mapping_file_path (str, optional): Path to mapping file
        mapping_str (str, optional): Inline mapping string
        spatial_tolerance (float): Tolerance for spatial alignment (default: 0.02)
    
    Returns:
        dict: Results including merged_graph, floor_graphs, output_paths
    """
    start_total = time.time()
    timing_info = {}
    
    print("\n" + "=" * 70)
    print("MULTI-FLOOR PROCESSING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Parse mapping
    start_step = time.time()
    if mapping_file_path:
        print(f"\nLoading mapping from file: {mapping_file_path}")
        transition_mapping = parse_mapping_file(mapping_file_path)
    elif mapping_str:
        print(f"\nParsing inline mapping...")
        transition_mapping = parse_inline_mapping(mapping_str)
    else:
        raise ValueError("Either mapping_file_path or mapping_str must be provided")
    
    timing_info['parse_mapping'] = time.time() - start_step
    print(f"  Parsed {sum(len(t) for t in transition_mapping.values())} mapping(s)")
    
    # Step 2: Validate mapping
    start_step = time.time()
    is_valid, errors, warnings = validate_mapping_semantics(transition_mapping)
    timing_info['validation'] = time.time() - start_step
    
    if not is_valid:
        raise ValueError(f"Mapping validation failed with {len(errors)} error(s)")
    
    validation_info = {'is_valid': is_valid, 'errors': errors, 'warnings': warnings}
    
    # Step 3: Ensure all graphs exist
    start_step = time.time()
    floor_graphs = ensure_all_graphs_exist(transition_mapping)
    timing_info['ensure_graphs'] = time.time() - start_step
    
    # Build floor-image mapping
    floor_image_map = {}
    for (src_floor, src_image, _), targets in transition_mapping.items():
        floor_image_map[src_image] = src_floor
        for tgt_floor, tgt_image, _ in targets:
            floor_image_map[tgt_image] = tgt_floor
    
    # Step 4: Merge graphs
    start_step = time.time()
    merged_graph, node_id_mapping = merge_floor_graphs(floor_graphs, floor_image_map)
    timing_info['merge_graphs'] = time.time() - start_step
    
    # Step 5: Connect transitions
    start_step = time.time()
    connections, connection_details = connect_transitions_across_floors(
        merged_graph, node_id_mapping, transition_mapping, floor_image_map
    )
    timing_info['connect_transitions'] = time.time() - start_step
    
    # Step 6: Generate floor sequence name
    floor_sequence_name = generate_floor_sequence_name(transition_mapping)
    print(f"\nFloor sequence: {floor_sequence_name}")
    
    # Step 7: Save results
    start_step = time.time()
    saved_files = save_multifloor_results(
        merged_graph, floor_graphs, floor_sequence_name,
        connection_details, timing_info, validation_info
    )
    timing_info['save_results'] = time.time() - start_step
    
    # Step 8: Generate plots
    start_step = time.time()
    plots_dir = os.path.join(MULTIFLOOR_RESULTS_DIR, "Plots", floor_sequence_name)
    plot_merged_graph(merged_graph, floor_graphs, floor_sequence_name, 
                     connection_details, plots_dir)
    timing_info['generate_plots'] = time.time() - start_step
    
    # Total time
    timing_info['total'] = time.time() - start_total
    
    # Summary
    print(f"\n{'=' * 70}")
    print("MULTI-FLOOR PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Floor sequence: {floor_sequence_name}")
    print(f"Total nodes: {merged_graph.return_graph_size()}")
    print(f"Total edges: {len(merged_graph.graph.edges())}")
    print(f"Inter-floor connections: {connections}")
    print(f"Total time: {timing_info['total']:.2f} seconds")
    print(f"\nResults saved to: {os.path.join(MULTIFLOOR_RESULTS_DIR, '*', floor_sequence_name)}")
    
    return {
        'merged_graph': merged_graph,
        'floor_graphs': floor_graphs,
        'floor_sequence_name': floor_sequence_name,
        'connection_details': connection_details,
        'saved_files': saved_files,
        'timing_info': timing_info
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Command-line interface for multi-floor processing."""
    parser = argparse.ArgumentParser(
        description="Multi-Floor Connectivity Module for Tesseract++",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mapping Format:
    (floor_num, image_name, node_id):(floor_num, image_name, node_id)

Examples:
    # Using mapping file:
    python MultiFloor.py --mapping-file mappings/FF_SF.txt

    # Using inline mapping:
    python MultiFloor.py --mapping "(1, FF part 1upE.png, stairs_1):(2, SF part 1upE.png, stairs_1)"

    # Multiple inline mappings (separated by semicolons):
    python MultiFloor.py --mapping "(1, FF part 1upE.png, stairs_1):(2, SF part 1upE.png, stairs_1);(1, FF part 1upE.png, elevator_1):(2, SF part 1upE.png, elevator_1)"
"""
    )
    
    parser.add_argument(
        '--mapping-file', '-f',
        type=str,
        help='Path to mapping file (.txt)'
    )
    
    parser.add_argument(
        '--mapping', '-m',
        type=str,
        help='Inline mapping string (multiple mappings separated by semicolons)'
    )
    
    parser.add_argument(
        '--spatial-tolerance', '-t',
        type=float,
        default=0.02,
        help='Spatial tolerance for alignment checks (default: 0.02)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.mapping_file and not args.mapping:
        parser.error("Either --mapping-file or --mapping must be provided")
    
    if args.mapping_file and args.mapping:
        parser.error("Cannot use both --mapping-file and --mapping")
    
    try:
        result = process_multi_floor(
            mapping_file_path=args.mapping_file,
            mapping_str=args.mapping,
            spatial_tolerance=args.spatial_tolerance
        )
        print("\n✓ Multi-floor processing completed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
