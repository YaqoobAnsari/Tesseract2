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
from collections import defaultdict

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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

# Academic color palette (colorblind-friendly)
ACADEMIC_COLORS = {
    'room': '#E69F00',        # Orange
    'door': '#009E73',        # Bluish Green
    'corridor': '#CC79A7',    # Reddish Purple
    'outside': '#56B4E9',     # Sky Blue
    'transition': '#D55E00',  # Vermillion
    'unknown': '#999999',     # Gray
    'inter_floor': '#0072B2', # Blue
    'floor_1': '#E69F00',     # Orange
    'floor_2': '#56B4E9',     # Sky Blue
    'floor_3': '#009E73',     # Green
    'floor_4': '#F0E442',     # Yellow
}


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
    """Convert floor number to display name."""
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
    """
    line = line.strip()
    
    # Skip empty lines and comments
    if not line or line.startswith('#'):
        return None
    
    # Pattern: (floor, image, node):(floor, image, node)
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
    """Parse a mapping file into internal dictionary format."""
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
                if line.strip() and not line.strip().startswith('#'):
                    errors.append(f"Line {line_num}: Invalid format: {line.strip()}")
                continue
            
            source, target = result
            
            if source not in transition_mapping:
                transition_mapping[source] = []
            transition_mapping[target] = transition_mapping.get(target, [])
            transition_mapping[source].append(target)
    
    if errors:
        print("\n⚠ Mapping file format errors:")
        for err in errors:
            print(f"  {err}")
        raise ValueError(f"Found {len(errors)} format error(s) in mapping file")
    
    return transition_mapping


def parse_inline_mapping(mapping_str):
    """Parse inline mapping string (from command line)."""
    transition_mapping = {}
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
    """Validate mapping semantics comprehensively."""
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
        print(f"  ✗ ERROR: {len(missing_images)} image(s) not found")
    else:
        print(f"  ✓ All {len(all_images)} images exist")
    
    # Check 2: Floor adjacency (CRITICAL)
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
        print(f"  ✗ ERROR: {len(adjacency_violations)} floor adjacency violation(s)")
    else:
        print(f"  ✓ All floor connections are adjacent (N to N±1)")
    
    # Check 3: One-to-one constraint
    print("\n[3/4] Checking one-to-one constraint...")
    floor_pair_transitions = {}
    
    for (src_floor, src_image, src_node), targets in transition_mapping.items():
        for tgt_floor, tgt_image, tgt_node in targets:
            pair = (min(src_floor, tgt_floor), max(src_floor, tgt_floor))
            if pair not in floor_pair_transitions:
                floor_pair_transitions[pair] = []
            floor_pair_transitions[pair].append((src_node, tgt_node, src_image, tgt_image))
    
    one_to_one_violations = []
    for (floor_a, floor_b), transitions in floor_pair_transitions.items():
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
            errors.append(f"One-to-one violation: Node '{v['src_node']}' has multiple connections")
        print(f"  ✗ ERROR: {len(one_to_one_violations)} one-to-one constraint violation(s)")
    else:
        print(f"  ✓ One-to-one constraint satisfied")
    
    # Check 4: Image-floor consistency
    print("\n[4/4] Checking image-floor consistency...")
    inconsistencies = []
    for img in all_images:
        detected_floor = detect_floor_from_filename(img)
        referenced_floors = set()
        for (src_floor, src_image, _), targets in transition_mapping.items():
            if src_image == img:
                referenced_floors.add(src_floor)
            for tgt_floor, tgt_image, _ in targets:
                if tgt_image == img:
                    referenced_floors.add(tgt_floor)
        
        for ref_floor in referenced_floors:
            if ref_floor != detected_floor:
                inconsistencies.append({'image': img, 'detected': detected_floor, 'referenced': ref_floor})
                warnings.append(f"Image '{img}' detected as floor {detected_floor} but referenced as floor {ref_floor}")
    
    if inconsistencies:
        print(f"  ⚠ WARNING: {len(inconsistencies)} image-floor inconsistency(ies)")
    else:
        print(f"  ✓ All image-floor references are consistent")
    
    is_valid = len(errors) == 0
    
    print(f"\n{'=' * 70}")
    if is_valid:
        print("✓ MAPPING VALIDATION PASSED")
        if warnings:
            print(f"  ({len(warnings)} warning(s) - see above)")
    else:
        print("✗ MAPPING VALIDATION FAILED")
        print(f"  {len(errors)} error(s), {len(warnings)} warning(s)")
    print("=" * 70)
    
    return is_valid, errors, warnings


# =============================================================================
# GRAPH MANAGEMENT
# =============================================================================

def check_graph_exists(image_name, results_dir=RESULTS_DIR):
    """Check if a graph exists for the given image."""
    image_name_no_ext = os.path.splitext(image_name)[0]
    json_dir = os.path.join(results_dir, "Json", image_name_no_ext)
    
    # Prefer post_pruning (has edges), then pre_pruning, then final
    possible_files = [
        f"{image_name_no_ext}_post_pruning.json",
        f"{image_name_no_ext}_pre_pruning.json",
        f"{image_name_no_ext}_final_graph.json",
    ]
    
    for fname in possible_files:
        fpath = os.path.join(json_dir, fname)
        if os.path.exists(fpath):
            return True, fpath
    
    return False, None


def get_graph_paths(image_name, results_dir=RESULTS_DIR):
    """Get paths to both pre and post pruning graphs."""
    image_name_no_ext = os.path.splitext(image_name)[0]
    json_dir = os.path.join(results_dir, "Json", image_name_no_ext)
    
    pre_path = os.path.join(json_dir, f"{image_name_no_ext}_pre_pruning.json")
    post_path = os.path.join(json_dir, f"{image_name_no_ext}_post_pruning.json")
    
    return {
        'pre_pruning': pre_path if os.path.exists(pre_path) else None,
        'post_pruning': post_path if os.path.exists(post_path) else None
    }


def load_graph_from_json(json_path):
    """Load a BuildingGraph from JSON file."""
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
        
        if node_type not in graph.node_types:
            node_type = 'room'
        
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
    
    # Set default floor
    for node_id in graph.graph.nodes():
        floor = graph.graph.nodes[node_id].get('floor', 'UNKNOWN')
        if floor not in ('NA', 'UNKNOWN'):
            graph.default_floor = floor
            break
    
    return graph


def parse_timer_file(timer_path):
    """Parse timing information from single-floor timer file."""
    timing = {}
    
    if not os.path.exists(timer_path):
        return timing
    
    with open(timer_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line and 'seconds' in line.lower():
                parts = line.split(':')
                if len(parts) >= 2:
                    key = parts[0].strip()
                    value_str = parts[1].strip().replace('seconds', '').strip()
                    try:
                        timing[key] = float(value_str)
                    except ValueError:
                        pass
    
    return timing


def ensure_all_graphs_exist(transition_mapping, input_images_dir=INPUT_IMAGES_DIR, results_dir=RESULTS_DIR):
    """Ensure all images in mapping have generated graphs."""
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
    graph_info = {}
    images_to_generate = []
    
    for img in all_images:
        exists, json_path = check_graph_exists(img, results_dir)
        graph_paths = get_graph_paths(img, results_dir)
        
        # Get timer info
        img_no_ext = os.path.splitext(img)[0]
        timer_path = os.path.join(results_dir, "Time&Meta", "Text files", f"{img_no_ext}_timer_info.txt")
        timer_info = parse_timer_file(timer_path)
        
        if exists:
            print(f"  ✓ {img}: Graph exists")
            graphs[img] = {'path': json_path, 'needs_generation': False}
            graph_info[img] = {
                'paths': graph_paths,
                'timer': timer_info,
                'generated': False
            }
        else:
            img_path = os.path.join(input_images_dir, img)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            print(f"  ○ {img}: Graph not found, will generate")
            images_to_generate.append(img)
            graphs[img] = {'path': None, 'needs_generation': True}
            graph_info[img] = {'paths': {}, 'timer': {}, 'generated': True}
    
    # Generate missing graphs
    if images_to_generate:
        print(f"\nGenerating graphs for {len(images_to_generate)} image(s)...")
        
        try:
            from Main import make_graph
        except ImportError:
            raise RuntimeError("Could not import make_graph from Main.py")
        
        for img in images_to_generate:
            print(f"\n{'─' * 50}")
            print(f"Generating graph for: {img}")
            print(f"{'─' * 50}")
            
            start_gen = time.time()
            try:
                graph, image_name_no_ext, floor_id = make_graph(img)
                gen_time = time.time() - start_gen
                
                exists, json_path = check_graph_exists(img, results_dir)
                graph_paths = get_graph_paths(img, results_dir)
                timer_path = os.path.join(results_dir, "Time&Meta", "Text files", f"{image_name_no_ext}_timer_info.txt")
                timer_info = parse_timer_file(timer_path)
                
                if exists:
                    graphs[img] = {'path': json_path, 'needs_generation': False}
                    graph_info[img] = {
                        'paths': graph_paths,
                        'timer': timer_info,
                        'generated': True,
                        'generation_time': gen_time
                    }
                    print(f"  ✓ Graph generated in {gen_time:.2f}s: {json_path}")
                else:
                    raise RuntimeError(f"Graph was not saved properly for {img}")
                    
            except Exception as e:
                raise RuntimeError(f"Failed to generate graph for {img}: {e}")
    
    print(f"\n✓ All {len(all_images)} graphs are ready")
    
    # Load all graphs (post_pruning preferred)
    loaded_graphs = {}
    for img, info in graphs.items():
        loaded_graphs[img] = load_graph_from_json(info['path'])
        n_nodes = loaded_graphs[img].return_graph_size()
        n_edges = len(loaded_graphs[img].graph.edges())
        print(f"  Loaded {img}: {n_nodes} nodes, {n_edges} edges")
    
    return loaded_graphs, graph_info


# =============================================================================
# FLOOR SEQUENCE NAMING
# =============================================================================

def generate_floor_sequence_name(image_names_or_mapping):
    """Generate folder name from floor sequence."""
    floors = set()
    
    if isinstance(image_names_or_mapping, dict):
        for (src_floor, src_image, _), targets in image_names_or_mapping.items():
            floors.add(src_floor)
            for tgt_floor, _, _ in targets:
                floors.add(tgt_floor)
    else:
        for img in image_names_or_mapping:
            floors.add(detect_floor_from_filename(img))
    
    sorted_floors = sorted(floors)
    name_parts = [get_floor_display_name(f) for f in sorted_floors]
    
    return "_".join(name_parts)


# =============================================================================
# GRAPH MERGING
# =============================================================================

def merge_floor_graphs(floor_graphs, floor_image_map):
    """Merge individual floor graphs into a unified multi-floor graph."""
    print(f"\n{'=' * 70}")
    print("MERGING FLOOR GRAPHS")
    print("=" * 70)
    
    merged_graph = BuildingGraph(default_floor="MULTI")
    node_id_mapping = {}
    
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
        
        # Add edges
        for u, v, edge_data in graph.graph.edges(data=True):
            new_u = f"{floor_prefix}{u}"
            new_v = f"{floor_prefix}{v}"
            
            merged_graph.graph.add_edge(
                new_u, new_v,
                weight=edge_data.get('weight', 1.0),
                distance=edge_data.get('distance'),
                floor=str(floor_num),
                edge_type='intra_floor'
            )
            total_edges += 1
    
    print(f"\n  Merged graph statistics:")
    print(f"    Total nodes: {total_nodes}")
    print(f"    Total edges: {total_edges}")
    
    return merged_graph, node_id_mapping


def connect_transitions_across_floors(merged_graph, node_id_mapping, transition_mapping, floor_image_map):
    """Connect transition nodes across floors based on mapping."""
    print(f"\n{'=' * 70}")
    print("CONNECTING TRANSITIONS ACROSS FLOORS")
    print("=" * 70)
    
    connections_created = 0
    connection_details = []
    
    for (src_floor, src_image, src_node), targets in transition_mapping.items():
        src_merged_id = node_id_mapping.get((src_image, src_node))
        
        if src_merged_id is None:
            src_merged_id = f"{src_floor}_{src_node}"
            if src_merged_id not in merged_graph.graph:
                print(f"  ⚠ Warning: Source node not found: {src_node} in {src_image}")
                continue
        
        src_type = merged_graph.graph.nodes[src_merged_id].get('type', '')
        src_pos = merged_graph.graph.nodes[src_merged_id].get('position')
        
        for tgt_floor, tgt_image, tgt_node in targets:
            tgt_merged_id = node_id_mapping.get((tgt_image, tgt_node))
            
            if tgt_merged_id is None:
                tgt_merged_id = f"{tgt_floor}_{tgt_node}"
                if tgt_merged_id not in merged_graph.graph:
                    print(f"  ⚠ Warning: Target node not found: {tgt_node} in {tgt_image}")
                    continue
            
            tgt_pos = merged_graph.graph.nodes[tgt_merged_id].get('position')
            
            # Create inter-floor edge
            merged_graph.graph.add_edge(
                src_merged_id, tgt_merged_id,
                weight=1.0,
                edge_type='inter_floor',
                src_floor=src_floor,
                tgt_floor=tgt_floor
            )
            
            connections_created += 1
            connection_details.append({
                'src': src_merged_id,
                'tgt': tgt_merged_id,
                'src_floor': src_floor,
                'tgt_floor': tgt_floor,
                'src_pos': src_pos,
                'tgt_pos': tgt_pos
            })
            
            print(f"  ✓ Connected: {src_merged_id} (Floor {src_floor}) ↔ {tgt_merged_id} (Floor {tgt_floor})")
    
    print(f"\n  Total inter-floor connections: {connections_created}")
    
    return connections_created, connection_details


# =============================================================================
# CONNECTIVITY VERIFICATION
# =============================================================================

def verify_full_connectivity(merged_graph, floor_graphs, connection_details):
    """
    Verify that all rooms are reachable from all other rooms across floors.
    
    Returns:
        dict: Connectivity report with statistics and any issues found
    """
    print(f"\n{'=' * 70}")
    print("CONNECTIVITY VERIFICATION")
    print("=" * 70)
    
    report = {
        'is_fully_connected': False,
        'total_components': 0,
        'largest_component_size': 0,
        'unreachable_pairs': [],
        'room_connectivity': {},
        'floor_connectivity': {}
    }
    
    # Get all room nodes (main rooms, not subnodes)
    room_nodes = [n for n in merged_graph.graph.nodes() 
                  if merged_graph.graph.nodes[n].get('type') == 'room' 
                  and not n.endswith('_sub')]
    
    # Check if graph is connected
    if len(merged_graph.graph.nodes()) == 0:
        print("  ⚠ Graph is empty!")
        return report
    
    # Find connected components
    components = list(nx.connected_components(merged_graph.graph))
    report['total_components'] = len(components)
    report['largest_component_size'] = max(len(c) for c in components)
    
    print(f"  Connected components: {len(components)}")
    print(f"  Largest component: {report['largest_component_size']} nodes")
    
    if len(components) == 1:
        report['is_fully_connected'] = True
        print(f"  ✓ Graph is fully connected!")
    else:
        print(f"  ✗ Graph has {len(components)} disconnected components")
        
        # Analyze which floors are in which components
        for i, comp in enumerate(components):
            floors_in_comp = set()
            for node in comp:
                floor = merged_graph.graph.nodes[node].get('floor', 'unknown')
                floors_in_comp.add(floor)
            print(f"    Component {i+1}: {len(comp)} nodes, floors: {sorted(floors_in_comp)}")
    
    # Check room-to-room connectivity (sample)
    print(f"\n  Checking room-to-room paths...")
    
    # Group rooms by floor
    rooms_by_floor = defaultdict(list)
    for room in room_nodes:
        floor = merged_graph.graph.nodes[room].get('floor', 'unknown')
        rooms_by_floor[floor].append(room)
    
    # Check paths between floors
    floors = sorted(rooms_by_floor.keys())
    cross_floor_paths = 0
    cross_floor_failures = 0
    
    for i, floor_a in enumerate(floors):
        for floor_b in floors[i+1:]:
            rooms_a = rooms_by_floor[floor_a][:3]  # Sample 3 rooms per floor
            rooms_b = rooms_by_floor[floor_b][:3]
            
            for room_a in rooms_a:
                for room_b in rooms_b:
                    try:
                        path = nx.shortest_path(merged_graph.graph, room_a, room_b)
                        cross_floor_paths += 1
                    except nx.NetworkXNoPath:
                        cross_floor_failures += 1
                        report['unreachable_pairs'].append((room_a, room_b))
    
    if cross_floor_paths > 0:
        print(f"  ✓ Cross-floor paths found: {cross_floor_paths}")
    if cross_floor_failures > 0:
        print(f"  ✗ Cross-floor path failures: {cross_floor_failures}")
    
    # Check within-floor connectivity
    print(f"\n  Checking within-floor connectivity...")
    for floor, rooms in rooms_by_floor.items():
        if len(rooms) < 2:
            continue
        
        connected = 0
        disconnected = 0
        
        for i, room_a in enumerate(rooms[:5]):
            for room_b in rooms[i+1:6]:
                try:
                    path = nx.shortest_path(merged_graph.graph, room_a, room_b)
                    connected += 1
                except nx.NetworkXNoPath:
                    disconnected += 1
        
        report['floor_connectivity'][floor] = {
            'connected_pairs': connected,
            'disconnected_pairs': disconnected
        }
        
        if disconnected == 0:
            print(f"    Floor {floor}: ✓ All sampled rooms connected")
        else:
            print(f"    Floor {floor}: ✗ {disconnected} disconnected pairs")
    
    return report


# =============================================================================
# ACADEMIC-QUALITY VISUALIZATIONS
# =============================================================================

def plot_timing_bar_chart(timing_info, floor_info, output_path):
    """
    Create an academic-quality timing bar chart.
    
    Args:
        timing_info: Dict of timing values
        floor_info: Dict with per-floor timing from single-floor runs
        output_path: Path to save the plot
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Collect all timing data
    all_timings = {}
    
    # Add multi-floor specific timings
    mf_steps = ['parse_mapping', 'validation', 'merge_graphs', 'connect_transitions', 
                'save_results', 'generate_plots', 'connectivity_check']
    
    for step in mf_steps:
        if step in timing_info and timing_info[step] > 0.001:
            all_timings[f"MF: {step.replace('_', ' ').title()}"] = timing_info[step]
    
    # Add per-floor graph generation times
    for img, info in floor_info.items():
        floor_num = detect_floor_from_filename(img)
        floor_name = get_floor_display_name(floor_num)
        
        if info.get('timer'):
            # Key steps from single-floor processing
            key_steps = [
                ('text detection check', 'Text Detection'),
                ('Interpreting bboxes check', 'OCR Interpretation'),
                ('Flood Filling check', 'Flood Fill'),
                ('Detecting doors check', 'Door Detection'),
                ('Graph pruning check', 'Graph Pruning'),
            ]
            
            for timer_key, display_name in key_steps:
                if timer_key in info['timer'] and info['timer'][timer_key] > 0.5:
                    all_timings[f"F{floor_num}: {display_name}"] = info['timer'][timer_key]
            
            # Total time for floor
            if 'Total Time' in info['timer']:
                all_timings[f"F{floor_num}: Total"] = info['timer']['Total Time']
    
    if not all_timings:
        print("  ⚠ No timing data available for bar chart")
        return
    
    # Sort by value (ascending)
    sorted_items = sorted(all_timings.items(), key=lambda x: x[1])
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.4)))
    
    # Color by category
    colors = []
    for label in labels:
        if label.startswith('MF:'):
            colors.append(ACADEMIC_COLORS['inter_floor'])
        elif 'F1:' in label:
            colors.append(ACADEMIC_COLORS['floor_1'])
        elif 'F2:' in label:
            colors.append(ACADEMIC_COLORS['floor_2'])
        else:
            colors.append(ACADEMIC_COLORS['corridor'])
    
    # Horizontal bar chart
    bars = ax.barh(labels, values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}s', va='center', fontsize=9, fontweight='medium')
    
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Multi-Floor Processing Time Breakdown\n(Ascending Order)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=ACADEMIC_COLORS['inter_floor'], label='Multi-Floor Steps'),
        mpatches.Patch(facecolor=ACADEMIC_COLORS['floor_1'], label='Floor 1 Processing'),
        mpatches.Patch(facecolor=ACADEMIC_COLORS['floor_2'], label='Floor 2 Processing'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    ax.set_xlim(0, max(values) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Timing bar chart: {output_path}")


def plot_stacked_floors(merged_graph, floor_graphs, floor_image_map, connection_details, 
                       input_images_dir, output_path):
    """
    Create a stacked floor visualization showing all floors vertically aligned.
    """
    floors = sorted(set(floor_image_map.values()))
    n_floors = len(floors)
    
    # Load floor images
    floor_images = {}
    for img, floor_num in floor_image_map.items():
        img_path = os.path.join(input_images_dir, img)
        if os.path.exists(img_path):
            floor_images[floor_num] = cv2.imread(img_path)
            floor_images[floor_num] = cv2.cvtColor(floor_images[floor_num], cv2.COLOR_BGR2RGB)
    
    if not floor_images:
        print("  ⚠ No floor images found for stacked visualization")
        return
    
    # Get max dimensions
    max_height = max(img.shape[0] for img in floor_images.values())
    max_width = max(img.shape[1] for img in floor_images.values())
    
    # Create figure
    fig, axes = plt.subplots(n_floors, 1, figsize=(14, 8 * n_floors))
    if n_floors == 1:
        axes = [axes]
    
    # Plot each floor
    for idx, floor_num in enumerate(sorted(floors, reverse=True)):  # Top floor first
        ax = axes[idx]
        
        if floor_num in floor_images:
            ax.imshow(floor_images[floor_num], alpha=0.7)
        
        # Get nodes for this floor
        floor_prefix = f"{floor_num}_"
        floor_nodes = [n for n in merged_graph.graph.nodes() 
                      if n.startswith(floor_prefix)]
        
        # Plot nodes by type
        for node_id in floor_nodes:
            node_data = merged_graph.graph.nodes[node_id]
            pos = node_data.get('position')
            node_type = node_data.get('type', 'unknown')
            
            if pos:
                color = ACADEMIC_COLORS.get(node_type, ACADEMIC_COLORS['unknown'])
                size = 100 if node_type == 'transition' else 20
                marker = '^' if node_type == 'transition' else 'o'
                ax.scatter(pos[0], pos[1], c=color, s=size, marker=marker, 
                          alpha=0.8, edgecolors='black', linewidths=0.5)
        
        # Plot intra-floor edges
        for u, v, edge_data in merged_graph.graph.edges(data=True):
            if u.startswith(floor_prefix) and v.startswith(floor_prefix):
                pos_u = merged_graph.graph.nodes[u].get('position')
                pos_v = merged_graph.graph.nodes[v].get('position')
                if pos_u and pos_v:
                    ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], 
                           color='gray', alpha=0.3, linewidth=0.5)
        
        # Highlight transition nodes
        for conn in connection_details:
            if conn['src_floor'] == floor_num or conn['tgt_floor'] == floor_num:
                pos = conn['src_pos'] if conn['src_floor'] == floor_num else conn['tgt_pos']
                if pos:
                    ax.scatter(pos[0], pos[1], c=ACADEMIC_COLORS['inter_floor'], 
                              s=300, marker='*', edgecolors='black', linewidths=1.5,
                              zorder=10, label='Inter-floor Connection' if idx == 0 else '')
        
        floor_name = get_floor_display_name(floor_num)
        ax.set_title(f'Floor {floor_num} ({floor_name})', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    # Add legend to first subplot
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ACADEMIC_COLORS['room'], 
               markersize=10, label='Room'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ACADEMIC_COLORS['corridor'], 
               markersize=10, label='Corridor'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=ACADEMIC_COLORS['transition'], 
               markersize=12, label='Transition (Stairs/Elevator)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=ACADEMIC_COLORS['inter_floor'], 
               markersize=15, label='Inter-floor Connection'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.suptitle('Multi-Floor Building Graph - Stacked View', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Stacked floors plot: {output_path}")


def plot_3d_building(merged_graph, floor_image_map, connection_details, output_path):
    """
    Create a 3D visualization of the multi-floor building.
    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    floors = sorted(set(floor_image_map.values()))
    floor_height = 100  # Vertical spacing between floors
    
    # Color map for floors
    floor_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(floors)))
    floor_color_map = {f: floor_colors[i] for i, f in enumerate(floors)}
    
    # Plot nodes
    for node_id in merged_graph.graph.nodes():
        node_data = merged_graph.graph.nodes[node_id]
        pos = node_data.get('position')
        floor = node_data.get('floor', '1')
        node_type = node_data.get('type', 'unknown')
        
        if pos:
            try:
                floor_num = int(floor)
            except:
                floor_num = 1
            
            x, y = pos
            z = floor_num * floor_height
            
            # Size and marker based on type
            if node_type == 'transition':
                size = 100
                marker = '^'
                color = ACADEMIC_COLORS['transition']
            elif node_type == 'room':
                size = 30
                marker = 'o'
                color = floor_color_map.get(floor_num, 'gray')
            elif node_type == 'corridor':
                size = 15
                marker = 's'
                color = ACADEMIC_COLORS['corridor']
            else:
                size = 10
                marker = '.'
                color = 'gray'
            
            ax.scatter(x, y, z, c=[color], s=size, marker=marker, alpha=0.7)
    
    # Plot intra-floor edges (sample to avoid clutter)
    edge_sample_rate = 0.1  # Plot 10% of edges
    intra_edges = [(u, v) for u, v, d in merged_graph.graph.edges(data=True) 
                   if d.get('edge_type') != 'inter_floor']
    
    sampled_edges = intra_edges[::int(1/edge_sample_rate)] if edge_sample_rate < 1 else intra_edges
    
    for u, v in sampled_edges:
        pos_u = merged_graph.graph.nodes[u].get('position')
        pos_v = merged_graph.graph.nodes[v].get('position')
        floor_u = merged_graph.graph.nodes[u].get('floor', '1')
        
        if pos_u and pos_v:
            try:
                z = int(floor_u) * floor_height
            except:
                z = floor_height
            
            ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [z, z],
                   color='gray', alpha=0.1, linewidth=0.3)
    
    # Plot inter-floor connections (highlighted)
    for conn in connection_details:
        src_pos = conn['src_pos']
        tgt_pos = conn['tgt_pos']
        src_floor = conn['src_floor']
        tgt_floor = conn['tgt_floor']
        
        if src_pos and tgt_pos:
            z_src = src_floor * floor_height
            z_tgt = tgt_floor * floor_height
            
            # Draw vertical connection line
            ax.plot([src_pos[0], tgt_pos[0]], [src_pos[1], tgt_pos[1]], [z_src, z_tgt],
                   color=ACADEMIC_COLORS['inter_floor'], linewidth=3, alpha=0.9,
                   linestyle='-', marker='o', markersize=8)
    
    # Labels and styling
    ax.set_xlabel('X (pixels)', fontsize=11, labelpad=10)
    ax.set_ylabel('Y (pixels)', fontsize=11, labelpad=10)
    ax.set_zlabel('Floor Level', fontsize=11, labelpad=10)
    
    # Set z-ticks to floor numbers
    z_ticks = [f * floor_height for f in floors]
    ax.set_zticks(z_ticks)
    ax.set_zticklabels([f'Floor {f}' for f in floors])
    
    ax.set_title('3D Multi-Floor Building Graph\nInter-floor Connections Highlighted', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor=ACADEMIC_COLORS['transition'], 
               markersize=12, label='Transition Node'),
        Line2D([0], [0], color=ACADEMIC_COLORS['inter_floor'], linewidth=3, 
               label='Inter-floor Connection'),
        Line2D([0], [0], color='gray', linewidth=1, alpha=0.5, label='Intra-floor Edge'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Adjust view angle
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ 3D building plot: {output_path}")


def plot_connectivity_matrix(merged_graph, floor_image_map, output_path):
    """
    Create a connectivity matrix showing which floors can reach which.
    """
    floors = sorted(set(floor_image_map.values()))
    n_floors = len(floors)
    
    # Build connectivity matrix
    connectivity = np.zeros((n_floors, n_floors))
    
    # Group nodes by floor
    nodes_by_floor = defaultdict(list)
    for node_id in merged_graph.graph.nodes():
        floor = merged_graph.graph.nodes[node_id].get('floor', '1')
        try:
            floor_num = int(floor)
            if floor_num in floors:
                nodes_by_floor[floor_num].append(node_id)
        except:
            pass
    
    # Check connectivity between floors
    for i, floor_a in enumerate(floors):
        for j, floor_b in enumerate(floors):
            if i == j:
                # Same floor - check internal connectivity
                nodes_a = nodes_by_floor[floor_a][:10]  # Sample
                connected = 0
                for n1 in nodes_a:
                    for n2 in nodes_a:
                        if n1 != n2:
                            try:
                                path = nx.shortest_path(merged_graph.graph, n1, n2)
                                connected += 1
                                break
                            except:
                                pass
                    if connected > 0:
                        break
                connectivity[i, j] = 1.0 if connected > 0 else 0.0
            else:
                # Different floors
                nodes_a = nodes_by_floor[floor_a][:5]
                nodes_b = nodes_by_floor[floor_b][:5]
                paths_found = 0
                paths_tried = 0
                
                for n1 in nodes_a:
                    for n2 in nodes_b:
                        paths_tried += 1
                        try:
                            path = nx.shortest_path(merged_graph.graph, n1, n2)
                            paths_found += 1
                        except:
                            pass
                
                connectivity[i, j] = paths_found / max(paths_tried, 1)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(connectivity, cmap='RdYlGn', vmin=0, vmax=1)
    
    # Labels
    floor_labels = [f'Floor {f}\n({get_floor_display_name(f)})' for f in floors]
    ax.set_xticks(range(n_floors))
    ax.set_yticks(range(n_floors))
    ax.set_xticklabels(floor_labels, fontsize=10)
    ax.set_yticklabels(floor_labels, fontsize=10)
    
    # Add values
    for i in range(n_floors):
        for j in range(n_floors):
            val = connectivity[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.0%}', ha='center', va='center', 
                   color=color, fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Target Floor', fontsize=12, fontweight='bold')
    ax.set_ylabel('Source Floor', fontsize=12, fontweight='bold')
    ax.set_title('Floor-to-Floor Connectivity Matrix\n(Percentage of Reachable Paths)', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Connectivity Rate', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Connectivity matrix: {output_path}")


def plot_node_distribution(merged_graph, floor_image_map, output_path):
    """
    Create a bar chart showing node distribution by type and floor.
    """
    floors = sorted(set(floor_image_map.values()))
    node_types = ['room', 'corridor', 'door', 'transition', 'outside']
    
    # Count nodes
    counts = {floor: {t: 0 for t in node_types} for floor in floors}
    
    for node_id in merged_graph.graph.nodes():
        node_data = merged_graph.graph.nodes[node_id]
        floor = node_data.get('floor', '1')
        node_type = node_data.get('type', 'unknown')
        
        try:
            floor_num = int(floor)
            if floor_num in floors and node_type in node_types:
                counts[floor_num][node_type] += 1
        except:
            pass
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(floors))
    width = 0.15
    
    for i, node_type in enumerate(node_types):
        values = [counts[f][node_type] for f in floors]
        offset = (i - len(node_types)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=node_type.title(),
                     color=ACADEMIC_COLORS.get(node_type, 'gray'), alpha=0.85,
                     edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                       str(val), ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Floor', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax.set_title('Node Distribution by Type and Floor', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Floor {f}\n({get_floor_display_name(f)})' for f in floors])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Node distribution plot: {output_path}")


def plot_pre_post_pruning_comparison(graph_info, floor_image_map, output_path):
    """
    Create comparison plots showing pre vs post pruning statistics.
    """
    floors = sorted(set(floor_image_map.values()))
    
    pre_nodes = []
    post_nodes = []
    pre_edges = []
    post_edges = []
    floor_labels = []
    
    for img, floor_num in floor_image_map.items():
        info = graph_info.get(img, {})
        timer = info.get('timer', {})
        
        if 'Total graph nodes (before pruning)' in timer:
            pre_nodes.append(timer['Total graph nodes (before pruning)'])
        else:
            pre_nodes.append(0)
        
        if 'Total graph nodes (after pruning)' in timer:
            post_nodes.append(timer['Total graph nodes (after pruning)'])
        else:
            post_nodes.append(0)
        
        # Load pre/post graphs to get edge counts
        paths = info.get('paths', {})
        if paths.get('pre_pruning'):
            try:
                with open(paths['pre_pruning']) as f:
                    data = json.load(f)
                pre_edges.append(len(data.get('edges', [])))
            except:
                pre_edges.append(0)
        else:
            pre_edges.append(0)
        
        if paths.get('post_pruning'):
            try:
                with open(paths['post_pruning']) as f:
                    data = json.load(f)
                post_edges.append(len(data.get('edges', [])))
            except:
                post_edges.append(0)
        else:
            post_edges.append(0)
        
        floor_labels.append(f'Floor {floor_num}\n({get_floor_display_name(floor_num)})')
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(floors))
    width = 0.35
    
    # Nodes comparison
    bars1 = ax1.bar(x - width/2, pre_nodes, width, label='Before Pruning', 
                    color=ACADEMIC_COLORS['floor_1'], alpha=0.85, edgecolor='black')
    bars2 = ax1.bar(x + width/2, post_nodes, width, label='After Pruning',
                    color=ACADEMIC_COLORS['floor_2'], alpha=0.85, edgecolor='black')
    
    ax1.set_xlabel('Floor', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax1.set_title('Node Count: Pre vs Post Pruning', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(floor_labels)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add reduction percentage
    for i, (pre, post) in enumerate(zip(pre_nodes, post_nodes)):
        if pre > 0:
            reduction = (pre - post) / pre * 100
            ax1.text(i, max(pre, post) + 50, f'-{reduction:.0f}%', 
                    ha='center', fontsize=9, color='red', fontweight='bold')
    
    # Edges comparison
    bars3 = ax2.bar(x - width/2, pre_edges, width, label='Before Pruning',
                    color=ACADEMIC_COLORS['floor_1'], alpha=0.85, edgecolor='black')
    bars4 = ax2.bar(x + width/2, post_edges, width, label='After Pruning',
                    color=ACADEMIC_COLORS['floor_2'], alpha=0.85, edgecolor='black')
    
    ax2.set_xlabel('Floor', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Edges', fontsize=12, fontweight='bold')
    ax2.set_title('Edge Count: Pre vs Post Pruning', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(floor_labels)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add reduction percentage
    for i, (pre, post) in enumerate(zip(pre_edges, post_edges)):
        if pre > 0:
            reduction = (pre - post) / pre * 100
            ax2.text(i, max(pre, post) + 200, f'-{reduction:.0f}%',
                    ha='center', fontsize=9, color='red', fontweight='bold')
    
    plt.suptitle('Graph Pruning Impact Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Pre/post pruning comparison: {output_path}")


# =============================================================================
# RESULT SAVING
# =============================================================================

def save_multifloor_results(merged_graph, floor_graphs, floor_sequence_name, 
                           connection_details, timing_info, validation_info, graph_info,
                           connectivity_report, multifloor_results_dir=MULTIFLOOR_RESULTS_DIR):
    """Save multi-floor results to organized directory structure."""
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
    
    # 3. Save comprehensive timing info
    timing_path = os.path.join(time_dir, "multifloor_timer_info.txt")
    with open(timing_path, 'w') as f:
        f.write(f"Multi-Floor Processing Timing Report\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Floor Sequence: {floor_sequence_name}\n\n")
        
        f.write("MULTI-FLOOR PROCESSING STEPS:\n")
        f.write("-" * 40 + "\n")
        mf_steps = ['parse_mapping', 'validation', 'ensure_graphs', 'merge_graphs', 
                    'connect_transitions', 'connectivity_check', 'save_results', 'generate_plots']
        for step in mf_steps:
            if step in timing_info:
                f.write(f"  {step.replace('_', ' ').title()}: {timing_info[step]:.3f} seconds\n")
        
        f.write(f"\nPER-FLOOR PROCESSING TIMES:\n")
        f.write("-" * 40 + "\n")
        for img, info in graph_info.items():
            floor_num = detect_floor_from_filename(img)
            timer = info.get('timer', {})
            total = timer.get('Total Time', 0)
            f.write(f"\n  Floor {floor_num} ({img}):\n")
            f.write(f"    Total: {total:.2f} seconds\n")
            
            # Key steps
            key_steps = ['text detection check', 'Interpreting bboxes check', 
                        'Flood Filling check', 'Detecting doors check', 'Graph pruning check']
            for step in key_steps:
                if step in timer:
                    f.write(f"    {step}: {timer[step]:.2f} seconds\n")
        
        f.write(f"\n{'=' * 60}\n")
        f.write(f"TOTAL PROCESSING TIME: {timing_info.get('total', 0):.2f} seconds\n")
    
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
            f.write(f"    Floors: {conn['src_floor']} → {conn['tgt_floor']}\n")
            if conn.get('src_pos') and conn.get('tgt_pos'):
                f.write(f"    Positions: {conn['src_pos']} → {conn['tgt_pos']}\n")
            f.write("\n")
    
    saved_files['mapping_summary'] = mapping_path
    print(f"  ✓ Mapping summary: {mapping_path}")
    
    # 5. Save connectivity report
    connectivity_path = os.path.join(time_dir, "connectivity_report.txt")
    with open(connectivity_path, 'w') as f:
        f.write(f"Connectivity Verification Report\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Fully Connected: {'Yes' if connectivity_report.get('is_fully_connected') else 'No'}\n")
        f.write(f"Connected Components: {connectivity_report.get('total_components', 0)}\n")
        f.write(f"Largest Component: {connectivity_report.get('largest_component_size', 0)} nodes\n\n")
        
        if connectivity_report.get('floor_connectivity'):
            f.write("Floor-wise Connectivity:\n")
            for floor, stats in connectivity_report['floor_connectivity'].items():
                f.write(f"  Floor {floor}: {stats['connected_pairs']} connected, "
                       f"{stats['disconnected_pairs']} disconnected\n")
    
    saved_files['connectivity'] = connectivity_path
    print(f"  ✓ Connectivity report: {connectivity_path}")
    
    # 6. Save validation report
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
    
    return saved_files, plots_dir, time_dir


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_multi_floor(mapping_file_path=None, mapping_str=None, spatial_tolerance=0.02):
    """Main entry point for multi-floor processing."""
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
    
    # Step 3: Ensure all graphs exist (with timing from single-floor runs)
    start_step = time.time()
    floor_graphs, graph_info = ensure_all_graphs_exist(transition_mapping)
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
    
    # Step 6: Verify connectivity
    start_step = time.time()
    connectivity_report = verify_full_connectivity(merged_graph, floor_graphs, connection_details)
    timing_info['connectivity_check'] = time.time() - start_step
    
    # Step 7: Generate floor sequence name
    floor_sequence_name = generate_floor_sequence_name(transition_mapping)
    print(f"\nFloor sequence: {floor_sequence_name}")
    
    # Step 8: Save results
    start_step = time.time()
    saved_files, plots_dir, time_dir = save_multifloor_results(
        merged_graph, floor_graphs, floor_sequence_name,
        connection_details, timing_info, validation_info, graph_info,
        connectivity_report
    )
    timing_info['save_results'] = time.time() - start_step
    
    # Step 9: Generate academic-quality plots
    start_step = time.time()
    print(f"\n{'=' * 70}")
    print("GENERATING ACADEMIC-QUALITY VISUALIZATIONS")
    print("=" * 70)
    
    # Timing bar chart
    plot_timing_bar_chart(timing_info, graph_info, 
                         os.path.join(plots_dir, "timing_breakdown.png"))
    
    # Stacked floors view
    plot_stacked_floors(merged_graph, floor_graphs, floor_image_map, connection_details,
                       INPUT_IMAGES_DIR, os.path.join(plots_dir, "stacked_floors.png"))
    
    # 3D building visualization
    plot_3d_building(merged_graph, floor_image_map, connection_details,
                    os.path.join(plots_dir, "3d_building.png"))
    
    # Connectivity matrix
    plot_connectivity_matrix(merged_graph, floor_image_map,
                            os.path.join(plots_dir, "connectivity_matrix.png"))
    
    # Node distribution
    plot_node_distribution(merged_graph, floor_image_map,
                          os.path.join(plots_dir, "node_distribution.png"))
    
    # Pre/post pruning comparison
    plot_pre_post_pruning_comparison(graph_info, floor_image_map,
                                    os.path.join(plots_dir, "pruning_comparison.png"))
    
    timing_info['generate_plots'] = time.time() - start_step
    
    # Total time
    timing_info['total'] = time.time() - start_total
    
    # Update timing file with final values
    timing_path = os.path.join(time_dir, "multifloor_timer_info.txt")
    with open(timing_path, 'a') as f:
        f.write(f"\nPlot Generation: {timing_info['generate_plots']:.2f} seconds\n")
        f.write(f"FINAL TOTAL: {timing_info['total']:.2f} seconds\n")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("MULTI-FLOOR PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Floor sequence: {floor_sequence_name}")
    print(f"Total nodes: {merged_graph.return_graph_size()}")
    print(f"Total edges: {len(merged_graph.graph.edges())}")
    print(f"Inter-floor connections: {connections}")
    print(f"Fully connected: {'Yes' if connectivity_report.get('is_fully_connected') else 'No'}")
    print(f"Total time: {timing_info['total']:.2f} seconds")
    print(f"\nResults saved to: {os.path.join(MULTIFLOOR_RESULTS_DIR, '*', floor_sequence_name)}")
    
    return {
        'merged_graph': merged_graph,
        'floor_graphs': floor_graphs,
        'floor_sequence_name': floor_sequence_name,
        'connection_details': connection_details,
        'saved_files': saved_files,
        'timing_info': timing_info,
        'connectivity_report': connectivity_report
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
    python MultiFloor.py --mapping-file mappings/FF_SF.txt
    python MultiFloor.py --mapping "(1, FF part 1upE.png, stairs_1):(2, SF part 1upE.png, stairs_1)"
"""
    )
    
    parser.add_argument('--mapping-file', '-f', type=str, help='Path to mapping file (.txt)')
    parser.add_argument('--mapping', '-m', type=str, help='Inline mapping string')
    parser.add_argument('--spatial-tolerance', '-t', type=float, default=0.02)
    
    args = parser.parse_args()
    
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
