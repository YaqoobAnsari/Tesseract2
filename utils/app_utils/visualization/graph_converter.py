"""
Convert Tesseract++ graph JSON to Cytoscape.js elements format.

Tesseract++ JSON format:
  nodes: [{id, type, position: [x,y], floor, ...}, ...]
  edges: [{source, target, weight, distance}, ...]

Cytoscape.js elements format:
  nodes: [{data: {id, type, floor, ...}, position: {x, y}}, ...]
  edges: [{data: {id, source, target, weight, edgeColorType}, ...}, ...]
"""

from typing import Dict, Any, List


def _determine_edge_color_type(
    source_id: str,
    target_id: str,
    node_type_map: Dict[str, str]
) -> str:
    """Determine edge color category based on endpoint node types."""
    src_type = node_type_map.get(source_id, "unknown")
    tgt_type = node_type_map.get(target_id, "unknown")

    types = {src_type, tgt_type}

    if "floor_transition" in types:
        return "transition"
    if "outside" in types:
        return "outside"
    if "corridor" in types:
        return "corridor"
    if "room" in types:
        return "room"
    return "default"


def convert_to_cytoscape(graph_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Tesseract++ graph JSON to Cytoscape.js elements format.

    Args:
        graph_json: Raw graph dict with 'nodes' and 'edges' lists.

    Returns:
        Dict with 'nodes', 'edges', and 'layout' keys in Cytoscape format.
    """
    raw_nodes = graph_json.get("nodes", [])
    raw_edges = graph_json.get("edges", [])

    # Build node type lookup for edge coloring
    node_type_map: Dict[str, str] = {}
    for node in raw_nodes:
        node_type_map[node["id"]] = node.get("type", "unknown")

    # Convert nodes
    cy_nodes: List[Dict[str, Any]] = []
    for node in raw_nodes:
        pos = node.get("position", [0, 0])
        x = pos[0] if isinstance(pos, (list, tuple)) else pos.get("x", 0)
        y = pos[1] if isinstance(pos, (list, tuple)) else pos.get("y", 0)

        data: Dict[str, Any] = {
            "id": node["id"],
            "label": node["id"],
            "type": node.get("type", "unknown"),
            "floor": node.get("floor", ""),
        }

        # Add room-specific attributes
        if node.get("type") == "room":
            if "room_area_px" in node:
                data["area"] = node["room_area_px"]
            if "anchor_door" in node:
                data["anchorDoor"] = node["anchor_door"]
            if "room_eq_radius" in node:
                data["eqRadius"] = round(node["room_eq_radius"], 1)

        # Add subnode info
        if node.get("is_subnode"):
            data["isSubnode"] = True
            data["parentRoom"] = node.get("parent_room_id", "")

        # Add door-specific attributes
        if node.get("type") == "door":
            if "door_type" in node:
                data["doorType"] = node["door_type"]

        cy_nodes.append({
            "data": data,
            "position": {"x": float(x), "y": float(y)}
        })

    # Convert edges
    cy_edges: List[Dict[str, Any]] = []
    for idx, edge in enumerate(raw_edges):
        source = edge["source"]
        target = edge["target"]
        edge_id = f"e{idx}_{source}_{target}"

        color_type = _determine_edge_color_type(source, target, node_type_map)

        cy_edges.append({
            "data": {
                "id": edge_id,
                "source": source,
                "target": target,
                "weight": edge.get("weight", 1.0),
                "edgeColorType": color_type
            }
        })

    return {
        "nodes": cy_nodes,
        "edges": cy_edges,
        "layout": "preset"
    }
