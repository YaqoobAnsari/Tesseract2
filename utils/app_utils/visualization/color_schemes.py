"""
Centralized academic color palette constants for Tesseract++ visualization.
Wong colorblind-safe palette adapted for architectural graph types.
"""

# Node colors by type (Wong colorblind-safe palette)
NODE_COLORS = {
    "room": "#E69F00",            # Orange
    "door": "#009E73",            # Teal green
    "corridor": "#CC79A7",        # Pink/mauve
    "outside": "#56B4E9",         # Sky blue
    "floor_transition": "#D55E00" # Vermillion
}

# Edge colors based on the types of connected nodes
EDGE_COLORS = {
    "room": "#FFA500",        # Orange - edges within/to rooms
    "corridor": "#3399FF",    # Blue - edges involving corridors
    "outside": "#FF0000",     # Red - edges to outside
    "transition": "#B40000",  # Dark red - floor transition edges
    "default": "#999999"      # Gray - default/mixed
}

# Node sizes by type (diameter in pixels for Cytoscape)
NODE_SIZES = {
    "room": 30,
    "door": 20,
    "corridor": 12,
    "outside": 25,
    "floor_transition": 35
}

# Node shapes by type (Cytoscape.js shape names)
NODE_SHAPES = {
    "room": "ellipse",
    "door": "rectangle",
    "corridor": "ellipse",
    "outside": "diamond",
    "floor_transition": "triangle"
}
