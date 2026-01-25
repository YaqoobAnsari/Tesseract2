"""
Pydantic models for API request/response validation
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ExampleImage(BaseModel):
    """Example image information"""
    name: str
    display_name: str
    size_kb: float
    has_cached_result: bool = False

class NodeData(BaseModel):
    """Graph node data structure"""
    id: str
    label: str
    type: str
    position: Dict[str, float]  # {x, y}
    floor: Optional[str] = None
    attributes: Dict[str, Any] = {}

class EdgeData(BaseModel):
    """Graph edge data structure"""
    id: str
    source: str
    target: str
    weight: float
    label: Optional[str] = None

class CytoscapeGraph(BaseModel):
    """Cytoscape.js compatible graph format"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    layout: str = "preset"  # Use preset positions from graph

class ProcessingResponse(BaseModel):
    """Response from image processing"""
    session_id: str
    status: str = Field(description="success, error, or processing")
    image_name: str
    processing_time: float
    graph_data: Optional[CytoscapeGraph] = None
    statistics: Dict[str, Any] = {}
    message: str = ""
    error: Optional[str] = None

class GraphVisualization(BaseModel):
    """Graph visualization configuration"""
    show_labels: bool = True
    node_size: Dict[str, int] = {
        "room": 30,
        "door": 20,
        "corridor": 15,
        "outside": 25,
        "floor_transition": 35
    }
    colors: Dict[str, str] = {
        "room": "#E69F00",
        "door": "#009E73",
        "corridor": "#CC79A7",
        "outside": "#56B4E9",
        "floor_transition": "#D55E00"
    }
    edge_colors: Dict[str, str] = {
        "room": "#FFA500",
        "corridor": "#3399FF",
        "outside": "#FF0000",
        "transition": "#B40000"
    }

class ExportRequest(BaseModel):
    """Request to export graph data"""
    session_id: str
    format: str = Field(description="json, png, or svg")
    include_image: bool = False
