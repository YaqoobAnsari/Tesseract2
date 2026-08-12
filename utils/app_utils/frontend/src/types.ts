/** Matches backend ExampleImage pydantic model */
export interface ExampleImage {
  name: string;
  display_name: string;
  size_kb: number;
  has_cached_result: boolean;
}

/** Cytoscape node data from backend */
export interface CyNodeData {
  id: string;
  label: string;
  type: string;
  floor: string;
  area?: number;
  anchorDoor?: string;
  eqRadius?: number;
  isSubnode?: boolean;
  parentRoom?: string;
  doorType?: string;
}

/** Cytoscape edge data from backend */
export interface CyEdgeData {
  id: string;
  source: string;
  target: string;
  weight: number;
  edgeColorType: string;
}

/** Cytoscape.js graph format from backend */
export interface CytoscapeGraph {
  nodes: Array<{ data: CyNodeData; position: { x: number; y: number } }>;
  edges: Array<{ data: CyEdgeData }>;
  layout: string;
}

/** Matches backend ProcessingResponse */
export interface ProcessingResponse {
  session_id: string;
  status: string;
  image_name: string;
  processing_time: number;
  graph_data: CytoscapeGraph | null;
  pre_pruning_graph_data: CytoscapeGraph | null;
  statistics: GraphStatistics;
  message: string;
  error?: string;
}

/** Graph processing stage selector */
export type GraphStage = 'post_pruning' | 'pre_pruning';

export interface GraphStatistics {
  total_nodes: number;
  total_edges: number;
  node_types: Record<string, number>;
  pruning_reduction?: number;
}

/** Application state machine */
export type AppState = 'idle' | 'processing' | 'results' | 'error';

/** Node type visibility toggles */
export type NodeTypeVisibility = Record<string, boolean>;

/** Node type sizes (px) */
export type NodeTypeSizes = Record<string, number>;

/** A node that can serve as a routing endpoint (start / destination) */
export interface RouteEndpoint {
  id: string;
  label: string;
  type: string;
  floor: string;
}

/** Result of an A* / Dijkstra routing query over the graph */
export interface RouteInfo {
  found: boolean;
  /** Total path weight (sum of edge Euclidean distances, in pixels) */
  distance: number;
  /** Number of edges (segments) on the path */
  segments: number;
  /** True when the path spans more than one floor */
  crossFloor: boolean;
  /** Number of nodes along the path */
  nodeCount: number;
  /** Count of nodes on the path by type (room subnodes counted as 'subnode') */
  nodeTypes: Record<string, number>;
}

/** Canvas interaction mode. Only one is active at a time. */
export type InteractionMode =
  | 'idle'
  | 'route-start'
  | 'route-end'
  | 'add-node'
  | 'add-edge'
  | 'delete';

/** Live graph counts, kept in sync with the canvas after edits */
export interface GraphCounts {
  total_nodes: number;
  total_edges: number;
  node_types: Record<string, number>;
}

/** Connectivity analysis of the current graph (computed client-side).
 *  We only care that the main room nodes are mutually connected and that
 *  the exit doors reach the same network. Any node with no edges is a hard
 *  red flag. */
export interface ConnectivityInfo {
  /** Health score: percent of tracked nodes (rooms, exits, isolated) that are OK */
  score: number;
  fullyConnected: boolean;
  /** Nodes with zero edges (must not exist) */
  isolatedCount: number;
  /** Main room nodes not connected to the main network */
  roomsDisconnected: number;
  /** Exit doors not connected to the main network */
  exitsDisconnected: number;
  componentCount: number;
  /** Offending nodes, with the reason each is flagged */
  offenders: { id: string; type: string; reason: 'isolated' | 'room' | 'exit' }[];
}
