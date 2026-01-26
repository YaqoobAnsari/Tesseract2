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
