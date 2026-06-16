/** Node colors by type — Wong colorblind-safe palette */
export const NODE_COLORS: Record<string, string> = {
  room: '#E69F00',
  door: '#009E73',
  corridor: '#CC79A7',
  outside: '#56B4E9',
  floor_transition: '#D55E00',
};

/** Edge colors based on endpoint node types */
export const EDGE_COLORS: Record<string, string> = {
  room: '#FFA500',
  corridor: '#3399FF',
  outside: '#FF0000',
  transition: '#B40000',
  default: '#999999',
};

/** Node sizes (width/height in px for Cytoscape) */
export const NODE_SIZES: Record<string, number> = {
  room: 15,
  door: 15,
  corridor: 12,
  outside: 15,
  floor_transition: 35,
};

/** Node shapes (Cytoscape shape names) */
export const NODE_SHAPES: Record<string, string> = {
  room: 'ellipse',
  door: 'rectangle',
  corridor: 'ellipse',
  outside: 'diamond',
  floor_transition: 'triangle',
};

/** All known node types in display order */
export const NODE_TYPES = ['room', 'door', 'corridor', 'outside', 'floor_transition'] as const;

/** Human-readable labels for node types */
export const NODE_TYPE_LABELS: Record<string, string> = {
  room: 'Room',
  door: 'Door',
  corridor: 'Corridor',
  outside: 'Outside',
  floor_transition: 'Floor Transition',
};

/** Colors for the interactive navigation route overlay */
export const ROUTE_COLORS = {
  path: '#0d6efd',   // highlighted route edges
  source: '#2e7d32', // start node (green)
  target: '#c62828', // destination node (red)
};

/** Node types that may be chosen as routing endpoints.
 *  Rooms are the primary endpoints; transitions (stairs/elevators) are
 *  offered too. The raw graph labels transitions as 'transition'; we also
 *  accept 'floor_transition' for forward compatibility. */
export const ROUTE_ENDPOINT_TYPES = ['room', 'transition', 'floor_transition'] as const;
