import { useEffect, useRef, useCallback, useState, type MutableRefObject } from 'react';
import cytoscape from 'cytoscape';
import type {
  CytoscapeGraph,
  NodeTypeVisibility,
  NodeTypeSizes,
  RouteInfo,
  InteractionMode,
} from '../types';
import {
  NODE_COLORS,
  NODE_SIZES,
  NODE_SHAPES,
  EDGE_COLORS,
  ROUTE_COLORS,
} from '../constants';

interface Props {
  graphData: CytoscapeGraph;
  visibility: NodeTypeVisibility;
  nodeSizes: NodeTypeSizes;
  showEdges: boolean;
  showFloorplan: boolean;
  floorplanUrl: string;
  floorplanOpacity: number;
  routeSource: string | null;
  routeTarget: string | null;
  mode: InteractionMode;
  addNodeType: string;
  editVersion: number;
  brokenNodeIds: string[];
  /** Identity of the underlying image. When it is unchanged across a graph
   *  swap (i.e. a pre/post pruning toggle), the viewport is preserved so the
   *  graph stays aligned with the floorplan instead of re-fitting. */
  viewKey: string;
  onCyInit: (cy: cytoscape.Core) => void;
  onTooltip: (
    info: { x: number; y: number; data: Record<string, unknown> } | null,
  ) => void;
  onNodeSelect: (id: string) => void;
  onRouteComputed: (info: RouteInfo | null) => void;
  onGraphMutated: () => void;
  onEditStateChange: (undoCount: number, redoCount: number) => void;
  editControls: MutableRefObject<{ undo: () => void; redo: () => void } | null>;
}

/** One reversible edit, with both directions. */
type EditEntry = { undo: () => void; redo: () => void };

const ROUTE_CLASSES = 'route-hl route-dim route-src route-dst';

function edgeColorOf(a: string, b: string): string {
  const s = new Set([a, b]);
  if (s.has('floor_transition') || s.has('transition')) return 'transition';
  if (s.has('outside')) return 'outside';
  if (s.has('corridor')) return 'corridor';
  if (s.has('room')) return 'room';
  return 'default';
}

function buildStylesheet(): cytoscape.StylesheetStyle[] {
  const nodeStyles = Object.entries(NODE_COLORS).map(([type, color]) => ({
    selector: `node[type="${type}"]`,
    style: {
      'background-color': color,
      shape: NODE_SHAPES[type] || 'ellipse',
      width: NODE_SIZES[type] || 20,
      height: NODE_SIZES[type] || 20,
      label: '',
    },
  }));

  const edgeStyles = Object.entries(EDGE_COLORS).map(([type, color]) => ({
    selector: `edge[edgeColorType="${type}"]`,
    style: { 'line-color': color, width: 1.5, 'curve-style': 'bezier', opacity: 0.7 },
  }));

  const defaults = [
    { selector: 'node', style: { 'background-color': '#999', width: 15, height: 15, label: '' } },
    { selector: 'edge', style: { 'line-color': '#999', width: 1.5, 'curve-style': 'bezier', opacity: 0.5 } },
    { selector: '.hidden', style: { display: 'none' } },
  ];

  const routeStyles = [
    { selector: '.route-dim', style: { opacity: 0.1, 'text-opacity': 0 } },
    { selector: 'edge.route-hl', style: { 'line-color': ROUTE_COLORS.path, width: 5, opacity: 1, 'z-index': 9999 } },
    { selector: 'node.route-hl', style: { 'border-width': 3, 'border-color': ROUTE_COLORS.path, opacity: 1, 'z-index': 9999 } },
    {
      selector: 'node.route-src',
      style: {
        'background-color': ROUTE_COLORS.source, 'border-width': 4, 'border-color': '#1b5e20',
        label: 'data(label)', color: '#1b5e20', 'font-size': 13, 'font-weight': 'bold',
        'text-outline-color': '#ffffff', 'text-outline-width': 2, opacity: 1, 'z-index': 10000,
      },
    },
    {
      selector: 'node.route-dst',
      style: {
        'background-color': ROUTE_COLORS.target, 'border-width': 4, 'border-color': '#b71c1c',
        label: 'data(label)', color: '#b71c1c', 'font-size': 13, 'font-weight': 'bold',
        'text-outline-color': '#ffffff', 'text-outline-width': 2, opacity: 1, 'z-index': 10000,
      },
    },
    // Edit highlights
    { selector: 'node.edge-pending', style: { 'border-width': 4, 'border-color': '#0d6efd', opacity: 1, 'z-index': 10001 } },
    // Hover glow (selection feedback during route / edge / delete modes)
    { selector: 'node.hover-target', style: { 'overlay-color': '#0d6efd', 'overlay-padding': 7, 'overlay-opacity': 0.28 } },
    { selector: 'node.hover-delete', style: { 'overlay-color': '#dc3545', 'overlay-padding': 7, 'overlay-opacity': 0.33 } },
    { selector: 'edge.hover-delete', style: { 'line-color': '#dc3545', width: 5, opacity: 1, 'z-index': 9998 } },
    // Flash when focusing a disconnected node from the stats panel
    { selector: '.focus-flash', style: { 'overlay-color': '#ffb300', 'overlay-padding': 9, 'overlay-opacity': 0.55 } },
    // Persistent red ring on broken / disconnected nodes (large glow, #5)
    {
      selector: 'node.broken',
      style: {
        'border-width': 4, 'border-color': '#dc3545',
        'overlay-color': '#dc3545', 'overlay-padding': 12, 'overlay-opacity': 0.28,
        opacity: 1, 'z-index': 9997,
      },
    },
  ];

  return [...defaults, ...nodeStyles, ...edgeStyles, ...routeStyles] as cytoscape.StylesheetStyle[];
}

export default function GraphViewer({
  graphData,
  visibility,
  nodeSizes,
  showEdges,
  showFloorplan,
  floorplanUrl,
  floorplanOpacity,
  routeSource,
  routeTarget,
  mode,
  addNodeType,
  editVersion,
  brokenNodeIds,
  viewKey,
  onCyInit,
  onTooltip,
  onNodeSelect,
  onRouteComputed,
  onGraphMutated,
  onEditStateChange,
  editControls,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const floorplanRef = useRef<HTMLImageElement>(null);
  const imgDimsRef = useRef<{ w: number; h: number } | null>(null);
  const [imgDims, setImgDims] = useState<{ w: number; h: number } | null>(null);

  // Latest callbacks/props in refs so cy event handlers never go stale.
  const onNodeSelectRef = useRef(onNodeSelect); onNodeSelectRef.current = onNodeSelect;
  const onRouteComputedRef = useRef(onRouteComputed); onRouteComputedRef.current = onRouteComputed;
  const onGraphMutatedRef = useRef(onGraphMutated); onGraphMutatedRef.current = onGraphMutated;
  const onEditStateChangeRef = useRef(onEditStateChange); onEditStateChangeRef.current = onEditStateChange;
  const onTooltipRef = useRef(onTooltip); onTooltipRef.current = onTooltip;
  const modeRef = useRef(mode); modeRef.current = mode;
  const addNodeTypeRef = useRef(addNodeType); addNodeTypeRef.current = addNodeType;
  const visibilityRef = useRef(visibility); visibilityRef.current = visibility;
  const nodeSizesRef = useRef(nodeSizes); nodeSizesRef.current = nodeSizes;

  const prevRouteKeyRef = useRef('');
  const undoStackRef = useRef<EditEntry[]>([]);
  const redoStackRef = useRef<EditEntry[]>([]);
  const editCounterRef = useRef(0);
  const pendingEdgeRef = useRef<string | null>(null);
  // Saved viewport, used to keep the view stable across a pre/post toggle.
  const savedViewRef = useRef<{ pan: { x: number; y: number }; zoom: number } | null>(null);
  const viewKeyRef = useRef<string>('');

  // ---- Floorplan natural dimensions ----
  useEffect(() => {
    if (!floorplanUrl) {
      setImgDims(null);
      imgDimsRef.current = null;
      return;
    }
    const img = new Image();
    img.onload = () => {
      const dims = { w: img.naturalWidth, h: img.naturalHeight };
      setImgDims(dims);
      imgDimsRef.current = dims;
    };
    img.onerror = () => { setImgDims(null); imgDimsRef.current = null; };
    img.src = floorplanUrl;
  }, [floorplanUrl]);

  const syncFloorplan = useCallback(() => {
    const cy = cyRef.current;
    const img = floorplanRef.current;
    const dims = imgDimsRef.current;
    if (!cy || !img || !dims) return;
    const zoom = cy.zoom();
    const pan = cy.pan();
    img.style.left = `${pan.x}px`;
    img.style.top = `${pan.y}px`;
    img.style.width = `${dims.w * zoom}px`;
    img.style.height = `${dims.h * zoom}px`;
  }, []);

  // ---- Edit helpers (operate directly on cy, with undo/redo via restore/remove) ----
  const reportEdit = useCallback(() => {
    onEditStateChangeRef.current(undoStackRef.current.length, redoStackRef.current.length);
  }, []);

  const pushEdit = useCallback((undo: () => void, redo: () => void) => {
    undoStackRef.current.push({ undo, redo });
    redoStackRef.current = []; // a fresh edit invalidates the redo stack
    reportEdit();
    onGraphMutatedRef.current();
  }, [reportEdit]);

  const clearPendingEdge = useCallback(() => {
    const cy = cyRef.current;
    if (cy && pendingEdgeRef.current) {
      cy.getElementById(pendingEdgeRef.current).removeClass('edge-pending');
    }
    pendingEdgeRef.current = null;
  }, []);

  const addNodeAt = useCallback((pos: { x: number; y: number }) => {
    const cy = cyRef.current;
    if (!cy) return;
    try {
      const type = addNodeTypeRef.current;
      const id = `added_${type}_${++editCounterRef.current}`;
      const floor = cy.nodes().nonempty() ? cy.nodes().first().data('floor') : '';
      const added = cy.add({
        group: 'nodes',
        data: { id, label: id, type, floor },
        position: { x: pos.x, y: pos.y },
      });
      // (#3) new nodes adopt the current per-type size from the sliders.
      const size = nodeSizesRef.current[type] || NODE_SIZES[type] || 20;
      added.style({ width: size, height: size });
      if (visibilityRef.current[type] === false) added.addClass('hidden');
      pushEdit(() => added.remove(), () => { added.restore(); });
    } catch (err) { console.error('add node failed', err); }
  }, [pushEdit]);

  const addEdgeBetween = useCallback((aId: string, bId: string) => {
    const cy = cyRef.current;
    if (!cy || aId === bId) return;
    try {
      if (cy.edges(`[source="${aId}"][target="${bId}"], [source="${bId}"][target="${aId}"]`).nonempty()) return;
      const a = cy.getElementById(aId) as unknown as cytoscape.NodeSingular;
      const b = cy.getElementById(bId) as unknown as cytoscape.NodeSingular;
      if (a.empty() || b.empty()) return;
      const pa = a.position();
      const pb = b.position();
      const weight = Math.hypot(pa.x - pb.x, pa.y - pb.y);
      const id = `eadded_${++editCounterRef.current}`;
      const added = cy.add({
        group: 'edges',
        data: {
          id, source: aId, target: bId, weight,
          edgeColorType: edgeColorOf(a.data('type'), b.data('type')),
        },
      });
      if (!showEdges) added.addClass('hidden');
      pushEdit(() => added.remove(), () => { added.restore(); });
    } catch (err) { console.error('add edge failed', err); }
  }, [pushEdit, showEdges]);

  // ---- Initialize Cytoscape (rebuilds only when the base graph changes) ----
  useEffect(() => {
    if (!containerRef.current) return;

    const elements: cytoscape.ElementDefinition[] = [
      ...graphData.nodes.map((n) => ({ group: 'nodes' as const, data: n.data, position: n.position })),
      ...graphData.edges.map((e) => ({ group: 'edges' as const, data: e.data })),
    ];

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: buildStylesheet(),
      layout: { name: 'preset' },
      minZoom: 0.05,
      maxZoom: 8,
      wheelSensitivity: 0.3,
      autoungrabify: true,      // (#9) no dragging nodes
      boxSelectionEnabled: false,
    });

    cyRef.current = cy;

    // Preserve the viewport across a pre/post pruning toggle (same image) so
    // the graph stays aligned with the floorplan; fit only for a new image.
    const sameImage = viewKey === viewKeyRef.current && savedViewRef.current !== null;
    if (sameImage && savedViewRef.current) {
      cy.viewport({ zoom: savedViewRef.current.zoom, pan: savedViewRef.current.pan });
    } else {
      cy.fit(undefined, 40);
    }
    viewKeyRef.current = viewKey;
    syncFloorplan();

    onCyInit(cy);
    prevRouteKeyRef.current = '';
    undoStackRef.current = [];
    redoStackRef.current = [];
    pendingEdgeRef.current = null;
    onEditStateChangeRef.current(0, 0);

    cy.on('viewport', syncFloorplan);

    // Hover: tooltip always, plus a selection glow when a pick mode is armed.
    cy.on('mouseover', 'node', (evt) => {
      const node = evt.target;
      const pos = node.renderedPosition();
      onTooltipRef.current({ x: pos.x + 15, y: pos.y - 10, data: node.data() });
      const m = modeRef.current;
      if (m === 'delete') node.addClass('hover-delete');
      else if (m === 'route-start' || m === 'route-end' || m === 'add-edge') node.addClass('hover-target');
    });
    cy.on('mouseout', 'node', (evt) => {
      onTooltipRef.current(null);
      evt.target.removeClass('hover-target hover-delete');
    });

    // Edges glow red on hover while in delete mode.
    cy.on('mouseover', 'edge', (evt) => {
      if (modeRef.current === 'delete') evt.target.addClass('hover-delete');
    });
    cy.on('mouseout', 'edge', (evt) => evt.target.removeClass('hover-delete'));

    // Node taps: route selection, edge building, or deletion (by mode).
    cy.on('tap', 'node', (evt) => {
      const node = evt.target as cytoscape.NodeSingular;
      const m = modeRef.current;
      if (m === 'route-start' || m === 'route-end') {
        onNodeSelectRef.current(node.id());
      } else if (m === 'add-edge') {
        const first = pendingEdgeRef.current;
        if (!first) {
          pendingEdgeRef.current = node.id();
          node.addClass('edge-pending');
        } else if (first === node.id()) {
          clearPendingEdge();
        } else {
          addEdgeBetween(first, node.id());
          clearPendingEdge();
        }
      } else if (m === 'delete') {
        try {
          const removed = node.remove();
          pushEdit(() => { removed.restore(); }, () => { removed.remove(); });
        } catch (err) { console.error('delete node failed', err); }
      }
    });

    // Edge taps: deletion.
    cy.on('tap', 'edge', (evt) => {
      if (modeRef.current === 'delete') {
        try {
          const removed = evt.target.remove();
          pushEdit(() => { removed.restore(); }, () => { removed.remove(); });
        } catch (err) { console.error('delete edge failed', err); }
      }
    });

    // Background taps: add node.
    cy.on('tap', (evt) => {
      if (evt.target !== cy) return;
      if (modeRef.current === 'add-node') addNodeAt(evt.position);
    });

    return () => {
      // Remember the viewport so the next init (a stage toggle) can restore it.
      try {
        const p = cy.pan();
        savedViewRef.current = { pan: { x: p.x, y: p.y }, zoom: cy.zoom() };
      } catch { /* ignore */ }
      cy.destroy();
      cyRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graphData]);

  // Expose undo/redo to the parent.
  useEffect(() => {
    editControls.current = {
      undo: () => {
        const e = undoStackRef.current.pop();
        if (!e) return;
        try { e.undo(); } catch (err) { console.error('undo failed', err); }
        redoStackRef.current.push(e);
        onEditStateChangeRef.current(undoStackRef.current.length, redoStackRef.current.length);
        onGraphMutatedRef.current();
      },
      redo: () => {
        const e = redoStackRef.current.pop();
        if (!e) return;
        try { e.redo(); } catch (err) { console.error('redo failed', err); }
        undoStackRef.current.push(e);
        onEditStateChangeRef.current(undoStackRef.current.length, redoStackRef.current.length);
        onGraphMutatedRef.current();
      },
    };
  }, [editControls]);

  // Clear pending edge and any stale hover glow when the mode changes.
  useEffect(() => {
    if (mode !== 'add-edge') clearPendingEdge();
    cyRef.current?.elements().removeClass('hover-target hover-delete');
  }, [mode, clearPendingEdge]);

  // ---- Combined display effect: visibility + routing overlay ----
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    try {
    cy.elements().removeClass(ROUTE_CLASSES);

    const pathNodeIds = new Set<string>();
    const pathEdgeIds = new Set<string>();
    let info: RouteInfo | null = null;

    if (routeSource && routeTarget && routeSource !== routeTarget) {
      const root = cy.getElementById(routeSource);
      const goal = cy.getElementById(routeTarget);
      if (!root.empty() && !goal.empty()) {
        const posById = new Map<string, { x: number; y: number; floor: string }>();
        graphData.nodes.forEach((n) =>
          posById.set(n.data.id, { x: n.position.x, y: n.position.y, floor: n.data.floor }));
        const goalEntry = posById.get(routeTarget);
        const goalFloor = goalEntry?.floor;

        const res = cy.elements().aStar({
          root, goal, directed: false,
          weight: (edge: cytoscape.EdgeCollection) => {
            const w = Number(edge.data('weight'));
            return isFinite(w) && w > 0 ? w : 1;
          },
          heuristic: (node: cytoscape.NodeSingular) => {
            const np = posById.get(node.id());
            if (!np || !goalEntry || np.floor !== goalFloor) return 0;
            return Math.hypot(np.x - goalEntry.x, np.y - goalEntry.y);
          },
        });

        if (res.found && res.path && res.path.length > 0) {
          const floors = new Set<string>();
          const nodeTypes: Record<string, number> = {};
          res.path.nodes().forEach((n) => {
            pathNodeIds.add(n.id());
            floors.add(String(n.data('floor')));
            let t = n.data('type') as string;
            if (t === 'room' && n.data('isSubnode')) t = 'subnode';
            nodeTypes[t] = (nodeTypes[t] || 0) + 1;
          });
          res.path.edges().forEach((e) => { pathEdgeIds.add(e.id()); });
          info = {
            found: true,
            distance: Math.round(res.distance),
            segments: pathEdgeIds.size,
            crossFloor: floors.size > 1,
            nodeCount: pathNodeIds.size,
            nodeTypes,
          };
        } else {
          info = { found: false, distance: 0, segments: 0, crossFloor: false, nodeCount: 0, nodeTypes: {} };
        }
      }
    }

    const hasRoute = pathNodeIds.size > 0;
    // Endpoints, the computed path, and flagged broken nodes stay visible.
    const keepVisible = new Set(pathNodeIds);
    if (routeSource) keepVisible.add(routeSource);
    if (routeTarget) keepVisible.add(routeTarget);
    brokenNodeIds.forEach((id) => keepVisible.add(id));
    const hiddenNodeIds = new Set<string>();

    cy.nodes().forEach((node) => {
      if (keepVisible.has(node.id())) { node.removeClass('hidden'); return; }
      const type = node.data('type') as string;
      if (visibility[type] === false) { node.addClass('hidden'); hiddenNodeIds.add(node.id()); }
      else node.removeClass('hidden');
    });

    cy.edges().forEach((edge) => {
      if (pathEdgeIds.has(edge.id())) { edge.removeClass('hidden'); return; }
      if (!showEdges) { edge.addClass('hidden'); return; }
      const src = edge.data('source') as string;
      const tgt = edge.data('target') as string;
      if (hiddenNodeIds.has(src) || hiddenNodeIds.has(tgt)) edge.addClass('hidden');
      else edge.removeClass('hidden');
    });

    if (hasRoute) {
      cy.elements().addClass('route-dim');
      cy.nodes().forEach((n) => { if (pathNodeIds.has(n.id())) n.removeClass('route-dim').addClass('route-hl'); });
      cy.edges().forEach((e) => { if (pathEdgeIds.has(e.id())) e.removeClass('route-dim').addClass('route-hl'); });
    }

    // Persistent endpoint markers: stay highlighted even before the
    // destination is chosen, so the chosen Start node is always visible.
    if (routeSource) cy.getElementById(routeSource).removeClass('route-dim').addClass('route-src');
    if (routeTarget) cy.getElementById(routeTarget).removeClass('route-dim').addClass('route-dst');

    onRouteComputedRef.current(info);

    const key = hasRoute ? `${routeSource}->${routeTarget}` : '';
    if (hasRoute && key !== prevRouteKeyRef.current) {
      const pathEles = cy.elements('.route-hl, .route-src, .route-dst');
      if (pathEles.nonempty()) cy.animate({ fit: { eles: pathEles, padding: 60 }, duration: 500 });
    }
    prevRouteKeyRef.current = key;
    } catch (err) {
      console.error('display update failed', err);
      onRouteComputedRef.current(null);
    }
    // editVersion is included so removing an edge on the active route breaks
    // (or reroutes) the highlighted path in real time.
  }, [graphData, visibility, showEdges, routeSource, routeTarget, editVersion, brokenNodeIds]);

  // Highlight broken / disconnected nodes while the stats list is open (#2).
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    try {
      cy.nodes().removeClass('broken');
      brokenNodeIds.forEach((id) => cy.getElementById(id).addClass('broken'));
    } catch (err) { console.error('broken highlight failed', err); }
  }, [brokenNodeIds, editVersion]);

  // ---- Node sizes ----
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.nodes().forEach((node) => {
      const type = node.data('type') as string;
      const size = nodeSizes[type] || NODE_SIZES[type] || 20;
      node.style({ width: size, height: size });
    });
  }, [nodeSizes]);

  // ---- Floorplan re-sync ----
  useEffect(() => { syncFloorplan(); }, [showFloorplan, imgDims, syncFloorplan]);

  // ---- Cursor hint by mode ----
  const cursor =
    mode === 'add-node' ? 'copy'
    : mode === 'delete' ? 'not-allowed'
    : mode === 'add-edge' || mode === 'route-start' || mode === 'route-end' ? 'crosshair'
    : 'default';

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden' }}>
      {showFloorplan && imgDims && floorplanUrl && (
        <img
          ref={floorplanRef}
          src={floorplanUrl}
          alt=""
          style={{ position: 'absolute', opacity: floorplanOpacity, pointerEvents: 'none' }}
        />
      )}
      <div ref={containerRef} style={{ width: '100%', height: '100%', cursor }} />
    </div>
  );
}
