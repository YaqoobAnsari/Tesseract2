import { useEffect, useRef, useCallback, useState } from 'react';
import cytoscape from 'cytoscape';
import type {
  CytoscapeGraph,
  NodeTypeVisibility,
  NodeTypeSizes,
  RouteInfo,
} from '../types';
import {
  NODE_COLORS,
  NODE_SIZES,
  NODE_SHAPES,
  EDGE_COLORS,
  ROUTE_COLORS,
  ROUTE_ENDPOINT_TYPES,
} from '../constants';

interface Props {
  graphData: CytoscapeGraph;
  visibility: NodeTypeVisibility;
  nodeSizes: NodeTypeSizes;
  showEdges: boolean;
  showFloorplan: boolean;
  floorplanUrl: string;
  routeSource: string | null;
  routeTarget: string | null;
  onCyInit: (cy: cytoscape.Core) => void;
  onTooltip: (
    info: { x: number; y: number; data: Record<string, unknown> } | null,
  ) => void;
  onNodePick: (id: string) => void;
  onRouteComputed: (info: RouteInfo | null) => void;
}

const ROUTE_CLASSES = 'route-hl route-dim route-src route-dst';

function isPickable(node: cytoscape.NodeSingular): boolean {
  const type = node.data('type') as string;
  if (node.data('isSubnode')) return false;
  return (ROUTE_ENDPOINT_TYPES as readonly string[]).includes(type);
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
    style: {
      'line-color': color,
      width: 1.5,
      'curve-style': 'bezier',
      opacity: 0.7,
    },
  }));

  const defaults = [
    {
      selector: 'node',
      style: { 'background-color': '#999', width: 15, height: 15, label: '' },
    },
    {
      selector: 'edge',
      style: {
        'line-color': '#999',
        width: 1.5,
        'curve-style': 'bezier',
        opacity: 0.5,
      },
    },
    { selector: '.hidden', style: { display: 'none' } },
  ];

  // Routing overlay. Declared last so it wins over the type styles above.
  const routeStyles = [
    {
      selector: '.route-dim',
      style: { opacity: 0.1, 'text-opacity': 0 },
    },
    {
      selector: 'edge.route-hl',
      style: {
        'line-color': ROUTE_COLORS.path,
        width: 5,
        opacity: 1,
        'z-index': 9999,
      },
    },
    {
      selector: 'node.route-hl',
      style: {
        'border-width': 3,
        'border-color': ROUTE_COLORS.path,
        opacity: 1,
        'z-index': 9999,
      },
    },
    {
      selector: 'node.route-src',
      style: {
        'background-color': ROUTE_COLORS.source,
        'border-width': 4,
        'border-color': '#1b5e20',
        label: 'data(label)',
        color: '#1b5e20',
        'font-size': 13,
        'font-weight': 'bold',
        'text-outline-color': '#ffffff',
        'text-outline-width': 2,
        opacity: 1,
        'z-index': 10000,
      },
    },
    {
      selector: 'node.route-dst',
      style: {
        'background-color': ROUTE_COLORS.target,
        'border-width': 4,
        'border-color': '#b71c1c',
        label: 'data(label)',
        color: '#b71c1c',
        'font-size': 13,
        'font-weight': 'bold',
        'text-outline-color': '#ffffff',
        'text-outline-width': 2,
        opacity: 1,
        'z-index': 10000,
      },
    },
  ];

  return [
    ...defaults,
    ...nodeStyles,
    ...edgeStyles,
    ...routeStyles,
  ] as cytoscape.StylesheetStyle[];
}

export default function GraphViewer({
  graphData,
  visibility,
  nodeSizes,
  showEdges,
  showFloorplan,
  floorplanUrl,
  routeSource,
  routeTarget,
  onCyInit,
  onTooltip,
  onNodePick,
  onRouteComputed,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const floorplanRef = useRef<HTMLImageElement>(null);
  const imgDimsRef = useRef<{ w: number; h: number } | null>(null);
  const [imgDims, setImgDims] = useState<{ w: number; h: number } | null>(null);

  // Keep latest callbacks in refs so the cytoscape event handlers
  // (bound once per graph) never go stale.
  const onNodePickRef = useRef(onNodePick);
  onNodePickRef.current = onNodePick;
  const onRouteComputedRef = useRef(onRouteComputed);
  onRouteComputedRef.current = onRouteComputed;

  // Remember the last routed endpoints so we only re-fit the viewport when
  // the route itself changes, not on every filter toggle.
  const prevRouteKeyRef = useRef('');

  // Pre-load floorplan image to get its natural dimensions
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
    img.onerror = () => {
      setImgDims(null);
      imgDimsRef.current = null;
    };
    img.src = floorplanUrl;
  }, [floorplanUrl]);

  // Position the floorplan <img> to match Cytoscape's viewport.
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

  // Initialize Cytoscape
  useEffect(() => {
    if (!containerRef.current) return;

    const elements: cytoscape.ElementDefinition[] = [
      ...graphData.nodes.map((n) => ({
        group: 'nodes' as const,
        data: n.data,
        position: n.position,
      })),
      ...graphData.edges.map((e) => ({
        group: 'edges' as const,
        data: e.data,
      })),
    ];

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: buildStylesheet(),
      layout: { name: 'preset' },
      minZoom: 0.1,
      maxZoom: 5,
      wheelSensitivity: 0.3,
    });

    cy.fit(undefined, 40);
    cyRef.current = cy;
    onCyInit(cy);
    prevRouteKeyRef.current = '';

    cy.on('viewport', syncFloorplan);

    cy.on('mouseover', 'node', (evt) => {
      const node = evt.target;
      const pos = node.renderedPosition();
      onTooltip({ x: pos.x + 15, y: pos.y - 10, data: node.data() });
    });

    cy.on('mouseout', 'node', () => onTooltip(null));

    // Tap a routing-eligible node to assign it as start / destination.
    cy.on('tap', 'node', (evt) => {
      const node = evt.target as cytoscape.NodeSingular;
      if (isPickable(node)) onNodePickRef.current(node.id());
    });

    return () => {
      cy.destroy();
      cyRef.current = null;
    };
    // Only re-init when graph data changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graphData]);

  // Combined display effect: applies node/edge visibility AND the routing
  // overlay together, so the two never clobber each other. Re-runs when
  // filters or the chosen endpoints change.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    cy.elements().removeClass(ROUTE_CLASSES);

    // ---- Compute the route (if both endpoints are chosen) ----
    const pathNodeIds = new Set<string>();
    const pathEdgeIds = new Set<string>();
    let info: RouteInfo | null = null;

    if (routeSource && routeTarget && routeSource !== routeTarget) {
      const root = cy.getElementById(routeSource);
      const goal = cy.getElementById(routeTarget);

      if (!root.empty() && !goal.empty()) {
        // Positions and floors come from the graph data, keyed by node id.
        const posById = new Map<string, { x: number; y: number; floor: string }>();
        graphData.nodes.forEach((n) =>
          posById.set(n.data.id, {
            x: n.position.x,
            y: n.position.y,
            floor: n.data.floor,
          }),
        );
        const goalEntry = posById.get(routeTarget);
        const goalFloor = goalEntry?.floor;

        const res = cy.elements().aStar({
          root,
          goal,
          directed: false,
          weight: (edge: cytoscape.EdgeCollection) => {
            const w = Number(edge.data('weight'));
            return isFinite(w) && w > 0 ? w : 1;
          },
          // Straight-line distance to the goal. Admissible within a floor,
          // and returns 0 across floors so A* stays optimal building-wide.
          heuristic: (node: cytoscape.NodeSingular) => {
            const np = posById.get(node.id());
            if (!np || !goalEntry || np.floor !== goalFloor) return 0;
            return Math.hypot(np.x - goalEntry.x, np.y - goalEntry.y);
          },
        });

        if (res.found && res.path && res.path.length > 0) {
          const floors = new Set<string>();
          res.path.nodes().forEach((n) => {
            pathNodeIds.add(n.id());
            floors.add(String(n.data('floor')));
          });
          res.path.edges().forEach((e) => {
            pathEdgeIds.add(e.id());
          });
          info = {
            found: true,
            distance: Math.round(res.distance),
            segments: pathEdgeIds.size,
            crossFloor: floors.size > 1,
          };
        } else {
          info = { found: false, distance: 0, segments: 0, crossFloor: false };
        }
      }
    }

    const hasRoute = pathNodeIds.size > 0;

    // ---- Base visibility (filters). Path elements are never hidden. ----
    const hiddenNodeIds = new Set<string>();
    cy.nodes().forEach((node) => {
      if (pathNodeIds.has(node.id())) {
        node.removeClass('hidden');
        return;
      }
      const type = node.data('type') as string;
      if (visibility[type] === false) {
        node.addClass('hidden');
        hiddenNodeIds.add(node.id());
      } else {
        node.removeClass('hidden');
      }
    });

    cy.edges().forEach((edge) => {
      if (pathEdgeIds.has(edge.id())) {
        edge.removeClass('hidden');
        return;
      }
      if (!showEdges) {
        edge.addClass('hidden');
        return;
      }
      const src = edge.data('source') as string;
      const tgt = edge.data('target') as string;
      if (hiddenNodeIds.has(src) || hiddenNodeIds.has(tgt)) {
        edge.addClass('hidden');
      } else {
        edge.removeClass('hidden');
      }
    });

    // ---- Routing overlay (dim the rest, highlight the path) ----
    if (hasRoute) {
      cy.elements().addClass('route-dim');
      cy.nodes().forEach((n) => {
        if (pathNodeIds.has(n.id())) n.removeClass('route-dim').addClass('route-hl');
      });
      cy.edges().forEach((e) => {
        if (pathEdgeIds.has(e.id())) e.removeClass('route-dim').addClass('route-hl');
      });
      cy.getElementById(routeSource!).removeClass('route-dim').addClass('route-src');
      cy.getElementById(routeTarget!).removeClass('route-dim').addClass('route-dst');
    }

    // ---- Report result and re-fit only when the route changed ----
    onRouteComputedRef.current(info);

    const key = hasRoute ? `${routeSource}->${routeTarget}` : '';
    if (hasRoute && key !== prevRouteKeyRef.current) {
      const pathEles = cy.elements('.route-hl, .route-src, .route-dst');
      if (pathEles.nonempty()) {
        cy.animate({ fit: { eles: pathEles, padding: 60 }, duration: 500 });
      }
    }
    prevRouteKeyRef.current = key;
  }, [graphData, visibility, showEdges, routeSource, routeTarget]);

  // Apply node sizes dynamically
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.nodes().forEach((node) => {
      const type = node.data('type') as string;
      const size = nodeSizes[type] || NODE_SIZES[type] || 20;
      node.style({ width: size, height: size });
    });
  }, [nodeSizes]);

  // Re-sync floorplan position when it becomes visible or dims load
  useEffect(() => {
    syncFloorplan();
  }, [showFloorplan, imgDims, syncFloorplan]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden' }}>
      {showFloorplan && imgDims && floorplanUrl && (
        <img
          ref={floorplanRef}
          src={floorplanUrl}
          alt=""
          style={{ position: 'absolute', opacity: 0.35, pointerEvents: 'none' }}
        />
      )}
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
}
