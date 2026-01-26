import { useEffect, useRef, useCallback, useState } from 'react';
import cytoscape from 'cytoscape';
import type { CytoscapeGraph, NodeTypeVisibility, NodeTypeSizes } from '../types';
import { NODE_COLORS, NODE_SIZES, NODE_SHAPES, EDGE_COLORS } from '../constants';

interface Props {
  graphData: CytoscapeGraph;
  visibility: NodeTypeVisibility;
  nodeSizes: NodeTypeSizes;
  showEdges: boolean;
  showFloorplan: boolean;
  floorplanUrl: string;
  onCyInit: (cy: cytoscape.Core) => void;
  onTooltip: (
    info: { x: number; y: number; data: Record<string, unknown> } | null,
  ) => void;
}

function buildStylesheet(): cytoscape.StylesheetStyle[] {
  const nodeStyles = Object.entries(NODE_COLORS).map(
    ([type, color]) => ({
      selector: `node[type="${type}"]`,
      style: {
        'background-color': color,
        shape: NODE_SHAPES[type] || 'ellipse',
        width: NODE_SIZES[type] || 20,
        height: NODE_SIZES[type] || 20,
        label: '',
      },
    }),
  );

  const edgeStyles = Object.entries(EDGE_COLORS).map(
    ([type, color]) => ({
      selector: `edge[edgeColorType="${type}"]`,
      style: {
        'line-color': color,
        width: 1.5,
        'curve-style': 'bezier',
        opacity: 0.7,
      },
    }),
  );

  const defaults = [
    {
      selector: 'node',
      style: {
        'background-color': '#999',
        width: 15,
        height: 15,
        label: '',
      },
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
    {
      selector: '.hidden',
      style: {
        display: 'none',
      },
    },
  ];

  return [...defaults, ...nodeStyles, ...edgeStyles] as cytoscape.StylesheetStyle[];
}

export default function GraphViewer({
  graphData,
  visibility,
  nodeSizes,
  showEdges,
  showFloorplan,
  floorplanUrl,
  onCyInit,
  onTooltip,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const floorplanRef = useRef<HTMLImageElement>(null);
  const imgDimsRef = useRef<{ w: number; h: number } | null>(null);
  const [imgDims, setImgDims] = useState<{ w: number; h: number } | null>(null);

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
  // Reads from refs so the cy 'viewport' listener never goes stale.
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

    // Keep floorplan image in sync with pan / zoom
    cy.on('viewport', syncFloorplan);

    // Hover events for tooltip
    cy.on('mouseover', 'node', (evt) => {
      const node = evt.target;
      const pos = node.renderedPosition();
      onTooltip({
        x: pos.x + 15,
        y: pos.y - 10,
        data: node.data(),
      });
    });

    cy.on('mouseout', 'node', () => {
      onTooltip(null);
    });

    return () => {
      cy.destroy();
      cyRef.current = null;
    };
    // Only re-init when graph data changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graphData]);

  // Apply visibility toggles (nodes + edges)
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    const hiddenNodeIds = new Set<string>();

    cy.nodes().forEach((node) => {
      const type = node.data('type') as string;
      if (visibility[type] === false) {
        node.addClass('hidden');
        hiddenNodeIds.add(node.id());
      } else {
        node.removeClass('hidden');
      }
    });

    cy.edges().forEach((edge) => {
      if (!showEdges) {
        edge.addClass('hidden');
      } else {
        const src = edge.data('source') as string;
        const tgt = edge.data('target') as string;
        if (hiddenNodeIds.has(src) || hiddenNodeIds.has(tgt)) {
          edge.addClass('hidden');
        } else {
          edge.removeClass('hidden');
        }
      }
    });
  }, [visibility, showEdges]);

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
          style={{
            position: 'absolute',
            opacity: 0.35,
            pointerEvents: 'none',
          }}
        />
      )}
      <div
        ref={containerRef}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}
