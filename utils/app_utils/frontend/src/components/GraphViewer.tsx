import { useEffect, useRef, useCallback } from 'react';
import cytoscape from 'cytoscape';
import type { CytoscapeGraph, NodeTypeVisibility } from '../types';
import { NODE_COLORS, NODE_SIZES, NODE_SHAPES, EDGE_COLORS } from '../constants';

interface Props {
  graphData: CytoscapeGraph;
  visibility: NodeTypeVisibility;
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
  showFloorplan,
  floorplanUrl,
  onCyInit,
  onTooltip,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);

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

  // Apply visibility toggles
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    // Build set of hidden node IDs for edge filtering
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
      const src = edge.data('source') as string;
      const tgt = edge.data('target') as string;
      if (hiddenNodeIds.has(src) || hiddenNodeIds.has(tgt)) {
        edge.addClass('hidden');
      } else {
        edge.removeClass('hidden');
      }
    });
  }, [visibility]);

  // Apply floorplan background
  const updateBackground = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    if (showFloorplan && floorplanUrl) {
      el.style.backgroundImage = `url(${floorplanUrl})`;
      el.style.backgroundSize = 'contain';
      el.style.backgroundRepeat = 'no-repeat';
      el.style.backgroundPosition = 'center';
      el.style.backgroundColor = '#f0f0f0';
    } else {
      el.style.backgroundImage = 'none';
      el.style.backgroundColor = '#ffffff';
    }
  }, [showFloorplan, floorplanUrl]);

  useEffect(() => {
    updateBackground();
  }, [updateBackground]);

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '100%' }}
    />
  );
}
