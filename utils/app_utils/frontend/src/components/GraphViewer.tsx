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
  const [imgDims, setImgDims] = useState<{ w: number; h: number } | null>(null);

  // Pre-load floorplan image to get its natural dimensions
  useEffect(() => {
    if (!floorplanUrl) {
      setImgDims(null);
      return;
    }
    const img = new Image();
    img.onload = () => setImgDims({ w: img.naturalWidth, h: img.naturalHeight });
    img.onerror = () => setImgDims(null);
    img.src = floorplanUrl;
  }, [floorplanUrl]);

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
      if (node.data('id') === '__floorplan_bg__') return;
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
      if (node.data('id') === '__floorplan_bg__') return;
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
      if (node.data('id') === '__floorplan_bg__') return;
      const type = node.data('type') as string;
      const size = nodeSizes[type] || NODE_SIZES[type] || 20;
      node.style({ width: size, height: size });
    });
  }, [nodeSizes]);

  // Manage floorplan background node — pans and zooms with the graph
  const updateFloorplan = useCallback(() => {
    const cy = cyRef.current;
    if (!cy) return;

    const existing = cy.$('#__floorplan_bg__');

    if (showFloorplan && imgDims && floorplanUrl) {
      if (!existing.length) {
        cy.add({
          group: 'nodes',
          data: { id: '__floorplan_bg__', type: '_background' },
          position: { x: imgDims.w / 2, y: imgDims.h / 2 },
        });
      }

      const bg = cy.$('#__floorplan_bg__');
      bg.style({
        width: imgDims.w,
        height: imgDims.h,
        shape: 'rectangle',
        'background-image': floorplanUrl,
        'background-fit': 'cover',
        'background-opacity': 0.35,
        'border-width': 0,
        'background-color': '#ffffff',
        'z-index': 0,
      } as Record<string, unknown>);
      bg.ungrabify();
      bg.unselectify();
      bg.removeClass('hidden');

      // Ensure regular nodes render on top
      cy.nodes().forEach((node) => {
        if (node.data('id') !== '__floorplan_bg__') {
          node.style('z-index', 10);
        }
      });
    } else {
      if (existing.length) {
        existing.addClass('hidden');
      }
    }
  }, [showFloorplan, imgDims, floorplanUrl]);

  useEffect(() => {
    updateFloorplan();
  }, [updateFloorplan]);

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '100%' }}
    />
  );
}
