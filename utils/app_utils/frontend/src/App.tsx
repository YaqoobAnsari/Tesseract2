import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import type {
  AppState,
  ProcessingResponse,
  NodeTypeVisibility,
  NodeTypeSizes,
  GraphStage,
  RouteEndpoint,
  RouteInfo,
  InteractionMode,
  GraphCounts,
  GraphStatistics,
  ConnectivityInfo,
} from './types';
import { processImage, processExample, fetchCachedResult, floorplanImageUrl } from './api';
import { NODE_TYPES, NODE_SIZES, ROUTE_ENDPOINT_TYPES } from './constants';
import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import ProcessingStatus from './components/ProcessingStatus';
import GraphViewer from './components/GraphViewer';
import VisualControls from './components/VisualControls';
import StatsPanel from './components/StatsPanel';
import RouteStats from './components/RouteStats';
import RoutePanel from './components/RoutePanel';
import EditPanel from './components/EditPanel';
import ExportPanel from './components/ExportPanel';
import CanvasToolbar from './components/CanvasToolbar';
import NodeTooltip from './components/NodeTooltip';

const MODE_BANNER: Partial<Record<InteractionMode, string>> = {
  'route-start': 'Click a node to set the START of the route',
  'route-end': 'Click a node to set the DESTINATION of the route',
  'add-node': 'Click an empty spot to add a node',
  'add-edge': 'Click one node, then another, to connect them',
  delete: 'Click a node or edge to delete it',
};

function countsOf(cy: cytoscape.Core): GraphCounts {
  const node_types: Record<string, number> = {};
  cy.nodes().forEach((n) => {
    const t = n.data('type') as string;
    node_types[t] = (node_types[t] || 0) + 1;
  });
  return { total_nodes: cy.nodes().length, total_edges: cy.edges().length, node_types };
}

const isMainRoom = (n: cytoscape.NodeSingular) =>
  n.data('type') === 'room' && !n.data('isSubnode');
const isExitDoor = (n: cytoscape.NodeSingular) =>
  n.data('type') === 'door' && /exit/i.test(n.id());

function connectivityOf(cy: cytoscape.Core): ConnectivityInfo {
  const empty: ConnectivityInfo = {
    score: 100, fullyConnected: true, isolatedCount: 0,
    roomsDisconnected: 0, exitsDisconnected: 0, componentCount: 0, offenders: [],
  };
  const nodes = cy.nodes();
  if (nodes.length === 0) return empty;

  const mainRooms = nodes.filter(isMainRoom);
  const exitDoors = nodes.filter(isExitDoor);
  const comps = cy.elements().components();

  // The navigable network is the component holding the most main rooms.
  let mainIds = new Set<string>();
  let bestRooms = -1;
  for (const c of comps) {
    const r = c.nodes().filter(isMainRoom).length;
    if (r > bestRooms) {
      bestRooms = r;
      mainIds = new Set<string>(c.nodes().map((n) => n.id()));
    }
  }

  // Any node with zero edges is a hard red flag.
  const isolated: string[] = [];
  const typeById = new Map<string, string>();
  nodes.forEach((n) => {
    typeById.set(n.id(), n.data('type') as string);
    if (n.connectedEdges().length === 0) isolated.push(n.id());
  });
  const isolatedSet = new Set(isolated);

  const roomsOut = mainRooms
    .filter((n) => !mainIds.has(n.id()) && !isolatedSet.has(n.id()))
    .map((n) => n.id());
  const exitsOut = exitDoors
    .filter((n) => !mainIds.has(n.id()) && !isolatedSet.has(n.id()))
    .map((n) => n.id());

  const offenders: ConnectivityInfo['offenders'] = [];
  const seen = new Set<string>();
  for (const id of isolated) { offenders.push({ id, type: typeById.get(id) || '', reason: 'isolated' }); seen.add(id); }
  for (const id of roomsOut) if (!seen.has(id)) { offenders.push({ id, type: 'room', reason: 'room' }); seen.add(id); }
  for (const id of exitsOut) if (!seen.has(id)) { offenders.push({ id, type: 'door', reason: 'exit' }); seen.add(id); }

  const tracked = new Set<string>();
  mainRooms.forEach((n) => { tracked.add(n.id()); });
  exitDoors.forEach((n) => { tracked.add(n.id()); });
  isolated.forEach((id) => tracked.add(id));
  const score = tracked.size === 0
    ? 100
    : Math.round((100 * (tracked.size - offenders.length)) / tracked.size);

  return {
    score,
    fullyConnected: offenders.length === 0,
    isolatedCount: isolated.length,
    roomsDisconnected: roomsOut.length,
    exitsDisconnected: exitsOut.length,
    componentCount: comps.length,
    offenders,
  };
}

function App() {
  const [appState, setAppState] = useState<AppState>('idle');
  const [result, setResult] = useState<ProcessingResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [processingName, setProcessingName] = useState('');
  const [cyRef, setCyRef] = useState<cytoscape.Core | null>(null);

  // Floorplan overlay: ON by default (#3), with adjustable opacity.
  const [showFloorplan, setShowFloorplan] = useState(true);
  const [floorplanUrl, setFloorplanUrl] = useState('');
  const [floorplanOpacity, setFloorplanOpacity] = useState(0.5);

  const [graphStage, setGraphStage] = useState<GraphStage>('post_pruning');

  const [visibility, setVisibility] = useState<NodeTypeVisibility>(() => {
    const v: NodeTypeVisibility = {};
    for (const t of NODE_TYPES) v[t] = true;
    return v;
  });
  const [nodeSizes, setNodeSizes] = useState<NodeTypeSizes>(() => ({ ...NODE_SIZES }));
  const [showEdges, setShowEdges] = useState(true);

  const [tooltip, setTooltip] = useState<{ x: number; y: number; data: Record<string, unknown> } | null>(null);

  // Routing
  const [routeSource, setRouteSource] = useState<string | null>(null);
  const [routeTarget, setRouteTarget] = useState<string | null>(null);
  const [routeInfo, setRouteInfo] = useState<RouteInfo | null>(null);
  const routeSourceRef = useRef<string | null>(null);
  const routeTargetRef = useRef<string | null>(null);
  useEffect(() => { routeSourceRef.current = routeSource; }, [routeSource]);
  useEffect(() => { routeTargetRef.current = routeTarget; }, [routeTarget]);

  // Interaction mode (shared by routing + editing)
  const [mode, setMode] = useState<InteractionMode>('idle');
  const modeRef = useRef<InteractionMode>('idle');
  modeRef.current = mode;
  const [addNodeType, setAddNodeType] = useState<string>('room');

  // Live counts after edits, undo depth, and connectivity analysis
  const [liveCounts, setLiveCounts] = useState<GraphCounts | null>(null);
  const [connectivity, setConnectivity] = useState<ConnectivityInfo | null>(null);
  const [undoCount, setUndoCount] = useState(0);
  const editControls = useRef<{ undo: () => void } | null>(null);
  // Bumped on every graph edit so the viewer re-runs routing/visibility live.
  const [editVersion, setEditVersion] = useState(0);
  // Whether to highlight broken/disconnected nodes on the canvas.
  const [showBroken, setShowBroken] = useState(false);

  // Zoom readout
  const [zoomPct, setZoomPct] = useState(100);

  const activeGraphData = useMemo(() => {
    if (!result) return null;
    if (graphStage === 'pre_pruning' && result.pre_pruning_graph_data) {
      return result.pre_pruning_graph_data;
    }
    return result.graph_data;
  }, [result, graphStage]);

  const routeEndpoints = useMemo<RouteEndpoint[]>(() => {
    if (!activeGraphData) return [];
    const types = ROUTE_ENDPOINT_TYPES as readonly string[];
    return activeGraphData.nodes
      .filter((n) => types.includes(n.data.type) && !n.data.isSubnode)
      .map((n) => ({ id: n.data.id, label: n.data.label, type: n.data.type, floor: n.data.floor }))
      .sort((a, b) => a.type.localeCompare(b.type) || a.label.localeCompare(b.label, undefined, { numeric: true }));
  }, [activeGraphData]);

  // Stats reflect what is on screen: live cy counts after edits, otherwise the
  // counts of the currently displayed stage (pre vs post pruning). Pruning
  // reduction is a fixed property of the image, kept from the backend.
  const statistics = useMemo<GraphStatistics>(() => {
    const pruning = result?.statistics.pruning_reduction;
    if (liveCounts) {
      return { ...liveCounts, pruning_reduction: pruning };
    }
    if (activeGraphData) {
      const node_types: Record<string, number> = {};
      for (const n of activeGraphData.nodes) {
        node_types[n.data.type] = (node_types[n.data.type] || 0) + 1;
      }
      return {
        total_nodes: activeGraphData.nodes.length,
        total_edges: activeGraphData.edges.length,
        node_types,
        pruning_reduction: pruning,
      };
    }
    return result?.statistics ?? { total_nodes: 0, total_edges: 0, node_types: {} };
  }, [result, liveCounts, activeGraphData]);

  const resetRoute = useCallback(() => {
    setRouteSource(null);
    setRouteTarget(null);
    setRouteInfo(null);
  }, []);

  const loadResult = useCallback((res: ProcessingResponse) => {
    setResult(res);
    setFloorplanUrl(floorplanImageUrl(res.image_name));
    setGraphStage('post_pruning');
    setAppState('results');
  }, []);

  const handleFile = useCallback(async (file: File) => {
    setProcessingName(file.name);
    setErrorMsg('');
    setAppState('processing');
    try {
      loadResult(await processImage(file));
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : 'Unknown error');
      setAppState('error');
    }
  }, [loadResult]);

  const handleExample = useCallback(async (name: string, hasCached: boolean) => {
    setProcessingName(name);
    setErrorMsg('');
    setAppState('processing'); // (#2) always show a smooth processing state
    try {
      let res: ProcessingResponse;
      if (hasCached) {
        try { res = await fetchCachedResult(name); }
        catch { res = await processExample(name); }
      } else {
        res = await processExample(name);
      }
      loadResult(res);
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : 'Unknown error');
      setAppState('error');
    }
  }, [loadResult]);

  const handleReset = useCallback(() => {
    setAppState('idle');
    setResult(null);
    setErrorMsg('');
    setShowFloorplan(true);
    setFloorplanUrl('');
    setGraphStage('post_pruning');
    setTooltip(null);
    setMode('idle');
    setLiveCounts(null);
    setConnectivity(null);
    setUndoCount(0);
    setShowBroken(false);
    const v: NodeTypeVisibility = {};
    for (const t of NODE_TYPES) v[t] = true;
    setVisibility(v);
    setNodeSizes({ ...NODE_SIZES });
    setShowEdges(true);
    resetRoute();
  }, [resetRoute]);

  // Node tapped on canvas while a route mode is armed.
  const handleNodeSelect = useCallback((id: string) => {
    const m = modeRef.current;
    if (m === 'route-start') { setRouteSource(id); setMode('idle'); }
    else if (m === 'route-end') { setRouteTarget(id); setMode('idle'); }
  }, []);

  const handleSwap = useCallback(() => {
    const s = routeSourceRef.current;
    const t = routeTargetRef.current;
    setRouteSource(t);
    setRouteTarget(s);
  }, []);

  const handleRouteComputed = useCallback((info: RouteInfo | null) => setRouteInfo(info), []);

  const refreshAfterEdit = useCallback(() => {
    const cy = cyRef;
    if (!cy) return;
    setLiveCounts(countsOf(cy));
    setConnectivity(connectivityOf(cy));
    setEditVersion((v) => v + 1); // (#1) force route/visibility to recompute live
  }, [cyRef]);

  // IDs to highlight on the canvas while the broken-node list is open (#2).
  const brokenNodeIds = useMemo(
    () => (showBroken && connectivity ? connectivity.offenders.map((o) => o.id) : []),
    [showBroken, connectivity],
  );

  // Center and flash a (usually disconnected) node when picked from the stats panel.
  const focusNode = useCallback((id: string) => {
    const cy = cyRef;
    if (!cy) return;
    const n = cy.getElementById(id);
    if (n.empty()) return;
    cy.animate({ center: { eles: n }, zoom: 1.8 }, { duration: 400 });
    n.flashClass('focus-flash', 1600);
  }, [cyRef]);

  const handleUndo = useCallback(() => editControls.current?.undo(), []);

  // Reset interaction state whenever the displayed graph changes.
  useEffect(() => {
    resetRoute();
    setMode('idle');
    setLiveCounts(null);
    setConnectivity(null);
    setUndoCount(0);
    setShowBroken(false);
  }, [result, graphStage, resetRoute]);

  // Track zoom + compute initial connectivity whenever the graph (re)loads.
  useEffect(() => {
    const cy = cyRef;
    if (!cy) return;
    const update = () => setZoomPct(Math.round(cy.zoom() * 100));
    update();
    cy.on('zoom', update);
    setConnectivity(connectivityOf(cy));
    return () => { cy.off('zoom', update); };
  }, [cyRef]);

  const zoomBy = useCallback((factor: number) => {
    const cy = cyRef;
    if (!cy) return;
    cy.zoom({ level: cy.zoom() * factor, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } });
  }, [cyRef]);

  const armRoute = (m: InteractionMode) => setMode((cur) => (cur === m ? 'idle' : m));

  return (
    <div className="app">
      <Header onBack={appState === 'results' ? handleReset : undefined} />

      <div className="main-content">
        {appState === 'idle' && <ImageUpload onFile={handleFile} onExample={handleExample} />}

        {appState === 'processing' && <ProcessingStatus name={processingName} />}

        {appState === 'error' && (
          <div className="error-container">
            <div className="error-message">{errorMsg}</div>
            <button className="btn btn-primary" onClick={handleReset}>Try Again</button>
          </div>
        )}

        {appState === 'results' && activeGraphData && (
          <div className="results-layout">
            {/* LEFT: stats + visual controls */}
            <aside className="sidebar sidebar-left">
              <StatsPanel
                statistics={statistics}
                processingTime={result!.processing_time}
                imageName={result!.image_name}
                edited={!!liveCounts}
                connectivity={connectivity}
                onFocusNode={focusNode}
                onToggleBroken={setShowBroken}
              />
              <RouteStats info={routeInfo} />
              <VisualControls
                visibility={visibility}
                onToggle={(type) => setVisibility((v) => ({ ...v, [type]: !v[type] }))}
                nodeSizes={nodeSizes}
                onNodeSizeChange={(type, size) => setNodeSizes((s) => ({ ...s, [type]: size }))}
                showEdges={showEdges}
                onEdgesToggle={() => setShowEdges((v) => !v)}
                showFloorplan={showFloorplan}
                onFloorplanToggle={() => setShowFloorplan((v) => !v)}
                floorplanOpacity={floorplanOpacity}
                onFloorplanOpacityChange={setFloorplanOpacity}
                hasFloorplan={!!floorplanUrl}
                graphStage={graphStage}
                onGraphStageChange={setGraphStage}
                hasPrePruning={!!result?.pre_pruning_graph_data}
              />
            </aside>

            {/* CENTER: graph */}
            <div className="graph-area">
              <GraphViewer
                graphData={activeGraphData}
                visibility={visibility}
                nodeSizes={nodeSizes}
                showEdges={showEdges}
                showFloorplan={showFloorplan}
                floorplanUrl={floorplanUrl}
                floorplanOpacity={floorplanOpacity}
                routeSource={routeSource}
                routeTarget={routeTarget}
                mode={mode}
                addNodeType={addNodeType}
                editVersion={editVersion}
                brokenNodeIds={brokenNodeIds}
                viewKey={floorplanUrl || result!.image_name}
                onCyInit={setCyRef}
                onTooltip={setTooltip}
                onNodeSelect={handleNodeSelect}
                onRouteComputed={handleRouteComputed}
                onGraphMutated={refreshAfterEdit}
                onEditStateChange={setUndoCount}
                editControls={editControls}
              />

              {mode !== 'idle' && (
                <div className="mode-banner">{MODE_BANNER[mode]}</div>
              )}

              <CanvasToolbar
                onZoomIn={() => zoomBy(1.2)}
                onZoomOut={() => zoomBy(1 / 1.2)}
                onFit={() => cyRef?.fit(undefined, 40)}
                zoomPct={zoomPct}
              />

              {tooltip && <NodeTooltip x={tooltip.x} y={tooltip.y} data={tooltip.data} />}
            </div>

            {/* RIGHT: navigation, edit, export */}
            <aside className="sidebar sidebar-right">
              <RoutePanel
                endpoints={routeEndpoints}
                source={routeSource}
                target={routeTarget}
                onSourceChange={setRouteSource}
                onTargetChange={setRouteTarget}
                onSwap={handleSwap}
                onClear={resetRoute}
                info={routeInfo}
                mode={mode}
                onArmStart={() => armRoute('route-start')}
                onArmEnd={() => armRoute('route-end')}
              />
              <EditPanel
                mode={mode}
                addNodeType={addNodeType}
                onAddNodeTypeChange={setAddNodeType}
                onSetMode={setMode}
                onUndo={handleUndo}
                undoCount={undoCount}
              />
              <ExportPanel cy={cyRef} />
            </aside>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
