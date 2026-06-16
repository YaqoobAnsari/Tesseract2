import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import type {
  AppState,
  ProcessingResponse,
  NodeTypeVisibility,
  NodeTypeSizes,
  GraphStage,
  RouteEndpoint,
  RouteInfo,
} from './types';
import { processImage, processExample, fetchCachedResult, floorplanImageUrl } from './api';
import { NODE_TYPES, NODE_SIZES, ROUTE_ENDPOINT_TYPES } from './constants';
import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import ProcessingStatus from './components/ProcessingStatus';
import GraphViewer from './components/GraphViewer';
import GraphControls from './components/GraphControls';
import StatsPanel from './components/StatsPanel';
import RoutePanel from './components/RoutePanel';
import ExportPanel from './components/ExportPanel';
import NodeTooltip from './components/NodeTooltip';

function App() {
  const [appState, setAppState] = useState<AppState>('idle');
  const [result, setResult] = useState<ProcessingResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [processingName, setProcessingName] = useState('');
  const [cyRef, setCyRef] = useState<cytoscape.Core | null>(null);

  // Floorplan background toggle
  const [showFloorplan, setShowFloorplan] = useState(false);
  const [floorplanUrl, setFloorplanUrl] = useState('');

  // Graph stage toggle
  const [graphStage, setGraphStage] = useState<GraphStage>('post_pruning');

  // Node type visibility
  const [visibility, setVisibility] = useState<NodeTypeVisibility>(() => {
    const v: NodeTypeVisibility = {};
    for (const t of NODE_TYPES) v[t] = true;
    return v;
  });

  // Node sizes (per-type)
  const [nodeSizes, setNodeSizes] = useState<NodeTypeSizes>(() => ({ ...NODE_SIZES }));

  // Edge visibility
  const [showEdges, setShowEdges] = useState(true);

  // Tooltip state
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    data: Record<string, unknown>;
  } | null>(null);

  // Navigation / routing state
  const [routeSource, setRouteSource] = useState<string | null>(null);
  const [routeTarget, setRouteTarget] = useState<string | null>(null);
  const [routeInfo, setRouteInfo] = useState<RouteInfo | null>(null);

  // Mirror routing endpoints into refs for the stable tap-to-select handler
  const routeSourceRef = useRef<string | null>(null);
  const routeTargetRef = useRef<string | null>(null);
  useEffect(() => { routeSourceRef.current = routeSource; }, [routeSource]);
  useEffect(() => { routeTargetRef.current = routeTarget; }, [routeTarget]);

  // Determine which graph data to display based on stage selection
  const activeGraphData = useMemo(() => {
    if (!result) return null;
    if (graphStage === 'pre_pruning' && result.pre_pruning_graph_data) {
      return result.pre_pruning_graph_data;
    }
    return result.graph_data;
  }, [result, graphStage]);

  // Nodes that can serve as routing endpoints (main rooms, transitions, outside)
  const routeEndpoints = useMemo<RouteEndpoint[]>(() => {
    if (!activeGraphData) return [];
    const types = ROUTE_ENDPOINT_TYPES as readonly string[];
    return activeGraphData.nodes
      .filter((n) => types.includes(n.data.type) && !n.data.isSubnode)
      .map((n) => ({
        id: n.data.id,
        label: n.data.label,
        type: n.data.type,
        floor: n.data.floor,
      }))
      .sort(
        (a, b) =>
          a.type.localeCompare(b.type) ||
          a.label.localeCompare(b.label, undefined, { numeric: true }),
      );
  }, [activeGraphData]);

  const resetRoute = useCallback(() => {
    setRouteSource(null);
    setRouteTarget(null);
    setRouteInfo(null);
  }, []);

  // Tap-to-select cycle: 1st node -> start, 2nd -> destination, 3rd -> new start
  const handleNodePick = useCallback((id: string) => {
    const s = routeSourceRef.current;
    const t = routeTargetRef.current;
    if (!s) {
      setRouteSource(id);
    } else if (!t) {
      if (id !== s) setRouteTarget(id);
    } else {
      setRouteSource(id);
      setRouteTarget(null);
    }
  }, []);

  const handleSwap = useCallback(() => {
    const s = routeSourceRef.current;
    const t = routeTargetRef.current;
    setRouteSource(t);
    setRouteTarget(s);
  }, []);

  const handleRouteComputed = useCallback((info: RouteInfo | null) => {
    setRouteInfo(info);
  }, []);

  // Clear any active route when the graph changes (new image or stage toggle),
  // since endpoint node IDs may no longer exist in the new graph.
  useEffect(() => {
    resetRoute();
  }, [result, graphStage, resetRoute]);

  const handleFile = useCallback(async (file: File) => {
    setAppState('processing');
    setProcessingName(file.name);
    setErrorMsg('');
    try {
      const res = await processImage(file);
      setResult(res);
      setFloorplanUrl(floorplanImageUrl(file.name));
      setGraphStage('post_pruning');
      setAppState('results');
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : 'Unknown error');
      setAppState('error');
    }
  }, []);

  const handleExample = useCallback(async (name: string, hasCached: boolean) => {
    setProcessingName(name);
    setErrorMsg('');

    if (hasCached) {
      // Load cached result instantly — no spinner needed
      try {
        const res = await fetchCachedResult(name);
        setResult(res);
        setFloorplanUrl(floorplanImageUrl(name));
        setGraphStage('post_pruning');
        setAppState('results');
      } catch {
        // Fallback to full processing
        setAppState('processing');
        try {
          const res = await processExample(name);
          setResult(res);
          setFloorplanUrl(floorplanImageUrl(name));
          setGraphStage('post_pruning');
          setAppState('results');
        } catch (e: unknown) {
          setErrorMsg(e instanceof Error ? e.message : 'Unknown error');
          setAppState('error');
        }
      }
    } else {
      setAppState('processing');
      try {
        const res = await processExample(name);
        setResult(res);
        setFloorplanUrl(floorplanImageUrl(name));
        setGraphStage('post_pruning');
        setAppState('results');
      } catch (e: unknown) {
        setErrorMsg(e instanceof Error ? e.message : 'Unknown error');
        setAppState('error');
      }
    }
  }, []);

  const handleReset = useCallback(() => {
    setAppState('idle');
    setResult(null);
    setErrorMsg('');
    setShowFloorplan(false);
    setFloorplanUrl('');
    setGraphStage('post_pruning');
    setTooltip(null);
    // Reset visibility, sizes, edges
    const v: NodeTypeVisibility = {};
    for (const t of NODE_TYPES) v[t] = true;
    setVisibility(v);
    setNodeSizes({ ...NODE_SIZES });
    setShowEdges(true);
  }, []);

  return (
    <div className="app">
      <Header />

      <div className="main-content">
        {appState === 'idle' && (
          <ImageUpload onFile={handleFile} onExample={handleExample} />
        )}

        {appState === 'processing' && (
          <ProcessingStatus name={processingName} />
        )}

        {appState === 'error' && (
          <div className="error-container">
            <div className="error-message">{errorMsg}</div>
            <button className="btn btn-primary" onClick={handleReset}>
              Try Again
            </button>
          </div>
        )}

        {appState === 'results' && activeGraphData && (
          <div className="results-layout">
            <div className="graph-area">
              <GraphViewer
                graphData={activeGraphData}
                visibility={visibility}
                nodeSizes={nodeSizes}
                showEdges={showEdges}
                showFloorplan={showFloorplan}
                floorplanUrl={floorplanUrl}
                routeSource={routeSource}
                routeTarget={routeTarget}
                onCyInit={setCyRef}
                onTooltip={setTooltip}
                onNodePick={handleNodePick}
                onRouteComputed={handleRouteComputed}
              />
              {tooltip && (
                <NodeTooltip x={tooltip.x} y={tooltip.y} data={tooltip.data} />
              )}
            </div>

            <div className="sidebar">
              <StatsPanel
                statistics={result!.statistics}
                processingTime={result!.processing_time}
                imageName={result!.image_name}
              />

              <RoutePanel
                endpoints={routeEndpoints}
                source={routeSource}
                target={routeTarget}
                onSourceChange={setRouteSource}
                onTargetChange={setRouteTarget}
                onSwap={handleSwap}
                onClear={resetRoute}
                info={routeInfo}
              />

              <GraphControls
                visibility={visibility}
                onToggle={(type) =>
                  setVisibility((v) => ({ ...v, [type]: !v[type] }))
                }
                nodeSizes={nodeSizes}
                onNodeSizeChange={(type, size) =>
                  setNodeSizes((s) => ({ ...s, [type]: size }))
                }
                showEdges={showEdges}
                onEdgesToggle={() => setShowEdges((v) => !v)}
                showFloorplan={showFloorplan}
                onFloorplanToggle={() => setShowFloorplan((v) => !v)}
                hasFloorplan={!!floorplanUrl}
                onFit={() => cyRef?.fit(undefined, 40)}
                graphStage={graphStage}
                onGraphStageChange={setGraphStage}
                hasPrePruning={!!result?.pre_pruning_graph_data}
              />

              <ExportPanel cy={cyRef} graphData={activeGraphData} />

              <div className="back-btn-row">
                <button className="btn btn-block" onClick={handleReset}>
                  Process Another Image
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
