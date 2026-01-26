import { useState, useCallback, useMemo } from 'react';
import type { AppState, ProcessingResponse, NodeTypeVisibility, NodeTypeSizes, GraphStage } from './types';
import { processImage, processExample, fetchCachedResult, floorplanImageUrl } from './api';
import { NODE_TYPES, NODE_SIZES } from './constants';
import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import ProcessingStatus from './components/ProcessingStatus';
import GraphViewer from './components/GraphViewer';
import GraphControls from './components/GraphControls';
import StatsPanel from './components/StatsPanel';
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

  // Determine which graph data to display based on stage selection
  const activeGraphData = useMemo(() => {
    if (!result) return null;
    if (graphStage === 'pre_pruning' && result.pre_pruning_graph_data) {
      return result.pre_pruning_graph_data;
    }
    return result.graph_data;
  }, [result, graphStage]);

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
                onCyInit={setCyRef}
                onTooltip={setTooltip}
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
