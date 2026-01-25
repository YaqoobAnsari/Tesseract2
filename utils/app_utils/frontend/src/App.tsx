import { useState, useCallback } from 'react';
import type { AppState, ProcessingResponse, NodeTypeVisibility } from './types';
import { processImage, processExample, fetchCachedResult, floorplanImageUrl } from './api';
import { NODE_TYPES } from './constants';
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

  // Node type visibility
  const [visibility, setVisibility] = useState<NodeTypeVisibility>(() => {
    const v: NodeTypeVisibility = {};
    for (const t of NODE_TYPES) v[t] = true;
    return v;
  });

  // Tooltip state
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    data: Record<string, unknown>;
  } | null>(null);

  const handleFile = useCallback(async (file: File) => {
    setAppState('processing');
    setProcessingName(file.name);
    setErrorMsg('');
    try {
      const res = await processImage(file);
      setResult(res);
      setFloorplanUrl('');  // no floorplan for uploaded files
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
        setAppState('results');
      } catch {
        // Fallback to full processing
        setAppState('processing');
        try {
          const res = await processExample(name);
          setResult(res);
          setFloorplanUrl(floorplanImageUrl(name));
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
    setTooltip(null);
    // Reset visibility
    const v: NodeTypeVisibility = {};
    for (const t of NODE_TYPES) v[t] = true;
    setVisibility(v);
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

        {appState === 'results' && result?.graph_data && (
          <div className="results-layout">
            <div className="graph-area">
              <GraphViewer
                graphData={result.graph_data}
                visibility={visibility}
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
                statistics={result.statistics}
                processingTime={result.processing_time}
                imageName={result.image_name}
              />

              <GraphControls
                visibility={visibility}
                onToggle={(type) =>
                  setVisibility((v) => ({ ...v, [type]: !v[type] }))
                }
                showFloorplan={showFloorplan}
                onFloorplanToggle={() => setShowFloorplan((v) => !v)}
                hasFloorplan={!!floorplanUrl}
                onFit={() => cyRef?.fit(undefined, 40)}
              />

              <ExportPanel cy={cyRef} graphData={result.graph_data} />

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
