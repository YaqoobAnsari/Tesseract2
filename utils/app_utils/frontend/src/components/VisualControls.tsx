import type { NodeTypeVisibility, NodeTypeSizes, GraphStage } from '../types';
import { NODE_TYPES, NODE_COLORS, NODE_TYPE_LABELS } from '../constants';

interface Props {
  visibility: NodeTypeVisibility;
  onToggle: (type: string) => void;
  nodeSizes: NodeTypeSizes;
  onNodeSizeChange: (type: string, size: number) => void;
  showEdges: boolean;
  onEdgesToggle: () => void;
  showFloorplan: boolean;
  onFloorplanToggle: () => void;
  floorplanOpacity: number;
  onFloorplanOpacityChange: (v: number) => void;
  hasFloorplan: boolean;
  graphStage: GraphStage;
  onGraphStageChange: (stage: GraphStage) => void;
  hasPrePruning: boolean;
}

export default function VisualControls({
  visibility,
  onToggle,
  nodeSizes,
  onNodeSizeChange,
  showEdges,
  onEdgesToggle,
  showFloorplan,
  onFloorplanToggle,
  floorplanOpacity,
  onFloorplanOpacityChange,
  hasFloorplan,
  graphStage,
  onGraphStageChange,
  hasPrePruning,
}: Props) {
  return (
    <div className="panel">
      <h3>Visual Controls</h3>

      {hasPrePruning && (
        <>
          <label className="stage-label">Graph Stage</label>
          <div className="stage-toggle">
            <button
              className={`stage-btn${graphStage === 'post_pruning' ? ' active' : ''}`}
              onClick={() => onGraphStageChange('post_pruning')}
            >
              Post-Pruning
            </button>
            <button
              className={`stage-btn${graphStage === 'pre_pruning' ? ' active' : ''}`}
              onClick={() => onGraphStageChange('pre_pruning')}
            >
              Pre-Pruning
            </button>
          </div>
          <hr className="stat-divider" />
        </>
      )}

      <label className="stage-label">Node Types &amp; Sizes</label>
      {NODE_TYPES.map((type) => (
        <div className="type-control-group" key={type}>
          <div className="type-toggle">
            <label>
              <input
                type="checkbox"
                checked={visibility[type] !== false}
                onChange={() => onToggle(type)}
              />
              <span className="type-swatch" style={{ backgroundColor: NODE_COLORS[type] }} />
              {NODE_TYPE_LABELS[type]}
            </label>
          </div>
          <div className="size-slider">
            <input
              type="range"
              min={4}
              max={60}
              value={nodeSizes[type] || 20}
              onChange={(e) => onNodeSizeChange(type, Number(e.target.value))}
            />
            <span className="size-value">{nodeSizes[type] || 20}</span>
          </div>
        </div>
      ))}

      <hr className="stat-divider" />

      <div className="type-toggle">
        <label>
          <input type="checkbox" checked={showEdges} onChange={onEdgesToggle} />
          Show Edges
        </label>
      </div>

      {hasFloorplan && (
        <>
          <div className="type-toggle">
            <label>
              <input
                type="checkbox"
                checked={showFloorplan}
                onChange={onFloorplanToggle}
              />
              Show Floorplan Background
            </label>
          </div>
          {showFloorplan && (
            <div className="size-slider opacity-slider">
              <span className="opacity-label">Opacity</span>
              <input
                type="range"
                min={0}
                max={100}
                value={Math.round(floorplanOpacity * 100)}
                onChange={(e) => onFloorplanOpacityChange(Number(e.target.value) / 100)}
              />
              <span className="size-value">{Math.round(floorplanOpacity * 100)}%</span>
            </div>
          )}
        </>
      )}
    </div>
  );
}
