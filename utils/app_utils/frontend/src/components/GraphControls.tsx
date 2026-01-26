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
  hasFloorplan: boolean;
  onFit: () => void;
  graphStage: GraphStage;
  onGraphStageChange: (stage: GraphStage) => void;
  hasPrePruning: boolean;
}

export default function GraphControls({
  visibility,
  onToggle,
  nodeSizes,
  onNodeSizeChange,
  showEdges,
  onEdgesToggle,
  showFloorplan,
  onFloorplanToggle,
  hasFloorplan,
  onFit,
  graphStage,
  onGraphStageChange,
  hasPrePruning,
}: Props) {
  return (
    <div className="panel">
      <h3>Controls</h3>

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

      {NODE_TYPES.map((type) => (
        <div className="type-control-group" key={type}>
          <div className="type-toggle">
            <label>
              <input
                type="checkbox"
                checked={visibility[type] !== false}
                onChange={() => onToggle(type)}
              />
              <span
                className="type-swatch"
                style={{ backgroundColor: NODE_COLORS[type] }}
              />
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
          <input
            type="checkbox"
            checked={showEdges}
            onChange={onEdgesToggle}
          />
          Show Edges
        </label>
      </div>

      {hasFloorplan && (
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
      )}

      <div style={{ marginTop: 8 }}>
        <button className="btn btn-block" onClick={onFit}>
          Fit to View
        </button>
      </div>
    </div>
  );
}
