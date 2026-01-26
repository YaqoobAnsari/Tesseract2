import type { NodeTypeVisibility, GraphStage } from '../types';
import { NODE_TYPES, NODE_COLORS, NODE_TYPE_LABELS } from '../constants';

interface Props {
  visibility: NodeTypeVisibility;
  onToggle: (type: string) => void;
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
        <div className="type-toggle" key={type}>
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
      ))}

      <hr className="stat-divider" />

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
