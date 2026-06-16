import type { InteractionMode } from '../types';
import { NODE_TYPES, NODE_COLORS, NODE_TYPE_LABELS } from '../constants';

interface Props {
  mode: InteractionMode;
  addNodeType: string;
  onAddNodeTypeChange: (type: string) => void;
  onSetMode: (mode: InteractionMode) => void;
  onUndo: () => void;
  canUndo: boolean;
}

const HINTS: Partial<Record<InteractionMode, string>> = {
  'add-node': 'Click an empty spot on the graph to drop a node.',
  'add-edge': 'Click one node, then another, to connect them.',
  delete: 'Click any node or edge to remove it.',
};

export default function EditPanel({
  mode,
  addNodeType,
  onAddNodeTypeChange,
  onSetMode,
  onUndo,
  canUndo,
}: Props) {
  const tool = (m: InteractionMode, label: string) => (
    <button
      className={`edit-tool${mode === m ? ' active' : ''}`}
      onClick={() => onSetMode(mode === m ? 'idle' : m)}
    >
      {label}
    </button>
  );

  return (
    <div className="panel">
      <h3>Edit Graph</h3>
      <p className="route-hint">
        Fix extraction defects by hand. Add or remove nodes and edges directly on
        the canvas.
      </p>

      <div className="edit-tools">
        {tool('add-node', 'Add Node')}
        {tool('add-edge', 'Add Edge')}
        {tool('delete', 'Delete')}
      </div>

      {mode === 'add-node' && (
        <div className="edit-nodetype">
          <label className="stage-label">New node type</label>
          <div className="nodetype-grid">
            {NODE_TYPES.map((t) => (
              <button
                key={t}
                className={`nodetype-chip${addNodeType === t ? ' active' : ''}`}
                onClick={() => onAddNodeTypeChange(t)}
              >
                <span className="type-swatch" style={{ backgroundColor: NODE_COLORS[t] }} />
                {NODE_TYPE_LABELS[t]}
              </button>
            ))}
          </div>
        </div>
      )}

      {HINTS[mode] && <div className="edit-hint">{HINTS[mode]}</div>}

      <div className="route-buttons" style={{ marginTop: 10 }}>
        <button className="btn" onClick={onUndo} disabled={!canUndo}>Undo</button>
        <button
          className="btn"
          onClick={() => onSetMode('idle')}
          disabled={mode === 'idle'}
        >
          Done
        </button>
      </div>
    </div>
  );
}
