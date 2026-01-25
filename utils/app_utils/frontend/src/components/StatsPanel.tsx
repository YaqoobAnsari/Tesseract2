import type { GraphStatistics } from '../types';
import { NODE_TYPE_LABELS, NODE_COLORS } from '../constants';

interface Props {
  statistics: GraphStatistics;
  processingTime: number;
  imageName: string;
}

export default function StatsPanel({ statistics, processingTime, imageName }: Props) {
  const { total_nodes, total_edges, node_types, pruning_reduction } = statistics;

  return (
    <div className="panel">
      <h3>Statistics</h3>

      <div className="stat-row">
        <span className="label">Image</span>
        <span className="value">{imageName}</span>
      </div>

      <div className="stat-row">
        <span className="label">Processing Time</span>
        <span className="value">
          {processingTime > 0 ? `${processingTime.toFixed(1)}s` : 'Cached'}
        </span>
      </div>

      <hr className="stat-divider" />

      <div className="stat-row">
        <span className="label">Total Nodes</span>
        <span className="value">{total_nodes}</span>
      </div>

      <div className="stat-row">
        <span className="label">Total Edges</span>
        <span className="value">{total_edges}</span>
      </div>

      {pruning_reduction != null && pruning_reduction > 0 && (
        <div className="stat-row">
          <span className="label">Pruning Reduction</span>
          <span className="value">{pruning_reduction}%</span>
        </div>
      )}

      <hr className="stat-divider" />

      {Object.entries(node_types).map(([type, count]) => (
        <div className="stat-row" key={type}>
          <span className="label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span
              className="type-swatch"
              style={{ backgroundColor: NODE_COLORS[type] || '#999' }}
            />
            {NODE_TYPE_LABELS[type] || type}
          </span>
          <span className="value">{count}</span>
        </div>
      ))}
    </div>
  );
}
