import { useState } from 'react';
import type { GraphStatistics, ConnectivityInfo } from '../types';
import { NODE_TYPE_LABELS, NODE_COLORS } from '../constants';

interface Props {
  statistics: GraphStatistics;
  processingTime: number;
  imageName: string;
  edited?: boolean;
  connectivity?: ConnectivityInfo | null;
  onFocusNode?: (id: string) => void;
}

const MAX_LISTED = 40;

export default function StatsPanel({
  statistics,
  processingTime,
  imageName,
  edited,
  connectivity,
  onFocusNode,
}: Props) {
  const [open, setOpen] = useState(true);
  const [connOpen, setConnOpen] = useState(false);
  const { total_nodes, total_edges, node_types, pruning_reduction } = statistics;
  const broken = connectivity && connectivity.score < 100;

  return (
    <div className="panel">
      <button className="panel-collapse-header" onClick={() => setOpen((o) => !o)}>
        <h3>Statistics{edited ? ' (edited)' : ''}</h3>
        <span className={`chevron${open ? ' open' : ''}`}>›</span>
      </button>

      {open && (
        <div className="panel-collapse-body">
          <div className="stat-row">
            <span className="label">Image</span>
            <span className="value stat-image" title={imageName}>{imageName}</span>
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

          {connectivity && (
            <>
              <div className="stat-row">
                <span className="label">Connectivity</span>
                <span className="value">
                  {connectivity.score}%{' '}
                  {broken ? (
                    <button
                      className="conn-warn"
                      title="Some nodes are disconnected. Click to inspect."
                      onClick={() => setConnOpen((o) => !o)}
                    >
                      &#9888;
                    </button>
                  ) : (
                    <span className="conn-ok" title="Fully connected">&#10003;</span>
                  )}
                </span>
              </div>

              {broken && connOpen && (
                <div className="conn-list">
                  <div className="conn-list-title">
                    {connectivity.disconnected.length} disconnected node
                    {connectivity.disconnected.length === 1 ? '' : 's'} across{' '}
                    {connectivity.componentCount} components. Click to locate.
                  </div>
                  {connectivity.disconnected.slice(0, MAX_LISTED).map((n) => (
                    <button
                      key={n.id}
                      className="conn-node"
                      onClick={() => onFocusNode?.(n.id)}
                    >
                      <span
                        className="type-swatch"
                        style={{ backgroundColor: NODE_COLORS[n.type] || '#999' }}
                      />
                      {n.id}
                    </button>
                  ))}
                  {connectivity.disconnected.length > MAX_LISTED && (
                    <div className="conn-more">
                      + {connectivity.disconnected.length - MAX_LISTED} more
                    </div>
                  )}
                </div>
              )}
            </>
          )}

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
      )}
    </div>
  );
}
