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
  onToggleBroken?: (show: boolean) => void;
}

const MAX_LISTED = 40;
const REASON_LABEL: Record<string, string> = { isolated: 'no edges', room: 'room', exit: 'exit' };

export default function StatsPanel({
  statistics,
  processingTime,
  imageName,
  edited,
  connectivity,
  onFocusNode,
  onToggleBroken,
}: Props) {
  const [open, setOpen] = useState(true);
  const [connOpen, setConnOpen] = useState(false);
  const { total_nodes, total_edges, node_types, pruning_reduction } = statistics;
  const broken = connectivity && !connectivity.fullyConnected;

  const toggleConn = () =>
    setConnOpen((o) => {
      const next = !o;
      onToggleBroken?.(next);
      return next;
    });

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
                      title="Issues found. Click to inspect and highlight on the graph."
                      onClick={toggleConn}
                    >
                      &#9888;
                    </button>
                  ) : (
                    <span className="conn-ok" title="Rooms and exits fully connected">&#10003;</span>
                  )}
                </span>
              </div>

              {broken && connOpen && (
                <div className="conn-list">
                  <div className="conn-list-title">
                    {connectivity.isolatedCount > 0 && (
                      <div className="conn-flag">
                        {connectivity.isolatedCount} node
                        {connectivity.isolatedCount === 1 ? '' : 's'} with no edges
                      </div>
                    )}
                    {connectivity.roomsDisconnected > 0 && (
                      <div>
                        {connectivity.roomsDisconnected} room
                        {connectivity.roomsDisconnected === 1 ? '' : 's'} unreachable
                      </div>
                    )}
                    {connectivity.exitsDisconnected > 0 && (
                      <div>
                        {connectivity.exitsDisconnected} exit door
                        {connectivity.exitsDisconnected === 1 ? '' : 's'} unreachable
                      </div>
                    )}
                    <div className="conn-locate">
                      Highlighted in red on the graph. Click one to zoom to it.
                    </div>
                  </div>
                  {connectivity.offenders.slice(0, MAX_LISTED).map((n) => (
                    <button key={n.id} className="conn-node" onClick={() => onFocusNode?.(n.id)}>
                      <span
                        className="type-swatch"
                        style={{ backgroundColor: NODE_COLORS[n.type] || '#999' }}
                      />
                      <span className="conn-node-id">{n.id}</span>
                      <span className={`conn-reason conn-reason-${n.reason}`}>
                        {REASON_LABEL[n.reason]}
                      </span>
                    </button>
                  ))}
                  {connectivity.offenders.length > MAX_LISTED && (
                    <div className="conn-more">
                      + {connectivity.offenders.length - MAX_LISTED} more
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
