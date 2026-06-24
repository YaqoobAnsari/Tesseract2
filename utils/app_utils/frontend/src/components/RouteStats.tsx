import type { RouteInfo } from '../types';
import { NODE_COLORS } from '../constants';

interface Props {
  info: RouteInfo | null;
}

const PATH_TYPE_LABELS: Record<string, string> = {
  room: 'Room',
  subnode: 'Room Subnode',
  corridor: 'Corridor',
  door: 'Door',
  outside: 'Outdoor',
  floor_transition: 'Floor Transition',
};

// Show the navigationally meaningful types first.
const TYPE_ORDER = ['room', 'door', 'floor_transition', 'corridor', 'outside', 'subnode'];

export default function RouteStats({ info }: Props) {
  if (!info || !info.found) return null;

  const entries = TYPE_ORDER.filter((t) => info.nodeTypes[t]).map(
    (t) => [t, info.nodeTypes[t]] as const,
  );

  return (
    <div className="panel route-stats">
      <h3>Current Route</h3>

      <div className="stat-row">
        <span className="label">Path length</span>
        <span className="value">{info.distance.toLocaleString()} px</span>
      </div>
      <div className="stat-row">
        <span className="label">Nodes on path</span>
        <span className="value">{info.nodeCount}</span>
      </div>
      <div className="stat-row">
        <span className="label">Segments</span>
        <span className="value">{info.segments}</span>
      </div>

      {info.crossFloor && <span className="route-badge">Crosses floors</span>}

      <hr className="stat-divider" />
      <label className="stage-label">Nodes traversed</label>
      {entries.map(([t, c]) => (
        <div className="stat-row" key={t}>
          <span className="label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span className="type-swatch" style={{ backgroundColor: NODE_COLORS[t] || '#999' }} />
            {PATH_TYPE_LABELS[t] || t}
          </span>
          <span className="value">{c}</span>
        </div>
      ))}
    </div>
  );
}
