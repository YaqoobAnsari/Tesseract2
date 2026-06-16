import type { RouteEndpoint, RouteInfo } from '../types';
import { NODE_TYPE_LABELS, ROUTE_COLORS } from '../constants';

interface Props {
  endpoints: RouteEndpoint[];
  source: string | null;
  target: string | null;
  onSourceChange: (id: string | null) => void;
  onTargetChange: (id: string | null) => void;
  onSwap: () => void;
  onClear: () => void;
  info: RouteInfo | null;
}

function optionLabel(ep: RouteEndpoint): string {
  const typeLabel = NODE_TYPE_LABELS[ep.type] || ep.type;
  return ep.type === 'room' ? ep.label : `${ep.label} (${typeLabel})`;
}

export default function RoutePanel({
  endpoints,
  source,
  target,
  onSourceChange,
  onTargetChange,
  onSwap,
  onClear,
  info,
}: Props) {
  const hasRoute = !!source && !!target;

  return (
    <div className="panel">
      <h3>Navigation</h3>

      <p className="route-hint">
        Pick a start and a destination from the menus, or click two nodes on
        the graph. The shortest route is computed with A* search.
      </p>

      <label className="route-field">
        <span className="route-dot" style={{ background: ROUTE_COLORS.source }} />
        Start
        <select
          className="route-select"
          value={source ?? ''}
          onChange={(e) => onSourceChange(e.target.value || null)}
        >
          <option value="">Select start…</option>
          {endpoints.map((ep) => (
            <option key={ep.id} value={ep.id} disabled={ep.id === target}>
              {optionLabel(ep)}
            </option>
          ))}
        </select>
      </label>

      <label className="route-field">
        <span className="route-dot" style={{ background: ROUTE_COLORS.target }} />
        Destination
        <select
          className="route-select"
          value={target ?? ''}
          onChange={(e) => onTargetChange(e.target.value || null)}
        >
          <option value="">Select destination…</option>
          {endpoints.map((ep) => (
            <option key={ep.id} value={ep.id} disabled={ep.id === source}>
              {optionLabel(ep)}
            </option>
          ))}
        </select>
      </label>

      <div className="route-buttons">
        <button className="btn" onClick={onSwap} disabled={!hasRoute}>
          Swap
        </button>
        <button className="btn" onClick={onClear} disabled={!source && !target}>
          Clear
        </button>
      </div>

      {hasRoute && info && (
        <div className={`route-result${info.found ? '' : ' route-result-fail'}`}>
          {info.found ? (
            <>
              <div className="route-result-row">
                <span className="label">Path length</span>
                <span className="value">{info.distance.toLocaleString()} px</span>
              </div>
              <div className="route-result-row">
                <span className="label">Segments</span>
                <span className="value">{info.segments}</span>
              </div>
              {info.crossFloor && (
                <span className="route-badge">Crosses floors</span>
              )}
            </>
          ) : (
            <span>No route found between these points.</span>
          )}
        </div>
      )}
    </div>
  );
}
