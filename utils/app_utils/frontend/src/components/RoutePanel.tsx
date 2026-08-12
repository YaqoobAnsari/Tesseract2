import type { RouteEndpoint, RouteInfo, InteractionMode } from '../types';
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
  mode: InteractionMode;
  onArmStart: () => void;
  onArmEnd: () => void;
}

function labelFor(endpoints: RouteEndpoint[], id: string | null): string {
  if (!id) return 'not set';
  const ep = endpoints.find((e) => e.id === id);
  return ep ? ep.label : id;
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
  mode,
  onArmStart,
  onArmEnd,
}: Props) {
  const hasRoute = !!source && !!target;

  return (
    <div className="panel">
      <h3>Navigation</h3>

      <p className="route-hint">
        Click <strong>Set Start</strong>, then click a node on the graph. Do the
        same for <strong>Set Destination</strong>. The shortest route is computed
        with A* search.
      </p>

      <div className="route-pick">
        <button
          className={`btn route-pick-btn${mode === 'route-start' ? ' arming' : ''}`}
          onClick={onArmStart}
        >
          <span className="route-dot" style={{ background: ROUTE_COLORS.source }} />
          {mode === 'route-start' ? 'Click a node…' : 'Set Start'}
        </button>
        <div className="route-current">{labelFor(endpoints, source)}</div>
      </div>

      <div className="route-pick">
        <button
          className={`btn route-pick-btn${mode === 'route-end' ? ' arming' : ''}`}
          onClick={onArmEnd}
        >
          <span className="route-dot" style={{ background: ROUTE_COLORS.target }} />
          {mode === 'route-end' ? 'Click a node…' : 'Set Destination'}
        </button>
        <div className="route-current">{labelFor(endpoints, target)}</div>
      </div>

      <div className="route-buttons">
        <button className="btn" onClick={onSwap} disabled={!hasRoute}>Swap</button>
        <button className="btn" onClick={onClear} disabled={!source && !target}>Clear</button>
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
              {info.crossFloor && <span className="route-badge">Crosses floors</span>}
            </>
          ) : (
            <span>No route found between these points.</span>
          )}
        </div>
      )}

      <details className="route-advanced">
        <summary>Pick from list instead</summary>
        <label className="route-field">
          <span className="route-dot" style={{ background: ROUTE_COLORS.source }} />
          <select
            className="route-select"
            value={source ?? ''}
            onChange={(e) => onSourceChange(e.target.value || null)}
          >
            <option value="">Start…</option>
            {endpoints.map((ep) => (
              <option key={ep.id} value={ep.id} disabled={ep.id === target}>
                {optionLabel(ep)}
              </option>
            ))}
          </select>
        </label>
        <label className="route-field">
          <span className="route-dot" style={{ background: ROUTE_COLORS.target }} />
          <select
            className="route-select"
            value={target ?? ''}
            onChange={(e) => onTargetChange(e.target.value || null)}
          >
            <option value="">Destination…</option>
            {endpoints.map((ep) => (
              <option key={ep.id} value={ep.id} disabled={ep.id === source}>
                {optionLabel(ep)}
              </option>
            ))}
          </select>
        </label>
      </details>
    </div>
  );
}
