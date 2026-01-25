interface Props {
  x: number;
  y: number;
  data: Record<string, unknown>;
}

export default function NodeTooltip({ x, y, data }: Props) {
  const type = (data.type as string) || 'unknown';
  const id = (data.id as string) || '';
  const floor = (data.floor as string) || '';

  return (
    <div
      className="node-tooltip"
      style={{ left: x, top: y }}
    >
      <div className="tooltip-type">{type}</div>
      <div>ID: {id}</div>
      {floor && <div>Floor: {floor}</div>}
      {data.area != null && <div>Area: {(data.area as number).toLocaleString()} px</div>}
      {data.eqRadius != null && <div>Eq. Radius: {String(data.eqRadius)}</div>}
      {data.doorType ? <div>Door Type: {String(data.doorType)}</div> : null}
      {data.isSubnode ? <div>Subnode of: {String(data.parentRoom)}</div> : null}
    </div>
  );
}
