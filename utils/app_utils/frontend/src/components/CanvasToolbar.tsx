interface Props {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFit: () => void;
  zoomPct: number;
}

export default function CanvasToolbar({ onZoomIn, onZoomOut, onFit, zoomPct }: Props) {
  return (
    <div className="canvas-toolbar">
      <button className="zoom-btn" onClick={onZoomOut} title="Zoom out">−</button>
      <span className="zoom-readout" title="Current zoom">{zoomPct}%</span>
      <button className="zoom-btn" onClick={onZoomIn} title="Zoom in">+</button>
      <button className="zoom-btn fit-btn" onClick={onFit} title="Fit graph to view">Fit</button>
    </div>
  );
}
