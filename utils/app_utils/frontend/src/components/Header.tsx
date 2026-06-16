interface Props {
  onBack?: () => void;
}

export default function Header({ onBack }: Props) {
  return (
    <header className="header">
      <div className="header-left">
        {onBack && (
          <button className="back-btn" onClick={onBack} title="Back to start">
            <span className="back-arrow">‹</span> Back
          </button>
        )}
        <h1>
          <span>Tesseract++</span> Floorplan Analyzer
        </h1>
      </div>
      <div className="header-subtitle">
        Architectural floorplan to navigable graph conversion
      </div>
    </header>
  );
}
