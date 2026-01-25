import { useState, useEffect, useCallback, useRef } from 'react';
import type { ExampleImage } from '../types';
import { fetchExamples, exampleImageUrl } from '../api';

interface Props {
  onFile: (file: File) => void;
  onExample: (name: string, hasCached: boolean) => void;
}

export default function ImageUpload({ onFile, onExample }: Props) {
  const [examples, setExamples] = useState<ExampleImage[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchExamples()
      .then(setExamples)
      .catch(() => {});
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file && file.type === 'image/png') {
        onFile(file);
      }
    },
    [onFile],
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onFile(file);
    },
    [onFile],
  );

  return (
    <div className="upload-container">
      <div
        className={`dropzone${dragOver ? ' drag-over' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <div className="dropzone-text">
          Drop a floorplan PNG here, or click to select
        </div>
        <div className="dropzone-sub">Maximum file size: 10 MB</div>
        <input
          ref={inputRef}
          type="file"
          accept="image/png"
          style={{ display: 'none' }}
          onChange={handleFileSelect}
        />
      </div>

      {examples.length > 0 && (
        <div className="examples-section">
          <h2>Or try an example</h2>
          <div className="example-grid">
            {examples.map((ex) => (
              <div
                key={ex.name}
                className="example-card"
                onClick={() => onExample(ex.name, ex.has_cached_result)}
              >
                <img
                  src={exampleImageUrl(ex.name)}
                  alt={ex.display_name}
                  loading="lazy"
                />
                <div className="example-card-info">
                  <div className="example-card-name">{ex.display_name}</div>
                  <div className="example-card-meta">
                    <span>{ex.size_kb.toFixed(0)} KB</span>
                    {ex.has_cached_result && (
                      <span className="cached-badge">Instant</span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
