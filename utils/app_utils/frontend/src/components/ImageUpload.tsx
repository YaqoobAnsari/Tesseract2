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
      <div className="intro">
        <h1 className="intro-title">
          <span>Tesseract</span> Floorplan Analyzer
        </h1>
        <p className="intro-abstract">
          Tesseract turns an annotated building floor plan into a navigable graph.
          It detects text labels, segments rooms and corridors, finds and
          classifies doors, and assembles a typed graph of rooms, corridors,
          doors, and floor transitions connected by walkable edges. Upload a floor
          plan or open an example to watch the pipeline run, explore the graph,
          route between any two rooms, and refine the result by hand.
        </p>
      </div>

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
        <svg
          className="dropzone-icon"
          width="52"
          height="52"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M14 3v4a1 1 0 0 0 1 1h4" />
          <path d="M17 21H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7l5 5v11a2 2 0 0 1-2 2z" />
          <path d="M12 17v-6" />
          <path d="m9.5 13.5 2.5-2.5 2.5 2.5" />
        </svg>
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

      <footer className="intro-footer">
        The Tesseract paper was published at ACM SIGSPATIAL 2025. Work by
        Carnegie Mellon University.{' '}
        <a
          href="https://dl.acm.org/doi/abs/10.1145/3748636.3762771"
          target="_blank"
          rel="noopener noreferrer"
        >
          Read the paper
        </a>
      </footer>
    </div>
  );
}
