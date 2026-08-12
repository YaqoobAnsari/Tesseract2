import { useState, useEffect } from 'react';

interface Props {
  name: string;
}

export default function ProcessingStatus({ name }: Props) {
  const [stage, setStage] = useState('');

  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const res = await fetch(`/api/progress/${encodeURIComponent(name)}`);
        if (res.ok && active) {
          const data = await res.json();
          if (data.stage) setStage(data.stage);
        }
      } catch { /* ignore */ }
    };

    const id = setInterval(poll, 800);
    poll(); // immediate first check
    return () => { active = false; clearInterval(id); };
  }, [name]);

  return (
    <div className="processing-overlay">
      <div className="spinner" />
      <div className="processing-text">
        Processing <strong>{name}</strong>
      </div>
      <div className="processing-stage">
        {stage || 'Starting pipeline...'}
      </div>
    </div>
  );
}
