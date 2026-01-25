interface Props {
  name: string;
}

export default function ProcessingStatus({ name }: Props) {
  return (
    <div className="processing-overlay">
      <div className="spinner" />
      <div className="processing-text">
        Processing <strong>{name}</strong>...
      </div>
      <div className="processing-text" style={{ fontSize: '0.8rem' }}>
        This may take up to 3 minutes
      </div>
    </div>
  );
}
