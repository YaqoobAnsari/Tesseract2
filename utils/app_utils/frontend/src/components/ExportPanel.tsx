interface Props {
  cy: cytoscape.Core | null;
}

export default function ExportPanel({ cy }: Props) {
  const handleExportPng = () => {
    if (!cy) return;
    const png = cy.png({ output: 'blob', scale: 2, bg: '#ffffff', full: true });
    const url = URL.createObjectURL(png as Blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'tesseract_graph.png';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExportJson = () => {
    if (!cy) return;
    // Serialize the CURRENT graph (includes any manual edits) to the
    // Tesseract graph schema.
    const nodes = cy.nodes().map((n) => {
      const d = n.data();
      const p = n.position();
      return { ...d, position: [Math.round(p.x), Math.round(p.y)] };
    });
    const edges = cy.edges().map((e) => {
      const d = e.data();
      return { source: d.source, target: d.target, weight: d.weight };
    });
    const blob = new Blob([JSON.stringify({ nodes, edges }, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'tesseract_graph.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="panel">
      <h3>Export</h3>
      <div className="export-buttons">
        <button className="btn" onClick={handleExportPng}>Export PNG</button>
        <button className="btn" onClick={handleExportJson}>Export JSON</button>
      </div>
    </div>
  );
}
