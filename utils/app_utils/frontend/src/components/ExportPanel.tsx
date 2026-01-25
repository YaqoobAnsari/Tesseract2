import type { CytoscapeGraph } from '../types';

interface Props {
  cy: cytoscape.Core | null;
  graphData: CytoscapeGraph;
}

export default function ExportPanel({ cy, graphData }: Props) {
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
    const blob = new Blob([JSON.stringify(graphData, null, 2)], {
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
        <button className="btn" onClick={handleExportPng}>
          Export PNG
        </button>
        <button className="btn" onClick={handleExportJson}>
          Export JSON
        </button>
      </div>
    </div>
  );
}
