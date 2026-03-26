export default function ScriptViewer({ lines, activeLineIndex = 0 }) {
  return (
    <div className="script-viewer" role="region" aria-label="Teleprompter script">
      {lines.map((line, index) => {
        const isActive = index === activeLineIndex;
        return (
          <p
            key={`${index}-${line.slice(0, 12)}`}
            className={`script-line ${isActive ? 'script-line--active' : ''}`}
          >
            <span className="line-number">{String(index + 1).padStart(2, '0')}</span>
            <span>{line}</span>
          </p>
        );
      })}
    </div>
  );
}
