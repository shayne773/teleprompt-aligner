import { useMemo } from 'react';
import ControlPanel from './src/components/ControlPanel';
import ScriptViewer from './src/components/SricptViewer';
import StatusBadge from './src/components/StatusBadge';
import useTeleprompterSocket from './src/hooks/useTeleprompterSocket';

const DEFAULT_LINES = [
  'Welcome to Teleprompt Aligner.',
  'Connect to the backend to begin syncing script progress.',
  'The active line will be highlighted as updates arrive.',
  'Use start and stop controls to manage teleprompter flow.',
  'Keep this shell minimal and easy to maintain.'
];

export default function App() {
  const {
    connectionState,
    isRunning,
    scriptLines,
    currentLineIndex,
    currentWordIndex,
    confidence,
    connect,
    disconnect,
    start,
    stop,
    socketUrl
  } = useTeleprompterSocket();

  const visibleLines = useMemo(() => {
    if (Array.isArray(scriptLines) && scriptLines.length > 0) return scriptLines;
    return DEFAULT_LINES;
  }, [scriptLines]);

  const activeLine = visibleLines[currentLineIndex] || 'Waiting for script data...';

  return (
    <div className="app-shell">
      <header className="top-bar">
        <div>
          <p className="top-bar__label">teleprompt-aligner</p>
          <h1>Teleprompter Console</h1>
        </div>
        <StatusBadge status={connectionState} />
      </header>

      <ControlPanel
        connected={connectionState === 'connected'}
        isRunning={isRunning}
        onConnect={connect}
        onDisconnect={disconnect}
        onStart={start}
        onStop={stop}
      />

      <main className="main-layout">
        <section className="script-panel">
          <ScriptViewer lines={visibleLines} activeLineIndex={currentLineIndex} />
        </section>

        <aside className="info-panel">
          <h2>status</h2>
          <dl>
            <div>
              <dt>socket</dt>
              <dd>{connectionState}</dd>
            </div>
            <div>
              <dt>line index</dt>
              <dd>{Math.max(currentLineIndex, 0)}</dd>
            </div>
            <div>
              <dt>word index</dt>
              <dd>{Math.max(currentWordIndex, 0)}</dd>
            </div>
            <div>
              <dt>confidence</dt>
              <dd>{confidence == null ? 'n/a' : confidence.toFixed(2)}</dd>
            </div>
            <div>
              <dt>endpoint</dt>
              <dd className="endpoint">{socketUrl}</dd>
            </div>
            <div>
              <dt>active line</dt>
              <dd className="active-line-readout">{activeLine}</dd>
            </div>
          </dl>
        </aside>
      </main>
    </div>
  );
}
