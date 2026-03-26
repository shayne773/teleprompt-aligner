function ControlButton({ onClick, disabled, children, variant = 'default' }) {
  return (
    <button type="button" className={`control-btn control-btn--${variant}`} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}

export default function ControlPanel({ connected, isRunning, onConnect, onDisconnect, onStart, onStop }) {
  return (
    <section className="control-panel" aria-label="Playback controls">
      <ControlButton onClick={onConnect} disabled={connected} variant="muted">
        Connect
      </ControlButton>
      <ControlButton onClick={onDisconnect} disabled={!connected} variant="muted">
        Disconnect
      </ControlButton>
      <ControlButton onClick={onStart} disabled={!connected || isRunning} variant="primary">
        Start
      </ControlButton>
      <ControlButton onClick={onStop} disabled={!connected || !isRunning} variant="danger">
        Stop
      </ControlButton>
    </section>
  );
}
