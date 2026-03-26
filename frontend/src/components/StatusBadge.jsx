const STATUS_LABELS = {
  connected: 'Connected',
  connecting: 'Connecting',
  disconnected: 'Disconnected',
  error: 'Error'
};

export default function StatusBadge({ status }) {
  return (
    <div className={`status-badge status-badge--${status}`}>
      <span className="status-dot" aria-hidden="true" />
      <span>{STATUS_LABELS[status] || status}</span>
    </div>
  );
}
