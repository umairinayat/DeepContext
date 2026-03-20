export default function ProgressBar({ value, max, color = 'var(--accent)', label, count }) {
  const pct = max > 0 ? (value / max) * 100 : 0
  return (
    <div className="progress-row">
      <div className="progress-label">
        <span>{label}</span>
        <span className="text-muted text-sm">{count}</span>
      </div>
      <div className="progress-bar">
        <div className="progress-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  )
}
