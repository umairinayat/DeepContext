import { Link } from 'react-router-dom'

export default function EmptyState({ message, actionLabel, actionTo, onAction }) {
  return (
    <div className="empty-state">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.5 }}>
        <circle cx="12" cy="12" r="10"/><path d="M8 15h8"/><circle cx="9" cy="9" r="1"/><circle cx="15" cy="9" r="1"/>
      </svg>
      <p>{message}</p>
      {actionTo && (
        <Link to={actionTo} className="btn btn-sm btn-primary btn-pill">{actionLabel}</Link>
      )}
      {onAction && !actionTo && (
        <button className="btn btn-sm btn-outline btn-pill" onClick={onAction}>{actionLabel}</button>
      )}
    </div>
  )
}
