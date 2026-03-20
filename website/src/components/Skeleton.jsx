export function SkeletonBar({ width = '100%', height = '1rem', className = '' }) {
  return <div className={`skeleton skeleton-bar ${className}`} style={{ width, height }} />
}

export function SkeletonCard({ height = '120px', className = '' }) {
  return <div className={`skeleton skeleton-card ${className}`} style={{ height }} />
}

export function SkeletonRow({ count = 5 }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className="skeleton skeleton-bar" style={{ height: '3rem', width: '100%' }} />
      ))}
    </div>
  )
}
