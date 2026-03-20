import { useEffect, useRef } from 'react'

export default function SlidePanel({ isOpen, onClose, title, children }) {
  const panelRef = useRef()
  const previousFocus = useRef()

  useEffect(() => {
    if (isOpen) {
      previousFocus.current = document.activeElement
      const firstFocusable = panelRef.current?.querySelector('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')
      firstFocusable?.focus()
    } else if (previousFocus.current) {
      previousFocus.current.focus()
    }
  }, [isOpen])

  useEffect(() => {
    if (!isOpen) return
    function handleKey(e) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [isOpen, onClose])

  if (!isOpen) return null

  return (
    <>
      <div className="slide-panel-backdrop" onClick={onClose} />
      <div className="slide-panel" ref={panelRef} role="dialog" aria-modal="true" aria-label={title}>
        <div className="slide-panel-header">
          <h3>{title}</h3>
          <button className="slide-panel-close" onClick={onClose} aria-label="Close">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>
        <div className="slide-panel-body">
          {children}
        </div>
      </div>
    </>
  )
}
