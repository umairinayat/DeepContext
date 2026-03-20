import { useEffect } from 'react'
import { NavLink } from 'react-router-dom'

export default function MobileMenu({ isOpen, onClose }) {
  useEffect(() => {
    if (!isOpen) return
    function handleKey(e) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [isOpen, onClose])

  // Prevent body scroll when open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }
    return () => { document.body.style.overflow = '' }
  }, [isOpen])

  return (
    <div className={`mobile-menu-overlay ${isOpen ? 'open' : ''}`}>
      <NavLink to="/" onClick={onClose} className={({ isActive }) => isActive ? 'active' : ''} end>Home</NavLink>
      <NavLink to="/docs" onClick={onClose} className={({ isActive }) => isActive ? 'active' : ''}>Docs</NavLink>
      <NavLink to="/demo" onClick={onClose} className={({ isActive }) => isActive ? 'active' : ''}>Demo</NavLink>
      <NavLink to="/dashboard" onClick={onClose} className={({ isActive }) => isActive ? 'active' : ''}>Dashboard</NavLink>
      <a href="https://github.com/umairinayat/DeepContext" target="_blank" rel="noreferrer" className="btn btn-primary btn-pill" style={{ marginTop: '1rem', width: '200px' }}>
        GitHub
      </a>
    </div>
  )
}
