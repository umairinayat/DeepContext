import { useEffect } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function MobileMenu({ isOpen, onClose }) {
  const { isAuthenticated, logout } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    if (!isOpen) return
    function handleKey(e) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [isOpen, onClose])

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }
    return () => { document.body.style.overflow = '' }
  }, [isOpen])

  function handleLogout() {
    logout()
    onClose()
    navigate('/')
  }

  return (
    <div className={`mobile-menu-overlay ${isOpen ? 'open' : ''}`}>
      <NavLink to="/" onClick={onClose} className={({ isActive }) => isActive ? 'active' : ''} end>Home</NavLink>
      <NavLink to="/docs" onClick={onClose} className={({ isActive }) => isActive ? 'active' : ''}>Docs</NavLink>
      <NavLink to="/demo" onClick={onClose} className={({ isActive }) => isActive ? 'active' : ''}>Demo</NavLink>
      <NavLink to="/dashboard" onClick={onClose} className={({ isActive }) => isActive ? 'active' : ''}>Dashboard</NavLink>
      {isAuthenticated ? (
        <>
          <NavLink to="/dashboard/settings" onClick={onClose} className={({ isActive }) => isActive ? 'active' : ''}>Settings</NavLink>
          <button type="button" className="btn btn-secondary btn-pill" onClick={handleLogout}>
            Logout
          </button>
        </>
      ) : (
        <NavLink to="/login" onClick={onClose} className="btn btn-primary btn-pill">
          Sign In
        </NavLink>
      )}
      <a href="https://github.com/umairinayat/DeepContext" target="_blank" rel="noreferrer" className="btn btn-primary btn-pill" style={{ marginTop: '1rem', width: '200px' }}>
        GitHub
      </a>
    </div>
  )
}
