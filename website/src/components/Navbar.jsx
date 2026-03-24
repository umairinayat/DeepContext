import { NavLink, useNavigate } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'
import ThemeToggle from './ThemeToggle'
import MobileMenu from './MobileMenu'

export default function Navbar() {
  const { isAuthenticated, user, logout } = useAuth()
  const navigate = useNavigate()
  const [scrolled, setScrolled] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  useEffect(() => {
    function handleScroll() {
      setScrolled(window.scrollY > 50)
    }
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  function handleLogout() {
    logout()
    navigate('/')
  }

  return (
    <>
      <nav className={`navbar ${scrolled ? 'navbar-scrolled' : ''}`}>
        <div className="navbar-inner">
          <NavLink to="/" className="navbar-logo">
            <span className="navbar-logo-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
              </svg>
            </span>
            <span>DeepContext</span>
          </NavLink>

          <div className="navbar-center">
            <NavLink to="/docs" className={({ isActive }) => isActive ? 'active' : ''}>Docs</NavLink>
            <NavLink to="/demo" className={({ isActive }) => isActive ? 'active' : ''}>Demo</NavLink>
            <NavLink to="/dashboard" className={({ isActive }) => isActive ? 'active' : ''}>Dashboard</NavLink>
            {isAuthenticated && (
              <NavLink to="/dashboard/settings" className={({ isActive }) => isActive ? 'active' : ''}>Settings</NavLink>
            )}
            <a href="https://github.com/umairinayat/DeepContext" target="_blank" rel="noreferrer">GitHub</a>
          </div>

          <div className="navbar-right">
            {isAuthenticated ? (
              <button
                type="button"
                className="btn btn-secondary btn-sm btn-pill"
                onClick={handleLogout}
                title={user?.username ? `Signed in as ${user.username}` : 'Sign out'}
              >
                Logout
              </button>
            ) : (
              <NavLink to="/login" className="btn btn-primary btn-sm btn-pill">
                Sign In
              </NavLink>
            )}
            <ThemeToggle />
            <button
              className="navbar-hamburger"
              onClick={() => setMobileMenuOpen(o => !o)}
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
              ) : (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/>
                </svg>
              )}
            </button>
          </div>
        </div>
      </nav>
      <MobileMenu isOpen={mobileMenuOpen} onClose={() => setMobileMenuOpen(false)} />
    </>
  )
}
