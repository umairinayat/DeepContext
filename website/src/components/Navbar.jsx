import { useState, useEffect } from 'react'
import { NavLink, useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const BASE = import.meta.env.BASE_URL

export default function Navbar() {
    const { isAuthenticated, user, logout } = useAuth()
    const navigate = useNavigate()
    const location = useLocation()
    const [menuOpen, setMenuOpen] = useState(false)

    // Close menu on route change
    useEffect(() => {
        setMenuOpen(false)
    }, [location.pathname])

    // Close menu on Escape
    useEffect(() => {
        if (!menuOpen) return
        function handleKey(e) {
            if (e.key === 'Escape') setMenuOpen(false)
        }
        window.addEventListener('keydown', handleKey)
        return () => window.removeEventListener('keydown', handleKey)
    }, [menuOpen])

    // Prevent body scroll when menu open
    useEffect(() => {
        document.body.style.overflow = menuOpen ? 'hidden' : ''
        return () => { document.body.style.overflow = '' }
    }, [menuOpen])

    function handleLogout() {
        logout()
        navigate('/')
    }

    return (
        <nav className="navbar">
            <div className="navbar-inner">
                <NavLink to="/" className="navbar-logo">
                    <img src={`${BASE}logo.png`} alt="DeepContext" className="navbar-logo-img" />
                    <span>DeepContext</span>
                </NavLink>

                <button
                    className={`navbar-hamburger ${menuOpen ? 'navbar-hamburger-open' : ''}`}
                    onClick={() => setMenuOpen(o => !o)}
                    aria-label="Toggle menu"
                    aria-expanded={menuOpen}
                >
                    <span className="hamburger-line" />
                    <span className="hamburger-line" />
                    <span className="hamburger-line" />
                </button>

                {menuOpen && <div className="navbar-overlay" onClick={() => setMenuOpen(false)} />}

                <div className={`navbar-links ${menuOpen ? 'navbar-links-open' : ''}`}>
                    <NavLink to="/" className={({ isActive }) => isActive ? 'active' : ''} end>Home</NavLink>
                    <NavLink to="/docs" className={({ isActive }) => isActive ? 'active' : ''}>Docs</NavLink>
                    <NavLink to="/demo" className={({ isActive }) => isActive ? 'active' : ''}>Demo</NavLink>
                    <NavLink to="/dashboard" className={({ isActive }) => isActive ? 'active' : ''}>Dashboard</NavLink>

                    {isAuthenticated ? (
                        <>
                            <NavLink to="/settings" className={({ isActive }) => isActive ? 'active' : ''}>Settings</NavLink>
                            <button className="btn-nav btn-nav-logout" onClick={handleLogout} title={`Signed in as ${user?.username}`}>
                                Logout
                            </button>
                        </>
                    ) : (
                        <NavLink to="/login" className="btn-nav">
                            Sign In
                        </NavLink>
                    )}

                    <a href="https://github.com/umairinayat/DeepContext" target="_blank" rel="noreferrer" className="btn-nav btn-nav-github">
                        GitHub
                    </a>
                </div>
            </div>
        </nav>
    )
}
