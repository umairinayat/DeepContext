import { NavLink } from 'react-router-dom'

const BASE = import.meta.env.BASE_URL

export default function Navbar() {
    return (
        <nav className="navbar">
            <div className="navbar-inner">
                <NavLink to="/" className="navbar-logo">
                    <img src={`${BASE}logo.png`} alt="DeepContext" className="navbar-logo-img" />
                    <span>DeepContext</span>
                </NavLink>
                <div className="navbar-links">
                    <NavLink to="/" className={({ isActive }) => isActive ? 'active' : ''} end>Home</NavLink>
                    <NavLink to="/docs" className={({ isActive }) => isActive ? 'active' : ''}>Docs</NavLink>
                    <NavLink to="/demo" className={({ isActive }) => isActive ? 'active' : ''}>Demo</NavLink>
                    <a href="https://github.com/umairinayat/DeepContext" target="_blank" rel="noreferrer" className="btn-nav">
                        ⭐ GitHub
                    </a>
                </div>
            </div>
        </nav>
    )
}
