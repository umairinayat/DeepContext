import { NavLink } from 'react-router-dom'

const BASE = import.meta.env.BASE_URL

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div className="footer-brand">
          <div className="footer-brand-name">
            <img src={`${BASE}logo.png`} alt="" width="20" height="20" style={{ borderRadius: 4 }} />
            DeepContext
          </div>
          <p>Hierarchical memory for AI agents</p>
          <p>&copy; {new Date().getFullYear()} DeepContext</p>
        </div>

        <div className="footer-links">
          <NavLink to="/docs">Docs</NavLink>
          <NavLink to="/demo">Demo</NavLink>
          <NavLink to="/dashboard">Dashboard</NavLink>
          <a href="https://github.com/umairinayat/DeepContext" target="_blank" rel="noreferrer">GitHub</a>
        </div>

        <div className="footer-right">
          <p>Built by <a href="https://github.com/umairinayat" target="_blank" rel="noreferrer">umairinayat</a></p>
          <span className="badge badge-muted" style={{ marginTop: '0.5rem', display: 'inline-flex' }}>MIT License</span>
        </div>
      </div>
    </footer>
  )
}
