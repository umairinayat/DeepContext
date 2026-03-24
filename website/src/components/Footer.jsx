import { NavLink } from 'react-router-dom'

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div className="footer-brand">
          <div className="footer-brand-name">DeepContext</div>
          <p>&copy; {new Date().getFullYear()} DeepContext</p>
        </div>

        <div className="footer-links">
          <a href="https://github.com/umairinayat/DeepContext" target="_blank" rel="noreferrer">GitHub</a>
          <NavLink to="/docs">Documentation</NavLink>
          <NavLink to="/dashboard">Dashboard</NavLink>
        </div>

        <div className="footer-right">
          <div className="footer-status-dot" />
          <span className="footer-status-text">All Systems Operational</span>
        </div>
      </div>
    </footer>
  )
}
