import { Link } from "react-router-dom";

const BASE = import.meta.env.BASE_URL;

export default function Footer() {
  const year = new Date().getFullYear();

  return (
    <footer className="footer">
      <div className="footer-inner">
        {/* Brand */}
        <div className="footer-brand">
          <Link to="/" className="footer-logo">
            <img
              src={`${BASE}logo.png`}
              alt="DeepContext"
              style={{ height: "22px", width: "auto", borderRadius: "4px" }}
            />
            <span>DeepContext</span>
          </Link>
          <p>
            Hierarchical memory for AI agents — graph-aware hybrid retrieval,
            lifecycle management, and a full REST API. Open source under MIT.
          </p>
        </div>

        {/* Product */}
        <div className="footer-col">
          <h4>Product</h4>
          <Link to="/">Home</Link>
          <Link to="/docs">Documentation</Link>
          <Link to="/demo">Live Demo</Link>
          <Link to="/dashboard">Dashboard</Link>
        </div>

        {/* Docs */}
        <div className="footer-col">
          <h4>Docs</h4>
          <Link to="/docs#installation">Installation</Link>
          <Link to="/docs#quickstart">Quick Start</Link>
          <Link to="/docs#api">REST API</Link>
          <Link to="/docs#retrieval">Hybrid Retrieval</Link>
          <Link to="/docs#lifecycle">Memory Lifecycle</Link>
        </div>

        {/* Links */}
        <div className="footer-col">
          <h4>Links</h4>
          <a
            href="https://github.com/umairinayat/DeepContext"
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
          <a
            href="https://github.com/umairinayat/DeepContext/issues"
            target="_blank"
            rel="noreferrer"
          >
            Issues
          </a>
          <a
            href="https://github.com/umairinayat/DeepContext/blob/main/CHANGELOG.md"
            target="_blank"
            rel="noreferrer"
          >
            Changelog
          </a>
          <a
            href="https://pypi.org/project/deepcontext/"
            target="_blank"
            rel="noreferrer"
          >
            PyPI
          </a>
        </div>
      </div>

      {/* Bottom bar */}
      <div className="footer-bottom">
        <span>© {year} DeepContext · MIT License</span>
        <span>
          Built by{" "}
          <a
            href="https://github.com/umairinayat"
            target="_blank"
            rel="noreferrer"
          >
            @umairinayat
          </a>
        </span>
        <span>
          <a
            href="https://github.com/umairinayat/DeepContext"
            target="_blank"
            rel="noreferrer"
          >
            Star on GitHub ★
          </a>
        </span>
      </div>
    </footer>
  );
}
