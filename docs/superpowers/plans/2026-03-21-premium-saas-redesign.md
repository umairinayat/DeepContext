# Premium SaaS Redesign — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the entire DeepContext website (Home, Docs, Demo, Dashboard) with a premium SaaS aesthetic, light/dark theme toggle, sidebar dashboard navigation, full-width responsive layouts, and modular CSS architecture.

**Architecture:** Replace the monolithic `index.css` with modular CSS files under `styles/`. Add a `ThemeProvider` context for light/dark mode. Convert the Dashboard from tab-based to sidebar + nested routes via React Router `<Outlet />`. All hardcoded colors become CSS variable references.

**Tech Stack:** React 19, React Router 7 (HashRouter), Vite 7, react-force-graph-2d/3d, Prism.js, pure CSS with custom properties (no framework).

**Spec:** `docs/superpowers/specs/2026-03-21-premium-saas-redesign-design.md`

---

## File Structure (New & Modified)

### New Files

```
website/src/
  styles/
    index.css              <- entry point, @imports all CSS files
    variables.css          <- light/dark theme tokens, accent, semantic colors
    reset.css              <- modern CSS reset
    typography.css         <- font imports, heading/body/code styles
    components.css         <- buttons, badges, pills, cards, inputs, selects
    layout.css             <- navbar, sidebar, footer, grid utilities
    animations.css         <- keyframes for gradient mesh, shimmer, transitions
    pages/
      home.css             <- hero, features bento grid, architecture, CTA
      dashboard.css        <- sidebar, header, stats, chat, graph, memories, lifecycle
      docs.css             <- two-column docs, TOC sidebar, mobile dropdown
      demo.css             <- demo page cards, terminal styling
  context/
    ThemeContext.jsx        <- React context for theme (light/dark) + provider
  components/
    ThemeToggle.jsx         <- sun/moon icon button
    Sidebar.jsx             <- dashboard sidebar nav + mobile bottom tab bar
    MobileMenu.jsx          <- full-screen overlay menu for navbar
    CopyButton.jsx          <- copy-to-clipboard button
    ProgressBar.jsx         <- colored horizontal bar for stats
    SlidePanel.jsx          <- right slide-in / bottom sheet for memory detail
    Skeleton.jsx            <- skeleton loader primitives (bar, card, row)
    ErrorBanner.jsx         <- inline error banner with retry
    EmptyState.jsx          <- centered empty state message with CTA
  pages/
    dashboard/
      StatsPage.jsx         <- stats wrapper page (uses StatsCards)
      ChatPage.jsx          <- chat wrapper page (uses ChatPanel)
      GraphPage.jsx         <- graph wrapper page (uses GraphViz)
      MemoriesPage.jsx      <- memories wrapper page (uses MemoryBrowser)
      LifecyclePage.jsx     <- lifecycle wrapper page (uses LifecycleControls)
```

### Modified Files

```
website/src/
  main.jsx                 <- import styles/index.css instead of index.css
  App.jsx                  <- add nested dashboard routes, wrap with ThemeProvider
  components/
    Navbar.jsx             <- theme toggle, mobile hamburger, scroll-aware bg
    Footer.jsx             <- three-column layout, responsive
    StatsCards.jsx          <- progress bars, new card styling, skeleton/empty states
    ChatPanel.jsx          <- card-based results, entity chips
    GraphViz.jsx           <- CSS variable colors, updated toolbar, entity color map
    MemoryBrowser.jsx      <- slide panel detail, cleaner rows, mobile sheet
    LifecycleControls.jsx  <- result cards with icons, "last run" timestamp
  pages/
    Home.jsx               <- full redesign: hero, bento features, architecture, CTA
    Dashboard.jsx          <- sidebar layout shell with <Outlet />, no tabs
    Docs.jsx               <- two-column with sticky TOC, mobile dropdown
    Demo.jsx               <- restyle with new card/button components

### Deleted Files

  index.css                <- replaced by styles/ directory
```

---

## Chunk 1: CSS Foundation & Theme System

### Task 1: Create CSS Variables File

**Files:**
- Create: `website/src/styles/variables.css`

- [ ] **Step 1: Create the variables.css file**

```css
/* website/src/styles/variables.css */

/* Light theme (default for system preference) */
[data-theme="light"] {
  --bg-primary: #FFFFFF;
  --bg-secondary: #F8F9FC;
  --bg-tertiary: #F1F3F9;
  --border: #E2E5F1;
  --text-primary: #0F1729;
  --text-secondary: #475467;
  --text-muted: #98A2B3;
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.04);
  --shadow-md: 0 1px 3px rgba(0,0,0,0.04), 0 6px 24px rgba(0,0,0,0.06);
  --glass-bg: rgba(255,255,255,0.6);
  --glass-border: rgba(255,255,255,0.2);
}

/* Dark theme */
[data-theme="dark"] {
  --bg-primary: #0A0B14;
  --bg-secondary: #12131E;
  --bg-tertiary: #1A1C2E;
  --border: #1F2137;
  --text-primary: #F1F3F9;
  --text-secondary: #A0A8C3;
  --text-muted: #5A6178;
  --shadow-sm: none;
  --shadow-md: none;
  --glass-bg: rgba(18, 19, 30, 0.8);
  --glass-border: rgba(31, 33, 55, 0.5);
}

/* Accent colors (shared both modes) */
:root {
  --accent: #6C5CE7;
  --accent-hover: #5A4BD6;
  --accent-dim: rgba(108, 92, 231, 0.1);
  --accent-glow: rgba(108, 92, 231, 0.2);
  --accent-gradient: linear-gradient(135deg, #6C5CE7, #4F8CFF);

  --green: #10B981;
  --green-dim: rgba(16, 185, 129, 0.1);
  --cyan: #06B6D4;
  --cyan-dim: rgba(6, 182, 212, 0.1);
  --orange: #F59E0B;
  --orange-dim: rgba(245, 158, 11, 0.1);
  --rose: #F43F5E;
  --rose-dim: rgba(244, 63, 94, 0.1);
  --purple: #A855F7;
  --purple-dim: rgba(168, 85, 247, 0.1);
  --pink: #EC4899;
  --pink-dim: rgba(236, 72, 153, 0.1);

  --radius: 12px;
  --radius-sm: 8px;
  --radius-lg: 16px;
  --radius-pill: 9999px;
  --transition: 150ms ease;
  --transition-reveal: 300ms ease-out;
}
```

- [ ] **Step 2: Verify the file was created**

Run: `ls website/src/styles/variables.css`

---

### Task 2: Create CSS Reset File

**Files:**
- Create: `website/src/styles/reset.css`

- [ ] **Step 1: Create reset.css**

```css
/* website/src/styles/reset.css */
*, *::before, *::after {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Theme transition on key containers */
body, .card, .sidebar, .navbar, .footer {
  transition: background-color 200ms ease, color 200ms ease, border-color 200ms ease;
}

img, video, svg {
  max-width: 100%;
  display: block;
}

a {
  color: var(--accent);
  text-decoration: none;
  transition: color var(--transition);
}

a:hover {
  color: var(--accent-hover);
}

/* Accessibility: focus indicators */
:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px var(--accent-dim);
}

/* Accessibility: reduced motion */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
  html { scroll-behavior: auto; }
}
```

- [ ] **Step 2: Commit**

```bash
git add website/src/styles/variables.css website/src/styles/reset.css
git commit -m "feat: add CSS variables (light/dark tokens) and reset"
```

---

### Task 3: Create Typography CSS

**Files:**
- Create: `website/src/styles/typography.css`

- [ ] **Step 1: Create typography.css**

```css
/* website/src/styles/typography.css */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

h1, h2, h3, h4, h5, h6 {
  color: var(--text-primary);
  line-height: 1.2;
  font-weight: 700;
}

h1 { font-size: 2.5rem; font-weight: 800; }
h2 { font-size: 2rem; font-weight: 700; }
h3 { font-size: 1.25rem; font-weight: 600; }
h4 { font-size: 1rem; font-weight: 600; }

p {
  color: var(--text-secondary);
  line-height: 1.6;
}

code, pre, kbd, samp, .mono {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

code {
  background: var(--bg-tertiary);
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
  font-size: 0.875rem;
  color: var(--accent);
}

pre {
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem;
  overflow-x: auto;
  font-size: 0.875rem;
  line-height: 1.5;
}

pre code {
  background: none;
  padding: 0;
  border-radius: 0;
  color: var(--text-primary);
}

.text-muted { color: var(--text-muted); }
.text-secondary { color: var(--text-secondary); }
.text-accent { color: var(--accent); }
.text-sm { font-size: 0.8125rem; }

/* Responsive typography */
@media (max-width: 768px) {
  h1 { font-size: 1.75rem; }
  h2 { font-size: 1.5rem; }
  h3 { font-size: 1.125rem; }
}
```

- [ ] **Step 2: Commit**

```bash
git add website/src/styles/typography.css
git commit -m "feat: add typography CSS with font imports and responsive sizes"
```

---

### Task 4: Create Components CSS (Buttons, Badges, Cards, Inputs)

**Files:**
- Create: `website/src/styles/components.css`

- [ ] **Step 1: Create components.css**

```css
/* website/src/styles/components.css */

/* ===== BUTTONS ===== */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.625rem 1.25rem;
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  font-family: inherit;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all var(--transition);
  text-decoration: none;
  white-space: nowrap;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background: var(--accent-gradient);
  color: #fff;
  border: none;
}
.btn-primary:hover:not(:disabled) {
  box-shadow: 0 4px 16px var(--accent-glow);
  transform: translateY(-1px);
}

.btn-secondary {
  background: var(--bg-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border);
}
.btn-secondary:hover:not(:disabled) {
  border-color: var(--accent);
}

.btn-outline {
  background: transparent;
  color: var(--text-secondary);
  border: 1px solid var(--border);
}
.btn-outline:hover:not(:disabled) {
  color: var(--accent);
  border-color: var(--accent);
}

.btn-ghost {
  background: transparent;
  color: var(--text-secondary);
  border: none;
}
.btn-ghost:hover:not(:disabled) {
  color: var(--accent);
  background: var(--accent-dim);
}

.btn-danger {
  background: var(--rose);
  color: #fff;
  border: none;
}
.btn-danger:hover:not(:disabled) {
  opacity: 0.9;
}

.btn-pill {
  border-radius: var(--radius-pill);
}

.btn-sm {
  padding: 0.375rem 0.875rem;
  font-size: 0.8125rem;
}

.btn-lg {
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
}

.btn-icon {
  padding: 0.5rem;
  border-radius: var(--radius-sm);
  line-height: 1;
}

/* White variant for CTA section */
.btn-white-outline {
  background: transparent;
  color: #fff;
  border: 1px solid rgba(255,255,255,0.3);
}
.btn-white-outline:hover {
  background: rgba(255,255,255,0.1);
  border-color: rgba(255,255,255,0.6);
}

/* ===== BADGES / PILLS ===== */
.badge {
  display: inline-flex;
  align-items: center;
  padding: 0.2rem 0.6rem;
  border-radius: var(--radius-pill);
  font-size: 0.75rem;
  font-weight: 500;
  white-space: nowrap;
}

.badge-accent { background: var(--accent-dim); color: var(--accent); }
.badge-green { background: var(--green-dim); color: var(--green); }
.badge-cyan { background: var(--cyan-dim); color: var(--cyan); }
.badge-orange { background: var(--orange-dim); color: var(--orange); }
.badge-rose { background: var(--rose-dim); color: var(--rose); }
.badge-purple { background: var(--purple-dim); color: var(--purple); }
.badge-pink { background: var(--pink-dim); color: var(--pink); }
.badge-muted { background: var(--bg-tertiary); color: var(--text-muted); }

/* Semantic badge aliases */
.badge-tier-working { background: var(--orange-dim); color: var(--orange); }
.badge-tier-short_term { background: var(--cyan-dim); color: var(--cyan); }
.badge-tier-long_term { background: var(--green-dim); color: var(--green); }
.badge-type-semantic { background: var(--accent-dim); color: var(--accent); }
.badge-type-episodic { background: var(--pink-dim); color: var(--pink); }
.badge-type-procedural { background: var(--purple-dim); color: var(--purple); }

/* ===== CARDS ===== */
.card {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem;
  box-shadow: var(--shadow-sm);
}

.card-interactive {
  cursor: pointer;
  transition: transform var(--transition), border-color var(--transition), box-shadow var(--transition);
}
.card-interactive:hover {
  transform: translateY(-2px);
  border-color: rgba(108, 92, 231, 0.3);
  box-shadow: var(--shadow-md);
}

.card-glass {
  background: var(--glass-bg);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border: 1px solid var(--glass-border);
}

/* ===== INPUTS ===== */
input[type="text"],
input[type="search"],
input[type="email"],
input[type="password"],
input[type="url"],
select,
textarea {
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 0.625rem 0.875rem;
  color: var(--text-primary);
  font-family: inherit;
  font-size: 0.875rem;
  transition: border-color var(--transition), box-shadow var(--transition);
  width: 100%;
}

input:focus,
select:focus,
textarea:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-dim);
}

input::placeholder,
textarea::placeholder {
  color: var(--text-muted);
}

textarea.mono {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.875rem;
  line-height: 1.5;
}

/* ===== SKELETON LOADERS ===== */
@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

.skeleton {
  background: linear-gradient(
    90deg,
    var(--bg-tertiary) 25%,
    var(--border) 50%,
    var(--bg-tertiary) 75%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s ease-in-out infinite;
  border-radius: var(--radius-sm);
}

.skeleton-bar {
  height: 1rem;
  width: 100%;
}
.skeleton-bar-sm { height: 0.75rem; width: 60%; }
.skeleton-bar-lg { height: 2rem; width: 40%; }
.skeleton-circle { border-radius: 50%; }
.skeleton-card {
  height: 120px;
  border-radius: var(--radius);
}

/* ===== ERROR BANNER ===== */
.error-banner {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  background: var(--rose-dim);
  border-left: 4px solid var(--rose);
  border-radius: var(--radius-sm);
  color: var(--rose);
  font-size: 0.875rem;
  margin-bottom: 1rem;
}

.error-banner .btn {
  margin-left: auto;
  flex-shrink: 0;
}

/* ===== EMPTY STATE ===== */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 3rem 1.5rem;
  text-align: center;
  gap: 0.75rem;
}

.empty-state p {
  color: var(--text-muted);
  max-width: 400px;
}

/* ===== PROGRESS BAR ===== */
.progress-bar {
  height: 8px;
  background: var(--bg-tertiary);
  border-radius: 4px;
  overflow: hidden;
  width: 100%;
}

.progress-bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.5s ease;
}
```

- [ ] **Step 2: Commit**

```bash
git add website/src/styles/components.css
git commit -m "feat: add component CSS (buttons, badges, cards, inputs, skeleton, errors)"
```

---

### Task 5: Create Layout CSS

**Files:**
- Create: `website/src/styles/layout.css`

- [ ] **Step 1: Create layout.css**

This file contains navbar, footer, sidebar, and utility layout classes. Write the full layout CSS covering:

- `.navbar` — sticky, blur-on-scroll, responsive (desktop center links + right actions, mobile hamburger)
- `.navbar-scrolled` — class added on scroll for blur bg + border
- `.mobile-menu-overlay` — full-screen slide-down menu for mobile
- `.footer` — three-column desktop, single column mobile
- `.sidebar` — fixed left, 240px, collapsible to 64px
- `.sidebar-collapsed` — 64px icon-only mode
- `.bottom-tab-bar` — mobile-only fixed bottom nav
- `.dashboard-layout` — flex container for sidebar + content
- `.dashboard-header` — sticky header inside content area
- `.section-full` — full-width section with `padding: 0 5%`
- Responsive media queries for all breakpoints (1024px, 768px, 480px)

- [ ] **Step 2: Commit**

```bash
git add website/src/styles/layout.css
git commit -m "feat: add layout CSS (navbar, footer, sidebar, dashboard layout)"
```

---

### Task 6: Create Animations CSS

**Files:**
- Create: `website/src/styles/animations.css`

- [ ] **Step 1: Create animations.css**

```css
/* website/src/styles/animations.css */

/* Hero gradient mesh — floating blobs */
@keyframes float-1 {
  0%, 100% { transform: translate(0, 0) scale(1); }
  33% { transform: translate(30px, -50px) scale(1.05); }
  66% { transform: translate(-20px, 20px) scale(0.95); }
}

@keyframes float-2 {
  0%, 100% { transform: translate(0, 0) scale(1); }
  33% { transform: translate(-40px, 30px) scale(1.1); }
  66% { transform: translate(25px, -40px) scale(0.9); }
}

@keyframes float-3 {
  0%, 100% { transform: translate(0, 0) scale(1); }
  33% { transform: translate(20px, 40px) scale(0.95); }
  66% { transform: translate(-35px, -25px) scale(1.05); }
}

.hero-blob {
  position: absolute;
  border-radius: 50%;
  filter: blur(80px);
  pointer-events: none;
}

.hero-blob-1 {
  width: 600px;
  height: 600px;
  background: rgba(108, 92, 231, 0.06);
  top: -10%;
  left: -5%;
  animation: float-1 20s ease-in-out infinite;
}

.hero-blob-2 {
  width: 500px;
  height: 500px;
  background: rgba(79, 140, 255, 0.05);
  top: 20%;
  right: -10%;
  animation: float-2 25s ease-in-out infinite;
}

.hero-blob-3 {
  width: 400px;
  height: 400px;
  background: rgba(108, 92, 231, 0.04);
  bottom: -5%;
  left: 30%;
  animation: float-3 15s ease-in-out infinite;
}

/* Gradient text */
.gradient-text {
  background: linear-gradient(135deg, #6C5CE7, #4F8CFF);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* Slide in from right (desktop slide panel) */
@keyframes slide-in-right {
  from { transform: translateX(100%); }
  to { transform: translateX(0); }
}

@keyframes slide-out-right {
  from { transform: translateX(0); }
  to { transform: translateX(100%); }
}

/* Slide up from bottom (mobile sheet) */
@keyframes slide-in-up {
  from { transform: translateY(100%); }
  to { transform: translateY(0); }
}

@keyframes slide-out-down {
  from { transform: translateY(0); }
  to { transform: translateY(100%); }
}

/* Fade in overlay backdrop */
@keyframes fade-in {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes fade-out {
  from { opacity: 1; }
  to { opacity: 0; }
}
```

- [ ] **Step 2: Commit**

```bash
git add website/src/styles/animations.css
git commit -m "feat: add animations CSS (hero blobs, slide panels, gradient text)"
```

---

### Task 7: Create Page-Specific CSS Files

**Files:**
- Create: `website/src/styles/pages/home.css`
- Create: `website/src/styles/pages/dashboard.css`
- Create: `website/src/styles/pages/docs.css`
- Create: `website/src/styles/pages/demo.css`

- [ ] **Step 1: Create home.css**

Covers: hero section (full viewport, centered, blob background), features bento grid (3-col asymmetric with `grid-template` from spec), architecture flow diagram (flexbox + pseudo connectors), CTA section (accent gradient bg).

Mobile: single-column stacks, reduced font sizes, full-width buttons.

- [ ] **Step 2: Create dashboard.css**

Covers: sidebar layout (240px fixed, collapsible to 64px), bottom tab bar (mobile), dashboard header (sticky, 56px), stats cards grid (3-col top, 2-col bottom), chat panel layout, graph container (fill remaining space), memory browser (list rows, search bar, filters), lifecycle page, slide panel overlay.

Mobile: bottom tab bar, stacked cards, full-width content.

- [ ] **Step 3: Create docs.css**

Covers: two-column layout (220px sticky sidebar + content), TOC sidebar (active section highlight, accent left border), mobile dropdown TOC (collapsible button + card), prose width constraint (720px), code block copy button position, table styling.

- [ ] **Step 4: Create demo.css**

Covers: demo card grid, input/output two-column layout, tab navigation, run button styling. Restyle existing demo components with new card/button classes.

- [ ] **Step 5: Commit**

```bash
git add website/src/styles/pages/
git commit -m "feat: add page-specific CSS (home, dashboard, docs, demo)"
```

---

### Task 8: Create CSS Entry Point & Wire Up

**Files:**
- Create: `website/src/styles/index.css`
- Modify: `website/src/main.jsx`

- [ ] **Step 1: Create styles/index.css entry point**

```css
/* website/src/styles/index.css — master entry point */
@import './variables.css';
@import './reset.css';
@import './typography.css';
@import './components.css';
@import './layout.css';
@import './animations.css';
@import './pages/home.css';
@import './pages/dashboard.css';
@import './pages/docs.css';
@import './pages/demo.css';
```

- [ ] **Step 2: Update main.jsx to import new CSS**

Change `import './index.css'` to `import './styles/index.css'`

```jsx
// website/src/main.jsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import { HashRouter } from 'react-router-dom'
import App from './App'
import './styles/index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </React.StrictMode>,
)
```

- [ ] **Step 3: Verify the dev server starts without CSS errors**

Run: `cd website && npm run dev`
Expected: Vite dev server starts, no CSS import errors in console.

- [ ] **Step 4: Delete old index.css**

Remove `website/src/index.css` (the old monolithic file).

- [ ] **Step 5: Commit**

```bash
git add website/src/styles/index.css website/src/main.jsx
git rm website/src/index.css
git commit -m "feat: wire up modular CSS, replace monolithic index.css"
```

---

### Task 9: Create Theme Context & ThemeToggle Component

**Files:**
- Create: `website/src/context/ThemeContext.jsx`
- Create: `website/src/components/ThemeToggle.jsx`
- Modify: `website/src/App.jsx`

- [ ] **Step 1: Create ThemeContext.jsx**

```jsx
// website/src/context/ThemeContext.jsx
import { createContext, useContext, useState, useEffect } from 'react'

const ThemeContext = createContext()

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('deepcontext-theme')
    if (saved) return saved
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('deepcontext-theme', theme)
  }, [theme])

  function toggleTheme() {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark')
  }

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const ctx = useContext(ThemeContext)
  if (!ctx) throw new Error('useTheme must be used within ThemeProvider')
  return ctx
}
```

- [ ] **Step 2: Create ThemeToggle.jsx**

```jsx
// website/src/components/ThemeToggle.jsx
import { useTheme } from '../context/ThemeContext'

export default function ThemeToggle({ className = '' }) {
  const { theme, toggleTheme } = useTheme()

  return (
    <button
      className={`btn btn-icon btn-ghost ${className}`}
      onClick={toggleTheme}
      aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
      title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
    >
      {theme === 'dark' ? (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="5"/>
          <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
          <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
        </svg>
      ) : (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
        </svg>
      )}
    </button>
  )
}
```

- [ ] **Step 3: Wrap App with ThemeProvider**

```jsx
// website/src/App.jsx
import { Routes, Route, Navigate } from 'react-router-dom'
import { ThemeProvider } from './context/ThemeContext'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Home from './pages/Home'
import Docs from './pages/Docs'
import Demo from './pages/Demo'
import Dashboard from './pages/Dashboard'
import StatsPage from './pages/dashboard/StatsPage'
import ChatPage from './pages/dashboard/ChatPage'
import GraphPage from './pages/dashboard/GraphPage'
import MemoriesPage from './pages/dashboard/MemoriesPage'
import LifecyclePage from './pages/dashboard/LifecyclePage'

export default function App() {
  return (
    <ThemeProvider>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/docs" element={<Docs />} />
        <Route path="/demo" element={<Demo />} />
        <Route path="/dashboard" element={<Dashboard />}>
          <Route index element={<Navigate to="stats" replace />} />
          <Route path="stats" element={<StatsPage />} />
          <Route path="chat" element={<ChatPage />} />
          <Route path="graph" element={<GraphPage />} />
          <Route path="memories" element={<MemoriesPage />} />
          <Route path="lifecycle" element={<LifecyclePage />} />
        </Route>
      </Routes>
      <Footer />
    </ThemeProvider>
  )
}
```

- [ ] **Step 4: Commit**

```bash
git add website/src/context/ThemeContext.jsx website/src/components/ThemeToggle.jsx website/src/App.jsx
git commit -m "feat: add ThemeProvider, ThemeToggle, and nested dashboard routes"
```

---

## Chunk 2: Global Components (Navbar, Footer, Utility Components)

### Task 10: Rewrite Navbar with Theme Toggle, Mobile Menu, Scroll-Aware Background

**Files:**
- Modify: `website/src/components/Navbar.jsx`
- Create: `website/src/components/MobileMenu.jsx`

- [ ] **Step 1: Create MobileMenu.jsx**

Full-screen overlay that slides down. Takes `isOpen` and `onClose` props. Renders nav links stacked vertically + "Get Started" full-width button. Closes on link click or backdrop click. Listens for Escape key.

- [ ] **Step 2: Rewrite Navbar.jsx**

Replace current Navbar with:
- Uses `useState` for `scrolled` state (toggled via scroll listener on `useEffect`)
- Uses `useState` for `mobileMenuOpen`
- Desktop: logo left, centered nav links with `NavLink`, right side has `ThemeToggle` + GitHub icon + "Get Started" pill
- Mobile: logo left, right side has `ThemeToggle` + hamburger button that opens `MobileMenu`
- `className` toggles `navbar-scrolled` when scrolled past 50px (adds blur bg + border)
- The navbar should NOT render on Dashboard pages (the Dashboard has its own header) — OR: render on all pages but Dashboard hides it via CSS. Decision: **keep navbar on all pages for consistency**; Dashboard content area scrolls below it.

- [ ] **Step 3: Verify navbar renders with theme toggle working**

Run: `cd website && npm run dev`
Open browser, click theme toggle — should switch between light and dark.

- [ ] **Step 4: Commit**

```bash
git add website/src/components/Navbar.jsx website/src/components/MobileMenu.jsx
git commit -m "feat: redesign Navbar with theme toggle, mobile menu, scroll blur"
```

---

### Task 11: Rewrite Footer

**Files:**
- Modify: `website/src/components/Footer.jsx`

- [ ] **Step 1: Rewrite Footer.jsx**

Three-column layout:
- Left: logo + "Hierarchical memory for AI agents" + `© 2026 DeepContext`
- Center: links (Docs, Demo, Dashboard, GitHub) using `NavLink` / `<a>`
- Right: "Built by umairinayat" + "MIT License" badge
- Uses `.footer` class from `layout.css`
- Mobile: single column, centered, stacked

- [ ] **Step 2: Commit**

```bash
git add website/src/components/Footer.jsx
git commit -m "feat: redesign Footer with three-column responsive layout"
```

---

### Task 12: Create Utility Components (CopyButton, ProgressBar, Skeleton, ErrorBanner, EmptyState)

**Files:**
- Create: `website/src/components/CopyButton.jsx`
- Create: `website/src/components/ProgressBar.jsx`
- Create: `website/src/components/Skeleton.jsx`
- Create: `website/src/components/ErrorBanner.jsx`
- Create: `website/src/components/EmptyState.jsx`

- [ ] **Step 1: Create CopyButton.jsx**

```jsx
// website/src/components/CopyButton.jsx
import { useState } from 'react'

export default function CopyButton({ text, className = '' }) {
  const [copied, setCopied] = useState(false)

  async function handleCopy() {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <button
      className={`btn btn-icon btn-ghost ${className}`}
      onClick={handleCopy}
      aria-label="Copy to clipboard"
      title={copied ? 'Copied!' : 'Copy'}
    >
      {copied ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--green)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
        </svg>
      )}
    </button>
  )
}
```

- [ ] **Step 2: Create ProgressBar.jsx**

```jsx
// website/src/components/ProgressBar.jsx
export default function ProgressBar({ value, max, color = 'var(--accent)', label, count }) {
  const pct = max > 0 ? (value / max) * 100 : 0
  return (
    <div className="progress-row">
      <div className="progress-label">
        <span>{label}</span>
        <span className="text-muted text-sm">{count}</span>
      </div>
      <div className="progress-bar">
        <div className="progress-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Create Skeleton.jsx**

```jsx
// website/src/components/Skeleton.jsx
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
```

- [ ] **Step 4: Create ErrorBanner.jsx**

```jsx
// website/src/components/ErrorBanner.jsx
export default function ErrorBanner({ message, onRetry }) {
  return (
    <div className="error-banner" role="alert">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
      </svg>
      <span>{message}</span>
      {onRetry && (
        <button className="btn btn-sm btn-outline" onClick={onRetry}>Retry</button>
      )}
    </div>
  )
}
```

- [ ] **Step 5: Create EmptyState.jsx**

```jsx
// website/src/components/EmptyState.jsx
import { Link } from 'react-router-dom'

export default function EmptyState({ message, actionLabel, actionTo, onAction }) {
  return (
    <div className="empty-state">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.5 }}>
        <circle cx="12" cy="12" r="10"/><path d="M8 15h8"/><circle cx="9" cy="9" r="1"/><circle cx="15" cy="9" r="1"/>
      </svg>
      <p>{message}</p>
      {actionTo && (
        <Link to={actionTo} className="btn btn-sm btn-primary btn-pill">{actionLabel}</Link>
      )}
      {onAction && !actionTo && (
        <button className="btn btn-sm btn-outline btn-pill" onClick={onAction}>{actionLabel}</button>
      )}
    </div>
  )
}
```

- [ ] **Step 6: Commit**

```bash
git add website/src/components/CopyButton.jsx website/src/components/ProgressBar.jsx website/src/components/Skeleton.jsx website/src/components/ErrorBanner.jsx website/src/components/EmptyState.jsx
git commit -m "feat: add utility components (CopyButton, ProgressBar, Skeleton, ErrorBanner, EmptyState)"
```

---

### Task 13: Create SlidePanel Component

**Files:**
- Create: `website/src/components/SlidePanel.jsx`

- [ ] **Step 1: Create SlidePanel.jsx**

Props: `isOpen`, `onClose`, `title`, `children`

Desktop (>= 768px): Fixed right panel, 320px wide, `transform: translateX` animation, backdrop overlay.
Mobile (< 768px): Fixed bottom sheet, `80vh` max-height, `transform: translateY` animation, backdrop overlay.

Close on: X button click, backdrop click, Escape key.
Focus trap: Tab cycles within panel content.
On close: returns focus to previously focused element.

Uses CSS classes `.slide-panel`, `.slide-panel-open`, `.slide-panel-backdrop`.
CSS for these goes in `dashboard.css`.

- [ ] **Step 2: Commit**

```bash
git add website/src/components/SlidePanel.jsx
git commit -m "feat: add SlidePanel component (desktop right panel, mobile bottom sheet)"
```

---

## Chunk 3: Dashboard Redesign (Sidebar + Nested Routes)

### Task 14: Create Sidebar Component

**Files:**
- Create: `website/src/components/Sidebar.jsx`

- [ ] **Step 1: Create Sidebar.jsx**

The sidebar renders differently based on viewport:
- Desktop (>= 768px): Fixed left sidebar, 240px, collapsible to 64px via chevron button
- Mobile (< 768px): Bottom tab bar, 56px, 5 icon tabs

Uses `NavLink` from react-router-dom for active state highlighting.
Collapse state saved to `localStorage` key `deepcontext-sidebar-collapsed`.
Includes `ThemeToggle` at bottom of sidebar (desktop) or hidden on mobile (already in navbar).

Nav items with icons (inline SVGs):
1. Stats — bar chart icon — links to `/dashboard/stats`
2. Chat Input — message icon — links to `/dashboard/chat`
3. Knowledge Graph — network icon — links to `/dashboard/graph`
4. Memories — brain icon — links to `/dashboard/memories`
5. Lifecycle — refresh icon — links to `/dashboard/lifecycle`

- [ ] **Step 2: Commit**

```bash
git add website/src/components/Sidebar.jsx
git commit -m "feat: add Sidebar component with collapsible desktop + mobile bottom tabs"
```

---

### Task 15: Rewrite Dashboard Page as Layout Shell

**Files:**
- Modify: `website/src/pages/Dashboard.jsx`

- [ ] **Step 1: Rewrite Dashboard.jsx**

Replace the tab-based layout with a sidebar layout shell:

```jsx
// website/src/pages/Dashboard.jsx
import { useState, useEffect } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import Sidebar from '../components/Sidebar'
import { healthCheck } from '../api/client'

const PAGE_TITLES = {
  stats: 'Stats',
  chat: 'Chat Input',
  graph: 'Knowledge Graph',
  memories: 'Memories',
  lifecycle: 'Lifecycle',
}

export default function Dashboard() {
  const [userId, setUserId] = useState('default_user')
  const [backendStatus, setBackendStatus] = useState('checking')
  const [refreshKey, setRefreshKey] = useState(0)
  const location = useLocation()

  const currentSection = location.pathname.split('/').pop() || 'stats'
  const pageTitle = PAGE_TITLES[currentSection] || 'Dashboard'

  useEffect(() => {
    healthCheck()
      .then(() => setBackendStatus('connected'))
      .catch(() => setBackendStatus('disconnected'))
  }, [])

  function handleRefresh() {
    setRefreshKey(k => k + 1)
  }

  return (
    <div className="dashboard-layout">
      <Sidebar />
      <div className="dashboard-main">
        <div className="dashboard-header">
          <h2 className="dashboard-title">{pageTitle}</h2>
          <div className="dashboard-header-right">
            <div className={`backend-status status-${backendStatus}`}>
              <span className="status-dot" />
              {backendStatus === 'connected' ? 'Connected' :
               backendStatus === 'disconnected' ? 'Disconnected' : 'Checking...'}
            </div>
            <input
              type="text"
              value={userId}
              onChange={e => setUserId(e.target.value)}
              placeholder="User ID"
              className="user-id-input"
            />
            <button className="btn btn-icon btn-ghost" onClick={handleRefresh} title="Refresh">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
              </svg>
            </button>
          </div>
        </div>

        {backendStatus === 'disconnected' && (
          <div className="error-banner" style={{ margin: '1rem' }}>
            Backend offline. Start it: <code>uvicorn deepcontext.api.server:app --reload</code>
          </div>
        )}

        <div className="dashboard-content" key={refreshKey}>
          <Outlet context={{ userId, refreshKey, backendStatus }} />
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add website/src/pages/Dashboard.jsx
git commit -m "feat: rewrite Dashboard as sidebar layout shell with Outlet"
```

---

### Task 16: Create Dashboard Sub-Page Wrappers

**Files:**
- Create: `website/src/pages/dashboard/StatsPage.jsx`
- Create: `website/src/pages/dashboard/ChatPage.jsx`
- Create: `website/src/pages/dashboard/GraphPage.jsx`
- Create: `website/src/pages/dashboard/MemoriesPage.jsx`
- Create: `website/src/pages/dashboard/LifecyclePage.jsx`

- [ ] **Step 1: Create all 5 sub-page wrappers**

Each wrapper reads `userId` and `refreshKey` from `useOutletContext()` and passes them to the existing component:

```jsx
// Example: website/src/pages/dashboard/StatsPage.jsx
import { useOutletContext } from 'react-router-dom'
import StatsCards from '../../components/StatsCards'

export default function StatsPage() {
  const { userId } = useOutletContext()
  return <StatsCards userId={userId} />
}
```

Same pattern for ChatPage (ChatPanel), GraphPage (GraphViz), MemoriesPage (MemoryBrowser), LifecyclePage (LifecycleControls).

- [ ] **Step 2: Verify dashboard routing works**

Run: `cd website && npm run dev`
Navigate to `/#/dashboard` — should redirect to `/#/dashboard/stats`.
Click sidebar items — URL changes, content switches.

- [ ] **Step 3: Commit**

```bash
git add website/src/pages/dashboard/
git commit -m "feat: add dashboard sub-page wrappers (Stats, Chat, Graph, Memories, Lifecycle)"
```

---

## Chunk 4: Landing Page Redesign

### Task 17: Rewrite Home Page

**Files:**
- Modify: `website/src/pages/Home.jsx`

- [ ] **Step 1: Rewrite Home.jsx**

Complete rewrite with these sections:

**Hero Section:**
- Full viewport height, centered content, gradient mesh background (3 `.hero-blob` divs)
- Accent badge pill "Hierarchical Memory for AI Agents"
- H1 with gradient text on "Context + Memory"
- Subtitle paragraph
- Two CTA buttons: "Read the Docs" (primary pill) + "Live Dashboard" (outline pill)
- Install pill: `pip install deepcontext` with `CopyButton`
- Quickstart code block in glassmorphic card with `CopyButton` and Prism.js highlighting

**Features Section (Bento Grid):**
- Section background `--bg-secondary`, full-width
- 6 cards in asymmetric CSS Grid (spec Section 3.2 grid template)
- Each card: inline SVG icon, H3 title, description, glassmorphic hover

**Architecture Section:**
- Flow diagram: 5 step cards connected with CSS pseudo-element arrows
- Horizontal on desktop, vertical on mobile

**CTA Section:**
- Accent gradient background, white text, install pill, two white outline buttons

- [ ] **Step 2: Verify landing page renders correctly in both themes**

Run dev server, check both light and dark modes. Check mobile viewport.

- [ ] **Step 3: Commit**

```bash
git add website/src/pages/Home.jsx
git commit -m "feat: redesign landing page (hero, bento features, architecture, CTA)"
```

---

## Chunk 5: Dashboard Component Restyling

### Task 18: Restyle StatsCards

**Files:**
- Modify: `website/src/components/StatsCards.jsx`

- [ ] **Step 1: Update StatsCards.jsx**

- Replace card styling with new `.card` class
- Add `ProgressBar` component for tier/type breakdowns
- Add `Skeleton` loading state
- Add `EmptyState` when no data
- Add `ErrorBanner` on fetch failure
- Large numbers use colored text: memories = accent, entities = green, relationships = purple
- Remove all hardcoded colors; use CSS variable classes

- [ ] **Step 2: Commit**

```bash
git add website/src/components/StatsCards.jsx
git commit -m "feat: restyle StatsCards with progress bars, skeleton, empty/error states"
```

---

### Task 19: Restyle ChatPanel

**Files:**
- Modify: `website/src/components/ChatPanel.jsx`

- [ ] **Step 1: Update ChatPanel.jsx**

- Textarea uses `.mono` class
- Extract button uses `.btn-primary .btn-pill`
- Results display:
  - Facts: card with bulleted list
  - Entities: color-coded badge pills using entity type color map
  - Relationships: styled inline flow `source --relation--> target (strength)`
- Add `ErrorBanner` on extraction failure
- Remove all hardcoded colors

- [ ] **Step 2: Commit**

```bash
git add website/src/components/ChatPanel.jsx
git commit -m "feat: restyle ChatPanel with card results, entity chips, error handling"
```

---

### Task 20: Restyle GraphViz

**Files:**
- Modify: `website/src/components/GraphViz.jsx`

- [ ] **Step 1: Update GraphViz.jsx**

- Replace hardcoded `TYPE_COLORS` with CSS variable-aware approach. Since react-force-graph needs hex values at runtime, use a JS map that reads from the spec's definitive entity color table:

```js
const TYPE_COLORS = {
  person: '#EC4899',
  organization: '#F59E0B',
  technology: '#6C5CE7',
  concept: '#06B6D4',
  location: '#10B981',
  event: '#A855F7',
  preference: '#F59E0B',
  other: '#5A6178',
}
```

- Update toolbar: pill toggle buttons (`.btn-pill .btn-sm`), fullscreen button
- Update fullscreen styles to use `--bg-primary` background
- Legend uses badge classes
- Add `Skeleton` loading state (centered spinner + text)
- Add `ErrorBanner` on fetch failure
- Add `EmptyState` when no graph data

- [ ] **Step 2: Commit**

```bash
git add website/src/components/GraphViz.jsx
git commit -m "feat: restyle GraphViz with updated colors, toolbar, loading/empty states"
```

---

### Task 21: Restyle MemoryBrowser with SlidePanel

**Files:**
- Modify: `website/src/components/MemoryBrowser.jsx`

- [ ] **Step 1: Update MemoryBrowser.jsx**

- Search input with search icon
- Filter dropdowns use new input styling
- Memory list: clean rows with subtle `.border-b` dividers, tier + type badge pills
- Click row opens `SlidePanel` with full memory detail
- SlidePanel shows: full text, all metadata, timestamps, Edit + Delete buttons
- Pagination: numbered pill buttons with accent active
- Add `Skeleton` loading state (5 skeleton rows)
- Add `EmptyState` for no results
- Add `ErrorBanner` on fetch failure
- Remove all hardcoded colors

- [ ] **Step 2: Commit**

```bash
git add website/src/components/MemoryBrowser.jsx
git commit -m "feat: restyle MemoryBrowser with SlidePanel detail, skeleton, badges"
```

---

### Task 22: Restyle LifecycleControls

**Files:**
- Modify: `website/src/components/LifecycleControls.jsx`

- [ ] **Step 1: Update LifecycleControls.jsx**

- Run button: `.btn-primary .btn-lg .btn-pill`
- Result cards: 3 in a row, each with icon above + large number + label
- Colors: decayed = orange, consolidated = cyan, cleaned = green
- Add "Last run: X ago" timestamp
- Add `ErrorBanner` on failure

- [ ] **Step 2: Commit**

```bash
git add website/src/components/LifecycleControls.jsx
git commit -m "feat: restyle LifecycleControls with result cards, icons, last-run timestamp"
```

---

## Chunk 6: Docs & Demo Page Restyling

### Task 23: Restyle Docs Page

**Files:**
- Modify: `website/src/pages/Docs.jsx`

- [ ] **Step 1: Update Docs.jsx**

- Two-column layout: sticky sidebar TOC (220px) + main content
- Sidebar: active section with accent left border + accent text
- Mobile: collapsible dropdown TOC (button at top, card with links on expand)
- Auto-close TOC on section click
- Prose constrained to 720px
- Code blocks use new styling + `CopyButton`
- Tables use clean borders from `--border`
- Heading anchor links on hover

- [ ] **Step 2: Commit**

```bash
git add website/src/pages/Docs.jsx
git commit -m "feat: restyle Docs page with sticky TOC, mobile dropdown, copy buttons"
```

---

### Task 24: Restyle Demo Page

**Files:**
- Modify: `website/src/pages/Demo.jsx`

- [ ] **Step 1: Update Demo.jsx**

- Restyle all demo cards with `.card` class
- Tab buttons use `.btn-pill .btn-sm` with active state
- Run buttons use `.btn-primary .btn-pill`
- Input textareas use `.mono` class
- Output code blocks use new styling
- Same functionality, just restyled

- [ ] **Step 2: Commit**

```bash
git add website/src/pages/Demo.jsx
git commit -m "feat: restyle Demo page with new card and button styles"
```

---

## Chunk 7: Final Polish & Verification

### Task 25: Update CodeBlock Component with CopyButton

**Files:**
- Modify: `website/src/components/CodeBlock.jsx`

- [ ] **Step 1: Add CopyButton to CodeBlock**

Import and render `CopyButton` in the top-right corner of every code block. Pass the raw code text as the `text` prop.

- [ ] **Step 2: Commit**

```bash
git add website/src/components/CodeBlock.jsx
git commit -m "feat: add CopyButton to CodeBlock component"
```

---

### Task 26: Responsive Testing & Fixes

- [ ] **Step 1: Test all pages at desktop (1440px) viewport**

Run dev server, check: Home, Docs, Demo, Dashboard (all sub-pages). Both themes.

- [ ] **Step 2: Test all pages at tablet (768px) viewport**

Check: navbar still shows links (not hamburger yet), sidebar collapses to icon-only, grids adjust.

- [ ] **Step 3: Test all pages at mobile (375px) viewport**

Check: navbar shows hamburger menu, dashboard shows bottom tab bar, all content stacks vertically, no horizontal overflow.

- [ ] **Step 4: Fix any responsive issues found**

- [ ] **Step 5: Commit fixes**

```bash
git add -A
git commit -m "fix: responsive design fixes across all breakpoints"
```

---

### Task 27: Build Verification

- [ ] **Step 1: Run production build**

Run: `cd website && npm run build`
Expected: Build succeeds with no errors.

- [ ] **Step 2: Preview production build**

Run: `cd website && npm run preview`
Expected: Site loads correctly, all pages work, theme toggle works.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: verify production build passes for redesigned site"
```

---

## Summary

| Chunk | Tasks | Description |
|-------|-------|-------------|
| 1 | Tasks 1-9 | CSS foundation (variables, reset, typography, components, layout, animations) + theme system |
| 2 | Tasks 10-13 | Global components (Navbar, Footer, utility components) |
| 3 | Tasks 14-16 | Dashboard sidebar + nested routes + sub-page wrappers |
| 4 | Task 17 | Landing page full redesign |
| 5 | Tasks 18-22 | Dashboard component restyling (Stats, Chat, Graph, Memories, Lifecycle) |
| 6 | Tasks 23-24 | Docs & Demo page restyling |
| 7 | Tasks 25-27 | Polish, responsive testing, build verification |

**Total: 27 tasks across 7 chunks.**

Each chunk produces a working, committable state. Chunks can be executed sequentially — each depends on the previous (CSS foundation must exist before components, components before pages).
