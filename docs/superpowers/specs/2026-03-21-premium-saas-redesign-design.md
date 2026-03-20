# DeepContext Premium SaaS Redesign — Design Spec

> **Date:** 2026-03-21
> **Status:** Approved
> **Scope:** Full site redesign — Home, Docs, Demo, Dashboard
> **Direction:** Premium SaaS (Linear / Vercel / Stripe inspired)

---

## 1. Design System Foundation

### 1.1 Color Tokens

All colors are defined as CSS custom properties. Zero hardcoded colors in component CSS.

**Light Mode** (`[data-theme="light"]`):

| Token | Value | Use |
|-------|-------|-----|
| `--bg-primary` | `#FFFFFF` | Page background |
| `--bg-secondary` | `#F8F9FC` | Card/section backgrounds |
| `--bg-tertiary` | `#F1F3F9` | Inputs, hover states |
| `--border` | `#E2E5F1` | Borders, dividers |
| `--text-primary` | `#0F1729` | Headings |
| `--text-secondary` | `#475467` | Body text |
| `--text-muted` | `#98A2B3` | Placeholders, hints |

**Dark Mode** (`[data-theme="dark"]`):

| Token | Value | Use |
|-------|-------|-----|
| `--bg-primary` | `#0A0B14` | Page background |
| `--bg-secondary` | `#12131E` | Card/section backgrounds |
| `--bg-tertiary` | `#1A1C2E` | Inputs, hover states |
| `--border` | `#1F2137` | Borders, dividers |
| `--text-primary` | `#F1F3F9` | Headings |
| `--text-secondary` | `#A0A8C3` | Body text |
| `--text-muted` | `#5A6178` | Placeholders, hints |

**Accent (shared both modes):**

| Token | Value | Use |
|-------|-------|-----|
| `--accent` | `#6C5CE7` | Primary accent (blue-violet) |
| `--accent-hover` | `#5A4BD6` | Hover state |
| `--accent-dim` | `#6C5CE7` at 10% opacity | Subtle backgrounds |
| `--accent-gradient` | `#6C5CE7 -> #4F8CFF` | Brand gradient (hero, CTAs) |

**Semantic colors (shared both modes):**

| Token | Value | Use |
|-------|-------|-----|
| `--green` | `#10B981` | Success, long-term memory |
| `--cyan` | `#06B6D4` | Info, short-term memory |
| `--orange` | `#F59E0B` | Warning, working memory |
| `--rose` | `#F43F5E` | Error, danger |
| `--purple` | `#A855F7` | Procedural memory |
| `--pink` | `#EC4899` | Episodic memory |

Each semantic color also gets a `--{color}-dim` variant at 10% opacity for badge/pill backgrounds.

### 1.2 Typography

| Element | Font | Weight | Size (Desktop) | Size (Mobile) |
|---------|------|--------|----------------|---------------|
| Hero H1 | Inter | 800 | 4rem | 2.25rem |
| Section H2 | Inter | 700 | 2.5rem | 1.75rem |
| Card H3 | Inter | 600 | 1.25rem | 1.125rem |
| Body | Inter | 400-500 | 1rem | 1rem |
| Small/Label | Inter | 500 | 0.8125rem | 0.8125rem |
| Code/Data | JetBrains Mono | 400-500 | 0.875rem | 0.8125rem |

Line heights: headings 1.2, body 1.6, code 1.5.

### 1.3 Spacing & Layout

- Full-width sections: backgrounds span `100vw`, content uses `padding: 0 5%` (desktop) / `padding: 0 1.25rem` (mobile)
- No `max-width` container on page sections — full bleed
- Prose text (docs, descriptions) constrained to `720px` for readability
- Border radius: `12px` cards, `8px` buttons/inputs, `16px` large containers, `9999px` pills
- Shadows (light mode): `0 1px 3px rgba(0,0,0,0.04), 0 6px 24px rgba(0,0,0,0.06)`
- Shadows (dark mode): none — use border + subtle background elevation
- Transitions: `150ms ease` for interactions, `300ms ease-out` for reveals

### 1.4 Theme Toggle

- Sun/moon icon button in navbar
- Saves preference to `localStorage` key `deepcontext-theme`
- Defaults to system preference via `prefers-color-scheme` media query
- Theme applied via `data-theme` attribute on `<html>` element
- All colors reference CSS variables — theme switch is instant, no flash

### 1.5 Responsive Breakpoints

| Breakpoint | Width | Target |
|------------|-------|--------|
| Desktop | `>= 1024px` | Full layout, sidebar nav |
| Tablet | `768px - 1023px` | Collapsed sidebar, adjusted grids |
| Mobile | `< 768px` | Bottom tab bar, stacked layouts |
| Small mobile | `< 480px` | Tighter padding, smaller text |

---

## 2. Global Components

### 2.1 Navbar

**Desktop (>= 768px):**
- Full-width sticky bar, `height: 64px`
- Left: Logo icon + "DeepContext" wordmark (Inter 700)
- Center: Nav links (Home, Docs, Demo, Dashboard) with subtle underline hover animation
- Right: Theme toggle (sun/moon icon) + GitHub icon button + "Get Started" pill button (accent gradient)
- Background: transparent over hero, then `backdrop-filter: blur(16px)` + semi-transparent `--bg-primary` on scroll (detected via `IntersectionObserver` or scroll listener)
- Subtle `1px` bottom border appears on scroll

**Mobile (< 768px):**
- Full-width sticky bar, `height: 56px`
- Left: Logo icon + "DeepContext"
- Right: Theme toggle + hamburger menu icon
- Menu: Full-screen overlay (slide-down), links stacked vertically centered, "Get Started" as full-width button at bottom
- Menu overlay uses `--bg-primary` with high opacity

### 2.2 Footer

**Desktop:** Full-width, `--bg-secondary` background, three columns
- Left: Logo + one-line description ("Hierarchical memory for AI agents") + copyright
- Center: Links (Docs, Demo, Dashboard, GitHub)
- Right: "Built by umairinayat" + MIT license badge

**Mobile:** Single column, centered, stacked.

### 2.3 Button Variants

| Variant | Style |
|---------|-------|
| Primary | Accent gradient background, white text, subtle glow on hover |
| Secondary | `--bg-tertiary` background, `--text-primary`, border on hover |
| Outline | Transparent, `1px` border `--border`, accent border on hover |
| Ghost | Transparent, no border, accent text on hover |
| Pill | `border-radius: 9999px`, used for CTAs and tags |
| Danger | `--rose` background, white text |

All buttons: `8px` border radius (except pill), `150ms` transition, `font-weight: 500`, disabled state at 50% opacity.

### 2.4 Badge / Pill Components

Small inline elements for tier, type, entity type labels:
- Colored dim background + bright text matching semantic color
- `border-radius: 9999px`, `padding: 0.2rem 0.6rem`, `font-size: 0.75rem`, `font-weight: 500`
- Tier badges: working (orange), short_term (cyan), long_term (green)
- Type badges: semantic (accent), episodic (pink), procedural (purple)
- Entity badges: person (pink), organization (orange), technology (accent), concept (cyan), location (green), event (purple), preference (orange), other (muted)

### 2.5 Card Component

- Background: `--bg-secondary`
- Border: `1px solid var(--border)`
- Border radius: `12px`
- Padding: `1.5rem`
- Light mode: subtle shadow
- Dark mode: no shadow, border only
- Hover (interactive cards): `translateY(-2px)`, border color shifts to `--accent` at 30% opacity

### 2.6 Input / Textarea

- Background: `--bg-tertiary`
- Border: `1px solid var(--border)`
- Border radius: `8px`
- Padding: `0.625rem 0.875rem`
- Focus: border color `--accent`, subtle accent glow `0 0 0 3px var(--accent-dim)`
- Placeholder: `--text-muted`
- Textarea code input: `font-family: JetBrains Mono`

---

## 3. Landing Page (`/`)

### 3.1 Hero Section

- Full viewport height (`min-height: 100vh`), content vertically centered
- Background: 2-3 large radial gradients using accent color at 5% opacity, animated with CSS `@keyframes` (slow drift, 20s cycle) — pure CSS, no JS/canvas
- Content stacked center-aligned:
  1. Small accent badge pill: "Hierarchical Memory for AI Agents"
  2. H1: "Give Your AI Agents" (line 1) + "Context + Memory" (line 2, accent gradient text via `background-clip: text`)
  3. Subtitle: 1-2 sentences, `--text-secondary`, `1.25rem`
  4. Two CTA buttons: "Read the Docs" (primary) + "Live Dashboard" (outline)
  5. Install pill: `pip install deepcontext` with monospace font + copy-to-clipboard icon button
  6. Code block: `quickstart.py` in a glassmorphic card (light: semi-transparent white + blur, dark: `--bg-secondary` + subtle border glow)

**Mobile:** Same content, `min-height: 100vh`, H1 at `2.25rem`, buttons stack full-width, code block horizontally scrollable.

### 3.2 Features Section

- Full-width, `--bg-secondary` background
- Section header: centered, "Everything your agents need to remember", `2.5rem`
- **Bento grid layout** (asymmetric, not uniform 3-col):

```
Desktop grid (CSS Grid):
┌──────────────────────┬─────────────┐
│  Hybrid Retrieval    │  Knowledge  │
│  (spans 2 cols)      │  Graph      │
├────────────┬─────────┴─────────────┤
│  Lifecycle │  Fact Extraction      │
│  Mgmt      │  (spans 2 cols)      │
├────────────┴───────────┬───────────┤
│  Embedding Storage     │  Multi-   │
│  (spans 2 cols)        │  User     │
└────────────────────────┴───────────┘
```

- 6 feature cards total
- Each card: icon (small, colored), title (H3, weight 600), description (`--text-secondary`), optional mini code snippet or visual
- Card style: glassmorphic in light mode (semi-transparent white bg, `backdrop-filter: blur(8px)`), elevated `--bg-secondary` in dark mode
- Hover: `translateY(-2px)` + accent border glow

**Mobile:** Single column, all cards full-width, stacked.

### 3.3 Architecture Section

- Full-width, alternate background (`--bg-primary`)
- Vertical flow diagram built with CSS (flexbox + pseudo-element connectors):

```
Conversation -> Extraction -> Embedding -> Storage -> Retrieval
                     |                        ^
               Knowledge Graph ───────────────┘
```

- Each step: small card with icon + label, connected by lines/arrows
- Beside each step (desktop): brief 1-sentence explanation
- Built purely with CSS flexbox and `::before`/`::after` pseudo-elements for connectors

**Mobile:** Stacked vertically, diagram becomes a vertical flow (top to bottom), explanations below each step card.

### 3.4 CTA / Install Section

- Full-width accent gradient background (the brand `--accent-gradient`)
- Centered content on top:
  - "Start building with DeepContext" — white text, `2rem`, weight 700
  - `pip install deepcontext` — dark pill with copy button
  - "Read the Docs" + "View on GitHub" — white outline buttons
- Slight padding: `4rem 5%`

**Mobile:** Same content, buttons stack full-width.

---

## 4. Dashboard (`/dashboard`)

### 4.1 Layout — Sidebar Navigation

Replaces the current tabbed layout with a sidebar.

**Desktop (>= 768px):**
- Sidebar: fixed left, `240px` wide, full viewport height
- Collapsible to `64px` (icon-only mode) via toggle chevron button at bottom
- Sidebar collapse state saved to `localStorage`
- Nav items: icon + label text, vertical stack
- Active item: accent dim background pill, accent text
- Hover: subtle `--bg-tertiary` background
- Sections: Stats, Chat Input, Knowledge Graph, Memories, Lifecycle
- Bottom of sidebar: theme toggle + collapse/expand button
- Content area: `margin-left: 240px` (or `64px` collapsed), full remaining width, scrollable

**Tablet (768px - 1023px):**
- Sidebar auto-collapses to `64px` icon-only mode
- Can expand on hover/click

**Mobile (< 768px):**
- Sidebar hidden entirely
- Bottom tab bar: fixed at bottom, `56px` height, 5 icon tabs (no labels)
- Active tab: accent color icon
- Content area: full width, `padding-bottom: 56px` to clear tab bar

### 4.2 Dashboard Header

- Sticky at top of content area (not page top — below navbar)
- Left: Page title (dynamic, matches selected section)
- Right: Backend status indicator (green/red dot + "Connected"/"Disconnected"), User ID input (compact, `180px`), Refresh icon button
- Height: `56px`
- Background: `--bg-primary` with bottom border

**Mobile:** Compact single row, user ID input accessible via settings icon overlay.

### 4.3 Stats Page

**Top row:** 3 summary cards, CSS Grid `1fr 1fr 1fr`
- Each card: icon, label ("Total Memories"), large number (`2rem`, weight 800, colored), subtitle ("+12 this week" or similar)
- Colors: Memories = accent, Entities = green, Relationships = purple

**Bottom row:** 2 breakdown cards, CSS Grid `1fr 1fr`
- "By Tier" card: horizontal colored progress bars (working orange, short_term cyan, long_term green) with labels + counts
- "By Type" card: horizontal colored progress bars (semantic accent, episodic pink, procedural purple) with labels + counts
- Progress bar: `height: 8px`, `border-radius: 4px`, background `--bg-tertiary`, fill colored

**Mobile:** All cards single column, full width.

### 4.4 Chat Input Page

- Top section: textarea (monospace, min 12 rows, resizable), conversation ID input, Extract button (accent gradient)
- Format hint: small muted text "Supports JSON array or ROLE: content format"
- Bottom section (after extraction): results in 3 card groups:
  - **Facts:** Bulleted list in a card, each fact as a row
  - **Entities:** Color-coded pills/chips, each showing entity name + type badge
  - **Relationships:** Inline flow items: `source --relation--> target (strength: X)` styled with monospace arrows

**Mobile:** Full-width stacked, textarea fills width, results cards full-width.

### 4.5 Knowledge Graph Page

- Toolbar: horizontal bar with pill toggle buttons (2D/3D), fullscreen button, node/link count text
- Graph container: fills **all remaining viewport space** (`height: calc(100vh - header - toolbar - legend)`)
- Uses existing `react-force-graph-2d` / `react-force-graph-3d` components
- Node styling: colored circles by entity type (matching semantic colors), size proportional to `mention_count`
- Link styling: width proportional to `strength`, labeled on hover with `relation`, directional arrows
- Interactions: click node to expand 1-hop neighbors, double-click for 2-hop, drag to reposition, scroll to zoom
- Fullscreen: `position: fixed; inset: 0; z-index: 9999;` with `--bg-primary` background, Esc to exit
- Legend: horizontal row of colored circles + entity type labels below the graph

**Mobile:** Graph fills full width, `60vh` height. Toolbar wraps to two rows. Legend scrolls horizontally. Touch gestures for zoom/pan.

### 4.6 Memory Browser Page

- Search bar: full-width input with search icon, accent border on focus
- Filter row: two compact select dropdowns (Tier, Type) side by side
- Memory list: clean rows with subtle dividers (not heavy card borders per row)
  - Each row: truncated memory text, tier pill, type pill, relevance score
  - Hover: subtle background highlight
  - Click: detail panel slides in
- Detail panel:
  - Desktop: slides in from right as a `320px` side panel (overlay, not pushing content)
  - Mobile: slides up from bottom as a sheet overlay (`80vh` max height, draggable)
  - Shows: full text, all metadata (tier, type, importance, decay_factor, access_count), timestamps, Edit + Delete buttons
- Pagination: numbered pills at bottom, accent on active page, prev/next arrows

### 4.7 Lifecycle Page

- Simple clean layout
- Description text: "Run decay, consolidation, and cleanup for the current user."
- Run button: accent gradient, prominent
- Results (after run): 3 cards in a row
  - Each: subtle icon above, large centered number (weight 800, `2rem`), label below ("Decayed", "Consolidated", "Cleaned")
  - Colors: decayed = orange, consolidated = cyan, cleaned = green
- "Last run: X ago" timestamp below results

**Mobile:** Button full-width, result cards stack or stay in row (they're small enough).

---

## 5. Docs Page (`/docs`)

**Desktop:**
- Two-column layout: sticky sidebar TOC (`220px`) + main content area
- Sidebar: scrollable independently, active section highlighted with accent left border + accent text
- Content: prose constrained to `720px` max-width for readability
- Code blocks: can break wider than prose (up to content area width), `--bg-tertiary` background, Prism.js syntax highlighting, copy button top-right
- Tables: clean with `--border` dividers, no heavy backgrounds
- Headings: anchor links on hover (# icon)

**Mobile:** Sidebar becomes a collapsible dropdown TOC at the top of the page (tap to expand/collapse). Content full-width with `1.25rem` padding.

---

## 6. Demo Page (`/demo`)

Same existing simulated demo functionality, restyled with the new design system:
- Each operation (add, search, graph, lifecycle) in a premium card
- Input area with monospace textarea
- "Run" button with accent gradient
- Output displayed in a code-block-style card with syntax highlighting
- Cards use the standard card component from the design system

---

## 7. CSS Architecture

Replace the single `website/src/index.css` (1345 lines) with a modular structure:

```
website/src/styles/
  variables.css        <- all CSS custom properties (light + dark tokens)
  reset.css            <- modern CSS reset (box-sizing, margin, etc.)
  typography.css       <- @font-face/imports, heading/body/code styles
  components.css       <- buttons, badges, pills, cards, inputs, selects
  layout.css           <- navbar, sidebar, footer, grid utilities
  animations.css       <- keyframes for gradient mesh, transitions
  pages/
    home.css           <- landing page (hero, features, architecture, CTA)
    dashboard.css      <- sidebar layout, header, all dashboard sub-pages
    docs.css           <- two-column docs layout, TOC sidebar
    demo.css           <- demo page cards and terminal styling
```

**Entry point:** `website/src/styles/index.css` that `@import`s all files in order.

Import order matters:
1. `variables.css` (tokens first)
2. `reset.css`
3. `typography.css`
4. `components.css`
5. `layout.css`
6. `animations.css`
7. Page-specific CSS (loaded per route or all in bundle — small enough to bundle)

---

## 8. Component Changes

### New Components Needed

| Component | Purpose |
|-----------|---------|
| `ThemeToggle.jsx` | Sun/moon icon button, reads/writes `localStorage`, applies `data-theme` |
| `Sidebar.jsx` | Dashboard sidebar nav, collapsible, active state, mobile bottom-tab variant |
| `MobileMenu.jsx` | Full-screen overlay menu for navbar on mobile |
| `CopyButton.jsx` | Small copy-to-clipboard button for code blocks and install pill |
| `ProgressBar.jsx` | Colored horizontal bar for stats breakdowns |
| `SlidePanel.jsx` | Right slide-in panel (desktop) / bottom sheet (mobile) for memory detail |

### Modified Components

| Component | Changes |
|-----------|---------|
| `Navbar.jsx` | Add theme toggle, mobile hamburger menu, scroll-aware background |
| `Footer.jsx` | Three-column layout, responsive |
| `StatsCards.jsx` | Add progress bars for breakdowns, new card styling |
| `ChatPanel.jsx` | New card-based result layout, entity chips |
| `GraphViz.jsx` | Updated toolbar styling, fullscreen improvements (already partially done) |
| `MemoryBrowser.jsx` | Slide panel for detail, cleaner list rows, mobile sheet |
| `LifecycleControls.jsx` | Result cards with icons, "last run" timestamp |
| `Dashboard.jsx` | Replace tabbed layout with sidebar layout + routing |

### Removed Components

None — all existing components are kept but restyled.

---

## 9. Technical Notes

- **No CSS framework added.** Stays pure CSS with custom properties. The design quality comes from tokens, spacing discipline, and consistent components — not a framework.
- **Theme toggle** uses `useEffect` to check `localStorage` on mount, falls back to `prefers-color-scheme`. Sets `document.documentElement.dataset.theme`.
- **Sidebar state** (collapsed/expanded) saved to `localStorage` separately.
- **Mobile bottom tab bar** uses the same React Router `NavLink` as the sidebar, just rendered differently based on viewport.
- **Glassmorphic cards** in light mode use `background: rgba(255,255,255,0.6); backdrop-filter: blur(8px);` — only on the landing page feature cards, not dashboard (dashboard cards are solid for readability).
- **Animated gradient mesh** on hero: 2-3 absolutely positioned `div`s with large `border-radius: 50%`, accent color at 5% opacity, animated `transform: translate()` with different durations (15s, 20s, 25s). Pure CSS `@keyframes`.
- **No new npm dependencies** required. Everything is achievable with existing React + react-router + react-force-graph stack.
- **Code blocks** keep using Prism.js (already installed) with updated theme colors matching the new tokens.
