# Design System Specification: Technical Precision & Editorial Depth

## 1. Overview & Creative North Star
**Creative North Star: The Sovereign Engine**
This design system moves away from the "chat-bubble" cliché of AI interfaces. Instead, it positions itself as a high-performance terminal for the mind—a "Sovereign Engine." It is built for technical mastery, where every pixel conveys data-dense utility through an editorial lens.

We break the "standard SaaS" look by rejecting the grid-line obsession. We use **intentional asymmetry** and **tonal layering** to create a sense of infinite digital space. By utilizing high-contrast typography (large, bold headers paired with tiny, precise mono labels), we create a "Pro-Tool" aesthetic that feels both authoritative and sophisticated.

---

## 2. Colors: Tonal Architecture
The palette is rooted in a deep charcoal foundation, using vibrant cyan and teal accents to highlight "intelligence" and "interactivity."

### The "No-Line" Rule
**Explicit Instruction:** Do not use 1px solid borders to section off the UI. Containers must be defined strictly through background color shifts. For example, a `surface-container-low` (#1a1b20) block sitting on a `surface` (#121317) background provides all the separation a professional eye needs.

### Surface Hierarchy & Nesting
Treat the UI as a series of physical layers. Each deeper level of data should use a progressively higher surface tier:
- **Base Layer:** `surface` (#121317)
- **Navigation/Sidebars:** `surface-container-low` (#1a1b20)
- **Primary Content Cards:** `surface-container` (#1f1f24)
- **Active/Hovered Elements:** `surface-container-high` (#292a2e)

### The Glass & Gradient Rule
To ensure the system feels premium:
- Use **Backdrop Blur** (12px–20px) on floating menus and modals using `surface-container-highest` at 80% opacity.
- **Signature Texture:** Primary CTAs should not be flat. Apply a subtle linear gradient from `primary` (#ffffff) to `primary-fixed-dim` (#3cdcd1) at a 45-degree angle to give the element a "machined" metallic finish.

---

## 3. Typography: The Editorial Scale
We pair the geometric precision of **Space Grotesk** for high-level branding with the Swiss-style clarity of **Inter** for functional data.

| Role | Font Family | Size | Intent |
| :--- | :--- | :--- | :--- |
| **Display-LG** | Space Grotesk | 3.5rem | Impactful technical headers. |
| **Headline-SM** | Space Grotesk | 1.5rem | Section titles. |
| **Body-MD** | Inter | 0.875rem | General readability and descriptions. |
| **Label-SM** | JetBrains Mono* | 0.6875rem | API keys, timestamps, and metadata. |

*\*Note: Use JetBrains Mono for all code-related and technical metadata to reinforce the "Engine" identity.*

---

## 4. Elevation & Depth: Tonal Layering
Traditional drop shadows are too "soft" for this system. We use **Tonal Layering** to convey importance.

- **The Layering Principle:** Place a `surface-container-lowest` (#0d0e12) element inside a `surface-container` (#1f1f24) to create a "recessed" well for code snippets.
- **Ambient Shadows:** For floating modals, use a shadow color derived from `surface-tint` (#3cdcd1) at 5% opacity with a 32px blur. It should feel like a soft glow from the engine, not a grey shadow.
- **The "Ghost Border" Fallback:** If accessibility requires a border, use `outline-variant` (#3c4948) at **15% opacity**. High-contrast white or grey borders are strictly forbidden.

---

## 5. Components: Functional Primitives

### Buttons
- **Primary:** `primary` (#ffffff) background, `on-primary` (#003734) text. 4px rounding (`DEFAULT`). No border.
- **Secondary:** Transparent background with a `Ghost Border` (15% opacity `outline`). 
- **State:** On hover, shift background to `primary-fixed-dim` (#3cdcd1).

### Cards & Lists
- **The Divider Rule:** Never use `<hr>` tags or border-bottoms. Use **Spacing Scale 4** (0.9rem) or a background shift to `surface-container-low` to separate items.
- **Interactive Lists:** Use `surface-container` for the item. On hover, transition to `surface-container-highest`.

### LLM Memory Nodes (Context Chips)
- Used for representing stored memory fragments.
- **Visuals:** `surface-container-high` background, `sm` (2px) rounding, with a 2px left-accent border of `secondary` (#7bd6d1).

### Code Blocks & API Inputs
- Background: `surface-container-lowest` (#0d0e12).
- Font: `JetBrains Mono`.
- Edge: Sharp (`none`) or `sm` (2px) rounding to emphasize the "Technical Pro" feel.

---

## 6. Do's and Don'ts

### Do
- **DO** use asymmetry. Large empty spaces on one side of the screen balanced by dense data on the other create an editorial, high-end feel.
- **DO** use `secondary` (#7bd6d1) for data visualization and success states.
- **DO** utilize `spacing-24` (5.5rem) for hero section padding to let the "Engine" breathe.

### Don't
- **DON'T** use 100% opaque borders. They clutter the UI and make it look like a template.
- **DON'T** use large corner radii. Stick to `DEFAULT` (4px) or `sm` (2px). Rounded "pills" contradict the technical nature of the system.
- **DON'T** use pure black (#000000). Always use the specified `surface` (#121317) to maintain tonal depth and reduce eye strain.