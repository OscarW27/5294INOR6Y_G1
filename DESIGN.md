# Design system — RunTracker analytics

Light-theme, data-first editorial aesthetic for the Endomondo running and heart-rate
analytics mockup. The source of truth is [site/css/tokens.css](site/css/tokens.css);
the living implementation lives at [site/design.html](site/design.html) (`/design` on the
deployed site).

## Visual rules

1. **Light theme only.** No dark mode, no glassmorphism, no blurred gradient blobs,
   no neon accents, no drop shadows heavier than a none-default flat surface.
2. **Messages in ink, chrome from borders.** Surfaces are white or near-white; dark
   ink text; hierarchy is carried by 1px borders, spacing and type — not shadows or fills.
3. **Tabular numerals everywhere numbers are read, compared or scanned**
   (`font-variant-numeric: tabular-nums`): BPM, pace (min/km), distance, elevation,
   times, percentages. See the `.tnum` utility and `.metric` component.
4. **Generous whitespace.** Cards breathe; sections are separated by at least one
   large spacing step; page content is capped at a readable measure.
5. **Token discipline.** No raw hex colors or arbitrary pixel sizes inside page
   templates or components. All values come from semantic tokens, which in turn
   reference primitives. The only allowed raw values are inside `tokens.css`.

## Spacing scale (4px base unit)

| Token | Value | Use |
|---|---|---|
| `--space-1` | 4px | Icon gaps, inline nudges |
| `--space-2` | 8px | Inside compact chips, list gaps |
| `--space-3` | 12px | Control padding, small card padding |
| `--space-4` | 16px | Standard card padding, grid gutter |
| `--space-5` | 24px | Feature-block padding, group separation |
| `--space-6` | 32px | Section separation |
| `--space-7` | 48px | Page-level rhythm |
| `--space-8` | 64px | Page top/bottom margins |

## Typography hierarchy

| Level | Token | Notes |
|---|---|---|
| Display | `--font-size-display` | Oversized editorial titles (use sparingly) |
| H1 | `--font-size-title` | Page titles |
| H2 | `--font-size-heading` | Section / card titles |
| H3 | `--font-size-subheading` | Sub-blocks |
| Body | `--font-size-body` | Paragraphs, list items |
| Metric | `--font-size-metric` | Large KPI numerals, tabular |
| Meta / label | `--font-size-meta` | Metadata labels, table headers (uppercase + letterspacing) |

Families: `--font-sans` (UI + reading), `--font-mono` (telemetry, code).

## Color usage (semantic layer)

Every color used in a template maps to one of these semantic groups:

- **Surfaces** — `--surface-page`, `--surface-card`, `--surface-subtle`, `--surface-inset`
- **Text hierarchy** — `--text-primary`, `--text-secondary`, `--text-muted`, `--text-inverted`
- **Borders** — `--border-subtle`, `--border-default`, `--border-strong`
- **Primary actions** — `--action-bg`, `--action-bg-hover`, `--action-fg`,
  `--action-soft-bg`, `--action-soft-text`, `--accent` / `--accent-soft`
  (persona-recolourable: Beginner=green, Advanced=blue, Coach=orange, Medical=red)
- **Running intensity zones** — `--zone-recovery-*` (low/green), `--zone-steady-*`
  (moderate/amber), `--zone-intensive-*` (high/red). The same trio drives status
  bar fills (`good` / `warn` / `bad`).

## Component inventory

All components live in `site/css/components.css` and are demonstrated on
[site/design.html](site/design.html):

- **Buttons** — `.btn` base, `.btn--primary`, `.btn--ghost`
- **Navigation bar** — `.site-header`, `.site-nav`, `.site-nav a[aria-current="page"]`
- **Persona / mode tabs** — `.persona-tabs` (Beginner / Expert toggle, segmented control)
- **Role cards** — `.role-card` (login persona chooser)
- **KPI stat cards** — `.kpi-card`, `.kpi-stat`, `.kpi-card--placeholder`
- **Intensity badges** — `.badge-intensity` with `--recovery`, `--steady`, `--intensive`
- **Flag badge** — `.badge`
- **Zone bars** — `.zonebar` with `.z-low` / `.z-mod` / `.z-high` segments
- **Bar chart rows** — `.bar-row`, `.bar-track`, `.bar-fill` (+ `.good/.warn/.bad/.mute`)
- **Chart container frame** — `.chart-frame` (1px bordered panel for any plot/SVG)
- **Data table rows** — `.data-table` / `.data-row`, `.run-list`
- **Forms** — text inputs, `.field-label`
- **Footer** — `.site-foot`

## Accessibility

- Focus states use `--border-focus` at 1px outline; keyboard reachable controls.
