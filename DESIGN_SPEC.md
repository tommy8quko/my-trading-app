# Trading Journal — Design Specification
> Copy this file into your Claude Code project and reference it in every prompt.

---

## Fonts
```
Space Grotesk — UI labels, headings, body text
Space Mono   — numeric values, tickers, data cells
```
Google Fonts import:
```html
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
```

---

## Color Tokens
```css
--bg-base:        #090c10;   /* page background */
--bg-surface:     #0f1318;   /* card / panel background */
--bg-elevated:    #141a24;   /* inputs, dropdowns */
--border:         rgba(255,255,255,0.06);  /* all dividers and card borders */
--border-focus:   rgba(68,153,255,0.5);   /* focused input ring */

--text-primary:   #c8d0e0;   /* main text */
--text-secondary: #5a6478;   /* labels, de-emphasized */
--text-dim:       #2e3648;   /* very muted / disabled */

--accent:         #4499ff;   /* blue — primary accent, links, active nav */
--gain:           #3ecf8e;   /* positive P&L */
--loss:           #f05b6b;   /* negative P&L */
```

---

## Typography Scale
```
Nav labels:        10px  Space Grotesk 500  --text-secondary  letter-spacing: 0.08em uppercase
KPI label:         10px  Space Grotesk 500  --text-secondary  letter-spacing: 0.08em uppercase
KPI value:         20px  Space Mono 700     --text-primary
Table header:      10px  Space Grotesk 500  --text-secondary  letter-spacing: 0.08em uppercase
Table ticker:      13px  Space Mono 700     --text-primary
Table data cell:   12px  Space Mono 400     --text-secondary
Section title:     11px  Space Grotesk 600  --text-secondary  letter-spacing: 0.12em uppercase
```

---

## Spacing & Radii
```
Panel padding:    16px
Cell padding:     10px 16px
Border radius:    6px  (cards, cells, buttons)
Border radius:    4px  (inputs, small chips)
Sidebar width:    200px (expanded), 48px (collapsed)
```

---

## Layout — Dashboard (Command Center)
```
┌──────────────────────────────────────────────────────┐
│ TOPBAR  [TJ logo] [Page title] [date · open · closed]│
├──────────────────────────────────────────────────────┤
│ SIDEBAR (48px collapsed)    │  MAIN CONTENT          │
│  [icon] Dashboard           │                        │
│  [icon] Journal             │  ┌──── EQUITY CURVE ──┐│
│  [icon] Analysis            │  │  blue line + blue  ││
│  [icon] Data                │  │  gradient fill      ││
│                             │  └────────────────────┘│
│  ── KPI COLUMN ──           │                        │
│  [HKD] [USD] toggle         │  ┌─ OPEN POSITIONS ───┐│
│                             │  │  table              ││
│  TOTAL P&L                  │  └────────────────────┘│
│  -75,287                    │                        │
│  WIN RATE                   │                        │
│  19.4%                      │                        │
│  MAX DRAWDOWN               │                        │
│  75,287                     │                        │
│  PROFIT FACTOR              │                        │
│  0.30                       │                        │
│  AVG WIN                    │                        │
│  +5,362                     │                        │
│  AVG LOSS                   │                        │
│  -4,298                     │                        │
│  TRADES                     │                        │
│  31                         │                        │
└─────────────────────────────┴────────────────────────┘
```

---

## Equity Curve Chart
- Background: `#0f1318`
- Line color: `#4499ff` (2px stroke)
- Fill: gradient from `rgba(68,153,255,0.25)` at top to `rgba(68,153,255,0)` at bottom
- Zero-line: dashed `rgba(255,255,255,0.15)`
- Grid lines: `rgba(255,255,255,0.04)`
- Axis labels: 10px Space Mono, `#5a6478`
- Dot at last point: 6px filled `#4499ff`
- Time filters: `1M 3M 6M ALL` pills top-right, active = `#4499ff` text

---

## Tables
```css
/* header row */
th {
  font: 500 10px/1 'Space Grotesk';
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #5a6478;
  padding: 8px 12px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* data rows */
td {
  font: 400 12px/1 'Space Mono';
  color: #5a6478;
  padding: 10px 12px;
  border-bottom: 1px solid rgba(255,255,255,0.04);
}

/* ticker column — bold, primary color */
td.ticker {
  font: 700 13px/1 'Space Mono';
  color: #c8d0e0;
}

/* P&L colors */
td.gain { color: #3ecf8e; }
td.loss { color: #f05b6b; }

/* hover row */
tr:hover td { background: rgba(255,255,255,0.02); }
```

---

## Sidebar & Nav
```
Background:     #090c10 with right border rgba(255,255,255,0.06)
Topbar height:  40px, background #090c10, bottom border rgba(255,255,255,0.06)

Nav item (default):  icon + label, color #5a6478
Nav item (active):   background rgba(68,153,255,0.1), left accent bar 2px #4499ff, color #c8d0e0
Nav item hover:      background rgba(255,255,255,0.04)
```

---

## No Direction Badges
Do NOT render "LONG" / "SHORT" as colored pill badges.  
Instead just show the text as a plain uppercase label in `--text-secondary`.

---

## Prompt Template for Claude Code
Use this as a prefix for every styling request:

```
Reference DESIGN_SPEC.md for all visual decisions. 
Use Space Grotesk + Space Mono fonts (Google Fonts).
Background #090c10, surfaces #0f1318, accent #4499ff.
Gains #3ecf8e, losses #f05b6b, text #c8d0e0, labels #5a6478.
The equity curve must have a blue (#4499ff) line with a blue gradient fill underneath.
No neon green. No direction pill badges. No pure black backgrounds.
Match the typography scale and spacing exactly from the spec.
```
