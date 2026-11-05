---
name: Relay mobile operations dashboard
description: A light, code-led mobile control surface for a personal AI voice agent.
colors:
  canvas: "#eee7dc"
  surface: "#f8f5ee"
  active-field: "#edf5ee"
  surface-sheet: "#fffaf3"
  control-surface: "#fffdf8"
  rule: "#ddd6ca"
  text: "#26372b"
  text-muted: "#536357"
  action-green: "#b9dfc6"
  status-green: "#4f8c67"
  terracotta: "#bd7541"
  terracotta-ink: "#934a20"
typography:
  greeting:
    fontFamily: "Aptos, Segoe UI Variable, ui-sans-serif, system-ui, -apple-system, sans-serif"
    fontSize: "27px"
    fontWeight: 700
    lineHeight: 1.04
    letterSpacing: "-0.052em"
  call-title:
    fontFamily: "Aptos, Segoe UI Variable, ui-sans-serif, system-ui, -apple-system, sans-serif"
    fontSize: "22px"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-0.047em"
  body:
    fontFamily: "Aptos, Segoe UI Variable, ui-sans-serif, system-ui, -apple-system, sans-serif"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.5
  section-title:
    fontFamily: "Aptos, Segoe UI Variable, ui-sans-serif, system-ui, -apple-system, sans-serif"
    fontSize: "20px"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-0.035em"
  label:
    fontFamily: "Aptos, Segoe UI Variable, ui-sans-serif, system-ui, -apple-system, sans-serif"
    fontSize: "10px"
    fontWeight: 750
    lineHeight: 1.3
    letterSpacing: "0.11em"
rounded:
  control: "8px"
  button: "10px"
  icon: "12px"
  agent-mark: "12px"
  capsule: "999px"
spacing:
  page: "18px"
  compact: "8px"
  control: "12px"
  section: "24px"
components:
  button-primary:
    backgroundColor: "{colors.action-green}"
    textColor: "#244836"
    rounded: "{rounded.button}"
    height: "48px"
  icon-button:
    backgroundColor: "{colors.control-surface}"
    textColor: "#365542"
    rounded: "{rounded.icon}"
    width: "44px"
    height: "44px"
---

# Design System: Relay mobile operations dashboard

## Overview

**Creative North Star: "Soft daylight."**

This is a calm personal-organizer surface for a personal voice agent: warm, light, and evidence-led rather than decorative. Rice-paper surfaces, eucalyptus state signals, and muted terracotta review markers make the interface comfortable for all-day phone use while preserving direct routes to calls, MiniLM candidate-label review, prompt setup, and live runtime activity.

The authored experience is a 390 × 844 CSS-pixel portrait app. A wider browser shows the same fixed-height phone preview centered in a warm rice-paper field; it must never become a desktop dashboard. This React surface is intended for later Flutter WebView embedding, so safe-area insets, touch-sized controls, and web-standard interaction remain first-class constraints.

## Colors

The palette is warm paper and pale eucalyptus, with green reserved for live/healthy state and muted terracotta reserved for review attention.

- **Rice-paper canvas** (`#eee7dc`) and **app surface** (`#f8f5ee`): the warm, low-contrast field that frames the phone and its content.
- **Pale eucalyptus active field** (`#edf5ee`): the full-bleed live-call band; it marks current activity without alarm.
- **Cream control and sheet surfaces** (`#fffdf8`, `#fffaf3`): quiet lift for controls and the review sheet.
- **Warm rule** (`#ddd6ca`): 1px dividers and quiet boundaries; avoid heavy card outlines.
- **Deep green ink** (`#26372b`) and **muted green-gray** (`#536357`): accessible default and supporting reading hierarchy.
- **Action green** (`#b9dfc6`) / **status green** (`#4f8c67`): confirmed action, live route, active navigation, and healthy state.
- **Muted terracotta** (`#bd7541`) / **terracotta ink** (`#934a20`): labels and counts that need review.

**The evidence rule.** Status color reinforces an explicit textual state and dot/icon; it never carries the meaning alone.

## Typography

Use `Aptos`, `Segoe UI Variable`, and the system sans fallback stack. The type system is compact but deliberately emphatic in the first viewport: the greeting is 27px with a 1.04 line height and constrained to 250px for a two-line welcome; the live-call title is 22px; supporting section titles are 20px; metric values are 25px tabular numerals; body copy is 12–13px; labels/meta remain 10–11px.

Small labels use 750 weight with measured tracking (`0.11em` for section labels). Stable figures, times, confidence, and counts use tabular numerals. Section and page headings use tight `-0.035em` tracking.

## Layout

The shell is 390px wide with 18px horizontal padding and safe-area-aware top and bottom padding. The phone frame fills the small viewport; at 520px and above it is centered, fixed to 844px high, lightly bordered, and visibly identified as a mobile-only preview. Within the shell, the content pane scrolls while the four-item bottom navigation remains anchored; operational lists never push navigation out of reach.

The opening scan order is: two-line greeting and demo disclosure, expanded Relay identity, **Call in motion**, today metrics, then the titled human-review priority, **One decision needs you**. Major content sections separate at 27px after the emphasis pass; full-bleed live-call content deliberately reaches past the page gutter. Persistent four-item bottom navigation reserves the lower 72px plus the safe-area inset.

## Elevation & Depth

Depth is mostly tonal. Cream controls and the review sheet lift from rice-paper surfaces while pale eucalyptus identifies active work; warm 1px rules organize lists and metrics. Shadows are restrained and structural only: the agent strip uses `0 9px 19px rgba(91,76,52,.08)`, the desktop phone preview uses `0 24px 65px rgba(78,62,40,.18)`, and the review sheet lifts with `0 -16px 38px rgba(78,62,40,.18)`.

## Shapes

The system is square-leaning with small, purposeful rounding: 8px selects, a 12px expanded agent mark, 10px primary action, and 12px icon controls. Route stops and status dots are the intentional circular exceptions. Borders are 1px; do not add glass, gradients, or pill-heavy card decoration.

## Components

### Navigation and controls

- Icon buttons and call controls are 44 × 44px minimum; all primary touch targets meet the 44px target floor.
- Bottom navigation has four equal targets. The active item is green and carries a 24 × 2px top indicator; review count uses a terracotta badge.
- The primary review action is a full-width 48px pale-green button. Quiet and text actions stay unfilled so the operator can distinguish primary commitment from navigation.
- Visible keyboard focus is a 3px dark-green outline with a 3px offset.

### Operational panels

- The expanded Relay strip uses a cream-raised rectangle, a 40px pale-green identity mark, 15px Relay name, concise “Your on-call assistant” copy, and a stacked “Ready on device” status treatment. It is a first-screen identity signal, not a passive metadata row.
- The live-call module is full bleed within the shell. Its signature operational route is **Caller → Relay → Reminder**, using numbered circular stops and 1px lines; exactly one signal dot moves between Caller and Relay while the demo call is live.
- Relay is the active handoff: its pale-green stop carries a restrained `0 0 0 5px rgba(79,140,103,.12)` halo, not a glow effect. The final Reminder stop remains dashed to communicate a prepared outcome.
- An outcome strip follows the transcript and route: **Next action**, the scheduled time, and the explicit state **Reminder draft ready**, with a quiet green check. It confirms the operational result without competing with the live-call controls.
- **Call in motion** is the first-screen operational focal point: its 22px title, 34px route stops, enlarged caller detail, transcript, and outcome strip must carry more weight than the surrounding summary sections.
- Metrics are a three-column ruled row, not cards. Values are 25px tabular numerals with a compact 10px label beneath.
- Human review is a named priority, not a generic queue: the title **One decision needs you** leads into a single candidate row whose candidate status and confidence stay explicit. The review sheet labels the MiniLM suggestion as candidate-only before allowing a saved correction; review state is reversible with Undo.

### Setup

Setup is a notebook of decisions, not another dashboard. Its 25px **Agent setup** title pairs one-sentence intent with an explicit Published, Stored · off, or Preview state. A two-option segmented control switches between **Instructions** and **MCP prep**; the selected cream surface is the only elevated segment.

- **Instructions** begins with a 40px pale-green instruction-stack mark and a compact Rules → Prompt → Call assembly route. Ordered rules use numbered circular markers, concise condition/priority metadata, and a 44px inspect action. “Add instruction” opens an inline editor; publishing language must state whether the change is local preview or applies to new calls.
- **MCP prep** uses a quiet terracotta-tinted readiness mark to distinguish connection preparation from live state. It documents read-only pre-call sources, the remote server’s credential boundary, and per-tool 44px toggles. Treat “browser never receives credentials” and “approved read-only tools” as visible safety information, not buried implementation notes.

### Live activity

Live activity is intentionally a notebook-like event stream, not a metric wall. The heading pairs a short explanation with a Live feed, Connecting, or Preview status. Event rows are ruled, compact, and timestamped: a 7px typed dot, a 12px event title, an 11px detail, and tabular time communicate sequence without cards or charts.

- Use quiet typed dots for transcript, classifier, agent, call, and prompt events; the classifier alone may use a restrained green halo. Color supports the event type but does not replace the written title.
- When no stream is available, show a specific waiting or empty state that tells the operator how events will appear. Keep **Recent calls** below a divider as a secondary reference list rather than mixing it into the runtime feed.

### Motion and accessibility

The Caller-to-Relay signal loops over 2.4 seconds; the Relay halo and Reminder outcome strip remain static. Sheet entry (`.22s`) and backdrop fade (`.16s`) are short state transitions. Under `prefers-reduced-motion`, animations effectively resolve immediately. The review sheet is a modal dialog with Escape close and tab focus containment.

## Do's and Don'ts

### Do:

- **Do** keep the surface explicitly mobile-only at 390 × 844 and account for Flutter WebView safe areas.
- **Do** disclose synthetic/demo data until dedicated dashboard endpoints replace the client adapter.
- **Do** pair every live, planned, connected, or review status with text as well as color.
- **Do** reserve green for active/healthy/confirmed signals and muted terracotta for review work.
- **Do** make publishing, preview, live-feed, and connection states explicit in Setup and Live activity.
- **Do** keep long operational content in the scrolling pane while bottom navigation stays available.

### Don't:

- **Don't** present VoiceLink as connected; it remains planned.
- **Don't** imply candidate MiniLM labels are final labels or that review automatically retrains the model.
- **Don't** turn the widened browser preview into a responsive desktop dashboard.
- **Don't** turn Setup into a metric dashboard or Live activity into a decorative chart/analytics wall.
- **Don't** introduce hover-dependent behavior, emoji icons, decorative charts, gradients, or glass surfaces.
