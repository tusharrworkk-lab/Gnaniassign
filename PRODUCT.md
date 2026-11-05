# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React frontend, selected by the user for later embedding inside a Flutter WebView. The initial implementation is a Vite-based single-page application served separately from the Python voice-agent API.

## Users

The primary user is the owner/operator of a personal AI voice agent. They check call activity, inspect conversation quality, review candidate MiniLM intent labels, and manage the operational state of the agent from a phone.

## Product Purpose

Give the operator a calm, fast, mobile-native control surface for understanding what the personal voice agent is doing and what requires human review.

## Positioning

The dashboard brings voice operations, conversation review, intent-learning review, and agent health into one phone-sized operating surface rather than forcing the owner to inspect logs, databases, and provider consoles separately.

## Operating Context

The surface is intended for a 9:16 phone viewport and eventual Flutter WebView embedding. Data shown initially is illustrative until the existing API gains dedicated dashboard read endpoints. The voice agent currently uses hosted Gnani STT/TTS and Bedrock LLM services, PostgreSQL for durable state, and optional Redis for prompt caching.

## Capabilities and Constraints

- Mobile-only first surface, designed at 390 × 844 logical pixels with 9:16 composition.
- React implementation; touch targets, safe-area padding, and no hover-dependent controls.
- The dashboard must distinguish live facts from candidate/AI-generated labels.
- Existing API endpoints are `/healthz`, `/v1/turn`, and normalized webhook endpoints; dashboard-specific read/write APIs are not yet implemented.
- VoiceLink media integration remains planned and must not be represented as already connected.

## Evidence on Hand

- Existing repository architecture, PostgreSQL schema, local cloud Compose stack, and architecture PDF.
- No established brand assets, production dashboard data endpoint, or approved visual system exist yet.

## Product Principles

- Make the agent's current state understandable in a few seconds.
- Keep review actions explicit and reversible.
- Use real system vocabulary: call, transcript, candidate label, final label, and rule.
- Preserve privacy: do not normalize exposing raw customer data as decoration.
- Treat the phone as the primary operating environment, not a shrunken desktop.

## Accessibility & Inclusion

- Touch targets must be at least 44 CSS pixels.
- Contrast must support normal text readability.
- Status must never rely on color alone.
- Motion must respect reduced-motion preferences.
