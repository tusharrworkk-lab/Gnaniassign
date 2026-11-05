# Mobile operations dashboard

Mode: Operate

Viewport: 390 × 844 CSS pixels, portrait 9:16. This is a mobile-web React surface intended for a future Flutter WebView.

Primary task: within a few seconds, the owner can tell whether the voice agent is live, whether a call needs attention, and whether nightly intent labels need review.

First viewport: a compact live-status header, the active-call path, today’s call state, and a review queue entry point. The signature interaction is the call path: caller → agent → resolution, with the active stage moving when a call is live.

Initial data: explicitly synthetic demonstration data in the frontend. The service API has no dedicated dashboard read endpoints yet; a client adapter keeps the eventual API replacement isolated.
