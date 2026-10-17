# Changelog

## 0.9.8 — 2026-10-05

Stable reference release used by AlitaYoko Conversations.

### Changed

- Expanded deterministic unsupported-recommendation detection to cover additional stability/safety/best-option phrasing.
- Neutralizes claims such as “for stability version X would still be safe” when the requested recommendation is not supported by the available evidence.
- Neutralizes exclusive recommendation inference such as “the only confirmed option” when evidence establishes support/compatibility but not a recommendation.

### Tests

- Added regression coverage for both recommendation-inference cases.
- 175 automated tests pass.

## 0.9.7

- Added deterministic exact-entry deduplication before semantic validation.
- Continued the existing Research, grounding and Final Audit pipeline.
