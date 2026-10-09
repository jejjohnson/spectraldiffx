---
applyTo: "**"
---

# Code Review Instructions

When performing code review, use `/CODE_REVIEW.md` as the source of truth for:

- Review checklist (reuse and boundaries, grids and conventions, operators
  and dealiasing, transforms, elliptic solvers, JAX numerics, public API and
  docs, tests, modern Python)
- spectraldiffx-specific checks (wavenumbers from the grid, truncated
  nonlinear products, BC ↔ transform pairing, dtype-preserving constants)
- Output format and priority levels
- Suggestion type emojis and review tone

Read "Boundaries", "Reuse before you write" and "The contracts" in
`/AGENTS.md` first.

Key principles:
- Sacrifice *cleverness* for *clarity*. Sacrifice *brevity* for *explicitness*.
- Don't worry about formatting — CI (ruff format, pre-commit) handles that automatically.
- Be **constructive** and **specific**. Acknowledge good patterns with 👍.
- Every suggestion must include a concrete alternative, not just criticism.
