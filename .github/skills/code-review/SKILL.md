---
name: code-review
description: Review a change or pull request in spectraldiffx against CODE_REVIEW.md, the contracts in AGENTS.md (grids, operators and dealiasing, transforms, elliptic solvers, JAX numerics), the boundaries with gaussx and finitevolX, and reuse of existing primitives.
---

# Code review

Read "Boundaries", "Reuse before you write" and "The contracts" in
[`AGENTS.md`](../../../AGENTS.md); for every function, class or module the
diff adds, search [`docs/api/capabilities.md`](../../../docs/api/capabilities.md)
for an existing equivalent; then apply [`CODE_REVIEW.md`](../../../CODE_REVIEW.md)
and report in its format.

The full procedure, with the reuse and spectral-numerics checklists, is the
`spectraldiffx-review` recipe in
[`.claude/skills/spectraldiffx-review/SKILL.md`](../../../.claude/skills/spectraldiffx-review/SKILL.md)
and the two reviewers in [`.claude/agents/`](../../../.claude/agents/).
