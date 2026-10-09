# CLAUDE.md

The rules for every agent live in `AGENTS.md`; this file adds only what is
specific to Claude Code.

@AGENTS.md

## Claude Code specifics

- **Reuse first.** Search `docs/api/capabilities.md` (every public
  spectraldiffx name, plus gaussx) before writing a helper, a transform, an
  eigenvalue formula or a solver; the "Reuse before you write" table in
  `AGENTS.md` maps the usual hand-rolled code to what already exists.
- **Run Python through `uv run`**, and run tests from the repo root
  (`tests/conftest.py` turns x64 on). Make targets take no paths; select
  tests with `uv run pytest tests/<file> -k <expr>`.
- **Skills** in `.claude/skills/` load on their own when a task matches
  their description (or run them as `/<name>`):
  - building: `add-elliptic-solver`, `add-transform`,
    `add-derivative-operator`, `add-spectral-filter`, `add-notebook`,
    `bump-upstream-pins`;
  - shipping: `pre-pr-check`, `spectraldiffx-review`, `squash-commit`.
- **Subagents** (`.claude/agents/`), both read-only, both used by
  `spectraldiffx-review`; run them on any diff that adds code, before
  committing:
  - `reuse-reviewer`: does the diff re-implement something in
    `docs/api/capabilities.md` (spectraldiffx, gaussx) or a shared private
    helper?
  - `spectral-numerics-reviewer`: BC ↔ transform ↔ eigenvalue pairing,
    normalisation, wavenumber scale and axis order, dealiasing, Nyquist,
    null modes and resonance, dtype / complex promotion, traced control
    flow.
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
