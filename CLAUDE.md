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
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
