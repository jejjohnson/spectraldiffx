# Copilot Instructions

Read [`AGENTS.md`](../AGENTS.md) at the repository root first: it is the
single source of truth for every coding agent working here (the family /
module map, the boundaries with gaussx and finitevolX, "reuse before you
write", the contracts, the tests that enforce them, commands, the pre-commit
checklist, the docs and notebook rules, git and PR rules).

The essentials, in case you only read this file:

- One package, `spectraldiffx/` (flat layout), three independent families
  under `spectraldiffx/_src/`: `fourier/` (grids, DCT / DST transforms,
  Laplacian eigenvalues, derivatives, filters, elliptic solvers, the
  capacitance solver), `chebyshev/` and `spherical/`. The public API is
  `spectraldiffx.__all__`. Search
  [`docs/api/capabilities.md`](../docs/api/capabilities.md) before writing a
  helper.
- gaussx (upstream) owns the structured linear algebra; finitevolX
  (downstream) imports the solvers, eigenvalues and transforms by name, so
  renames go through a `DeprecationWarning`.
- Keep the contracts in `AGENTS.md`:
  - **grids**: per-family domain conventions (Fourier `[0, L)`, Chebyshev
    `[−L, L]` half-length, spherical Gauss–Legendre), axis order
    `(…, z, y, x)`, strict 2/3 mask;
  - **operators**: linear operators never dealias, nonlinear products do;
    complex physical input raises;
  - **transforms**: scipy conventions, `norm=None | "ortho"`, static
    `type` / `norm` / `axes`, dtype preserved;
  - **elliptic solvers**: `(∇² − λ)ψ = f`; the BC picks the transform
    (`_BC_DISPATCH`); FD2 vs spectral eigenvalues; one null-mode policy;
    resonance raises through `eqx.error_if`;
  - **JAX numerics**: no Python control flow on traced values, no dtype
    promotion (the transforms and the `solve_*` functions keep float32
    even with x64 on), new
    public operators join `CASES` in `tests/test_tracing.py`.
- Docstrings are numpy style, with plain-text (Unicode / ASCII) equations and
  array shapes. Notebooks are jupytext `.py` files in `notebooks/` (never
  `.ipynb`).
- Before committing, from the repo root: `make test-fast`,
  `uv run ruff check .`, `uv run ruff format --check .`, `make typecheck`;
  `make capabilities` after a public API change; `make docs` after a docs
  change.
- Step-by-step recipes (add an elliptic solver or BC, a transform, a
  derivative operator, a filter, a notebook; bump gaussx; pre-PR check;
  review; squash message) are plain Markdown in
  `.claude/skills/<name>/SKILL.md`, and the two review checklists
  (`reuse-reviewer`, `spectral-numerics-reviewer`) in `.claude/agents/`;
  follow them as written.
- Path-scoped standards live in `.github/instructions/`; code review follows
  [`CODE_REVIEW.md`](../CODE_REVIEW.md).

## Behavioral guidelines

- **Do not nitpick**: ignore what ruff and the formatter catch, and code you
  were not asked to change; match existing patterns.
- **Always propose tests**: a test that shows the expected behaviour (an
  analytic field, a dense matrix, `scipy.fft`), then the change.
- **Never suggest without a proposal**: "Add validation here", followed by
  the code.
- **Simplicity first, surgical changes**: no abstractions for single-use
  code, no speculative features; remove only what your change made unused.
