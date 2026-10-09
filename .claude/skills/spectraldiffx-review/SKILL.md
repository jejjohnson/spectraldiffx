---
name: spectraldiffx-review
description: Review a change or pull request in spectraldiffx against the repo's own rules — CODE_REVIEW.md, the grid / operator / transform / elliptic-solver / JAX-numerics contracts in AGENTS.md, the boundaries with gaussx and finitevolX, reuse of existing primitives, and spectral numerical correctness. Use when asked to review a diff, branch or PR in this repo.
---

# Review a spectraldiffx change

1. **Get the diff** as `CODE_REVIEW.md` describes, or from the PR. Note
   which families (fourier, chebyshev, spherical) and which layers (grid,
   transforms / eigenvalues, operators, filters, solvers, capacitance) it
   touches.
2. **Reuse** — run the `reuse-reviewer` subagent on the diff: hand-written
   wavenumbers, 2/3 masks, DCT / DST via FFT tricks, eigenvalue formulas,
   BC dispatch, divide-by-`k²` solves and dense linear algebra are the main
   way the package drifts from itself and from gaussx.
3. **Spectral numerics** — run the `spectral-numerics-reviewer` subagent in
   parallel with step 2: BC ↔ transform ↔ eigenvalue pairing, grid
   placement, transform normalisation, wavenumber scale and order,
   dealiasing of products (and not of linear operators), Nyquist handling,
   null mode and resonance, dtype / complex promotion, traced control flow.
4. **Boundaries** — gaussx through its public API only; no import across
   families; a renamed, removed or re-signatured public name has a
   `DeprecationWarning` alias (finitevolX imports the solvers, eigenvalues,
   transforms and capacitance solver by name).
5. **Contracts** — for each layer touched, the rules in `AGENTS.md`'s
   contracts: domain conventions and axis order; linear vs nonlinear
   dealiasing; scipy conventions and static `type` / `norm` / `axes`;
   `(∇² − λ)ψ = f`, `_BC_DISPATCH`, the null-mode policy, `_check_finite`;
   a `CASES` entry for a new operator or solver; the `docs/api` entry and
   `docs/api/capabilities.md`.
6. **Checklist** — the rest of `CODE_REVIEW.md` (numpy-style docstrings with
   plain-text equations and shapes, tests against closed forms / scipy /
   dense matrices with commented tolerances, tier markers), skipping what
   ruff, ty or the tests already enforce.
7. **Verify claims** — for anything you flag as a bug, run it: the
   operator on a single resolved mode against its closed form, the solver
   against `tests/test_solvers_dense.py`'s dense matrix, the transform
   against `scipy.fft`, a float32 input, `jax.jit` / `jax.grad` around the
   call, a non-square grid with different lengths per axis.

Report in the format `CODE_REVIEW.md` gives (overview, suggestions with
priority, file, lines and a concrete change, summary).
