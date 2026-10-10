# Building with agents

spectraldiffx exists so that a grid owns its wavenumbers and dealiasing
mask, a derivative is a method on an operator built from that grid, and a
Poisson or Helmholtz solve picks the transform its boundary conditions
require. Coding agents tend to re-implement what they cannot see — a
`2π·fftfreq` array, an FFT derivative, a product left aliased, a DCT built
from an FFT, an FFT solve on a domain with walls — and each one loses
accuracy, boundary conditions or differentiability. spectraldiffx ships
three things that let them find it.

## The capability index

The [capability index](api/capabilities.md) lists every public name in
`spectraldiffx`, grouped as the API reference groups it, with a one-line
summary; then the public API of gaussx, which spectraldiffx builds on. It
is regenerated from the code and checked in the test suite, so it never
drifts.

## The Claude Code plugin

The repository is a Claude Code plugin marketplace. In any project:

```text
/plugin marketplace add jejjohnson/spectraldiffx
/plugin install spectraldiffx@spectraldiffx
```

The plugin adds:

- **`spectral-methods-with-spectraldiffx`** (skill) — loads whenever a task
  differentiates a field spectrally, solves an elliptic equation on a
  rectangle, basin or sphere, takes a DCT / DST or dealiases a nonlinear
  term: what lives where, the rules (wavenumbers from the grid, dealias
  products not linear operators, the boundary condition picks the
  transform, which eigenvalues a solver uses), a worked example, and a
  "don't write it — use spectraldiffx" table.
- **`spectraldiffx-reuse-reviewer`** (subagent) — a read-only check of a
  diff for pseudospectral code spectraldiffx already provides, and for
  misuse.

## llms.txt

For other agents and tools, the docs site serves
[`llms.txt`](https://jejjohnson.github.io/spectraldiffx/llms.txt): a
curated map of spectraldiffx and its key pages.

## Rules for your project's `AGENTS.md`

Paste this into the agent instructions of a project that builds on
spectraldiffx:

```markdown
## Spectral methods: build on spectraldiffx

This project uses spectraldiffx (pseudospectral grids, derivatives,
filters, DCT / DST transforms and Poisson / Helmholtz solvers in JAX).
Before writing wavenumbers, an FFT derivative, a dealiasing mask, a
DCT / DST or an elliptic solve, search the capability index
(https://jejjohnson.github.io/spectraldiffx/api/capabilities/) or
`spectraldiffx.__all__`, and compose what exists:

- Wavenumbers and the 2/3 mask come from the grid (`FourierGrid2D.k`,
  `.KX`, `.K2`, `.dealias_filter()`); fields are `(Ny, Nx)` with x last.
- Derivatives are methods of `SpectralDerivative2D` (and the Chebyshev and
  spherical operators); dealias products with `jacobian`,
  `advection_scalar` or `apply_dealias`, never linear operators.
- Solve `(∇² − λ)ψ = f` with `solve_helmholtz_2d` / `_3d`, passing the
  boundary condition of each axis (`"periodic"`, `"dirichlet"`,
  `"neumann"`, staggered or mixed); a masked basin with
  `build_capacitance_solver`. Never solve a walled domain with an FFT.
- Use `spectraldiffx.dct` / `dst` / `dctn` / … rather than scipy.fft in
  JAX code; keep BC, `type`, `norm` and `axes` arguments static under jit.
```

## Working on spectraldiffx itself

Contributors (and their agents) follow
[`AGENTS.md`](https://github.com/jejjohnson/spectraldiffx/blob/main/AGENTS.md)
in the repository: the module map, the boundaries with gaussx and
finitevolX, "reuse before you write", the contracts and the tests that
enforce them, and recipe skills for adding elliptic solvers, transforms,
derivative operators, filters and notebooks.
