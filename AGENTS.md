# AGENTS.md

Standing instructions for **every** coding agent working in this repository
(Claude Code, Copilot, Codex, Gemini, …). This is the single source of truth:
`CLAUDE.md` and `.github/copilot-instructions.md` point here.

## What this repo is

spectraldiffx is pseudospectral differentiation, filtering, spectral
transforms and elliptic (Poisson / Helmholtz) solvers in JAX, on Fourier,
Chebyshev and spherical-harmonic bases, in 1-D, 2-D and 3-D. Everything is
an Equinox module or a pure function, so it runs under `jit`, `vmap` and
`grad`. finitevolX builds on it (it re-exports the elliptic solvers, the
eigenvalue helpers, the DCT / DST transforms and the capacitance solver), so
**spectraldiffx is a library of primitives**: new code composes what is
here, and anything genuinely new lands where the next person will find it.

It is one package, `spectraldiffx/` at the repo root (flat layout, no
`src/`). Everything under `spectraldiffx/_src/` is private; the public API is
`spectraldiffx.__all__`, re-exported from `spectraldiffx/__init__.py` (each
family's `_src/<family>/__init__.py` re-exports its modules first). Three
independent families, each built bottom-up from its grid:

| Family | Path | Grids | Transforms and primitives | Operators and filters | Solvers |
|---|---|---|---|---|---|
| Fourier | `_src/fourier/` | `FourierGrid1D/2D/3D` (`grid.py`): wavenumbers, 2/3 mask, FFT | `transforms.py`: `dct` / `dst` / `idct` / `idst` types I–IV and the n-D `dctn` … `idstn`; `eigenvalues.py`: 1-D Laplacian eigenvalues, FD2 (`*_eigenvalues`) and pseudo-spectral (`*_eigenvalues_ps`) | `operators.py`: `SpectralDerivative1D/2D/3D`; `filters.py`: `SpectralFilter1D/2D/3D` | `solvers.py`: the `solve_helmholtz_*` / `solve_poisson_*` functions (layer 0) and the solver classes (layer 1), per-axis BCs (`BoundaryCondition`), inhomogeneous BCs (`modify_rhs_*`); `capacitance.py`: `build_capacitance_solver` → `CapacitanceSolver` for masked domains, on gaussx |
| Chebyshev | `_src/chebyshev/` | `ChebyshevGrid1D/2D/3D` (`grid.py`): nodes, differentiation matrices, DCT-based transform | `transforms.py`: `ChebyshevTransform1D/2D`, coefficient calculus, `dealias_product` (public as `cheb_dealias_product`); `quadrature.py`: Clenshaw–Curtis | `operators.py`: `ChebyshevDerivative1D/2D/3D`; `filters.py`: `ChebyshevFilter1D/2D` | `solvers.py`: `ChebyshevHelmholtzSolver1D/2D`, `ChebyshevPoissonSolver1D/2D`, on gaussx |
| Spherical | `_src/spherical/` | `SphericalGrid1D/2D` (`grid.py`): Gauss–Legendre colatitude × uniform longitude | `harmonics.py`: `SphericalHarmonicTransform` | `operators.py`: `SphericalDerivative1D/2D`; `filters.py`: `SphericalFilter1D/2D` | `solvers.py`: Poisson, Helmholtz, vorticity / divergence inversion, Helmholtz decomposition |

Inside a family, imports point one way: the grid at the bottom, then
transforms / eigenvalues, then operators, filters and solvers. The families
never import each other. The Fourier solvers module calls its free
functions "Layer 0" and its `eqx.Module` wrappers "Layer 1"; keep that split
when adding a solver.

### Boundaries

- **Upstream: gaussx** (pinned to a release tag in `[tool.uv.sources]` plus
  a `>=` floor in `dependencies`). It owns the structured linear algebra:
  the capacitance solver is a `gaussx.MaskedOperator` over a
  `gaussx.DiagonalisedOperator` / `circulant_from_symbol`, and the Chebyshev
  Helmholtz solvers diagonalise once with `gaussx.EigenFactorization` and
  solve 2-D systems with `gaussx.kronecker_sum_solve`. Use gaussx's public
  API only; gaussx never imports spectraldiffx.
- **Downstream: finitevolX** pins a spectraldiffx release tag and imports
  the elliptic solvers, eigenvalue functions, transforms, `BoundaryCondition`
  and the capacitance solver by name. A rename, a removed name or a changed
  signature breaks it: keep the old name working with a `DeprecationWarning`
  that names the replacement for at least one release, and say in the PR
  that finitevolX must follow. (The solvers module already binds two names
  to one function where both are in use, e.g. `solve_helmholtz_dst1 =
  solve_helmholtz_dst`.)
- **Numerics are JAX.** NumPy and SciPy run only at construction time, on
  concrete values (Chebyshev differentiation matrices, Gauss–Legendre nodes
  and associated Legendre functions via `scipy.special`, capacitance cell
  classification); SciPy is otherwise a test reference (`scipy.fft`).
  `import spectraldiffx` loads gaussx, nothing heavier.

## What spectraldiffx is built on

Each foundation brings a rule. Breaking one runs fine on one example and
fails under a transform, in float32, or in someone else's pipeline.

| Library | spectraldiffx uses it for | The rule it brings |
|---|---|---|
| **jax** | `jnp.fft`, `jit`, `grad`, `vmap`, x64 | Pure functions; no Python control flow on traced values (validate only concrete values, raise at run time with `eqx.error_if`); don't promote the input's dtype (the transforms and the Fourier `solve_*` functions return float32 for float32 input even with x64 on); physical-space input is real. |
| **equinox** | Every grid, operator, filter, transform class and solver is an `eqx.Module`; `__check_init__`; `eqx.error_if` | Never a dataclass or a plain class for anything that holds arrays (a dataclass is not a pytree). Settings that pick a code path are `eqx.field(static=True)` (`method`, `bc_x`, `base_bc` and the capacitance `shape` are; older grid fields such as `dealias` and `node_type` are not yet, #102); validation goes in `__check_init__`. |
| **gaussx** | Capacitance correction, masked and diagonalised operators, eigen-factorised shifted solves, Kronecker-sum solves | A solve, factorisation, Woodbury or capacitance correction comes from gaussx, not a hand-written `jnp.linalg` call; the dense `jnp.linalg.solve` fallbacks in the Chebyshev solvers (traced grids only) are the documented exception. |
| **jaxtyping** | Shape annotations (`Float[Array, "Ny Nx"]`) | Annotate every public array argument and return; shape strings stay quoted (ruff's `UP037` is off for that reason). |
| **numpy / scipy** | Construction-time precomputation; `scipy.fft` as the test reference for the transforms | Never on a traced value; keep it out of `__call__` / `solve` (gh-98: the spherical GFD solvers still rebuild their scipy tables inside `solve` and cannot be jitted). |

## Reuse before you write

Before writing a helper, a transform, an eigenvalue formula or a solver,
find out whether it exists:

1. **Search the capability index.**
   [`docs/api/capabilities.md`](docs/api/capabilities.md) lists every public
   spectraldiffx name, grouped as the API reference groups it, with a
   one-line summary; then the public API of gaussx. It is generated
   (`make capabilities`) and checked in the fast tier, so it is current.
2. **Search the private helpers** in the table below: several invariants
   (real input, strict 2/3 mask, resonance check, BC dispatch) live in one
   place each.
3. **If it is missing, add it in the family that owns it**, at the lowest
   layer every caller can reach (a transform in `transforms.py`, an
   eigenvalue formula in `eigenvalues.py`, a per-axis BC in `_BC_DISPATCH`),
   not inline in the one caller you have today; structured linear algebra
   with no spectral content goes to gaussx.
4. **One object, one name.** The deliberate exceptions are the
   backwards-compatible aliases (`solve_helmholtz_dst1` = `solve_helmholtz_dst`,
   `solve_helmholtz_dct2` = `solve_helmholtz_dct`, and their Poisson twins)
   and `CapacitanceSolver`, which gaussx also exports for its generic
   correction (`ALLOWED_SHARED_NAMES` in `scripts/capabilities.py` gives the
   reason).

| You are about to write… | Use instead |
|---|---|
| `2 * jnp.pi * jnp.fft.fftfreq(N, dx)`, a `k²` array, a wavenumber meshgrid | `FourierGrid*.k`, `kx` / `ky` / `kz`, `KX`, `K2` |
| A 2/3 dealiasing mask | `grid.dealias_filter()` (`_src/fourier/grid.py::_two_thirds_mask`, strict `3\|n\| < N`), `SpectralDerivative*.apply_dealias`; Chebyshev `cheb_dealias_product`; spherical `SphericalGrid*.dealias_filter()` |
| `ifft(1j * k * fft(u))`, a Laplacian, curl, divergence, Jacobian or advection term | `SpectralDerivative1D/2D/3D` (`gradient`, `laplacian`, `divergence`, `curl`, `jacobian`, `advection_scalar`, `velocity_from_streamfunction`, `project_vector`, `inverse_laplacian`, `biharmonic`, `hyperviscosity`); `ChebyshevDerivative*`, `SphericalDerivative*` |
| A DCT / DST through FFT tricks, or `scipy.fft.dct` in library code | `dct` / `dst` / `idct` / `idst` (types I–IV, `norm=None` or `"ortho"`), `dctn` / `dstn` / `idctn` / `idstn` |
| `-4 / dx**2 * sin(...)**2` or `-(pi * k / L)**2` | `dst1_eigenvalues` … `fft_eigenvalues` (FD2), `*_eigenvalues_ps` (pseudo-spectral) |
| A Poisson / Helmholtz solve on a rectangle (divide by `-k²`) | `solve_helmholtz_2d` / `_3d` with per-axis `bc_x` / `bc_y` / `bc_z`, the `solve_*_fft` / `_dst*` / `_dct*` functions, `SpectralHelmholtzSolver*`, `MixedBCHelmholtzSolver2D/3D` |
| Ghost-point corrections for non-zero boundary values | `bc_x_values=` / `bc_y_values=` on `solve_helmholtz_2d` / `_3d`, `modify_rhs_1d/2d/3d` |
| A Poisson solve on a basin with a land mask | `build_capacitance_solver(mask, dx, dy, lambda_, base_bc)` → `CapacitanceSolver` |
| A Chebyshev differentiation matrix, nodes or coefficients | `ChebyshevGrid*.D` / `Dx` / `Dx2`, `.x`, `.transform`; `ChebyshevTransform1D/2D`; `chebyshev_derivative_coeffs`, `chebyshev_antiderivative_coeffs`, `chebyshev_integral_coeffs` |
| Clenshaw–Curtis weights or an integral on a Chebyshev grid | `clenshaw_curtis_weights`, `clenshaw_curtis_integrate_1d/2d`, `ChebyshevDerivative*.integrate` |
| A Chebyshev collocation BVP with boundary rows | `ChebyshevHelmholtzSolver1D/2D`, `ChebyshevPoissonSolver1D/2D` |
| Gauss–Legendre nodes, associated Legendre functions, a spherical-harmonic transform | `SphericalGrid1D/2D`, `SphericalHarmonicTransform` |
| A streamfunction / velocity potential on the sphere | `SphericalVorticityInversionSolver`, `SphericalDivergenceInversionSolver`, `SphericalHelmholtzDecomposition` |
| An exponential or hyperviscosity filter | `SpectralFilter*`, `ChebyshevFilter*`, `SphericalFilter*` |
| Repeated shifted solves with one small dense matrix; a Kronecker-sum solve | `gaussx.EigenFactorization`, `gaussx.kronecker_sum_solve` |
| A masked operator, a capacitance / Woodbury correction | `gaussx.MaskedOperator`, `gaussx.CapacitanceSolver`, `gaussx.DiagonalisedOperator` |
| Rejecting complex physical input; `(-1)**n` in the input dtype | `_src/fourier/operators.py::_real`, `_src/fourier/transforms.py::_validate_real`, `_alternating_sign` |
| A run-time error for a singular / resonant solve under `jit` | `_src/fourier/solvers.py::_check_finite` (`eqx.error_if`) |
| BC → transform and eigenvalue lookup, 1-D forward / inverse along an axis | `_src/fourier/solvers.py::_BC_DISPATCH`, `_lookup_bc`, `_forward_1d`, `_inverse_1d` |

## The contracts

### 1. Grids

- **An `eqx.Module` with factory constructors** `from_N_L`, `from_N_dx` and
  (Fourier, spherical) `from_L_dx`. Fourier grids store the redundant
  `L` and `dx` and check `L = N·dx` in `__check_init__` for concrete values
  only (`_check_lengths`), so a grid can still be built inside `jit`.
- **Domain conventions differ per family; keep them.** Fourier: `[0, L)`,
  `x_j = j·L/N` (endpoint excluded), wavenumbers in FFT order
  `k = 2π·fftfreq(N, dx)`. Chebyshev: `[−L, L]` with `L` the
  *half*-length, nodes decreasing from `x_0 = +L`; Gauss–Lobatto has `N + 1`
  points, Gauss `N`. Spherical: colatitude on Gauss–Legendre nodes,
  longitude uniform, `Lx = 2π` and `Ly = π` by default; the solvers and
  operators read the radius as `R = Ly / π`.
- **Axis order is `(…, z, y, x)`**: a 2-D field is `(Ny, Nx)`, a 3-D one
  `(Nz, Ny, Nx)`; the last axis is x (`test_anisotropic.py` catches swaps).
- **Dealiasing is a grid setting** (`dealias="2/3"` or `None`; the Fourier
  grids reject anything else, `_validate_dealias`). The Fourier mask keeps mode `n` iff `3|n| < N`, strictly,
  on integer mode numbers (gh-88).
- **Transforms are unnormalised forward, normalised inverse**
  (`grid.transform(u)` is `fft`, `transform(û, inverse=True)` is `ifft`).

### 2. Operators and filters

- **An `eqx.Module` holding its grid** (plus static settings such as the
  Chebyshev `method="matrix" | "fft"`). Methods take a physical field and
  return a **real** physical field of the same shape. With `spectral=True`
  (Fourier and spherical operators) they take coefficients instead; the
  Fourier derivatives still return a physical field, while `apply_dealias`
  and the spherical Laplacians return coefficients: say which in the
  docstring.
- **Linear operators never dealias** (`__call__`, `gradient`, `laplacian`,
  `biharmonic`, `hyperviscosity`, `inverse_laplacian` keep every resolved
  mode, so `laplacian(inverse_laplacian(u)) == u`, gh-91). **Nonlinear
  products do**: `jacobian` and `advection_scalar` truncate both factors and
  the product; anything you multiply yourself goes through `apply_dealias`.
- **Complex physical input raises** `TypeError` (`_real`, gh-93): transform
  the real and imaginary parts separately.
- **Signs are physical**: `hyperviscosity` is always dissipative
  (`−ν|k|^{2n}`), `inverse_laplacian` returns the zero-mean solution;
  parameters are validated (`_validate_hyperviscosity`).
- **Join the tests**: a closed-form check against an analytic field
  (`test_operators.py`, `test_physics_operators.py`, `test_chebyshev_*`,
  `test_spherical_*`), a case in `CASES` (`tests/test_tracing.py`) for
  `jit` / `vmap` / `grad`, and anisotropic grids where axes could swap.

### 3. Transforms

- **scipy's conventions**, unnormalised (`norm=None`) and orthonormal
  (`norm="ortho"`); every `(type, norm)` round-trips through its inverse.
  The 1-D functions take a 1-D vector (anything else raises); the n-D
  wrappers apply the 1-D kernel along each of `axes` in turn.
- **Implementations are registered**: a 1-D kernel `_dctK` / `_dstK` in
  `_DCT_IMPLS` / `_DST_IMPLS`, its inverse in `_idct_along_axis` /
  `_idst_along_axis`, and the ortho scaling in `_apply_ortho_forward` /
  `_remove_ortho_forward` (`_dct1_ortho` and `_prescale_type3` for the
  asymmetric types).
- **`type`, `norm` and `axes` are static** (they pick Python code paths:
  `jax.jit(dctn, static_argnames=("type", "norm", "axes"))`); input is real
  (`_validate_real`) and the output keeps the input dtype; `type` is
  validated as the int 1–4 (`_validate_type`, which also rejects `True`).
- **Checked against scipy** at odd and prime lengths, both norms, inverses,
  n-D and negative axes (`tests/test_transforms_scipy.py`), plus float32
  (`tests/test_float32.py`).

### 4. Elliptic solvers

- **One equation, one sign: `(∇² − λ)ψ = f`.** The functions take
  `lambda_` (any real; a negative value can resonate), the classes take
  `alpha`, documented as `≥ 0` (the periodic, spherical and Chebyshev
  classes reject a concrete negative value: `_check_zero_mean`,
  `_maybe_check_alpha`). Poisson is `λ = 0`.
- **The BC picks the transform** (`_BC_DISPATCH`; never pair them by hand):

  | BC | Grid | Transform | FD2 eigenvalues |
  |---|---|---|---|
  | `"periodic"` | any | FFT | `fft_eigenvalues` |
  | `"dirichlet"` | regular (vertex), interior points | DST-I | `dst1_eigenvalues` |
  | `"dirichlet_stag"` | staggered (cell centre) | DST-II | `dst2_eigenvalues` |
  | `"neumann"` | regular | DCT-I | `dct1_eigenvalues` |
  | `"neumann_stag"` | staggered | DCT-II | `dct2_eigenvalues` |
  | `("dirichlet", "neumann")` / `("neumann", "dirichlet")` | regular | DST-III / DCT-III | `dst3_` / `dct3_eigenvalues` |
  | `("dirichlet_stag", "neumann_stag")` / reverse | staggered | DST-IV / DCT-IV | `dst4_` / `dct4_eigenvalues` |

- **Which eigenvalues.** The functions default to `approximation="fd2"`
  (the exact inverse of the 3 / 5 / 7-point FD Laplacian; required for the
  inhomogeneous-BC corrections); `"spectral"` selects the continuous
  eigenvalues. The `SpectralHelmholtzSolver1D/2D/3D` classes use the
  continuous `k²`. Say which one a new solver uses.
- **Null mode, one policy** (gh-92): a mode whose denominator is zero (the
  constant at `λ = 0` with periodic / Neumann on every axis) is set to
  zero, never to an arbitrary value; `zero_mean=None` zeroes the mean only
  when it is undefined, `zero_mean=False` at `α = 0` raises.
- **Resonance raises** under `jit`: wrap the result in `_check_finite`
  (`eqx.error_if`), so a `λ` on an eigenvalue fails instead of returning
  inf / NaN or a silently zeroed mode (gh-94).
- **Inhomogeneous BCs** go through `_BC_RHS_FORMULAS` / `modify_rhs_*`
  (ghost-point corrections, FD2 only; periodic axes reject values).
- **BC arguments are static** (`jax.jit(solve_helmholtz_2d,
  static_argnames=("bc_x", "bc_y"))`; class fields `eqx.field(static=True)`);
  `λ` may be traced.
- **The capacitance solver** is built eagerly from a concrete NumPy mask
  (`build_capacitance_solver`), on the 5-point FD2 base operator of
  `base_bc` (`"fft"`, `"dst"` or `"dct"`), with `ψ = 0` on the inner
  boundary; the linear algebra is gaussx's.
- **Checked against dense matrices**: every `_BC_DISPATCH` entry is swept by
  `tests/test_solvers_dense.py` (eigenpairs of the dense FD2 matrix, 2-D and
  3-D solves against a dense Kronecker sum), so a new BC needs its
  boundary-row closure in `dense_laplacian_1d` there.

### 5. JAX numerics

- **Pure and traceable.** Arrays and modules in, arrays out; no global
  state; no Python `if` / `float()` / `.item()` / `np.asarray` on a traced
  value (validation that needs a concrete value skips tracers, as
  `_check_lengths`, `_check_zero_mean` and `_concrete_numpy` do).
- **Dtypes.** The transforms and the Fourier `solve_*` functions (and the
  solver classes that only call them, such as `DirichletHelmholtzSolver2D`)
  return float32 for float32 input even with x64 on: build constants from
  the input's dtype (`_alternating_sign(n, x.dtype)`, not `(-1.0) ** n`).
  Everything that multiplies by precomputed grid or solver arrays — the
  derivative operators, filters, `SpectralHelmholtzSolver*`, the
  capacitance, Chebyshev and spherical solvers — computes in the default
  float (float32 with x64 off, float64 with it on).
  `tests/test_float32.py` runs float32 cases with x64 off.
- **Modules under `jit`.** Close over a module rather than passing it as a
  `jit` argument (`jax.jit(lambda u: deriv.laplacian(u))`): module fields
  are not all jit-safe yet (#102).
- **Under every transform.** A new public operator or solver joins `CASES`
  in `tests/test_tracing.py` (`jit` equals eager, `vmap` equals a loop,
  `grad` matches a central finite difference to 1e-6); a known gap is an
  `xfail(strict=True)` naming its issue, like `_JIT_XFAIL` (gh-98).

### The public API

- **Export** a new public name from its `_src/<family>/__init__.py`, then
  from `spectraldiffx/__init__.py` (import and `__all__`, kept sorted —
  ruff's `RUF022`); add a `::: spectraldiffx.<Name>` entry on its
  `docs/api/<family>/<page>.md` page (`tests/test_api_docs.py`); run
  `make capabilities`.
- **Docstrings are numpy style** (mkdocstrings renders `docstring_style:
  numpy`; ruff's `D405`–`D414` check the section format) on every module,
  class and function, private ones included. Equations in plain text,
  Unicode or ASCII (`λ_k = −4/dx² · sin²(…)`, `nabla^2 psi`), never LaTeX
  (that is for the docs pages and notebooks); every array argument and
  return with its shape (`rhs : Float[Array, "Ny Nx"]`); the BC, grid
  placement and eigenvalue choice for a solver; a reference for a published
  method; an `Examples` section. Be pedagogical and use the terminology the
  rest of the package uses. Doctests are not collected by CI, so run a new
  example yourself.
- **Breaking changes** go through a `DeprecationWarning` first (see
  "Boundaries": finitevolX imports these names).

## What enforces them

Most rules here are tests in the fast tier, so CI tells you when one breaks.
Read the test's docstring before changing what it checks.

| Test | Enforces |
|---|---|
| `tests/test_api_docs.py` | Every name in `__all__` has a `::: spectraldiffx.<name>` entry under `docs/api/`, and every entry is a public name |
| `tests/test_capabilities.py` | `docs/api/capabilities.md` is current; no spectraldiffx name shadows a gaussx one (except `ALLOWED_SHARED_NAMES`) |
| `tests/test_docs.py` | Every `python` fence in `README.md`, `docs/*.md` and `docs/theory/*.md` runs (mktestdocs, one namespace per page; `docs/theory/elliptic_solvers.md` is excluded as signature sketches) |
| `tests/test_tracing.py` | `jit`, `vmap` and `grad` across `CASES`, a representative set of the public transforms, operators, filters and solvers |
| `tests/test_float32.py` | float32 stays float32 with x64 off and matches the float64 result |
| `tests/test_transforms_scipy.py` | DCT / DST I–IV and inverses agree with `scipy.fft` (odd and prime N, both norms, n-D, negative axes, dtypes) |
| `tests/test_solvers_dense.py` | Every `_BC_DISPATCH` entry: the transform basis diagonalises the dense FD2 matrix with the package's eigenvalues; 2-D / 3-D solves match a dense Kronecker-sum solve |
| `tests/test_null_mode.py` | The null-mode / `zero_mean` policy of the Fourier and spherical solvers |
| `tests/test_guards.py`, `tests/test_edge_cases.py` | Input guards (DCT-I length, transform type, `dealias`, `L = N·dx`, resonant `λ`, negative `alpha`) and input corners (N = 1, complex input, `spectral=True`) |
| `tests/test_anisotropic.py` | Non-square grids with a different length per axis: no swapped axes |
| `tests/test_correctness.py` | Parseval, spectral convergence, dealiasing of products, conservation in model dynamics |
| `tests/test_release_please_config.py` | Release tags are plain semver (`0.1.1`, no component, no `v`) |
| ruff (`make lint`), ty (`make typecheck`) | Lint (including the numpydoc section format and sorted `__all__`) and types on `spectraldiffx/` |

## Working in the repo

Always run Python tools through `uv run` (never the system Python); `git`,
`ls` and other non-Python commands need no `uv run`.

```bash
make install              # uv sync --all-extras + pre-commit hooks
make test-fast            # the fast tier, what PR CI runs (-n auto)
make test-slow            # the slow + integration tiers
make test                 # every test, in parallel
make test-cov             # every test with coverage (reports/)
make lint                 # ruff check .   (entire repo)
make format-check         # ruff format --check .
make format               # ruff format . && ruff check --fix .
make typecheck            # ty check spectraldiffx (what CI runs)
make capabilities         # regenerate docs/api/capabilities.md
make docs                 # mkdocs build --strict
make docs-serve           # local MkDocs preview
make nb-check             # no .ipynb committed under notebooks/
```

Make targets take no paths. Run one test from the repo root with
`uv run pytest tests/test_fourier_solvers.py -k dirichlet -v`
(`tests/conftest.py` turns x64 on). Lint and format have no dependency
group of their own: `uv run ruff …` uses the dev extras `uv sync` installs.

### Test tiers

- **Unmarked (fast):** unit tests, each well under a few seconds.
- **`@pytest.mark.slow`:** individually expensive tests (over ~3 s); for a
  parametrised sweep, mark only the expensive cases
  (`pytest.param(..., marks=pytest.mark.slow)`, as `test_solvers_dense.py`
  does).
- **`@pytest.mark.integration`:** end-to-end workflows (registered; no test
  uses it yet).

PR CI (`ci.yml`) runs `pytest tests -n auto -m "not slow and not
integration"` with coverage (`fail_under = 50`) on Ubuntu and macOS,
Python 3.12 and 3.13. `tests-extended.yml` runs the slow and integration
tiers nightly and on demand (`gh workflow run tests-extended.yml`, `-f
suite=full` for everything); run the slow tests of what you touched
locally (`uv run pytest -m slow tests/<file>`). `filterwarnings = error`
turns any warning into a failure (JAX deprecation notices stay visible);
`xfail_strict = true`.

### Tests that assert on random draws

The library draws no random numbers; tests use random *fields* as inputs.

- Draw them from a seeded `np.random.default_rng(seed)` (or a fixed key),
  so a failure reproduces.
- Compare against a closed form, `scipy.fft`, a dense matrix or a float64
  reference, and say in a comment where the tolerance comes from (as
  `test_float32.py` and `test_tracing.py` do: the measured error and the
  margin above it).
- One behaviour per test; use fixtures for different equations and grids.

### Before every commit

All of these must pass, from the repo root:

1. `make test-fast` (zero failures), plus the slow tests of what you touched.
2. `uv run ruff check .` — the **entire** repo, which includes `tests/` and
   `scripts/`. Never lint a subdirectory.
3. `uv run ruff format --check .`
4. `make typecheck` (`ty check spectraldiffx`, what CI runs).
5. After changing a public API: `make capabilities`, and the `docs/api`
   entry.
6. After changing docs, docstrings or `mkdocs.yml`: `make docs`
   (`mkdocs build --strict`, what `docs.yml` runs on every PR).
7. After changing a dependency: `uv lock`, and commit `uv.lock`.

## Coding principles

1. **Think before coding.** State assumptions; if a request has several
   readings, name them instead of picking one silently; if something is
   unclear, stop and ask.
2. **Simplicity first.** The minimum code that solves the problem: no
   speculative features, no single-use abstractions, no configurability
   nobody asked for, no error handling for impossible cases.
3. **Surgical changes.** Touch only what the task needs; match the existing
   style; don't refactor or add docstrings to code you didn't change;
   remove only what your change made unused (mention other dead code, don't
   delete it).
4. **Goal-driven.** Turn the task into a check (a failing test, a
   reproduced bug, an analytic solution or dense reference to match) and
   loop until it passes; for multi-step work, state the plan with a check
   per step.

Also: Python 3.12+, type hints on every public function, `X | None` over
`Optional`, specific exceptions with `raise … from …`, and prefer
JAX-native vectorised code over Python loops on the numerical path.

## Git, commits and pull requests

- **Never** push to or merge into `main` unless explicitly told to ("push
  to main", "merge to main"). Work on a feature branch and commit locally;
  never run `git push` unless asked. "Merge the branch" means push the
  feature branch, not merge into `main`. Confirm before any action that
  affects a shared branch.
- Commit messages and PR titles follow
  [Conventional Commits](https://www.conventionalcommits.org/) with a
  lowercase subject; CI validates PR titles. Types: `feat`, `fix`, `docs`,
  `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
  Scopes name the area: `fourier`, `transforms`, `solvers`, `capacitance`,
  `operators`, `filters`, `grid`, `chebyshev`, `spherical`, `docs`, `deps`.
  Breaking changes use `!` and a `BREAKING CHANGE:` footer.
- Releases are cut by release-please with plain semver tags (`0.1.1`, no
  `v`); don't bump versions by hand.
- **Never replace or remove an existing PR title or description.** An
  agent asked for a small follow-up tends to write a fresh description
  scoped to its own work and silently discard the rest; read the existing
  description first and only append checklist items or update their status.
- Code review follows [`CODE_REVIEW.md`](CODE_REVIEW.md).

### Pull Request Review Comments

After fixing a review comment, resolve its thread. Don't resolve threads you
didn't address.

```bash
# 1. List the review threads and their IDs
gh api graphql -f query='
  query($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100) {
          nodes { id isResolved comments(first: 1) { nodes { body path line } } }
        }
      }
    }
  }' -f owner=OWNER -f repo=REPO -F pr=PR_NUMBER

# 2. Resolve an addressed thread
gh api graphql -f query='mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) { thread { isResolved } } }' \
  -f threadId=THREAD_ID
```

When the `gh` CLI is unavailable, use the GitHub MCP tools for the same
operations.

## Documentation

MkDocs + Material + mkdocstrings (numpy style) + mkdocs-jupyter, from
`mkdocs.yml` (`docs_dir: docs`). `docs.yml` builds it with `--strict` on
every PR; `pages.yml` deploys it on every push to `main`
(<https://jejjohnson.github.io/spectraldiffx/>).

- **Pages**: guides (`docs/*_guide.md`), theory (`docs/theory/`), and the
  API reference (`docs/api/<family>/<page>.md`, one
  `::: spectraldiffx.<Name>` per public name). Every `python` fence in
  `README.md`, `docs/*.md` and `docs/theory/*.md` runs in
  `tests/test_docs.py`, so make examples self-contained per page; use a
  `text` fence for anything that is not runnable.
- **Notebooks** live in `notebooks/` (`docs/notebooks` is a symlink to it)
  as **jupytext percent-format `.py` files only**: `.ipynb` files are
  gitignored and rejected by the `forbid-ipynb` pre-commit hook
  (`make nb-check`). mkdocs-jupyter renders them **without executing**
  (`execute: false`), so each figure is written with `fig.savefig` to
  `docs/images/<notebook>/` (committed) and shown in the next markdown cell
  with `![…](../../images/<notebook>/<figure>.png)`. Add a new notebook to
  the "Examples" nav in `mkdocs.yml`.
- `make docs` must pass after any change to docstrings, `docs/` or
  `mkdocs.yml`.

## Plans

Plans and scratch design notes go in `.plans/` (gitignored, never
committed); track work in GitHub issues.
