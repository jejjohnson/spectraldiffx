---
name: add-elliptic-solver
description: Add or extend an elliptic (Poisson / Helmholtz) solver in spectraldiffx — a new per-axis boundary condition (eigenvalues + _BC_DISPATCH entry + ghost-point correction), a new Fourier solve function or eqx.Module solver class, a Chebyshev or spherical solver, or a capacitance base — with the BC ↔ transform ↔ eigenvalue pairing, the null-mode and resonance policy, the dense-matrix tests, docs and capability index. Use when asked to add, port or fix a Poisson, Helmholtz, Laplace or inversion solver, a boundary condition, or an eigenvalue formula in spectraldiffx/_src/*/solvers.py, eigenvalues.py or capacitance.py.
---

# Add an elliptic solver or boundary condition

Read "4. Elliptic solvers" and "5. JAX numerics" under "The contracts" in
`AGENTS.md` first; this is the step-by-step.

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md` (Fourier: Spectral Elliptic Solvers,
  Eigenvalue Helpers, Capacitance Solver; Chebyshev and spherical solvers).
  Any combination of the nine per-axis BCs in `_BC_DISPATCH`
  (`_src/fourier/solvers.py`) is already solvable in 2-D and 3-D through
  `solve_helmholtz_2d` / `solve_helmholtz_3d` (classes
  `MixedBCHelmholtzSolver2D/3D`), with inhomogeneous values through
  `bc_x_values=` / `modify_rhs_*`. A "new solver" for one such combination
  is a call, not new code.
- Issue #106 plans one N-D Helmholtz kernel driven by a per-axis BC table,
  with the fixed-BC `solve_*` functions as generated wrappers: prefer
  extending `solve_helmholtz_2d` / `_3d` and `_BC_DISPATCH` over adding
  another hand-written `solve_*` variant.
- A masked / irregular domain is `build_capacitance_solver`; repeated
  shifted solves with a small dense operator are `gaussx.EigenFactorization`;
  a Kronecker-sum system is `gaussx.kronecker_sum_solve`.

## 2. Pick the extension point

| You are adding… | Where | Exemplar |
|---|---|---|
| A per-axis BC (new transform pairing) | `_src/fourier/eigenvalues.py` (FD2 `name_eigenvalues(N, dx)` and PS `name_eigenvalues_ps(N, L)`), then `_BC_DISPATCH` (and the `SameBC` / `MixedBC` aliases) in `_src/fourier/solvers.py`, and `_BC_RHS_FORMULAS` if it introduces a new side type | `"dirichlet_stag"` → `("dst", 2, dst2_eigenvalues, False)` |
| A fixed-BC Fourier solve function (1-D, 2-D, 3-D) | `_src/fourier/solvers.py`, layer 0 | `solve_helmholtz_dst2` / `solve_poisson_dst2` |
| An `eqx.Module` wrapper | `_src/fourier/solvers.py`, layer 1 | `StaggeredDirichletHelmholtzSolver2D` (`__call__`), `SpectralHelmholtzSolver2D` (`solve`, `zero_mean`), `MixedBCHelmholtzSolver2D` (static BC fields) |
| A Chebyshev solver | `_src/chebyshev/solvers.py` | `ChebyshevHelmholtzSolver2D` (eigen-factorised at construction, dense fallback for a traced grid) |
| A spherical solver | `_src/spherical/solvers.py` | `SphericalHelmholtzSolver`, `SphericalVorticityInversionSolver` |
| A capacitance base | `_BASE_BCS`, `_base_operator` in `_src/fourier/capacitance.py` | `"dct"` → `gaussx.DiagonalisedOperator(eig, _dct2, _idct2, …)` |

## 3. Write it

- **Equation and sign**: `(∇² − λ)ψ = f`; functions take `lambda_`, classes
  `alpha` (reject a concrete negative value as `_check_zero_mean` /
  `_maybe_check_alpha` do). Poisson twins call the Helmholtz function with
  `lambda_=0.0`.
- **The BC picks the transform** and its eigenvalues: get the triple from
  `_lookup_bc` (or the fixed transform the function is named after), and
  state the grid placement in the docstring (regular = vertex, interior
  points only for Dirichlet; staggered = cell centres).
- **Eigenvalues**: an `approximation: Approximation = "fd2"` keyword routed
  through `_eig_1d(fd2_fn, ps_fn, N, dx, L, approximation)`; pass the right
  `L` for the PS eigenvalues (DST-I: `(N + 1)·dx`; staggered and periodic:
  `N·dx`; DCT-I: `(N − 1)·dx` — copy the existing call, gh-94). Build them
  with `jnp.arange(N)` and Python scalars only, so a float32 right-hand
  side stays float32.
- **Null mode** (gh-92): only a zero denominator at the constant mode is
  replaced (`denom_safe = jnp.where(is_null, 1.0, denom)`, then the mode set
  to zero); `jnp.where` before the division keeps gradients finite.
- **Resonance** (gh-94): return `_check_finite(psi)` so a `λ` on an
  eigenvalue raises under `jit` instead of returning inf / NaN.
- **Mixed FFT / real transforms**: transform the real and imaginary parts
  separately along the non-periodic axis (as `solve_helmholtz_2d` does);
  the output is real.
- **Static vs traced**: BC arguments and class BC fields are static
  (`eqx.field(static=True)`); `λ` / `alpha` may be traced, so no Python
  `if` on them except through a concrete-value check that skips tracers.
- **Inhomogeneous BCs**: only with FD2 eigenvalues; a new side type needs
  its ghost-point formula in `_BC_RHS_FORMULAS` (returns the corrections to
  add to `rhs[0]` / `rhs[-1]`); periodic axes reject values.
- **Chebyshev**: precompute on concrete matrices at construction
  (`_concrete_numpy`, `gaussx.EigenFactorization.from_matrix`), keep the
  dense per-call fallback for a traced grid, document the gauge for the
  singular Neumann–Poisson case.
- **Docstring** (numpy style): the equation, the spectral algorithm as
  numbered steps with shapes, BC and grid placement, which eigenvalues,
  the null-mode behaviour, `Raises`, an `Examples` section you have run.

## 4. Export and document

- `_src/fourier/__init__.py` (or the family's `__init__.py`), then
  `spectraldiffx/__init__.py`: import and `__all__` (sorted, ruff `RUF022`).
- `::: spectraldiffx.<name>` on `docs/api/fourier/solvers.md` (layer 0 or
  layer 1 section), `docs/api/fourier/eigenvalues.md`,
  `docs/api/chebyshev/solvers.md` or `docs/api/spherical/solvers.md`
  (`tests/test_api_docs.py`); the BC table in
  `docs/elliptic_solvers_guide.md` / `docs/theory/elliptic_solvers.md` if
  you added a BC (fences in the guide run in `tests/test_docs.py`).
- `make capabilities`.
- finitevolX re-exports these solvers by name: never rename one without a
  `DeprecationWarning` alias.

## 5. Tests

- **New BC**: its boundary-row closure in `dense_laplacian_1d`
  (`tests/test_solvers_dense.py`; add it to `_NULL_BCS` if it has a
  constant null mode). `BCS = list(_BC_DISPATCH)`, so the eigenpair test
  and the 2-D sweep (same-BC pairs fast, mixed pairs `slow`) then cover it;
  add a row to `_COMBOS_3D` for 3-D.
- **Eigenvalues**: shape, sign, closed form and the FD2 → PS convergence in
  `tests/test_fourier_eigenvalues.py` and `tests/test_ps_eigenvalues.py`.
- **Solver**: an analytic or manufactured solution in
  `tests/test_fourier_solvers.py` (or `test_fourier_mixed_bc_solvers.py`,
  `test_fourier_mixed_bc_3d_solvers.py`, `test_fourier_capacitance.py`;
  Chebyshev: `test_chebyshev_solvers.py`, `test_chebyshev_new_features.py`,
  `test_chebyshev_extensions.py`; spherical: `test_spherical_solvers.py`,
  `test_spherical_new_features.py`); the null
  mode and `zero_mean` in `tests/test_null_mode.py`; inhomogeneous values
  in `tests/test_inhomogeneous_bcs.py`; guards (resonance, negative
  `alpha`) in `tests/test_guards.py`.
- **Transforms of the solver**: a case in `CASES` (`tests/test_tracing.py`)
  for `jit` / `vmap` / `grad`; a float32 case in `tests/test_float32.py`
  for a new solve path.

## 6. Verify

```bash
uv run pytest tests/test_solvers_dense.py tests/test_fourier_eigenvalues.py \
  tests/test_ps_eigenvalues.py tests/test_fourier_solvers.py \
  tests/test_fourier_mixed_bc_solvers.py tests/test_fourier_mixed_bc_3d_solvers.py \
  tests/test_null_mode.py tests/test_inhomogeneous_bcs.py tests/test_guards.py \
  tests/test_tracing.py tests/test_float32.py tests/test_api_docs.py \
  tests/test_capabilities.py -n auto
```

(no `-m`: this runs the slow cases of those files too), then the
`pre-pr-check` skill.
