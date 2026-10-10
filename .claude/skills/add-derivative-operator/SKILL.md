---
name: add-derivative-operator
description: Add a derivative or physics operator to spectraldiffx — a method on SpectralDerivative1D/2D/3D, ChebyshevDerivative1D/2D/3D or SphericalDerivative1D/2D (gradient-type, Laplacian-type, Jacobian, advection, inversion), or a new operator class — with the right spectral symbol, the linear-vs-nonlinear dealiasing rule, real-input guards, analytic tests, the tracing registry, docs and capability index. Use when asked to add, port or fix a spectral derivative, differential operator, vorticity / streamfunction / advection term in spectraldiffx/_src/*/operators.py.
---

# Add a derivative operator

Read "1. Grids", "2. Operators and filters" and "5. JAX numerics" under
"The contracts" in `AGENTS.md` first; this is the step-by-step.

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md` and the classes' methods:
  `SpectralDerivative2D` already has `gradient`, `divergence`, `curl`,
  `laplacian`, `biharmonic`, `hyperviscosity`, `inverse_laplacian`,
  `velocity_from_streamfunction`, `jacobian`, `apply_dealias`,
  `project_vector`, `advection_scalar` (3-D has the same set; 1-D the scalar
  ones). `ChebyshevDerivative2D` adds `vector_laplacian` and `integrate`;
  `SphericalDerivative2D` has `iterated_laplacian`. A composition of these
  needs no new method.
- An elliptic inverse (Poisson, Helmholtz, vorticity inversion) is a solver:
  use the `add-elliptic-solver` skill.

## 2. Where it goes

| Family | File | Pattern to copy |
|---|---|---|
| Fourier (periodic) | `_src/fourier/operators.py`, on `SpectralDerivative1D/2D/3D` | `u_hat = u if spectral else self.grid.transform(_real(u))`, multiply by the symbol built from `self.grid.k` / `KX` / `K2`, `self.grid.transform(…, inverse=True).real` |
| Chebyshev | `_src/chebyshev/operators.py`, on `ChebyshevDerivative1D/2D/3D` | compose `self._dx` / `self._dy` (`_d(u, axis, order)` in 3-D), which honour `method="matrix" \| "fft"` through `_diff_along_axis` |
| Spherical | `_src/spherical/operators.py`, on `SphericalDerivative1D/2D` | spectral symbol in `(l, m)` from `self.grid.l` / `m` and the radius, through the grid's SHT |

Add the method to every dimension of the family where it makes sense, with
the same name and signature (1-D, 2-D and 3-D Fourier operators mirror one
another).

## 3. Write it

- **Symbol and scale**: wavenumbers from the grid (they carry 2π/L;
  Chebyshev matrices carry 1/L for the half-length; spherical eigenvalues
  carry 1/R²). Axis order `(…, z, y, x)`: `KX` multiplies along the last
  axis.
- **Linear or nonlinear**: a linear operator applies its symbol to every
  resolved mode and **never** the 2/3 mask (gh-91). A nonlinear term
  truncates its factors (`* self.grid.dealias_filter()` in spectral space,
  or `self.apply_dealias(v)`) and the product (`self.apply_dealias(...)`),
  as `jacobian` and `advection_scalar` do (gh-90); Chebyshev products go
  through `cheb_dealias_product`.
- **Input and output**: physical input must be real (`_real(u)` raises for
  complex, gh-93); return `.real` of the inverse transform, the input's
  shape. `spectral=True` takes coefficients; document what comes back.
- **Odd derivatives** must not keep a Nyquist component on even grids
  (taking `.real` of the inverse does this for the FFT path; check a new
  path).
- **Parameters**: validate in Python before any tracing
  (`_validate_hyperviscosity` style), state signs (`hyperviscosity` is
  dissipative for every order).
- **Docstring** (numpy style): the operator in plain-text math, its
  spectral form (`lap_hat = -(k^2) * u_hat`), shapes, the dealiasing
  behaviour, an `Examples` section you have run.

## 4. Export and document

- A new **method** is documented by the existing
  `::: spectraldiffx.<Class>` entry (mkdocstrings lists members): nothing
  to export, and the capability index (classes, not methods) is
  unchanged.
- A new **class**: `_src/<family>/__init__.py`, `spectraldiffx/__init__.py`
  (import and sorted `__all__`), `::: spectraldiffx.<Class>` on
  `docs/api/<family>/operators.md` (`tests/test_api_docs.py`),
  `make capabilities`.
- If users will reach for it, a short runnable example in the matching
  guide or theory page (fences run in `tests/test_docs.py`).

## 5. Tests

- **Closed form** on a single resolved mode (exact to round-off): Fourier
  in `tests/test_operators.py` / `tests/test_physics_operators.py`;
  Chebyshev in `tests/test_chebyshev_operators.py` /
  `tests/test_chebyshev_extensions.py`; spherical in
  `tests/test_spherical_operators.py` / `tests/test_spherical_new_features.py`.
- **Identities and convergence** where they apply (div curl = 0, spectral
  convergence, dealiasing of products, conservation):
  `tests/test_correctness.py`.
- **Anisotropic grids** (different N and L per axis) for anything with
  more than one axis: `tests/test_anisotropic.py`.
- **`spectral=True`** gives the same result as the physical path.
- **Transforms**: a case in `CASES` (`tests/test_tracing.py`) for `jit`,
  `vmap` and `grad` (wrap complex outputs with `_real`).
- Tolerances: say in a comment where they come from.

## 6. Verify

```bash
uv run pytest tests/test_operators.py tests/test_physics_operators.py \
  tests/test_correctness.py tests/test_anisotropic.py tests/test_tracing.py \
  tests/test_api_docs.py tests/test_capabilities.py -n auto
```

plus the family's files (`tests/test_chebyshev_operators.py
tests/test_chebyshev_extensions.py` or `tests/test_spherical_operators.py
tests/test_spherical_new_features.py`), then the `pre-pr-check` skill.
