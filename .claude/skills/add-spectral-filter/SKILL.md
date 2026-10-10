---
name: add-spectral-filter
description: Add a spectral filter to spectraldiffx — a method on SpectralFilter1D/2D/3D, ChebyshevFilter1D/2D or SphericalFilter1D/2D (exponential, hyperviscous, sharp cut-off, Vandeven-style), or a new filter class — as a multiplicative mask in coefficient space normalised by each axis' highest mode, with its tests, tracing case, docs and capability index. Use when asked to add, port or fix a filter, a damping / smoothing mask, or spectral viscosity in spectraldiffx/_src/*/filters.py.
---

# Add a spectral filter

Read "1. Grids" and "2. Operators and filters" under "The contracts" in
`AGENTS.md` first; this is the step-by-step.

## 1. Make sure it does not exist yet

- Every family has `exponential_filter(u, alpha=36.0, power=16,
  spectral=False)` and `hyperviscosity(u, nu_hyper, dt, power=4,
  spectral=False)` (Fourier 1-D / 2-D / 3-D, Chebyshev 1-D / 2-D,
  spherical 1-D / 2-D). The 2/3 truncation is the grid's
  `dealias_filter()`, not a filter.
- A hyperviscous *tendency* (a right-hand-side term rather than a damping
  factor per step) is `SpectralDerivative*.hyperviscosity`, an operator.

## 2. Where it goes

| Family | File | Pattern to copy |
|---|---|---|
| Fourier | `_src/fourier/filters.py` | `_exponential_1d(k, alpha, power)` per axis, combined as a tensor product (gh-89) |
| Chebyshev | `_src/chebyshev/filters.py` | mask over the mode index `k`, normalised by the highest mode (`N` for Gauss–Lobatto, `N − 1` for Gauss), in `_coeff_dtype(a)` |
| Spherical | `_src/spherical/filters.py` | mask over the degree `l` |

Add the method to each dimension of the family, with the same signature.

## 3. Write it

- **A multiplicative mask in coefficient space**: forward transform (unless
  `spectral=True`), multiply, inverse transform; with `spectral=True`
  return the filtered coefficients, otherwise the real physical field.
- **Normalise per axis** by that axis' highest resolved mode, so a filter
  does the same thing at each axis' Nyquist on anisotropic grids
  (`F(kx_max) = exp(−α)` on every axis; a radial `|k|` over the corner of
  the wavenumber box leaves the axis Nyquist modes almost undamped, gh-89).
  Guard the normalisation against a zero maximum (`jnp.where(k_max == 0,
  1.0, k_max)`).
- **Parameters**: validate concrete values in Python (the Chebyshev filters
  reject `alpha < 0` and `power <= 0`); document defaults and what they
  mean (`alpha = 36` puts the top mode at about double-precision ε).
- **dtype**: build the mask in the coefficients' real dtype
  (`_coeff_dtype` in the Chebyshev filters), not a fresh default-float
  array.
- **Docstring** (numpy style): the kernel `F(k)` in plain text, the
  normalisation, shapes, an `Examples` section you have run.

## 4. Export and document

- A new method is documented by the class's existing
  `::: spectraldiffx.<Class>` entry; a new class goes into the family's
  `__init__.py`, `spectraldiffx/__init__.py` (sorted `__all__`), a
  `::: spectraldiffx.<Class>` entry on `docs/api/<family>/filters.md`
  (`tests/test_api_docs.py`), and `make capabilities`.

## 5. Tests

- `tests/test_filters.py` (Fourier), `tests/test_chebyshev_filters.py`,
  `tests/test_spherical_filters.py`: low modes kept to round-off, the top
  mode damped by exactly `exp(−α)` on **each** axis of a non-square grid,
  `spectral=True` consistent with the physical path, invalid parameters
  raising.
- A case in `CASES` (`tests/test_tracing.py`), as the three
  `exponential_filter`s have, for `jit` / `vmap` / `grad`.

## 6. Verify

```bash
uv run pytest tests/test_filters.py tests/test_chebyshev_filters.py \
  tests/test_spherical_filters.py tests/test_tracing.py tests/test_api_docs.py \
  tests/test_capabilities.py -n auto
```

then the `pre-pr-check` skill.
