---
name: add-transform
description: Add or change a spectral transform in spectraldiffx — a DCT / DST kernel or normalisation in spectraldiffx/_src/fourier/transforms.py (registered in _DCT_IMPLS / _DST_IMPLS with its inverse and ortho scaling), its n-D wrapper behaviour, or a Chebyshev / spherical transform or coefficient-space routine — checked against scipy.fft, round trips, float32 and jit / vmap / grad. Use when asked to add, port, speed up or fix a DCT, DST, FFT-based, Chebyshev or spherical-harmonic transform.
---

# Add or change a transform

Read "3. Transforms" and "5. JAX numerics" under "The contracts" in
`AGENTS.md` first; this is the step-by-step.

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md`: `dct` / `dst` / `idct` / `idst` (types
  I–IV, `norm=None` or `"ortho"`) and `dctn` / `dstn` / `idctn` / `idstn`
  cover every scipy DCT / DST; `FourierGrid*.transform` is the FFT;
  `ChebyshevGrid*.transform` / `ChebyshevTransform1D/2D` the Chebyshev
  coefficients (with `chebyshev_derivative_coeffs` and friends);
  `SphericalHarmonicTransform` the SHT.
- A transform that diagonalises a new boundary condition also needs
  eigenvalues and a `_BC_DISPATCH` entry: follow the `add-elliptic-solver`
  skill after this one.

## 2. Where it goes

| You are adding… | Where | Exemplar |
|---|---|---|
| A DCT / DST kernel (unnormalised, one axis, any ndim) | `_dctK(x, axis)` / `_dstK(x, axis)` in `_src/fourier/transforms.py`, registered in `_DCT_IMPLS` / `_DST_IMPLS` | `_dct2` (Makhoul: an N-point FFT of a reordered sequence), `_dst1` (odd extension and `rfft`) |
| Its inverse | `_idct_along_axis` / `_idst_along_axis` (the scale and the dual type) | `IDCT-II = DCT-III / (2N)` |
| Its orthonormal form | `_ortho_uniform_factor`, `_apply_ortho_forward`, `_remove_ortho_forward`; `_prescale_type3` / `_dct1_ortho` for asymmetric weights | DCT-I ortho is computed directly by `_dct1_ortho` |
| A Chebyshev transform or coefficient routine | `_src/chebyshev/grid.py` (`_transform_gl`, `_transform_gauss`, `_transform_along_axis`) or `_src/chebyshev/transforms.py` | `chebyshev_derivative_coeffs` |
| A spherical transform | `_src/spherical/grid.py` / `harmonics.py` (ALPs precomputed at construction) | `SphericalHarmonicTransform` |

## 3. Write it

- **Definition**: write the transform's sum in the module docstring and the
  function docstring, in scipy's normalisation; if scipy has it, match
  scipy exactly (including which elements get the `1/√2` corrections in
  `"ortho"`).
- **Validation, once, up front** (in the public function, before the axis
  loop): `_validate_norm`, `_validate_real` (complex input raises
  `TypeError`), `_validate_type` (the int 1–4; `True` is rejected), and a
  length guard like `_validate_dct1_length` if the kernel divides by
  `N − 1` or similar.
- **dtype**: the output has the input's dtype, also with x64 on. Build
  phases and signs from the input's dtype (`_alternating_sign(n, x.dtype)`,
  `jnp.result_type(x.dtype, jnp.complex64)` for the complex intermediate),
  never `(-1.0) ** n` or a bare `jnp.ones(N)`.
- **Static arguments**: `type`, `norm` and `axes` select Python code paths;
  document that `jax.jit` needs them in `static_argnames`.
- **Shapes**: the 1-D functions reject `ndim != 1`; the n-D wrappers
  normalise negative axes (`_norm_axis`) and loop over `axes` (all axes for
  `None`). Use `_make_idx` / `_phase_shape` / `_sl` for axis-generic
  indexing.
- **Chebyshev**: Gauss–Lobatto coefficients are the true coefficients
  (`a₀` and `a_N` halved, since 0.1.0); keep forward / inverse a pair on
  both node types.

## 4. Export and document

- A new public function: `_src/fourier/__init__.py` (or the family's
  `__init__.py`), then `spectraldiffx/__init__.py` (import and sorted
  `__all__`); `::: spectraldiffx.<name>` on `docs/api/fourier/transforms.md`
  (or the family's page); `make capabilities`.
- The definitions table in `docs/theory/spectral_transforms.md` and, for a
  user-facing change, an example in `docs/transforms_guide.md` (its
  `python` fences run in `tests/test_docs.py`).

## 5. Tests

- **Against scipy**: `tests/test_transforms_scipy.py` parametrises over
  `SIZES` (odd and prime lengths), `TYPES`, `NORMS` and the `_PAIRS` /
  `_PAIRS_N` tables (forward, inverse, n-D with partial and negative axes,
  float32 preserved under x64): extend those lists, don't write a parallel
  test.
- **Round trips and multi-axis**: `tests/test_fourier_transforms.py`.
- **Guards and corners**: `tests/test_guards.py` (type, length),
  `tests/test_edge_cases.py` (`_TRANSFORMS`: N = 1, complex input).
- **float32 with x64 off**: `tests/test_float32.py::test_transform_roundtrip`.
- **Transforms of the transform**: a case in `CASES`
  (`tests/test_tracing.py`), as `dctn` and `idstn` have.
- Chebyshev: `tests/test_chebyshev_grid.py`,
  `tests/test_chebyshev_new_features.py` (`ChebyshevTransform1D/2D`),
  `tests/test_chebyshev_extensions.py` (coefficient conventions and
  calculus); spherical:
  `tests/test_spherical_harmonics.py`, `tests/test_spherical_ylm.py`.

## 6. Verify

```bash
uv run pytest tests/test_transforms_scipy.py tests/test_fourier_transforms.py \
  tests/test_guards.py tests/test_edge_cases.py tests/test_float32.py \
  tests/test_tracing.py tests/test_solvers_dense.py tests/test_api_docs.py \
  tests/test_capabilities.py -n auto
```

(`test_solvers_dense.py` because every DST / DCT solver goes through these
kernels), then the `pre-pr-check` skill.
