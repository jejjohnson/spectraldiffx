# SpectralDiffX

> Pseudospectral differentiation, filtering, and PDE solvers in [JAX](https://github.com/google/jax) — fully differentiable and JIT-compatible.

**SpectralDiffX** provides composable building blocks for pseudospectral methods built on JAX and [Equinox](https://github.com/patrick-kidger/equinox). It supports **Fourier**, **Chebyshev**, and **spherical harmonic** bases for grids, derivatives, spectral filters, and Helmholtz/Poisson solvers in 1D, 2D, and 3D.

---

## Features

| Module | Components | Dimensions |
|--------|-----------|------------|
| **Fourier** | `FourierGrid` | 1D, 2D, 3D |
| **Spectral Operators** | `SpectralDerivative`, `SpectralFilter`, `SpectralHelmholtzSolver` | 1D, 2D, 3D |
| **Chebyshev** | `ChebyshevGrid`, `ChebyshevDerivative`, `ChebyshevFilter`, `ChebyshevHelmholtzSolver` | 1D, 2D |
| **Spherical Harmonics** | `SphericalGrid`, `SphericalDerivative`, `SphericalFilter`, `SphericalHarmonicTransform`, `SphericalHelmholtzSolver`, `SphericalPoissonSolver` | 1D, 2D |

All classes are Equinox modules — fully pytree-compatible, immutable, and composable with `jax.jit`, `jax.vmap`, and `jax.grad`.

---

## Quick Start

```python
import jax.numpy as jnp
from spectraldiffx import FourierGrid1D, SpectralDerivative1D

# Create a 1D Fourier grid
grid = FourierGrid1D.from_N_L(N=64, L=2 * jnp.pi)

# Create a spectral derivative operator
deriv = SpectralDerivative1D(grid=grid)

# Compute the derivative of sin(x)
x = grid.x
u = jnp.sin(x)
du_dx = deriv(u, order=1)  # ≈ cos(x)
```

---

## Navigation

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)**

    Install SpectralDiffX from GitHub or set up a development environment.

- :material-rocket-launch: **[Quick Start](quickstart.md)**

    Get up and running with your first spectral computation in minutes.

- :material-book-open-variant: **[Theory](theory/index.md)**

    Rigorous mathematical background on Fourier, Chebyshev, and spherical harmonic methods.

- :material-code-braces: **[Examples](notebooks/demo_1d.py)**

    Worked examples including KdV solitons, 2D turbulence, and QG models.

- :material-api: **[API Reference](api/fourier/grids.md)**

    Complete API documentation for all public classes and functions.

</div>
