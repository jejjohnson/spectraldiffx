# SpectralDiffX

<a href="https://www.codefactor.io/repository/github/jejjohnson/spectraldiffx"><img src="https://www.codefactor.io/repository/github/jejjohnson/spectraldiffx/badge"></a>
<a href="https://codecov.io/gh/jejjohnson/spectraldiffx"><img src="https://codecov.io/gh/jejjohnson/spectraldiffx/branch/main/graph/badge.svg?token=YGPQQEAK91"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg"></a>
<a href="https://github.com/jejjohnson/spectraldiffx"><img src="https://img.shields.io/badge/status-pre--alpha-orange"></a>

> Pseudospectral differentiation, filtering, and PDE solvers in <a href="https://github.com/google/jax">JAX</a> — fully differentiable and JIT-compatible.

**SpectralDiffX** provides composable building blocks for pseudospectral methods built on JAX and <a href="https://github.com/patrick-kidger/equinox">Equinox</a>. It supports **Fourier**, **Chebyshev**, and **spherical harmonic** bases for grids, derivatives, spectral filters, and Helmholtz/Poisson solvers in 1D, 2D, and 3D.

---

## Features

| Module | Components | Dimensions |
|--------|-----------|------------|
| **Fourier** | `FourierGrid` | 1D, 2D, 3D |
| **Spectral Operators** | `SpectralDerivative`, `SpectralFilter`, `SpectralHelmholtzSolver` | 1D, 2D, 3D |
| **Chebyshev** | `ChebyshevGrid`, `ChebyshevDerivative`, `ChebyshevFilter`, `ChebyshevHelmholtzSolver` | 1D, 2D |
| **Spherical Harmonics** | `SphericalGrid`, `SphericalDerivative`, `SphericalFilter`, `SphericalHarmonicTransform`, `SphericalHelmholtzSolver`, `SphericalPoissonSolver` | 1D, 2D |

All classes are Equinox modules, meaning they are:
- ✅ Pytree-compatible (works with `jax.jit`, `jax.vmap`, `jax.grad`)
- ✅ Immutable and functional
- ✅ Composable with other JAX/Equinox code

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/jejjohnson/spectraldiffx
```

### Development Setup

We use <a href="https://docs.astral.sh/uv/">uv</a> for dependency management:

```bash
git clone https://github.com/jejjohnson/spectraldiffx.git
cd spectraldiffx
uv sync --group dev
```

To install with all optional dependencies (tests, docs, experiments, examples):

```bash
uv sync --all-extras
```

> **Note:** Requires Python ≥ 3.12, < 3.14.

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

## References

**Software**

* <a href="https://github.com/ASEM000/kernex">kernex</a> – differentiable stencils in JAX
* <a href="https://github.com/ASEM000/finitediffX">FiniteDiffX</a> – finite difference tools in JAX
* <a href="https://github.com/patrick-kidger/equinox">Equinox</a> – JAX neural network library
* <a href="https://github.com/patrick-kidger/diffrax">Diffrax</a> – differential equation solvers in JAX

**Algorithms**

* <a href="https://egusphere.copernicus.org/preprints/2023/egusphere-2023-1715/">Thiry et al, 2023</a> | <a href="https://github.com/louity/MQGeometry">MQGeometry 1.0</a> – WENO reconstructions applied to the multilayer Quasi-Geostrophic equations

---

## License

<a>MIT</a> © J Emmanuel Johnson
