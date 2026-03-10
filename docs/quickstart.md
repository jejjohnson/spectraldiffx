# Quick Start

This page walks through the most common use cases.

## Enable 64-bit precision

JAX defaults to 32-bit floats. For spectral methods the extra precision matters:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

---

## 1D Fourier Differentiation

Compute derivatives of a smooth periodic function on a uniform grid.

```python
import jax.numpy as jnp
from spectraldiffx import FourierGrid1D, SpectralDerivative1D

# Build grid: N points on [0, 2π)
grid = FourierGrid1D.from_N_L(N=64, L=2 * jnp.pi)
deriv = SpectralDerivative1D(grid=grid)

x = grid.x
u = jnp.sin(2 * x) + 0.5 * jnp.cos(5 * x)

# First derivative
du_dx = deriv(u, order=1)

# Second derivative
d2u_dx2 = deriv(u, order=2)
```

Spectral accuracy means machine-precision errors for smooth periodic functions — the error does not grow with the derivative order.

---

## 2D Fourier Operations

```python
import jax.numpy as jnp
from spectraldiffx import FourierGrid2D, SpectralDerivative2D

grid = FourierGrid2D.from_N_L(Nx=64, Ny=64, Lx=2 * jnp.pi, Ly=2 * jnp.pi)
deriv = SpectralDerivative2D(grid=grid)

X, Y = grid.X
u = jnp.sin(X) * jnp.cos(Y)

# Partial derivatives
du_dx, du_dy = deriv.gradient(u)

# Laplacian
lap_u = deriv.laplacian(u)  # should equal -2 * sin(x) * cos(y)
```

---

## Chebyshev Differentiation

For non-periodic problems on $[-1, 1]$, use the Chebyshev basis:

```python
import jax.numpy as jnp
from spectraldiffx import ChebyshevGrid1D, ChebyshevDerivative1D

# Chebyshev-Gauss-Lobatto nodes on [-1, 1]
grid = ChebyshevGrid1D(N=32)
deriv = ChebyshevDerivative1D(grid=grid)

x = grid.x
u = jnp.exp(x)          # smooth, non-periodic

du_dx = deriv(u, order=1)   # ≈ exp(x)
```

Nodes cluster near the boundaries, which controls the Runge phenomenon and enables high accuracy without the need for a periodic extension.

---

## Spectral Filtering

Suppress aliasing or high-frequency noise with a spectral filter:

```python
from spectraldiffx import FourierGrid1D, SpectralFilter1D

grid = FourierGrid1D.from_N_L(N=128, L=2 * jnp.pi)
filt = SpectralFilter1D(grid=grid, filter_type="exponential", order=8)

u_filtered = filt(u)
```

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **`FourierGrid*D`** | Uniform periodic grids; stores wavenumbers and transform helpers |
| **`SpectralDerivative*D`** | Differentiation via multiplication in spectral space |
| **`SpectralFilter*D`** | Spectral-space low-pass filters (exponential, raised cosine) |
| **`SpectralHelmholtzSolver*D`** | Solve $(\nabla^2 - \alpha)u = f$ in spectral space |
| **`ChebyshevGrid*D`** | Gauss-Lobatto nodes on $[-1,1]$; DCT-based transforms |
| **`SphericalGrid*D`** | Gauss-Legendre grids for spherical harmonic transforms |

All objects are [Equinox](https://github.com/patrick-kidger/equinox) modules, which means they are JAX pytrees and work seamlessly inside `jax.jit`, `jax.vmap`, and `jax.grad`.
