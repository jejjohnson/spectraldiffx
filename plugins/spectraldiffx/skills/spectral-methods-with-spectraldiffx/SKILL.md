---
name: spectral-methods-with-spectraldiffx
description: Write pseudospectral code in JAX on spectraldiffx — Fourier, Chebyshev and spherical-harmonic grids and wavenumbers, spectral derivatives (gradient, Laplacian, curl, Jacobian, advection) with correct 2/3 dealiasing, DCT / DST transforms (types I–IV), Poisson / Helmholtz solvers with periodic, Dirichlet, Neumann and mixed per-axis boundary conditions, masked-domain (land-mask) capacitance solves, Chebyshev collocation BVPs, spherical vorticity inversion and spectral filters. Use whenever a task differentiates a field spectrally, solves an elliptic equation on a rectangle, basin or sphere, takes a DCT / DST, or dealiases a nonlinear term, in a project that uses (or could use) spectraldiffx.
---

# Pseudospectral methods on spectraldiffx

spectraldiffx turns a grid into an Equinox module that owns its
wavenumbers and dealiasing mask, a derivative into a method on an operator
built from that grid, and a Poisson / Helmholtz solve into one call whose
boundary conditions pick the right transform (FFT, DST, DCT). Everything is
a pytree or a pure function, so it runs under `jit`, `vmap` and `grad`.
Before writing `fftfreq`, a 2/3 mask, a DCT through an FFT trick or a
divide-by-`k²` solve, look it up:

1. **The capability index** lists every public name with a one-line
   summary, grouped like the API reference, plus gaussx:
   <https://jejjohnson.github.io/spectraldiffx/api/capabilities/>. Or list
   the installed version:

   ```python
   import inspect

   import spectraldiffx

   for name in spectraldiffx.__all__:
       doc = (inspect.getdoc(getattr(spectraldiffx, name)) or "").split("\n")[0]
       print(f"spectraldiffx.{name}: {doc}")
   ```

2. Compose what exists. Neither spectraldiffx nor gaussx is on PyPI yet;
   install both from GitHub, gaussx at the tag spectraldiffx pins:
   `uv add "gaussx @ git+https://github.com/jejjohnson/gaussx.git@v0.1.0"
   "spectraldiffx @ git+https://github.com/jejjohnson/spectraldiffx.git"`.

## What lives where

Everything is importable from the top level (`import spectraldiffx as sdx`).

| You need… | Use |
|---|---|
| A periodic grid, its wavenumbers and 2/3 mask | `sdx.FourierGrid1D/2D/3D` (`.from_N_L`, `.x`, `.X`, `.k`, `.KX`, `.K2`, `.dealias_filter()`) |
| Derivatives on a periodic domain | `sdx.SpectralDerivative1D/2D/3D` (`gradient`, `laplacian`, `divergence`, `curl`, `biharmonic`, `hyperviscosity`, `inverse_laplacian`, `velocity_from_streamfunction`, `project_vector`) |
| A dealiased nonlinear term | `deriv.jacobian(f, g)`, `deriv.advection_scalar(u, v, q)`, or `deriv.apply_dealias(product)` |
| DCT / DST (scipy conventions, types I–IV, `norm=None` or `"ortho"`) | `sdx.dct`, `sdx.dst`, `sdx.idct`, `sdx.idst`; along axes `sdx.dctn`, `sdx.dstn`, `sdx.idctn`, `sdx.idstn` |
| 1-D Laplacian eigenvalues (FD2 or continuous) | `sdx.dst1_eigenvalues` … `sdx.fft_eigenvalues`; `sdx.dst1_eigenvalues_ps` … `sdx.fft_eigenvalues_ps` |
| Poisson / Helmholtz `(∇² − λ)ψ = f` on a rectangle, any BC per axis | `sdx.solve_helmholtz_2d` / `sdx.solve_helmholtz_3d` (`bc_x`, `bc_y`, `bc_z` ∈ `"periodic"`, `"dirichlet"`, `"dirichlet_stag"`, `"neumann"`, `"neumann_stag"` or a mixed `(left, right)` pair), `sdx.solve_poisson_2d`; as modules `sdx.MixedBCHelmholtzSolver2D/3D` |
| One BC everywhere | `sdx.solve_helmholtz_fft` / `sdx.solve_helmholtz_dst` / `sdx.solve_helmholtz_dct` (and the `_1d`, `_3d`, `dst2`, `dct1` variants); periodic modules `sdx.SpectralHelmholtzSolver1D/2D/3D` |
| Non-zero boundary values | `bc_x_values=` / `bc_y_values=` on `solve_helmholtz_2d`, or `sdx.modify_rhs_1d/2d/3d` |
| A basin with a land mask | `sdx.build_capacitance_solver(mask, dx, dy, lambda_, base_bc)` → a callable `sdx.CapacitanceSolver` |
| Non-periodic, high-order (Chebyshev) | `sdx.ChebyshevGrid1D/2D/3D`, `sdx.ChebyshevDerivative1D/2D/3D`, `sdx.ChebyshevHelmholtzSolver1D/2D`, `sdx.ChebyshevPoissonSolver1D/2D`, `sdx.clenshaw_curtis_weights`, `sdx.cheb_dealias_product` |
| The sphere | `sdx.SphericalGrid2D`, `sdx.SphericalHarmonicTransform`, `sdx.SphericalDerivative2D`, `sdx.SphericalHelmholtzSolver`, `sdx.SphericalVorticityInversionSolver`, `sdx.SphericalHelmholtzDecomposition` |
| Damping the grid scale | `sdx.SpectralFilter1D/2D/3D`, `sdx.ChebyshevFilter1D/2D`, `sdx.SphericalFilter1D/2D` (`exponential_filter`, `hyperviscosity`) |

The structured linear algebra underneath (masked and diagonalised
operators, eigen-factorised shifted solves) lives in gaussx; finite-volume
operators on staggered grids live in finitevolX, which re-exports these
solvers.

## The rules your code must keep

- **Wavenumbers come from the grid.** `grid.k` is `2π·fftfreq(N, dx)` in
  FFT order; fields are `(Ny, Nx)` / `(Nz, Ny, Nx)` with x on the last
  axis. Fourier grids are `[0, L)`; Chebyshev grids are `[−L, L]` with `L`
  the *half*-length and decreasing nodes.
- **Dealias products, not linear operators.** Derivatives keep every
  resolved mode; a product of fields must be truncated (factors and
  product) with `jacobian`, `advection_scalar` or `apply_dealias`.
- **The boundary condition picks the transform.** Never solve a walled
  domain with an FFT: pass the BC per axis and let the solver choose
  FFT / DST / DCT. `"dirichlet"` means ψ = 0 one grid step outside the
  array (the array holds the interior points); `"dirichlet_stag"` /
  `"neumann_stag"` put the boundary half a cell outside.
- **Know which eigenvalues you get.** The `solve_*` functions use the
  second-order finite-difference eigenvalues by default (exact inverses of
  the 5-point Laplacian, second-order accurate against the PDE;
  `approximation="spectral"` on the fixed-BC ones for spectral accuracy);
  the `SpectralHelmholtzSolver*` classes use the continuous `k²`.
- **Null modes and resonance.** With `λ = 0` and periodic or Neumann
  boundaries everywhere the constant mode is undefined and its coefficient
  is set to zero; in the `solve_*` functions a `λ` equal to an eigenvalue
  raises (also under `jit`).
- **JAX rules.** Physical-space input is real (complex raises); BC, `type`,
  `norm` and `axes` arguments are static under `jit` (`λ` may be traced);
  turn on x64 for spectral accuracy; close over a module rather than
  passing it as a `jit` argument.

## Worked example

```python
import jax
import jax.numpy as jnp
import numpy as np

import spectraldiffx as sdx

jax.config.update("jax_enable_x64", True)  # spectral accuracy needs float64

# 1. A doubly periodic grid; fields are (Ny, Nx), x along the last axis
grid = sdx.FourierGrid2D.from_N_L(Nx=64, Ny=48, Lx=2 * jnp.pi, Ly=1.0)
X, Y = grid.X  # (Ny, Nx) each
psi = jnp.sin(3 * X) * jnp.cos(2 * jnp.pi * Y)  # streamfunction, (Ny, Nx)
deriv = sdx.SpectralDerivative2D(grid)

# 2. Derivatives are exact for resolved modes; linear operators never dealias
dpsi_dx, dpsi_dy = deriv.gradient(psi)  # (Ny, Nx) each
grad_err = jnp.max(jnp.abs(dpsi_dx - 3 * jnp.cos(3 * X) * jnp.cos(2 * jnp.pi * Y)))

# 3. Nonlinear terms go through the dealiased operators: (u·∇)ψ = 0
u, v = deriv.velocity_from_streamfunction(psi)  # u = −∂ψ/∂y, v = ∂ψ/∂x
self_advection = jnp.max(jnp.abs(deriv.advection_scalar(u, v, psi)))

# 4. Periodic Poisson ∇²ψ = ζ with the continuous k² (zero-mean gauge)
zeta = deriv.laplacian(psi)  # (Ny, Nx)
psi_rec = sdx.SpectralHelmholtzSolver2D(grid).solve(zeta, alpha=0.0)
poisson_err = jnp.max(jnp.abs(psi_rec - psi))

# 5. A channel: periodic in x, Dirichlet walls in y (FFT × DST-I, FD2 eigenvalues)
nx, ny = 64, 31
dx, dy = 1.0 / nx, 1.0 / (ny + 1)  # ψ = 0 on the walls y = 0 and y = 1
XC, YC = jnp.meshgrid(dx * jnp.arange(nx), dy * jnp.arange(1, ny + 1))  # (ny, nx)
psi_ch = jnp.sin(2 * jnp.pi * XC) * jnp.sin(jnp.pi * YC)  # interior points only
rhs = -((2 * jnp.pi) ** 2 + jnp.pi**2) * psi_ch  # the continuous ∇²ψ
psi_fd2 = sdx.solve_poisson_2d(rhs, dx, dy, bc_x="periodic", bc_y="dirichlet")
channel_err = jnp.max(jnp.abs(psi_fd2 - psi_ch))  # O(h²): the 5-point Laplacian


# 6. λ may be traced: differentiate through the solver
def energy(lam):
    sol = sdx.solve_helmholtz_2d(rhs, dx, dy, "periodic", "dirichlet", lam)
    return jnp.sum(sol**2)


d_energy = jax.grad(energy)(1.0)  # ∂(Σψ²)/∂λ, ()

# 7. A basin with a land mask: the capacitance solver (gaussx underneath)
j, i = np.mgrid[:40, :48]
mask = np.hypot(j - 19.5, i - 23.5) < 17.0  # True = water
basin = sdx.build_capacitance_solver(mask, dx=1.0, dy=1.0, lambda_=0.0, base_bc="fft")
f = jnp.ones(mask.shape)  # (40, 48)
psi_b = basin(f)  # ψ = 0 on land and on the coast
lap5 = (
    jnp.roll(psi_b, 1, 0)
    + jnp.roll(psi_b, -1, 0)
    + jnp.roll(psi_b, 1, 1)
    + jnp.roll(psi_b, -1, 1)
    - 4 * psi_b
)  # five-point Laplacian, dx = dy = 1
interior = jnp.zeros(mask.size, bool).at[basin.interior_indices].set(True)
interior = interior.reshape(mask.shape)  # (40, 48)
basin_residual = jnp.max(jnp.abs(jnp.where(interior, lap5 - f, 0.0)))

# 8. Chebyshev collocation on [−1, 1]: u″ = −π² sin(πx), u(±1) = 0
cgrid = sdx.ChebyshevGrid1D.from_N_L(N=32, L=1.0)
cheb = sdx.ChebyshevHelmholtzSolver1D(cgrid)
u_cheb = cheb.solve(-(jnp.pi**2) * jnp.sin(jnp.pi * cgrid.x), alpha=0.0)
cheb_err = jnp.max(jnp.abs(u_cheb - jnp.sin(jnp.pi * cgrid.x)))
```

With x64 on, the spectral derivative, the self-advection `(u·∇)ψ` and the
periodic Poisson round trip are all exact to round-off (errors around
1e-14); the channel solve is off by about 8e-4, exactly the second-order
error of the 5-point Laplacian it inverts; the gradient with respect to
`λ` is negative (a larger `λ` shrinks the solution); the basin solution
satisfies the five-point equation in every interior cell to about 1e-13;
and the Chebyshev solve recovers `sin(πx)` to about 4e-14 with 33 nodes.

## Don't write it — use spectraldiffx

| Don't write… | Use |
|---|---|
| `2 * jnp.pi * jnp.fft.fftfreq(N, dx)`, `KX**2 + KY**2` | `grid.k`, `grid.KX`, `grid.K2` |
| `jnp.fft.ifft(1j * k * jnp.fft.fft(u)).real` | `sdx.SpectralDerivative1D(grid).gradient(u)` |
| `jnp.where(jnp.abs(k) < k.max() * 2 / 3, ...)` | `grid.dealias_filter()` (the strict rule `3\|n\| < N`), `deriv.apply_dealias` |
| `u * dq_dx + v * dq_dy` in physical space | `deriv.advection_scalar(u, v, q)`, `deriv.jacobian(psi, q)` |
| `scipy.fft.dst` in JAX code, or a DCT through a hand-built FFT | `sdx.dst`, `sdx.dctn`, … (jit- and grad-compatible, float32-preserving) |
| `ifft2(fft2(f) / -K2)` with a hand-zeroed mean | `sdx.SpectralHelmholtzSolver2D(grid).solve(f)` or `sdx.solve_poisson_fft` |
| An FFT solve on a domain with walls | `sdx.solve_helmholtz_2d(f, dx, dy, bc_x=..., bc_y="dirichlet")` |
| A dense or iterative solve of the masked 5-point Laplacian | `sdx.build_capacitance_solver(mask, dx, dy)` |
| A Chebyshev differentiation matrix and boundary-row bookkeeping | `sdx.ChebyshevGrid1D(...).D`, `sdx.ChebyshevHelmholtzSolver1D` |
| A Legendre transform or vorticity inversion on the sphere | `sdx.SphericalHarmonicTransform`, `sdx.SphericalVorticityInversionSolver` |

## Self-check before you finish

- Every wavenumber comes from a grid, and every BC is passed by name, not
  emulated with a transform you picked.
- Every product of fields is dealiased; no linear operator is.
- A non-square grid with a different length per axis gives the same
  answer as your square test (no swapped `kx` / `ky`).
- `jax.jit` and `jax.grad` run through your code; with x64 off,
  everything runs in float32 (the transforms and the `solve_*` functions
  also keep a float32 input float32 with x64 on).

If spectraldiffx lacks what you need, keep your addition small and shaped
like spectraldiffx (a method on the derivative classes, a new entry in the
per-axis BC table, an `eqx.Module` holding its grid) and consider proposing
it upstream at <https://github.com/jejjohnson/spectraldiffx/issues>.
