# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pseudospectral Differentiation - 2D
#
# This notebook demonstrates how to use the `spectraldiffx` library for
# computing derivatives using the pseudospectral (Fourier) method in 2D.

# %%
import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

jax.config.update("jax_enable_x64", True)
sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)

# %% [markdown]
# ## Define Test Function
#
# We use a 2D periodic function:
# $$
# u(x,y) = \cos \left(m_x \frac{2\pi}{L_x}x \right) \sin \left(m_y \frac{2\pi}{L_y}y \right)
# $$

# %%
def f(x, y, Lx, Ly, mx, my):
    return jnp.cos(mx * 2 * jnp.pi * x / Lx) * jnp.sin(my * 2 * jnp.pi * y / Ly)


# Define gradient functions using JAX autodiff
df_dx = jax.grad(f, argnums=0)
df_dy = jax.grad(f, argnums=1)

# Second order derivatives
d2f_dx2 = jax.grad(df_dx, argnums=0)
d2f_dy2 = jax.grad(df_dy, argnums=1)

# %% [markdown]
# ## Setup Grid Using New API
#
# We use `FourierGrid2D` to define the 2D computational domain.

# %%
from spectraldiffx._src.grid import FourierGrid2D
from spectraldiffx._src.operators import SpectralDerivative2D
from spectraldiffx._src.solvers import SpectralHelmholtzSolver2D

# Grid and function parameters
mx, my = 3, 2
Nx, Ny = 64, 64
Lx, Ly = 2 * math.pi, 2 * math.pi

# Create 2D grid using the new API
grid = FourierGrid2D.from_N_L(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dealias="2/3")

# Get meshgrid coordinates
X, Y = grid.X

# %%
print(f"Grid points: Nx={grid.Nx}, Ny={grid.Ny}")
print(f"Domain size: Lx={grid.Lx}, Ly={grid.Ly}")
print(f"Grid spacing: dx={grid.dx:.4f}, dy={grid.dy:.4f}")
print(f"Dealiasing: {grid.dealias}")

# %% [markdown]
# ## Compute Fields and Analytical Derivatives

# %%
# Helper to compute field values on grid
kernel = lambda x, y: f(x, y, Lx, Ly, mx, my)
kernel_grad_x = lambda x, y: df_dx(x, y, Lx, Ly, mx, my)
kernel_grad_y = lambda x, y: df_dy(x, y, Lx, Ly, mx, my)
kernel_grad2_x2 = lambda x, y: d2f_dx2(x, y, Lx, Ly, mx, my)
kernel_grad2_y2 = lambda x, y: d2f_dy2(x, y, Lx, Ly, mx, my)


def compute_on_grid(func, x, y):
    """Evaluate function on meshgrid."""
    return jax.vmap(lambda xi: jax.vmap(lambda yi: func(xi, yi))(y))(x)


# Note: grid.x and grid.y are 1D arrays
u = compute_on_grid(kernel, grid.x, grid.y)
dudx_analytical = compute_on_grid(kernel_grad_x, grid.x, grid.y)
dudy_analytical = compute_on_grid(kernel_grad_y, grid.x, grid.y)
d2udx2_analytical = compute_on_grid(kernel_grad2_x2, grid.x, grid.y)
d2udy2_analytical = compute_on_grid(kernel_grad2_y2, grid.x, grid.y)

# %%
fig, ax = plt.subplots(ncols=3, figsize=(10, 3))

ax[0].contourf(X.T, Y.T, u.T)
ax[1].contourf(X.T, Y.T, dudx_analytical.T)
ax[2].contourf(X.T, Y.T, dudy_analytical.T)

ax[0].set(title="$u(x,y)$", xlabel="$x$", ylabel="$y$")
ax[1].set(title=r"$\partial_x u(x,y)$", xlabel="$x$", ylabel="$y$")
ax[2].set(title=r"$\partial_y u(x,y)$", xlabel="$x$", ylabel="$y$")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Compute Derivatives Using SpectralDerivative2D

# %%
# Create the 2D derivative operator
deriv = SpectralDerivative2D(grid=grid)

# Compute gradient (returns tuple of du/dx, du/dy)
dudx_spectral, dudy_spectral = deriv.gradient(u)

# %%
fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(7, 8))

x_levels = np.linspace(dudx_analytical.min(), dudx_analytical.max(), 10)
y_levels = np.linspace(dudy_analytical.min(), dudy_analytical.max(), 10)

# Analytical
ax[0, 0].contourf(X.T, Y.T, dudx_analytical.T, levels=x_levels)
ax[0, 0].set(title=r"$\partial_x u$ (analytical)", xlabel="$x$", ylabel="$y$")
ax[0, 1].contourf(X.T, Y.T, dudy_analytical.T, levels=y_levels)
ax[0, 1].set(title=r"$\partial_y u$ (analytical)", xlabel="$x$", ylabel="$y$")

# Spectral
ax[1, 0].contourf(X.T, Y.T, dudx_spectral.T, levels=x_levels)
ax[1, 0].set(title=r"$\partial_x u$ (spectral)", xlabel="$x$", ylabel="$y$")
ax[1, 1].contourf(X.T, Y.T, dudy_spectral.T, levels=y_levels)
ax[1, 1].set(title=r"$\partial_y u$ (spectral)", xlabel="$x$", ylabel="$y$")

# Error
err_x = np.abs(dudx_analytical - dudx_spectral)
err_y = np.abs(dudy_analytical - dudy_spectral)
pts = ax[2, 0].contourf(X.T, Y.T, err_x.T, cmap="Reds")
plt.colorbar(pts, ax=ax[2, 0])
ax[2, 0].set(title=r"$|\partial_x u|$ error", xlabel="$x$", ylabel="$y$")
pts = ax[2, 1].contourf(X.T, Y.T, err_y.T, cmap="Reds")
plt.colorbar(pts, ax=ax[2, 1])
ax[2, 1].set(title=r"$|\partial_y u|$ error", xlabel="$x$", ylabel="$y$")

plt.tight_layout()
plt.show()

# %%
print(f"Max error in du/dx: {err_x.max():.2e}")
print(f"Max error in du/dy: {err_y.max():.2e}")

# %% [markdown]
# ## Laplacian
#
# The Laplacian operator $\nabla^2 u = \partial^2 u/\partial x^2 + \partial^2 u/\partial y^2$
# is computed efficiently in spectral space as $-|k|^2 \hat{u}$.

# %%
# Compute Laplacian
laplacian_spectral = deriv.laplacian(u)
laplacian_analytical = d2udx2_analytical + d2udy2_analytical

# %%
fig, ax = plt.subplots(ncols=3, figsize=(10, 3))

levels = np.linspace(laplacian_analytical.min(), laplacian_analytical.max(), 10)

ax[0].contourf(X.T, Y.T, laplacian_analytical.T, levels=levels)
ax[0].set(title=r"$\nabla^2 u$ (analytical)", xlabel="$x$", ylabel="$y$")

ax[1].contourf(X.T, Y.T, laplacian_spectral.T, levels=levels)
ax[1].set(title=r"$\nabla^2 u$ (spectral)", xlabel="$x$", ylabel="$y$")

err_lap = np.abs(laplacian_analytical - laplacian_spectral)
pts = ax[2].contourf(X.T, Y.T, err_lap.T, cmap="Reds")
plt.colorbar(pts, ax=ax[2])
ax[2].set(title=r"$|\nabla^2 u|$ error", xlabel="$x$", ylabel="$y$")

plt.tight_layout()
plt.show()

print(f"Max error in Laplacian: {err_lap.max():.2e}")

# %% [markdown]
# ## Divergence and Curl
#
# For vector fields, we can compute divergence and curl.

# %%
# Create a vector field: V = (u, -u) which has zero divergence
vx = u
vy = -u

# Compute divergence: div(V) = du/dx + dv/dy
div_V = deriv.divergence(vx, vy)

# Compute curl (2D scalar): curl(V) = dv/dx - du/dy
curl_V = deriv.curl(vx, vy)

# %%
fig, ax = plt.subplots(ncols=2, figsize=(8, 3))

pts = ax[0].contourf(X.T, Y.T, div_V.T)
plt.colorbar(pts, ax=ax[0])
ax[0].set(title=r"$\nabla \cdot V$", xlabel="$x$", ylabel="$y$")

pts = ax[1].contourf(X.T, Y.T, curl_V.T)
plt.colorbar(pts, ax=ax[1])
ax[1].set(title=r"$\nabla \times V$", xlabel="$x$", ylabel="$y$")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Poisson Solver
#
# Solve $\nabla^2 \phi = f$ for $\phi$ given source term $f$.

# %%
# Create solver
solver = SpectralHelmholtzSolver2D(grid=grid)

# Use the Laplacian of u as source term, should recover u (up to a constant)
source = laplacian_analytical

# Solve Poisson equation
phi = solver.solve(source, alpha=0.0, zero_mean=True)

# The solution should match u (shifted to zero mean)
u_zero_mean = u - u.mean()
phi_normalized = phi - phi.mean()

# %%
fig, ax = plt.subplots(ncols=3, figsize=(10, 3))

ax[0].contourf(X.T, Y.T, source.T)
ax[0].set(title=r"Source $f = \nabla^2 u$", xlabel="$x$", ylabel="$y$")

ax[1].contourf(X.T, Y.T, phi.T)
ax[1].set(title=r"Solution $\phi$", xlabel="$x$", ylabel="$y$")

ax[2].contourf(X.T, Y.T, u_zero_mean.T)
ax[2].set(title=r"Original $u$ (zero mean)", xlabel="$x$", ylabel="$y$")

plt.tight_layout()
plt.show()

# %%
err_poisson = np.abs(phi_normalized - u_zero_mean)
print(f"Max error in Poisson solution: {err_poisson.max():.2e}")

# %% [markdown]
# ## Dealiasing Filter

# %%
# View the 2D dealiasing filter
dealias_filter = grid.dealias_filter()

fig, ax = plt.subplots(figsize=(5, 4))
KX, KY = grid.KX
pts = ax.contourf(KX.T, KY.T, dealias_filter.T)
plt.colorbar(pts)
ax.set(xlabel="$k_x$", ylabel="$k_y$", title="2/3 Dealiasing Filter")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Advection Term
#
# For fluid dynamics, we often need to compute the advection term $(v \cdot \nabla) q$.

# %%
# Create velocity field
vx = jnp.sin(2 * jnp.pi * X / Lx)
vy = jnp.cos(2 * jnp.pi * Y / Ly)

# Scalar field to advect
q = u

# Compute advection term
advection = deriv.advection_scalar(vx, vy, q)

# %%
fig, ax = plt.subplots(ncols=3, figsize=(10, 3))

ax[0].contourf(X.T, Y.T, vx.T)
ax[0].set(title=r"$v_x$", xlabel="$x$", ylabel="$y$")

ax[1].contourf(X.T, Y.T, vy.T)
ax[1].set(title=r"$v_y$", xlabel="$x$", ylabel="$y$")

ax[2].contourf(X.T, Y.T, advection.T)
ax[2].set(title=r"$(v \cdot \nabla) q$", xlabel="$x$", ylabel="$y$")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# The 2D API provides:
#
# 1. **FourierGrid2D**: 2D grid with wavenumber meshgrids
#    - Properties: `x`, `y`, `X` (meshgrid), `kx`, `ky`, `KX` (wavenumber meshgrid), `K2`
#    - Methods: `transform()`, `dealias_filter()`, `check_consistency()`
#
# 2. **SpectralDerivative2D**: 2D derivative operators
#    - `gradient(u)`: Returns (du/dx, du/dy)
#    - `divergence(vx, vy)`: Computes div(V)
#    - `curl(vx, vy)`: Computes 2D curl (scalar)
#    - `laplacian(u)`: Computes nabla^2 u
#    - `advection_scalar(vx, vy, q)`: Computes (v . grad) q
#    - `apply_dealias(u)`: Apply dealiasing filter
#    - `project_vector(vx, vy)`: Leray projection for incompressible flows
#
# 3. **SpectralHelmholtzSolver2D**: Solve elliptic equations
#    - `solve(f, alpha)`: Solves (nabla^2 - alpha) phi = f
