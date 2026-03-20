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
# # 2D Pseudospectral Operators
#
# ## Introduction
#
# This notebook extends the 1D pseudospectral method to **two dimensions** on a
# doubly-periodic rectangular domain $[0, L_x) \times [0, L_y)$. All the same
# spectral accuracy guarantees hold: for smooth periodic functions, derivatives
# are computed to machine precision.
#
# In 2D, the Fourier expansion of a field $u(x, y)$ is:
#
# $$
# u(x, y) = \sum_{k_x} \sum_{k_y} \hat{u}_{k_x, k_y} \, e^{i(k_x x + k_y y)}
# $$
#
# Partial derivatives become multiplications in spectral space:
#
# $$
# \frac{\partial u}{\partial x} \leftrightarrow i k_x \hat{u}, \qquad
# \frac{\partial u}{\partial y} \leftrightarrow i k_y \hat{u}
# $$
#
# The `spectraldiffx` 2D API provides a full catalog of differential operators:
#
# | Operator | Formula | Method |
# |---|---|---|
# | Gradient | $\nabla u = (\partial_x u, \partial_y u)$ | `deriv.gradient(u)` |
# | Divergence | $\nabla \cdot \mathbf{v} = \partial_x v_x + \partial_y v_y$ | `deriv.divergence(vx, vy)` |
# | Curl (2D) | $\nabla \times \mathbf{v} = \partial_x v_y - \partial_y v_x$ | `deriv.curl(vx, vy)` |
# | Laplacian | $\nabla^2 u = \partial_{xx} u + \partial_{yy} u$ | `deriv.laplacian(u)` |
# | Advection | $(\mathbf{v} \cdot \nabla) q$ | `deriv.advection_scalar(vx, vy, q)` |
# | Poisson solve | $\nabla^2 \psi = f$ | `solver.solve(f)` |

# %%
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from spectraldiffx import (
    FourierGrid2D,
    SpectralDerivative2D,
    SpectralFilter2D,
    SpectralHelmholtzSolver2D,
)

matplotlib.use("Agg")

jax.config.update("jax_enable_x64", True)
sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)

IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "images" / "demo_2d"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Grid Setup
#
# The `FourierGrid2D` creates a doubly-periodic rectangular grid with $N_x \times N_y$
# points. The grid provides:
# - Physical coordinates: `grid.x` ($N_x$), `grid.y` ($N_y$), `grid.X` (meshgrid tuple)
# - Wavenumbers: `grid.kx` ($N_x$), `grid.ky` ($N_y$), `grid.KX` (meshgrid tuple)
# - Combined wavenumber magnitude: `grid.K2` $= k_x^2 + k_y^2$ ($N_y \times N_x$)
#
# ```
# 2D periodic grid (Ny x Nx points):
#
#   y ^
# Ly  |--o--o--o--o--o--|
#     |  .  .  .  .  .  |
#     |  .  .  .  .  .  |     Field shape: [Ny, Nx]
#     |  .  .  .  .  .  |     Indexing: u[j, i] = u(x_i, y_j)
#  0  |--o--o--o--o--o--|
#     0                 Lx --> x
# ```

# %%
# Grid and function parameters
mx, my = 3, 2   # Mode numbers in x and y directions
Nx, Ny = 64, 64  # Grid resolution (powers of 2 for efficient FFT)
Lx, Ly = 2 * math.pi, 2 * math.pi  # Domain lengths

# Why these values?
# - mx=3, my=2: low modes, well-resolved on 64-point grids
# - Nx=Ny=64: enough resolution for spectral accuracy; N/3 ~ 21 > max(mx, my)
# - Lx=Ly=2*pi: convenient domain where wavenumber k_n = n

# Create the 2D Fourier grid with 2/3-rule dealiasing
grid = FourierGrid2D.from_N_L(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dealias="2/3")

# Physical coordinates
X, Y = grid.X  # X: [Ny, Nx], Y: [Ny, Nx] -- meshgrid arrays

# Wavenumber grids
KX, KY = grid.KX  # KX: [Ny, Nx], KY: [Ny, Nx] -- wavenumber meshgrids
K2 = grid.K2       # K2: [Ny, Nx] -- kx^2 + ky^2

print(f"Grid: Nx={grid.Nx}, Ny={grid.Ny}")
print(f"Domain: Lx={grid.Lx:.4f}, Ly={grid.Ly:.4f}")
print(f"Spacing: dx={grid.dx:.6f}, dy={grid.dy:.6f}")
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"KX shape: {KX.shape}")
print(f"K2 shape: {K2.shape}")
print(f"K2 = kx^2 + ky^2, max value: {float(K2.max()):.2f}")

# %% [markdown]
# ## 2. Test Function
#
# We use a separable 2D Fourier mode:
#
# $$
# u(x, y) = \cos\!\left(m_x \frac{2\pi}{L_x} x\right) \cdot \sin\!\left(m_y \frac{2\pi}{L_y} y\right)
# $$
#
# This function is smooth and periodic in both directions. Its analytical
# derivatives are straightforward to compute.
#
# **Analytical gradient** (using the product rule):
#
# $$
# \frac{\partial u}{\partial x} = -k_x \sin(k_x x) \sin(k_y y), \qquad
# \frac{\partial u}{\partial y} = k_y \cos(k_x x) \cos(k_y y)
# $$
#
# where $k_x = m_x \cdot 2\pi / L_x$ and $k_y = m_y \cdot 2\pi / L_y$.
#
# **Analytical Laplacian**:
#
# $$
# \nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}
# = -(k_x^2 + k_y^2) \cos(k_x x) \sin(k_y y) = -(k_x^2 + k_y^2) \, u
# $$

# %%
# Effective wavenumbers
kx = mx * 2 * jnp.pi / Lx  # kx = mx for Lx = 2*pi
ky = my * 2 * jnp.pi / Ly  # ky = my for Ly = 2*pi

print(f"Mode numbers: mx={mx}, my={my}")
print(f"Wavenumbers: kx={float(kx):.4f}, ky={float(ky):.4f}")

# %%
# Evaluate the test function on the meshgrid
u = jnp.cos(kx * X) * jnp.sin(ky * Y)  # u: [Ny, Nx]

# Analytical derivatives (closed-form on the meshgrid)
dudx_analytical = -kx * jnp.sin(kx * X) * jnp.sin(ky * Y)  # dudx: [Ny, Nx]
dudy_analytical = ky * jnp.cos(kx * X) * jnp.cos(ky * Y)   # dudy: [Ny, Nx]

# Analytical second derivatives
d2udx2_analytical = -(kx**2) * jnp.cos(kx * X) * jnp.sin(ky * Y)  # [Ny, Nx]
d2udy2_analytical = -(ky**2) * jnp.cos(kx * X) * jnp.sin(ky * Y)  # [Ny, Nx]
laplacian_analytical = d2udx2_analytical + d2udy2_analytical         # [Ny, Nx]

print(f"u shape: {u.shape}")
print(f"dudx_analytical shape: {dudx_analytical.shape}")
print(f"laplacian_analytical shape: {laplacian_analytical.shape}")

# %%
fig, ax = plt.subplots(ncols=3, figsize=(10, 3))

c0 = ax[0].contourf(X.T, Y.T, u.T)
plt.colorbar(c0, ax=ax[0])
ax[0].set(title=r"$u(x,y)$", xlabel="$x$", ylabel="$y$")

c1 = ax[1].contourf(X.T, Y.T, dudx_analytical.T)
plt.colorbar(c1, ax=ax[1])
ax[1].set(title=r"$\partial_x u$ (analytical)", xlabel="$x$", ylabel="$y$")

c2 = ax[2].contourf(X.T, Y.T, dudy_analytical.T)
plt.colorbar(c2, ax=ax[2])
ax[2].set(title=r"$\partial_y u$ (analytical)", xlabel="$x$", ylabel="$y$")

plt.tight_layout()
fig.savefig(IMG_DIR / "field_and_gradients.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Field and analytical gradients](../images/demo_2d/field_and_gradients.png)

# %% [markdown]
# ## 3. Gradient
#
# The spectral gradient computes both partial derivatives in one call:
#
# $$
# \nabla u = \left(\frac{\partial u}{\partial x}, \frac{\partial u}{\partial y}\right)
# \quad \leftrightarrow \quad
# (i k_x \hat{u}, \; i k_y \hat{u})
# $$
#
# We compare the spectral result against the closed-form analytical formulas.

# %%
# Create the 2D derivative operator
deriv = SpectralDerivative2D(grid=grid)

# Compute gradient: returns (du/dx, du/dy)
dudx_spectral, dudy_spectral = deriv.gradient(u)  # each: [Ny, Nx]

print(f"dudx_spectral shape: {dudx_spectral.shape}")
print(f"dudy_spectral shape: {dudy_spectral.shape}")

# %%
fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(7, 8))

x_levels = np.linspace(float(dudx_analytical.min()), float(dudx_analytical.max()), 10)
y_levels = np.linspace(float(dudy_analytical.min()), float(dudy_analytical.max()), 10)

# Row 1: Analytical
ax[0, 0].contourf(X.T, Y.T, dudx_analytical.T, levels=x_levels)
ax[0, 0].set(title=r"$\partial_x u$ (analytical)", xlabel="$x$", ylabel="$y$")
ax[0, 1].contourf(X.T, Y.T, dudy_analytical.T, levels=y_levels)
ax[0, 1].set(title=r"$\partial_y u$ (analytical)", xlabel="$x$", ylabel="$y$")

# Row 2: Spectral
ax[1, 0].contourf(X.T, Y.T, dudx_spectral.T, levels=x_levels)
ax[1, 0].set(title=r"$\partial_x u$ (spectral)", xlabel="$x$", ylabel="$y$")
ax[1, 1].contourf(X.T, Y.T, dudy_spectral.T, levels=y_levels)
ax[1, 1].set(title=r"$\partial_y u$ (spectral)", xlabel="$x$", ylabel="$y$")

# Row 3: Error
err_x = np.abs(dudx_analytical - dudx_spectral)
err_y = np.abs(dudy_analytical - dudy_spectral)
pts0 = ax[2, 0].contourf(X.T, Y.T, err_x.T, cmap="Reds")
plt.colorbar(pts0, ax=ax[2, 0])
ax[2, 0].set(title=r"$|\partial_x u|$ error", xlabel="$x$", ylabel="$y$")
pts1 = ax[2, 1].contourf(X.T, Y.T, err_y.T, cmap="Reds")
plt.colorbar(pts1, ax=ax[2, 1])
ax[2, 1].set(title=r"$|\partial_y u|$ error", xlabel="$x$", ylabel="$y$")

plt.tight_layout()
fig.savefig(IMG_DIR / "gradient_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Gradient comparison: analytical vs spectral](../images/demo_2d/gradient_comparison.png)

# %%
print(f"Max error in du/dx: {float(err_x.max()):.2e}")
print(f"Max error in du/dy: {float(err_y.max()):.2e}")
print(f"(Machine epsilon for float64: {jnp.finfo(jnp.float64).eps:.2e})")

# %% [markdown]
# ## 4. Laplacian
#
# The Laplacian in 2D is:
#
# $$
# \nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}
# \quad \leftrightarrow \quad -(k_x^2 + k_y^2) \hat{u} = -|k|^2 \hat{u}
# $$
#
# For our test function: $\nabla^2 u = -(k_x^2 + k_y^2) u$.

# %%
# Compute Laplacian
laplacian_spectral = deriv.laplacian(u)  # laplacian_spectral: [Ny, Nx]

print(f"laplacian_spectral shape: {laplacian_spectral.shape}")

# %%
fig, ax = plt.subplots(ncols=3, figsize=(10, 3))

levels = np.linspace(float(laplacian_analytical.min()), float(laplacian_analytical.max()), 10)

c0 = ax[0].contourf(X.T, Y.T, laplacian_analytical.T, levels=levels)
plt.colorbar(c0, ax=ax[0])
ax[0].set(title=r"$\nabla^2 u$ (analytical)", xlabel="$x$", ylabel="$y$")

c1 = ax[1].contourf(X.T, Y.T, laplacian_spectral.T, levels=levels)
plt.colorbar(c1, ax=ax[1])
ax[1].set(title=r"$\nabla^2 u$ (spectral)", xlabel="$x$", ylabel="$y$")

err_lap = np.abs(laplacian_analytical - laplacian_spectral)
pts = ax[2].contourf(X.T, Y.T, err_lap.T, cmap="Reds")
plt.colorbar(pts, ax=ax[2])
ax[2].set(title=r"$|\nabla^2 u|$ error", xlabel="$x$", ylabel="$y$")

plt.tight_layout()
fig.savefig(IMG_DIR / "laplacian_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Laplacian comparison](../images/demo_2d/laplacian_comparison.png)

# %%
print(f"Max error in Laplacian: {float(err_lap.max()):.2e}")

# %% [markdown]
# ## 5. Divergence and Curl
#
# For a 2D vector field $\mathbf{v} = (v_x, v_y)$:
#
# $$
# \text{div}(\mathbf{v}) = \frac{\partial v_x}{\partial x} + \frac{\partial v_y}{\partial y}
# $$
#
# $$
# \text{curl}(\mathbf{v}) = \frac{\partial v_y}{\partial x} - \frac{\partial v_x}{\partial y}
# \qquad \text{(scalar in 2D)}
# $$
#
# A fundamental vector identity is $\nabla \cdot (\nabla \times \mathbf{F}) = 0$:
# the divergence of a curl is always zero. We can verify this numerically by
# constructing a vector field as the curl of a scalar potential.
#
# Let $\psi(x,y) = u(x,y)$ be a stream function. Then
# $\mathbf{v} = \nabla^\perp \psi = (-\partial_y \psi, \partial_x \psi)$
# is automatically divergence-free:
# $\nabla \cdot \mathbf{v} = -\partial_{xy} \psi + \partial_{xy} \psi = 0$.

# %%
# Construct a divergence-free vector field from a stream function
# v = (-du/dy, du/dx) -- guaranteed div(v) = 0
vx = -dudy_analytical   # vx: [Ny, Nx]
vy = dudx_analytical     # vy: [Ny, Nx]

# Compute divergence and curl
div_v = deriv.divergence(vx, vy)  # div_v: [Ny, Nx] -- should be ~0
curl_v = deriv.curl(vx, vy)       # curl_v: [Ny, Nx]

print(f"div_v shape: {div_v.shape}")
print(f"curl_v shape: {curl_v.shape}")
print(f"Max |div(v)|: {float(jnp.abs(div_v).max()):.2e} (should be ~0)")
print(f"Max |curl(v)|: {float(jnp.abs(curl_v).max()):.4f}")

# %%
fig, ax = plt.subplots(ncols=2, figsize=(8, 3))

pts0 = ax[0].contourf(X.T, Y.T, div_v.T, cmap="RdBu_r")
plt.colorbar(pts0, ax=ax[0])
ax[0].set(title=r"$\nabla \cdot \mathbf{v}$ (should be $\approx 0$)", xlabel="$x$", ylabel="$y$")

pts1 = ax[1].contourf(X.T, Y.T, curl_v.T)
plt.colorbar(pts1, ax=ax[1])
ax[1].set(title=r"$\nabla \times \mathbf{v}$ (vorticity)", xlabel="$x$", ylabel="$y$")

plt.tight_layout()
fig.savefig(IMG_DIR / "divergence_curl.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Divergence and curl](../images/demo_2d/divergence_curl.png)
#
# The divergence is at machine precision ($\sim 10^{-14}$), confirming both the
# vector identity and the accuracy of the spectral operators.

# %% [markdown]
# ## 6. Poisson Solver
#
# The **Poisson equation** is a fundamental elliptic PDE:
#
# $$
# \nabla^2 \psi = f
# $$
#
# In spectral space, this becomes an algebraic equation:
#
# $$
# -(k_x^2 + k_y^2) \hat{\psi}_{k_x, k_y} = \hat{f}_{k_x, k_y}
# \quad \Rightarrow \quad
# \hat{\psi} = -\frac{\hat{f}}{|k|^2}
# $$
#
# The `SpectralHelmholtzSolver2D` solves the more general **Helmholtz equation**
# $(\nabla^2 - \alpha) \psi = f$. Setting $\alpha = 0$ gives the Poisson solver.
#
# **Test**: We use $f = \nabla^2 u$ as the source term. The solver should recover
# $u$ (up to a constant, since the Poisson equation has a free additive constant
# for the $k = 0$ mode).

# %%
# Create the Helmholtz solver
solver = SpectralHelmholtzSolver2D(grid=grid)

# Source term: the Laplacian of our test function
source = laplacian_analytical  # source: [Ny, Nx]

# Solve: nabla^2 psi = source, with zero_mean=True to fix the constant
psi = solver.solve(source, alpha=0.0, zero_mean=True)  # psi: [Ny, Nx]

# Compare to original field (shifted to zero mean for fair comparison)
u_zero_mean = u - u.mean()
psi_zero_mean = psi - psi.mean()

print(f"psi shape: {psi.shape}")
print(f"Max |psi - u| (zero-mean): {float(jnp.abs(psi_zero_mean - u_zero_mean).max()):.2e}")

# %%
fig, ax = plt.subplots(ncols=3, figsize=(10, 3))

c0 = ax[0].contourf(X.T, Y.T, source.T)
plt.colorbar(c0, ax=ax[0])
ax[0].set(title=r"Source $f = \nabla^2 u$", xlabel="$x$", ylabel="$y$")

c1 = ax[1].contourf(X.T, Y.T, psi.T)
plt.colorbar(c1, ax=ax[1])
ax[1].set(title=r"Solution $\psi$", xlabel="$x$", ylabel="$y$")

c2 = ax[2].contourf(X.T, Y.T, u_zero_mean.T)
plt.colorbar(c2, ax=ax[2])
ax[2].set(title=r"Original $u$ (zero mean)", xlabel="$x$", ylabel="$y$")

plt.tight_layout()
fig.savefig(IMG_DIR / "poisson_solver.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Poisson solver results](../images/demo_2d/poisson_solver.png)

# %%
err_poisson = np.abs(psi_zero_mean - u_zero_mean)
print(f"Max error in Poisson solution: {float(err_poisson.max()):.2e}")

# %% [markdown]
# ## 7. Advection
#
# In fluid dynamics and transport problems, we frequently need the **advection
# term** $(\mathbf{v} \cdot \nabla) q$, which describes how a velocity field
# $\mathbf{v} = (v_x, v_y)$ transports a scalar field $q$:
#
# $$
# (\mathbf{v} \cdot \nabla) q = v_x \frac{\partial q}{\partial x} + v_y \frac{\partial q}{\partial y}
# $$
#
# The `advection_scalar` method computes the gradients of $q$ in spectral space
# (with dealiasing), then multiplies by the velocity components in physical
# space. This is the standard pseudospectral approach for nonlinear terms.

# %%
# Create a velocity field
vx_adv = jnp.sin(2 * jnp.pi * X / Lx)  # vx: [Ny, Nx] -- x-component
vy_adv = jnp.cos(2 * jnp.pi * Y / Ly)  # vy: [Ny, Nx] -- y-component

# Scalar field to advect
q = u  # q: [Ny, Nx]

# Compute advection: (v . grad) q
advection = deriv.advection_scalar(vx_adv, vy_adv, q)  # advection: [Ny, Nx]

print(f"vx shape: {vx_adv.shape}")
print(f"vy shape: {vy_adv.shape}")
print(f"q shape: {q.shape}")
print(f"advection shape: {advection.shape}")
print(f"advection range: [{float(advection.min()):.4f}, {float(advection.max()):.4f}]")

# %%
fig, ax = plt.subplots(ncols=3, figsize=(10, 3))

c0 = ax[0].contourf(X.T, Y.T, vx_adv.T)
plt.colorbar(c0, ax=ax[0])
ax[0].set(title=r"$v_x = \sin(2\pi x / L_x)$", xlabel="$x$", ylabel="$y$")

c1 = ax[1].contourf(X.T, Y.T, vy_adv.T)
plt.colorbar(c1, ax=ax[1])
ax[1].set(title=r"$v_y = \cos(2\pi y / L_y)$", xlabel="$x$", ylabel="$y$")

c2 = ax[2].contourf(X.T, Y.T, advection.T)
plt.colorbar(c2, ax=ax[2])
ax[2].set(title=r"$(\mathbf{v} \cdot \nabla) q$", xlabel="$x$", ylabel="$y$")

plt.tight_layout()
fig.savefig(IMG_DIR / "advection.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Advection term](../images/demo_2d/advection.png)

# %% [markdown]
# ## 8. Dealiasing
#
# The 2D dealiasing filter is the outer product of 1D filters applied in each
# direction. It zeros out any mode where $|k_x| > \frac{2}{3} k_{x,\max}$ **or**
# $|k_y| > \frac{2}{3} k_{y,\max}$:
#
# $$
# D(k_x, k_y) = D_x(k_x) \cdot D_y(k_y)
# $$
#
# This creates a rectangular "window" in spectral space that retains only the
# well-resolved, alias-free modes.

# %%
# Get the 2D dealiasing filter
dealias_filter = grid.dealias_filter()  # dealias_filter: [Ny, Nx]

print(f"Dealias filter shape: {dealias_filter.shape}")
print(f"Retained modes: {int(dealias_filter.sum())} / {Nx * Ny}")
print(f"Fraction retained: {float(dealias_filter.sum()) / (Nx * Ny):.4f}")
print(f"(Expected ~ (2/3)^2 = {(2/3)**2:.4f})")

# %%
fig, ax = plt.subplots(figsize=(5, 4))

pts = ax.contourf(KX.T, KY.T, dealias_filter.T, levels=[0, 0.5, 1.0], cmap="RdYlGn")
plt.colorbar(pts, ax=ax, ticks=[0, 1], label="Retained (1) / Zeroed (0)")
ax.set(xlabel="$k_x$", ylabel="$k_y$",
       title="2/3 Dealiasing Filter in Spectral Space")
plt.tight_layout()
fig.savefig(IMG_DIR / "dealiasing_filter.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![2D dealiasing filter](../images/demo_2d/dealiasing_filter.png)
#
# The rectangular region in the center represents the retained modes. The
# zeroed-out modes along the edges would otherwise cause aliasing errors
# when computing nonlinear products.

# %% [markdown]
# ## Summary
#
# This tutorial demonstrated the full 2D pseudospectral operator toolkit in
# `spectraldiffx`:
#
# | Class / Method | Purpose |
# |---|---|
# | `FourierGrid2D.from_N_L(Nx, Ny, Lx, Ly, dealias)` | Create 2D periodic grid |
# | `grid.X` | Meshgrid tuple `(X, Y)`, each `[Ny, Nx]` |
# | `grid.KX` | Wavenumber meshgrid `(KX, KY)`, each `[Ny, Nx]` |
# | `grid.K2` | $k_x^2 + k_y^2$ array `[Ny, Nx]` |
# | `grid.dealias_filter()` | 2D dealiasing mask `[Ny, Nx]` |
# | `SpectralDerivative2D(grid)` | 2D derivative operator |
# | `deriv.gradient(u)` | $\nabla u = (\partial_x u, \partial_y u)$ |
# | `deriv.divergence(vx, vy)` | $\nabla \cdot \mathbf{v}$ |
# | `deriv.curl(vx, vy)` | $\nabla \times \mathbf{v}$ (scalar) |
# | `deriv.laplacian(u)` | $\nabla^2 u$ |
# | `deriv.advection_scalar(vx, vy, q)` | $(\mathbf{v} \cdot \nabla) q$ |
# | `deriv.apply_dealias(u)` | Apply 2/3-rule filter |
# | `deriv.project_vector(vx, vy)` | Leray projection (divergence-free) |
# | `SpectralHelmholtzSolver2D(grid)` | Elliptic solver |
# | `solver.solve(f, alpha, zero_mean)` | Solve $(\nabla^2 - \alpha)\psi = f$ |
# | `SpectralFilter2D(grid)` | Smoothing / hyperviscosity filters |
#
# Key takeaways:
# - All operators achieve **machine-precision** accuracy for resolved modes.
# - The gradient comparison uses closed-form analytical derivatives (not `jax.grad`) to avoid shape-mismatch pitfalls on meshgrids.
# - The divergence of a curl-derived vector field is verified to be zero to machine precision.
# - The Poisson solver recovers the original field from its Laplacian (up to a constant).
# - The 2D dealiasing filter retains roughly $(2/3)^2 \approx 44\%$ of modes.
