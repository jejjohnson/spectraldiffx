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
# # Inhomogeneous Boundary Conditions
#
# By default, the spectral solvers enforce **homogeneous** BCs (zero values).
# This notebook shows how to solve with **non-zero** Dirichlet values and
# Neumann fluxes using the `bc_x_values` / `bc_y_values` parameters.
#
# We also demonstrate the 3D mixed-BC solver for physically motivated setups.
#
# ## Key concepts
#
# - **Inhomogeneous Dirichlet:** prescribe non-zero psi on the boundary
# - **Inhomogeneous Neumann:** prescribe non-zero dpsi/dn on the boundary
# - **RHS modification:** the solver modifies the RHS internally using FD2 stencil formulas
# - **3D mixed BCs:** different BC types on each axis (e.g., periodic x/y, Neumann z)

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
jax.config.update("jax_enable_x64", True)

IMGDIR = Path("docs/images/inhomogeneous_bcs")
IMGDIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Heated Plate: Inhomogeneous Dirichlet
#
# Solve the Poisson equation on a unit square with non-zero temperature
# on the left wall and zero on all other walls:
#
# $$\nabla^2 T = 0, \quad T(0, y) = \sin(\pi y), \quad T = 0 \text{ elsewhere}$$
#
# This is a Laplace equation (zero RHS) with inhomogeneous Dirichlet BCs.

# %%
from spectraldiffx import solve_helmholtz_2d

Nx, Ny = 64, 64
dx = 1.0 / (Nx + 1)
dy = 1.0 / (Ny + 1)

# Interior grid points (regular grid, DST-I)
x = jnp.arange(1, Nx + 1) * dx
y = jnp.arange(1, Ny + 1) * dy

# Zero RHS (Laplace equation)
rhs = jnp.zeros((Ny, Nx))

# Boundary values: sin(pi*y) on left wall, zero everywhere else
left_wall = jnp.sin(jnp.pi * y)

# Solve with Helmholtz (lambda=0 would be Poisson, but pure Laplace with
# Dirichlet is well-posed; use small lambda to avoid any numerical issues)
T = solve_helmholtz_2d(
    rhs, dx, dy,
    bc_x="dirichlet", bc_y="dirichlet",
    bc_x_values=(left_wall, None),  # left = sin(pi*y), right = 0
    bc_y_values=(None, None),       # bottom = 0, top = 0
)

# %% [markdown]
# The analytic solution is $T(x,y) = \sinh(\pi(1-x))/\sinh(\pi) \cdot \sin(\pi y)$.

# %%
X, Y = jnp.meshgrid(x, y, indexing="xy")
T_exact = jnp.sinh(jnp.pi * (1.0 - X)) / jnp.sinh(jnp.pi) * jnp.sin(jnp.pi * Y)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

im0 = axes[0].pcolormesh(np.array(X), np.array(Y), np.array(T), shading="auto")
axes[0].set_title("Numerical solution")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(np.array(X), np.array(Y), np.array(T_exact), shading="auto")
axes[1].set_title("Analytic solution")
axes[1].set_xlabel("x")
fig.colorbar(im1, ax=axes[1])

err = np.array(jnp.abs(T - T_exact))
im2 = axes[2].pcolormesh(np.array(X), np.array(Y), err, shading="auto")
axes[2].set_title(f"Error (max = {err.max():.2e})")
axes[2].set_xlabel("x")
fig.colorbar(im2, ax=axes[2])

fig.savefig(IMGDIR / "heated_plate.png", dpi=150)
plt.close()

# %% [markdown]
# ![Heated plate solution](../images/inhomogeneous_bcs/heated_plate.png)
#
# The solver recovers the analytic solution to $O(h^2)$ accuracy.

# %% [markdown]
# ## 2. Convergence Study
#
# Verify that the inhomogeneous Dirichlet solver converges at $O(h^2)$.
# We use $\psi(x,y) = \sin(\pi x)\sin(\pi y) + x + y$ which has non-zero
# boundary values from the $x + y$ term.

# %%
from spectraldiffx import solve_poisson_2d

resolutions = [16, 32, 64, 128]
errors = []

for N in resolutions:
    ddx = 1.0 / (N + 1)
    xx = jnp.arange(1, N + 1) * ddx
    XX, YY = jnp.meshgrid(xx, xx, indexing="xy")

    psi_exact = jnp.sin(jnp.pi * XX) * jnp.sin(jnp.pi * YY) + XX + YY
    f = -2 * jnp.pi**2 * jnp.sin(jnp.pi * XX) * jnp.sin(jnp.pi * YY)

    Lx = 1.0
    psi_got = solve_poisson_2d(
        f, ddx, ddx,
        bc_x="dirichlet", bc_y="dirichlet",
        bc_x_values=(
            jnp.sin(jnp.pi * 0.0) * jnp.sin(jnp.pi * xx) + 0.0 + xx,
            jnp.sin(jnp.pi * Lx) * jnp.sin(jnp.pi * xx) + Lx + xx,
        ),
        bc_y_values=(
            jnp.sin(jnp.pi * xx) * jnp.sin(jnp.pi * 0.0) + xx + 0.0,
            jnp.sin(jnp.pi * xx) * jnp.sin(jnp.pi * Lx) + xx + Lx,
        ),
    )
    err = float(jnp.max(jnp.abs(psi_got - psi_exact)))
    errors.append(err)

# %%
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
h_vals = [1.0 / (N + 1) for N in resolutions]
ax.loglog(h_vals, errors, "o-", label="Measured error")
ax.loglog(h_vals, [3 * h**2 for h in h_vals], "k--", alpha=0.5, label="$O(h^2)$ ref")
ax.set_xlabel("Grid spacing $h$")
ax.set_ylabel("$L_\\infty$ error")
ax.set_title("Convergence: Inhomogeneous Dirichlet")
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(IMGDIR / "convergence.png", dpi=150)
plt.close()

# %% [markdown]
# ![Convergence plot](../images/inhomogeneous_bcs/convergence.png)
#
# The error decreases as $O(h^2)$, confirming second-order accuracy
# from the FD2 eigenvalues.

# %% [markdown]
# ## 3. Mixed BCs with Inhomogeneous Values
#
# Combine different BC types on each axis with non-zero values.
# Here: Dirichlet in x (non-zero walls) + Neumann staggered in y (non-zero flux).
#
# Solution: $\psi(x,y) = x^2 + y^2$, so $\nabla^2\psi = 4$.

# %%
Nx, Ny = 32, 32
dx, dy = 0.1, 0.1

# Regular grid in x (interior points), staggered in y (cell centres)
x_reg = jnp.arange(1, Nx + 1) * dx
y_stag = (jnp.arange(Ny) + 0.5) * dy
X_m, Y_m = jnp.meshgrid(x_reg, y_stag, indexing="xy")
psi_exact = X_m**2 + Y_m**2

Lx = (Nx + 1) * dx
Ly = Ny * dy

lam = 1.0
rhs_mixed = 4.0 * jnp.ones((Ny, Nx)) - lam * psi_exact

# Dirichlet values in x
y_arr = (jnp.arange(Ny) + 0.5) * dy

# Neumann flux in y: dpsi/dn at y=0 is 0, dpsi/dn at y=Ly is 2*Ly
psi_mixed = solve_helmholtz_2d(
    rhs_mixed, dx, dy,
    bc_x="dirichlet", bc_y="neumann_stag",
    lambda_=lam,
    bc_x_values=(y_arr**2, Lx**2 + y_arr**2),
    bc_y_values=(jnp.zeros(Nx), 2.0 * Ly * jnp.ones(Nx)),
)

err_mixed = float(jnp.max(jnp.abs(psi_mixed - psi_exact)))

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
im0 = axes[0].pcolormesh(
    np.array(X_m), np.array(Y_m), np.array(psi_mixed), shading="auto"
)
axes[0].set_title("Solution (Dirichlet x + Neumann y)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(
    np.array(X_m), np.array(Y_m),
    np.array(jnp.abs(psi_mixed - psi_exact)), shading="auto",
)
axes[1].set_title(f"Error (max = {err_mixed:.2e})")
axes[1].set_xlabel("x")
fig.colorbar(im1, ax=axes[1])

fig.savefig(IMGDIR / "mixed_inhomogeneous.png", dpi=150)
plt.close()

# %% [markdown]
# ![Mixed inhomogeneous](../images/inhomogeneous_bcs/mixed_inhomogeneous.png)
#
# The quadratic solution is recovered to machine precision for this
# combination of regular Dirichlet and staggered Neumann BCs.

# %% [markdown]
# ## 4. 3D Mixed BCs: Atmospheric Boundary Layer
#
# A common 3D setup: periodic in x/y (horizontal), Neumann staggered in z
# (vertical). This models an atmospheric boundary layer where the horizontal
# directions are homogeneous and the vertical has no-flux conditions.

# %%
from spectraldiffx import solve_helmholtz_3d
from spectraldiffx._src.fourier.eigenvalues import (
    dct2_eigenvalues,
    fft_eigenvalues,
)

Nx3, Ny3, Nz3 = 16, 16, 12
dx3, dy3, dz3 = 1.0, 1.0, 1.0


def _ef_fft(N, k):
    return jnp.cos(2 * jnp.pi * k * jnp.arange(N) / N)


def _ef_dct2(N, k):
    return jnp.cos(jnp.pi * k * (2 * jnp.arange(N) + 1) / (2 * N))


kx, ky, kz = 2, 3, 2
fx = _ef_fft(Nx3, kx)
fy = _ef_fft(Ny3, ky)
fz = _ef_dct2(Nz3, kz)
psi_exact_3d = fz[:, None, None] * fy[None, :, None] * fx[None, None, :]

eigx = fft_eigenvalues(Nx3, dx3)[kx]
eigy = fft_eigenvalues(Ny3, dy3)[ky]
eigz = dct2_eigenvalues(Nz3, dz3)[kz]
lam3 = 1.0
rhs_3d = (eigx + eigy + eigz - lam3) * psi_exact_3d

psi_got_3d = solve_helmholtz_3d(
    rhs_3d, dx3, dy3, dz3,
    bc_x="periodic", bc_y="periodic", bc_z="neumann_stag",
    lambda_=lam3,
)

err_3d = float(jnp.max(jnp.abs(psi_got_3d - psi_exact_3d)))

# Plot a vertical slice
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
z_idx = jnp.arange(Nz3)
x_idx = jnp.arange(Nx3)
Z_sl, X_sl = jnp.meshgrid(z_idx, x_idx, indexing="ij")

im0 = axes[0].pcolormesh(
    np.array(X_sl), np.array(Z_sl),
    np.array(psi_got_3d[:, Ny3 // 2, :]), shading="auto",
)
axes[0].set_title("3D solution (y-slice)")
axes[0].set_xlabel("x index")
axes[0].set_ylabel("z index")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(
    np.array(X_sl), np.array(Z_sl),
    np.array(jnp.abs(psi_got_3d[:, Ny3 // 2, :] - psi_exact_3d[:, Ny3 // 2, :])),
    shading="auto",
)
axes[1].set_title(f"Error (max = {err_3d:.2e})")
axes[1].set_xlabel("x index")
fig.colorbar(im1, ax=axes[1])

fig.savefig(IMGDIR / "atmospheric_bl_3d.png", dpi=150)
plt.close()

# %% [markdown]
# ![3D atmospheric BL](../images/inhomogeneous_bcs/atmospheric_bl_3d.png)
#
# The 3D mixed-BC solver recovers the eigenfunction to machine precision.

# %% [markdown]
# ## Summary
#
# | Feature | Function | Notes |
# |---------|----------|-------|
# | Inhomogeneous Dirichlet | `bc_x_values=(left, right)` | Prescribe non-zero psi |
# | Inhomogeneous Neumann | `bc_x_values=(g_left, g_right)` | Prescribe non-zero dpsi/dn |
# | Mixed per-axis BCs (2D) | `solve_helmholtz_2d(bc_x=..., bc_y=...)` | Any combination |
# | Mixed per-axis BCs (3D) | `solve_helmholtz_3d(bc_x=..., bc_y=..., bc_z=...)` | Any combination |
# | Layer 0 helpers | `modify_rhs_1d/2d/3d` | Manual RHS modification |
# | Module wrappers | `MixedBCHelmholtzSolver2D/3D` | BC values at call time |
