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
# # Spectral Elliptic Solvers Demo
#
# This notebook demonstrates how to solve the Helmholtz equation
#
# $$(\nabla^2 - \lambda)\psi = f$$
#
# using three spectral methods, each corresponding to a different boundary condition:
#
# | BC type | Transform | Solver function |
# |---------|-----------|-----------------|
# | **Dirichlet** ($\psi = 0$ on boundary) | DST-I | `solve_poisson_dst` / `solve_helmholtz_dst` |
# | **Neumann** ($\partial\psi/\partial n = 0$ on boundary) | DCT-II | `solve_poisson_dct` / `solve_helmholtz_dct` |
# | **Periodic** ($\psi(0) = \psi(L)$) | FFT | `solve_poisson_fft` / `solve_helmholtz_fft` |
#
# Setting $\lambda = 0$ reduces the Helmholtz equation to the **Poisson equation** $\nabla^2\psi = f$.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from spectraldiffx import (
    DirichletHelmholtzSolver2D,
    dst1_eigenvalues,
    dct2_eigenvalues,
    fft_eigenvalues,
    solve_helmholtz_dct,
    solve_helmholtz_dst,
    solve_poisson_dst,
    solve_poisson_dct,
    solve_poisson_fft,
)

# %% [markdown]
# ## Section 1: Poisson with Dirichlet BCs (DST-I)
#
# We solve $\nabla^2 \psi = f$ on an interior grid where $\psi = 0$ on all four edges.
#
# **Strategy**: pick a known eigenfunction of the discrete Laplacian, compute
# its Laplacian analytically using the DST-I eigenvalues, then verify the solver
# recovers the original field.

# %%
# Grid parameters
Nx, Ny = 64, 64
dx, dy = 1.0, 1.0

# Known eigenfunction: sin(pi*p*(i+1)/(Nx+1)) * sin(pi*q*(j+1)/(Ny+1))
p, q = 2, 3
i_idx = jnp.arange(Nx)
j_idx = jnp.arange(Ny)
II, JJ = jnp.meshgrid(i_idx, j_idx)

psi_exact = (
    jnp.sin(jnp.pi * p * (II + 1) / (Nx + 1))
    * jnp.sin(jnp.pi * q * (JJ + 1) / (Ny + 1))
)

# The Laplacian of this eigenfunction equals (eigx[p-1] + eigy[q-1]) * psi_exact
eigx = dst1_eigenvalues(Nx, dx)
eigy = dst1_eigenvalues(Ny, dy)
eigenvalue = eigx[p - 1] + eigy[q - 1]
print(f"Combined eigenvalue (p={p}, q={q}): {eigenvalue:.6f}")

rhs_dst = eigenvalue * psi_exact

# %%
# Solve and check error
psi_dst = solve_poisson_dst(rhs_dst, dx, dy)
error_dst = jnp.max(jnp.abs(psi_dst - psi_exact))
print(f"Max error (Poisson DST): {error_dst:.2e}")

# %% [markdown]
# ## Section 2: Helmholtz with Neumann BCs (DCT-II)
#
# We solve $(\nabla^2 - \lambda)\psi = f$ with homogeneous Neumann BCs
# ($\partial\psi/\partial n = 0$) using the DCT-II.
#
# The eigenfunctions are cosines: $\cos(\pi p \, i / N_x) \cos(\pi q \, j / N_y)$.

# %%
lambda_ = 1.0

# Known eigenfunction using cosines (Neumann basis)
p, q = 2, 3
psi_exact_dct = (
    jnp.cos(jnp.pi * p * II / Nx)
    * jnp.cos(jnp.pi * q * JJ / Ny)
)

# Eigenvalues for the DCT-II basis
eigx_dct = dct2_eigenvalues(Nx, dx)
eigy_dct = dct2_eigenvalues(Ny, dy)
eigenvalue_dct = eigx_dct[p] + eigy_dct[q]

# RHS = (eigenvalue - lambda) * psi_exact
rhs_dct = (eigenvalue_dct - lambda_) * psi_exact_dct

# %%
psi_dct = solve_helmholtz_dct(rhs_dct, dx, dy, lambda_=lambda_)
error_dct = jnp.max(jnp.abs(psi_dct - psi_exact_dct))
print(f"Max error (Helmholtz DCT, lambda={lambda_}): {error_dct:.2e}")

# %% [markdown]
# ## Section 3: Poisson with Periodic BCs (FFT)
#
# We solve $\nabla^2 \psi = f$ with periodic BCs using the 2-D FFT.
#
# The eigenfunctions are complex exponentials, but for a real test we use
# $\cos(2\pi p \, i / N_x) \cos(2\pi q \, j / N_y)$.

# %%
p, q = 2, 3
psi_exact_fft = (
    jnp.cos(2 * jnp.pi * p * II / Nx)
    * jnp.cos(2 * jnp.pi * q * JJ / Ny)
)

eigx_fft = fft_eigenvalues(Nx, dx)
eigy_fft = fft_eigenvalues(Ny, dy)
eigenvalue_fft = eigx_fft[p] + eigy_fft[q]

rhs_fft = eigenvalue_fft * psi_exact_fft

# %%
psi_fft = solve_poisson_fft(rhs_fft, dx, dy)
error_fft = jnp.max(jnp.abs(psi_fft - psi_exact_fft))
print(f"Max error (Poisson FFT): {error_fft:.2e}")

# %% [markdown]
# ## Section 4: Comparison of All Three Solvers
#
# We define a Gaussian-bump RHS and solve with all three BC types,
# then plot the solutions side by side.

# %%
# Gaussian bump RHS centered at the middle of the domain
cx, cy = Nx / 2, Ny / 2
sigma = Nx / 8
rhs_gauss = jnp.exp(-((II - cx) ** 2 + (JJ - cy) ** 2) / (2 * sigma**2))

# Solve with all three methods
psi_dir = solve_poisson_dst(rhs_gauss, dx, dy)
psi_neu = solve_poisson_dct(rhs_gauss, dx, dy)
psi_per = solve_poisson_fft(rhs_gauss, dx, dy)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

titles = ["Dirichlet (DST)", "Neumann (DCT)", "Periodic (FFT)"]
solutions = [psi_dir, psi_neu, psi_per]

for ax, title, psi in zip(axes, titles, solutions):
    im = ax.contourf(np.array(psi), levels=32, cmap="RdBu_r")
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("Poisson solutions for a Gaussian bump RHS", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Section 5: Using Layer 1 Classes
#
# The `DirichletHelmholtzSolver2D` class wraps `solve_helmholtz_dst` in an
# `eqx.Module` that stores grid parameters and is callable.

# %%
solver = DirichletHelmholtzSolver2D(dx=dx, dy=dy, alpha=0.0)

# Solve the same Gaussian problem
psi_class = solver(rhs_gauss)

# Verify it matches the functional API
diff = jnp.max(jnp.abs(psi_class - psi_dir))
print(f"Max difference (class vs function): {diff:.2e}")
print(f"Solver is callable: {callable(solver)}")

# %% [markdown]
# ## Section 6: Batched Solve with vmap
#
# We can use `jax.vmap` to solve multiple RHS in parallel. Here we
# create a batch of 5 Gaussian bumps with different centers.

# %%
# Create 5 different RHS by shifting the Gaussian center
centers_x = jnp.array([16.0, 24.0, 32.0, 40.0, 48.0])
centers_y = jnp.array([32.0, 32.0, 32.0, 32.0, 32.0])

rhs_batch = jnp.stack(
    [
        jnp.exp(-((II - cx) ** 2 + (JJ - cy) ** 2) / (2 * sigma**2))
        for cx, cy in zip(centers_x, centers_y)
    ]
)
print(f"Batched RHS shape: {rhs_batch.shape}")

# %%
# vmap over the batch dimension
batched_solve = jax.vmap(solve_helmholtz_dst, in_axes=(0, None, None, None))
psi_batch = batched_solve(rhs_batch, dx, dy, 0.0)
print(f"Batched solution shape: {psi_batch.shape}")

# Verify each slice matches the individual solve
for k in range(5):
    psi_k = solve_helmholtz_dst(rhs_batch[k], dx, dy, 0.0)
    err = jnp.max(jnp.abs(psi_batch[k] - psi_k))
    print(f"  Batch element {k}: max error vs individual solve = {err:.2e}")
