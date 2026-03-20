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
# # Spectral Elliptic PDE Solvers: Poisson & Helmholtz Equations
#
# ## What this tutorial covers
#
# Many problems in physics and engineering reduce to solving an elliptic PDE
# of the form
#
# $$(\nabla^2 - \lambda)\,\psi = f,$$
#
# known as the **Helmholtz equation**. Setting $\lambda = 0$ gives the
# **Poisson equation** $\nabla^2 \psi = f$.
#
# These equations appear everywhere:
#
# - **Pressure projection** in incompressible Navier--Stokes solvers
# - **Geostrophic balance** and quasi-geostrophic streamfunction inversion
# - **Electrostatics** (Coulomb potential from a charge distribution)
# - **Quantum mechanics** (time-independent Schrodinger equation)
# - **Screened Poisson** problems in plasma physics ($\lambda > 0$)
#
# This notebook demonstrates how `spectraldiffx` solves these equations
# on a 2-D rectangular grid using three spectral methods, each corresponding
# to a different boundary condition:
#
# | BC type | Transform | Eigenfunctions | Solver |
# |---------|-----------|----------------|--------|
# | **Dirichlet** $\psi = 0$ on boundary | DST-I | $\sin$ | `solve_poisson_dst` / `solve_helmholtz_dst` |
# | **Neumann** $\partial_n \psi = 0$ on boundary | DCT-II | $\cos$ | `solve_poisson_dct` / `solve_helmholtz_dct` |
# | **Periodic** $\psi(0) = \psi(L)$ | FFT | $e^{ikx}$ | `solve_poisson_fft` / `solve_helmholtz_fft` |
#
# We also show how to use the `eqx.Module` class wrappers and `jax.vmap`
# for batched solves.

# %%
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

jax.config.update("jax_enable_x64", True)

IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "images" / "demo_elliptic_solvers"
IMG_DIR.mkdir(parents=True, exist_ok=True)

from spectraldiffx import (
    DirichletHelmholtzSolver2D,
    NeumannHelmholtzSolver2D,
    dct2_eigenvalues,
    dst1_eigenvalues,
    fft_eigenvalues,
    solve_helmholtz_dct,
    solve_helmholtz_dst,
    solve_helmholtz_fft,
    solve_poisson_dct,
    solve_poisson_dst,
    solve_poisson_fft,
)

# %% [markdown]
# ## 1. The Spectral Solution Method
#
# ### Continuous problem
#
# Given a source $f(x, y)$, find $\psi(x, y)$ satisfying
#
# $$\nabla^2 \psi - \lambda\,\psi = f, \qquad \nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}.$$
#
# ### Discrete spectral approach
#
# On an $N_y \times N_x$ grid with spacing $(\Delta x, \Delta y)$, the
# second-order finite-difference Laplacian has known eigenvectors
# for DST, DCT, and FFT bases. The algorithm is:
#
# ```
# Algorithm: Spectral Helmholtz Solve
# ────────────────────────────────────
# 1. f_hat = Transform(f)               ← forward DST / DCT / FFT
# 2. psi_hat[p,q] = f_hat[p,q]          ← spectral division
#                    ─────────────────
#                    lambda_x[p] + lambda_y[q] - alpha
# 3. psi = InverseTransform(psi_hat)     ← backward transform
# ```
#
# Each transform type has its own eigenvalue formula:
#
# - **DST-I** (Dirichlet): $\lambda^x_p = \frac{2}{\Delta x^2}\left[\cos\!\left(\frac{\pi\,p}{N_x+1}\right) - 1\right], \quad p = 1, \dots, N_x$
# - **DCT-II** (Neumann): $\lambda^x_p = \frac{2}{\Delta x^2}\left[\cos\!\left(\frac{\pi\,p}{N_x}\right) - 1\right], \quad p = 0, \dots, N_x-1$
# - **FFT** (Periodic): $\lambda^x_p = \frac{2}{\Delta x^2}\left[\cos\!\left(\frac{2\pi\,p}{N_x}\right) - 1\right], \quad p = 0, \dots, N_x-1$
#
# The $y$-direction eigenvalues follow the same pattern with $N_y$ and $\Delta y$.
#
# **Null-mode caveat**: For Neumann and periodic BCs, the $(0, 0)$ eigenvalue
# is zero, making the Poisson equation singular. The solution is determined
# only up to an additive constant. The solver handles this by projecting
# out the zero mode (enforcing zero-mean $\psi$).

# %%
# ─── Grid parameters ───
# 64x64 is large enough to see smooth fields, small enough to run instantly.
# dx = dy = 1.0 gives eigenvalues in simple units.
Nx, Ny = 64, 64
dx, dy = 1.0, 1.0

# Index arrays for building eigenfunctions
i_idx = jnp.arange(Nx)
j_idx = jnp.arange(Ny)
II, JJ = jnp.meshgrid(i_idx, j_idx)  # shape: (Ny, Nx)

print(f"Grid shape: ({Ny}, {Nx})")
print(f"Grid spacing: dx={dx}, dy={dy}")
print(f"Index arrays II, JJ shape: {II.shape}")

# %% [markdown]
# ## 2. Dirichlet BCs (DST-I): Eigenfunction Recovery
#
# The DST-I eigenfunctions on an $N$-point interior grid are
#
# $$\phi_{p,q}(i, j) = \sin\!\left(\frac{\pi\,p\,(i+1)}{N_x+1}\right)\sin\!\left(\frac{\pi\,q\,(j+1)}{N_y+1}\right), \quad p \in [1, N_x],\; q \in [1, N_y].$$
#
# These satisfy homogeneous Dirichlet conditions: $\phi = 0$ at $i = -1$
# and $i = N_x$ (the ghost boundary points).
#
# **Test**: Build an eigenfunction, compute its Laplacian analytically
# using the eigenvalue, then verify the solver recovers $\phi$ exactly.

# %%
# Mode indices (low modes for a smooth field)
p, q = 2, 3

# Eigenfunction
psi_exact = (
    jnp.sin(jnp.pi * p * (II + 1) / (Nx + 1))
    * jnp.sin(jnp.pi * q * (JJ + 1) / (Ny + 1))
)
print(f"psi_exact shape: {psi_exact.shape}")

# Eigenvalues from the DST-I formula
eigx = dst1_eigenvalues(Nx, dx)
eigy = dst1_eigenvalues(Ny, dy)
eigenvalue = eigx[p - 1] + eigy[q - 1]  # 1-indexed: mode p uses eigx[p-1]
print(f"DST-I eigenvalue for (p={p}, q={q}): {eigenvalue:.6f}")

# The Laplacian of the eigenfunction is eigenvalue * psi_exact
rhs_dst = eigenvalue * psi_exact
print(f"RHS shape: {rhs_dst.shape}")

# %%
# Solve and measure error
psi_dst = solve_poisson_dst(rhs_dst, dx, dy)
error_dst = float(jnp.max(jnp.abs(psi_dst - psi_exact)))
print(f"Max error (Poisson DST): {error_dst:.2e}")
print(f"Solution shape: {psi_dst.shape}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

im0 = axes[0].imshow(np.array(psi_exact), cmap="RdBu_r", origin="lower")
axes[0].set_title(r"Exact $\psi$ (eigenfunction)")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(np.array(psi_dst), cmap="RdBu_r", origin="lower")
axes[1].set_title(r"Computed $\psi$ (DST-I)")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

im2 = axes[2].imshow(np.array(jnp.abs(psi_dst - psi_exact)), cmap="hot_r", origin="lower")
axes[2].set_title(f"Error (max = {error_dst:.1e})")
plt.colorbar(im2, ax=axes[2], shrink=0.8, format="%.0e")

for ax in axes:
    ax.set_xlabel("i")
    ax.set_ylabel("j")

plt.suptitle(f"Eigenfunction Recovery: DST-I Poisson ($p={p}, q={q}$)", fontsize=13)
plt.tight_layout()
fig.savefig(IMG_DIR / "eigenfunction_recovery.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Eigenfunction recovery](../../images/demo_elliptic_solvers/eigenfunction_recovery.png)
#
# The error is at machine precision ($\sim 10^{-14}$) because the solver
# uses the exact same eigenvector basis as the test function.

# %% [markdown]
# ## 3. Neumann BCs (DCT-II): Eigenfunction Recovery
#
# The DCT-II eigenfunctions are cosines defined on cell centers:
#
# $$\phi_{p,q}(i, j) = \cos\!\left(\frac{\pi\,p\,i}{N_x}\right)\cos\!\left(\frac{\pi\,q\,j}{N_y}\right), \quad p \in [0, N_x-1],\; q \in [0, N_y-1].$$
#
# These satisfy homogeneous Neumann conditions ($\partial_n \phi = 0$) at
# the half-grid boundaries.
#
# **Zero-mean gauge**: When $\lambda = 0$, the $(0,0)$ mode is in the
# null space of the Laplacian. The solver projects it out, yielding the
# unique zero-mean solution. We choose $p, q > 0$ to avoid the null mode.

# %%
lambda_neu = 0.0
p, q = 2, 3

# Cosine eigenfunction
psi_exact_dct = (
    jnp.cos(jnp.pi * p * II / Nx)
    * jnp.cos(jnp.pi * q * JJ / Ny)
)

# DCT-II eigenvalues (0-indexed)
eigx_dct = dct2_eigenvalues(Nx, dx)
eigy_dct = dct2_eigenvalues(Ny, dy)
eigenvalue_dct = eigx_dct[p] + eigy_dct[q]
print(f"DCT-II eigenvalue for (p={p}, q={q}): {eigenvalue_dct:.6f}")

# RHS = eigenvalue * psi_exact  (Poisson: lambda=0)
rhs_dct = eigenvalue_dct * psi_exact_dct

# %%
psi_dct = solve_poisson_dct(rhs_dct, dx, dy)
error_dct = float(jnp.max(jnp.abs(psi_dct - psi_exact_dct)))
print(f"Max error (Poisson DCT): {error_dct:.2e}")
print(f"Solution shape: {psi_dct.shape}")

# %% [markdown]
# ## 4. Periodic BCs (FFT): Eigenfunction Recovery
#
# The FFT eigenfunctions are complex exponentials $e^{2\pi i\,p\,k/N}$.
# For a real-valued test we use the cosine:
#
# $$\phi_{p,q}(i, j) = \cos\!\left(\frac{2\pi\,p\,i}{N_x}\right)\cos\!\left(\frac{2\pi\,q\,j}{N_y}\right).$$
#
# Periodic BCs wrap the domain: $\psi(0) = \psi(N)$.

# %%
p, q = 2, 3

psi_exact_fft = (
    jnp.cos(2 * jnp.pi * p * II / Nx)
    * jnp.cos(2 * jnp.pi * q * JJ / Ny)
)

eigx_fft = fft_eigenvalues(Nx, dx)
eigy_fft = fft_eigenvalues(Ny, dy)
eigenvalue_fft = eigx_fft[p] + eigy_fft[q]
print(f"FFT eigenvalue for (p={p}, q={q}): {eigenvalue_fft:.6f}")

rhs_fft = eigenvalue_fft * psi_exact_fft

# %%
psi_fft = solve_poisson_fft(rhs_fft, dx, dy)
error_fft = float(jnp.max(jnp.abs(psi_fft - psi_exact_fft)))
print(f"Max error (Poisson FFT): {error_fft:.2e}")
print(f"Solution shape: {psi_fft.shape}")

# %% [markdown]
# ### Summary of eigenfunction recovery tests
#
# All three solvers recover the exact eigenfunction to machine precision,
# confirming that the eigenvalue formulas and transform implementations
# are consistent.

# %% [markdown]
# ## 5. Helmholtz Equation ($\lambda \neq 0$): Screening Effect
#
# When $\lambda > 0$, the equation becomes
#
# $$(\nabla^2 - \lambda)\,\psi = f.$$
#
# The $-\lambda\,\psi$ term acts as a **screening** (or damping) term:
# it penalizes large values of $\psi$, pulling the solution toward zero
# and localizing the response near the source. The effective length scale
# is $\ell \sim 1/\sqrt{\lambda}$.
#
# We demonstrate this with the DST (Dirichlet) eigenfunction test,
# adding a nonzero $\lambda$.

# %%
lambda_helm = 1.0
p, q = 2, 3

# Reuse the DST eigenfunction from Section 2
eigx = dst1_eigenvalues(Nx, dx)
eigy = dst1_eigenvalues(Ny, dy)
eigenvalue = eigx[p - 1] + eigy[q - 1]

psi_exact_helm = (
    jnp.sin(jnp.pi * p * (II + 1) / (Nx + 1))
    * jnp.sin(jnp.pi * q * (JJ + 1) / (Ny + 1))
)

# RHS = (eigenvalue - lambda) * psi_exact  for Helmholtz
rhs_helm = (eigenvalue - lambda_helm) * psi_exact_helm

# %%
psi_helm = solve_helmholtz_dst(rhs_helm, dx, dy, lambda_=lambda_helm)
error_helm = float(jnp.max(jnp.abs(psi_helm - psi_exact_helm)))
print(f"Max error (Helmholtz DST, lambda={lambda_helm}): {error_helm:.2e}")

# %%
# Compare solution magnitudes: Helmholtz screening reduces amplitude
# Solve same RHS with Poisson (lambda=0) and Helmholtz (lambda=1)
rhs_compare = eigenvalue * psi_exact_helm  # true Laplacian
psi_poisson = solve_poisson_dst(rhs_compare, dx, dy)
psi_screened = solve_helmholtz_dst(rhs_compare, dx, dy, lambda_=lambda_helm)

print(f"Poisson max |psi|:   {float(jnp.max(jnp.abs(psi_poisson))):.6f}")
print(f"Helmholtz max |psi|: {float(jnp.max(jnp.abs(psi_screened))):.6f}")
print(f"Ratio (Helmholtz/Poisson): {float(jnp.max(jnp.abs(psi_screened)) / jnp.max(jnp.abs(psi_poisson))):.4f}")

# %% [markdown]
# The Helmholtz solution is smaller because the spectral denominator
# changes from $\lambda_k$ to $\lambda_k - \alpha$, increasing the
# magnitude of the divisor and shrinking $\hat\psi_k$.

# %% [markdown]
# ## 6. Visual Comparison: Gaussian Bump with Three BC Types
#
# This is the key figure. We define a Gaussian bump RHS and solve with
# all three boundary condition types. The same source produces noticeably
# different solutions because the BCs determine how the field "sees" the
# domain boundaries.
#
# ```
# Dirichlet:  psi pinned to 0 at edges  →  solution pulled toward zero near walls
# Neumann:    no flux at edges           →  solution "pools" at boundaries
# Periodic:   wraps around               →  no boundary effect, but translational symmetry
# ```

# %%
# Gaussian bump centered in the domain
# sigma = Nx/8 gives a bump that is well-resolved but doesn't touch the edges
cx, cy = Nx / 2, Ny / 2
sigma = Nx / 8
rhs_gauss = jnp.exp(-((II - cx) ** 2 + (JJ - cy) ** 2) / (2 * sigma**2))

print(f"Gaussian center: ({cx}, {cy})")
print(f"Gaussian sigma: {sigma}")
print(f"RHS shape: {rhs_gauss.shape}")
print(f"RHS range: [{float(jnp.min(rhs_gauss)):.4f}, {float(jnp.max(rhs_gauss)):.4f}]")

# Solve with all three methods
psi_dir = solve_poisson_dst(rhs_gauss, dx, dy)
psi_neu = solve_poisson_dct(rhs_gauss, dx, dy)
psi_per = solve_poisson_fft(rhs_gauss, dx, dy)

print(f"\nSolution ranges:")
print(f"  Dirichlet: [{float(jnp.min(psi_dir)):.4f}, {float(jnp.max(psi_dir)):.4f}]")
print(f"  Neumann:   [{float(jnp.min(psi_neu)):.4f}, {float(jnp.max(psi_neu)):.4f}]")
print(f"  Periodic:  [{float(jnp.min(psi_per)):.4f}, {float(jnp.max(psi_per)):.4f}]")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

titles = ["Dirichlet (DST-I)", "Neumann (DCT-II)", "Periodic (FFT)"]
solutions = [psi_dir, psi_neu, psi_per]

for ax, title, psi in zip(axes, titles, solutions):
    im = ax.contourf(np.array(psi), levels=32, cmap="RdBu_r")
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle(r"Poisson $\nabla^2\psi = f$ with Gaussian bump RHS", fontsize=14)
plt.tight_layout()
fig.savefig(IMG_DIR / "poisson_comparison_three_bcs.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Poisson solutions for three BC types](../../images/demo_elliptic_solvers/poisson_comparison_three_bcs.png)

# %% [markdown]
# **Observations**:
#
# - **Dirichlet**: The solution is forced to zero at all four edges, creating
#   a bowl shape.
# - **Neumann**: No-flux boundaries allow the solution to accumulate near
#   the walls. The zero-mean constraint shifts the baseline.
# - **Periodic**: The solution wraps smoothly from one edge to the opposite
#   edge, with no boundary artifacts.

# %% [markdown]
# ## 7. Module Classes: `DirichletHelmholtzSolver2D` and `NeumannHelmholtzSolver2D`
#
# The functional API (`solve_poisson_dst`, etc.) is convenient for one-off
# solves. For repeated solves with the same grid parameters, the
# `eqx.Module` class wrappers store the configuration and provide a
# clean callable interface.
#
# ```
# ┌─────────────────────────────────────┐
# │  DirichletHelmholtzSolver2D         │
# │  ─────────────────────────────────  │
# │  Fields:  dx, dy, alpha             │
# │  __call__(rhs) → psi               │
# │  Compatible with eqx.filter_jit    │
# └─────────────────────────────────────┘
# ```
#
# Since these are `equinox.Module` instances (frozen dataclasses), they
# are valid JAX pytrees and can be passed through `jax.jit`, `jax.vmap`,
# and `jax.grad` without special handling.

# %%
# Dirichlet solver (alpha=0 → Poisson)
solver_dir = DirichletHelmholtzSolver2D(dx=dx, dy=dy, alpha=0.0)
print(f"Dirichlet solver: {solver_dir}")
print(f"  dx={solver_dir.dx}, dy={solver_dir.dy}, alpha={solver_dir.alpha}")

# Neumann solver
solver_neu = NeumannHelmholtzSolver2D(dx=dx, dy=dy, alpha=0.0)
print(f"Neumann solver: {solver_neu}")

# %%
# Solve with the class API
psi_class_dir = solver_dir(rhs_gauss)
psi_class_neu = solver_neu(rhs_gauss)

# Verify they match the functional API
diff_dir = float(jnp.max(jnp.abs(psi_class_dir - psi_dir)))
diff_neu = float(jnp.max(jnp.abs(psi_class_neu - psi_neu)))
print(f"Dirichlet: class vs function max diff = {diff_dir:.2e}")
print(f"Neumann:   class vs function max diff = {diff_neu:.2e}")

# %%
# eqx.filter_jit compiles the solver, tracing only the dynamic leaves
jit_solver = eqx.filter_jit(solver_dir)
psi_jit = jit_solver(rhs_gauss)
diff_jit = float(jnp.max(jnp.abs(psi_jit - psi_dir)))
print(f"JIT'd solver vs functional: {diff_jit:.2e}")

# %% [markdown]
# ## 8. Batched Solves with `jax.vmap`
#
# A powerful feature of JAX is `vmap` — vectorized map. Instead of
# looping over multiple RHS arrays, we solve them all in parallel with
# a single call. This is especially useful for:
#
# - Ensemble simulations (different initial conditions)
# - Multi-layer ocean models (solve each layer independently)
# - Sensitivity analysis (perturbed forcings)

# %%
# Create a batch of 5 Gaussian bumps with different centers
centers_x = jnp.array([16.0, 24.0, 32.0, 40.0, 48.0])
centers_y = jnp.array([32.0, 32.0, 32.0, 32.0, 32.0])

rhs_batch = jnp.stack(
    [
        jnp.exp(-((II - cx) ** 2 + (JJ - cy) ** 2) / (2 * sigma**2))
        for cx, cy in zip(centers_x, centers_y)
    ]
)
print(f"Batched RHS shape: {rhs_batch.shape}")  # (5, 64, 64)

# %%
# vmap over the leading (batch) dimension
# in_axes=(0, None, None, None) means: batch axis 0 for rhs, broadcast dx, dy, lambda_
batched_solve = jax.vmap(solve_helmholtz_dst, in_axes=(0, None, None, None))
psi_batch = batched_solve(rhs_batch, dx, dy, 0.0)
print(f"Batched solution shape: {psi_batch.shape}")  # (5, 64, 64)

# %%
# Verify each batch element matches the individual solve
print("Verification (batch vs individual):")
for k in range(5):
    psi_k = solve_helmholtz_dst(rhs_batch[k], dx, dy, 0.0)
    err = float(jnp.max(jnp.abs(psi_batch[k] - psi_k)))
    print(f"  Batch element {k} (cx={float(centers_x[k]):.0f}): max error = {err:.2e}")

# %% [markdown]
# All errors are zero (or machine epsilon), confirming that `vmap`
# produces identical results to sequential solves, but executes them
# as a single vectorized operation on the accelerator.

# %% [markdown]
# ## Summary
#
# | Feature | Function | Class |
# |---------|----------|-------|
# | Dirichlet (DST-I) | `solve_poisson_dst` / `solve_helmholtz_dst` | `DirichletHelmholtzSolver2D` |
# | Neumann (DCT-II) | `solve_poisson_dct` / `solve_helmholtz_dct` | `NeumannHelmholtzSolver2D` |
# | Periodic (FFT) | `solve_poisson_fft` / `solve_helmholtz_fft` | `SpectralHelmholtzSolver2D` |
#
# Key takeaways:
#
# 1. **Spectral methods** solve elliptic PDEs in $O(N \log N)$ time via
#    fast transforms.
# 2. **Boundary conditions** are encoded in the choice of transform
#    (DST / DCT / FFT) and the corresponding eigenvalue formula.
# 3. **Helmholtz screening** ($\lambda > 0$) reduces solution magnitude
#    and localizes the response.
# 4. **Module wrappers** provide a clean OOP interface compatible with
#    `eqx.filter_jit`.
# 5. **`jax.vmap`** enables embarrassingly parallel batch solves with
#    no code changes.
