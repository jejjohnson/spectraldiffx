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
# # Capacitance Matrix Solver Demo
#
# Solving Poisson or Helmholtz equations on **irregular domains** (e.g. ocean
# basins with coastlines) is more challenging than on rectangles because
# standard spectral transforms (DST/DCT/FFT) assume a rectangular grid.
#
# The **capacitance matrix method** (Buzbee, Golub & Nielson, 1970) extends
# a fast rectangular spectral solver to masked domains.  The idea:
#
# 1. Solve on the full rectangle, ignoring the mask.
# 2. The solution violates $\psi = 0$ at inner-boundary points.
# 3. Correct using precomputed Green's functions so that $\psi = 0$ is
#    enforced at all boundary points of the irregular domain.
#
# The offline cost is $O(N_b)$ rectangular solves (one per boundary point).
# The online cost per solve is $O(N_b^2 + N \log N)$.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from spectraldiffx import build_capacitance_solver

# %% [markdown]
# ## Section 1: Build a Circular Ocean Basin Mask
#
# We create a binary mask on a 32x32 grid representing a circular ocean
# basin (True = ocean interior, False = land/exterior).

# %%
Ny, Nx = 32, 32
dx, dy = 1.0, 1.0

# Coordinate arrays (cell centers)
j_idx, i_idx = np.mgrid[0:Ny, 0:Nx]
center_y, center_x = Ny / 2, Nx / 2
radius = 0.35 * min(Ny, Nx)

# Circular mask
dist = np.sqrt((i_idx - center_x) ** 2 + (j_idx - center_y) ** 2)
mask = dist < radius

print(f"Grid size: {Ny} x {Nx}")
print(f"Circle center: ({center_x}, {center_y}), radius: {radius}")
print(f"Interior points: {mask.sum()} / {Ny * Nx}")

# %%
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(mask.astype(float), origin="lower", cmap="Blues", vmin=0, vmax=1)
ax.set_title("Ocean basin mask (blue = interior)")
ax.set_xlabel("i")
ax.set_ylabel("j")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Section 2: Build the Capacitance Solver
#
# `build_capacitance_solver` performs the offline precomputation:
# detecting inner-boundary points, computing Green's functions, and
# inverting the capacitance matrix.

# %%
solver_dst = build_capacitance_solver(mask, dx, dy, lambda_=0.0, base_bc="dst")

n_boundary = len(solver_dst._j_b)
print(f"Number of inner-boundary points: {n_boundary}")
print(f"Base spectral solver: {solver_dst.base_bc}")
print(f"Helmholtz parameter: {solver_dst.lambda_}")

# %% [markdown]
# ## Section 3: Solve Poisson on the Masked Domain
#
# We set the RHS to ones inside the mask and zeros outside, then solve.

# %%
rhs = jnp.where(jnp.array(mask), 1.0, 0.0)
psi = solver_dst(rhs)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

im0 = axes[0].imshow(np.array(rhs), origin="lower", cmap="viridis")
axes[0].set_title("RHS (ones inside mask)")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(np.array(psi), origin="lower", cmap="RdBu_r")
axes[1].set_title("Solution psi")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

for ax in axes:
    ax.set_xlabel("i")
    ax.set_ylabel("j")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Section 4: Verify Boundary Enforcement
#
# The capacitance method enforces $\psi = 0$ at the inner-boundary points.
# Let us check.

# %%
# Extract solution values at inner-boundary points
psi_at_boundary = psi[solver_dst._j_b, solver_dst._i_b]

print(f"Number of boundary points: {n_boundary}")
print(f"Max |psi| at boundary: {jnp.max(jnp.abs(psi_at_boundary)):.2e}")
print(f"Mean |psi| at boundary: {jnp.mean(jnp.abs(psi_at_boundary)):.2e}")

# %%
# Visualize the boundary points and their psi values
boundary_field = jnp.zeros((Ny, Nx))
boundary_field = boundary_field.at[solver_dst._j_b, solver_dst._i_b].set(
    psi_at_boundary
)

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(np.array(boundary_field), origin="lower", cmap="RdBu_r")
ax.set_title("psi at inner-boundary points")
ax.set_xlabel("i")
ax.set_ylabel("j")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Section 5: Compare DST vs FFT Base Solver
#
# The capacitance method can use different rectangular base solvers.
# Inside the masked domain, the solutions should agree (they solve the
# same PDE with the same mask-boundary conditions).

# %%
solver_fft = build_capacitance_solver(mask, dx, dy, lambda_=0.0, base_bc="fft")

psi_from_dst = solver_dst(rhs)
psi_from_fft = solver_fft(rhs)

# Difference inside the mask
diff_inside = jnp.where(jnp.array(mask), jnp.abs(psi_from_dst - psi_from_fft), 0.0)

print(f"Max |DST - FFT| inside mask: {jnp.max(diff_inside):.2e}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].imshow(np.array(psi_from_dst), origin="lower", cmap="RdBu_r")
axes[0].set_title("DST base")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(np.array(psi_from_fft), origin="lower", cmap="RdBu_r")
axes[1].set_title("FFT base")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

im2 = axes[2].imshow(np.array(diff_inside), origin="lower", cmap="hot")
axes[2].set_title("|DST - FFT| inside mask")
plt.colorbar(im2, ax=axes[2], shrink=0.8)

for ax in axes:
    ax.set_xlabel("i")
    ax.set_ylabel("j")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Section 6: Helmholtz with Nonzero Lambda
#
# Setting `lambda_ > 0` solves the Helmholtz equation
# $(\nabla^2 - \lambda)\psi = f$ instead of pure Poisson.
# The solution is more localized because the $-\lambda\psi$ term
# acts as a damping/screening term.

# %%
solver_helmholtz = build_capacitance_solver(
    mask, dx, dy, lambda_=1.0, base_bc="dst"
)

psi_poisson = solver_dst(rhs)
psi_helmholtz = solver_helmholtz(rhs)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

vmax = max(float(jnp.max(jnp.abs(psi_poisson))), float(jnp.max(jnp.abs(psi_helmholtz))))

im0 = axes[0].imshow(
    np.array(psi_poisson), origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax
)
axes[0].set_title("Poisson (lambda=0)")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(
    np.array(psi_helmholtz), origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax
)
axes[1].set_title("Helmholtz (lambda=1)")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

for ax in axes:
    ax.set_xlabel("i")
    ax.set_ylabel("j")

plt.suptitle("Capacitance solver: Poisson vs Helmholtz", fontsize=14)
plt.tight_layout()
plt.show()

# %%
print(f"Poisson max |psi|: {jnp.max(jnp.abs(psi_poisson)):.4f}")
print(f"Helmholtz max |psi|: {jnp.max(jnp.abs(psi_helmholtz)):.4f}")
print("The Helmholtz solution is smaller due to the screening term.")
