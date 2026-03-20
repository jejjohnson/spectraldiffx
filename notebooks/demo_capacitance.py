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
# # Capacitance Matrix Method: Poisson & Helmholtz on Irregular Domains
#
# ## What this tutorial covers
#
# Standard spectral elliptic solvers (DST, DCT, FFT) assume a
# **rectangular domain**. But many real-world problems live on
# irregular domains:
#
# - **Ocean basins** bounded by coastlines
# - **Aerodynamic bodies** with complex cross-sections
# - **Semiconductor devices** with etched geometries
# - **Medical imaging** domains defined by organ boundaries
#
# The **capacitance matrix method** (Buzbee, Golub & Nielson, 1970)
# extends fast rectangular spectral solvers to handle **masked
# (irregular) domains** where $\psi = 0$ is enforced at the boundary
# of an arbitrary interior region.
#
# This notebook demonstrates the full workflow:
#
# 1. Build a circular ocean-basin mask
# 2. Construct the capacitance solver (offline precomputation)
# 3. Solve Poisson on the masked domain
# 4. Verify boundary enforcement
# 5. Compare DST vs FFT base solvers
# 6. Demonstrate Helmholtz screening ($\lambda > 0$)

# %% [markdown]
# ## 1. How the Capacitance Matrix Method Works
#
# ### The problem
#
# We want to solve $\nabla^2 \psi = f$ (or $(\nabla^2 - \lambda)\psi = f$)
# on an irregular sub-domain $\Omega$ of a rectangle, with $\psi = 0$ on
# the boundary $\partial\Omega$.
#
# ### The idea
#
# ```
# Masked domain (ocean):        Full rectangular domain:
# ┌─────────────────┐           ┌─────────────────┐
# │ ░░░░░░░░░░░░░░░ │           │                 │
# │ ░░░┌───────┐░░░ │           │                 │
# │ ░░░│ ocean │░░░ │    →      │   spectral      │
# │ ░░░│ (psi) │░░░ │           │   solve here    │
# │ ░░░└───────┘░░░ │           │                 │
# │ ░░░░░░░░░░░░░░░ │           │                 │
# └─────────────────┘           └─────────────────┘
# ░ = land (psi = 0)
# ```
#
# ### Algorithm (three steps)
#
# **Offline** (done once per mask):
#
# 1. Identify the $N_b$ inner-boundary points — ocean cells adjacent to land.
# 2. For each boundary point $x_j$, solve $\nabla^2 G_j = \delta_{x_j}$ on
#    the full rectangle. This gives the Green's function column $G_j$.
# 3. Build the $N_b \times N_b$ **capacitance matrix** $C$ where
#    $C_{ij} = G_j(x_i)$ (Green's function of point $j$ evaluated at
#    boundary point $i$). Invert $C$.
#
# **Online** (each solve):
#
# $$\boxed{\psi = \text{mask} \cdot \bigl[\psi_0 - G \, C^{-1} \, \psi_0\big|_{\partial\Omega}\bigr]}$$
#
# where $\psi_0$ is the uncorrected full-domain solve.
#
# In LaTeX:
#
# 1. Solve $\nabla^2 \psi_0 = f$ on the full rectangle → fast spectral solve, $O(N \log N)$
# 2. Extract $\psi_0$ at boundary points: $\mathbf{b} = \psi_0\big|_{\partial\Omega}$
# 3. Compute correction coefficients: $\mathbf{c} = C^{-1}\,\mathbf{b}$, cost $O(N_b^2)$
# 4. Correct: $\psi = \text{mask} \cdot (\psi_0 - \sum_j c_j\,G_j)$
#
# **Total online cost**: $O(N \log N + N_b^2)$ — still dominated by the
# spectral transform for typical masks where $N_b \ll N$.

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

jax.config.update("jax_enable_x64", True)

IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "images" / "demo_capacitance"
IMG_DIR.mkdir(parents=True, exist_ok=True)

from spectraldiffx import build_capacitance_solver

# %% [markdown]
# ## 2. Build a Circular Ocean Basin
#
# We create a binary mask on a $32 \times 32$ grid representing a circular
# ocean basin. Points inside the circle are "ocean" (`True`); points
# outside are "land" (`False`).
#
# **Parameter choices**:
#
# - Grid size $32 \times 32$: small enough for fast precomputation, large
#   enough to see smooth solutions.
# - Radius $= 0.35 \cdot \min(N_y, N_x)$: leaves a comfortable margin
#   from the rectangular boundary so the circle is fully interior.
# - $\Delta x = \Delta y = 1.0$: unit spacing for simplicity.

# %%
Ny, Nx = 32, 32
dx, dy = 1.0, 1.0

# Cell-center coordinates
j_idx, i_idx = np.mgrid[0:Ny, 0:Nx]
center_y, center_x = Ny / 2, Nx / 2
radius = 0.35 * min(Ny, Nx)

# Distance from center
dist = np.sqrt((i_idx - center_x) ** 2 + (j_idx - center_y) ** 2)

# Binary mask: True = ocean interior
mask = dist < radius

print(f"Grid shape:      ({Ny}, {Nx})")
print(f"Grid spacing:    dx={dx}, dy={dy}")
print(f"Circle center:   ({center_x}, {center_y})")
print(f"Circle radius:   {radius}")
print(f"Interior points: {mask.sum()} / {Ny * Nx} ({100 * mask.sum() / (Ny * Nx):.1f}%)")
print(f"Mask shape:      {mask.shape}")
print(f"Mask dtype:      {mask.dtype}")

# %%
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(mask.astype(float), origin="lower", cmap="Blues", vmin=0, vmax=1)
ax.set_title("Ocean basin mask (blue = ocean interior)")
ax.set_xlabel("i")
ax.set_ylabel("j")
plt.tight_layout()
fig.savefig(IMG_DIR / "ocean_basin_mask.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Ocean basin mask](../images/demo_capacitance/ocean_basin_mask.png)

# %% [markdown]
# ## 3. Build the Capacitance Solver
#
# `build_capacitance_solver` performs all the offline precomputation:
#
# 1. Detect inner-boundary points (ocean cells with at least one land neighbor)
# 2. Compute Green's function columns ($N_b$ spectral solves)
# 3. Build and invert the capacitance matrix
#
# The returned `CapacitanceSolver` is an `eqx.Module` — a frozen pytree
# that stores all precomputed data and is callable.

# %%
solver_dst = build_capacitance_solver(mask, dx, dy, lambda_=0.0, base_bc="dst")

n_boundary = len(solver_dst._j_b)
print(f"Base spectral solver:       {solver_dst.base_bc}")
print(f"Helmholtz parameter:        lambda = {solver_dst.lambda_}")
print(f"Number of boundary points:  N_b = {n_boundary}")
print(f"Capacitance matrix size:    ({n_boundary}, {n_boundary})")
print(f"Total grid points:          N = {Ny * Nx}")
print(f"Ratio N_b / N:              {n_boundary / (Ny * Nx):.2%}")

# %% [markdown]
# The number of boundary points $N_b$ is much smaller than $N$, which is
# what makes the method efficient. The online cost per solve is
# $O(N \log N + N_b^2)$ rather than $O(N^2)$ or $O(N^{3/2})$.

# %% [markdown]
# ## 4. Solve Poisson on the Masked Domain
#
# We create a Gaussian-bump RHS centered in the basin and solve
# $\nabla^2 \psi = f$ with $\psi = 0$ on the basin boundary.

# %%
# Gaussian bump RHS (only nonzero inside the mask)
sigma_rhs = 0.25 * radius  # width relative to basin radius
rhs = jnp.where(
    jnp.array(mask),
    jnp.exp(-((i_idx - center_x) ** 2 + (j_idx - center_y) ** 2) / (2 * sigma_rhs**2)),
    0.0,
)

print(f"RHS shape: {rhs.shape}")
print(f"RHS range: [{float(jnp.min(rhs)):.4f}, {float(jnp.max(rhs)):.4f}]")
print(f"RHS nonzero points: {int(jnp.sum(rhs > 0))}")

# %%
psi = solver_dst(rhs)

print(f"Solution shape: {psi.shape}")
print(f"Solution range: [{float(jnp.min(psi)):.4f}, {float(jnp.max(psi)):.4f}]")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

im0 = axes[0].imshow(np.array(rhs), origin="lower", cmap="viridis")
axes[0].set_title("RHS (Gaussian bump)")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(np.array(psi), origin="lower", cmap="RdBu_r")
axes[1].set_title(r"Solution $\psi$")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

for ax in axes:
    ax.set_xlabel("i")
    ax.set_ylabel("j")

plt.suptitle(r"Capacitance solver: $\nabla^2\psi = f$ on circular basin", fontsize=13)
plt.tight_layout()
fig.savefig(IMG_DIR / "rhs_and_solution.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![RHS and Poisson solution](../images/demo_capacitance/rhs_and_solution.png)

# %% [markdown]
# The solution is smooth inside the basin and drops to zero at the
# boundary, as enforced by the capacitance correction.

# %% [markdown]
# ## 5. Boundary Enforcement Verification
#
# The whole point of the capacitance method is to enforce $\psi = 0$ at
# the inner-boundary points. Let us verify this quantitatively.

# %% [markdown]
# ### Before vs After Capacitance Correction
#
# To appreciate what the capacitance method does, we compare the
# **uncorrected** rectangular DST solve (which ignores the mask) with the
# **corrected** capacitance solve.

# %%
from scipy.ndimage import binary_dilation
from spectraldiffx import solve_poisson_dst

# Uncorrected: solve on full rectangle, ignoring the mask
psi_rect = solve_poisson_dst(rhs, dx, dy)

# Detect inner boundary for visualization
exterior = ~mask
struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
dilated = binary_dilation(exterior, structure=struct)
inner_boundary = mask & dilated
j_b_vis, i_b_vis = np.where(inner_boundary)

print(f"psi_rect shape:    {psi_rect.shape}")
print(f"Max |psi| at boundary (uncorrected): {float(jnp.abs(psi_rect[j_b_vis, i_b_vis]).max()):.4f}")
print(f"Max |psi| at boundary (capacitance): {float(jnp.abs(psi[j_b_vis, i_b_vis]).max()):.2e}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Uncorrected
psi_rect_masked = np.array(psi_rect) * mask
im0 = axes[0].imshow(psi_rect_masked, origin="lower", cmap="RdBu_r")
axes[0].scatter(i_b_vis, j_b_vis, c="red", s=12, zorder=5, label="Boundary pts")
bnd_max_rect = float(jnp.abs(psi_rect[j_b_vis, i_b_vis]).max())
axes[0].set_title(f"Rectangular solve (no correction)\nmax |ψ| at boundary = {bnd_max_rect:.3f}")
axes[0].legend(fontsize=8, loc="upper right")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

# Corrected
psi_masked = np.array(psi) * mask
im1 = axes[1].imshow(psi_masked, origin="lower", cmap="RdBu_r")
axes[1].scatter(i_b_vis, j_b_vis, c="red", s=12, zorder=5, label="Boundary pts")
bnd_max_cap = float(jnp.abs(psi[j_b_vis, i_b_vis]).max())
axes[1].set_title(f"Capacitance solve (corrected)\nmax |ψ| at boundary = {bnd_max_cap:.1e}")
axes[1].legend(fontsize=8, loc="upper right")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

for ax in axes:
    ax.set_xlabel("i")
    ax.set_ylabel("j")

plt.suptitle("Boundary Enforcement: Before vs After Capacitance Correction", fontsize=13)
plt.tight_layout()
fig.savefig(IMG_DIR / "boundary_before_after.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Boundary before vs after correction](../images/demo_capacitance/boundary_before_after.png)
#
# The uncorrected rectangular solve has significant nonzero values at the
# boundary points (red dots). The capacitance correction drives these to
# machine precision.

# %%
# Extract solution values at inner-boundary points
psi_at_boundary = psi[solver_dst._j_b, solver_dst._i_b]

print(f"Number of boundary points:  {n_boundary}")
print(f"Max  |psi| at boundary:     {float(jnp.max(jnp.abs(psi_at_boundary))):.2e}")
print(f"Mean |psi| at boundary:     {float(jnp.mean(jnp.abs(psi_at_boundary))):.2e}")
print(f"psi_at_boundary shape:      {psi_at_boundary.shape}")

# %%
# Visualize: place boundary-point psi values on a field
boundary_field = jnp.zeros((Ny, Nx))
boundary_field = boundary_field.at[solver_dst._j_b, solver_dst._i_b].set(
    psi_at_boundary
)

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(np.array(boundary_field), origin="lower", cmap="RdBu_r")
ax.set_title(r"$\psi$ at inner-boundary points (should be $\approx 0$)")
ax.set_xlabel("i")
ax.set_ylabel("j")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
fig.savefig(IMG_DIR / "boundary_enforcement.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Psi at inner-boundary points](../images/demo_capacitance/boundary_enforcement.png)

# %% [markdown]
# The boundary values are at machine precision ($\sim 10^{-14}$),
# confirming that the capacitance correction works as expected.

# %% [markdown]
# ## 6. DST vs FFT Base Solver
#
# The capacitance method works with **any** fast rectangular solver as
# the base. Two natural choices are:
#
# - `base_bc="dst"` — Discrete Sine Transform (Dirichlet on rectangle)
# - `base_bc="fft"` — FFT (periodic on rectangle)
#
# Inside the masked domain, both should produce the same solution because
# the capacitance correction enforces $\psi = 0$ at the mask boundary
# regardless of the rectangular boundary conditions.

# %%
solver_fft = build_capacitance_solver(mask, dx, dy, lambda_=0.0, base_bc="fft")
print(f"FFT-based solver boundary points: {len(solver_fft._j_b)}")

# Solve with both bases
psi_from_dst = solver_dst(rhs)
psi_from_fft = solver_fft(rhs)

# Difference inside the mask only
diff_inside = jnp.where(jnp.array(mask), jnp.abs(psi_from_dst - psi_from_fft), 0.0)

print(f"\npsi_from_dst shape: {psi_from_dst.shape}")
print(f"psi_from_fft shape: {psi_from_fft.shape}")
print(f"Max |DST - FFT| inside mask: {float(jnp.max(diff_inside)):.2e}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].imshow(np.array(psi_from_dst), origin="lower", cmap="RdBu_r")
axes[0].set_title("DST base solver")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(np.array(psi_from_fft), origin="lower", cmap="RdBu_r")
axes[1].set_title("FFT base solver")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

im2 = axes[2].imshow(np.array(diff_inside), origin="lower", cmap="hot")
axes[2].set_title("|DST - FFT| inside mask")
plt.colorbar(im2, ax=axes[2], shrink=0.8)

for ax in axes:
    ax.set_xlabel("i")
    ax.set_ylabel("j")

plt.suptitle("Capacitance solver: DST vs FFT base comparison", fontsize=13)
plt.tight_layout()
fig.savefig(IMG_DIR / "dst_vs_fft_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![DST vs FFT base solver comparison](../images/demo_capacitance/dst_vs_fft_comparison.png)

# %% [markdown]
# The two solutions agree to near machine precision inside the mask.
# The choice of base solver is a matter of convenience — both give the
# same physical answer.

# %% [markdown]
# ## 7. Helmholtz Screening ($\lambda > 0$)
#
# Setting $\lambda > 0$ solves the Helmholtz equation
#
# $$(\nabla^2 - \lambda)\,\psi = f$$
#
# instead of pure Poisson. The $-\lambda\,\psi$ term acts as **exponential
# screening**: it penalizes large $|\psi|$ and localizes the response
# near the source. The characteristic decay length is
# $\ell \sim 1/\sqrt{\lambda}$.
#
# **Physical interpretation**: In geophysical fluid dynamics, the screened
# Poisson equation arises in quasi-geostrophic potential vorticity
# inversion, where $\lambda$ is related to the Rossby deformation radius.
# Larger $\lambda$ means stronger stratification and more localized
# response to forcing.
#
# **Important**: We use **independent color scales** for the two panels
# because the Helmholtz solution can be $\sim 30\times$ smaller than the
# Poisson solution. A shared color scale would make the Helmholtz panel
# appear blank.

# %%
# Build a Helmholtz solver with lambda = 1.0
lambda_helm = 1.0
solver_helmholtz = build_capacitance_solver(
    mask, dx, dy, lambda_=lambda_helm, base_bc="dst"
)
print(f"Helmholtz solver: lambda = {solver_helmholtz.lambda_}")

# Solve both Poisson and Helmholtz with the same RHS
psi_poisson = solver_dst(rhs)       # lambda = 0
psi_helmholtz = solver_helmholtz(rhs)  # lambda = 1

print(f"\nPoisson   max |psi|: {float(jnp.max(jnp.abs(psi_poisson))):.6f}")
print(f"Helmholtz max |psi|: {float(jnp.max(jnp.abs(psi_helmholtz))):.6f}")
ratio = float(jnp.max(jnp.abs(psi_poisson)) / jnp.max(jnp.abs(psi_helmholtz)))
print(f"Ratio (Poisson / Helmholtz): {ratio:.1f}x")
print(f"\nScreening localizes the response: the Helmholtz solution is ~{ratio:.0f}x smaller.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# INDEPENDENT color scales — critical for visual comparison
vmax_p = float(jnp.max(jnp.abs(psi_poisson)))
vmax_h = float(jnp.max(jnp.abs(psi_helmholtz)))

im0 = axes[0].imshow(
    np.array(psi_poisson), origin="lower", cmap="RdBu_r",
    vmin=-vmax_p, vmax=vmax_p,
)
axes[0].set_title(r"Poisson ($\lambda = 0$)")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(
    np.array(psi_helmholtz), origin="lower", cmap="RdBu_r",
    vmin=-vmax_h, vmax=vmax_h,
)
axes[1].set_title(rf"Helmholtz ($\lambda = {lambda_helm}$)")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

for ax in axes:
    ax.set_xlabel("i")
    ax.set_ylabel("j")

fig.suptitle("Capacitance solver: Poisson vs Helmholtz (independent color scales)", fontsize=13)
plt.tight_layout()
fig.savefig(IMG_DIR / "poisson_vs_helmholtz.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Poisson vs Helmholtz comparison](../images/demo_capacitance/poisson_vs_helmholtz.png)

# %% [markdown]
# **Observations**:
#
# - The **Poisson solution** ($\lambda = 0$) spreads broadly across the
#   entire basin, with values that decay slowly from the source.
# - The **Helmholtz solution** ($\lambda = 1$) is concentrated near the
#   Gaussian source and decays exponentially away from it.
# - The amplitude ratio confirms the screening: the Helmholtz peak is
#   much smaller than the Poisson peak.
# - Both solutions satisfy $\psi = 0$ at the basin boundary (enforced by
#   the capacitance correction).
#
# ## Summary
#
# | Step | What happens | Cost |
# |------|-------------|------|
# | **Offline** | Detect boundary, build Green's functions, invert $C$ | $O(N_b \cdot N \log N + N_b^3)$ |
# | **Online** | Full-domain spectral solve + capacitance correction | $O(N \log N + N_b^2)$ |
#
# The capacitance matrix method turns an irregular-domain PDE into a
# sequence of rectangular spectral solves plus a small dense linear
# system, combining the speed of FFT-based methods with the flexibility
# of arbitrary domain shapes.
