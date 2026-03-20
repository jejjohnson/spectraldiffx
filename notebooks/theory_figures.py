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
# # Theory Figures for spectraldiffx Documentation
#
# This notebook generates all the theory/documentation figures for spectraldiffx.
# Each section builds a figure that illustrates a key concept: basis functions,
# eigenvalue structure, solver behaviour under different boundary conditions,
# eigenfunction recovery, the capacitance matrix method, and Parseval's theorem.

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

jax.config.update("jax_enable_x64", True)

IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "images" / "theory_figures"
IMG_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150
COMPRESS = {"dpi": DPI, "bbox_inches": "tight", "pad_inches": 0.15}

# %% [markdown]
# ---
# ## 1. DST-I and DCT-II Basis Functions
#
# Spectral solvers expand the solution in a complete set of **basis functions**
# chosen to satisfy the boundary conditions automatically.
#
# **DST-I** (Discrete Sine Transform, Type I) uses sine basis functions:
#
# $$
# \phi_k(n) = \sin\!\left(\frac{\pi\,(n+1)(k+1)}{N+1}\right),
# \quad k = 0, \ldots, N-1
# $$
#
# These vanish at $n = -1$ and $n = N$, which corresponds to **Dirichlet**
# (zero-value) boundary conditions at both ends of the domain.
#
# **DCT-II** (Discrete Cosine Transform, Type II) uses cosine basis functions:
#
# $$
# \phi_k(n) = \cos\!\left(\frac{\pi\,k\,(2n+1)}{2N}\right),
# \quad k = 0, \ldots, N-1
# $$
#
# The derivative of each cosine basis function vanishes at the boundaries,
# which corresponds to **Neumann** (zero-gradient) boundary conditions.
#
# Below we plot the first four basis functions for each transform with $N = 8$.

# %%
N = 8
n = np.arange(N)
print(f"Grid points: n = {n}  (N = {N})")

fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharex=True, sharey=True)
fig.suptitle("Spectral Basis Functions  (N = 8)", fontsize=13, y=1.02)

# DST-I: phi_k(n) = sin(pi (n+1)(k+1) / (N+1))
for k in range(4):
    ax = axes[0, k]
    vals = np.sin(np.pi * (n + 1) * (k + 1) / (N + 1))
    ax.stem(
        n,
        vals,
        linefmt="C0-",
        markerfmt="C0o",
        basefmt="k-",
    )
    ax.set_title(f"DST-I  k={k + 1}", fontsize=10)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(0, color="k", lw=0.5)
    if k == 0:
        ax.set_ylabel("Dirichlet (sin)")

# DCT-II: phi_k(n) = cos(pi k (2n+1) / (2N))
for k in range(4):
    ax = axes[1, k]
    vals = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    ax.stem(
        n,
        vals,
        linefmt="C1-",
        markerfmt="C1o",
        basefmt="k-",
    )
    ax.set_title(f"DCT-II  k={k}", fontsize=10)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("n")
    if k == 0:
        ax.set_ylabel("Neumann (cos)")

fig.tight_layout()
fig.savefig(IMG_DIR / "basis_functions.png", **COMPRESS)
plt.close(fig)
print(f"Saved: {IMG_DIR / 'basis_functions.png'}")

# %% [markdown]
# ![Spectral basis functions for DST-I (Dirichlet) and DCT-II (Neumann)](../docs/images/theory_figures/basis_functions.png)

# %% [markdown]
# ---
# ## 2. Discrete vs Continuous Eigenvalues
#
# When we discretize the 1D Laplacian $\partial^2/\partial x^2$ with second-order
# centred finite differences on a grid of spacing $dx$, the eigenvalues are:
#
# $$
# \lambda_k^{\text{discrete}} = \frac{2(\cos\theta_k - 1)}{dx^2}
# $$
#
# where $\theta_k$ depends on the transform type (DST-I, DCT-II, or FFT).
# The continuous Laplacian, by contrast, has eigenvalues $-k^2$.
#
# At **low wavenumbers** the two agree closely, but near the **Nyquist frequency**
# the discrete eigenvalues flatten out while the continuous values keep growing
# quadratically. This divergence means that high-wavenumber modes are damped less
# aggressively by the discrete Laplacian than one might expect from continuum
# theory -- an important consideration for spectral solvers.

# %%
from spectraldiffx import dct2_eigenvalues, dst1_eigenvalues, fft_eigenvalues

N = 32
dx = 1.0

eig_dst = dst1_eigenvalues(N, dx)
eig_dct = dct2_eigenvalues(N, dx)
eig_fft = fft_eigenvalues(N, dx)

print(f"DST-I eigenvalues shape: {jnp.array(eig_dst).shape}")
print(f"DCT-II eigenvalues shape: {jnp.array(eig_dct).shape}")
print(f"FFT eigenvalues shape:   {jnp.array(eig_fft).shape}")

k_dst = np.arange(N)
k_dct = np.arange(N)
k_fft = np.arange(N)

# Continuous eigenvalues for comparison
cont_dst = -((np.pi * (k_dst + 1) / (N + 1)) ** 2) / dx**2
cont_dct = -((np.pi * k_dct / N) ** 2) / dx**2  # approximate
cont_fft = -((2 * np.pi * k_fft / (N * dx)) ** 2)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

for ax, eig, cont, label, k in [
    (axes[0], eig_dst, cont_dst, "DST-I (Dirichlet)", k_dst),
    (axes[1], eig_dct, cont_dct, "DCT-II (Neumann)", k_dct),
    (axes[2], eig_fft, cont_fft, "FFT (Periodic)", k_fft),
]:
    ax.plot(k, np.array(eig), "o-", ms=4, label="Discrete FD", color="C0")
    ax.plot(k, cont, "s--", ms=3, label=r"Continuous $-k^2$", color="C3", alpha=0.7)
    ax.set_xlabel("Mode index k")
    ax.set_ylabel(r"$\lambda_k$")
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    f"Discrete vs Continuous Laplacian Eigenvalues  (N = {N})", fontsize=12, y=1.03
)
fig.tight_layout()
fig.savefig(IMG_DIR / "eigenvalues_comparison.png", **COMPRESS)
plt.close(fig)
print(f"Saved: {IMG_DIR / 'eigenvalues_comparison.png'}")

# %% [markdown]
# ![Discrete vs continuous Laplacian eigenvalues for DST-I, DCT-II, and FFT](../docs/images/theory_figures/eigenvalues_comparison.png)

# %% [markdown]
# ---
# ## 3. Poisson Equation with Three BC Types
#
# We solve the 2D Poisson equation $\nabla^2 \psi = f$ on a $64 \times 64$ grid
# with a Gaussian-bump right-hand side, using three different boundary conditions:
#
# - **Dirichlet** ($\psi = 0$ on the boundary) via DST
# - **Neumann** ($\partial\psi/\partial n = 0$ on the boundary) via DCT
# - **Periodic** (wrap-around) via FFT
#
# The boundary conditions dramatically shape the solution. Dirichlet forces the
# solution to zero at the edges, pulling it downward. Neumann allows a free
# boundary that adjusts to maintain zero flux. Periodic wraps the domain into a
# torus, so the solution "sees" copies of itself.
#
# Note: the RHS is made zero-mean for compatibility with the Neumann and periodic
# solvers (both require $\int f = 0$ for a solution to exist).

# %%
from spectraldiffx import solve_poisson_dct, solve_poisson_dst, solve_poisson_fft

Nx, Ny = 64, 64
dx = dy = 1.0

# Gaussian bump RHS
j, i = jnp.mgrid[0:Ny, 0:Nx]
cx, cy = Nx / 2, Ny / 2
rhs = jnp.exp(-((i - cx) ** 2 + (j - cy) ** 2) / (2 * 8**2))
rhs = rhs - rhs.mean()  # zero mean for compatibility

print(f"RHS shape: {rhs.shape}")
print(f"RHS mean:  {float(rhs.mean()):.2e}")

psi_dst = solve_poisson_dst(rhs, dx, dy)
psi_dct = solve_poisson_dct(rhs, dx, dy)
psi_fft = solve_poisson_fft(rhs, dx, dy)

print(f"psi_dst shape: {psi_dst.shape},  range: [{float(psi_dst.min()):.4f}, {float(psi_dst.max()):.4f}]")
print(f"psi_dct shape: {psi_dct.shape},  range: [{float(psi_dct.min()):.4f}, {float(psi_dct.max()):.4f}]")
print(f"psi_fft shape: {psi_fft.shape},  range: [{float(psi_fft.min()):.4f}, {float(psi_fft.max()):.4f}]")

# %%
fig, axes = plt.subplots(1, 4, figsize=(14, 3))

im0 = axes[0].imshow(np.array(rhs), cmap="RdBu_r", origin="lower")
axes[0].set_title("RHS  f(x, y)")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

vmin = min(float(psi_dst.min()), float(psi_dct.min()), float(psi_fft.min()))
vmax = max(float(psi_dst.max()), float(psi_dct.max()), float(psi_fft.max()))

for ax, psi, label in [
    (axes[1], psi_dst, r"Dirichlet ($\psi=0$)"),
    (axes[2], psi_dct, r"Neumann ($\partial\psi/\partial n=0$)"),
    (axes[3], psi_fft, "Periodic"),
]:
    im = ax.imshow(
        np.array(psi), cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax
    )
    ax.set_title(label, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(
    r"Poisson Solve: $\nabla^2 \psi = f$ with Different BCs", fontsize=12, y=1.03
)
fig.tight_layout()
fig.savefig(IMG_DIR / "solver_comparison.png", **COMPRESS)
plt.close(fig)
print(f"Saved: {IMG_DIR / 'solver_comparison.png'}")

# %% [markdown]
# ![Poisson solutions with Dirichlet, Neumann, and periodic BCs](../docs/images/theory_figures/solver_comparison.png)

# %% [markdown]
# ---
# ## 4. Eigenfunction Recovery
#
# A powerful validation of spectral solvers is **eigenfunction recovery**. If the
# right-hand side $f$ is itself a discrete eigenfunction of the Laplacian, then
# the spectral solve should recover the exact eigenfunction $\psi$ to machine
# precision.
#
# For DST-I on an $N$-point grid, the eigenfunctions are:
#
# $$
# \psi_{k_x, k_y}(i, j) = \sin\!\left(\frac{\pi k_x (i+1)}{N_x+1}\right)
# \sin\!\left(\frac{\pi k_y (j+1)}{N_y+1}\right)
# $$
#
# with eigenvalue $\lambda_{k_x} + \lambda_{k_y}$. Setting $f = (\lambda_{k_x}
# + \lambda_{k_y})\,\psi$ and solving $\nabla^2 \psi = f$ should return $\psi$
# exactly. The error is at the level of floating-point round-off (~$10^{-15}$),
# confirming that the spectral solve is **exact** for discrete eigenfunctions --
# no truncation error, only machine epsilon.

# %%
from spectraldiffx import dst1_eigenvalues, solve_poisson_dst

Nx, Ny = 32, 32
dx = dy = 1.0
kx, ky = 2, 3

i = jnp.arange(Nx)
j = jnp.arange(Ny)
psi_exact = (
    jnp.sin(jnp.pi * kx * (i + 1) / (Nx + 1))[None, :]
    * jnp.sin(jnp.pi * ky * (j + 1) / (Ny + 1))[:, None]
)
print(f"psi_exact shape: {psi_exact.shape}")

eigx = dst1_eigenvalues(Nx, dx)
eigy = dst1_eigenvalues(Ny, dy)
print(f"eigx shape: {jnp.array(eigx).shape},  eigx[kx-1] = {float(eigx[kx - 1]):.6f}")
print(f"eigy shape: {jnp.array(eigy).shape},  eigy[ky-1] = {float(eigy[ky - 1]):.6f}")

rhs = (eigx[kx - 1] + eigy[ky - 1]) * psi_exact
print(f"rhs shape: {rhs.shape}")

psi_computed = solve_poisson_dst(rhs, dx, dy)
error = psi_computed - psi_exact
print(f"psi_computed shape: {psi_computed.shape}")
print(f"Max absolute error: {float(jnp.max(jnp.abs(error))):.2e}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(11, 3))

im0 = axes[0].imshow(np.array(psi_exact), cmap="RdBu_r", origin="lower")
axes[0].set_title(r"Exact $\psi$ (eigenfunction)")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(np.array(psi_computed), cmap="RdBu_r", origin="lower")
axes[1].set_title(r"Computed $\psi$ (DST-I)")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

im2 = axes[2].imshow(np.array(error), cmap="RdBu_r", origin="lower")
axes[2].set_title(f"Error (max = {float(jnp.max(jnp.abs(error))):.1e})")
plt.colorbar(im2, ax=axes[2], shrink=0.8)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(
    f"Eigenfunction Recovery: DST-I Poisson  (kx={kx}, ky={ky})",
    fontsize=12,
    y=1.03,
)
fig.tight_layout()
fig.savefig(IMG_DIR / "eigenfunction_recovery.png", **COMPRESS)
plt.close(fig)
print(f"Saved: {IMG_DIR / 'eigenfunction_recovery.png'}")

# %% [markdown]
# ![Eigenfunction recovery showing exact solution, computed solution, and machine-precision error](../docs/images/theory_figures/eigenfunction_recovery.png)

# %% [markdown]
# ---
# ## 5. Capacitance Matrix Solver on an Irregular Domain
#
# Real geophysical domains (ocean basins, lakes) are rarely rectangular. The
# **capacitance matrix method** extends fast spectral solvers to irregular domains
# defined by a binary mask.
#
# The idea is:
# 1. Solve the Poisson equation on the full rectangular domain using a fast
#    spectral solver (DST/DCT/FFT).
# 2. Identify the **inner boundary points** -- mask points adjacent to the
#    exterior -- where the rectangular solution violates the desired BCs.
# 3. Precompute a small dense matrix (the **capacitance matrix**) that maps
#    boundary corrections to their effect on the boundary values.
# 4. At solve time, apply a correction using this matrix to enforce the BCs
#    exactly at all inner boundary points.
#
# The cost is dominated by the rectangular solve ($O(N \log N)$), plus a small
# dense solve of size $n_b \times n_b$ where $n_b$ is the number of boundary
# points.
#
# Below we demonstrate this on a circular ocean basin mask.

# %%
from scipy.ndimage import binary_dilation

from spectraldiffx import build_capacitance_solver

Ny, Nx = 32, 32
dx = dy = 1.0

# Circular mask
j, i = np.mgrid[0:Ny, 0:Nx]
cy, cx = Ny / 2, Nx / 2
mask = ((j - cy) ** 2 + (i - cx) ** 2) < (0.35 * min(Ny, Nx)) ** 2
print(f"Mask shape: {mask.shape},  ocean points: {mask.sum()},  land points: {(~mask).sum()}")

# Inner boundary
exterior = ~mask
struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
dilated = binary_dilation(exterior, structure=struct)
inner_boundary = mask & dilated
j_b, i_b = np.where(inner_boundary)
print(f"Inner boundary points: {len(j_b)}")

# Build solver and solve
solver = build_capacitance_solver(mask, dx, dy, base_bc="dst")
rhs = jnp.array(mask, dtype=float)
print(f"RHS shape: {rhs.shape}")

psi = solver(rhs)
print(f"Solution shape: {psi.shape}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

# Mask with boundary
mask_vis = np.zeros((Ny, Nx, 3))
mask_vis[mask] = [0.7, 0.85, 1.0]  # light blue = ocean
mask_vis[~mask] = [0.85, 0.75, 0.6]  # tan = land
mask_vis[inner_boundary] = [1.0, 0.3, 0.3]  # red = boundary
axes[0].imshow(mask_vis, origin="lower")
axes[0].set_title(f"Mask  ({len(j_b)} boundary pts)", fontsize=10)

# RHS
im1 = axes[1].imshow(np.array(rhs), cmap="Blues", origin="lower")
axes[1].set_title("RHS (ones inside mask)", fontsize=10)
plt.colorbar(im1, ax=axes[1], shrink=0.8)

# Solution
psi_masked = np.array(psi) * mask
im2 = axes[2].imshow(psi_masked, cmap="RdBu_r", origin="lower")
axes[2].set_title(r"Solution $\psi$ (capacitance)", fontsize=10)
plt.colorbar(im2, ax=axes[2], shrink=0.8)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("Capacitance Matrix Solver: Circular Ocean Basin", fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(IMG_DIR / "capacitance_mask_solution.png", **COMPRESS)
plt.close(fig)
print(f"Saved: {IMG_DIR / 'capacitance_mask_solution.png'}")

# %% [markdown]
# ![Capacitance method: mask with boundary points, RHS, and solution](../docs/images/theory_figures/capacitance_mask_solution.png)

# %% [markdown]
# ---
# ## 6. Boundary Enforcement: Before vs After Capacitance Correction
#
# The key benefit of the capacitance method is exact enforcement of boundary
# conditions on the irregular domain. Without the correction, the rectangular
# spectral solve has no awareness of the mask boundary, and the solution takes
# non-zero values at inner boundary points.
#
# After the capacitance correction, the solution satisfies $\psi = 0$ at all
# inner boundary points to machine precision (~$10^{-14}$). The red dots below
# mark the inner boundary points where this enforcement is applied.

# %%
from spectraldiffx import solve_poisson_dst

# Reuse mask from previous section
solver = build_capacitance_solver(mask, dx, dy, base_bc="dst")
rhs = jnp.array(mask, dtype=float)
psi = solver(rhs)

# Uncorrected rectangular solve for comparison
psi_rect = solve_poisson_dst(rhs, dx, dy)

print(f"psi_rect shape: {psi_rect.shape}")
print(f"psi (capacitance) shape: {psi.shape}")

bnd_vals_rect = np.array(psi_rect)[j_b, i_b]
bnd_vals = np.array(psi)[j_b, i_b]
print(f"Max |psi| at boundary (rectangular): {np.max(np.abs(bnd_vals_rect)):.4f}")
print(f"Max |psi| at boundary (capacitance): {np.max(np.abs(bnd_vals)):.2e}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

# Rectangular solve (no correction)
psi_rect_masked = np.array(psi_rect) * mask
im0 = axes[0].imshow(psi_rect_masked, cmap="RdBu_r", origin="lower")
axes[0].scatter(i_b, j_b, c="red", s=10, zorder=5)
axes[0].set_title(
    f"Rectangular solve (no correction)\n"
    f"max |psi| at boundary = {np.max(np.abs(bnd_vals_rect)):.3f}",
    fontsize=9,
)
plt.colorbar(im0, ax=axes[0], shrink=0.8)

# Capacitance solve
psi_masked = np.array(psi) * mask
im1 = axes[1].imshow(psi_masked, cmap="RdBu_r", origin="lower")
axes[1].scatter(i_b, j_b, c="red", s=10, zorder=5)
axes[1].set_title(
    f"Capacitance solve (corrected)\n"
    f"max |psi| at boundary = {np.max(np.abs(bnd_vals)):.1e}",
    fontsize=9,
)
plt.colorbar(im1, ax=axes[1], shrink=0.8)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(
    "Boundary Enforcement: Before vs After Capacitance Correction",
    fontsize=11,
    y=1.03,
)
fig.tight_layout()
fig.savefig(IMG_DIR / "capacitance_boundary.png", **COMPRESS)
plt.close(fig)
print(f"Saved: {IMG_DIR / 'capacitance_boundary.png'}")

# %% [markdown]
# ![Before vs after capacitance boundary correction](../docs/images/theory_figures/capacitance_boundary.png)

# %% [markdown]
# ---
# ## 7. Parseval's Theorem and the `ortho` Normalization
#
# **Parseval's theorem** states that a unitary (energy-preserving) transform
# satisfies:
#
# $$
# \|x\|^2 = \|X\|^2
# $$
#
# i.e., the sum of squares in the physical domain equals the sum of squares in the
# spectral domain. For the DCT-II, this holds when using `norm="ortho"`, which
# applies symmetric scaling factors of $1/\sqrt{2N}$ (and an extra $1/\sqrt{2}$
# for the $k=0$ mode) so that the transform matrix is orthogonal.
#
# With the default `norm=None`, the DCT-II is **not** unitary -- the spectral
# energy scales as $O(N)$ relative to the physical energy. The plot below shows
# how the energy ratio $\|DCT(x)\|^2 / \|x\|^2$ behaves for increasing $N$:
# it stays exactly 1.0 with `"ortho"` but grows with `None`.

# %%
from spectraldiffx import dct

Ns = [8, 16, 32, 64, 128]
ratios_none = []
ratios_ortho = []

for N in Ns:
    x = jnp.sin(jnp.linspace(0.1, 3.0, N)) + 0.5
    energy_x = float(jnp.sum(x**2))

    y_none = dct(x, type=2, norm=None)
    energy_none = float(jnp.sum(y_none**2))
    ratios_none.append(energy_none / energy_x)

    y_ortho = dct(x, type=2, norm="ortho")
    energy_ortho = float(jnp.sum(y_ortho**2))
    ratios_ortho.append(energy_ortho / energy_x)

    print(f"N={N:3d}:  x shape={x.shape},  ratio(None)={energy_none / energy_x:.2f},  ratio(ortho)={energy_ortho / energy_x:.6f}")

# %%
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(Ns, ratios_none, "s-", label=r"norm=None: $\|Y\|^2 / \|x\|^2$", color="C3")
ax.plot(
    Ns, ratios_ortho, "o-", label=r'norm="ortho": $\|Y\|^2 / \|x\|^2$', color="C0"
)
ax.axhline(1.0, color="k", ls="--", lw=0.8, label="Parseval (ratio = 1)")
ax.set_xlabel("N (vector length)")
ax.set_ylabel(r"$\|DCT(x)\|^2 \,/\, \|x\|^2$")
ax.set_title("DCT-II Energy Ratio: norm=None vs norm='ortho'")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(ratios_none) * 1.1)

fig.tight_layout()
fig.savefig(IMG_DIR / "ortho_parseval.png", **COMPRESS)
plt.close(fig)
print(f"Saved: {IMG_DIR / 'ortho_parseval.png'}")

# %% [markdown]
# ![Parseval's theorem: ortho normalization preserves energy, default does not](../docs/images/theory_figures/ortho_parseval.png)
