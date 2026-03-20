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
# # Solver Comparison: Regular vs Staggered Grids
#
# spectraldiffx provides spectral Poisson/Helmholtz solvers for **four**
# distinct grid–BC combinations, plus an FFT solver for periodic domains.
# This notebook compares all five on a single canonical problem to build
# intuition for when to use each.
#
# | Solver | Transform | BC type | Grid type | Eigenvalue denominator |
# |--------|-----------|---------|-----------|----------------------|
# | `solve_poisson_dst` | DST-I | Dirichlet | Regular (vertex-centred) | $2(N+1)$ |
# | `solve_poisson_dst2` | DST-II | Dirichlet | Staggered (cell-centred) | $2N$ |
# | `solve_poisson_dct1` | DCT-I | Neumann | Regular (vertex-centred) | $2(N-1)$ |
# | `solve_poisson_dct` | DCT-II | Neumann | Staggered (cell-centred) | $2N$ |
# | `solve_poisson_fft` | FFT | Periodic | — | $N$ |
#
# **Key insight:** "regular" means grid points sit *on* the boundary (vertices);
# "staggered" means grid points sit at *cell centres*, with boundaries located
# half a grid spacing outside the first/last point.

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
jax.config.update("jax_enable_x64", True)

from spectraldiffx import (
    dct1_eigenvalues,
    dct2_eigenvalues,
    dst1_eigenvalues,
    dst2_eigenvalues,
    fft_eigenvalues,
    solve_poisson_dct,
    solve_poisson_dct1,
    solve_poisson_dst,
    solve_poisson_dst2,
    solve_poisson_fft,
)

IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "images" / "solver_comparison"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Grid Geometry
#
# The fundamental difference between regular and staggered grids is **where
# the grid points sit** relative to the domain boundary.
#
# ```
# Regular (vertex-centred)            Staggered (cell-centred)
#
#   boundary                            boundary (half-spacing outside)
#   |                                   |
#   o----o----o----o----o               |--x----x----x----x--|
#   |    |    |    |    |               |  |    |    |    |  |
#   o    o    o    o    o               |--x    x    x    x--|
#   |    |    |    |    |               |  |    |    |    |  |
#   o    o    o    o    o               |--x    x    x    x--|
#   |    |    |    |    |               |  |    |    |    |  |
#   o----o----o----o----o               |--x----x----x----x--|
#   |                                   |
#   boundary                            boundary (half-spacing outside)
#
#   o = vertex (on boundary)            x = cell centre (interior)
#   N=5 spans [0, L]                    N=4 spans [dx/2, L-dx/2]
#   dx = L/(N-1)                        dx = L/N
# ```
#
# This distinction matters because the **spectral transform** that diagonalises
# the discrete Laplacian depends on where the unknowns live.

# %%
# Build small grids for visualisation
N_vis = 6
L = 1.0

# Regular grid: points at vertices, including boundary
x_reg = np.linspace(0, L, N_vis)
y_reg = np.linspace(0, L, N_vis)

# Staggered grid: points at cell centres
dx_stag = L / N_vis
x_stag = np.linspace(dx_stag / 2, L - dx_stag / 2, N_vis)
y_stag = np.linspace(dx_stag / 2, L - dx_stag / 2, N_vis)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Regular grid
ax = axes[0]
Xr, Yr = np.meshgrid(x_reg, y_reg)
ax.plot(Xr.ravel(), Yr.ravel(), "ko", ms=8, label="Grid points (vertices)")
ax.axhline(0, color="C3", ls="--", lw=2, label="Boundary")
ax.axhline(L, color="C3", ls="--", lw=2)
ax.axvline(0, color="C3", ls="--", lw=2)
ax.axvline(L, color="C3", ls="--", lw=2)
ax.set_title("Regular (vertex-centred)", fontsize=13)
ax.set_xlim(-0.15, L + 0.15)
ax.set_ylim(-0.15, L + 0.15)
ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=9)

# Staggered grid
ax = axes[1]
Xs, Ys = np.meshgrid(x_stag, y_stag)
ax.plot(Xs.ravel(), Ys.ravel(), "s", color="C0", ms=8, label="Grid points (cell centres)")
ax.axhline(0, color="C3", ls="--", lw=2, label="Boundary (off-grid)")
ax.axhline(L, color="C3", ls="--", lw=2)
ax.axvline(0, color="C3", ls="--", lw=2)
ax.axvline(L, color="C3", ls="--", lw=2)
ax.set_title("Staggered (cell-centred)", fontsize=13)
ax.set_xlim(-0.15, L + 0.15)
ax.set_ylim(-0.15, L + 0.15)
ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=9)

plt.suptitle("Grid Geometry: Regular vs Staggered", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(IMG_DIR / "grid_geometry.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Grid Geometry](../images/solver_comparison/grid_geometry.png)

# %% [markdown]
# ## 2. Eigenvalue Spectra
#
# Each transform type produces a different set of 1-D Laplacian eigenvalues.
# The formulas are:
#
# | Transform | Formula | Null mode? |
# |-----------|---------|------------|
# | DST-I | $\lambda_k = -\frac{4}{dx^2}\sin^2\!\left(\frac{\pi(k+1)}{2(N+1)}\right)$ | No |
# | DST-II | $\lambda_k = -\frac{4}{dx^2}\sin^2\!\left(\frac{\pi(k+1)}{2N}\right)$ | No |
# | DCT-I | $\lambda_k = -\frac{4}{dx^2}\sin^2\!\left(\frac{\pi k}{2(N-1)}\right)$ | Yes ($k=0$) |
# | DCT-II | $\lambda_k = -\frac{4}{dx^2}\sin^2\!\left(\frac{\pi k}{2N}\right)$ | Yes ($k=0$) |
# | FFT | $\lambda_k = -\frac{4}{dx^2}\sin^2\!\left(\frac{\pi k}{N}\right)$ | Yes ($k=0$) |
#
# The key difference is the **denominator** inside the sine: $2(N+1)$, $2N$,
# or $2(N-1)$.  This shifts the eigenvalue curves relative to each other.

# %%
N_eig = 32
dx_eig = 1.0
k = np.arange(N_eig)

eig_dst1 = np.array(dst1_eigenvalues(N_eig, dx_eig))
eig_dst2 = np.array(dst2_eigenvalues(N_eig, dx_eig))
eig_dct1 = np.array(dct1_eigenvalues(N_eig, dx_eig))
eig_dct2 = np.array(dct2_eigenvalues(N_eig, dx_eig))
eig_fft = np.array(fft_eigenvalues(N_eig, dx_eig))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k, eig_dst1, "o-", ms=4, label="DST-I (Dirichlet, regular)")
ax.plot(k, eig_dst2, "s-", ms=4, label="DST-II (Dirichlet, staggered)")
ax.plot(k, eig_dct1, "^-", ms=4, label="DCT-I (Neumann, regular)")
ax.plot(k, eig_dct2, "v-", ms=4, label="DCT-II (Neumann, staggered)")
ax.plot(k, eig_fft, "D-", ms=4, label="FFT (Periodic)")
ax.set_xlabel("Mode index $k$", fontsize=12)
ax.set_ylabel(r"Eigenvalue $\lambda_k$", fontsize=12)
ax.set_title(f"1-D Laplacian Eigenvalue Spectra ($N={N_eig}$, $dx={dx_eig}$)", fontsize=13)
ax.legend(fontsize=10)
ax.axhline(0, color="k", ls=":", lw=0.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(IMG_DIR / "eigenvalue_spectra.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Eigenvalue Spectra](../images/solver_comparison/eigenvalue_spectra.png)
#
# **Observations:**
#
# - **DST-I and DST-II** (Dirichlet) are all strictly negative — no null mode.
# - **DCT-I, DCT-II, and FFT** have $\lambda_0 = 0$ (constant null mode).
#   For Poisson solves, this mode is projected out (zero-mean gauge).
# - DST-II eigenvalues are slightly more negative than DST-I at each $k$
#   because $2N < 2(N+1)$ makes the sine argument larger.
# - FFT eigenvalues are symmetric about $k = N/2$ (aliasing).

# %% [markdown]
# ## 3. Manufactured Solution: Convergence Test
#
# We use a smooth manufactured solution to test convergence:
#
# $$\psi_{\text{exact}}(x, y) = \sin(2\pi x)\,\sin(2\pi y)$$
#
# so the RHS is:
#
# $$f(x, y) = \nabla^2 \psi = -8\pi^2 \sin(2\pi x)\,\sin(2\pi y)$$
#
# We solve on the unit square $[0,1]^2$ at increasing resolutions and
# measure the $L^\infty$ error for each solver.

# %%
resolutions = [16, 32, 64, 128, 256]

errors = {
    "DST-I (Dirichlet, regular)": [],
    "DST-II (Dirichlet, staggered)": [],
    "DCT-I (Neumann, regular)": [],
    "DCT-II (Neumann, staggered)": [],
    "FFT (Periodic)": [],
}

for N in resolutions:
    # --- Regular grids (DST-I, DCT-I): points at vertices ---
    # DST-I: N interior points, boundary at 0 and L (excluded)
    dx_r = L / (N + 1)
    x_r = jnp.linspace(dx_r, L - dx_r, N)  # interior only
    Xr, Yr = jnp.meshgrid(x_r, x_r)
    psi_exact_r = jnp.sin(2 * jnp.pi * Xr) * jnp.sin(2 * jnp.pi * Yr)
    rhs_r = -8.0 * jnp.pi**2 * psi_exact_r

    psi_dst1 = solve_poisson_dst(rhs_r, dx_r, dx_r)
    errors["DST-I (Dirichlet, regular)"].append(
        float(jnp.max(jnp.abs(psi_dst1 - psi_exact_r)))
    )

    # DCT-I: N points including both boundaries
    dx_c1 = L / (N - 1) if N > 1 else L
    x_c1 = jnp.linspace(0, L, N)
    Xc1, Yc1 = jnp.meshgrid(x_c1, x_c1)
    psi_exact_c1 = jnp.sin(2 * jnp.pi * Xc1) * jnp.sin(2 * jnp.pi * Yc1)
    rhs_c1 = -8.0 * jnp.pi**2 * psi_exact_c1

    psi_dct1 = solve_poisson_dct1(rhs_c1, dx_c1, dx_c1)
    errors["DCT-I (Neumann, regular)"].append(
        float(jnp.max(jnp.abs(psi_dct1 - psi_exact_c1)))
    )

    # --- Staggered grids (DST-II, DCT-II): points at cell centres ---
    dx_s = L / N
    x_s = jnp.linspace(dx_s / 2, L - dx_s / 2, N)
    Xs, Ys = jnp.meshgrid(x_s, x_s)
    psi_exact_s = jnp.sin(2 * jnp.pi * Xs) * jnp.sin(2 * jnp.pi * Ys)
    rhs_s = -8.0 * jnp.pi**2 * psi_exact_s

    psi_dst2 = solve_poisson_dst2(rhs_s, dx_s, dx_s)
    errors["DST-II (Dirichlet, staggered)"].append(
        float(jnp.max(jnp.abs(psi_dst2 - psi_exact_s)))
    )

    psi_dct2 = solve_poisson_dct(rhs_s, dx_s, dx_s)
    errors["DCT-II (Neumann, staggered)"].append(
        float(jnp.max(jnp.abs(psi_dct2 - psi_exact_s)))
    )

    # --- Periodic (FFT) ---
    # Periodic: sin(2*pi*x) is exactly periodic on [0,1)
    dx_p = L / N
    x_p = jnp.linspace(0, L - dx_p, N)
    Xp, Yp = jnp.meshgrid(x_p, x_p)
    psi_exact_p = jnp.sin(2 * jnp.pi * Xp) * jnp.sin(2 * jnp.pi * Yp)
    rhs_p = -8.0 * jnp.pi**2 * psi_exact_p

    psi_fft = solve_poisson_fft(rhs_p, dx_p, dx_p)
    errors["FFT (Periodic)"].append(
        float(jnp.max(jnp.abs(psi_fft - psi_exact_p)))
    )

# %%
fig, ax = plt.subplots(figsize=(9, 6))
markers = ["o", "s", "^", "v", "D"]
for (name, errs), marker in zip(errors.items(), markers):
    ax.loglog(resolutions, errs, f"{marker}-", ms=7, label=name)

# Reference slope: O(dx^2)
N_ref = np.array(resolutions, dtype=float)
ref = 2.0 * (N_ref[0] / N_ref) ** 2 * max(errors["DST-I (Dirichlet, regular)"][0], 1e-6)
ax.loglog(N_ref, ref, "k--", lw=1, alpha=0.5, label=r"$O(N^{-2})$ reference")

ax.set_xlabel("Grid size $N$", fontsize=12)
ax.set_ylabel(r"$L^\infty$ error", fontsize=12)
ax.set_title("Convergence: Manufactured Solution on Unit Square", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
fig.savefig(IMG_DIR / "convergence.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Convergence](../images/solver_comparison/convergence.png)
#
# **Observations:**
#
# - **FFT** achieves spectral (exponential) convergence because $\sin(2\pi x)$
#   is exactly representable in the Fourier basis — errors are at machine precision.
# - **DST/DCT solvers** converge at second order ($O(dx^2)$), consistent with the
#   second-order finite-difference Laplacian they invert.
# - Regular and staggered variants converge at the same rate but with slightly
#   different constants.

# %% [markdown]
# ## 4. Solution Visualisation
#
# We solve $\nabla^2 \psi = f$ for a Gaussian bump RHS and compare the
# five solvers side by side.  The different boundary conditions produce
# qualitatively different solutions near the domain edges.

# %%
N = 64
dx_s = L / N
x_s = jnp.linspace(dx_s / 2, L - dx_s / 2, N)
Xs, Ys = jnp.meshgrid(x_s, x_s)

# Gaussian bump centred at (0.5, 0.5)
sigma = 0.12
rhs_gauss = jnp.exp(-((Xs - 0.5) ** 2 + (Ys - 0.5) ** 2) / (2 * sigma**2))

# Solve with staggered solvers (cell-centred grids)
psi_dst2_g = solve_poisson_dst2(rhs_gauss, dx_s, dx_s)
psi_dct2_g = solve_poisson_dct(rhs_gauss, dx_s, dx_s)
psi_fft_g = solve_poisson_fft(rhs_gauss, dx_s, dx_s)

# Regular grids need their own coordinate arrays
dx_r_dir = L / (N + 1)
x_r_dir = jnp.linspace(dx_r_dir, L - dx_r_dir, N)
Xrd, Yrd = jnp.meshgrid(x_r_dir, x_r_dir)
rhs_gauss_rd = jnp.exp(-((Xrd - 0.5) ** 2 + (Yrd - 0.5) ** 2) / (2 * sigma**2))
psi_dst1_g = solve_poisson_dst(rhs_gauss_rd, dx_r_dir, dx_r_dir)

dx_r_neu = L / (N - 1)
x_r_neu = jnp.linspace(0, L, N)
Xrn, Yrn = jnp.meshgrid(x_r_neu, x_r_neu)
rhs_gauss_rn = jnp.exp(-((Xrn - 0.5) ** 2 + (Yrn - 0.5) ** 2) / (2 * sigma**2))
psi_dct1_g = solve_poisson_dct1(rhs_gauss_rn, dx_r_neu, dx_r_neu)

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

data = [
    (axes[0, 0], psi_dst1_g, Xrd, Yrd, "DST-I\n(Dirichlet, regular)"),
    (axes[0, 1], psi_dst2_g, Xs, Ys, "DST-II\n(Dirichlet, staggered)"),
    (axes[0, 2], psi_dct1_g, Xrn, Yrn, "DCT-I\n(Neumann, regular)"),
    (axes[1, 0], psi_dct2_g, Xs, Ys, "DCT-II\n(Neumann, staggered)"),
    (axes[1, 1], psi_fft_g, Xs, Ys, "FFT\n(Periodic)"),
]

# Common colour scale
vmin = min(float(jnp.min(d[1])) for d in data)
vmax = max(float(jnp.max(d[1])) for d in data)

for ax, psi, X, Y, title in data:
    im = ax.pcolormesh(
        np.array(X), np.array(Y), np.array(psi),
        cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto",
    )
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, shrink=0.8)

axes[1, 2].set_visible(False)

plt.suptitle(
    r"Poisson solutions $\nabla^2\psi = f$ for a Gaussian bump RHS",
    fontsize=14, y=1.01,
)
plt.tight_layout()
fig.savefig(IMG_DIR / "solutions.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Solutions](../images/solver_comparison/solutions.png)
#
# **Observations:**
#
# - **Dirichlet** solutions (DST-I, DST-II) are pulled to zero at the boundary,
#   creating a sharp gradient near the edges.
# - **Neumann** solutions (DCT-I, DCT-II) have zero normal derivative at the
#   boundary, so the solution "flattens out" at the edges.
# - **Periodic** solution wraps around — the response appears on both sides.
# - Regular vs staggered variants look very similar at this resolution; the
#   difference is in where exactly the boundary condition is enforced.

# %% [markdown]
# ## 5. Error Maps: Regular vs Staggered
#
# To see the difference between regular and staggered more clearly, we
# compare the Dirichlet solvers (DST-I vs DST-II) on the **same**
# manufactured solution, evaluated at the appropriate grid points for each.

# %%
N = 64

# DST-I (regular): interior points
dx1 = L / (N + 1)
x1 = jnp.linspace(dx1, L - dx1, N)
X1, Y1 = jnp.meshgrid(x1, x1)
psi_exact_1 = jnp.sin(2 * jnp.pi * X1) * jnp.sin(2 * jnp.pi * Y1)
rhs1 = -8.0 * jnp.pi**2 * psi_exact_1
err_dst1 = jnp.abs(solve_poisson_dst(rhs1, dx1, dx1) - psi_exact_1)

# DST-II (staggered): cell centres
dx2 = L / N
x2 = jnp.linspace(dx2 / 2, L - dx2 / 2, N)
X2, Y2 = jnp.meshgrid(x2, x2)
psi_exact_2 = jnp.sin(2 * jnp.pi * X2) * jnp.sin(2 * jnp.pi * Y2)
rhs2 = -8.0 * jnp.pi**2 * psi_exact_2
err_dst2 = jnp.abs(solve_poisson_dst2(rhs2, dx2, dx2) - psi_exact_2)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].pcolormesh(np.array(X1), np.array(Y1), np.array(err_dst1), cmap="hot_r", shading="auto")
axes[0].set_title("DST-I error (regular)", fontsize=11)
axes[0].set_aspect("equal")
plt.colorbar(im0, ax=axes[0], shrink=0.8, format="%.1e")

im1 = axes[1].pcolormesh(np.array(X2), np.array(Y2), np.array(err_dst2), cmap="hot_r", shading="auto")
axes[1].set_title("DST-II error (staggered)", fontsize=11)
axes[1].set_aspect("equal")
plt.colorbar(im1, ax=axes[1], shrink=0.8, format="%.1e")

# Cross-section through centre
j_mid = N // 2
axes[2].semilogy(np.array(x1), np.array(err_dst1[j_mid, :]), "o-", ms=3, label="DST-I (regular)")
axes[2].semilogy(np.array(x2), np.array(err_dst2[j_mid, :]), "s-", ms=3, label="DST-II (staggered)")
axes[2].set_xlabel("$x$")
axes[2].set_ylabel(r"$|\psi - \psi_{\rm exact}|$")
axes[2].set_title("Error cross-section at $y = 0.5$", fontsize=11)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle(
    r"Point-wise error: $|\psi_{\rm computed} - \psi_{\rm exact}|$",
    fontsize=13, y=1.02,
)
plt.tight_layout()
fig.savefig(IMG_DIR / "error_maps.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Error Maps](../images/solver_comparison/error_maps.png)

# %% [markdown]
# ## 6. Summary
#
# | Question | Answer |
# |----------|--------|
# | **Which solver for cell-centred Dirichlet?** | `solve_poisson_dst2` (DST-II) |
# | **Which solver for vertex-centred Dirichlet?** | `solve_poisson_dst` (DST-I) |
# | **Which solver for cell-centred Neumann?** | `solve_poisson_dct` (DCT-II) |
# | **Which solver for vertex-centred Neumann?** | `solve_poisson_dct1` (DCT-I) |
# | **Which solver for periodic?** | `solve_poisson_fft` |
# | **Convergence rate?** | $O(dx^2)$ for all DST/DCT; spectral for FFT |
# | **When does the choice matter?** | Always — using the wrong grid type introduces $O(dx^2)$ boundary error |
#
# The choice of solver must match where your unknowns live on the grid.
# In finite-volume methods (e.g., Arakawa C-grids), pressure and tracers
# live at **cell centres** (staggered), so DST-II/DCT-II are the correct
# choice.  Vorticity/streamfunction at corners use DST-I/DCT-I.
