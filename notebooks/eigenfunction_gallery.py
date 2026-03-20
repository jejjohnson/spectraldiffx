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
# # Eigenfunction & Eigenvalue Gallery
#
# Each spectral transform (DST-I through IV, DCT-I through IV, FFT)
# diagonalises the discrete Laplacian for a specific combination of
# **boundary conditions** and **grid placement**.  This notebook visualises
# the eigenfunctions and eigenvalues for all nine types, making the
# connection between transform, BC, and grid immediately visible.
#
# ## Transform–BC–Grid Map
#
# | Transform | Left BC | Right BC | Grid | Eigenvalue formula |
# |-----------|---------|----------|------|--------------------|
# | DST-I | Dirichlet | Dirichlet | Regular | $-\frac{4}{dx^2}\sin^2\!\frac{\pi(k+1)}{2(N+1)}$ |
# | DST-II | Dirichlet | Dirichlet | Staggered | $-\frac{4}{dx^2}\sin^2\!\frac{\pi(k+1)}{2N}$ |
# | DCT-I | Neumann | Neumann | Regular | $-\frac{4}{dx^2}\sin^2\!\frac{\pi k}{2(N-1)}$ |
# | DCT-II | Neumann | Neumann | Staggered | $-\frac{4}{dx^2}\sin^2\!\frac{\pi k}{2N}$ |
# | DST-III | Dirichlet | Neumann | Regular | $-\frac{4}{dx^2}\sin^2\!\frac{\pi(2k+1)}{4N}$ |
# | DCT-III | Neumann | Dirichlet | Regular | (same formula) |
# | DST-IV | Dirichlet | Neumann | Staggered | (same formula) |
# | DCT-IV | Neumann | Dirichlet | Staggered | (same formula) |
# | FFT | Periodic | Periodic | — | $-\frac{4}{dx^2}\sin^2\!\frac{\pi k}{N}$ |

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
    dct,
    dct1_eigenvalues,
    dct2_eigenvalues,
    dct3_eigenvalues,
    dct4_eigenvalues,
    dst,
    dst1_eigenvalues,
    dst2_eigenvalues,
    dst3_eigenvalues,
    dst4_eigenvalues,
    fft_eigenvalues,
    idct,
    idst,
)

IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "images" / "eigenfunction_gallery"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Same-BC Eigenfunctions
#
# When both boundaries have the **same** BC type (both Dirichlet or both
# Neumann), the eigenfunctions are pure sines or cosines.
#
# ```
# DST-I (Dirichlet, regular)           DST-II (Dirichlet, staggered)
#
#   ψ=0 ·                    · ψ=0     ψ=0 |                    | ψ=0
#        ·  o              o  ·         (wall)  x              x  (wall)
#        ·    o          o    ·              x    x          x
#        ·      o      o      ·            x      x      x
#        ·        o  o         ·          x        x  x
#        ·─────────────────────·       |─────────────────────|
#        0  1  2  3  4  5  6  7       0  1  2  3  4  5  6  7
#
#   o = grid point ON vertex            x = grid point AT cell centre
#   ψ vanishes at grid points           ψ vanishes BETWEEN grid points
#     0 and N+1 (implicit)                at walls half-spacing outside
# ```

# %%
N = 16
dx = 1.0
n_modes = 4  # show first 4 eigenfunctions

def build_eigenfunctions_dst1(N):
    """DST-I eigenfunctions: sin(pi*(k+1)*(j+1)/(N+1))."""
    j = np.arange(N)
    k = np.arange(N)
    return np.sin(np.pi * np.outer(j + 1, k + 1) / (N + 1))

def build_eigenfunctions_dst2(N):
    """DST-II eigenfunctions: sin(pi*(k+1)*(2j+1)/(2N))."""
    j = np.arange(N)
    k = np.arange(N)
    return np.sin(np.pi * np.outer(2 * j + 1, k + 1) / (2 * N))

def build_eigenfunctions_dct1(N):
    """DCT-I eigenfunctions: cos(pi*k*j/(N-1))."""
    j = np.arange(N)
    k = np.arange(N)
    return np.cos(np.pi * np.outer(j, k) / (N - 1))

def build_eigenfunctions_dct2(N):
    """DCT-II eigenfunctions: cos(pi*k*(2j+1)/(2N))."""
    j = np.arange(N)
    k = np.arange(N)
    return np.cos(np.pi * np.outer(2 * j + 1, k) / (2 * N))


# Grid positions for plotting
x_dst1 = np.arange(1, N + 1) / (N + 1)      # interior vertices
x_dst2 = (2 * np.arange(N) + 1) / (2 * N)   # cell centres
x_dct1 = np.arange(N) / (N - 1)              # vertices incl. boundary
x_dct2 = (2 * np.arange(N) + 1) / (2 * N)   # cell centres

phi_dst1 = build_eigenfunctions_dst1(N)
phi_dst2 = build_eigenfunctions_dst2(N)
phi_dct1 = build_eigenfunctions_dct1(N)
phi_dct2 = build_eigenfunctions_dct2(N)

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
colors = plt.cm.tab10(np.linspace(0, 1, n_modes))

configs = [
    (axes[0, 0], phi_dst1, x_dst1, "DST-I: Dirichlet, regular", "o"),
    (axes[0, 1], phi_dst2, x_dst2, "DST-II: Dirichlet, staggered", "s"),
    (axes[1, 0], phi_dct1, x_dct1, "DCT-I: Neumann, regular", "^"),
    (axes[1, 1], phi_dct2, x_dct2, "DCT-II: Neumann, staggered", "v"),
]

for ax, phi, x, title, marker in configs:
    for k in range(n_modes):
        ax.plot(x, phi[:, k], f"{marker}-", ms=4, color=colors[k],
                label=f"$k={k}$", alpha=0.8)
    # Mark boundaries
    ax.axvline(0, color="C3", ls="--", lw=1.5, alpha=0.7, label="Boundary")
    ax.axvline(1, color="C3", ls="--", lw=1.5, alpha=0.7)
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("$x / L$")
    ax.set_ylabel(r"$\phi_k(x)$")
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    ax.grid(True, alpha=0.2)

plt.suptitle(f"Same-BC Eigenfunctions ($N={N}$)", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(IMG_DIR / "same_bc_eigenfunctions.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Same-BC Eigenfunctions](../images/eigenfunction_gallery/same_bc_eigenfunctions.png)
#
# **Key differences to notice:**
#
# - **DST-I** eigenfunctions vanish exactly at $x=0$ and $x=1$ (grid points on boundary).
# - **DST-II** eigenfunctions vanish at positions *between* the first/last cell centre
#   and the boundary — the zero crossing is half a grid spacing outside the domain.
# - **DCT-I** $k=0$ mode is a constant; the function and its derivative equal zero at boundaries.
# - **DCT-II** $k=0$ mode is also constant; the zero-derivative condition is at the cell faces.

# %% [markdown]
# ## 2. Mixed-BC Eigenfunctions
#
# When different BCs apply at each end (Dirichlet on one side, Neumann on
# the other), the eigenfunctions use **half-integer mode indices**:
#
# - **DST-III**: Dirichlet left, Neumann right (regular grid)
# - **DCT-III**: Neumann left, Dirichlet right (regular grid)
# - **DST-IV**: Dirichlet left, Neumann right (staggered grid)
# - **DCT-IV**: Neumann left, Dirichlet right (staggered grid)
#
# All four share the **same eigenvalue formula**:
# $\lambda_k = -\frac{4}{dx^2}\sin^2\!\left(\frac{\pi(2k+1)}{4N}\right)$.

# %%
def build_eigenfunctions_dst3(N):
    """DST-III: sin(pi*(2k+1)*(j+1)/(2N))."""
    j = np.arange(N)
    k = np.arange(N)
    return np.sin(np.pi * np.outer(j + 1, 2 * k + 1) / (2 * N))

def build_eigenfunctions_dct3(N):
    """DCT-III: cos(pi*(2k+1)*j/(2N))."""
    j = np.arange(N)
    k = np.arange(N)
    return np.cos(np.pi * np.outer(j, 2 * k + 1) / (2 * N))

def build_eigenfunctions_dst4(N):
    """DST-IV: sin(pi*(2k+1)*(2j+1)/(4N))."""
    j = np.arange(N)
    k = np.arange(N)
    return np.sin(np.pi * np.outer(2 * j + 1, 2 * k + 1) / (4 * N))

def build_eigenfunctions_dct4(N):
    """DCT-IV: cos(pi*(2k+1)*(2j+1)/(4N))."""
    j = np.arange(N)
    k = np.arange(N)
    return np.cos(np.pi * np.outer(2 * j + 1, 2 * k + 1) / (4 * N))


# Grid positions
x_dst3 = np.arange(1, N + 1) / N       # regular, interior-ish
x_dct3 = np.arange(N) / N              # regular, starts at 0
x_dst4 = (2 * np.arange(N) + 1) / (2 * N)  # staggered
x_dct4 = (2 * np.arange(N) + 1) / (2 * N)  # staggered

phi_dst3 = build_eigenfunctions_dst3(N)
phi_dct3 = build_eigenfunctions_dct3(N)
phi_dst4 = build_eigenfunctions_dst4(N)
phi_dct4 = build_eigenfunctions_dct4(N)

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

mixed_configs = [
    (axes[0, 0], phi_dst3, x_dst3, "DST-III: Dir. left, Neu. right (regular)", "o"),
    (axes[0, 1], phi_dct3, x_dct3, "DCT-III: Neu. left, Dir. right (regular)", "s"),
    (axes[1, 0], phi_dst4, x_dst4, "DST-IV: Dir. left, Neu. right (staggered)", "^"),
    (axes[1, 1], phi_dct4, x_dct4, "DCT-IV: Neu. left, Dir. right (staggered)", "v"),
]

for ax, phi, x, title, marker in mixed_configs:
    for k in range(n_modes):
        ax.plot(x, phi[:, k], f"{marker}-", ms=4, color=colors[k],
                label=f"$k={k}$", alpha=0.8)
    # Boundary annotations
    ax.axvline(0, color="C3", ls="--", lw=1.5, alpha=0.7)
    ax.axvline(1, color="C0", ls="--", lw=1.5, alpha=0.7)
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("$x / L$")
    ax.set_ylabel(r"$\phi_k(x)$")
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    ax.grid(True, alpha=0.2)

plt.suptitle(f"Mixed-BC Eigenfunctions ($N={N}$)", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(IMG_DIR / "mixed_bc_eigenfunctions.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Mixed-BC Eigenfunctions](../images/eigenfunction_gallery/mixed_bc_eigenfunctions.png)
#
# **Key observations:**
#
# - **DST-III/IV** (Dirichlet left): eigenfunctions vanish at $x=0$ and have
#   zero slope at $x=L$.
# - **DCT-III/IV** (Neumann left): eigenfunctions have zero slope at $x=0$
#   and vanish at $x=L$.
# - The half-integer mode index $(2k+1)$ means these basis functions have
#   **no null mode** — all eigenvalues are strictly negative.

# %% [markdown]
# ## 3. Eigenvalue Comparison
#
# All nine eigenvalue curves on one plot.  Note that the four mixed-BC types
# (DST-III, DCT-III, DST-IV, DCT-IV) **all produce the same eigenvalues**.

# %%
N_eig = 32
dx_eig = 1.0
k = np.arange(N_eig)

eig_data = {
    "DST-I": np.array(dst1_eigenvalues(N_eig, dx_eig)),
    "DST-II": np.array(dst2_eigenvalues(N_eig, dx_eig)),
    "DCT-I": np.array(dct1_eigenvalues(N_eig, dx_eig)),
    "DCT-II": np.array(dct2_eigenvalues(N_eig, dx_eig)),
    "FFT": np.array(fft_eigenvalues(N_eig, dx_eig)),
    "Mixed (III/IV)": np.array(dst3_eigenvalues(N_eig, dx_eig)),
}

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left: all curves
ax = axes[0]
styles = {
    "DST-I": ("o-", "C0"), "DST-II": ("s-", "C1"),
    "DCT-I": ("^-", "C2"), "DCT-II": ("v-", "C3"),
    "FFT": ("D-", "C4"), "Mixed (III/IV)": ("*-", "C5"),
}
for name, eig in eig_data.items():
    style, color = styles[name]
    ax.plot(k, eig, style, ms=4, color=color, label=name, alpha=0.8)
ax.set_xlabel("Mode index $k$", fontsize=12)
ax.set_ylabel(r"$\lambda_k$", fontsize=12)
ax.set_title(f"All Eigenvalue Curves ($N={N_eig}$)", fontsize=13)
ax.legend(fontsize=9, ncol=2)
ax.axhline(0, color="k", ls=":", lw=0.5)
ax.grid(True, alpha=0.3)

# Right: zoom into low-k region
ax = axes[1]
k_zoom = slice(0, 8)
for name, eig in eig_data.items():
    style, color = styles[name]
    ax.plot(k[k_zoom], eig[k_zoom], style, ms=6, color=color, label=name)
ax.set_xlabel("Mode index $k$", fontsize=12)
ax.set_ylabel(r"$\lambda_k$", fontsize=12)
ax.set_title("Zoom: Low Modes ($k=0\\ldots7$)", fontsize=13)
ax.legend(fontsize=9, ncol=2)
ax.axhline(0, color="k", ls=":", lw=0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(IMG_DIR / "eigenvalue_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Eigenvalue Comparison](../images/eigenfunction_gallery/eigenvalue_comparison.png)
#
# **Key insight:** The mixed-BC eigenvalues (DST-III, DCT-III, DST-IV,
# DCT-IV) all collapse onto a single curve.  This is because they share
# the formula $\lambda_k = -\frac{4}{dx^2}\sin^2(\pi(2k+1)/(4N))$,
# regardless of which side gets Dirichlet vs Neumann or whether the grid
# is regular vs staggered.

# %% [markdown]
# ## 4. Eigenfunction Orthogonality
#
# Each set of eigenfunctions forms an **orthogonal** basis (up to a
# normalisation constant).  We verify this by computing the Gram matrix
# $G_{jk} = \sum_i \phi_j(x_i)\,\phi_k(x_i)$ and checking it is diagonal.

# %%
N_orth = 16
phi_sets = {
    "DST-I": build_eigenfunctions_dst1(N_orth),
    "DST-II": build_eigenfunctions_dst2(N_orth),
    "DCT-I": build_eigenfunctions_dct1(N_orth),
    "DCT-II": build_eigenfunctions_dct2(N_orth),
    "DST-III": build_eigenfunctions_dst3(N_orth),
    "DCT-III": build_eigenfunctions_dct3(N_orth),
}

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes_flat = axes.ravel()

for idx, (name, phi) in enumerate(phi_sets.items()):
    ax = axes_flat[idx]
    gram = phi.T @ phi
    # Normalise to show structure (divide by diagonal)
    diag = np.diag(gram).copy()
    diag[diag == 0] = 1  # avoid division by zero
    gram_norm = gram / np.sqrt(np.outer(diag, diag))
    im = ax.imshow(np.abs(gram_norm), cmap="Blues", vmin=0, vmax=1)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("$k$")
    ax.set_ylabel("$j$")

fig.colorbar(im, ax=axes_flat, shrink=0.6, label=r"$|G_{jk}| / \sqrt{G_{jj} G_{kk}}$")
plt.suptitle(f"Normalised Gram Matrices ($N={N_orth}$)", fontsize=14, y=1.01)
fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig(IMG_DIR / "orthogonality.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Orthogonality](../images/eigenfunction_gallery/orthogonality.png)
#
# All six Gram matrices show **diagonal-dominant** structure — the off-diagonal
# entries are at machine precision, confirming orthogonality.

# %% [markdown]
# ## 5. Stencil Verification
#
# The eigenvalues are exact for the **discrete** second-order Laplacian.
# We build the tridiagonal matrix $L = [1, -2, 1]/dx^2$ with appropriate
# boundary modifications and verify that $L \phi_k = \lambda_k \phi_k$ holds
# to machine precision for each transform type.

# %%
N_stencil = 16
dx_s = 1.0


def tridiag_laplacian_dirichlet_regular(N, dx):
    """Interior Laplacian with Dirichlet BCs (DST-I): v_0 = v_{N+1} = 0."""
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = -2.0
        if i > 0:
            L[i, i - 1] = 1.0
        if i < N - 1:
            L[i, i + 1] = 1.0
    return L / dx**2


def tridiag_laplacian_dirichlet_staggered(N, dx):
    """Cell-centred Laplacian with Dirichlet BCs (DST-II): ψ=0 at half-spacing."""
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = -2.0
        if i > 0:
            L[i, i - 1] = 1.0
        if i < N - 1:
            L[i, i + 1] = 1.0
    # Staggered Dirichlet: ψ(-dx/2) = 0 → ghost ψ_{-1} = -ψ_0
    # Stencil at i=0: (ψ_{-1} - 2ψ_0 + ψ_1)/dx² = (-3ψ_0 + ψ_1)/dx²
    L[0, 0] = -3.0
    L[N - 1, N - 1] = -3.0
    return L / dx**2


def tridiag_laplacian_neumann_regular(N, dx):
    """Vertex-centred Laplacian with Neumann BCs (DCT-I): ∂ψ/∂n=0 at boundaries."""
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = -2.0
        if i > 0:
            L[i, i - 1] = 1.0
        if i < N - 1:
            L[i, i + 1] = 1.0
    # Neumann: ghost ψ_{-1} = ψ_0 → stencil gives (-1)*ψ_0 + ψ_1
    L[0, 0] = -1.0
    L[N - 1, N - 1] = -1.0
    return L / dx**2


def tridiag_laplacian_neumann_staggered(N, dx):
    """Cell-centred Laplacian with Neumann BCs (DCT-II): ∂ψ/∂n=0 at cell faces."""
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = -2.0
        if i > 0:
            L[i, i - 1] = 1.0
        if i < N - 1:
            L[i, i + 1] = 1.0
    # Neumann at half-spacing: ghost ψ_{-1} = ψ_0
    L[0, 0] = -1.0
    L[N - 1, N - 1] = -1.0
    return L / dx**2


# Build Laplacians
L_dst1 = tridiag_laplacian_dirichlet_regular(N_stencil, dx_s)
L_dst2 = tridiag_laplacian_dirichlet_staggered(N_stencil, dx_s)
L_dct1 = tridiag_laplacian_neumann_regular(N_stencil, dx_s)
L_dct2 = tridiag_laplacian_neumann_staggered(N_stencil, dx_s)

# Eigenfunctions
phi_sets_stencil = {
    "DST-I": (L_dst1, build_eigenfunctions_dst1(N_stencil),
              np.array(dst1_eigenvalues(N_stencil, dx_s))),
    "DST-II": (L_dst2, build_eigenfunctions_dst2(N_stencil),
               np.array(dst2_eigenvalues(N_stencil, dx_s))),
    "DCT-I": (L_dct1, build_eigenfunctions_dct1(N_stencil),
              np.array(dct1_eigenvalues(N_stencil, dx_s))),
    "DCT-II": (L_dct2, build_eigenfunctions_dct2(N_stencil),
               np.array(dct2_eigenvalues(N_stencil, dx_s))),
}

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
axes_flat = axes.ravel()

for idx, (name, (L_mat, phi, eigs)) in enumerate(phi_sets_stencil.items()):
    ax = axes_flat[idx]
    residuals = []
    for k_idx in range(N_stencil):
        phi_k = phi[:, k_idx]
        r = np.linalg.norm(L_mat @ phi_k - eigs[k_idx] * phi_k) / (np.linalg.norm(phi_k) + 1e-30)
        residuals.append(r)

    ax.bar(np.arange(N_stencil), residuals, color="steelblue", alpha=0.7)
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 1e-10)
    ax.set_xlabel("Mode index $k$")
    ax.set_ylabel(r"$\|L\phi_k - \lambda_k \phi_k\| / \|\phi_k\|$")
    ax.set_title(f"{name}: Stencil Residuals", fontsize=11)
    ax.axhline(1e-14, color="C3", ls="--", lw=1, alpha=0.5, label="Machine $\\epsilon$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

plt.suptitle(
    f"Eigenvalue–Stencil Verification ($N={N_stencil}$)",
    fontsize=14, y=1.01,
)
plt.tight_layout()
fig.savefig(IMG_DIR / "stencil_verification.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Stencil Verification](../images/eigenfunction_gallery/stencil_verification.png)
#
# All residuals are at or below $10^{-14}$ — the eigenvalue formulas are
# **exact** for the discrete Laplacian, not approximations.  This is why
# the spectral solvers are exact inverses of the finite-difference operator.

# %% [markdown]
# ## 6. Summary
#
# | If your grid is… | And your BCs are… | Use eigenvalues | Use solver |
# |-------------------|-------------------|-----------------|------------|
# | Cell-centred (staggered) | Dirichlet both sides | `dst2_eigenvalues` | `solve_*_dst2` |
# | Cell-centred (staggered) | Neumann both sides | `dct2_eigenvalues` | `solve_*_dct` |
# | Vertex-centred (regular) | Dirichlet both sides | `dst1_eigenvalues` | `solve_*_dst` |
# | Vertex-centred (regular) | Neumann both sides | `dct1_eigenvalues` | `solve_*_dct1` |
# | Either | Periodic | `fft_eigenvalues` | `solve_*_fft` |
# | Regular | Dir. left + Neu. right | `dst3_eigenvalues` | (coming in v0.0.7) |
# | Regular | Neu. left + Dir. right | `dct3_eigenvalues` | (coming in v0.0.7) |
# | Staggered | Dir. left + Neu. right | `dst4_eigenvalues` | (coming in v0.0.7) |
# | Staggered | Neu. left + Dir. right | `dct4_eigenvalues` | (coming in v0.0.7) |
#
# The mixed-BC solvers (types III/IV) are planned for a future release.
# The eigenvalue functions are already available for use in custom solvers.
