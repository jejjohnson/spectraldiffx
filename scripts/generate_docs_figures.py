"""Generate all documentation figures for spectraldiffx.

Usage:
    uv run python scripts/generate_docs_figures.py

Outputs compressed PNGs to docs/images/.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
OUT.mkdir(exist_ok=True)
DPI = 150
COMPRESS = {"dpi": DPI, "bbox_inches": "tight", "pad_inches": 0.15}


# =========================================================================
# 1. DST/DCT basis functions (spectral_transforms theory)
# =========================================================================


def fig_basis_functions():
    """Plot DST-I and DCT-II basis functions for N=8."""
    N = 8
    n = np.arange(N)

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
    fig.savefig(OUT / "basis_functions.png", **COMPRESS)
    plt.close(fig)
    print("  basis_functions.png")


# =========================================================================
# 2. Eigenvalue comparison: discrete FD vs continuous (spectral_transforms theory)
# =========================================================================


def fig_eigenvalues():
    """Compare discrete FD eigenvalues with continuous k^2 for FFT."""
    from spectraldiffx._src.fourier.eigenvalues import (
        dct2_eigenvalues,
        dst1_eigenvalues,
        fft_eigenvalues,
    )

    N = 32
    dx = 1.0

    eig_dst = dst1_eigenvalues(N, dx)
    eig_dct = dct2_eigenvalues(N, dx)
    eig_fft = fft_eigenvalues(N, dx)

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
    fig.savefig(OUT / "eigenvalues_comparison.png", **COMPRESS)
    plt.close(fig)
    print("  eigenvalues_comparison.png")


# =========================================================================
# 3. Elliptic solver comparison: three BC types (elliptic_solvers theory/guide)
# =========================================================================


def fig_solver_comparison():
    """Solve Poisson with a Gaussian bump RHS using DST/DCT/FFT."""
    from spectraldiffx import solve_poisson_dct, solve_poisson_dst, solve_poisson_fft

    Nx, Ny = 64, 64
    dx = dy = 1.0

    # Gaussian bump RHS
    j, i = jnp.mgrid[0:Ny, 0:Nx]
    cx, cy = Nx / 2, Ny / 2
    rhs = jnp.exp(-((i - cx) ** 2 + (j - cy) ** 2) / (2 * 8**2))
    rhs = rhs - rhs.mean()  # zero mean for compatibility

    psi_dst = solve_poisson_dst(rhs, dx, dy)
    psi_dct = solve_poisson_dct(rhs, dx, dy)
    psi_fft = solve_poisson_fft(rhs, dx, dy)

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
    fig.savefig(OUT / "solver_comparison.png", **COMPRESS)
    plt.close(fig)
    print("  solver_comparison.png")


# =========================================================================
# 4. Eigenfunction recovery (elliptic_solvers guide)
# =========================================================================


def fig_eigenfunction_recovery():
    """Show exact eigenfunction vs computed solution for DST-I Poisson."""
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

    eigx = dst1_eigenvalues(Nx, dx)
    eigy = dst1_eigenvalues(Ny, dy)
    rhs = (eigx[kx - 1] + eigy[ky - 1]) * psi_exact

    psi_computed = solve_poisson_dst(rhs, dx, dy)
    error = psi_computed - psi_exact

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
    fig.savefig(OUT / "eigenfunction_recovery.png", **COMPRESS)
    plt.close(fig)
    print("  eigenfunction_recovery.png")


# =========================================================================
# 5. Capacitance: mask + boundary + solution (capacitance theory/guide)
# =========================================================================


def fig_capacitance_mask():
    """Show the mask, inner-boundary points, and solution."""
    from scipy.ndimage import binary_dilation

    from spectraldiffx import build_capacitance_solver

    Ny, Nx = 32, 32
    dx = dy = 1.0

    # Circular mask
    j, i = np.mgrid[0:Ny, 0:Nx]
    cy, cx = Ny / 2, Nx / 2
    mask = ((j - cy) ** 2 + (i - cx) ** 2) < (0.35 * min(Ny, Nx)) ** 2

    # Inner boundary
    exterior = ~mask
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    dilated = binary_dilation(exterior, structure=struct)
    inner_boundary = mask & dilated

    # Build solver and solve
    solver = build_capacitance_solver(mask, dx, dy, base_bc="dst")
    rhs = jnp.array(mask, dtype=float)
    psi = solver(rhs)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    # Mask with boundary
    mask_vis = np.zeros((Ny, Nx, 3))
    mask_vis[mask] = [0.7, 0.85, 1.0]  # light blue = ocean
    mask_vis[~mask] = [0.85, 0.75, 0.6]  # tan = land
    mask_vis[inner_boundary] = [1.0, 0.3, 0.3]  # red = boundary
    axes[0].imshow(mask_vis, origin="lower")
    j_b, _i_b = np.where(inner_boundary)
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
    fig.savefig(OUT / "capacitance_mask_solution.png", **COMPRESS)
    plt.close(fig)
    print("  capacitance_mask_solution.png")


def fig_capacitance_boundary():
    """Show boundary enforcement: |psi| at boundary points."""
    from spectraldiffx import build_capacitance_solver

    Ny, Nx = 32, 32
    dx = dy = 1.0

    j, i = np.mgrid[0:Ny, 0:Nx]
    mask = ((j - Ny / 2) ** 2 + (i - Nx / 2) ** 2) < (0.35 * min(Ny, Nx)) ** 2

    solver = build_capacitance_solver(mask, dx, dy, base_bc="dst")
    rhs = jnp.array(mask, dtype=float)
    psi = solver(rhs)

    # Uncorrected rectangular solve for comparison
    from spectraldiffx import solve_poisson_dst

    psi_rect = solve_poisson_dst(rhs, dx, dy)

    # Inner boundary
    from scipy.ndimage import binary_dilation

    exterior = ~mask
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    dilated = binary_dilation(exterior, structure=struct)
    inner_boundary = mask & dilated
    j_b, _i_b = np.where(inner_boundary)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Rectangular solve (no correction)
    psi_rect_masked = np.array(psi_rect) * mask
    im0 = axes[0].imshow(psi_rect_masked, cmap="RdBu_r", origin="lower")
    bnd_vals_rect = np.array(psi_rect)[j_b, i_b]
    axes[0].scatter(i_b, j_b, c="red", s=10, zorder=5)
    axes[0].set_title(
        f"Rectangular solve (no correction)\n"
        f"max |ψ| at boundary = {np.max(np.abs(bnd_vals_rect)):.3f}",
        fontsize=9,
    )
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Capacitance solve
    psi_masked = np.array(psi) * mask
    im1 = axes[1].imshow(psi_masked, cmap="RdBu_r", origin="lower")
    bnd_vals = np.array(psi)[j_b, i_b]
    axes[1].scatter(i_b, j_b, c="red", s=10, zorder=5)
    axes[1].set_title(
        f"Capacitance solve (corrected)\n"
        f"max |ψ| at boundary = {np.max(np.abs(bnd_vals)):.1e}",
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
    fig.savefig(OUT / "capacitance_boundary.png", **COMPRESS)
    plt.close(fig)
    print("  capacitance_boundary.png")


# =========================================================================
# 6. Ortho normalization: Parseval's theorem (transforms guide)
# =========================================================================


def fig_ortho_parseval():
    """Show that ortho DCT preserves energy (Parseval)."""
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
    fig.savefig(OUT / "ortho_parseval.png", **COMPRESS)
    plt.close(fig)
    print("  ortho_parseval.png")


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print("Generating docs figures...")
    fig_basis_functions()
    fig_eigenvalues()
    fig_solver_comparison()
    fig_eigenfunction_recovery()
    fig_capacitance_mask()
    fig_capacitance_boundary()
    fig_ortho_parseval()
    print(f"Done — {len(list(OUT.glob('*.png')))} PNGs in {OUT}")
