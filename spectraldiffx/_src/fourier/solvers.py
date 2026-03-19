"""Spectral elliptic solvers for Helmholtz, Poisson, and Laplace equations.

Layer 0 — Pure functions
------------------------
Functional solvers that take arrays and grid spacings directly.

**Same-BC solvers** (same BC type on all boundaries):

* **Periodic (FFT):** 1D, 2D, 3D
* **Dirichlet, regular (DST-I):** 1D, 2D, 3D
* **Dirichlet, staggered (DST-II/III):** 1D, 2D, 3D
* **Neumann, regular (DCT-I):** 1D, 2D, 3D
* **Neumann, staggered (DCT-II/III):** 1D, 2D, 3D

The DST/DCT solvers use discrete finite-difference eigenvalues, making
them exact inverses of the second-order finite-difference Laplacian
(3-point stencil in 1D, 5-point in 2D, 7-point in 3D).

**Backwards-compatible aliases:**

* ``solve_helmholtz_dst1`` = ``solve_helmholtz_dst`` (Dirichlet, regular)
* ``solve_helmholtz_dct2`` = ``solve_helmholtz_dct`` (Neumann, staggered)

Layer 1 — Module classes
-------------------------
``eqx.Module`` wrappers that store grid/solver parameters:

* ``SpectralHelmholtzSolver1D/2D/3D`` — periodic (FFT)
* ``DirichletHelmholtzSolver2D`` — Dirichlet BCs, regular grid (DST-I)
* ``StaggeredDirichletHelmholtzSolver2D`` — Dirichlet BCs, staggered grid (DST-II)
* ``NeumannHelmholtzSolver2D`` — Neumann BCs, staggered grid (DCT-II)
* ``RegularNeumannHelmholtzSolver2D`` — Neumann BCs, regular grid (DCT-I)
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .eigenvalues import (
    dct1_eigenvalues,
    dct2_eigenvalues,
    dst1_eigenvalues,
    dst2_eigenvalues,
    fft_eigenvalues,
)
from .grid import FourierGrid1D, FourierGrid2D, FourierGrid3D
from .transforms import dctn, dstn, idctn, idstn

# ===========================================================================
# Layer 0 — Pure functional solvers
# ===========================================================================

# ---------------------------------------------------------------------------
# Periodic (FFT) — 1D
# ---------------------------------------------------------------------------


def solve_helmholtz_fft_1d(
    rhs: Float[Array, " N"],
    dx: float,
    lambda_: float = 0.0,
) -> Float[Array, " N"]:
    """Solve (∇² − λ)ψ = f in 1-D with periodic BCs using the FFT.

    Spectral algorithm:

    1. Forward FFT:  f̂ = FFT(f)                                    [N]
    2. Eigenvalues:  Λ_k = −4/dx² sin²(πk/N)                      [N]
    3. Spectral division:  ψ̂[k] = f̂[k] / (Λ_k − λ)              [N]
    4. Inverse FFT:  ψ = Re(IFFT(ψ̂))                              [N]

    When ``lambda_ == 0`` the k=0 mode has Λ_0 = 0, making the denominator
    singular.  This is handled by setting ψ̂[0] = 0 (zero-mean gauge).

    Parameters
    ----------
    rhs : Float[Array, " N"]
        Right-hand side on the periodic domain.
    dx : float
        Grid spacing.
    lambda_ : float
        Helmholtz parameter λ.  Default: 0.0 (Poisson).

    Returns
    -------
    Float[Array, " N"]
        Solution ψ (real), same shape as *rhs*.
    """
    (N,) = rhs.shape
    rhs_hat = jnp.fft.fft(rhs)
    eig = fft_eigenvalues(N, dx)
    denom = eig - lambda_
    is_null = denom == 0.0
    denom_safe = jnp.where(is_null, 1.0, denom)
    psi_hat = rhs_hat / denom_safe
    psi_hat = jnp.where(is_null, 0.0, psi_hat)
    return jnp.real(jnp.fft.ifft(psi_hat))


def solve_poisson_fft_1d(
    rhs: Float[Array, " N"],
    dx: float,
) -> Float[Array, " N"]:
    """Solve ∇²ψ = f in 1-D with periodic BCs using FFT.

    Convenience wrapper around :func:`solve_helmholtz_fft_1d` with ``lambda_=0``.

    Parameters
    ----------
    rhs : Float[Array, " N"]
        Right-hand side on the periodic domain.
    dx : float
        Grid spacing.

    Returns
    -------
    Float[Array, " N"]
        Zero-mean solution ψ (real), same shape as *rhs*.
    """
    return solve_helmholtz_fft_1d(rhs, dx, lambda_=0.0)


# ---------------------------------------------------------------------------
# Dirichlet, regular (DST-I) — 1D
# ---------------------------------------------------------------------------


def solve_helmholtz_dst1_1d(
    rhs: Float[Array, " N"],
    dx: float,
    lambda_: float = 0.0,
) -> Float[Array, " N"]:
    """Solve (∇² − λ)ψ = f in 1-D with Dirichlet BCs on a regular grid (DST-I).

    Parameters
    ----------
    rhs : Float[Array, " N"]
        Right-hand side at interior grid points.
    dx : float
        Grid spacing.
    lambda_ : float
        Helmholtz parameter λ.  Default: 0.0 (Poisson).

    Returns
    -------
    Float[Array, " N"]
        Solution ψ, same shape as *rhs*.
    """
    (N,) = rhs.shape
    rhs_hat = dstn(rhs, type=1, axes=[0])
    eig = dst1_eigenvalues(N, dx) - lambda_
    psi_hat = rhs_hat / eig
    return idstn(psi_hat, type=1, axes=[0])


def solve_poisson_dst1_1d(
    rhs: Float[Array, " N"],
    dx: float,
) -> Float[Array, " N"]:
    """Solve ∇²ψ = f in 1-D with Dirichlet BCs on a regular grid (DST-I).

    Parameters
    ----------
    rhs : Float[Array, " N"]
        Right-hand side at interior grid points.
    dx : float
        Grid spacing.

    Returns
    -------
    Float[Array, " N"]
        Solution ψ, same shape as *rhs*.
    """
    return solve_helmholtz_dst1_1d(rhs, dx, lambda_=0.0)


# ---------------------------------------------------------------------------
# Dirichlet, staggered (DST-II/III) — 1D
# ---------------------------------------------------------------------------


def solve_helmholtz_dst2_1d(
    rhs: Float[Array, " N"],
    dx: float,
    lambda_: float = 0.0,
) -> Float[Array, " N"]:
    """Solve (∇² − λ)ψ = f in 1-D with Dirichlet BCs on a staggered grid (DST-II).

    Parameters
    ----------
    rhs : Float[Array, " N"]
        Right-hand side at cell-centred grid points.
    dx : float
        Grid spacing.
    lambda_ : float
        Helmholtz parameter λ.  Default: 0.0 (Poisson).

    Returns
    -------
    Float[Array, " N"]
        Solution ψ, same shape as *rhs*.
    """
    (N,) = rhs.shape
    rhs_hat = dstn(rhs, type=2, axes=[0])
    eig = dst2_eigenvalues(N, dx) - lambda_
    psi_hat = rhs_hat / eig
    return idstn(psi_hat, type=2, axes=[0])


def solve_poisson_dst2_1d(
    rhs: Float[Array, " N"],
    dx: float,
) -> Float[Array, " N"]:
    """Solve ∇²ψ = f in 1-D with Dirichlet BCs on a staggered grid (DST-II).

    Parameters
    ----------
    rhs : Float[Array, " N"]
        Right-hand side at cell-centred grid points.
    dx : float
        Grid spacing.

    Returns
    -------
    Float[Array, " N"]
        Solution ψ, same shape as *rhs*.
    """
    return solve_helmholtz_dst2_1d(rhs, dx, lambda_=0.0)


# ---------------------------------------------------------------------------
# Neumann, regular (DCT-I) — 1D
# ---------------------------------------------------------------------------


def solve_helmholtz_dct1_1d(
    rhs: Float[Array, " N"],
    dx: float,
    lambda_: float = 0.0,
) -> Float[Array, " N"]:
    """Solve (∇² − λ)ψ = f in 1-D with Neumann BCs on a regular grid (DCT-I).

    The k=0 eigenvalue is zero.  For λ=0 (Poisson), the null mode is
    projected out (zero-mean gauge).

    Parameters
    ----------
    rhs : Float[Array, " N"]
        Right-hand side (including boundary grid points).
    dx : float
        Grid spacing.
    lambda_ : float
        Helmholtz parameter λ.  Default: 0.0 (Poisson).

    Returns
    -------
    Float[Array, " N"]
        Solution ψ, same shape as *rhs*.
    """
    (N,) = rhs.shape
    rhs_hat = dctn(rhs, type=1, axes=[0])
    eig = dct1_eigenvalues(N, dx)
    denom = eig - lambda_
    is_null = denom == 0.0
    denom_safe = jnp.where(is_null, 1.0, denom)
    psi_hat = rhs_hat / denom_safe
    psi_hat = jnp.where(is_null, 0.0, psi_hat)
    return idctn(psi_hat, type=1, axes=[0])


def solve_poisson_dct1_1d(
    rhs: Float[Array, " N"],
    dx: float,
) -> Float[Array, " N"]:
    """Solve ∇²ψ = f in 1-D with Neumann BCs on a regular grid (DCT-I).

    Parameters
    ----------
    rhs : Float[Array, " N"]
        Right-hand side (including boundary grid points).
    dx : float
        Grid spacing.

    Returns
    -------
    Float[Array, " N"]
        Zero-mean solution ψ, same shape as *rhs*.
    """
    return solve_helmholtz_dct1_1d(rhs, dx, lambda_=0.0)


# ---------------------------------------------------------------------------
# Neumann, staggered (DCT-II/III) — 1D
# ---------------------------------------------------------------------------


def solve_helmholtz_dct2_1d(
    rhs: Float[Array, " N"],
    dx: float,
    lambda_: float = 0.0,
) -> Float[Array, " N"]:
    """Solve (∇² − λ)ψ = f in 1-D with Neumann BCs on a staggered grid (DCT-II).

    The k=0 eigenvalue is zero.  For λ=0 (Poisson), the null mode is
    projected out (zero-mean gauge).

    Parameters
    ----------
    rhs : Float[Array, " N"]
        Right-hand side at cell-centred grid points.
    dx : float
        Grid spacing.
    lambda_ : float
        Helmholtz parameter λ.  Default: 0.0 (Poisson).

    Returns
    -------
    Float[Array, " N"]
        Solution ψ, same shape as *rhs*.
    """
    (N,) = rhs.shape
    rhs_hat = dctn(rhs, type=2, axes=[0])
    eig = dct2_eigenvalues(N, dx)
    denom = eig - lambda_
    is_null = denom == 0.0
    denom_safe = jnp.where(is_null, 1.0, denom)
    psi_hat = rhs_hat / denom_safe
    psi_hat = jnp.where(is_null, 0.0, psi_hat)
    return idctn(psi_hat, type=2, axes=[0])


def solve_poisson_dct2_1d(
    rhs: Float[Array, " N"],
    dx: float,
) -> Float[Array, " N"]:
    """Solve ∇²ψ = f in 1-D with Neumann BCs on a staggered grid (DCT-II).

    Parameters
    ----------
    rhs : Float[Array, " N"]
        Right-hand side at cell-centred grid points.
    dx : float
        Grid spacing.

    Returns
    -------
    Float[Array, " N"]
        Zero-mean solution ψ, same shape as *rhs*.
    """
    return solve_helmholtz_dct2_1d(rhs, dx, lambda_=0.0)


# ---------------------------------------------------------------------------
# Periodic (FFT) — 2D
# ---------------------------------------------------------------------------


def solve_helmholtz_fft(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float = 0.0,
) -> Float[Array, "Ny Nx"]:
    """Solve (∇² − λ)ψ = f with periodic BCs using the 2-D FFT.

    Spectral algorithm:

    1. Forward 2-D FFT:  f̂ = FFT2(f)                        [Ny, Nx]
    2. Eigenvalue matrix:
           Λ[j,i] = Λ_j^y + Λ_i^x − λ                      [Ny, Nx]
       where Λ^x = fft_eigenvalues(Nx, dx), Λ^y = fft_eigenvalues(Ny, dy)
    3. Spectral division:  ψ̂[j,i] = f̂[j,i] / Λ[j,i]       [Ny, Nx]
    4. Inverse 2-D FFT:  ψ = Re(IFFT2(ψ̂))                  [Ny, Nx]

    When ``lambda_ == 0`` the (0,0) mode is singular (Λ[0,0] = 0).
    This is handled by setting ψ̂[0,0] = 0 (zero-mean gauge).

    Parameters
    ----------
    rhs : Float[Array, "Ny Nx"]
        Right-hand side on the periodic domain.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter λ.  Default: 0.0 (Poisson).

    Returns
    -------
    Float[Array, "Ny Nx"]
        Solution ψ (real), same shape as *rhs*.
    """
    Ny, Nx = rhs.shape
    rhs_hat = jnp.fft.fft2(rhs)
    eigx = fft_eigenvalues(Nx, dx)
    eigy = fft_eigenvalues(Ny, dy)
    eig2d = eigy[:, None] + eigx[None, :] - lambda_
    is_null = eig2d[0, 0] == 0.0
    eig2d_safe = eig2d.at[0, 0].set(jnp.where(is_null, 1.0, eig2d[0, 0]))
    psi_hat = rhs_hat / eig2d_safe
    psi_hat = psi_hat.at[0, 0].set(
        jnp.where(is_null, jnp.zeros_like(psi_hat[0, 0]), psi_hat[0, 0])
    )
    return jnp.real(jnp.fft.ifft2(psi_hat))


def solve_poisson_fft(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "Ny Nx"]:
    """Solve ∇²ψ = f with periodic BCs using the 2-D FFT.

    Convenience wrapper around :func:`solve_helmholtz_fft` with ``lambda_=0``.

    Parameters
    ----------
    rhs : Float[Array, "Ny Nx"]
        Right-hand side on the periodic domain.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Zero-mean solution ψ (real), same shape as *rhs*.
    """
    return solve_helmholtz_fft(rhs, dx, dy, lambda_=0.0)


# ---------------------------------------------------------------------------
# Dirichlet (DST-I) — 2D
# ---------------------------------------------------------------------------


def solve_helmholtz_dst(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float = 0.0,
) -> Float[Array, "Ny Nx"]:
    """Solve (∇² − λ)ψ = f with homogeneous Dirichlet BCs using DST-I.

    The input *rhs* lives on the **interior** grid (boundary values are
    implicitly zero: ψ = 0 on all four edges).

    Spectral algorithm:

    1. Forward 2-D DST-I:  f̂ = DST-I_y(DST-I_x(f))          [Ny, Nx]
    2. Eigenvalue matrix:
           Λ[j,i] = Λ_j^y + Λ_i^x − λ                       [Ny, Nx]
       where Λ^x = dst1_eigenvalues(Nx, dx),
             Λ^y = dst1_eigenvalues(Ny, dy)
       All eigenvalues are strictly negative, so the system is always
       non-singular for λ ≥ 0.
    3. Spectral division:  ψ̂[j,i] = f̂[j,i] / Λ[j,i]        [Ny, Nx]
    4. Inverse 2-D DST-I:  ψ = IDST-I_y(IDST-I_x(ψ̂))       [Ny, Nx]

    Parameters
    ----------
    rhs : Float[Array, "Ny Nx"]
        Right-hand side at interior grid points.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter λ.  Default: 0.0 (Poisson).

    Returns
    -------
    Float[Array, "Ny Nx"]
        Solution ψ at interior grid points, same shape as *rhs*.
    """
    Ny, Nx = rhs.shape
    rhs_hat = dstn(rhs, type=1, axes=[0, 1])
    eigx = dst1_eigenvalues(Nx, dx)
    eigy = dst1_eigenvalues(Ny, dy)
    eig2d = eigy[:, None] + eigx[None, :] - lambda_
    psi_hat = rhs_hat / eig2d
    return idstn(psi_hat, type=1, axes=[0, 1])


def solve_poisson_dst(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "Ny Nx"]:
    """Solve ∇²ψ = f with homogeneous Dirichlet BCs using DST-I.

    Convenience wrapper around :func:`solve_helmholtz_dst` with ``lambda_=0``.

    Parameters
    ----------
    rhs : Float[Array, "Ny Nx"]
        Right-hand side at interior grid points.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Solution ψ, same shape as *rhs*.
    """
    return solve_helmholtz_dst(rhs, dx, dy, lambda_=0.0)


# ---------------------------------------------------------------------------
# Neumann (DCT-II) — 2D
# ---------------------------------------------------------------------------


def solve_helmholtz_dct(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float = 0.0,
) -> Float[Array, "Ny Nx"]:
    """Solve (∇² − λ)ψ = f with homogeneous Neumann BCs using DCT-II.

    Spectral algorithm:

    1. Forward 2-D DCT-II:  f̂ = DCT-II_y(DCT-II_x(f))       [Ny, Nx]
    2. Eigenvalue matrix:
           Λ[j,i] = Λ_j^y + Λ_i^x − λ                       [Ny, Nx]
       where Λ^x = dct2_eigenvalues(Nx, dx),
             Λ^y = dct2_eigenvalues(Ny, dy)
    3. Spectral division:  ψ̂[j,i] = f̂[j,i] / Λ[j,i]        [Ny, Nx]
    4. Inverse 2-D DCT-II:  ψ = IDCT-II_y(IDCT-II_x(ψ̂))    [Ny, Nx]

    When ``lambda_ == 0`` the (0,0) mode is singular (Λ[0,0] = 0,
    corresponding to the constant null mode of the Neumann Laplacian).
    This is handled by setting ψ̂[0,0] = 0 (zero-mean gauge).

    Parameters
    ----------
    rhs : Float[Array, "Ny Nx"]
        Right-hand side.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter λ.  Default: 0.0 (Poisson).

    Returns
    -------
    Float[Array, "Ny Nx"]
        Solution ψ, same shape as *rhs*.
    """
    Ny, Nx = rhs.shape
    rhs_hat = dctn(rhs, type=2, axes=[0, 1])
    eigx = dct2_eigenvalues(Nx, dx)
    eigy = dct2_eigenvalues(Ny, dy)
    eig2d = eigy[:, None] + eigx[None, :] - lambda_
    is_null = eig2d[0, 0] == 0.0
    eig2d_safe = eig2d.at[0, 0].set(jnp.where(is_null, 1.0, eig2d[0, 0]))
    psi_hat = rhs_hat / eig2d_safe
    psi_hat = psi_hat.at[0, 0].set(jnp.where(is_null, 0.0, psi_hat[0, 0]))
    return idctn(psi_hat, type=2, axes=[0, 1])


def solve_poisson_dct(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "Ny Nx"]:
    """Solve ∇²ψ = f with homogeneous Neumann BCs using DCT-II.

    The Poisson problem has a one-dimensional null space (constant solutions).
    This function fixes the gauge by forcing the domain-mean of ψ to zero.

    Parameters
    ----------
    rhs : Float[Array, "Ny Nx"]
        Right-hand side.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Zero-mean solution ψ, same shape as *rhs*.
    """
    return solve_helmholtz_dct(rhs, dx, dy, lambda_=0.0)


# ---------------------------------------------------------------------------
# Dirichlet, regular (DST-I) — 2D (explicit name)
# ---------------------------------------------------------------------------

# solve_helmholtz_dst1 and solve_poisson_dst1 are aliases defined below.


# ---------------------------------------------------------------------------
# Dirichlet, staggered (DST-II/III) — 2D
# ---------------------------------------------------------------------------


def solve_helmholtz_dst2(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float = 0.0,
) -> Float[Array, "Ny Nx"]:
    """Solve (∇² − λ)ψ = f with Dirichlet BCs on a staggered grid (DST-II).

    The input *rhs* lives on a cell-centred (staggered) grid; boundary
    values ψ = 0 are located half a grid spacing outside the first and
    last rows/columns.

    Spectral algorithm:

    1. Forward 2-D DST-II:  f̂ = DST-II_y(DST-II_x(f))
    2. Eigenvalue matrix:
           Λ[j,i] = Λ_j^y + Λ_i^x − λ
       where Λ^x = dst2_eigenvalues(Nx, dx),
             Λ^y = dst2_eigenvalues(Ny, dy)
       All eigenvalues are strictly negative, so the system is always
       non-singular for λ ≥ 0.
    3. Spectral division:  ψ̂[j,i] = f̂[j,i] / Λ[j,i]
    4. Inverse 2-D DST-II:  ψ = IDST-II_y(IDST-II_x(ψ̂))

    Parameters
    ----------
    rhs : Float[Array, "Ny Nx"]
        Right-hand side at cell-centred grid points.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter λ.  Default: 0.0 (Poisson).

    Returns
    -------
    Float[Array, "Ny Nx"]
        Solution ψ at cell-centred grid points, same shape as *rhs*.
    """
    Ny, Nx = rhs.shape
    rhs_hat = dstn(rhs, type=2, axes=[0, 1])
    eigx = dst2_eigenvalues(Nx, dx)
    eigy = dst2_eigenvalues(Ny, dy)
    eig2d = eigy[:, None] + eigx[None, :] - lambda_
    psi_hat = rhs_hat / eig2d
    return idstn(psi_hat, type=2, axes=[0, 1])


def solve_poisson_dst2(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "Ny Nx"]:
    """Solve ∇²ψ = f with Dirichlet BCs on a staggered grid (DST-II).

    Parameters
    ----------
    rhs : Float[Array, "Ny Nx"]
        Right-hand side at cell-centred grid points.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Solution ψ, same shape as *rhs*.
    """
    return solve_helmholtz_dst2(rhs, dx, dy, lambda_=0.0)


# ---------------------------------------------------------------------------
# Neumann, regular (DCT-I) — 2D
# ---------------------------------------------------------------------------


def solve_helmholtz_dct1(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float = 0.0,
) -> Float[Array, "Ny Nx"]:
    """Solve (∇² − λ)ψ = f with Neumann BCs on a regular grid (DCT-I).

    The input *rhs* lives on a vertex-centred (regular) grid; the first
    and last rows/columns coincide with the domain boundary where
    ∂ψ/∂n = 0.

    Spectral algorithm:

    1. Forward 2-D DCT-I:  f̂ = DCT-I_y(DCT-I_x(f))
    2. Eigenvalue matrix:
           Λ[j,i] = Λ_j^y + Λ_i^x − λ
       where Λ^x = dct1_eigenvalues(Nx, dx),
             Λ^y = dct1_eigenvalues(Ny, dy)
    3. Spectral division:  ψ̂[j,i] = f̂[j,i] / Λ[j,i]
    4. Inverse 2-D DCT-I:  ψ = IDCT-I_y(IDCT-I_x(ψ̂))

    When ``lambda_ == 0`` the (0,0) mode is singular (Λ[0,0] = 0,
    corresponding to the constant null mode of the Neumann Laplacian).
    This is handled by setting ψ̂[0,0] = 0 (zero-mean gauge).

    Parameters
    ----------
    rhs : Float[Array, "Ny Nx"]
        Right-hand side (including boundary grid points).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter λ.  Default: 0.0 (Poisson).

    Returns
    -------
    Float[Array, "Ny Nx"]
        Solution ψ, same shape as *rhs*.
    """
    Ny, Nx = rhs.shape
    rhs_hat = dctn(rhs, type=1, axes=[0, 1])
    eigx = dct1_eigenvalues(Nx, dx)
    eigy = dct1_eigenvalues(Ny, dy)
    eig2d = eigy[:, None] + eigx[None, :] - lambda_
    is_null = eig2d[0, 0] == 0.0
    eig2d_safe = eig2d.at[0, 0].set(jnp.where(is_null, 1.0, eig2d[0, 0]))
    psi_hat = rhs_hat / eig2d_safe
    psi_hat = psi_hat.at[0, 0].set(jnp.where(is_null, 0.0, psi_hat[0, 0]))
    return idctn(psi_hat, type=1, axes=[0, 1])


def solve_poisson_dct1(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "Ny Nx"]:
    """Solve ∇²ψ = f with Neumann BCs on a regular grid (DCT-I).

    Parameters
    ----------
    rhs : Float[Array, "Ny Nx"]
        Right-hand side (including boundary grid points).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Zero-mean solution ψ, same shape as *rhs*.
    """
    return solve_helmholtz_dct1(rhs, dx, dy, lambda_=0.0)


# ---------------------------------------------------------------------------
# Neumann, staggered (DCT-II) — 2D (explicit name)
# ---------------------------------------------------------------------------

# solve_helmholtz_dct2 and solve_poisson_dct2 are aliases defined below.


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Old names → explicit DST/DCT type-number names
solve_helmholtz_dst1 = solve_helmholtz_dst
solve_poisson_dst1 = solve_poisson_dst
solve_helmholtz_dct2 = solve_helmholtz_dct
solve_poisson_dct2 = solve_poisson_dct

# ---------------------------------------------------------------------------
# Periodic (FFT) — 3D
# ---------------------------------------------------------------------------


def solve_helmholtz_fft_3d(
    rhs: Float[Array, "Nz Ny Nx"],
    dx: float,
    dy: float,
    dz: float,
    lambda_: float = 0.0,
) -> Float[Array, "Nz Ny Nx"]:
    """Solve (∇² − λ)ψ = f with periodic BCs using the 3-D FFT.

    Parameters
    ----------
    rhs : Float[Array, "Nz Ny Nx"]
        Right-hand side on the triply periodic domain.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    dz : float
        Grid spacing in z.
    lambda_ : float
        Helmholtz parameter λ.  Default: 0.0 (Poisson).

    Returns
    -------
    Float[Array, "Nz Ny Nx"]
        Solution ψ (real), same shape as *rhs*.
    """
    Nz, Ny, Nx = rhs.shape
    rhs_hat = jnp.fft.fftn(rhs)
    eigx = fft_eigenvalues(Nx, dx)
    eigy = fft_eigenvalues(Ny, dy)
    eigz = fft_eigenvalues(Nz, dz)
    eig3d = eigz[:, None, None] + eigy[None, :, None] + eigx[None, None, :] - lambda_
    is_null = eig3d[0, 0, 0] == 0.0
    eig3d_safe = eig3d.at[0, 0, 0].set(jnp.where(is_null, 1.0, eig3d[0, 0, 0]))
    psi_hat = rhs_hat / eig3d_safe
    psi_hat = psi_hat.at[0, 0, 0].set(
        jnp.where(is_null, jnp.zeros_like(psi_hat[0, 0, 0]), psi_hat[0, 0, 0])
    )
    return jnp.real(jnp.fft.ifftn(psi_hat))


def solve_poisson_fft_3d(
    rhs: Float[Array, "Nz Ny Nx"],
    dx: float,
    dy: float,
    dz: float,
) -> Float[Array, "Nz Ny Nx"]:
    """Solve ∇²ψ = f with periodic BCs using the 3-D FFT.

    Parameters
    ----------
    rhs : Float[Array, "Nz Ny Nx"]
        Right-hand side on the triply periodic domain.
    dx, dy, dz : float
        Grid spacings.

    Returns
    -------
    Float[Array, "Nz Ny Nx"]
        Zero-mean solution ψ (real), same shape as *rhs*.
    """
    return solve_helmholtz_fft_3d(rhs, dx, dy, dz, lambda_=0.0)


# ---------------------------------------------------------------------------
# Dirichlet, regular (DST-I) — 3D
# ---------------------------------------------------------------------------


def solve_helmholtz_dst1_3d(
    rhs: Float[Array, "Nz Ny Nx"],
    dx: float,
    dy: float,
    dz: float,
    lambda_: float = 0.0,
) -> Float[Array, "Nz Ny Nx"]:
    """Solve (∇² − λ)ψ = f with Dirichlet BCs on a regular 3-D grid (DST-I).

    Parameters
    ----------
    rhs : Float[Array, "Nz Ny Nx"]
        Right-hand side at interior grid points.
    dx, dy, dz : float
        Grid spacings.
    lambda_ : float
        Helmholtz parameter λ.  Default: 0.0 (Poisson).

    Returns
    -------
    Float[Array, "Nz Ny Nx"]
        Solution ψ at interior grid points, same shape as *rhs*.
    """
    Nz, Ny, Nx = rhs.shape
    rhs_hat = dstn(rhs, type=1, axes=[0, 1, 2])
    eigx = dst1_eigenvalues(Nx, dx)
    eigy = dst1_eigenvalues(Ny, dy)
    eigz = dst1_eigenvalues(Nz, dz)
    eig3d = eigz[:, None, None] + eigy[None, :, None] + eigx[None, None, :] - lambda_
    psi_hat = rhs_hat / eig3d
    return idstn(psi_hat, type=1, axes=[0, 1, 2])


def solve_poisson_dst1_3d(
    rhs: Float[Array, "Nz Ny Nx"],
    dx: float,
    dy: float,
    dz: float,
) -> Float[Array, "Nz Ny Nx"]:
    """Solve ∇²ψ = f with Dirichlet BCs on a regular 3-D grid (DST-I).

    Parameters
    ----------
    rhs : Float[Array, "Nz Ny Nx"]
        Right-hand side at interior grid points.
    dx, dy, dz : float
        Grid spacings.

    Returns
    -------
    Float[Array, "Nz Ny Nx"]
        Solution ψ, same shape as *rhs*.
    """
    return solve_helmholtz_dst1_3d(rhs, dx, dy, dz, lambda_=0.0)


# ---------------------------------------------------------------------------
# Dirichlet, staggered (DST-II/III) — 3D
# ---------------------------------------------------------------------------


def solve_helmholtz_dst2_3d(
    rhs: Float[Array, "Nz Ny Nx"],
    dx: float,
    dy: float,
    dz: float,
    lambda_: float = 0.0,
) -> Float[Array, "Nz Ny Nx"]:
    """Solve (∇² − λ)ψ = f with Dirichlet BCs on a staggered 3-D grid (DST-II).

    Parameters
    ----------
    rhs : Float[Array, "Nz Ny Nx"]
        Right-hand side at cell-centred grid points.
    dx, dy, dz : float
        Grid spacings.
    lambda_ : float
        Helmholtz parameter λ.  Default: 0.0 (Poisson).

    Returns
    -------
    Float[Array, "Nz Ny Nx"]
        Solution ψ, same shape as *rhs*.
    """
    Nz, Ny, Nx = rhs.shape
    rhs_hat = dstn(rhs, type=2, axes=[0, 1, 2])
    eigx = dst2_eigenvalues(Nx, dx)
    eigy = dst2_eigenvalues(Ny, dy)
    eigz = dst2_eigenvalues(Nz, dz)
    eig3d = eigz[:, None, None] + eigy[None, :, None] + eigx[None, None, :] - lambda_
    psi_hat = rhs_hat / eig3d
    return idstn(psi_hat, type=2, axes=[0, 1, 2])


def solve_poisson_dst2_3d(
    rhs: Float[Array, "Nz Ny Nx"],
    dx: float,
    dy: float,
    dz: float,
) -> Float[Array, "Nz Ny Nx"]:
    """Solve ∇²ψ = f with Dirichlet BCs on a staggered 3-D grid (DST-II).

    Parameters
    ----------
    rhs : Float[Array, "Nz Ny Nx"]
        Right-hand side at cell-centred grid points.
    dx, dy, dz : float
        Grid spacings.

    Returns
    -------
    Float[Array, "Nz Ny Nx"]
        Solution ψ, same shape as *rhs*.
    """
    return solve_helmholtz_dst2_3d(rhs, dx, dy, dz, lambda_=0.0)


# ---------------------------------------------------------------------------
# Neumann, regular (DCT-I) — 3D
# ---------------------------------------------------------------------------


def solve_helmholtz_dct1_3d(
    rhs: Float[Array, "Nz Ny Nx"],
    dx: float,
    dy: float,
    dz: float,
    lambda_: float = 0.0,
) -> Float[Array, "Nz Ny Nx"]:
    """Solve (∇² − λ)ψ = f with Neumann BCs on a regular 3-D grid (DCT-I).

    Parameters
    ----------
    rhs : Float[Array, "Nz Ny Nx"]
        Right-hand side (including boundary grid points).
    dx, dy, dz : float
        Grid spacings.
    lambda_ : float
        Helmholtz parameter λ.  Default: 0.0 (Poisson).

    Returns
    -------
    Float[Array, "Nz Ny Nx"]
        Solution ψ, same shape as *rhs*.
    """
    Nz, Ny, Nx = rhs.shape
    rhs_hat = dctn(rhs, type=1, axes=[0, 1, 2])
    eigx = dct1_eigenvalues(Nx, dx)
    eigy = dct1_eigenvalues(Ny, dy)
    eigz = dct1_eigenvalues(Nz, dz)
    eig3d = eigz[:, None, None] + eigy[None, :, None] + eigx[None, None, :] - lambda_
    is_null = eig3d[0, 0, 0] == 0.0
    eig3d_safe = eig3d.at[0, 0, 0].set(jnp.where(is_null, 1.0, eig3d[0, 0, 0]))
    psi_hat = rhs_hat / eig3d_safe
    psi_hat = psi_hat.at[0, 0, 0].set(jnp.where(is_null, 0.0, psi_hat[0, 0, 0]))
    return idctn(psi_hat, type=1, axes=[0, 1, 2])


def solve_poisson_dct1_3d(
    rhs: Float[Array, "Nz Ny Nx"],
    dx: float,
    dy: float,
    dz: float,
) -> Float[Array, "Nz Ny Nx"]:
    """Solve ∇²ψ = f with Neumann BCs on a regular 3-D grid (DCT-I).

    Parameters
    ----------
    rhs : Float[Array, "Nz Ny Nx"]
        Right-hand side (including boundary grid points).
    dx, dy, dz : float
        Grid spacings.

    Returns
    -------
    Float[Array, "Nz Ny Nx"]
        Zero-mean solution ψ, same shape as *rhs*.
    """
    return solve_helmholtz_dct1_3d(rhs, dx, dy, dz, lambda_=0.0)


# ---------------------------------------------------------------------------
# Neumann, staggered (DCT-II/III) — 3D
# ---------------------------------------------------------------------------


def solve_helmholtz_dct2_3d(
    rhs: Float[Array, "Nz Ny Nx"],
    dx: float,
    dy: float,
    dz: float,
    lambda_: float = 0.0,
) -> Float[Array, "Nz Ny Nx"]:
    """Solve (∇² − λ)ψ = f with Neumann BCs on a staggered 3-D grid (DCT-II).

    Parameters
    ----------
    rhs : Float[Array, "Nz Ny Nx"]
        Right-hand side at cell-centred grid points.
    dx, dy, dz : float
        Grid spacings.
    lambda_ : float
        Helmholtz parameter λ.  Default: 0.0 (Poisson).

    Returns
    -------
    Float[Array, "Nz Ny Nx"]
        Solution ψ, same shape as *rhs*.
    """
    Nz, Ny, Nx = rhs.shape
    rhs_hat = dctn(rhs, type=2, axes=[0, 1, 2])
    eigx = dct2_eigenvalues(Nx, dx)
    eigy = dct2_eigenvalues(Ny, dy)
    eigz = dct2_eigenvalues(Nz, dz)
    eig3d = eigz[:, None, None] + eigy[None, :, None] + eigx[None, None, :] - lambda_
    is_null = eig3d[0, 0, 0] == 0.0
    eig3d_safe = eig3d.at[0, 0, 0].set(jnp.where(is_null, 1.0, eig3d[0, 0, 0]))
    psi_hat = rhs_hat / eig3d_safe
    psi_hat = psi_hat.at[0, 0, 0].set(jnp.where(is_null, 0.0, psi_hat[0, 0, 0]))
    return idctn(psi_hat, type=2, axes=[0, 1, 2])


def solve_poisson_dct2_3d(
    rhs: Float[Array, "Nz Ny Nx"],
    dx: float,
    dy: float,
    dz: float,
) -> Float[Array, "Nz Ny Nx"]:
    """Solve ∇²ψ = f with Neumann BCs on a staggered 3-D grid (DCT-II).

    Parameters
    ----------
    rhs : Float[Array, "Nz Ny Nx"]
        Right-hand side at cell-centred grid points.
    dx, dy, dz : float
        Grid spacings.

    Returns
    -------
    Float[Array, "Nz Ny Nx"]
        Zero-mean solution ψ, same shape as *rhs*.
    """
    return solve_helmholtz_dct2_3d(rhs, dx, dy, dz, lambda_=0.0)


# ===========================================================================
# Layer 1 — Module classes (eqx.Module wrappers)
# ===========================================================================

# ---------------------------------------------------------------------------
# Periodic (FFT) — existing, refactored
# ---------------------------------------------------------------------------


class SpectralHelmholtzSolver1D(eqx.Module):
    """1D Helmholtz/Poisson solver with periodic BCs using FFT.

    Solves ``(d²/dx² − α)φ = f`` on a periodic 1-D domain [0, L) using
    continuous Fourier wavenumbers:

        φ̂_k = −f̂_k / (k² + α)

    where k = 2πm/L are the Fourier wavenumbers from ``grid.k``.

    For α = 0 (Poisson), the k=0 mode is singular; ``zero_mean=True``
    projects it out (sets φ̂_0 = 0).

    Attributes
    ----------
    grid : FourierGrid1D
        1-D Fourier grid providing wavenumbers and FFT methods.
    """

    grid: FourierGrid1D

    def solve(
        self,
        f: Array,
        alpha: float = 0.0,
        zero_mean: bool = True,
        spectral: bool = False,
    ) -> Array:
        """Solve (d²/dx² − α)φ = f.

        Parameters
        ----------
        f : Float[Array, "N"]
            Source term in physical space.
        alpha : float
            Helmholtz parameter (α ≥ 0).  Default: 0.0 (Poisson).
        zero_mean : bool
            Force the mean (k=0 mode) to zero.  Default: True.
        spectral : bool
            If True, *f* is already in spectral space.

        Returns
        -------
        Float[Array, "N"]
            Solution φ in physical space.
        """
        f_hat = f if spectral else self.grid.transform(f)
        k2 = self.grid.k**2
        denom = k2 + alpha  # k^2 + alpha  [N]

        denom_safe = jnp.where(denom == 0.0, 1.0, denom)
        phi_hat = -f_hat / denom_safe  # phi_hat = -f_hat/(k^2+alpha)  [N]

        if zero_mean:
            phi_hat = jnp.where(k2 == 0.0, 0.0, phi_hat)

        return self.grid.transform(phi_hat, inverse=True).real


class SpectralHelmholtzSolver2D(eqx.Module):
    """2D Helmholtz/Poisson solver with periodic BCs using FFT.

    Solves ``(∇² − α)φ = f`` on a doubly periodic domain using continuous
    Fourier wavenumbers:

        φ̂[j,i] = −f̂[j,i] / (kx_i² + ky_j² + α)

    where |k|² = kx² + ky² is provided by ``grid.K2``.

    For α = 0 (Poisson), the (0,0) mode is singular; ``zero_mean=True``
    projects it out.

    Attributes
    ----------
    grid : FourierGrid2D
        2-D Fourier grid providing wavenumber meshgrid and FFT methods.
    """

    grid: FourierGrid2D

    def solve(
        self,
        f: Array,
        alpha: float = 0.0,
        zero_mean: bool = True,
        spectral: bool = False,
    ) -> Array:
        """Solve (∇² − α)φ = f.

        Parameters
        ----------
        f : Array [Ny, Nx]
            Source term in physical space.
        alpha : float
            Helmholtz parameter (α ≥ 0).  Default: 0.0 (Poisson).
        zero_mean : bool
            Force the (0,0) mode to zero.  Default: True.
        spectral : bool
            If True, *f* is already in spectral space.

        Returns
        -------
        Array [Ny, Nx]
            Solution φ in physical space.
        """
        f_hat = f if spectral else self.grid.transform(f)
        K2 = self.grid.K2  # kx^2 + ky^2  [Ny, Nx]
        denom = K2 + alpha

        denom_safe = jnp.where(denom == 0.0, 1.0, denom)
        phi_hat = -f_hat / denom_safe  # -f_hat / (|k|^2 + alpha)

        if zero_mean:
            phi_hat = jnp.where(K2 == 0.0, 0.0, phi_hat)

        return self.grid.transform(phi_hat, inverse=True).real


class SpectralHelmholtzSolver3D(eqx.Module):
    """3D Helmholtz/Poisson solver with periodic BCs using FFT.

    Solves ``(∇² − α)φ = f`` on a triply periodic domain using continuous
    Fourier wavenumbers:

        φ̂[l,j,i] = −f̂[l,j,i] / (kx_i² + ky_j² + kz_l² + α)

    where |k|² = kx² + ky² + kz² is provided by ``grid.K2``.

    For α = 0 (Poisson), the (0,0,0) mode is singular; ``zero_mean=True``
    projects it out.

    Attributes
    ----------
    grid : FourierGrid3D
        3-D Fourier grid providing wavenumber meshgrid and FFT methods.
    """

    grid: FourierGrid3D

    def solve(
        self,
        f: Array,
        alpha: float = 0.0,
        zero_mean: bool = True,
        spectral: bool = False,
    ) -> Array:
        """Solve (∇² − α)φ = f.

        Parameters
        ----------
        f : Array [Nz, Ny, Nx]
            Source term in physical space.
        alpha : float
            Helmholtz parameter (α ≥ 0).  Default: 0.0 (Poisson).
        zero_mean : bool
            Force the (0,0,0) mode to zero.  Default: True.
        spectral : bool
            If True, *f* is already in spectral space.

        Returns
        -------
        Array [Nz, Ny, Nx]
            Solution φ in physical space.
        """
        f_hat = f if spectral else self.grid.transform(f)
        K2 = self.grid.K2  # kx^2 + ky^2 + kz^2  [Nz, Ny, Nx]
        denom = K2 + alpha

        denom_safe = jnp.where(denom == 0.0, 1.0, denom)
        phi_hat = -f_hat / denom_safe  # -f_hat / (|k|^2 + alpha)

        if zero_mean:
            phi_hat = jnp.where(K2 == 0.0, 0.0, phi_hat)

        return self.grid.transform(phi_hat, inverse=True).real


# ---------------------------------------------------------------------------
# Dirichlet (DST-I) — new
# ---------------------------------------------------------------------------


class DirichletHelmholtzSolver2D(eqx.Module):
    """2D Helmholtz/Poisson/Laplace solver with homogeneous Dirichlet BCs.

    Solves ``(∇² − α)ψ = f`` where ψ = 0 on all four edges, using the
    DST-I spectral method (see :func:`solve_helmholtz_dst`).

    The input *rhs* contains values at the Ny × Nx **interior** grid
    points; boundary values are implicitly zero.

    With ``alpha=0.0`` (default), solves the Poisson equation ∇²ψ = f.

    Attributes
    ----------
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    alpha : float
        Helmholtz parameter α ≥ 0.  Default: 0.0 (Poisson/Laplace).
    """

    dx: float
    dy: float
    alpha: float = 0.0

    def __call__(self, rhs: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Solve (∇² − α)ψ = rhs.

        Parameters
        ----------
        rhs : Float[Array, "Ny Nx"]
            Right-hand side at interior grid points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Solution ψ at interior grid points.
        """
        return solve_helmholtz_dst(rhs, self.dx, self.dy, self.alpha)


# ---------------------------------------------------------------------------
# Neumann (DCT-II) — new
# ---------------------------------------------------------------------------


class NeumannHelmholtzSolver2D(eqx.Module):
    """2D Helmholtz/Poisson/Laplace solver with homogeneous Neumann BCs.

    Solves ``(∇² − α)ψ = f`` where ∂ψ/∂n = 0 on all four edges, using
    the DCT-II spectral method (see :func:`solve_helmholtz_dct`).

    With ``alpha=0.0`` (default), solves the Poisson equation ∇²ψ = f.
    The Poisson null space (constant mode) is removed by enforcing
    zero-mean: ψ̂[0,0] = 0.

    Attributes
    ----------
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    alpha : float
        Helmholtz parameter α ≥ 0.  Default: 0.0 (Poisson/Laplace).
    """

    dx: float
    dy: float
    alpha: float = 0.0

    def __call__(self, rhs: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Solve (∇² − α)ψ = rhs.

        Parameters
        ----------
        rhs : Float[Array, "Ny Nx"]
            Right-hand side.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Solution ψ (zero-mean gauge when α = 0).
        """
        return solve_helmholtz_dct(rhs, self.dx, self.dy, self.alpha)


# ---------------------------------------------------------------------------
# Dirichlet, staggered (DST-II) — new
# ---------------------------------------------------------------------------


class StaggeredDirichletHelmholtzSolver2D(eqx.Module):
    """2D Helmholtz/Poisson solver with Dirichlet BCs on a staggered grid.

    Solves ``(∇² − α)ψ = f`` where ψ = 0 at the cell edges (half a grid
    spacing outside the first and last cell centres), using the DST-II
    spectral method (see :func:`solve_helmholtz_dst2`).

    Attributes
    ----------
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    alpha : float
        Helmholtz parameter α ≥ 0.  Default: 0.0 (Poisson).
    """

    dx: float
    dy: float
    alpha: float = 0.0

    def __call__(self, rhs: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Solve (∇² − α)ψ = rhs.

        Parameters
        ----------
        rhs : Float[Array, "Ny Nx"]
            Right-hand side at cell-centred grid points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Solution ψ at cell-centred grid points.
        """
        return solve_helmholtz_dst2(rhs, self.dx, self.dy, self.alpha)


# ---------------------------------------------------------------------------
# Neumann, regular (DCT-I) — new
# ---------------------------------------------------------------------------


class RegularNeumannHelmholtzSolver2D(eqx.Module):
    """2D Helmholtz/Poisson solver with Neumann BCs on a regular grid.

    Solves ``(∇² − α)ψ = f`` where ∂ψ/∂n = 0 on all four edges, using
    the DCT-I spectral method (see :func:`solve_helmholtz_dct1`).

    Attributes
    ----------
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    alpha : float
        Helmholtz parameter α ≥ 0.  Default: 0.0 (Poisson).
    """

    dx: float
    dy: float
    alpha: float = 0.0

    def __call__(self, rhs: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Solve (∇² − α)ψ = rhs.

        Parameters
        ----------
        rhs : Float[Array, "Ny Nx"]
            Right-hand side (including boundary grid points).

        Returns
        -------
        Float[Array, "Ny Nx"]
            Solution ψ (zero-mean gauge when α = 0).
        """
        return solve_helmholtz_dct1(rhs, self.dx, self.dy, self.alpha)
