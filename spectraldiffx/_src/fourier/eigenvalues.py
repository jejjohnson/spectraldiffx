"""1-D Laplacian eigenvalues for spectral elliptic solvers.

Two families of eigenvalues are provided:

**Finite-difference (FD2)** — exact inverse of the 3-point FD Laplacian.
Takes grid spacing *dx* as parameter.

**Pseudo-spectral (PS)** — continuous Laplacian eigenvalues.
Takes domain length *L* as parameter.  Gives spectral accuracy for smooth
solutions but is *not* the exact inverse of the FD stencil.

FD2 same-BC eigenvalues (both boundaries identical):

* DST-I   (Dirichlet, regular grid):
    λ_k = −4/dx² · sin²(π(k+1) / (2(N+1)))    k = 0, …, N−1

* DST-II  (Dirichlet, staggered grid):
    λ_k = −4/dx² · sin²(π(k+1) / (2N))         k = 0, …, N−1

* DCT-I   (Neumann, regular grid):
    λ_k = −4/dx² · sin²(πk / (2(N−1)))          k = 0, …, N−1

* DCT-II  (Neumann, staggered grid):
    λ_k = −4/dx² · sin²(πk / (2N))              k = 0, …, N−1

* FFT     (Periodic):
    λ_k = −4/dx² · sin²(πk / N)                 k = 0, …, N−1

FD2 mixed-BC eigenvalues (different BCs on left/right):

* DST-III (Dirichlet left + Neumann right, regular grid):
    λ_k = −4/dx² · sin²(π(2k+1) / (4N))        k = 0, …, N−1

* DCT-III (Neumann left + Dirichlet right, regular grid):
    λ_k = −4/dx² · sin²(π(2k+1) / (4N))        k = 0, …, N−1

* DST-IV  (Dirichlet left + Neumann right, staggered grid):
    λ_k = −4/dx² · sin²(π(2k+1) / (4N))        k = 0, …, N−1

* DCT-IV  (Neumann left + Dirichlet right, staggered grid):
    λ_k = −4/dx² · sin²(π(2k+1) / (4N))        k = 0, …, N−1

PS same-BC eigenvalues:

* DST-I / DST-II  (Dirichlet):
    λ_k = −(π(k+1)/L)²                          k = 0, …, N−1

* DCT-I / DCT-II  (Neumann):
    λ_k = −(πk/L)²                               k = 0, …, N−1

* FFT (Periodic):
    λ_k = −(2πk̃/L)²   k̃ = k if k < N/2 else N−k

PS mixed-BC eigenvalues:

* DST-III / DST-IV (Dirichlet left + Neumann right):
    λ_k = −(π(2k+1)/(2L))²                      k = 0, …, N−1

* DCT-III / DCT-IV (Neumann left + Dirichlet right):
    λ_k = −(π(2k+1)/(2L))²                      k = 0, …, N−1
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def dst1_eigenvalues(N: int, dx: float) -> Float[Array, " N"]:
    """1-D Laplacian eigenvalues for the DST-I basis (homogeneous Dirichlet BCs).

    Returns the eigenvalues of the second-order finite-difference Laplacian

        (L·v)_i = (v_{i-1} − 2v_i + v_{i+1}) / dx²

    on *N* interior points with v_0 = v_{N+1} = 0.  The DST-I diagonalises
    this operator, yielding:

        λ_k = −4/dx² · sin²(π(k+1) / (2(N+1))),   k = 0, …, N−1

    All eigenvalues are strictly negative (the discrete Dirichlet Laplacian
    is negative definite).

    Parameters
    ----------
    N : int
        Number of interior grid points (excluding the two zero-boundary points).
    dx : float
        Uniform grid spacing.

    Returns
    -------
    Float[Array, " N"]
        1-D array of *N* eigenvalues, ordered k = 0, …, N−1.  All < 0.
    """
    k = jnp.arange(N)
    return -4.0 / dx**2 * jnp.sin(jnp.pi * (k + 1) / (2 * (N + 1))) ** 2


def dct2_eigenvalues(N: int, dx: float) -> Float[Array, " N"]:
    """1-D Laplacian eigenvalues for the DCT-II basis (homogeneous Neumann BCs).

    Returns the eigenvalues of the second-order finite-difference Laplacian

        (L·v)_i = (v_{i-1} − 2v_i + v_{i+1}) / dx²

    on *N* points with ∂v/∂n = 0 at both boundaries (ghost-point Neumann).
    The DCT-II diagonalises this operator, yielding:

        λ_k = −4/dx² · sin²(πk / (2N)),   k = 0, …, N−1

    The k=0 eigenvalue is exactly zero (the constant mode), corresponding
    to the one-dimensional null space of the Neumann Laplacian.

    Parameters
    ----------
    N : int
        Number of grid points.
    dx : float
        Uniform grid spacing.

    Returns
    -------
    Float[Array, " N"]
        1-D array of *N* eigenvalues, ordered k = 0, …, N−1.
        λ_0 = 0; all others < 0.
    """
    k = jnp.arange(N)
    return -4.0 / dx**2 * jnp.sin(jnp.pi * k / (2 * N)) ** 2


def dst2_eigenvalues(N: int, dx: float) -> Float[Array, " N"]:
    """1-D Laplacian eigenvalues for the DST-II basis (Dirichlet BCs, staggered grid).

    Returns the eigenvalues of the second-order finite-difference Laplacian
    on *N* cell-centred points with homogeneous Dirichlet BCs at both
    boundaries (ψ = 0 at the cell edges, half a grid spacing outside the
    first and last points).

    The DST-II diagonalises this operator, yielding:

        λ_k = −4/dx² · sin²(π(k+1) / (2N)),   k = 0, …, N−1

    All eigenvalues are strictly negative.  The forward transform is DST-II;
    the inverse is DST-III (SciPy convention: ``idstn(type=2)``).

    Parameters
    ----------
    N : int
        Number of cell-centred grid points.
    dx : float
        Uniform grid spacing.

    Returns
    -------
    Float[Array, " N"]
        1-D array of *N* eigenvalues, ordered k = 0, …, N−1.  All < 0.
    """
    k = jnp.arange(N)
    return -4.0 / dx**2 * jnp.sin(jnp.pi * (k + 1) / (2 * N)) ** 2


def dct1_eigenvalues(N: int, dx: float) -> Float[Array, " N"]:
    """1-D Laplacian eigenvalues for the DCT-I basis (Neumann BCs, regular grid).

    Returns the eigenvalues of the second-order finite-difference Laplacian
    on *N* vertex-centred points with homogeneous Neumann BCs
    (∂ψ/∂n = 0 at both boundaries, which coincide with the first and last
    grid points).

    The DCT-I diagonalises this operator, yielding:

        λ_k = −4/dx² · sin²(πk / (2(N−1))),   k = 0, …, N−1

    The k = 0 eigenvalue is exactly zero (constant null mode).  Requires
    N ≥ 2.  The DCT-I is self-inverse (up to normalisation).

    Parameters
    ----------
    N : int
        Number of grid points (including both boundary points).
    dx : float
        Uniform grid spacing.

    Returns
    -------
    Float[Array, " N"]
        1-D array of *N* eigenvalues, ordered k = 0, …, N−1.
        λ_0 = 0; all others < 0.

    Raises
    ------
    ValueError
        If N < 2 (DCT-I requires at least two grid points).
    """
    if N < 2:
        raise ValueError(f"dct1_eigenvalues requires N >= 2, got N={N}")
    k = jnp.arange(N)
    return -4.0 / dx**2 * jnp.sin(jnp.pi * k / (2 * (N - 1))) ** 2


def _mixed_bc_eigenvalues(N: int, dx: float) -> Float[Array, " N"]:
    """Shared eigenvalue formula for all mixed-BC transforms (DST-III/IV, DCT-III/IV).

        λ_k = −4/dx² · sin²(π(2k+1) / (4N)),   k = 0, …, N−1

    All eigenvalues are strictly negative (no null mode).
    """
    k = jnp.arange(N)
    return -4.0 / dx**2 * jnp.sin(jnp.pi * (2 * k + 1) / (4 * N)) ** 2


def dst3_eigenvalues(N: int, dx: float) -> Float[Array, " N"]:
    """1-D Laplacian eigenvalues for Dirichlet-left + Neumann-right on a regular grid.

    Pairs with DST-III forward / DST-II backward.

        λ_k = −4/dx² · sin²(π(2k+1) / (4N)),   k = 0, …, N−1

    All eigenvalues are strictly negative.

    Parameters
    ----------
    N : int
        Number of grid points.
    dx : float
        Uniform grid spacing.

    Returns
    -------
    Float[Array, " N"]
        1-D array of *N* eigenvalues, ordered k = 0, …, N−1.  All < 0.
    """
    return _mixed_bc_eigenvalues(N, dx)


def dct3_eigenvalues(N: int, dx: float) -> Float[Array, " N"]:
    """1-D Laplacian eigenvalues for Neumann-left + Dirichlet-right on a regular grid.

    Pairs with DCT-III forward / DCT-II backward.

        λ_k = −4/dx² · sin²(π(2k+1) / (4N)),   k = 0, …, N−1

    All eigenvalues are strictly negative.

    Parameters
    ----------
    N : int
        Number of grid points.
    dx : float
        Uniform grid spacing.

    Returns
    -------
    Float[Array, " N"]
        1-D array of *N* eigenvalues, ordered k = 0, …, N−1.  All < 0.
    """
    return _mixed_bc_eigenvalues(N, dx)


def dst4_eigenvalues(N: int, dx: float) -> Float[Array, " N"]:
    """1-D Laplacian eigenvalues for Dirichlet-left + Neumann-right on a staggered grid.

    Pairs with DST-IV (self-inverse).

        λ_k = −4/dx² · sin²(π(2k+1) / (4N)),   k = 0, …, N−1

    All eigenvalues are strictly negative.

    Parameters
    ----------
    N : int
        Number of cell-centred grid points.
    dx : float
        Uniform grid spacing.

    Returns
    -------
    Float[Array, " N"]
        1-D array of *N* eigenvalues, ordered k = 0, …, N−1.  All < 0.
    """
    return _mixed_bc_eigenvalues(N, dx)


def dct4_eigenvalues(N: int, dx: float) -> Float[Array, " N"]:
    """1-D Laplacian eigenvalues for Neumann-left + Dirichlet-right on a staggered grid.

    Pairs with DCT-IV (self-inverse).

        λ_k = −4/dx² · sin²(π(2k+1) / (4N)),   k = 0, …, N−1

    All eigenvalues are strictly negative.

    Parameters
    ----------
    N : int
        Number of cell-centred grid points.
    dx : float
        Uniform grid spacing.

    Returns
    -------
    Float[Array, " N"]
        1-D array of *N* eigenvalues, ordered k = 0, …, N−1.  All < 0.
    """
    return _mixed_bc_eigenvalues(N, dx)


def fft_eigenvalues(N: int, dx: float) -> Float[Array, " N"]:
    """1-D Laplacian eigenvalues for the FFT basis (periodic BCs).

    Returns the eigenvalues of the second-order finite-difference Laplacian

        (L·v)_i = (v_{i-1} − 2v_i + v_{i+1}) / dx²

    on *N* points with periodic boundary conditions v_{N} = v_0.  The DFT
    diagonalises this operator, yielding:

        λ_k = −4/dx² · sin²(πk / N),   k = 0, …, N−1

    The k=0 eigenvalue is exactly zero (the constant / zero-wavenumber mode),
    corresponding to the one-dimensional null space of the periodic Laplacian.

    Parameters
    ----------
    N : int
        Number of grid points in one period.
    dx : float
        Uniform grid spacing.

    Returns
    -------
    Float[Array, " N"]
        1-D array of *N* eigenvalues in FFT order (k = 0, …, N−1).
        λ_0 = 0; all others ≤ 0.
    """
    k = jnp.arange(N)
    return -4.0 / dx**2 * jnp.sin(jnp.pi * k / N) ** 2


# ===========================================================================
# Pseudo-spectral (PS) eigenvalues — continuous Laplacian
# ===========================================================================


def dst1_eigenvalues_ps(N: int, L: float) -> Float[Array, " N"]:
    """PS eigenvalues for DST-I (Dirichlet BCs, regular grid).

    Eigenvalues of the continuous Laplacian d²/dx² on [0, L] with
    homogeneous Dirichlet BCs, projected onto the DST-I basis:

        λ_k = −(π(k+1)/L)²,   k = 0, …, N−1

    These agree with the FD2 eigenvalues for low wavenumbers (k << N)
    but diverge near Nyquist.

    Parameters
    ----------
    N : int
        Number of interior grid points.
    L : float
        Domain length.

    Returns
    -------
    Float[Array, " N"]
        1-D array of *N* eigenvalues, ordered k = 0, …, N−1.  All < 0.
    """
    k = jnp.arange(N)
    return -((jnp.pi * (k + 1) / L) ** 2)


def dst2_eigenvalues_ps(N: int, L: float) -> Float[Array, " N"]:
    """PS eigenvalues for DST-II (Dirichlet BCs, staggered grid).

    Same formula as DST-I PS (the continuous eigenvalues depend only on BC
    type, not grid placement):

        λ_k = −(π(k+1)/L)²,   k = 0, …, N−1

    Parameters
    ----------
    N : int
        Number of cell-centred grid points.
    L : float
        Domain length.

    Returns
    -------
    Float[Array, " N"]
        1-D array of *N* eigenvalues, ordered k = 0, …, N−1.  All < 0.
    """
    k = jnp.arange(N)
    return -((jnp.pi * (k + 1) / L) ** 2)


def dct1_eigenvalues_ps(N: int, L: float) -> Float[Array, " N"]:
    """PS eigenvalues for DCT-I (Neumann BCs, regular grid).

    Eigenvalues of the continuous Laplacian d²/dx² on [0, L] with
    homogeneous Neumann BCs, projected onto the DCT-I basis:

        λ_k = −(πk/L)²,   k = 0, …, N−1

    The k=0 eigenvalue is exactly zero (constant mode).

    Parameters
    ----------
    N : int
        Number of grid points (including both boundary points).
    L : float
        Domain length.

    Returns
    -------
    Float[Array, " N"]
        1-D array of *N* eigenvalues.  λ_0 = 0; all others < 0.
    """
    k = jnp.arange(N)
    return -((jnp.pi * k / L) ** 2)


def dct2_eigenvalues_ps(N: int, L: float) -> Float[Array, " N"]:
    """PS eigenvalues for DCT-II (Neumann BCs, staggered grid).

    Same formula as DCT-I PS:

        λ_k = −(πk/L)²,   k = 0, …, N−1

    The k=0 eigenvalue is exactly zero (constant mode).

    Parameters
    ----------
    N : int
        Number of grid points.
    L : float
        Domain length.

    Returns
    -------
    Float[Array, " N"]
        1-D array of *N* eigenvalues.  λ_0 = 0; all others < 0.
    """
    k = jnp.arange(N)
    return -((jnp.pi * k / L) ** 2)


def _mixed_bc_eigenvalues_ps(N: int, L: float) -> Float[Array, " N"]:
    """Shared PS eigenvalue formula for all mixed-BC transforms (DST-III/IV, DCT-III/IV).

        λ_k = −(π(2k+1) / (2L))²,   k = 0, …, N−1

    All eigenvalues are strictly negative (no null mode).
    """
    k = jnp.arange(N)
    return -((jnp.pi * (2 * k + 1) / (2 * L)) ** 2)


def dst3_eigenvalues_ps(N: int, L: float) -> Float[Array, " N"]:
    """PS eigenvalues for Dirichlet-left + Neumann-right on a regular grid.

        λ_k = −(π(2k+1) / (2L))²,   k = 0, …, N−1

    Parameters
    ----------
    N : int
        Number of grid points.
    L : float
        Domain length.

    Returns
    -------
    Float[Array, " N"]
        1-D array of *N* eigenvalues.  All < 0.
    """
    return _mixed_bc_eigenvalues_ps(N, L)


def dct3_eigenvalues_ps(N: int, L: float) -> Float[Array, " N"]:
    """PS eigenvalues for Neumann-left + Dirichlet-right on a regular grid.

        λ_k = −(π(2k+1) / (2L))²,   k = 0, …, N−1

    Parameters
    ----------
    N : int
        Number of grid points.
    L : float
        Domain length.

    Returns
    -------
    Float[Array, " N"]
        1-D array of *N* eigenvalues.  All < 0.
    """
    return _mixed_bc_eigenvalues_ps(N, L)


def dst4_eigenvalues_ps(N: int, L: float) -> Float[Array, " N"]:
    """PS eigenvalues for Dirichlet-left + Neumann-right on a staggered grid.

        λ_k = −(π(2k+1) / (2L))²,   k = 0, …, N−1

    Parameters
    ----------
    N : int
        Number of cell-centred grid points.
    L : float
        Domain length.

    Returns
    -------
    Float[Array, " N"]
        1-D array of *N* eigenvalues.  All < 0.
    """
    return _mixed_bc_eigenvalues_ps(N, L)


def dct4_eigenvalues_ps(N: int, L: float) -> Float[Array, " N"]:
    """PS eigenvalues for Neumann-left + Dirichlet-right on a staggered grid.

        λ_k = −(π(2k+1) / (2L))²,   k = 0, …, N−1

    Parameters
    ----------
    N : int
        Number of cell-centred grid points.
    L : float
        Domain length.

    Returns
    -------
    Float[Array, " N"]
        1-D array of *N* eigenvalues.  All < 0.
    """
    return _mixed_bc_eigenvalues_ps(N, L)


def fft_eigenvalues_ps(N: int, L: float) -> Float[Array, " N"]:
    """PS eigenvalues for the FFT basis (periodic BCs).

    Eigenvalues of the continuous Laplacian d²/dx² on a periodic domain
    of length L:

        λ_k = −(2πk̃/L)²,   k̃ = k for k < N/2,  k̃ = N−k for k ≥ N/2

    The k=0 eigenvalue is exactly zero (constant mode).

    Parameters
    ----------
    N : int
        Number of grid points in one period.
    L : float
        Domain length (period).

    Returns
    -------
    Float[Array, " N"]
        1-D array of *N* eigenvalues in FFT order.
        λ_0 = 0; all others ≤ 0.
    """
    k = jnp.arange(N)
    k_phys = jnp.where(k < N // 2, k, N - k)
    return -((2 * jnp.pi * k_phys / L) ** 2)
