"""1-D Laplacian eigenvalues for spectral elliptic solvers.

Each helper returns the eigenvalues of the second-order finite-difference
Laplacian discretised on *N* points with grid spacing *dx*, diagonalised by
the corresponding spectral transform.

* DST-I  (Dirichlet BCs):
    λ_k = −4/dx² · sin²(π(k+1) / (2(N+1)))   k = 0, …, N−1

* DCT-II (Neumann BCs):
    λ_k = −4/dx² · sin²(πk / (2N))             k = 0, …, N−1

* FFT    (Periodic BCs):
    λ_k = −4/dx² · sin²(πk / N)               k = 0, …, N−1
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
