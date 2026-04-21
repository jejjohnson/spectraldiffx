"""
Clenshaw–Curtis Quadrature on Chebyshev Grids
==============================================

Clenshaw–Curtis is the natural quadrature rule for Chebyshev–Gauss–Lobatto
nodes; it has rapid convergence for smooth integrands and is computed
from a single DCT, making it inexpensive to evaluate.  For an integrand
f on the Gauss–Lobatto nodes xⱼ = L·cos(πj/N) ∈ [−L, L],

    ∫_{−L}^{L} f(x) dx ≈ Σⱼ wⱼ f(xⱼ)

where the weights wⱼ depend only on N and L.

The weight formula used here (Waldvogel 2006, eqn. (2.8)) is:

    wⱼ = (c_j L / N) · [1 − Σ_{k=1..(N-1)/2} (2 / (4k² − 1)) · cos(2 k π j / N)
                        − (1/(N² − 1)) · cos(π j)   (only if N even)]

with cⱼ = 1 for j = 0, N and cⱼ = 2 otherwise.

References
----------
[1] Trefethen, L. N. (2000). Spectral Methods in MATLAB, ch. 12.
[2] Waldvogel, J. (2006). Fast construction of the Fejér and Clenshaw–Curtis
    quadrature rules.  BIT Numerical Mathematics, 46(1), 195–202.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float, Num
import numpy as np

from .grid import ChebyshevGrid1D, ChebyshevGrid2D

# Array-shape aliases mirror the rest of the backend:
#   "Npts"          — 1D Chebyshev grid size (N+1 for GL)
#   "Nypts Nxpts"   — 2D tensor-product grid


def _cc_weights_numpy(N: int, L: float = 1.0) -> np.ndarray:
    """Compute Clenshaw–Curtis weights in NumPy (scale-free, multiplied by L).

    Uses the Waldvogel (2006) closed-form series.  Returns a length-(N+1)
    vector that integrates polynomials of degree ≤ 2N−1 over [−L, L].
    """
    if N <= 0:
        raise ValueError(f"N must be >= 1, got {N}")
    n = np.arange(N + 1, dtype=np.float64)

    # v[k] = 2/(4k² − 1), k = 1 .. floor((N-1)/2), length ⌊(N-1)/2⌋
    K = (N - 1) // 2
    cos_term = np.zeros(N + 1)
    for k in range(1, K + 1):
        cos_term += 2.0 / (4.0 * k * k - 1.0) * np.cos(2.0 * k * np.pi * n / N)

    # Correction for even N (the l = N/2 term)
    if N % 2 == 0:
        cos_term += np.cos(np.pi * n) / (N * N - 1.0)

    base = (1.0 - cos_term) / N
    c = np.full(N + 1, 2.0)
    c[0] = 1.0
    c[N] = 1.0
    w = c * base * L
    return w


def clenshaw_curtis_weights(N: int, L: float = 1.0) -> Float[Array, "Npts"]:
    """Clenshaw–Curtis quadrature weights on Gauss–Lobatto nodes of [−L, L].

    Parameters
    ----------
    N : int
        Chebyshev polynomial degree.  The grid has N+1 Gauss–Lobatto nodes.
    L : float
        Domain half-length (default 1).  The weights scale linearly with L.

    Returns
    -------
    Float[Array, "Npts"]
        Weights ``w`` such that ``∫_{−L}^{L} f(x) dx ≈ Σⱼ w[j] · f(xⱼ)`` for
        any integrand, exact for polynomials of degree ≤ 2N−1 on smooth f.

    Examples
    --------
    Integrate cos(πx) on [−1, 1] (exact value 2 sin(π)/π ≈ 0):

    >>> import jax.numpy as jnp
    >>> from jax.scipy.special import ...  # doctest: +SKIP
    >>> w = clenshaw_curtis_weights(N=32, L=1.0)
    >>> x = 1.0 * jnp.cos(jnp.pi * jnp.arange(33) / 32)
    >>> float(jnp.sum(w * jnp.cos(jnp.pi * x)))  # doctest: +SKIP
    """
    return jnp.asarray(_cc_weights_numpy(N, L))


def clenshaw_curtis_integrate_1d(
    grid: ChebyshevGrid1D,
    f: Num[Array, "Npts"],
) -> Float[Array, ""]:
    """Integrate a 1D nodal field over [−L, L] using Clenshaw–Curtis.

    The grid must use Gauss–Lobatto nodes (Gauss nodes would require a
    different quadrature rule — Gauss–Chebyshev — which is not provided
    here because it is less accurate for smooth non-periodic f).

    Parameters
    ----------
    grid : ChebyshevGrid1D
        Must have ``node_type == 'gauss-lobatto'``.
    f : Num[Array, "Npts"]
        Nodal values of the integrand.

    Returns
    -------
    Float[Array, ""]
        Scalar approximation to ∫_{−L}^{L} f(x) dx.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> grid = ChebyshevGrid1D.from_N_L(N=32, L=1.0)
    >>> f = jnp.exp(grid.x)
    >>> I = clenshaw_curtis_integrate_1d(grid, f)  # ≈ e − 1/e
    """
    if grid.node_type != "gauss-lobatto":
        raise ValueError(
            "clenshaw_curtis_integrate_1d requires Gauss–Lobatto nodes; got "
            f"node_type={grid.node_type!r}."
        )
    w = clenshaw_curtis_weights(grid.N, grid.L)
    return jnp.sum(w * f)


def clenshaw_curtis_integrate_2d(
    grid: ChebyshevGrid2D,
    f: Num[Array, "Nypts Nxpts"],
) -> Float[Array, ""]:
    """Integrate a 2D nodal field over [−Lx, Lx] × [−Ly, Ly].

    Uses the tensor product of 1D Clenshaw–Curtis rules:

        ∫∫ f(x, y) dx dy ≈ Σⱼᵢ w_y[j] · w_x[i] · f[j, i]

    Parameters
    ----------
    grid : ChebyshevGrid2D
        Must use Gauss–Lobatto nodes in both directions.
    f : Num[Array, "Nypts Nxpts"]
        Nodal values of the integrand on the (Nᵧ+1, Nₓ+1) grid.

    Returns
    -------
    Float[Array, ""]
        Scalar approximation to ∫∫ f dA.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> grid = ChebyshevGrid2D.from_N_L(Nx=24, Ny=24, Lx=1.0, Ly=1.0)
    >>> X, Y = grid.X
    >>> f = jnp.exp(X + Y)
    >>> I = clenshaw_curtis_integrate_2d(grid, f)  # ≈ (e − 1/e)²
    """
    if grid.node_type != "gauss-lobatto":
        raise ValueError(
            "clenshaw_curtis_integrate_2d requires Gauss–Lobatto nodes; got "
            f"node_type={grid.node_type!r}."
        )
    wx = clenshaw_curtis_weights(grid.Nx, grid.Lx)  # (Nxpts,)
    wy = clenshaw_curtis_weights(grid.Ny, grid.Ly)  # (Nypts,)
    W = wy[:, None] * wx[None, :]  # (Nypts, Nxpts)
    return jnp.sum(W * f)
