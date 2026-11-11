"""
Clenshaw–Curtis Quadrature on Chebyshev Grids
==============================================

Clenshaw–Curtis is the natural quadrature rule for Chebyshev–Gauss–Lobatto
nodes; it has rapid convergence for smooth integrands and is computed
from a single DCT, making it inexpensive to evaluate.  For an integrand
f on the Gauss–Lobatto nodes xⱼ = L·cos(πj/N) ∈ [−L, L],

    ∫_{−L}^{L} f(x) dx ≈ Σⱼ wⱼ f(xⱼ)

where the weights wⱼ depend only on N and L.  They come from
``functax.quad.clenshaw_curtis`` (Waldvogel 2006, eqn. (2.8)), whose nodes
ascend from −1; the weights are reversed to pair with the descending
Gauss–Lobatto nodes of :class:`ChebyshevGrid1D` and scaled by L.

References
----------
[1] Trefethen, L. N. (2000). Spectral Methods in MATLAB, ch. 12.
[2] Waldvogel, J. (2006). Fast construction of the Fejér and Clenshaw–Curtis
    quadrature rules.  BIT Numerical Mathematics, 46(1), 195–202.
"""

from __future__ import annotations

from functax.quad import clenshaw_curtis
import jax.numpy as jnp
from jaxtyping import Array, Float, Num
import numpy as np

from .grid import ChebyshevGrid1D, ChebyshevGrid2D

# Array-shape aliases mirror the rest of the backend:
#   "Npts"          — 1D Chebyshev grid size (N+1 for GL)
#   "Nypts Nxpts"   — 2D tensor-product grid


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
        Weights ``w`` such that ``∫_{−L}^{L} f(x) dx ≈ Σⱼ w[j] · f(xⱼ)``.
        Exact for polynomials of degree ≤ N (and degree N+1 when N is
        even).  For analytic integrands CC converges at the same
        asymptotic rate as Gauss quadrature despite the smaller degree of
        precision.

    Examples
    --------
    Integrate exp(x) on [−1, 1] (exact value e − 1/e):

    >>> import jax.numpy as jnp
    >>> w = clenshaw_curtis_weights(N=32, L=1.0)
    >>> x = jnp.cos(jnp.pi * jnp.arange(33) / 32)
    >>> float(jnp.sum(w * jnp.exp(x)))  # doctest: +SKIP
    """
    if N <= 0:
        raise ValueError(f"N must be >= 1, got {N}")
    # The rule on [−1, 1] (nodes ascending), reversed to the descending
    # Gauss–Lobatto order and scaled by L (which may be traced).
    weights = np.asarray(clenshaw_curtis(N + 1).weights)[::-1]
    return jnp.asarray(weights * L)


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
