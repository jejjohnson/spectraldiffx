"""
Chebyshev Transforms
====================

Standalone forward/inverse Chebyshev transform classes that wrap the
:class:`ChebyshevGrid1D` / :class:`ChebyshevGrid2D` transform methods with a
dedicated interface (mirroring the Fourier backend's
:class:`SpectralDerivative1D` → :class:`SpectralDerivative2D` wrappers).

Mathematical framework
----------------------
For u(x) on a Chebyshev grid over [−L, L] the expansion is

    u(x) = Σₖ aₖ Tₖ(x/L)     where Tₖ(ξ) = cos(k·arccos(ξ))

Forward transform (aₖ from uⱼ):
    Gauss-Lobatto:  aₖ = DCT-I{uⱼ}   implemented via a length-2N rFFT
    Gauss       :  aₖ = DCT-II{uⱼ}  with half-sample shift
Inverse:
    uⱼ = Σₖ aₖ Tₖ(xⱼ/L)

Conventions follow Trefethen (2000), ch. 6.  2D transforms are tensor
products of the 1D transforms.

Also provided
-------------
* :func:`dealias_product` — compute the dealiased pointwise product a·b
  on a Chebyshev grid using zero-padding in coefficient space.

References
----------
[1] Trefethen, L. N. (2000). Spectral Methods in MATLAB. SIAM.
[2] Boyd, J. P. (2001). Chebyshev and Fourier Spectral Methods. Dover.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Num

from .grid import ChebyshevGrid1D, ChebyshevGrid2D

# Array-shape aliases:
#   "Npts"          — 1D Chebyshev grid size (N+1 for GL, N for Gauss)
#   "Nypts Nxpts"   — 2D tensor-product grid


class ChebyshevTransform1D(eqx.Module):
    """Forward/inverse 1D Chebyshev transform.

    Thin wrapper around :meth:`ChebyshevGrid1D.transform` that exposes
    :meth:`to_spectral` / :meth:`from_spectral` methods.  Useful when you
    want to pass the transform around as a first-class object (e.g. into
    a PDE residual, or to a higher-order library).

    Attributes
    ----------
    grid : ChebyshevGrid1D
        Underlying grid carrying the node convention (GL or Gauss).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> grid = ChebyshevGrid1D.from_N_L(N=16, L=1.0)
    >>> cheb = ChebyshevTransform1D(grid=grid)
    >>> u = jnp.sin(jnp.pi * grid.x)
    >>> a = cheb.to_spectral(u)
    >>> u_roundtrip = cheb.from_spectral(a)  # ≈ u
    """

    grid: ChebyshevGrid1D

    def to_spectral(self, u: Num[Array, Npts]) -> Num[Array, Npts]:
        """Forward Chebyshev transform: nodal values → coefficients aₖ."""
        return self.grid.transform(u, inverse=False)

    def from_spectral(self, a: Num[Array, Npts]) -> Num[Array, Npts]:
        """Inverse Chebyshev transform: coefficients aₖ → nodal values."""
        return self.grid.transform(a, inverse=True)

    def __call__(self, u: Num[Array, Npts], inverse: bool = False) -> Num[Array, Npts]:
        """Alias for ``self.grid.transform(u, inverse=inverse)``."""
        return self.grid.transform(u, inverse=inverse)


class ChebyshevTransform2D(eqx.Module):
    """Forward/inverse 2D Chebyshev transform (tensor product of 1D).

    Attributes
    ----------
    grid : ChebyshevGrid2D
        Underlying 2D grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> grid = ChebyshevGrid2D.from_N_L(Nx=16, Ny=16, Lx=1.0, Ly=1.0)
    >>> cheb = ChebyshevTransform2D(grid=grid)
    >>> X, Y = grid.X
    >>> u = jnp.sin(jnp.pi * X) * jnp.cos(jnp.pi * Y)
    >>> a = cheb.to_spectral(u)
    >>> u_roundtrip = cheb.from_spectral(a)
    """

    grid: ChebyshevGrid2D

    def to_spectral(self, u: Num[Array, "Nypts Nxpts"]) -> Num[Array, "Nypts Nxpts"]:
        """Forward 2D Chebyshev transform: nodal values → coefficients a_{jk}."""
        return self.grid.transform(u, inverse=False)

    def from_spectral(self, a: Num[Array, "Nypts Nxpts"]) -> Num[Array, "Nypts Nxpts"]:
        """Inverse 2D Chebyshev transform."""
        return self.grid.transform(a, inverse=True)

    def __call__(
        self,
        u: Num[Array, "Nypts Nxpts"],
        inverse: bool = False,
    ) -> Num[Array, "Nypts Nxpts"]:
        """Alias for ``self.grid.transform(u, inverse=inverse)``."""
        return self.grid.transform(u, inverse=inverse)


# ============================================================================
# Dealiased product (2/3-rule style) on a Chebyshev grid
# ============================================================================


def _apply_dealias_1d(
    a: Num[Array, Npts], n_modes: int, cutoff: int
) -> Num[Array, Npts]:
    """Zero out Chebyshev coefficients with index > cutoff."""
    k = jnp.arange(n_modes)
    mask = (k <= cutoff).astype(a.dtype)
    return a * mask


def dealias_product(
    grid: ChebyshevGrid1D | ChebyshevGrid2D,
    a: Num[Array, ...],
    b: Num[Array, ...],
) -> Num[Array, ...]:
    """Compute the dealiased pointwise product a·b on a Chebyshev grid.

    Implements a 2/3-style truncation in Chebyshev-coefficient space:

        1. Forward-transform a and b to coefficient space.
        2. Zero modes with index > 2N/3 on both inputs.
        3. Inverse-transform, multiply pointwise.
        4. Forward-transform the product, zero modes with index > 2N/3,
           and inverse-transform once more.

    This is the Chebyshev analogue of Orszag's 2/3 rule for Fourier grids.
    It is exact for products whose highest relevant Chebyshev mode is
    below the cut-off, and otherwise prevents aliasing into the retained
    modes at the cost of some high-mode truncation error.

    Parameters
    ----------
    grid : ChebyshevGrid1D or ChebyshevGrid2D
        Grid providing the transform; must have ``dealias='2/3'`` to have
        any effect (otherwise this function falls back to the naive
        product ``a * b``).
    a, b : Num[Array, ...]
        Nodal fields on the same grid.  Shapes ``(Npts,)`` for 1D or
        ``(Nypts, Nxpts)`` for 2D.

    Returns
    -------
    Num[Array, ...]
        Dealiased pointwise product on the same grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> grid = ChebyshevGrid1D.from_N_L(N=32, L=1.0, dealias="2/3")
    >>> x = grid.x
    >>> u = jnp.sin(jnp.pi * x)
    >>> v = jnp.cos(jnp.pi * x)
    >>> uv = dealias_product(grid, u, v)  # ≈ ½ sin(2πx)
    """
    if grid.dealias != "2/3":
        # No dealiasing requested → return the naive product.
        return a * b

    if isinstance(grid, ChebyshevGrid1D):
        n_modes = grid.N + 1 if grid.node_type == "gauss-lobatto" else grid.N
        cutoff = int(2 * grid.N / 3)

        a_hat = grid.transform(a)
        b_hat = grid.transform(b)
        a_hat = _apply_dealias_1d(a_hat, n_modes, cutoff)
        b_hat = _apply_dealias_1d(b_hat, n_modes, cutoff)
        a_f = grid.transform(a_hat, inverse=True)
        b_f = grid.transform(b_hat, inverse=True)

        ab = a_f * b_f
        ab_hat = _apply_dealias_1d(grid.transform(ab), n_modes, cutoff)
        return grid.transform(ab_hat, inverse=True)

    if isinstance(grid, ChebyshevGrid2D):
        nx_modes = grid.Nx + 1 if grid.node_type == "gauss-lobatto" else grid.Nx
        ny_modes = grid.Ny + 1 if grid.node_type == "gauss-lobatto" else grid.Ny
        cutoff_x = int(2 * grid.Nx / 3)
        cutoff_y = int(2 * grid.Ny / 3)
        mask_x = (jnp.arange(nx_modes) <= cutoff_x).astype(a.dtype)
        mask_y = (jnp.arange(ny_modes) <= cutoff_y).astype(a.dtype)
        mask = mask_y[:, None] * mask_x[None, :]

        a_hat = grid.transform(a) * mask
        b_hat = grid.transform(b) * mask
        a_f = grid.transform(a_hat, inverse=True)
        b_f = grid.transform(b_hat, inverse=True)

        ab_hat = grid.transform(a_f * b_f) * mask
        return grid.transform(ab_hat, inverse=True)

    raise TypeError(
        f"grid must be a ChebyshevGrid1D or ChebyshevGrid2D, got {type(grid).__name__}."
    )
