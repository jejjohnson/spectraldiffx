# ============================================================================
# Chebyshev Derivative Operators
# ============================================================================

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float, Num

from .grid import ChebyshevGrid1D, ChebyshevGrid2D

# Array-shape aliases used across this module:
#   "Npts"  — number of 1D Chebyshev nodes (N+1 for Gauss-Lobatto, N for Gauss)
#   "Nypts Nxpts" — 2D tensor-product grid (y-fast inner axis is x)


class ChebyshevDerivative1D(eqx.Module):
    """1D Chebyshev derivative operator using the precomputed differentiation matrix.

    Mathematical Formulation
    ------------------------
    For a function u(x) sampled at Chebyshev nodes xⱼ on [−L, L]:

        (du/dx)ⱼ = Σₖ D_{jk} uₖ

    where D is the (N+1)×(N+1) (Gauss–Lobatto) or N×N (Gauss) differentiation
    matrix precomputed in :class:`ChebyshevGrid1D`.  Higher-order derivatives
    are matrix powers:

        d²u/dx² = D · D · u = D² · u

    Attributes
    ----------
    grid : ChebyshevGrid1D
        1D Chebyshev grid carrying the differentiation matrix D.

    Examples
    --------
    Derivative of sin(πx) on [−1, 1]:

    >>> import jax.numpy as jnp
    >>> grid = ChebyshevGrid1D.from_N_L(N=32, L=1.0)
    >>> deriv = ChebyshevDerivative1D(grid=grid)
    >>> u = jnp.sin(jnp.pi * grid.x)
    >>> du_dx = deriv(u)  # ≈ π cos(πx)
    >>> d2u_dx2 = deriv(u, order=2)  # ≈ −π² sin(πx)
    """

    grid: ChebyshevGrid1D

    def __call__(self, u: Num[Array, Npts], order: int = 1) -> Float[Array, Npts]:
        """Apply the n-th derivative Dⁿ to a nodal field.

        Parameters
        ----------
        u : Float[Array, "Npts"]
            Nodal values at Chebyshev nodes (Npts = N+1 for GL, N for Gauss).
        order : int
            Derivative order (≥ 0).  ``order=0`` returns a copy of ``u``.

        Returns
        -------
        Float[Array, "Npts"]
            n-th derivative at the Chebyshev nodes.
        """
        if order < 0:
            raise ValueError(f"order must be >= 0, got {order}")
        D = self.grid.D
        result = u
        for _ in range(order):
            result = D @ result
        return result

    def gradient(self, u: Num[Array, Npts]) -> Float[Array, Npts]:
        """First derivative ``du/dx`` at Chebyshev nodes."""
        return self(u, order=1)

    def laplacian(self, u: Num[Array, Npts]) -> Float[Array, Npts]:
        """Second derivative ``d²u/dx²`` at Chebyshev nodes."""
        return self(u, order=2)


class ChebyshevDerivative2D(eqx.Module):
    """2D Chebyshev derivative operators on [−Lx, Lx] × [−Ly, Ly].

    Mathematical Formulation
    ------------------------
    For u(x, y) on a (Nypts, Nxpts) grid with differentiation matrices Dx, Dy
    stored on the grid:

        (∂u/∂x)[j, i] = (u · Dxᵀ)[j, i]   # applied along axis 1 (x)
        (∂u/∂y)[j, i] = (Dy · u)[j, i]    # applied along axis 0 (y)

    The scalar Laplacian and 2D divergence/curl follow directly:

        ∇²u   = ∂²u/∂x² + ∂²u/∂y²
        ∇·V  = ∂vₓ/∂x + ∂vᵧ/∂y
        (∇×V)_z = ∂vᵧ/∂x − ∂vₓ/∂y

    Attributes
    ----------
    grid : ChebyshevGrid2D
        2D Chebyshev grid carrying Dx, Dy and the precomputed Dx², Dy².

    Examples
    --------
    Laplacian of u(x, y) = sin(πx)·sin(πy) on [−1, 1]²:

    >>> import jax.numpy as jnp
    >>> grid = ChebyshevGrid2D.from_N_L(Nx=24, Ny=24, Lx=1.0, Ly=1.0)
    >>> deriv = ChebyshevDerivative2D(grid=grid)
    >>> X, Y = grid.X
    >>> u = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    >>> lap_u = deriv.laplacian(u)  # ≈ −2π² u

    Divergence of V = (y, −x) (should be ~0):

    >>> vx, vy = Y, -X
    >>> div = deriv.divergence(vx, vy)  # ≈ 0
    """

    grid: ChebyshevGrid2D

    def gradient(
        self, u: Num[Array, "Nypts Nxpts"]
    ) -> tuple[Float[Array, "Nypts Nxpts"], Float[Array, "Nypts Nxpts"]]:
        """Partial derivatives (∂u/∂x, ∂u/∂y) of a 2D nodal field."""
        Dx = self.grid.Dx
        Dy = self.grid.Dy
        du_dx = u @ Dx.T  # apply Dx along x-axis (axis 1)
        du_dy = Dy @ u  # apply Dy along y-axis (axis 0)
        return du_dx, du_dy

    def laplacian(self, u: Num[Array, "Nypts Nxpts"]) -> Float[Array, "Nypts Nxpts"]:
        """2D Laplacian ∇²u = ∂²u/∂x² + ∂²u/∂y².

        Uses precomputed Dx² and Dy² from the grid so the per-call cost is
        two matrix–matrix multiplies and an add (no O(N³) recomputation).
        """
        d2u_dx2 = u @ self.grid.Dx2.T
        d2u_dy2 = self.grid.Dy2 @ u
        return d2u_dx2 + d2u_dy2

    def divergence(
        self,
        vx: Num[Array, "Nypts Nxpts"],
        vy: Num[Array, "Nypts Nxpts"],
    ) -> Float[Array, "Nypts Nxpts"]:
        """Cartesian divergence ∇·V = ∂vₓ/∂x + ∂vᵧ/∂y."""
        Dx = self.grid.Dx
        Dy = self.grid.Dy
        dvx_dx = vx @ Dx.T
        dvy_dy = Dy @ vy
        return dvx_dx + dvy_dy

    def curl(
        self,
        vx: Num[Array, "Nypts Nxpts"],
        vy: Num[Array, "Nypts Nxpts"],
    ) -> Float[Array, "Nypts Nxpts"]:
        """Scalar curl ζ = ∂vᵧ/∂x − ∂vₓ/∂y (z-component of ∇×V)."""
        Dx = self.grid.Dx
        Dy = self.grid.Dy
        dvy_dx = vy @ Dx.T
        dvx_dy = Dy @ vx
        return dvy_dx - dvx_dy

    def advection_scalar(
        self,
        vx: Num[Array, "Nypts Nxpts"],
        vy: Num[Array, "Nypts Nxpts"],
        q: Num[Array, "Nypts Nxpts"],
    ) -> Float[Array, "Nypts Nxpts"]:
        """Scalar advection (V·∇)q = vₓ·∂q/∂x + vᵧ·∂q/∂y."""
        dq_dx, dq_dy = self.gradient(q)
        return vx * dq_dx + vy * dq_dy
