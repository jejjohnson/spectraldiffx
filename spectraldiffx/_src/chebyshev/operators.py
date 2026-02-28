# ============================================================================
# Chebyshev Derivative Operators
# ============================================================================

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .grid import ChebyshevGrid1D, ChebyshevGrid2D


class ChebyshevDerivative1D(eqx.Module):
    """
    1D Chebyshev derivative operator using the precomputed differentiation matrix.

    Mathematical Formulation:
    -------------------------
    For a function u(x) sampled at Chebyshev nodes xⱼ on [-L, L]:

        du/dx |ⱼ = Σₖ D_{jk} u_k

    where D is the (N+1)×(N+1) [or N×N for Gauss] differentiation matrix
    precomputed in ChebyshevGrid1D.

    Higher-order derivatives are obtained by matrix powers:
        d²u/dx² = D @ (D @ u) = D² @ u

    Attributes:
    -----------
        grid : ChebyshevGrid1D
            The 1D Chebyshev grid containing the differentiation matrix.
    """

    grid: ChebyshevGrid1D

    def __call__(self, u: Array, order: int = 1) -> Float[Array, "N1"]:
        """
        Compute the n-th derivative of a field using the differentiation matrix.

        Operation:
            du/dx    = D @ u
            d²u/dx²  = D @ D @ u

        Parameters:
        -----------
        u : Array [N+1] for GL, [N] for Gauss
            Physical-space field values at Chebyshev nodes.
        order : int
            Derivative order (1 = first, 2 = second, ...). Default is 1.

        Returns:
        --------
        dnu_dxn : Array [N+1] or [N]
            n-th derivative at Chebyshev nodes.
        """
        D = self.grid.D
        result = u
        for _ in range(order):
            result = D @ result
        return result

    def gradient(self, u: Array) -> Float[Array, "N1"]:
        """
        Compute the first derivative du/dx.

        Parameters:
        -----------
        u : Array [N1]
            Physical-space field at Chebyshev nodes.

        Returns:
        --------
        du_dx : Array [N1]
        """
        return self(u, order=1)

    def laplacian(self, u: Array) -> Float[Array, "N1"]:
        """
        Compute the second derivative d²u/dx².

        Parameters:
        -----------
        u : Array [N1]
            Physical-space field at Chebyshev nodes.

        Returns:
        --------
        d2u_dx2 : Array [N1]
        """
        return self(u, order=2)


class ChebyshevDerivative2D(eqx.Module):
    """
    2D Chebyshev derivative operators on [-Lx, Lx] × [-Ly, Ly].

    Mathematical Formulation:
    -------------------------
    For u(x, y) on a (Ny_pts, Nx_pts) grid:

        ∂u/∂x [j,i] = (u @ Dxᵀ)[j,i]   (differentiation along axis 1)
        ∂u/∂y [j,i] = (Dy @ u)[j,i]     (differentiation along axis 0)

    Laplacian: ∇²u = ∂²u/∂x² + ∂²u/∂y²

    Attributes:
    -----------
        grid : ChebyshevGrid2D
            The 2D Chebyshev grid containing Dx and Dy matrices.
    """

    grid: ChebyshevGrid2D

    def gradient(
        self, u: Array
    ) -> tuple[Float[Array, "Ny1 Nx1"], Float[Array, "Ny1 Nx1"]]:
        """
        Compute partial derivatives (∂u/∂x, ∂u/∂y).

        Parameters:
        -----------
        u : Array [Ny_pts, Nx_pts]
            2D field at Chebyshev nodes.

        Returns:
        --------
        (du_dx, du_dy) : tuple of Arrays [Ny_pts, Nx_pts]
        """
        Dx = self.grid.Dx
        Dy = self.grid.Dy
        du_dx = u @ Dx.T  # apply Dx along x-axis (axis 1)
        du_dy = Dy @ u    # apply Dy along y-axis (axis 0)
        return du_dx, du_dy

    def laplacian(self, u: Array) -> Float[Array, "Ny1 Nx1"]:
        """
        Compute the 2D Laplacian ∇²u = ∂²u/∂x² + ∂²u/∂y².

        Parameters:
        -----------
        u : Array [Ny_pts, Nx_pts]

        Returns:
        --------
        lap_u : Array [Ny_pts, Nx_pts]
        """
        Dx = self.grid.Dx
        Dy = self.grid.Dy
        d2u_dx2 = u @ (Dx @ Dx).T
        d2u_dy2 = (Dy @ Dy) @ u
        return d2u_dx2 + d2u_dy2

    def divergence(self, vx: Array, vy: Array) -> Float[Array, "Ny1 Nx1"]:
        """
        Compute the divergence ∇·V = ∂vx/∂x + ∂vy/∂y.

        Parameters:
        -----------
        vx : Array [Ny_pts, Nx_pts]
            x-component of the vector field.
        vy : Array [Ny_pts, Nx_pts]
            y-component of the vector field.

        Returns:
        --------
        div : Array [Ny_pts, Nx_pts]
        """
        Dx = self.grid.Dx
        Dy = self.grid.Dy
        dvx_dx = vx @ Dx.T
        dvy_dy = Dy @ vy
        return dvx_dx + dvy_dy

    def curl(self, vx: Array, vy: Array) -> Float[Array, "Ny1 Nx1"]:
        """
        Compute the 2D scalar curl ζ = ∂vy/∂x - ∂vx/∂y.

        Parameters:
        -----------
        vx : Array [Ny_pts, Nx_pts]
        vy : Array [Ny_pts, Nx_pts]

        Returns:
        --------
        curl : Array [Ny_pts, Nx_pts]
        """
        Dx = self.grid.Dx
        Dy = self.grid.Dy
        dvy_dx = vy @ Dx.T
        dvx_dy = Dy @ vx
        return dvy_dx - dvx_dy

    def advection_scalar(
        self, vx: Array, vy: Array, q: Array
    ) -> Float[Array, "Ny1 Nx1"]:
        """
        Compute scalar advection (V·∇)q = vx·∂q/∂x + vy·∂q/∂y.

        Parameters:
        -----------
        vx : Array [Ny_pts, Nx_pts]
            x-velocity.
        vy : Array [Ny_pts, Nx_pts]
            y-velocity.
        q : Array [Ny_pts, Nx_pts]
            Scalar tracer field.

        Returns:
        --------
        adv : Array [Ny_pts, Nx_pts]
        """
        dq_dx, dq_dy = self.gradient(q)
        return vx * dq_dx + vy * dq_dy
