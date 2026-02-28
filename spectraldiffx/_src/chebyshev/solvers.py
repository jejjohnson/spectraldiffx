# ============================================================================
# Chebyshev Elliptic Solver (Helmholtz / Poisson)
# ============================================================================

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .grid import ChebyshevGrid1D


class ChebyshevHelmholtzSolver1D(eqx.Module):
    """
    1D Chebyshev-collocation Helmholtz/Poisson solver with Dirichlet BCs.

    Solves the boundary-value problem:

        d²u/dx² - alpha * u = f(x),   x ∈ [-L, L]
        u(-L) = bc_left,  u(L) = bc_right

    For alpha=0 this reduces to the Poisson equation.

    Method — Boundary-Row Replacement:
    ------------------------------------
    Discretise using the Chebyshev differentiation matrix D on Gauss-Lobatto
    nodes (which include the endpoints ±L):

        (D² - alpha * I) u = f

    Then replace the first and last rows with the Dirichlet conditions:

        Row 0  →  u[0]  = bc_right   (node x[0] = +L)
        Row N  →  u[N]  = bc_left    (node x[N] = -L)

    Solve the resulting (N+1)×(N+1) linear system with jnp.linalg.solve.

    Attributes:
    -----------
        grid : ChebyshevGrid1D
            Must use 'gauss-lobatto' nodes (endpoints required for Dirichlet BCs).
    """

    grid: ChebyshevGrid1D

    def solve(
        self,
        f: Array,
        alpha: float = 0.0,
        bc_left: float = 0.0,
        bc_right: float = 0.0,
    ) -> Array:
        """
        Solve (d²/dx² - alpha) u = f with Dirichlet boundary conditions.

        Mathematical Formulation:
        -------------------------
        System to solve (physical space, GL nodes x[0]=+L ... x[N]=-L):

            A u = b

        where A = D² - alpha * I  (modified at boundary rows), b = f
        (modified at boundary rows with BC values).

        Parameters:
        -----------
        f : Array [N+1]
            Right-hand side (source term) at all N+1 GL nodes.
        alpha : float
            Helmholtz parameter. Default 0.0 (Poisson equation).
        bc_left : float
            Dirichlet value at x = -L (node x[N]). Default 0.0.
        bc_right : float
            Dirichlet value at x = +L (node x[0]). Default 0.0.

        Returns:
        --------
        u : Array [N+1]
            Solution field at all N+1 GL nodes.

        Example:
        --------
        Solve u'' = -π² sin(πx), u(±1)=0 → solution u = sin(πx):

        >>> import jax.numpy as jnp
        >>> grid = ChebyshevGrid1D.from_N_L(N=32, L=1.0)
        >>> solver = ChebyshevHelmholtzSolver1D(grid)
        >>> x = grid.x
        >>> f = -(jnp.pi**2) * jnp.sin(jnp.pi * x)
        >>> u = solver.solve(f, alpha=0.0, bc_left=0.0, bc_right=0.0)
        """
        D = self.grid.D
        N = self.grid.N

        # Build operator: A = D @ D - alpha * I
        A = D @ D - alpha * jnp.eye(N + 1)

        # Right-hand side
        b = f

        # Enforce Dirichlet BCs by replacing boundary rows
        # x[0] = +L (right endpoint), x[N] = -L (left endpoint)
        A = A.at[0, :].set(0.0).at[0, 0].set(1.0)
        b = b.at[0].set(bc_right)

        A = A.at[N, :].set(0.0).at[N, N].set(1.0)
        b = b.at[N].set(bc_left)

        return jnp.linalg.solve(A, b)
