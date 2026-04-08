from __future__ import annotations

# ============================================================================
# Chebyshev Elliptic Solver (Helmholtz / Poisson)
# ============================================================================
import equinox as eqx
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.typing import DTypeLike
from jaxtyping import Array

from .grid import ChebyshevGrid1D

_CACHE_UNAVAILABLE_MESSAGE = (
    "Cached factorization is unavailable. This solver requires 'gauss-lobatto' nodes."
)


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
        alpha : float
            Default Helmholtz parameter whose LU factorization is precomputed at
            construction time for repeated solves.
    """

    grid: ChebyshevGrid1D
    alpha: float = 0.0
    _base_operator: Array | None = eqx.field(default=None, repr=False)
    _interior_identity: Array | None = eqx.field(default=None, repr=False)
    _lu: Array | None = eqx.field(default=None, repr=False)
    _pivots: Array | None = eqx.field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.grid.node_type != "gauss-lobatto":
            return

        base_operator = self._build_base_operator()
        interior_identity = self._build_interior_identity(base_operator.dtype)
        lu, pivots = jsp_linalg.lu_factor(
            base_operator - self.alpha * interior_identity
        )

        object.__setattr__(self, "_base_operator", base_operator)
        object.__setattr__(self, "_interior_identity", interior_identity)
        object.__setattr__(self, "_lu", lu)
        object.__setattr__(self, "_pivots", pivots)

    def _build_base_operator(self) -> Array:
        D = self.grid.D
        N = self.grid.N

        base_operator = D @ D
        base_operator = base_operator.at[0, :].set(0.0).at[0, 0].set(1.0)
        base_operator = base_operator.at[N, :].set(0.0).at[N, N].set(1.0)
        return base_operator

    def _build_interior_identity(self, dtype: DTypeLike) -> Array:
        N = self.grid.N

        interior_identity = jnp.eye(N + 1, dtype=dtype)
        interior_identity = interior_identity.at[0, 0].set(0.0)
        interior_identity = interior_identity.at[N, N].set(0.0)
        return interior_identity

    def _factor_operator(self, alpha: float) -> tuple[Array, Array]:
        if self._base_operator is None or self._interior_identity is None:
            raise ValueError(_CACHE_UNAVAILABLE_MESSAGE)
        return jsp_linalg.lu_factor(
            self._base_operator - alpha * self._interior_identity
        )

    def solve(
        self,
        f: Array,
        alpha: float | None = None,
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
        alpha : float | None
            Helmholtz parameter. If omitted, uses the factorization precomputed
            from the solver's default ``alpha`` value.
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
        >>> solver = ChebyshevHelmholtzSolver1D(grid, alpha=0.0)
        >>> x = grid.x
        >>> f = -(jnp.pi**2) * jnp.sin(jnp.pi * x)
        >>> u = solver.solve(f, bc_left=0.0, bc_right=0.0)
        """
        if self.grid.node_type != "gauss-lobatto":
            raise ValueError(
                "ChebyshevHelmholtzSolver1D requires 'gauss-lobatto' nodes "
                "because the boundary-row replacement method uses the endpoints "
                f"x[0]=+L and x[N]=-L. Got node_type='{self.grid.node_type}'."
            )
        if f.shape[0] != self.grid.N + 1:
            raise ValueError(
                f"f must have length N+1={self.grid.N + 1} for Gauss-Lobatto nodes, "
                f"got length {f.shape[0]}."
            )

        N = self.grid.N
        b = f

        # Enforce Dirichlet BCs by replacing boundary rows
        # x[0] = +L (right endpoint), x[N] = -L (left endpoint)
        b = b.at[0].set(bc_right)
        b = b.at[N].set(bc_left)

        if alpha is None:
            if self._lu is None or self._pivots is None:
                raise ValueError(_CACHE_UNAVAILABLE_MESSAGE)
            return jsp_linalg.lu_solve((self._lu, self._pivots), b)

        lu, pivots = self._factor_operator(alpha)
        return jsp_linalg.lu_solve((lu, pivots), b)
