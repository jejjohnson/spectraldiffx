# ============================================================================
# Chebyshev Elliptic Solvers (Helmholtz / Poisson, 1D and 2D)
# ============================================================================

from __future__ import annotations

from typing import Literal

import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Num
import numpy as np

from .grid import ChebyshevGrid1D, ChebyshevGrid2D

# Array-shape aliases:
#   "Npts"         — 1D Chebyshev grid size (N+1 for Gauss-Lobatto)
#   "Nypts Nxpts"  — 2D Chebyshev grid (Ny+1, Nx+1 for GL)

BCType = Literal["dirichlet", "neumann"]


# ============================================================================
# Construction-time helpers (NumPy, run once outside JIT)
# ============================================================================


def _concrete_numpy(a: Array) -> np.ndarray | None:
    """Return ``a`` as a NumPy array, or ``None`` if it is a JAX tracer.

    Solvers precompute eigendecompositions with NumPy at construction time.
    When a solver is built *inside* a traced function from a traced grid,
    the matrices are unavailable and the solver falls back to a dense
    per-call solve.
    """
    try:
        return np.asarray(a)
    except (TypeError, jax.errors.TracerArrayConversionError):
        return None


class _Neumann1D(eqx.Module):
    """Neumann elimination data for the 1D solver.

    Boundary rows D_BB u_B + D_BI u_I = g give u_B = K_g g − K_I u_I with
    K_g = D_BB⁻¹ (2, 2) and K_I = D_BB⁻¹ D_BI (2, N−1).  ``null`` (bool,
    (N−1,)) marks the constant eigenvector of the eliminated operator.
    """

    factorization: gaussx.EigenFactorization
    Kg: Array
    KI: Array
    null: Array


def _maybe_check_alpha(alpha: float | Array) -> None:
    """Raise if ``alpha`` is concrete and negative (skip for tracers)."""
    try:
        value = float(alpha)
    except (TypeError, jax.errors.ConcretizationTypeError):
        return
    if value < 0:
        raise ValueError(f"alpha must be >= 0, got {value}")


# ============================================================================
# 1D solvers
# ============================================================================


class ChebyshevHelmholtzSolver1D(eqx.Module):
    """1D Chebyshev-collocation Helmholtz/Poisson solver with Dirichlet or Neumann BCs.

    Solves the boundary-value problem on [−L, L]:

        d²u/dx² − α·u = f(x),     x ∈ [−L, L]

    with boundary conditions selected via ``bc_type``:

        Dirichlet: u(+L) = bc_right,       u(−L) = bc_left
        Neumann:   u'(+L) = bc_right,     u'(−L) = bc_left

    For α = 0 this reduces to Poisson.

    Method — Boundary Elimination + Matrix Diagonalisation
    -------------------------------------------------------
    On Gauss–Lobatto nodes the endpoints x[0]=+L and x[N]=−L are collocation
    points.  Split the nodes into interior I = {1, …, N−1} and boundary
    B = {0, N}.  Collocating D²u − αu = f at the interior nodes gives

        (D²_II − α) u_I + D²_IB u_B = f_I                            (1)

    Dirichlet: u_B = g is given, so

        (D²_II − α) u_I = f_I − D²_IB g,       E := D²_II

    Neumann: the boundary rows D_BB u_B + D_BI u_I = g give
    u_B = D_BB⁻¹ (g − D_BI u_I); substituting into (1),

        (E − α) u_I = f_I − D²_IB D_BB⁻¹ g,    E := D²_II − D²_IB D_BB⁻¹ D_BI

    In both cases E depends only on the grid, so it is diagonalised once
    at construction (:class:`gaussx.EigenFactorization`), E = Q Λ Q⁻¹, and
    every solve is

        u_I = Q · diag(1 / (λ − α)) · Q⁻¹ · r                         O(N²)

    for *any* α — no per-call O(N³) factorisation, and α may be a traced
    JAX value (so the solve is ``jit``/``grad``-compatible in α).  E has
    real, non-positive eigenvalues and well-conditioned eigenvectors, so
    the diagonalisation is as accurate as a direct LU solve.

    Pure Neumann + Poisson (α = 0) is only solvable up to a constant: the
    Neumann E has an exact null vector (the constant).  The solver drops
    that eigen-component (i.e. projects out the incompatible part of f)
    and then fixes the gauge u[N//2] = 0.  Shift the returned field by any
    constant if a different gauge is needed.

    Gauss-node grids do not include the endpoints, so this method is
    inapplicable; ``solve`` validates the grid and raises.

    If the solver is constructed inside a traced function from a traced
    grid (so its matrices cannot be read at construction time), it falls
    back to an equivalent dense boundary-row solve on every call.

    Attributes
    ----------
    grid : ChebyshevGrid1D
        Must use ``'gauss-lobatto'`` nodes.

    Examples
    --------
    Solve u″ = −π² sin(πx) with u(±1) = 0 (analytic solution u = sin(πx)):

    >>> import jax.numpy as jnp
    >>> grid = ChebyshevGrid1D.from_N_L(N=32, L=1.0)
    >>> solver = ChebyshevHelmholtzSolver1D(grid=grid)
    >>> x = grid.x
    >>> f = -(jnp.pi**2) * jnp.sin(jnp.pi * x)
    >>> u = solver.solve(f, alpha=0.0, bc_left=0.0, bc_right=0.0)

    Neumann example — solve u″ = cos(πx) with u'(±1) = 0:

    >>> f = jnp.cos(jnp.pi * grid.x)
    >>> u = solver.solve(f, alpha=0.0, bc_type="neumann")
    """

    grid: ChebyshevGrid1D
    # Boundary columns of D², D²[I, B] with B = {0, N}; shape (N−1, 2)
    _D2_IB: Array | None
    # Dirichlet interior operator E = D²_II
    _dirichlet: gaussx.EigenFactorization | None
    # Neumann-eliminated operator and boundary lifting
    _neumann: _Neumann1D | None

    def __init__(self, grid: ChebyshevGrid1D):
        self.grid = grid
        D = _concrete_numpy(grid.D) if grid.node_type == "gauss-lobatto" else None
        if D is None or grid.N < 2:
            self._D2_IB = self._dirichlet = self._neumann = None
            return

        N = grid.N
        D2 = D @ D
        I = np.arange(1, N)
        B = np.array([0, N])
        self._D2_IB = jnp.asarray(D2[np.ix_(I, B)])
        self._dirichlet = gaussx.EigenFactorization.from_matrix(D2[np.ix_(I, I)])

        Kg = np.linalg.inv(D[np.ix_(B, B)])
        KI = Kg @ D[np.ix_(B, I)]
        fac = gaussx.EigenFactorization.from_matrix(
            D2[np.ix_(I, I)] - D2[np.ix_(I, B)] @ KI
        )
        null = jnp.zeros(N - 1, dtype=bool).at[jnp.argmin(jnp.abs(fac.eigenvalues))]
        self._neumann = _Neumann1D(
            fac, jnp.asarray(Kg), jnp.asarray(KI), null.set(True)
        )

    def solve(
        self,
        f: Num[Array, "Npts"],
        alpha: float | Float[Array, ""] = 0.0,
        bc_left: float | Float[Array, ""] = 0.0,
        bc_right: float | Float[Array, ""] = 0.0,
        bc_type: BCType = "dirichlet",
    ) -> Float[Array, "Npts"]:
        """Solve (d²/dx² − α) u = f on [−L, L] with Dirichlet or Neumann BCs.

        Parameters
        ----------
        f : Num[Array, "Npts"]
            Source term sampled at the N+1 Gauss–Lobatto nodes
            (ordered x[0]=+L, …, x[N]=−L).  The boundary entries f[0] and
            f[N] are ignored (they are replaced by the BCs).
        alpha : float or scalar Array
            Helmholtz parameter (≥ 0).  α=0 gives the Poisson equation.
            May be a traced value.
        bc_left : float or scalar Array
            BC value at x = −L.  Dirichlet: u(−L); Neumann: u'(−L).
        bc_right : float or scalar Array
            BC value at x = +L.  Dirichlet: u(+L); Neumann: u'(+L).
        bc_type : {"dirichlet", "neumann"}
            Boundary-condition flavour (static).

        Returns
        -------
        Float[Array, "Npts"]
            Solution at the N+1 GL nodes.

        Raises
        ------
        ValueError
            If the grid uses Gauss nodes, the length of ``f`` is wrong,
            or a concrete ``alpha`` is negative.
        """
        if self.grid.node_type != "gauss-lobatto":
            raise ValueError(
                "ChebyshevHelmholtzSolver1D requires 'gauss-lobatto' nodes — "
                "the boundary conditions evaluate u (or u') at the endpoints "
                "x[0]=+L and x[N]=−L, which Gauss nodes exclude. Got "
                f"node_type='{self.grid.node_type}'."
            )
        if f.shape[0] != self.grid.N + 1:
            raise ValueError(
                f"f must have length N+1={self.grid.N + 1} (Gauss–Lobatto), "
                f"got length {f.shape[0]}."
            )
        _maybe_check_alpha(alpha)
        if bc_type not in ("dirichlet", "neumann"):
            raise ValueError(
                f"bc_type must be 'dirichlet' or 'neumann', got {bc_type!r}"
            )
        D2_IB, dirichlet, neumann = self._D2_IB, self._dirichlet, self._neumann
        if D2_IB is None or dirichlet is None or neumann is None:
            return self._solve_dense(f, alpha, bc_left, bc_right, bc_type)

        N = self.grid.N
        g = jnp.stack([jnp.asarray(bc_right), jnp.asarray(bc_left)]).astype(
            D2_IB.dtype
        )  # (2,), ordered like B = {0, N}
        f_I = f[1:N]

        if bc_type == "dirichlet":
            r = f_I - D2_IB @ g
            u_I = dirichlet.solve_shifted(r, alpha)
            u_B = g
        else:
            r = f_I - D2_IB @ (neumann.Kg @ g)
            # α = 0: drop the constant null mode (compatibility projection)
            drop = neumann.null & (alpha == 0)
            u_I = neumann.factorization.solve_shifted(r, alpha, drop=drop)
            u_B = neumann.Kg @ g - neumann.KI @ u_I

        u = jnp.concatenate([u_B[:1], u_I, u_B[1:]])
        if bc_type == "neumann":
            # Gauge for the pure-Neumann Poisson case: u[N//2] = 0.
            u = jnp.where(alpha == 0, u - u[N // 2], u)
        return u

    def _solve_dense(
        self,
        f: Array,
        alpha: float | Array,
        bc_left: float | Array,
        bc_right: float | Array,
        bc_type: str,
    ) -> Array:
        """Dense boundary-row solve, used when the grid is traced.

        A = D² − α·I with rows 0 and N replaced by the boundary equations
        (identity rows for Dirichlet, rows of D for Neumann); O(N³) per call.
        """
        D = self.grid.D
        N = self.grid.N
        A = D @ D - alpha * jnp.eye(N + 1)
        if bc_type == "dirichlet":
            A = A.at[0, :].set(0.0).at[0, 0].set(1.0)
            A = A.at[N, :].set(0.0).at[N, N].set(1.0)
        else:
            A = A.at[0, :].set(D[0, :]).at[N, :].set(D[N, :])
        b = f.at[0].set(bc_right).at[N].set(bc_left)
        if bc_type == "neumann":
            # Pure-Neumann Poisson: replace one interior equation by u[mid] = 0.
            mid = N // 2
            pinned = A.at[mid, :].set(jnp.zeros(N + 1).at[mid].set(1.0))
            A = jnp.where(alpha == 0, pinned, A)
            b = jnp.where(alpha == 0, b.at[mid].set(0.0), b)
        return jnp.linalg.solve(A, b)


class ChebyshevPoissonSolver1D(eqx.Module):
    """1D Chebyshev Poisson solver: d²u/dx² = f on [−L, L].

    Convenience wrapper around :class:`ChebyshevHelmholtzSolver1D` with α = 0.
    The wrapped solver (and its precomputed eigendecomposition) is built
    once at construction.

    Attributes
    ----------
    grid : ChebyshevGrid1D
        Must use ``'gauss-lobatto'`` nodes.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> grid = ChebyshevGrid1D.from_N_L(N=32, L=1.0)
    >>> solver = ChebyshevPoissonSolver1D(grid=grid)
    >>> f = -(jnp.pi**2) * jnp.sin(jnp.pi * grid.x)
    >>> u = solver.solve(f)  # ≈ sin(πx)
    """

    grid: ChebyshevGrid1D
    _helmholtz: ChebyshevHelmholtzSolver1D

    def __init__(self, grid: ChebyshevGrid1D):
        self.grid = grid
        self._helmholtz = ChebyshevHelmholtzSolver1D(grid)

    def solve(
        self,
        f: Num[Array, "Npts"],
        bc_left: float | Float[Array, ""] = 0.0,
        bc_right: float | Float[Array, ""] = 0.0,
        bc_type: BCType = "dirichlet",
    ) -> Float[Array, "Npts"]:
        """Solve d²u/dx² = f with Dirichlet or Neumann BCs."""
        return self._helmholtz.solve(
            f,
            alpha=0.0,
            bc_left=bc_left,
            bc_right=bc_right,
            bc_type=bc_type,
        )


# ============================================================================
# 2D solvers
# ============================================================================


class ChebyshevHelmholtzSolver2D(eqx.Module):
    """2D Chebyshev-collocation Helmholtz/Poisson solver with Dirichlet BCs.

    Solves on [−Lx, Lx] × [−Ly, Ly]:

        ∇²u − α·u = f(x, y)

    with Dirichlet data on all four edges.  The boundary data are provided
    as four 1D arrays (top, bottom, left, right), evaluated at the
    Gauss–Lobatto nodes along each edge.

    Method — Matrix Diagonalisation (Haidvogel & Zang 1979)
    -------------------------------------------------------
    With u[j, i] = u(xᵢ, yⱼ), the collocated operator is

        ∇²u = Dy² · u + u · Dx²ᵀ

    Split each direction into interior (I) and boundary (B) nodes.  The
    boundary values u_B are known, so the interior unknowns U = u[I, I]
    satisfy the Sylvester equation

        Ay U + U Axᵀ − α U = R,
        Ay = Dy²[I, I],   Ax = Dx²[I, I],
        R  = f[I, I] − Dy²[I, B] u[B, I] − u[I, B] Dx²[I, B]ᵀ

    (the corner values never enter).  Diagonalising the 1D operators once
    at construction, Ay = Qy Λy Qy⁻¹ and Ax = Qx Λx Qx⁻¹, the solve is

        Û = Qy⁻¹ R Qx⁻ᵀ
        Û[j, i] ← Û[j, i] / (λy_j + λx_i − α)
        U = Qy Û Qxᵀ

    (:func:`gaussx.kronecker_sum_solve`), i.e. four small matrix–matrix
    products: O(Nx·Ny·(Nx + Ny)) per call
    for any α (vs O((Nx·Ny)³) for the dense Kronecker system), and α may
    be traced.  If the solver is constructed inside a traced function from
    a traced grid, it falls back to the dense Kronecker solve.

    Notes
    -----
    • For pure-Neumann Poisson in 2D we do not provide a solver here; use
      a Fourier backend.

    Attributes
    ----------
    grid : ChebyshevGrid2D
        Must use ``'gauss-lobatto'`` nodes in both directions.

    Examples
    --------
    Solve ∇²u = −2π² sin(πx) sin(πy) with homogeneous Dirichlet BCs:

    >>> import jax.numpy as jnp
    >>> grid = ChebyshevGrid2D.from_N_L(Nx=16, Ny=16, Lx=1.0, Ly=1.0)
    >>> solver = ChebyshevHelmholtzSolver2D(grid=grid)
    >>> X, Y = grid.X
    >>> f = -2 * jnp.pi**2 * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    >>> u = solver.solve(f, alpha=0.0)
    """

    grid: ChebyshevGrid2D
    # Interior diagonalisations Ax = Dx²[I, I] (Nx−1), Ay = Dy²[I, I] (Ny−1)
    _diag_x: gaussx.EigenFactorization | None
    _diag_y: gaussx.EigenFactorization | None

    def __init__(self, grid: ChebyshevGrid2D):
        self.grid = grid
        Dx2 = Dy2 = None
        if grid.node_type == "gauss-lobatto" and min(grid.Nx, grid.Ny) >= 2:
            Dx2 = _concrete_numpy(grid.Dx2)
            Dy2 = _concrete_numpy(grid.Dy2)
        if Dx2 is None or Dy2 is None:
            self._diag_x = self._diag_y = None
            return
        self._diag_x = gaussx.EigenFactorization.from_matrix(Dx2[1:-1, 1:-1])
        self._diag_y = gaussx.EigenFactorization.from_matrix(Dy2[1:-1, 1:-1])

    def solve(
        self,
        f: Num[Array, "Nypts Nxpts"],
        alpha: float | Float[Array, ""] = 0.0,
        bc_top: float | Num[Array, "Nxpts"] = 0.0,
        bc_bottom: float | Num[Array, "Nxpts"] = 0.0,
        bc_left: float | Num[Array, "Nypts"] = 0.0,
        bc_right: float | Num[Array, "Nypts"] = 0.0,
    ) -> Float[Array, "Nypts Nxpts"]:
        """Solve (∇² − α) u = f with Dirichlet BCs on all four edges.

        Boundary indexing (Gauss–Lobatto orientation):

            top    row is grid.y[0]     (y = +Ly)   at axis 0, index 0
            bottom row is grid.y[-1]    (y = −Ly)   at axis 0, index Nᵧ
            right  col is grid.x[0]     (x = +Lx)   at axis 1, index 0
            left   col is grid.x[-1]    (x = −Lx)   at axis 1, index Nₓ

        At the four corners the left/right values take precedence.

        Parameters
        ----------
        f : Num[Array, "Nypts Nxpts"]
            Source term at the 2D GL nodes (boundary entries are ignored).
        alpha : float or scalar Array
            Helmholtz parameter (≥ 0).  May be a traced value.
        bc_top, bc_bottom : float or Num[Array, "Nxpts"]
            Dirichlet values along the top and bottom edges.  Scalars broadcast.
        bc_left, bc_right : float or Num[Array, "Nypts"]
            Dirichlet values along the left and right edges.

        Returns
        -------
        Float[Array, "Nypts Nxpts"]
            Solution on the (Nᵧ+1, Nₓ+1) GL grid.
        """
        if self.grid.node_type != "gauss-lobatto":
            raise ValueError(
                "ChebyshevHelmholtzSolver2D requires 'gauss-lobatto' nodes."
            )
        _maybe_check_alpha(alpha)

        Nxpts = self.grid.Nx + 1
        Nypts = self.grid.Ny + 1
        if f.shape != (Nypts, Nxpts):
            raise ValueError(
                f"f must have shape (Ny+1, Nx+1)=({Nypts}, {Nxpts}), got {f.shape}."
            )

        # Boundary-value array in the same orientation as u.
        dtype = self.grid.Dx2.dtype
        bc = jnp.zeros((Nypts, Nxpts), dtype=dtype)
        bc = bc.at[0, :].set(jnp.broadcast_to(jnp.asarray(bc_top), (Nxpts,)))
        bc = bc.at[-1, :].set(jnp.broadcast_to(jnp.asarray(bc_bottom), (Nxpts,)))
        bc = bc.at[:, 0].set(jnp.broadcast_to(jnp.asarray(bc_right), (Nypts,)))
        bc = bc.at[:, -1].set(jnp.broadcast_to(jnp.asarray(bc_left), (Nypts,)))

        dx, dy = self._diag_x, self._diag_y
        if dx is None or dy is None:
            return self._solve_dense(f, alpha, bc)

        Dx2, Dy2 = self.grid.Dx2, self.grid.Dy2
        # bc is zero in the interior, so Dy²[I, :] bc[:, I] = Dy²[I, B] u[B, I]
        # and bc[I, :] Dx²[I, :]ᵀ = u[I, B] Dx²[I, B]ᵀ — the boundary lifting.
        lift = Dy2[1:-1, :] @ bc[:, 1:-1] + bc[1:-1, :] @ Dx2[1:-1, :].T
        R = f[1:-1, 1:-1] - lift  # (Ny−1, Nx−1)

        # Ay U + U Axᵀ − α U = R: a shifted Kronecker sum, Ay on axis 0 (y)
        # and Ax on axis 1 (x), solved by rotation into the eigenbases.
        U = gaussx.kronecker_sum_solve((dy, dx), R, alpha)
        return bc.at[1:-1, 1:-1].set(U)

    def _solve_dense(self, f: Array, alpha: float | Array, bc: Array) -> Array:
        """Dense Kronecker solve, used when the grid is traced.

        A = I_y ⊗ Dx² + Dy² ⊗ I_x − α·I with boundary rows replaced by
        identity rows; O((Nx·Ny)³) per call.
        """
        Nypts, Nxpts = bc.shape
        Dx2, Dy2 = self.grid.Dx2, self.grid.Dy2
        A = (
            jnp.kron(jnp.eye(Nypts), Dx2)
            + jnp.kron(Dy2, jnp.eye(Nxpts))
            - alpha * jnp.eye(Nxpts * Nypts)
        )
        mask = jnp.ones((Nypts, Nxpts), dtype=bool).at[1:-1, 1:-1].set(False)
        m = mask.reshape(-1)
        idx = jnp.arange(Nxpts * Nypts)
        A = jnp.where(m[:, None], 0.0, A)
        A = A.at[idx, idx].set(jnp.where(m, 1.0, A[idx, idx]))
        b = jnp.where(m, bc.reshape(-1), f.reshape(-1))
        return jnp.linalg.solve(A, b).reshape(Nypts, Nxpts)


class ChebyshevPoissonSolver2D(eqx.Module):
    """2D Chebyshev Poisson solver: ∇²u = f with Dirichlet BCs.

    Convenience wrapper around :class:`ChebyshevHelmholtzSolver2D` with α = 0.
    The wrapped solver (and its precomputed eigendecompositions) is built
    once at construction.

    Attributes
    ----------
    grid : ChebyshevGrid2D
        Must use ``'gauss-lobatto'`` nodes in both directions.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> grid = ChebyshevGrid2D.from_N_L(Nx=16, Ny=16, Lx=1.0, Ly=1.0)
    >>> solver = ChebyshevPoissonSolver2D(grid=grid)
    >>> X, Y = grid.X
    >>> f = -2 * jnp.pi**2 * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    >>> u = solver.solve(f)  # ≈ sin(πx) sin(πy)
    """

    grid: ChebyshevGrid2D
    _helmholtz: ChebyshevHelmholtzSolver2D

    def __init__(self, grid: ChebyshevGrid2D):
        self.grid = grid
        self._helmholtz = ChebyshevHelmholtzSolver2D(grid)

    def solve(
        self,
        f: Num[Array, "Nypts Nxpts"],
        bc_top: float | Num[Array, "Nxpts"] = 0.0,
        bc_bottom: float | Num[Array, "Nxpts"] = 0.0,
        bc_left: float | Num[Array, "Nypts"] = 0.0,
        bc_right: float | Num[Array, "Nypts"] = 0.0,
    ) -> Float[Array, "Nypts Nxpts"]:
        """Solve ∇²u = f with Dirichlet BCs on all four edges."""
        return self._helmholtz.solve(
            f,
            alpha=0.0,
            bc_top=bc_top,
            bc_bottom=bc_bottom,
            bc_left=bc_left,
            bc_right=bc_right,
        )
