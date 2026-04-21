# ============================================================================
# Chebyshev Elliptic Solvers (Helmholtz / Poisson, 1D and 2D)
# ============================================================================

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Num

from .grid import ChebyshevGrid1D, ChebyshevGrid2D

# Array-shape aliases:
#   "Npts"         — 1D Chebyshev grid size (N+1 for Gauss-Lobatto)
#   "Nypts Nxpts"  — 2D Chebyshev grid (Ny+1, Nx+1 for GL)

BCType = Literal["dirichlet", "neumann"]


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

    Method — Boundary-Row Replacement
    ---------------------------------
    On Gauss–Lobatto nodes the endpoints x[0]=+L and x[N]=−L are collocation
    points, so we discretise as

        A u = b,   A = D² − α·I,   b = f

    and then overwrite rows 0 and N with the boundary equations:

        Dirichlet : row 0 ← eᵀ₀,       b[0]  ← bc_right
                    row N ← eᵀ_N,      b[N]  ← bc_left
        Neumann   : row 0 ← D[0, :],   b[0]  ← bc_right
                    row N ← D[N, :],   b[N]  ← bc_left

    The resulting (N+1)×(N+1) linear system is solved with :func:`jnp.linalg.solve`.

    Gauss-node grids do not include the endpoints, so this boundary-row
    method is inapplicable; the constructor validates the grid and raises.

    Pure Neumann + Poisson (α = 0) is only solvable up to a constant
    (constant nullspace of the discretisation); the solver pins the gauge
    inside the linear system by replacing one interior equation with the
    point constraint ``u[N//2] = 0``, so the solve is well-posed.  Shift
    the returned field by any constant if a different gauge is needed.

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

    def solve(
        self,
        f: Num[Array, "Npts"],
        alpha: float = 0.0,
        bc_left: float = 0.0,
        bc_right: float = 0.0,
        bc_type: BCType = "dirichlet",
    ) -> Float[Array, "Npts"]:
        """Solve (d²/dx² − α) u = f on [−L, L] with Dirichlet or Neumann BCs.

        Parameters
        ----------
        f : Num[Array, "Npts"]
            Source term sampled at the N+1 Gauss–Lobatto nodes
            (ordered x[0]=+L, …, x[N]=−L).
        alpha : float
            Helmholtz parameter (≥ 0).  α=0 gives the Poisson equation.
        bc_left : float
            BC value at x = −L.  Dirichlet: u(−L); Neumann: u'(−L).
        bc_right : float
            BC value at x = +L.  Dirichlet: u(+L); Neumann: u'(+L).
        bc_type : {"dirichlet", "neumann"}
            Boundary-condition flavour.

        Returns
        -------
        Float[Array, "Npts"]
            Solution at the N+1 GL nodes.

        Raises
        ------
        ValueError
            If the grid uses Gauss nodes, the length of ``f`` is wrong,
            or ``alpha < 0``.
        """
        if self.grid.node_type != "gauss-lobatto":
            raise ValueError(
                "ChebyshevHelmholtzSolver1D requires 'gauss-lobatto' nodes — "
                "the boundary-row method evaluates u (or u') at the endpoints "
                "x[0]=+L and x[N]=−L, which Gauss nodes exclude. Got "
                f"node_type='{self.grid.node_type}'."
            )
        if f.shape[0] != self.grid.N + 1:
            raise ValueError(
                f"f must have length N+1={self.grid.N + 1} (Gauss–Lobatto), "
                f"got length {f.shape[0]}."
            )
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if bc_type not in ("dirichlet", "neumann"):
            raise ValueError(
                f"bc_type must be 'dirichlet' or 'neumann', got {bc_type!r}"
            )

        D = self.grid.D
        N = self.grid.N

        # A = D² − α·I  (interior operator; boundary rows replaced below)
        A = D @ D - alpha * jnp.eye(N + 1)
        b = f

        if bc_type == "dirichlet":
            # Row 0 → u(+L) = bc_right, row N → u(−L) = bc_left
            A = A.at[0, :].set(0.0).at[0, 0].set(1.0)
            A = A.at[N, :].set(0.0).at[N, N].set(1.0)
        else:  # neumann
            # Row 0 → u'(+L) = D[0,:]·u, row N → u'(−L) = D[N,:]·u
            A = A.at[0, :].set(D[0, :])
            A = A.at[N, :].set(D[N, :])
        b = b.at[0].set(bc_right)
        b = b.at[N].set(bc_left)

        if bc_type == "neumann" and alpha == 0.0:
            # Pure-Neumann Poisson is rank-deficient (constant nullspace:
            # D²·1 = 0 and D·1 = 0, so A·1 = 0).  Pin a gauge inside the
            # linear system by replacing one interior equation with
            # u[middle] = 0.  This removes the singularity before the solve,
            # making it robust across RHS / grid sizes.  The user can shift
            # the result by any constant afterwards if a different gauge is
            # needed.
            mid = N // 2
            gauge_row = jnp.zeros(N + 1).at[mid].set(1.0)
            A = A.at[mid, :].set(gauge_row)
            b = b.at[mid].set(0.0)

        return jnp.linalg.solve(A, b)


class ChebyshevPoissonSolver1D(eqx.Module):
    """1D Chebyshev Poisson solver: d²u/dx² = f on [−L, L].

    Convenience wrapper around :class:`ChebyshevHelmholtzSolver1D` with α = 0.

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

    def solve(
        self,
        f: Num[Array, "Npts"],
        bc_left: float = 0.0,
        bc_right: float = 0.0,
        bc_type: BCType = "dirichlet",
    ) -> Float[Array, "Npts"]:
        """Solve d²u/dx² = f with Dirichlet or Neumann BCs."""
        inner = ChebyshevHelmholtzSolver1D(grid=self.grid)
        return inner.solve(
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

    Method — Vectorised Boundary-Row Replacement
    --------------------------------------------
    Let Dx, Dy be the 1D Chebyshev differentiation matrices.  The 2D
    Helmholtz operator in the Kronecker form acting on vec(u) is

        A = Iᵧ ⊗ Dx²  +  Dy² ⊗ Iₓ  − α·I

    where ``vec(u) = u.reshape(-1)`` with row-major (C) ordering.  The
    boundary rows of A are overwritten with the identity so that the
    corresponding unknowns equal the supplied BC values.  The dense
    (Nₓ+1)(Nᵧ+1) system is factored once per call with
    :func:`jnp.linalg.solve`.

    Notes
    -----
    • The system size is (Nₓ+1)(Nᵧ+1); for Nₓ = Nᵧ ≈ 32 the dense solve
      is very fast on GPU but becomes costly (~O(N⁶)) beyond ≈ 48.
    • For pure-Neumann Poisson in 2D we do not provide a solver here; use
      a Fourier backend or ADD a tau-style compatibility correction.

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

    def solve(
        self,
        f: Num[Array, "Nypts Nxpts"],
        alpha: float = 0.0,
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

        Parameters
        ----------
        f : Num[Array, "Nypts Nxpts"]
            Source term at the 2D GL nodes.
        alpha : float
            Helmholtz parameter (≥ 0).
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
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")

        Nx, Ny = self.grid.Nx, self.grid.Ny
        Nxpts = Nx + 1
        Nypts = Ny + 1
        if f.shape != (Nypts, Nxpts):
            raise ValueError(
                f"f must have shape (Ny+1, Nx+1)=({Nypts}, {Nxpts}), got {f.shape}."
            )

        Dx2 = self.grid.Dx2
        Dy2 = self.grid.Dy2

        # Build the Kronecker operator
        #   A = I_y ⊗ Dx²  +  Dy² ⊗ I_x  − α·I
        Ix = jnp.eye(Nxpts)
        Iy = jnp.eye(Nypts)
        A = jnp.kron(Iy, Dx2) + jnp.kron(Dy2, Ix) - alpha * jnp.eye(Nxpts * Nypts)

        # Flatten f (row-major); for vec() of a (Ny, Nx) matrix, the row
        # index is the outer (slower) index → Kronecker order is (Iy ⊗ Dx).
        b = f.reshape(-1)

        # Build a boolean mask that marks boundary DOFs.
        boundary_mask = jnp.zeros((Nypts, Nxpts), dtype=bool)
        boundary_mask = boundary_mask.at[0, :].set(True)
        boundary_mask = boundary_mask.at[-1, :].set(True)
        boundary_mask = boundary_mask.at[:, 0].set(True)
        boundary_mask = boundary_mask.at[:, -1].set(True)
        bmask_flat = boundary_mask.reshape(-1)

        # Assemble the boundary-value array using the same orientation as u.
        bc = jnp.zeros((Nypts, Nxpts))
        bc = bc.at[0, :].set(jnp.broadcast_to(jnp.asarray(bc_top), (Nxpts,)))
        bc = bc.at[-1, :].set(jnp.broadcast_to(jnp.asarray(bc_bottom), (Nxpts,)))
        bc = bc.at[:, 0].set(jnp.broadcast_to(jnp.asarray(bc_right), (Nypts,)))
        bc = bc.at[:, -1].set(jnp.broadcast_to(jnp.asarray(bc_left), (Nypts,)))
        bc_flat = bc.reshape(-1)

        # Replace boundary rows of A with identity rows and boundary entries of b
        # with the prescribed BC values.
        idx = jnp.arange(Nxpts * Nypts)
        A_b = jnp.where(bmask_flat[:, None], 0.0, A)
        # Set A[i, i] = 1 for boundary i
        A_b = A_b.at[idx, idx].set(jnp.where(bmask_flat, 1.0, A_b[idx, idx]))
        b_b = jnp.where(bmask_flat, bc_flat, b)

        u_flat = jnp.linalg.solve(A_b, b_b)
        return u_flat.reshape(Nypts, Nxpts)


class ChebyshevPoissonSolver2D(eqx.Module):
    """2D Chebyshev Poisson solver: ∇²u = f with Dirichlet BCs.

    Convenience wrapper around :class:`ChebyshevHelmholtzSolver2D` with α = 0.

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

    def solve(
        self,
        f: Num[Array, "Nypts Nxpts"],
        bc_top: float | Num[Array, "Nxpts"] = 0.0,
        bc_bottom: float | Num[Array, "Nxpts"] = 0.0,
        bc_left: float | Num[Array, "Nypts"] = 0.0,
        bc_right: float | Num[Array, "Nypts"] = 0.0,
    ) -> Float[Array, "Nypts Nxpts"]:
        """Solve ∇²u = f with Dirichlet BCs on all four edges."""
        inner = ChebyshevHelmholtzSolver2D(grid=self.grid)
        return inner.solve(
            f,
            alpha=0.0,
            bc_top=bc_top,
            bc_bottom=bc_bottom,
            bc_left=bc_left,
            bc_right=bc_right,
        )
