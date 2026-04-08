from __future__ import annotations

"""
Tests for ChebyshevHelmholtzSolver1D.
"""

import jax.numpy as jnp
import pytest

from spectraldiffx._src.chebyshev.grid import ChebyshevGrid1D
import spectraldiffx._src.chebyshev.solvers as cheb_solvers
from spectraldiffx._src.chebyshev.solvers import ChebyshevHelmholtzSolver1D

# ============================================================================
# 1D Solver tests
# ============================================================================


def test_cheb_poisson_1d_dirichlet():
    """
    Poisson: u'' = f  with u(-1)=u(1)=0.

    Exact solution: u(x) = sin(πx)
    Source term:    f(x) = -π² sin(πx)
    """
    N = 32
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    solver = ChebyshevHelmholtzSolver1D(grid)
    x = grid.x
    f = -(jnp.pi**2) * jnp.sin(jnp.pi * x)
    u_exact = jnp.sin(jnp.pi * x)

    u_sol = solver.solve(f, alpha=0.0, bc_left=0.0, bc_right=0.0)
    assert jnp.allclose(u_sol, u_exact, atol=1e-8), (
        f"Poisson max error = {jnp.abs(u_sol - u_exact).max()}"
    )


def test_cheb_poisson_1d_nonzero_bcs():
    """
    Poisson: u'' = 0  with u(-1)=0, u(1)=1.

    Exact solution: u(x) = (x + 1) / 2  (linear function)
    """
    N = 16
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    solver = ChebyshevHelmholtzSolver1D(grid)
    x = grid.x
    f = jnp.zeros(N + 1)
    u_exact = (x + 1.0) / 2.0
    # x[0]=+1 → u=1.0, x[N]=-1 → u=0.0
    u_sol = solver.solve(f, alpha=0.0, bc_left=0.0, bc_right=1.0)
    assert jnp.allclose(u_sol, u_exact, atol=1e-8), (
        f"Poisson (nonzero BCs) max error = {jnp.abs(u_sol - u_exact).max()}"
    )


def test_cheb_helmholtz_1d():
    """
    Helmholtz: (u'' - alpha * u) = f  with u(-1)=u(1)=0.

    Exact solution: u(x) = sin(πx)
    Source term:    f(x) = -(π² + alpha) sin(πx)
    """
    N = 32
    alpha = 2.0
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    solver = ChebyshevHelmholtzSolver1D(grid)
    x = grid.x
    f = -(jnp.pi**2 + alpha) * jnp.sin(jnp.pi * x)
    u_exact = jnp.sin(jnp.pi * x)

    u_sol = solver.solve(f, alpha=alpha, bc_left=0.0, bc_right=0.0)
    assert jnp.allclose(u_sol, u_exact, atol=1e-8), (
        f"Helmholtz max error = {jnp.abs(u_sol - u_exact).max()}"
    )


def test_cheb_helmholtz_1d_reuses_cached_factorization(monkeypatch):
    """
    Repeated solves with the constructor-configured alpha should reuse the
    prefactored operator instead of refactoring on every solve call.
    """
    N = 32
    alpha = 2.0
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    x = grid.x
    f = -(jnp.pi**2 + alpha) * jnp.sin(jnp.pi * x)
    u_exact = jnp.sin(jnp.pi * x)

    call_count = 0
    original_lu_factor = cheb_solvers.jsp_linalg.lu_factor

    def counting_lu_factor(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_lu_factor(*args, **kwargs)

    monkeypatch.setattr(cheb_solvers.jsp_linalg, "lu_factor", counting_lu_factor)

    solver = ChebyshevHelmholtzSolver1D(grid, alpha=alpha)
    u_sol_1 = solver.solve(f, bc_left=0.0, bc_right=0.0)
    u_sol_2 = solver.solve(f, bc_left=0.0, bc_right=0.0)

    assert call_count == 1
    assert jnp.allclose(u_sol_1, u_exact, atol=1e-8)
    assert jnp.allclose(u_sol_2, u_exact, atol=1e-8)


def test_cheb_helmholtz_1d_scaled_domain():
    """
    Helmholtz on [-L, L] with L=2: exact solution u(x) = sin(πx/L).
    """
    N = 32
    L = 2.0
    alpha = 1.0
    grid = ChebyshevGrid1D.from_N_L(N=N, L=L)
    solver = ChebyshevHelmholtzSolver1D(grid)
    x = grid.x
    # u = sin(πx/L), u'' = -(π/L)² sin(πx/L), (u'' - alpha u) = -(π²/L² + alpha) sin(πx/L)
    f = -(jnp.pi**2 / L**2 + alpha) * jnp.sin(jnp.pi * x / L)
    u_exact = jnp.sin(jnp.pi * x / L)

    u_sol = solver.solve(f, alpha=alpha, bc_left=0.0, bc_right=0.0)
    assert jnp.allclose(u_sol, u_exact, atol=1e-7), (
        f"Scaled Helmholtz max error = {jnp.abs(u_sol - u_exact).max()}"
    )


def test_cheb_poisson_1d_smooth_rhs():
    """
    Poisson with a smooth RHS: u = sin(2πx)(1-x²), chosen to satisfy u(±1)=0.
    """
    N = 48
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    solver = ChebyshevHelmholtzSolver1D(grid)
    x = grid.x

    # u = (1 - x²) sin(2πx), u(±1) = 0
    u_exact = (1.0 - x**2) * jnp.sin(2 * jnp.pi * x)
    # f = u'' (computed analytically)
    # u' = -2x sin(2πx) + 2π(1-x²) cos(2πx)
    # u'' = -2 sin(2πx) - 2x*2π cos(2πx) - 4πx cos(2πx) - (2πx)²(1-x²) sin(2πx) ... simplified:
    f = (
        -2.0 * jnp.sin(2 * jnp.pi * x)
        - 8 * jnp.pi * x * jnp.cos(2 * jnp.pi * x)
        - (2 * jnp.pi) ** 2 * (1.0 - x**2) * jnp.sin(2 * jnp.pi * x)
    )

    u_sol = solver.solve(f, alpha=0.0, bc_left=0.0, bc_right=0.0)
    assert jnp.allclose(u_sol, u_exact, atol=1e-8), (
        f"Smooth Poisson max error = {jnp.abs(u_sol - u_exact).max()}"
    )


def test_cheb_solver1d_raises_for_gauss_nodes():
    """
    ChebyshevHelmholtzSolver1D must raise ValueError when given Gauss (not
    Gauss-Lobatto) nodes, because the boundary-row replacement method requires
    the endpoints x[0]=+L and x[N]=-L.
    """
    grid_gauss = ChebyshevGrid1D.from_N_L(N=16, L=1.0, node_type="gauss")
    solver = ChebyshevHelmholtzSolver1D(grid_gauss)
    with pytest.raises(ValueError, match="gauss-lobatto"):
        solver.solve(jnp.zeros(16), alpha=0.0)


def test_cheb_solver1d_raises_for_wrong_f_shape():
    """
    ChebyshevHelmholtzSolver1D must raise ValueError when f has wrong length.

    For N GL nodes, f must have length N+1; any other length should fail.
    """
    N = 16
    grid_gl = ChebyshevGrid1D.from_N_L(N=N, L=1.0, node_type="gauss-lobatto")
    solver = ChebyshevHelmholtzSolver1D(grid_gl)
    with pytest.raises(ValueError, match=rf"N\+1={N + 1}"):
        solver.solve(jnp.zeros(N), alpha=0.0)  # N instead of N+1
