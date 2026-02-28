"""
Tests for ChebyshevHelmholtzSolver1D.
"""

import jax.numpy as jnp

from spectraldiffx._src.chebyshev.grid import ChebyshevGrid1D
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
