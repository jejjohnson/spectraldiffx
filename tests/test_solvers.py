import jax.numpy as jnp
import pytest

from spectraldiffx._src.grid import FourierGrid1D, FourierGrid2D, FourierGrid3D
from spectraldiffx._src.solvers import (
    SpectralHelmholtzSolver1D,
    SpectralHelmholtzSolver2D,
    SpectralHelmholtzSolver3D,
)

# --- Fixtures ---


@pytest.fixture
def grid1d():
    return FourierGrid1D.from_N_L(64, 2 * jnp.pi, dealias=None)


@pytest.fixture
def grid2d():
    return FourierGrid2D.from_N_L(32, 32, 2 * jnp.pi, 2 * jnp.pi, dealias=None)


@pytest.fixture
def grid3d():
    return FourierGrid3D.from_N_L(
        16, 16, 16, 2 * jnp.pi, 2 * jnp.pi, 2 * jnp.pi, dealias=None
    )


# --- 1D Solver Tests ---


def test_solver1d_poisson(grid1d):
    """Test 1D Poisson: d^2phi/dx^2 = f. For phi = sin(x), f = -sin(x)."""
    solver = SpectralHelmholtzSolver1D(grid1d)
    phi_exact = jnp.sin(grid1d.x)
    f = -jnp.sin(grid1d.x)
    phi_sol = solver.solve(f, alpha=0.0)
    assert jnp.allclose(phi_sol, phi_exact, atol=1e-4)


def test_solver1d_helmholtz(grid1d):
    """Test 1D Helmholtz: (d^2/dx^2 - alpha)phi = f.
    For phi = sin(x), alpha=1: f = -sin(x) - 1*sin(x) = -2*sin(x).
    """
    solver = SpectralHelmholtzSolver1D(grid1d)
    phi_exact = jnp.sin(grid1d.x)
    alpha = 1.0
    f = -2.0 * jnp.sin(grid1d.x)
    phi_sol = solver.solve(f, alpha=alpha)
    assert jnp.allclose(phi_sol, phi_exact, atol=1e-4)


# --- 2D Solver Tests ---


def test_solver2d_poisson(grid2d):
    """Test 2D Poisson: ∇²phi = f. For phi = sin(x)sin(y), f = -2*sin(x)sin(y)."""
    solver = SpectralHelmholtzSolver2D(grid2d)
    X, Y = grid2d.X
    phi_exact = jnp.sin(X) * jnp.sin(Y)
    f = -2.0 * phi_exact
    phi_sol = solver.solve(f, alpha=0.0)
    assert jnp.allclose(phi_sol, phi_exact, atol=1e-4)


def test_solver2d_helmholtz(grid2d):
    """Test 2D Helmholtz: (∇² - alpha)phi = f.
    For phi = sin(x)sin(y), alpha=1: f = -2*phi - 1*phi = -3*phi.
    """
    solver = SpectralHelmholtzSolver2D(grid2d)
    X, Y = grid2d.X
    phi_exact = jnp.sin(X) * jnp.sin(Y)
    alpha = 1.0
    f = -3.0 * phi_exact
    phi_sol = solver.solve(f, alpha=alpha)
    assert jnp.allclose(phi_sol, phi_exact, atol=1e-4)


# --- 3D Solver Tests ---


def test_solver3d_poisson(grid3d):
    """Test 3D Poisson: ∇²phi = f. For phi = sin(x)sin(y)sin(z), f = -3*phi."""
    solver = SpectralHelmholtzSolver3D(grid3d)
    Z, Y, X = grid3d.X
    phi_exact = jnp.sin(X) * jnp.sin(Y) * jnp.sin(Z)
    f = -3.0 * phi_exact
    phi_sol = solver.solve(f, alpha=0.0)
    assert jnp.allclose(phi_sol, phi_exact, atol=1e-4)


def test_solver3d_helmholtz(grid3d):
    """Test 3D Helmholtz: (∇² - alpha)phi = f.
    For phi = sin(x)sin(y)sin(z), alpha=2.0: f = -3*phi - 2*phi = -5*phi.
    """
    solver = SpectralHelmholtzSolver3D(grid3d)
    Z, Y, X = grid3d.X
    phi_exact = jnp.sin(X) * jnp.sin(Y) * jnp.sin(Z)
    alpha = 2.0
    f = -5.0 * phi_exact
    phi_sol = solver.solve(f, alpha=alpha)
    assert jnp.allclose(phi_sol, phi_exact, atol=1e-4)
