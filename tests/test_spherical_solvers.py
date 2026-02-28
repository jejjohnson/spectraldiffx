"""
Tests for SphericalPoissonSolver and SphericalHelmholtzSolver.
"""

import jax.numpy as jnp
import numpy as np

from spectraldiffx._src.spherical.grid import SphericalGrid1D, SphericalGrid2D
from spectraldiffx._src.spherical.operators import SphericalDerivative2D
from spectraldiffx._src.spherical.solvers import (
    SphericalHelmholtzSolver,
    SphericalPoissonSolver,
)


def test_spherical_poisson_eigenfunction():
    """
    nabla^2 psi = f where f = -l*(l+1) * Y_l^m.
    Poisson solution should give psi = Y_l^m (up to sign convention).

    Test with Y_1^0 = P_1(cos(theta)) = cos(theta).
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    solver = SphericalPoissonSolver(grid=g)
    d = SphericalDerivative2D(g)

    THETA = g.y[:, None] * jnp.ones((Ny, Nx))
    psi = jnp.cos(THETA)  # Y_1^0 (l=1)
    f = d.laplacian(psi)  # f = -2 * psi (l*(l+1) = 2)

    phi = solver.solve(f)
    # phi should recover psi (up to additive constant forced to zero)
    assert jnp.allclose(phi, psi, atol=1e-6)


def test_spherical_poisson_mean_zero():
    """The l=0 (global mean) mode of the Poisson solution should be zero."""
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    solver = SphericalPoissonSolver(grid=g)

    # Source that has a mean component
    PHI, THETA = g.X
    f = jnp.sin(THETA) * jnp.cos(PHI) + jnp.ones((Ny, Nx))
    phi = solver.solve(f, zero_mean=True)

    # Check mean is (approximately) zero
    weights = g.weights
    mean = float(jnp.sum(weights * phi) / jnp.sum(weights))
    assert abs(mean) < 1e-8


def test_spherical_helmholtz():
    """
    Verify (nabla^2 - alpha) phi = f round-trip.

    Use Y_1^0 as the solution, compute f = (nabla^2 - alpha) * phi, then
    solve for phi from f.
    """
    Nx, Ny = 32, 16
    alpha = 2.0
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    solver = SphericalHelmholtzSolver(grid=g)
    d = SphericalDerivative2D(g)

    THETA = g.y[:, None] * jnp.ones((Ny, Nx))
    phi_true = jnp.cos(THETA)  # Y_1^0 (l=1)
    lap_phi = d.laplacian(phi_true)  # = -2 * phi_true
    f = lap_phi - alpha * phi_true

    phi_sol = solver.solve(f, alpha=alpha, zero_mean=False)
    assert jnp.allclose(phi_sol, phi_true, atol=1e-6)


def test_spherical_poisson_1d():
    """1D Poisson solver with P_2 eigenfunction."""
    from scipy.special import eval_legendre

    N = 32
    g = SphericalGrid1D.from_N_L(N, np.pi)  # R=1
    solver = SphericalPoissonSolver(grid=g)

    mu = np.array(g.cos_theta)
    psi = jnp.asarray(eval_legendre(2, mu))  # P_2(cos(theta)) — l=2 eigenvalue -6
    f = -6.0 * psi  # nabla^2 P_2 = -2*(2+1) * P_2 = -6 * P_2

    phi = solver.solve(f, zero_mean=True)
    assert jnp.allclose(phi, psi, atol=1e-8)
