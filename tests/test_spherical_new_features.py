"""Tests for the new Spherical additions from the code-review follow-up.

Covers:
    * :meth:`SphericalDerivative2D.iterated_laplacian` and ``biharmonic``
    * Standard GFD sign of :meth:`SphericalDerivative2D.curl`
    * :class:`SphericalVorticityInversionSolver`
    * :class:`SphericalDivergenceInversionSolver`
    * :class:`SphericalHelmholtzDecomposition` (for sphere-smooth inputs)
"""

from __future__ import annotations

import jax.numpy as jnp

from spectraldiffx import (
    SphericalDerivative2D,
    SphericalDivergenceInversionSolver,
    SphericalGrid2D,
    SphericalHelmholtzDecomposition,
    SphericalVorticityInversionSolver,
)

# ---------------------------------------------------------------------------
# Iterated Laplacian / biharmonic
# ---------------------------------------------------------------------------


def test_biharmonic_cos_theta():
    """For u = cos θ (l = 1), ∇⁴ u = (2/R²)² u = 4/R⁴ · cos θ."""
    grid = SphericalGrid2D.from_N_L(Nx=64, Ny=32)
    deriv = SphericalDerivative2D(grid=grid)
    _, THETA = grid.X
    u = jnp.cos(THETA)
    bih = deriv.biharmonic(u)
    R = grid.Ly / jnp.pi
    expected = (4.0 / R**4) * u
    assert jnp.allclose(bih, expected, atol=5e-7)


def test_iterated_laplacian_n3_cos_theta():
    """∇^6 cos θ = −(2/R²)³ cos θ = −8/R⁶ cos θ."""
    grid = SphericalGrid2D.from_N_L(Nx=64, Ny=32)
    deriv = SphericalDerivative2D(grid=grid)
    _, THETA = grid.X
    u = jnp.cos(THETA)
    it3 = deriv.iterated_laplacian(u, n=3)
    R = grid.Ly / jnp.pi
    expected = -(8.0 / R**6) * u
    assert jnp.allclose(it3, expected, atol=5e-5)


def test_iterated_laplacian_rejects_nonpositive_n():
    import pytest

    grid = SphericalGrid2D.from_N_L(Nx=16, Ny=8)
    deriv = SphericalDerivative2D(grid=grid)
    u = jnp.ones((grid.Ny, grid.Nx))
    with pytest.raises(ValueError, match="n must be >= 1"):
        deriv.iterated_laplacian(u, n=0)


# ---------------------------------------------------------------------------
# curl sign (standard GFD convention)
# ---------------------------------------------------------------------------


def test_curl_solid_body_rotation():
    """For V_φ = Ω R sin θ, (∇×V)·r̂ should equal 2 Ω cos θ."""
    grid = SphericalGrid2D.from_N_L(Nx=64, Ny=32)
    deriv = SphericalDerivative2D(grid=grid)
    _, THETA = grid.X
    Omega = 1.5
    R = grid.Ly / jnp.pi
    v_theta = jnp.zeros_like(THETA)
    v_phi = Omega * R * jnp.sin(THETA)
    zeta = deriv.curl(v_theta, v_phi)
    expected = 2.0 * Omega * jnp.cos(THETA)
    assert jnp.allclose(zeta, expected, atol=1e-10)


def test_curl_zonal_streamfunction():
    """For V = ẑ × ∇ψ with ψ = cos θ, (∇×V)·r̂ = ∇²ψ = −2 cos θ / R²."""
    grid = SphericalGrid2D.from_N_L(Nx=64, Ny=32)
    deriv = SphericalDerivative2D(grid=grid)
    _, THETA = grid.X
    R = grid.Ly / jnp.pi
    v_theta = jnp.zeros_like(THETA)
    v_phi = -jnp.sin(THETA) / R
    zeta = deriv.curl(v_theta, v_phi)
    expected = -2.0 * jnp.cos(THETA) / R**2
    assert jnp.allclose(zeta, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Vorticity inversion (zonal test case)
# ---------------------------------------------------------------------------


def test_vorticity_inversion_zonal():
    """ζ = −2 cos θ / R²  ⇒  ψ = cos θ (up to constant), V_θ = 0, V_φ = −sin θ / R."""
    grid = SphericalGrid2D.from_N_L(Nx=64, Ny=32)
    R = grid.Ly / jnp.pi
    _, THETA = grid.X
    zeta = -2.0 / R**2 * jnp.cos(THETA)

    solver = SphericalVorticityInversionSolver(grid=grid)
    psi, (v_theta, v_phi) = solver.solve(zeta)

    psi_ex = jnp.cos(THETA)
    # ψ is unique up to an additive constant — compare after mean-removal.
    assert jnp.allclose(psi - jnp.mean(psi), psi_ex - jnp.mean(psi_ex), atol=1e-12)
    assert jnp.allclose(v_theta, 0.0, atol=1e-10)
    assert jnp.allclose(v_phi, -jnp.sin(THETA) / R, atol=1e-10)


# ---------------------------------------------------------------------------
# Divergence inversion (zonal test case)
# ---------------------------------------------------------------------------


def test_divergence_inversion_zonal():
    """δ = −2 cos θ / R²  ⇒  χ = cos θ (up to constant), V_θ = −sin θ / R, V_φ = 0."""
    grid = SphericalGrid2D.from_N_L(Nx=64, Ny=32)
    R = grid.Ly / jnp.pi
    _, THETA = grid.X
    delta = -2.0 / R**2 * jnp.cos(THETA)

    solver = SphericalDivergenceInversionSolver(grid=grid)
    chi, (v_theta, v_phi) = solver.solve(delta)

    chi_ex = jnp.cos(THETA)
    assert jnp.allclose(chi - jnp.mean(chi), chi_ex - jnp.mean(chi_ex), atol=1e-12)
    assert jnp.allclose(v_theta, -jnp.sin(THETA) / R, atol=1e-10)
    assert jnp.allclose(v_phi, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Helmholtz decomposition — reconstruct a purely rotational field
# ---------------------------------------------------------------------------


def test_helmholtz_decomposition_pure_rotation():
    """For V = ẑ × ∇(cos θ): V_rot = V, V_div = 0."""
    grid = SphericalGrid2D.from_N_L(Nx=64, Ny=32)
    R = grid.Ly / jnp.pi
    _, THETA = grid.X
    v_theta = jnp.zeros_like(THETA)
    v_phi = -jnp.sin(THETA) / R

    decomp = SphericalHelmholtzDecomposition(grid=grid)
    psi, chi, (vth_rot, vph_rot), (vth_div, vph_div) = decomp.decompose(v_theta, v_phi)

    # v_rot should match input; v_div should vanish.
    assert jnp.allclose(vth_rot, v_theta, atol=1e-10)
    assert jnp.allclose(vph_rot, v_phi, atol=1e-10)
    assert jnp.allclose(vth_div, 0.0, atol=1e-10)
    assert jnp.allclose(vph_div, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Helmholtz solver input validation
# ---------------------------------------------------------------------------


def test_helmholtz_solver_rejects_negative_alpha():
    import pytest

    from spectraldiffx import SphericalHelmholtzSolver

    grid = SphericalGrid2D.from_N_L(Nx=16, Ny=8)
    solver = SphericalHelmholtzSolver(grid=grid)
    with pytest.raises(ValueError, match="alpha must be >= 0"):
        solver.solve(jnp.zeros((grid.Ny, grid.Nx)), alpha=-1.0)
