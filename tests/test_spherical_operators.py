"""
Tests for SphericalDerivative1D and SphericalDerivative2D.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from spectraldiffx._src.spherical.grid import SphericalGrid1D, SphericalGrid2D
from spectraldiffx._src.spherical.operators import (
    SphericalDerivative1D,
    SphericalDerivative2D,
)


# ---------------------------------------------------------------------------
# SphericalDerivative1D tests
# ---------------------------------------------------------------------------


def test_spherical_deriv1d_transform_roundtrip():
    """to_spectral then from_spectral should recover the original field."""
    g = SphericalGrid1D.from_N_L(16, np.pi)
    d = SphericalDerivative1D(g)
    theta = g.x
    u = jnp.sin(theta) ** 2
    c = d.to_spectral(u)
    u_rec = d.from_spectral(c)
    assert jnp.allclose(u, u_rec, atol=1e-10)


def test_spherical_deriv1d_constant():
    """Gradient and Laplacian of a constant field should be zero."""
    N = 16
    g = SphericalGrid1D.from_N_L(N, np.pi)
    d = SphericalDerivative1D(g)
    u = jnp.ones(N)
    grad = d.gradient(u)
    lap = d.laplacian(u)
    assert jnp.allclose(grad, 0.0, atol=1e-10)
    assert jnp.allclose(lap, 0.0, atol=1e-10)


def test_spherical_deriv1d_laplacian_eigenvalue():
    """
    For P_l(cos(theta)), the spherical Laplacian should give -l*(l+1)*P_l.

    Tests l = 1, 2, 3, 5 (on unit sphere, R=1).
    """
    from scipy.special import eval_legendre

    N = 32
    g = SphericalGrid1D.from_N_L(N, np.pi)  # R = pi/pi = 1
    d = SphericalDerivative1D(g)
    mu = np.array(g.cos_theta)

    for l in [1, 2, 3, 5]:
        if l >= N:
            continue
        u = jnp.asarray(eval_legendre(l, mu))
        lap = d.laplacian(u)
        expected = -float(l * (l + 1)) * u
        assert jnp.allclose(lap, expected, atol=1e-8), (
            f"l={l}: max error = {float(jnp.max(jnp.abs(lap - expected)))}"
        )


def test_spherical_deriv1d_legendre_gradient():
    """
    d/d_theta P_2(cos(theta)) = -3 * sin(theta) * cos(theta) (analytic).

    Since P_2(mu) = (3mu^2 - 1)/2:
        d/d_theta P_2(cos(theta)) = d P_2/d_mu * (-sin(theta))
                                   = 3*cos(theta) * (-sin(theta))
                                   = -3*sin(theta)*cos(theta)
    """
    N = 32
    g = SphericalGrid1D.from_N_L(N, np.pi)
    d = SphericalDerivative1D(g)
    theta = g.x
    mu = g.cos_theta
    u = 0.5 * (3 * mu**2 - 1)  # P_2(cos(theta))
    grad = d.gradient(u)
    expected = -3.0 * jnp.sin(theta) * jnp.cos(theta)
    assert jnp.allclose(grad, expected, atol=1e-8), (
        f"Max gradient error: {float(jnp.max(jnp.abs(grad - expected)))}"
    )


# ---------------------------------------------------------------------------
# SphericalDerivative2D tests
# ---------------------------------------------------------------------------


def test_spherical_deriv2d_gradient_zonal():
    """
    For u = cos(theta) (zonal, no phi-dependence), the phi-gradient should be zero
    and the theta-gradient should match d(cos(theta))/d_theta = -sin(theta).
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    d = SphericalDerivative2D(g)
    THETA = g.y[:, None] * jnp.ones((Ny, Nx))  # broadcast colatitude
    u = jnp.cos(THETA)
    grad_theta, grad_phi = d.gradient(u)

    R = g.Ly / np.pi  # R = 1
    expected_theta = -jnp.sin(THETA) / R
    assert jnp.allclose(grad_theta, expected_theta, atol=1e-8)
    assert jnp.allclose(grad_phi, 0.0, atol=1e-8)


def test_spherical_deriv2d_laplacian_harmonic():
    """
    For u proportional to Y_1^0 = P_1(cos(theta)) = cos(theta),
    nabla^2 u = -l*(l+1)/R^2 * u = -2/R^2 * u (l=1 eigenvalue).
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    d = SphericalDerivative2D(g)
    R = g.Ly / np.pi  # R = 1
    THETA = g.y[:, None] * jnp.ones((Ny, Nx))
    u = jnp.cos(THETA)  # Y_1^0 ∝ P_1(cos(theta))
    lap = d.laplacian(u)
    expected = -2.0 / R**2 * u
    assert jnp.allclose(lap, expected, atol=1e-8)


def test_spherical_deriv2d_div_of_curl_zero():
    """curl(grad(u)) = 0 for any smooth scalar u — validates via a helper identity."""
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    d = SphericalDerivative2D(g)
    PHI, THETA = g.X
    # curl(grad(u)) = 0 (identity): compute directly using a smooth field
    u = jnp.sin(THETA) * jnp.cos(PHI)
    grad_theta_u, grad_phi_u = d.gradient(u)
    curl_of_grad = d.curl(grad_theta_u, grad_phi_u)
    assert jnp.allclose(curl_of_grad, 0.0, atol=1e-6)


def test_spherical_deriv2d_curl_of_gradient_zero():
    """curl(grad(u)) = 0 for any smooth scalar u (vector calculus identity)."""
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    d = SphericalDerivative2D(g)
    PHI, THETA = g.X
    u = jnp.sin(THETA) ** 2 * jnp.cos(PHI)
    grad_theta, grad_phi = d.gradient(u)
    zeta = d.curl(grad_theta, grad_phi)
    assert jnp.allclose(zeta, 0.0, atol=1e-6)
