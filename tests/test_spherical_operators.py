"""
Tests for SphericalDerivative1D and SphericalDerivative2D.
"""

import jax
import jax.numpy as jnp
import numpy as np

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


# ---------------------------------------------------------------------------
# SphericalDerivative2D.divergence tests
# ---------------------------------------------------------------------------


def test_spherical_deriv2d_divergence_of_zonal_gradient():
    """
    div(grad(cos(theta))) = laplacian(cos(theta)) for a zonal field.

    For a purely zonal field (no phi dependence), the theta-gradient is
    accurately computed by the 1D Legendre derivative, so div(grad(u))
    should match laplacian(u) from the eigenvalue computation.
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    d = SphericalDerivative2D(g)
    THETA = g.y[:, None] * jnp.ones((Ny, Nx))

    u = jnp.cos(THETA)  # P_1(cos(theta)), l=1 eigenvalue -2/R^2
    grad_th, grad_ph = d.gradient(u)
    div_grad = d.divergence(grad_th, grad_ph)
    lap = d.laplacian(u)

    assert jnp.allclose(div_grad, lap, atol=1e-8), (
        f"div(grad(cos(theta))) vs laplacian: max error = "
        f"{float(jnp.max(jnp.abs(div_grad - lap))):.2e}"
    )


def test_spherical_deriv2d_divergence_free_streamfunction_velocity():
    """
    The velocity field derived from a streamfunction must be divergence-free.

    For ψ(theta, phi) = (1 - cos²(theta)) * cos(phi), the geostrophic
    velocity components are:
        v_theta = -1/(R*sin) * dψ/dphi = sin(phi)*sin(theta)/R
        v_phi   = 1/R * dψ/dtheta     = 2*sin(theta)*cos(theta)*cos(phi)/R

    Analytically: div(v_theta, v_phi) = 0.
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    d = SphericalDerivative2D(g)
    PHI, THETA = g.X
    R = g.Ly / jnp.pi

    # ψ = (1 - cos²θ)*cos(φ) = sin²(θ)*cos(φ) — theta part is polynomial in cos(θ)
    psi = (1.0 - jnp.cos(THETA) ** 2) * jnp.cos(PHI)
    sin_theta = jnp.sin(THETA)

    psi_hat = jnp.fft.fft(psi, axis=-1)
    m_phys = 2 * jnp.pi * jnp.fft.fftfreq(Nx, g.dx)
    dpsi_dphi = jnp.fft.ifft(1j * m_phys[None, :] * psi_hat, axis=-1).real
    dpsi_dtheta = jax.vmap(d.deriv_theta.gradient, in_axes=1, out_axes=1)(psi)

    v_theta = -dpsi_dphi / (R * sin_theta)
    v_phi = dpsi_dtheta / R

    div_v = d.divergence(v_theta, v_phi)
    assert jnp.allclose(div_v, 0.0, atol=1e-10), (
        f"Streamfunction velocity should be divergence-free; "
        f"max |div| = {float(jnp.max(jnp.abs(div_v))):.2e}"
    )


def test_spherical_deriv2d_divergence_known_div_free_field():
    """
    Analytically divergence-free field should have zero divergence.

    For V = (sin(phi)*sin(theta)/R, sin(2*theta)*cos(phi)/R):
        d(V_theta*sin)/dtheta = d(sin²θ*sin(phi)/R)/dtheta = sin(2θ)*sin(phi)/R
        dV_phi/dphi = d(sin(2θ)*cos(phi)/R)/dphi = -sin(2θ)*sin(phi)/R
    They cancel, so div = 0.
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    d = SphericalDerivative2D(g)
    PHI, THETA = g.X
    R = g.Ly / jnp.pi

    v_theta = jnp.sin(PHI) * jnp.sin(THETA) / R
    v_phi = jnp.sin(2 * THETA) * jnp.cos(PHI) / R

    div_v = d.divergence(v_theta, v_phi)
    assert jnp.allclose(div_v, 0.0, atol=1e-10), (
        f"Known div-free field: max |div| = {float(jnp.max(jnp.abs(div_v))):.2e}"
    )


# ---------------------------------------------------------------------------
# SphericalDerivative2D.advection_scalar tests
# ---------------------------------------------------------------------------


def test_spherical_deriv2d_advection_scalar_zero_velocity():
    """Zero velocity must give zero advection for any tracer field."""
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    d = SphericalDerivative2D(g)
    THETA = g.y[:, None] * jnp.ones((Ny, Nx))

    q = jnp.cos(THETA)
    zero = jnp.zeros((Ny, Nx))
    adv = d.advection_scalar(zero, zero, q)
    assert jnp.allclose(adv, 0.0, atol=1e-15), (
        "Zero velocity must give zero advection"
    )


def test_spherical_deriv2d_advection_scalar_zonal_flow():
    """
    For v_theta = 1/R, v_phi = 0, q = cos(theta) (zonal field):

        (V·∇)q = v_theta * grad_theta(q)
               = (1/R) * (-sin(theta)/R) = -sin(theta) / R²
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    d = SphericalDerivative2D(g)
    PHI, THETA = g.X
    R = g.Ly / jnp.pi

    v_theta = jnp.ones((Ny, Nx)) / R
    v_phi = jnp.zeros((Ny, Nx))
    q = jnp.cos(THETA)

    adv = d.advection_scalar(v_theta, v_phi, q)
    expected = -jnp.sin(THETA) / R**2
    assert jnp.allclose(adv, expected, atol=1e-10), (
        f"Zonal advection: max error = {float(jnp.max(jnp.abs(adv - expected))):.2e}"
    )


def test_spherical_deriv2d_advection_scalar_phi_flow():
    """
    For v_theta = 0, v_phi = 1.0, q = cos(phi):

        (V·∇)q = v_phi * grad_phi(cos(phi))
               = 1.0 * (-sin(phi) / (R * sin(theta)))
               = -sin(phi) / (R * sin(theta))
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    d = SphericalDerivative2D(g)
    PHI, THETA = g.X
    R = g.Ly / jnp.pi

    v_theta = jnp.zeros((Ny, Nx))
    v_phi = jnp.ones((Ny, Nx))
    q = jnp.cos(PHI)

    adv = d.advection_scalar(v_theta, v_phi, q)
    expected = -jnp.sin(PHI) / (R * jnp.sin(THETA))
    assert jnp.allclose(adv, expected, atol=1e-10), (
        f"Phi-flow advection: max error = {float(jnp.max(jnp.abs(adv - expected))):.2e}"
    )
