"""
Tests for SphericalGrid1D and SphericalGrid2D.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from spectraldiffx._src.spherical.grid import SphericalGrid1D, SphericalGrid2D

# ---------------------------------------------------------------------------
# SphericalGrid1D tests
# ---------------------------------------------------------------------------


def test_spherical_grid_1d_node_count():
    """Gauss-Legendre grid should have exactly N nodes."""
    N = 16
    g = SphericalGrid1D.from_N_L(N, np.pi)
    assert g.cos_theta.shape == (N,)
    assert g.weights.shape == (N,)
    assert g.x.shape == (N,)


def test_spherical_grid_1d_nodes_in_range():
    """GL nodes (cos(theta)) must lie strictly in (-1, 1)."""
    g = SphericalGrid1D.from_N_L(20, np.pi)
    mu = g.cos_theta
    assert jnp.all(mu > -1.0)
    assert jnp.all(mu < 1.0)


def test_spherical_grid_1d_weight_sum():
    """GL weights should sum to 2 (integral of 1 over [-1, 1])."""
    g = SphericalGrid1D.from_N_L(16, np.pi)
    assert jnp.isclose(g.weights.sum(), 2.0, atol=1e-12)


def test_spherical_grid_1d_quadrature_exact():
    """
    GL quadrature should be exact for polynomials of degree <= 2*N-1.

    Test orthogonality: integral_{-1}^{1} P_l(mu)*P_l'(mu) dmu = 2/(2l+1)*delta_{ll'}.
    """
    from orthax.legendre import legvander

    N = 16
    g = SphericalGrid1D.from_N_L(N, np.pi)
    mu = np.array(g.cos_theta)
    w = np.array(g.weights)
    V = np.array(legvander(mu, 4))  # V[j, l] = P_l(mu[j])

    # Check P_2 and P_4 orthogonality: sum w_j * P_2(mu_j) * P_4(mu_j) ≈ 0
    inner = float(np.sum(w * V[:, 2] * V[:, 4]))
    assert abs(inner) < 1e-12, f"Orthogonality failed: {inner}"

    # Check P_3 norm: sum w_j * P_3^2(mu_j) = 2/(2*3+1) = 2/7
    norm_sq = float(np.sum(w * V[:, 3] ** 2))
    assert abs(norm_sq - 2.0 / 7.0) < 1e-12


def test_spherical_grid_1d_theta_range():
    """Colatitude points theta should lie in (0, pi) — poles excluded."""
    g = SphericalGrid1D.from_N_L(16, np.pi)
    theta = g.x
    assert jnp.all(theta > 0.0)
    assert jnp.all(theta < np.pi)


def test_spherical_grid_1d_factory_consistency():
    """from_N_L, from_N_dx, and from_L_dx should give the same grid."""
    N, L = 16, np.pi
    g1 = SphericalGrid1D.from_N_L(N, L)
    g2 = SphericalGrid1D.from_N_dx(N, L / N)
    g3 = SphericalGrid1D.from_L_dx(L, L / N)

    assert g1.N == g2.N == g3.N
    assert jnp.isclose(g1.L, g2.L) and jnp.isclose(g1.L, g3.L)
    assert jnp.isclose(g1.dx, g2.dx) and jnp.isclose(g1.dx, g3.dx)
    g1.check_consistency()


def test_spherical_grid_1d_dlt_roundtrip():
    """Forward then inverse DLT should recover the original field."""
    g = SphericalGrid1D.from_N_L(16, np.pi)
    theta = g.x
    u = jnp.sin(2 * theta) + jnp.cos(theta)
    c = g.transform(u)
    u_rec = g.transform(c, inverse=True)
    assert jnp.allclose(u, u_rec, atol=1e-10)


def test_spherical_grid_1d_dealias_filter_shape():
    """Dealiasing filter should have the right shape."""
    N = 16
    g = SphericalGrid1D.from_N_L(N, np.pi)
    mask = g.dealias_filter()
    assert mask.shape == (N,)
    # DC mode (l=0) must be kept
    assert float(mask[0]) == 1.0


# ---------------------------------------------------------------------------
# SphericalGrid2D tests
# ---------------------------------------------------------------------------


def test_spherical_grid_2d_longitude_uniform():
    """Longitude grid should be uniformly spaced in [0, 2*pi)."""
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    phi = g.x
    diffs = jnp.diff(phi)
    assert jnp.allclose(diffs, diffs[0], atol=1e-12)
    assert float(phi[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(phi[-1]) < float(2 * np.pi)


def test_spherical_grid_2d_latitude_nonuniform():
    """Latitude grid should be non-uniform (Gauss-Legendre, not uniform)."""
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    theta = g.y  # colatitude
    diffs = jnp.diff(theta)
    # For a non-uniform grid, differences must not all be equal
    assert not jnp.allclose(diffs, diffs[0], atol=1e-5)


def test_spherical_grid_2d_meshgrid_shapes():
    """X (meshgrid) should have shape (Ny, Nx)."""
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    PHI, THETA = g.X
    assert PHI.shape == (Ny, Nx)
    assert THETA.shape == (Ny, Nx)


def test_spherical_grid_2d_dealias_filter_shape():
    """2D dealiasing filter should have shape (Ny, Nx)."""
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    mask = g.dealias_filter()
    assert mask.shape == (Ny, Nx)


def test_spherical_grid_2d_sht_roundtrip():
    """Forward + inverse SHT should recover the original field."""
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    PHI, THETA = g.X
    u = jnp.sin(THETA) * jnp.cos(PHI)
    u_hat = g.transform(u)
    u_rec = g.transform(u_hat, inverse=True)
    assert jnp.allclose(u, u_rec, atol=1e-10)


def test_spherical_grid_2d_laplacian_eigenvalues_shape():
    """Laplacian eigenvalues should have shape (Ny, Nx)."""
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    eig = g.laplacian_eigenvalues
    assert eig.shape == (Ny, Nx)
    # l=0 row should be 0
    assert float(eig[0, 0]) == pytest.approx(0.0)
    # l=1 row should be -1*(1+1) = -2
    assert float(eig[1, 0]) == pytest.approx(-2.0)


def test_spherical_grid_2d_weights_shape():
    """SphericalGrid2D.weights must have shape (Ny, Nx)."""
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    assert g.weights.shape == (Ny, Nx)


def test_spherical_grid_2d_weights_sphere_area():
    """
    The sum of SphericalGrid2D.weights should equal the area of the sphere.

    Integrating u=1 over the sphere:
        ∫∫ dΩ = ∫₀²π dphi * ∫₀^π sin(theta) dtheta = 2π * 2 = 4π

    For radius R = Ly/π: area = 4π * R².
    The weights w[j,k] = w_lat[j] * dx_lon absorb the sin(theta) Jacobian
    (GL weights integrate mu=cos(theta) over [-1,1]), so:
        Σ_jk w[j,k] * 1 = (Σ_j w_lat[j]) * Nx * dx = 2 * Lx = 4π (for R=1).
    """
    import math

    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    R = g.Ly / np.pi

    area = float(jnp.sum(g.weights))
    expected = 4 * math.pi * R**2

    assert abs(area - expected) < 1e-12, (
        f"Sphere area: sum(weights) = {area:.12f}, expected 4πR² = {expected:.12f}"
    )
