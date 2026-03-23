"""Tests for Fourier physics operators (biharmonic, hyperviscosity,
inverse Laplacian, Jacobian, velocity from streamfunction)."""

import jax.numpy as jnp

from spectraldiffx._src.fourier.grid import FourierGrid1D, FourierGrid2D, FourierGrid3D
from spectraldiffx._src.fourier.operators import (
    SpectralDerivative1D,
    SpectralDerivative2D,
    SpectralDerivative3D,
)

PI = jnp.pi
TWO_PI = 2.0 * jnp.pi

# ---------------------------------------------------------------------------
# 1D Tests
# ---------------------------------------------------------------------------


def test_1d_biharmonic():
    """d^4/dx^4 sin(x) = sin(x)."""
    grid = FourierGrid1D.from_N_L(64, TWO_PI, dealias=None)
    deriv = SpectralDerivative1D(grid)
    u = jnp.sin(grid.x)
    bih = deriv.biharmonic(u)
    assert jnp.allclose(bih, jnp.sin(grid.x), atol=1e-5)


def test_1d_biharmonic_cos2x():
    """d^4/dx^4 cos(2x) = 16*cos(2x)."""
    grid = FourierGrid1D.from_N_L(64, TWO_PI, dealias=None)
    deriv = SpectralDerivative1D(grid)
    u = jnp.cos(2 * grid.x)
    bih = deriv.biharmonic(u)
    assert jnp.allclose(bih, 16 * jnp.cos(2 * grid.x), atol=1e-4)


def test_1d_hyperviscosity_order1():
    """Order 1 hyperviscosity is always dissipative: -nu * k^2 * u_hat."""
    grid = FourierGrid1D.from_N_L(64, TWO_PI, dealias=None)
    deriv = SpectralDerivative1D(grid)
    u = jnp.sin(grid.x)
    nu = 0.5
    hyp = deriv.hyperviscosity(u, nu=nu, order=1)
    # -nu * k^2 * sin_hat → nu * laplacian(sin(x)) = nu * (-sin(x))
    lap = deriv.laplacian(u)
    assert jnp.allclose(hyp, nu * lap, atol=1e-5)


def test_1d_hyperviscosity_order2():
    """Order 2 hyperviscosity: -nu * k^4 * u_hat."""
    grid = FourierGrid1D.from_N_L(64, TWO_PI, dealias=None)
    deriv = SpectralDerivative1D(grid)
    u = jnp.sin(grid.x)
    nu = 0.5
    hyp = deriv.hyperviscosity(u, nu=nu, order=2)
    # -nu * k^4 * sin_hat = -nu * biharmonic(sin(x)) = -nu * sin(x)
    bih = deriv.biharmonic(u)
    assert jnp.allclose(hyp, -nu * bih, atol=1e-5)


def test_1d_inverse_laplacian():
    """nabla^{-2}(-sin(x)) = sin(x) (since d^2/dx^2 sin(x) = -sin(x))."""
    grid = FourierGrid1D.from_N_L(64, TWO_PI, dealias=None)
    deriv = SpectralDerivative1D(grid)
    rhs = -jnp.sin(grid.x)
    psi = deriv.inverse_laplacian(rhs)
    assert jnp.allclose(psi, jnp.sin(grid.x), atol=1e-5)


def test_1d_inverse_laplacian_roundtrip():
    """laplacian(inverse_laplacian(f)) = f for zero-mean f."""
    grid = FourierGrid1D.from_N_L(64, TWO_PI, dealias=None)
    deriv = SpectralDerivative1D(grid)
    f = jnp.sin(2 * grid.x) + 0.5 * jnp.cos(3 * grid.x)
    psi = deriv.inverse_laplacian(f)
    lap_psi = deriv.laplacian(psi)
    assert jnp.allclose(lap_psi, f, atol=1e-5)


# ---------------------------------------------------------------------------
# 2D Tests
# ---------------------------------------------------------------------------


def test_2d_biharmonic():
    """nabla^4 sin(x)sin(y) = 4*sin(x)sin(y)."""
    grid = FourierGrid2D.from_N_L(32, 32, TWO_PI, TWO_PI, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    u = jnp.sin(X) * jnp.sin(Y)
    bih = deriv.biharmonic(u)
    # nabla^4 = (kx^2+ky^2)^2 = (1+1)^2 = 4
    assert jnp.allclose(bih, 4 * jnp.sin(X) * jnp.sin(Y), atol=1e-4)


def test_2d_hyperviscosity_order1():
    """Order 1: (-1)^2 * nu * nabla^2 u = nu * nabla^2 u (diffusion)."""
    grid = FourierGrid2D.from_N_L(32, 32, TWO_PI, TWO_PI, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    u = jnp.sin(X) * jnp.sin(Y)
    nu = 0.5
    hyp = deriv.hyperviscosity(u, nu=nu, order=1)
    lap = deriv.laplacian(u)
    assert jnp.allclose(hyp, nu * lap, atol=1e-5)


def test_2d_inverse_laplacian():
    """nabla^{-2}(-2*sin(x)*sin(y)) = sin(x)*sin(y)."""
    grid = FourierGrid2D.from_N_L(32, 32, TWO_PI, TWO_PI, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    rhs = -2 * jnp.sin(X) * jnp.sin(Y)
    psi = deriv.inverse_laplacian(rhs)
    assert jnp.allclose(psi, jnp.sin(X) * jnp.sin(Y), atol=1e-5)


def test_2d_inverse_laplacian_roundtrip():
    """laplacian(inverse_laplacian(f)) = f for zero-mean f."""
    grid = FourierGrid2D.from_N_L(32, 32, TWO_PI, TWO_PI, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    f = jnp.sin(2 * X) * jnp.cos(Y) + 0.5 * jnp.cos(3 * X) * jnp.sin(2 * Y)
    psi = deriv.inverse_laplacian(f)
    lap_psi = deriv.laplacian(psi)
    assert jnp.allclose(lap_psi, f, atol=1e-4)


def test_2d_velocity_from_streamfunction():
    """psi = sin(x)sin(y) => u = -sin(x)cos(y), v = cos(x)sin(y)."""
    grid = FourierGrid2D.from_N_L(32, 32, TWO_PI, TWO_PI, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    psi = jnp.sin(X) * jnp.sin(Y)
    u, v = deriv.velocity_from_streamfunction(psi)
    # u = -dpsi/dy = -sin(x)cos(y)
    # v =  dpsi/dx =  cos(x)sin(y)
    assert jnp.allclose(u, -jnp.sin(X) * jnp.cos(Y), atol=1e-5)
    assert jnp.allclose(v, jnp.cos(X) * jnp.sin(Y), atol=1e-5)


def test_2d_velocity_divergence_free():
    """Velocity from streamfunction must be divergence-free."""
    grid = FourierGrid2D.from_N_L(32, 32, TWO_PI, TWO_PI, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    psi = jnp.sin(2 * X) * jnp.cos(3 * Y)
    u, v = deriv.velocity_from_streamfunction(psi)
    div = deriv.divergence(u, v)
    assert jnp.allclose(div, 0.0, atol=1e-10)


def test_2d_jacobian_antisymmetric():
    """J(f, g) = -J(g, f)."""
    grid = FourierGrid2D.from_N_L(32, 32, TWO_PI, TWO_PI, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    f = jnp.sin(X) * jnp.cos(Y)
    g = jnp.cos(2 * X) * jnp.sin(Y)
    j_fg = deriv.jacobian(f, g)
    j_gf = deriv.jacobian(g, f)
    assert jnp.allclose(j_fg, -j_gf, atol=1e-5)


def test_2d_jacobian_analytic():
    """J(sin(x), sin(y)) = cos(x)*cos(y)."""
    grid = FourierGrid2D.from_N_L(32, 32, TWO_PI, TWO_PI, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    f = jnp.sin(X)
    g = jnp.sin(Y)
    jac = deriv.jacobian(f, g)
    # J = df/dx * dg/dy - df/dy * dg/dx = cos(x)*cos(y) - 0*0 = cos(x)*cos(y)
    assert jnp.allclose(jac, jnp.cos(X) * jnp.cos(Y), atol=1e-5)


def test_2d_jacobian_self_zero():
    """J(f, f) = 0."""
    grid = FourierGrid2D.from_N_L(32, 32, TWO_PI, TWO_PI, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    f = jnp.sin(X) * jnp.cos(Y)
    jac = deriv.jacobian(f, f)
    assert jnp.allclose(jac, 0.0, atol=1e-10)


def test_2d_curl_matches_vorticity():
    """curl(u, v) = dv/dx - du/dy = nabla^2(psi) when u=-dpsi/dy, v=dpsi/dx."""
    grid = FourierGrid2D.from_N_L(32, 32, TWO_PI, TWO_PI, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    psi = jnp.sin(X) * jnp.sin(Y)
    u, v = deriv.velocity_from_streamfunction(psi)
    vort = deriv.curl(u, v)
    lap_psi = deriv.laplacian(psi)
    assert jnp.allclose(vort, lap_psi, atol=1e-4)


# ---------------------------------------------------------------------------
# 3D Tests
# ---------------------------------------------------------------------------


def test_3d_biharmonic():
    """nabla^4 sin(x)sin(y)sin(z) = 9*sin(x)sin(y)sin(z)."""
    grid = FourierGrid3D.from_N_L(16, 16, 16, TWO_PI, TWO_PI, TWO_PI, dealias=None)
    deriv = SpectralDerivative3D(grid)
    Z, Y, X = grid.X
    u = jnp.sin(X) * jnp.sin(Y) * jnp.sin(Z)
    bih = deriv.biharmonic(u)
    # nabla^4 = (1+1+1)^2 = 9
    assert jnp.allclose(bih, 9 * u, atol=1e-3)


def test_3d_inverse_laplacian_roundtrip():
    """laplacian(inverse_laplacian(f)) = f."""
    grid = FourierGrid3D.from_N_L(16, 16, 16, TWO_PI, TWO_PI, TWO_PI, dealias=None)
    deriv = SpectralDerivative3D(grid)
    Z, Y, X = grid.X
    f = jnp.sin(X) * jnp.cos(Y) * jnp.sin(Z)
    psi = deriv.inverse_laplacian(f)
    lap_psi = deriv.laplacian(psi)
    assert jnp.allclose(lap_psi, f, atol=1e-3)


def test_3d_velocity_from_streamfunction():
    """psi = sin(x)sin(y) (z-independent) => u = -sin(x)cos(y), v = cos(x)sin(y)."""
    grid = FourierGrid3D.from_N_L(8, 16, 16, TWO_PI, TWO_PI, TWO_PI, dealias=None)
    deriv = SpectralDerivative3D(grid)
    _Z, Y, X = grid.X
    psi = jnp.sin(X) * jnp.sin(Y)
    u, v = deriv.velocity_from_streamfunction(psi)
    assert jnp.allclose(u, -jnp.sin(X) * jnp.cos(Y), atol=1e-4)
    assert jnp.allclose(v, jnp.cos(X) * jnp.sin(Y), atol=1e-4)


def test_3d_jacobian_antisymmetric():
    """J(f, g) = -J(g, f)."""
    grid = FourierGrid3D.from_N_L(8, 16, 16, TWO_PI, TWO_PI, TWO_PI, dealias=None)
    deriv = SpectralDerivative3D(grid)
    Z, Y, X = grid.X
    f = jnp.sin(X) * jnp.cos(Y) * jnp.ones_like(Z)
    g = jnp.cos(X) * jnp.sin(Y) * jnp.ones_like(Z)
    j_fg = deriv.jacobian(f, g)
    j_gf = deriv.jacobian(g, f)
    assert jnp.allclose(j_fg, -j_gf, atol=1e-4)


def test_3d_hyperviscosity_order1():
    """Order 1 hyperviscosity = nu * nabla^2 u."""
    grid = FourierGrid3D.from_N_L(16, 16, 16, TWO_PI, TWO_PI, TWO_PI, dealias=None)
    deriv = SpectralDerivative3D(grid)
    Z, Y, X = grid.X
    u = jnp.sin(X) * jnp.sin(Y) * jnp.sin(Z)
    nu = 0.5
    hyp = deriv.hyperviscosity(u, nu=nu, order=1)
    lap = deriv.laplacian(u)
    assert jnp.allclose(hyp, nu * lap, atol=1e-3)


# ---------------------------------------------------------------------------
# Hyperviscosity validation
# ---------------------------------------------------------------------------


def test_hyperviscosity_invalid_order():
    """order < 1 raises ValueError."""
    grid = FourierGrid1D.from_N_L(64, TWO_PI, dealias=None)
    deriv = SpectralDerivative1D(grid)
    u = jnp.sin(grid.x)
    import pytest

    with pytest.raises(ValueError, match="order must be >= 1"):
        deriv.hyperviscosity(u, nu=1.0, order=0)


def test_hyperviscosity_negative_nu():
    """nu < 0 raises ValueError."""
    grid = FourierGrid2D.from_N_L(16, 16, TWO_PI, TWO_PI, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    u = jnp.sin(X) * jnp.sin(Y)
    import pytest

    with pytest.raises(ValueError, match="nu must be >= 0"):
        deriv.hyperviscosity(u, nu=-0.1, order=2)


# ---------------------------------------------------------------------------
# Jacobian dealiasing
# ---------------------------------------------------------------------------


def test_2d_jacobian_dealiased():
    """Jacobian with dealias='2/3' should have zero energy above cutoff."""
    N = 32
    grid = FourierGrid2D.from_N_L(N, N, TWO_PI, TWO_PI, dealias="2/3")
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    # High-frequency functions that produce aliased products
    f = jnp.sin(8 * X) * jnp.cos(6 * Y)
    g = jnp.cos(7 * X) * jnp.sin(9 * Y)
    jac = deriv.jacobian(f, g)
    # Check that the result is dealiased: FFT should have zero above cutoff
    jac_hat = grid.transform(jac)
    mask = grid.dealias_filter()
    # Energy outside the dealias mask should be zero
    energy_outside = jnp.sum(jnp.abs(jac_hat * (1 - mask)) ** 2)
    assert energy_outside < 1e-20
