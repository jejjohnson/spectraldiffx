import jax.numpy as jnp
import pytest
from spectraldiffx._src.grid import FourierGrid1D, FourierGrid2D, FourierGrid3D
from spectraldiffx._src.operators import (
    SpectralDerivative1D, SpectralDerivative2D, SpectralDerivative3D
)

# --- 1D Tests ---

def test_deriv1d_gradient():
    """Verify du/dx for u = sin(x) -> exact = cos(x)."""
    grid = FourierGrid1D.from_N_L(64, 2 * jnp.pi, dealias=None)
    deriv = SpectralDerivative1D(grid)
    u = jnp.sin(grid.x)
    du_dx = deriv.gradient(u)
    assert jnp.allclose(du_dx, jnp.cos(grid.x), atol=1e-5)

def test_deriv1d_laplacian():
    """Verify d^2u/dx^2 for u = sin(x) -> exact = -sin(x)."""
    grid = FourierGrid1D.from_N_L(64, 2 * jnp.pi, dealias=None)
    deriv = SpectralDerivative1D(grid)
    u = jnp.sin(grid.x)
    d2u_dx2 = deriv.laplacian(u)
    assert jnp.allclose(d2u_dx2, -jnp.sin(grid.x), atol=1e-3, rtol=1e-3)

# --- 2D Tests ---

def test_deriv2d_gradient():
    """Check ∇u components for u = sin(x)cos(y)."""
    grid = FourierGrid2D.from_N_L(32, 32, 2*jnp.pi, 2*jnp.pi, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    u = jnp.sin(X) * jnp.cos(Y)
    gx, gy = deriv.gradient(u)
    # ∂u/∂x = cos(x)cos(y), ∂u/∂y = -sin(x)sin(y)
    assert jnp.allclose(gx, jnp.cos(X) * jnp.cos(Y), atol=1e-5)
    assert jnp.allclose(gy, -jnp.sin(X) * jnp.sin(Y), atol=1e-5)

def test_deriv2d_divergence():
    """Check div(V) for V = (sin(x), cos(y))."""
    grid = FourierGrid2D.from_N_L(32, 32, 2*jnp.pi, 2*jnp.pi, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    vx, vy = jnp.sin(X), jnp.cos(Y)
    div = deriv.divergence(vx, vy)
    # div = ∂vx/∂x + ∂vy/∂y = cos(x) - sin(y)
    assert jnp.allclose(div, jnp.cos(X) - jnp.sin(Y), atol=1e-5)

def test_deriv2d_curl():
    grid = FourierGrid2D.from_N_L(32, 32, 2*jnp.pi, 2*jnp.pi, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    vx, vy = jnp.cos(Y), jnp.sin(X)
    curl = deriv.curl(vx, vy)
    # ∂vy/∂x - ∂vx/∂y = cos(X) - (-sin(Y))
    assert jnp.allclose(curl, jnp.cos(X) + jnp.sin(Y), atol=1e-5)

def test_deriv2d_laplacian():
    grid = FourierGrid2D.from_N_L(32, 32, 2*jnp.pi, 2*jnp.pi, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    u = jnp.sin(X) * jnp.sin(Y)
    lap = deriv.laplacian(u)
    # Use rtol for relative comparison
    assert jnp.allclose(lap, -2 * jnp.sin(X) * jnp.sin(Y), atol=1e-4, rtol=1e-4)

def test_deriv2d_project_vector():
    """
    Check Leray Projection by decomposing a field into potential and solenoidal parts.
    V = grad(sin(x)sin(y)) [irrotational] + (-sin(y), cos(x)) [solenoidal]
    """
    grid = FourierGrid2D.from_N_L(32, 32, 2*jnp.pi, 2*jnp.pi, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    
    vx = jnp.cos(X) * jnp.sin(Y) - jnp.sin(Y)
    vy = jnp.sin(X) * jnp.cos(Y) + jnp.cos(X)
    
    # Projecting should result in zero divergence
    vx_proj, vy_proj = deriv.project_vector(vx, vy)
    div = deriv.divergence(vx_proj, vy_proj)
    assert jnp.allclose(div, 0.0, atol=1e-5)

def test_deriv2d_advection():
    """Verify (u·∇)q for uniform advection u=1, v=0 -> exact = ∂q/∂x."""
    grid = FourierGrid2D.from_N_L(32, 32, 2*jnp.pi, 2*jnp.pi, dealias=None)
    deriv = SpectralDerivative2D(grid)
    X, Y = grid.X
    u, v = jnp.ones_like(X), jnp.zeros_like(Y)
    q = jnp.sin(X)
    adv = deriv.advection_scalar(u, v, q)
    assert jnp.allclose(adv, jnp.cos(X), atol=1e-5)

# --- 3D Tests ---

def test_deriv3d_gradient():
    grid = FourierGrid3D.from_N_L(16, 16, 16, 2*jnp.pi, 2*jnp.pi, 2*jnp.pi, dealias=None)
    deriv = SpectralDerivative3D(grid)
    Z, Y, X = grid.X
    u = jnp.sin(Z) * jnp.cos(Y) * jnp.sin(X)
    gz, gy, gx = deriv.gradient(u)
    assert jnp.allclose(gz, jnp.cos(Z) * jnp.cos(Y) * jnp.sin(X), atol=1e-5)
    assert jnp.allclose(gy, -jnp.sin(Z) * jnp.sin(Y) * jnp.sin(X), atol=1e-5)
    assert jnp.allclose(gx, jnp.sin(Z) * jnp.cos(Y) * jnp.cos(X), atol=1e-5)

def test_deriv3d_divergence():
    """Verify ∇·V for 3D vector field."""
    grid = FourierGrid3D.from_N_L(16, 16, 16, 2*jnp.pi, 2*jnp.pi, 2*jnp.pi, dealias=None)
    deriv = SpectralDerivative3D(grid)
    Z, Y, X = grid.X
    vz, vy, vx = jnp.sin(Z), jnp.cos(Y), jnp.sin(X)
    div = deriv.divergence(vz, vy, vx)
    # div = cos(z) - sin(y) + cos(x)
    assert jnp.allclose(div, jnp.cos(Z) - jnp.sin(Y) + jnp.cos(X), atol=1e-5)

def test_deriv3d_curl():
    """Verify ∇ x V components."""
    grid = FourierGrid3D.from_N_L(16, 16, 16, 2*jnp.pi, 2*jnp.pi, 2*jnp.pi, dealias=None)
    deriv = SpectralDerivative3D(grid)
    Z, Y, X = grid.X
    # Specific setup to isolate components
    vz, vy, vx = jnp.sin(Y), jnp.sin(X), jnp.sin(Z)
    wz, wy, wx = deriv.curl(vz, vy, vx)
    # ωz = ∂vy/∂x - ∂vx/∂y = cos(x) - 0
    assert jnp.allclose(wz, jnp.cos(X), atol=1e-5)
    # ωy = ∂vx/∂z - ∂vz/∂x = cos(z) - 0
    assert jnp.allclose(wy, jnp.cos(Z), atol=1e-5)
    # ωx = ∂vz/∂y - ∂vy/∂z = cos(y) - 0
    assert jnp.allclose(wx, jnp.cos(Y), atol=1e-5)

def test_deriv3d_laplacian():
    grid = FourierGrid3D.from_N_L(16, 16, 16, 2*jnp.pi, 2*jnp.pi, 2*jnp.pi, dealias=None)
    deriv = SpectralDerivative3D(grid)
    Z, Y, X = grid.X
    u = jnp.sin(Z) * jnp.sin(Y) * jnp.sin(X)
    lap = deriv.laplacian(u)
    assert jnp.allclose(lap, -3 * jnp.sin(Z) * jnp.sin(Y) * jnp.sin(X), atol=1e-5)

def test_deriv3d_project_vector():
    grid = FourierGrid3D.from_N_L(16, 16, 16, 2*jnp.pi, 2*jnp.pi, 2*jnp.pi, dealias=None)
    deriv = SpectralDerivative3D(grid)
    Z, Y, X = grid.X
    # Random vector field
    vz = jnp.sin(X+Y)
    vy = jnp.cos(Z)
    vx = jnp.sin(Y*Z)
    
    vz_p, vy_p, vx_p = deriv.project_vector(vz, vy, vx)
    div = deriv.divergence(vz_p, vy_p, vx_p)
    assert jnp.allclose(div, 0.0, atol=1e-5)
