"""
Tests for ChebyshevDerivative1D and ChebyshevDerivative2D.
"""

import jax.numpy as jnp
import pytest

from spectraldiffx._src.chebyshev.grid import ChebyshevGrid1D, ChebyshevGrid2D
from spectraldiffx._src.chebyshev.operators import (
    ChebyshevDerivative1D,
    ChebyshevDerivative2D,
)


# ============================================================================
# 1D derivative tests
# ============================================================================


def test_cheb_deriv1d_polynomial():
    """u = x³ → du/dx = 3x² (exact for polynomials of degree ≤ N)."""
    N = 16
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    deriv = ChebyshevDerivative1D(grid)
    x = grid.x
    u = x**3
    du_dx = deriv.gradient(u)
    expected = 3.0 * x**2
    assert jnp.allclose(du_dx, expected, atol=1e-8), (
        f"max error = {jnp.abs(du_dx - expected).max()}"
    )


def test_cheb_deriv1d_trig():
    """u = sin(πx) → du/dx = π cos(πx), spectral accuracy."""
    N = 32
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    deriv = ChebyshevDerivative1D(grid)
    x = grid.x
    u = jnp.sin(jnp.pi * x)
    du_dx = deriv.gradient(u)
    expected = jnp.pi * jnp.cos(jnp.pi * x)
    assert jnp.allclose(du_dx, expected, atol=1e-5), (
        f"max error = {jnp.abs(du_dx - expected).max()}"
    )


def test_cheb_deriv1d_laplacian_polynomial():
    """u = x⁴ → d²u/dx² = 12x² (exact)."""
    N = 16
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    deriv = ChebyshevDerivative1D(grid)
    x = grid.x
    u = x**4
    d2u = deriv.laplacian(u)
    expected = 12.0 * x**2
    assert jnp.allclose(d2u, expected, atol=1e-7), (
        f"max error = {jnp.abs(d2u - expected).max()}"
    )


def test_cheb_deriv1d_laplacian_trig():
    """u = cos(2πx) → d²u/dx² = -4π²cos(2πx), spectral accuracy."""
    N = 32
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    deriv = ChebyshevDerivative1D(grid)
    x = grid.x
    u = jnp.cos(2 * jnp.pi * x)
    d2u = deriv.laplacian(u)
    expected = -(2 * jnp.pi) ** 2 * jnp.cos(2 * jnp.pi * x)
    assert jnp.allclose(d2u, expected, atol=1e-4), (
        f"max error = {jnp.abs(d2u - expected).max()}"
    )


def test_cheb_deriv1d_constant_field():
    """Gradient and Laplacian of a constant field should be zero."""
    N = 16
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    deriv = ChebyshevDerivative1D(grid)
    u = jnp.ones(N + 1) * 5.0
    assert jnp.allclose(deriv.gradient(u), 0.0, atol=1e-10)
    assert jnp.allclose(deriv.laplacian(u), 0.0, atol=1e-10)


def test_cheb_deriv1d_higher_order():
    """__call__ with order=2 matches laplacian."""
    N = 16
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    deriv = ChebyshevDerivative1D(grid)
    x = grid.x
    u = x**4
    assert jnp.allclose(deriv(u, order=2), deriv.laplacian(u), atol=1e-10)


def test_cheb_deriv1d_scaled_domain():
    """Derivative on [-L, L] with L=2: du/dx on scaled domain."""
    N = 16
    L = 2.0
    grid = ChebyshevGrid1D.from_N_L(N=N, L=L)
    deriv = ChebyshevDerivative1D(grid)
    x = grid.x
    # u = x², du/dx = 2x (exact regardless of L)
    u = x**2
    du_dx = deriv.gradient(u)
    expected = 2.0 * x
    assert jnp.allclose(du_dx, expected, atol=1e-7), (
        f"max error = {jnp.abs(du_dx - expected).max()}"
    )


# ============================================================================
# 2D derivative tests
# ============================================================================


def test_cheb_deriv2d_gradient():
    """u = sin(πx)cos(πy): ∂u/∂x = π cos(πx)cos(πy), ∂u/∂y = -π sin(πx)sin(πy)."""
    N = 24
    grid = ChebyshevGrid2D.from_N_L(Nx=N, Ny=N, Lx=1.0, Ly=1.0)
    deriv = ChebyshevDerivative2D(grid)
    X, Y = grid.X
    u = jnp.sin(jnp.pi * X) * jnp.cos(jnp.pi * Y)
    du_dx, du_dy = deriv.gradient(u)
    assert jnp.allclose(
        du_dx, jnp.pi * jnp.cos(jnp.pi * X) * jnp.cos(jnp.pi * Y), atol=1e-5
    )
    assert jnp.allclose(
        du_dy, -jnp.pi * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y), atol=1e-5
    )


def test_cheb_deriv2d_laplacian():
    """u = sin(πx)sin(πy): ∇²u = -2π²sin(πx)sin(πy)."""
    N = 24
    grid = ChebyshevGrid2D.from_N_L(Nx=N, Ny=N, Lx=1.0, Ly=1.0)
    deriv = ChebyshevDerivative2D(grid)
    X, Y = grid.X
    u = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    lap = deriv.laplacian(u)
    expected = -2 * jnp.pi**2 * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    assert jnp.allclose(lap, expected, atol=1e-4), (
        f"max error = {jnp.abs(lap - expected).max()}"
    )


def test_cheb_deriv2d_divergence():
    """div(sin(πx), cos(πy)) = π cos(πx) - π sin(πy)."""
    N = 20
    grid = ChebyshevGrid2D.from_N_L(Nx=N, Ny=N, Lx=1.0, Ly=1.0)
    deriv = ChebyshevDerivative2D(grid)
    X, Y = grid.X
    vx = jnp.sin(jnp.pi * X)
    vy = jnp.cos(jnp.pi * Y)
    div = deriv.divergence(vx, vy)
    expected = jnp.pi * jnp.cos(jnp.pi * X) - jnp.pi * jnp.sin(jnp.pi * Y)
    assert jnp.allclose(div, expected, atol=1e-5)


def test_cheb_deriv2d_curl():
    """curl(cos(πy), sin(πx)) = π cos(πx) + π sin(πy)."""
    N = 20
    grid = ChebyshevGrid2D.from_N_L(Nx=N, Ny=N, Lx=1.0, Ly=1.0)
    deriv = ChebyshevDerivative2D(grid)
    X, Y = grid.X
    vx = jnp.cos(jnp.pi * Y)
    vy = jnp.sin(jnp.pi * X)
    curl = deriv.curl(vx, vy)
    # dvy/dx - dvx/dy = π cos(πx) - (-π sin(πy)) = π cos(πx) + π sin(πy)
    expected = jnp.pi * jnp.cos(jnp.pi * X) + jnp.pi * jnp.sin(jnp.pi * Y)
    assert jnp.allclose(curl, expected, atol=1e-5)


def test_cheb_deriv2d_advection_scalar():
    """(V·∇)q with V=(1,0): advection = ∂q/∂x = π cos(πx)."""
    N = 20
    grid = ChebyshevGrid2D.from_N_L(Nx=N, Ny=N, Lx=1.0, Ly=1.0)
    deriv = ChebyshevDerivative2D(grid)
    X, Y = grid.X
    vx = jnp.ones_like(X)
    vy = jnp.zeros_like(Y)
    q = jnp.sin(jnp.pi * X)
    adv = deriv.advection_scalar(vx, vy, q)
    expected = jnp.pi * jnp.cos(jnp.pi * X)
    assert jnp.allclose(adv, expected, atol=1e-5)
