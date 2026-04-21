"""Tests for the new Chebyshev additions from the code-review follow-up.

Covers:
    * :class:`ChebyshevTransform1D`, :class:`ChebyshevTransform2D` (standalone)
    * :class:`ChebyshevPoissonSolver1D`, :class:`ChebyshevPoissonSolver2D`
    * :class:`ChebyshevHelmholtzSolver2D`
    * 1D Helmholtz solver with Neumann BCs
    * :func:`clenshaw_curtis_weights` + 1D / 2D integrators
    * :func:`dealias_product`
"""

from __future__ import annotations

import jax.numpy as jnp

from spectraldiffx import (
    ChebyshevGrid1D,
    ChebyshevGrid2D,
    ChebyshevHelmholtzSolver1D,
    ChebyshevHelmholtzSolver2D,
    ChebyshevPoissonSolver1D,
    ChebyshevPoissonSolver2D,
    ChebyshevTransform1D,
    ChebyshevTransform2D,
    cheb_dealias_product,
    clenshaw_curtis_integrate_1d,
    clenshaw_curtis_integrate_2d,
    clenshaw_curtis_weights,
)

# ---------------------------------------------------------------------------
# Standalone transforms
# ---------------------------------------------------------------------------


def test_cheb_transform1d_roundtrip_gl():
    grid = ChebyshevGrid1D.from_N_L(N=32, L=1.0)
    t = ChebyshevTransform1D(grid=grid)
    u = jnp.sin(jnp.pi * grid.x) + 0.25 * jnp.cos(3 * jnp.pi * grid.x)
    a = t.to_spectral(u)
    assert jnp.allclose(t.from_spectral(a), u, atol=1e-12)


def test_cheb_transform1d_roundtrip_gauss():
    grid = ChebyshevGrid1D.from_N_L(N=24, L=2.0, node_type="gauss")
    t = ChebyshevTransform1D(grid=grid)
    u = jnp.cos(jnp.pi * grid.x / grid.L)
    a = t.to_spectral(u)
    assert jnp.allclose(t.from_spectral(a), u, atol=1e-12)


def test_cheb_transform2d_roundtrip():
    grid = ChebyshevGrid2D.from_N_L(Nx=16, Ny=16, Lx=1.0, Ly=1.0)
    t = ChebyshevTransform2D(grid=grid)
    X, Y = grid.X
    u = jnp.sin(jnp.pi * X) * jnp.cos(jnp.pi * Y)
    a = t.to_spectral(u)
    assert jnp.allclose(t.from_spectral(a), u, atol=1e-12)


# ---------------------------------------------------------------------------
# 1D Poisson / Helmholtz + Neumann BCs
# ---------------------------------------------------------------------------


def test_cheb_poisson1d_manufactured_solution():
    grid = ChebyshevGrid1D.from_N_L(N=32, L=1.0)
    solver = ChebyshevPoissonSolver1D(grid=grid)
    x = grid.x
    f = -(jnp.pi**2) * jnp.sin(jnp.pi * x)  # u = sin(πx)
    u = solver.solve(f, bc_left=0.0, bc_right=0.0)
    assert jnp.allclose(u, jnp.sin(jnp.pi * x), atol=1e-12)


def test_cheb_helmholtz1d_neumann_manufactured_solution():
    """(d²/dx² − α) cos(πx) = −(π² + α) cos(πx), with u'(±1) = 0."""
    grid = ChebyshevGrid1D.from_N_L(N=32, L=1.0)
    solver = ChebyshevHelmholtzSolver1D(grid=grid)
    x = grid.x
    alpha = 2.0
    f = -(jnp.pi**2 + alpha) * jnp.cos(jnp.pi * x)
    u = solver.solve(f, alpha=alpha, bc_type="neumann", bc_left=0.0, bc_right=0.0)
    assert jnp.allclose(u, jnp.cos(jnp.pi * x), atol=1e-10)


def test_cheb_helmholtz1d_rejects_gauss_nodes():
    import pytest

    grid = ChebyshevGrid1D.from_N_L(N=16, L=1.0, node_type="gauss")
    solver = ChebyshevHelmholtzSolver1D(grid=grid)
    with pytest.raises(ValueError, match="gauss-lobatto"):
        solver.solve(jnp.zeros(16))


# ---------------------------------------------------------------------------
# 2D Poisson / Helmholtz
# ---------------------------------------------------------------------------


def test_cheb_poisson2d_manufactured_solution():
    """∇² [sin(πx) sin(πy)] = −2π² sin(πx) sin(πy), zero Dirichlet BCs."""
    grid = ChebyshevGrid2D.from_N_L(Nx=16, Ny=16, Lx=1.0, Ly=1.0)
    solver = ChebyshevPoissonSolver2D(grid=grid)
    X, Y = grid.X
    u_exact = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    f = -2 * jnp.pi**2 * u_exact
    u = solver.solve(f)
    assert jnp.allclose(u, u_exact, atol=1e-10)


def test_cheb_helmholtz2d_manufactured_solution():
    grid = ChebyshevGrid2D.from_N_L(Nx=16, Ny=16, Lx=1.0, Ly=1.0)
    solver = ChebyshevHelmholtzSolver2D(grid=grid)
    X, Y = grid.X
    alpha = 4.0
    u_exact = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    f = -(2 * jnp.pi**2 + alpha) * u_exact
    u = solver.solve(f, alpha=alpha)
    assert jnp.allclose(u, u_exact, atol=1e-10)


def test_cheb_poisson2d_nonhomogeneous_dirichlet():
    """BC values on the interior of each edge are honoured.

    Corners, where two edges meet with possibly inconsistent BCs, are
    resolved deterministically (left/right overwrite top/bottom in the
    current implementation) but are not part of the contract — the test
    only checks interior edge points.
    """
    grid = ChebyshevGrid2D.from_N_L(Nx=16, Ny=16, Lx=1.0, Ly=1.0)
    solver = ChebyshevPoissonSolver2D(grid=grid)
    Nxpts = grid.Nx + 1
    Nypts = grid.Ny + 1
    f = jnp.zeros((Nypts, Nxpts))
    u = solver.solve(f, bc_top=1.0)
    # Top row, interior columns
    assert jnp.allclose(u[0, 1:-1], 1.0, atol=1e-12)
    # Bottom row, interior columns
    assert jnp.allclose(u[-1, 1:-1], 0.0, atol=1e-12)
    # Left and right edges (interior rows) — zero BCs
    assert jnp.allclose(u[1:-1, 0], 0.0, atol=1e-12)
    assert jnp.allclose(u[1:-1, -1], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Clenshaw–Curtis quadrature
# ---------------------------------------------------------------------------


def test_cc_weights_sum_to_2L():
    for N in (4, 8, 16, 33):
        w = clenshaw_curtis_weights(N, L=1.0)
        assert float(jnp.sum(w)) == _approx(2.0, rel=1e-13, abs_=1e-13)
    for N in (8, 32):
        w = clenshaw_curtis_weights(N, L=2.5)
        assert float(jnp.sum(w)) == _approx(5.0, rel=1e-13, abs_=1e-13)


def test_cc_integrate1d_polynomial_exact():
    grid = ChebyshevGrid1D.from_N_L(N=16, L=1.0)
    x = grid.x
    # CC on N=16 GL nodes is exact for polynomials of degree ≤ 2N−1 = 31.
    f = 3 * x**5 - 2 * x**3 + x
    I = clenshaw_curtis_integrate_1d(grid, f)
    # ∫_{-1}^{1} (3x^5 - 2x^3 + x) dx = 0 (odd)
    assert abs(float(I)) < 1e-13


def test_cc_integrate1d_smooth_function():
    grid = ChebyshevGrid1D.from_N_L(N=32, L=1.0)
    f = jnp.exp(grid.x)
    I = clenshaw_curtis_integrate_1d(grid, f)
    assert abs(float(I) - float(jnp.e - 1 / jnp.e)) < 1e-12


def test_cc_integrate2d_smooth_function():
    grid = ChebyshevGrid2D.from_N_L(Nx=24, Ny=24, Lx=1.0, Ly=1.0)
    X, Y = grid.X
    I = clenshaw_curtis_integrate_2d(grid, jnp.exp(X + Y))
    I_exact = (jnp.e - 1 / jnp.e) ** 2
    assert abs(float(I) - float(I_exact)) < 1e-12


# ---------------------------------------------------------------------------
# Dealias product
# ---------------------------------------------------------------------------


def test_dealias_product_sin_cos():
    grid = ChebyshevGrid1D.from_N_L(N=32, L=1.0, dealias="2/3")
    x = grid.x
    u = jnp.sin(jnp.pi * x)
    v = jnp.cos(jnp.pi * x)
    uv = cheb_dealias_product(grid, u, v)
    assert jnp.allclose(uv, 0.5 * jnp.sin(2 * jnp.pi * x), atol=1e-10)


def test_dealias_product_passthrough_when_disabled():
    """With dealias=None the helper returns the naive product."""
    grid = ChebyshevGrid1D.from_N_L(N=16, L=1.0, dealias=None)
    x = grid.x
    u = jnp.sin(jnp.pi * x)
    v = jnp.cos(jnp.pi * x)
    uv = cheb_dealias_product(grid, u, v)
    assert jnp.allclose(uv, u * v, atol=1e-14)


def test_dealias_product_2d():
    grid = ChebyshevGrid2D.from_N_L(Nx=24, Ny=24, Lx=1.0, Ly=1.0, dealias="2/3")
    X, Y = grid.X
    u = jnp.sin(jnp.pi * X)
    v = jnp.cos(jnp.pi * Y)
    uv = cheb_dealias_product(grid, u, v)
    assert jnp.allclose(uv, u * v, atol=1e-10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _approx:
    def __init__(self, value, rel: float = 1e-6, abs_: float = 1e-12):
        self.value = value
        self.rel = rel
        self.abs_ = abs_

    def __eq__(self, other):
        return abs(other - self.value) <= max(self.rel * abs(self.value), self.abs_)
