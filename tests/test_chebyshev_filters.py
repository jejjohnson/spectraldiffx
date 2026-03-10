"""
Tests for ChebyshevFilter1D and ChebyshevFilter2D.
"""

import jax.numpy as jnp

from spectraldiffx._src.chebyshev.filters import ChebyshevFilter1D, ChebyshevFilter2D
from spectraldiffx._src.chebyshev.grid import ChebyshevGrid1D, ChebyshevGrid2D

# ============================================================================
# 1D Filter tests
# ============================================================================


def test_cheb_filter1d_shape_preserved():
    """Output shape matches input shape."""
    N = 16
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    filt = ChebyshevFilter1D(grid)
    u = jnp.ones(N + 1)
    u_f = filt.exponential_filter(u)
    assert u_f.shape == u.shape


def test_cheb_filter1d_dc_preserved():
    """Constant field is (approximately) unchanged by the exponential filter.

    For a constant u = c, only the k=0 Chebyshev mode is non-zero.
    F(0) = exp(0) = 1, so the filtered field should equal the original.
    """
    N = 16
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    filt = ChebyshevFilter1D(grid)
    u = jnp.ones(N + 1) * 3.7
    u_f = filt.exponential_filter(u)
    assert jnp.allclose(u_f, u, atol=1e-5), (
        f"Constant field changed: max error = {jnp.abs(u_f - u).max()}"
    )


def test_cheb_filter1d_high_modes_damped():
    """High-frequency Chebyshev modes are significantly damped."""
    N = 16
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    filt = ChebyshevFilter1D(grid)

    # Build a field with energy only in the highest mode (T_N)
    x = grid.x
    u_high = jnp.cos(N * jnp.arccos(x / 1.0))  # T_N(x/L) — uses arccos

    u_f = filt.exponential_filter(u_high, alpha=36.0, power=16)
    # The amplitude should be reduced compared to the original
    assert jnp.abs(u_f).max() < jnp.abs(u_high).max()


def test_cheb_filter1d_spectral_input():
    """spectral=True: input is treated as Chebyshev coefficients."""
    N = 16
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    filt = ChebyshevFilter1D(grid)

    # Build coefficients: put energy in k=0 only
    a = jnp.zeros(N + 1).at[0].set(1.0)
    a_f = filt.exponential_filter(a, spectral=True)
    # k=0 coefficient should be unchanged (F(0)=1), output is still coefficients
    assert jnp.isclose(a_f[0], 1.0, atol=1e-6)
    assert a_f.shape == a.shape


def test_cheb_filter1d_hyperviscosity_shape():
    """Hyperviscosity filter preserves output shape."""
    N = 16
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    filt = ChebyshevFilter1D(grid)
    u = jnp.sin(jnp.pi * grid.x)
    u_f = filt.hyperviscosity(u, nu_hyper=1e-4, dt=0.01)
    assert u_f.shape == u.shape


def test_cheb_filter1d_hyperviscosity_dc_preserved():
    """Hyperviscosity at k=0 gives F(0)=1: constant field unchanged."""
    N = 16
    grid = ChebyshevGrid1D.from_N_L(N=N, L=1.0)
    filt = ChebyshevFilter1D(grid)
    u = jnp.ones(N + 1) * 2.5
    u_f = filt.hyperviscosity(u, nu_hyper=1e-3, dt=0.01)
    assert jnp.allclose(u_f, u, atol=1e-5)


# ============================================================================
# 2D Filter tests
# ============================================================================


def test_cheb_filter2d_shape_preserved():
    """2D filter preserves output shape."""
    Nx, Ny = 12, 10
    grid = ChebyshevGrid2D.from_N_L(Nx=Nx, Ny=Ny, Lx=1.0, Ly=1.0)
    filt = ChebyshevFilter2D(grid)
    u = jnp.ones((Ny + 1, Nx + 1))
    u_f = filt.exponential_filter(u)
    assert u_f.shape == u.shape


def test_cheb_filter2d_dc_preserved():
    """2D constant field is unchanged by the exponential filter."""
    Nx, Ny = 10, 10
    grid = ChebyshevGrid2D.from_N_L(Nx=Nx, Ny=Ny, Lx=1.0, Ly=1.0)
    filt = ChebyshevFilter2D(grid)
    u = jnp.ones((Ny + 1, Nx + 1)) * 4.2
    u_f = filt.exponential_filter(u)
    assert jnp.allclose(u_f, u, atol=1e-4), f"max error = {jnp.abs(u_f - u).max()}"


def test_cheb_filter2d_hyperviscosity_dc_preserved():
    """
    ChebyshevFilter2D.hyperviscosity(): the k=0 coefficient must be unchanged.

    F(kx=0, ky=0) = exp(-nu * 0^power * dt) = 1.
    A constant field has only the k=0 Chebyshev mode and should pass through.
    """
    N = 8
    grid = ChebyshevGrid2D.from_N_L(Nx=N, Ny=N, Lx=1.0, Ly=1.0)
    filt = ChebyshevFilter2D(grid)
    n_pts = N + 1  # Gauss-Lobatto has N+1 points
    u = jnp.ones((n_pts, n_pts))
    u_filt = filt.hyperviscosity(u, nu_hyper=1e-3, dt=0.1)
    assert jnp.allclose(u_filt, u, atol=1e-13), (
        f"Constant field should be unchanged by hyperviscosity filter; "
        f"max diff = {float(jnp.max(jnp.abs(u_filt - u))):.2e}"
    )


def test_cheb_filter2d_hyperviscosity_high_modes_damped():
    """ChebyshevFilter2D.hyperviscosity(): high-index coefficients are reduced."""
    N = 8
    grid = ChebyshevGrid2D.from_N_L(Nx=N, Ny=N, Lx=1.0, Ly=1.0)
    filt = ChebyshevFilter2D(grid)
    n_modes = N + 1  # Gauss-Lobatto

    a_high = jnp.zeros((n_modes, n_modes)).at[-1, -1].set(1.0)
    a_filt = filt.hyperviscosity(a_high, nu_hyper=1e-3, dt=0.1, spectral=True)
    assert float(jnp.abs(a_filt[-1, -1])) < 1.0, (
        "Highest Chebyshev coefficient should be damped by hyperviscosity"
    )


def test_cheb_filter2d_exponential_spectral_input_consistent():
    """
    ChebyshevFilter2D.exponential_filter(spectral=True) must match
    transforming to spectral space, filtering, then transforming back.
    """
    N = 8
    grid = ChebyshevGrid2D.from_N_L(Nx=N, Ny=N, Lx=1.0, Ly=1.0)
    filt = ChebyshevFilter2D(grid)
    X, Y = grid.X
    u = jnp.sin(jnp.pi * X / 2) * jnp.cos(jnp.pi * Y / 2)

    u_filt_phys = filt.exponential_filter(u, spectral=False)

    a = grid.transform(u)
    a_filt = filt.exponential_filter(a, spectral=True)
    u_filt_from_spec = grid.transform(a_filt, inverse=True)

    assert jnp.allclose(u_filt_phys, u_filt_from_spec, atol=1e-12), (
        f"exponential_filter spectral=True vs False: max diff = "
        f"{float(jnp.max(jnp.abs(u_filt_phys - u_filt_from_spec))):.2e}"
    )
