"""
Tests for SphericalFilter1D and SphericalFilter2D.
"""

import jax.numpy as jnp
import numpy as np

from spectraldiffx._src.spherical.filters import SphericalFilter1D, SphericalFilter2D
from spectraldiffx._src.spherical.grid import SphericalGrid1D, SphericalGrid2D


def test_spherical_filter1d_dc_preserved():
    """
    A constant field has only the l=0 mode.  The exponential filter with any
    (alpha, power) should preserve it (F(0) = exp(0) = 1).
    """
    N = 16
    g = SphericalGrid1D.from_N_L(N, np.pi)
    f = SphericalFilter1D(grid=g)
    u = jnp.ones(N)
    u_f = f.exponential_filter(u)
    assert jnp.allclose(u_f, u, atol=1e-10)


def test_spherical_filter1d_high_modes_damped():
    """High-l modes should be reduced in amplitude after exponential filtering."""
    N = 32
    g = SphericalGrid1D.from_N_L(N, np.pi)
    f = SphericalFilter1D(grid=g)
    c = jnp.zeros(N).at[-1].set(1.0)  # only highest mode
    c_f = f.exponential_filter(c, spectral=True)
    assert float(jnp.abs(c_f[-1])) < float(jnp.abs(c[-1]))


def test_spherical_filter1d_shape_preserved():
    """Output shape must match input shape."""
    N = 16
    g = SphericalGrid1D.from_N_L(N, np.pi)
    f = SphericalFilter1D(grid=g)
    u = jnp.ones(N)
    assert f.exponential_filter(u).shape == (N,)
    assert f.hyperviscosity(u, nu_hyper=1e-4, dt=0.1).shape == (N,)


def test_spherical_filter2d_dc_preserved():
    """Constant 2D field should be unchanged by the exponential filter."""
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    f = SphericalFilter2D(grid=g)
    u = jnp.ones((Ny, Nx))
    u_f = f.exponential_filter(u)
    assert jnp.allclose(u_f, u, atol=1e-10)


def test_spherical_filter2d_shape_preserved():
    """Output shape must match input shape."""
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    f = SphericalFilter2D(grid=g)
    u = jnp.ones((Ny, Nx))
    assert f.exponential_filter(u).shape == (Ny, Nx)
    assert f.hyperviscosity(u, nu_hyper=1e-4, dt=0.1).shape == (Ny, Nx)
