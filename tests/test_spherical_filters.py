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


def test_spherical_filter1d_spectral_input_consistent():
    """
    SphericalFilter1D.exponential_filter(u, spectral=False) and
    (transform → filter → inverse-transform) must give the same result.
    """
    N = 32
    g = SphericalGrid1D.from_N_L(N, np.pi)
    filt = SphericalFilter1D(g)
    theta = g.x
    u = jnp.sin(theta) ** 2 + 0.1 * (1 - jnp.cos(theta) ** 2)

    u_filt_phys = filt.exponential_filter(u, spectral=False)

    c = g.transform(u)
    c_filt = filt.exponential_filter(c, spectral=True)
    u_filt_from_spec = g.transform(c_filt, inverse=True)

    assert jnp.allclose(u_filt_phys, u_filt_from_spec, atol=1e-14), (
        f"SphericalFilter1D spectral=True vs False: max diff = "
        f"{float(jnp.max(jnp.abs(u_filt_phys - u_filt_from_spec))):.2e}"
    )


def test_spherical_filter1d_hyperviscosity_dc_preserved():
    """
    SphericalFilter1D.hyperviscosity(): l=0 (mean) mode must be preserved.

    F(l=0) = exp(-nu * [0*(0+1)/R²]^(p/2) * dt) = exp(0) = 1.
    """
    N = 32
    g = SphericalGrid1D.from_N_L(N, np.pi)
    filt = SphericalFilter1D(g)
    u = jnp.ones(N)  # constant field → only l=0 mode
    u_filt = filt.hyperviscosity(u, nu_hyper=1e-3, dt=0.1)
    assert jnp.allclose(u_filt, u, atol=1e-13), (
        f"SphericalFilter1D hyperviscosity: constant field changed; "
        f"max diff = {float(jnp.max(jnp.abs(u_filt - u))):.2e}"
    )


def test_spherical_filter1d_hyperviscosity_high_modes_damped():
    """SphericalFilter1D.hyperviscosity(): the highest-l mode must be damped."""
    N = 32
    g = SphericalGrid1D.from_N_L(N, np.pi)
    filt = SphericalFilter1D(g)
    c_high = jnp.zeros(N).at[-1].set(1.0)
    c_filt = filt.hyperviscosity(c_high, nu_hyper=1e-4, dt=0.1, spectral=True)
    assert float(jnp.abs(c_filt[-1])) < 1.0, (
        "Highest-l mode should be damped by hyperviscosity"
    )


def test_spherical_filter2d_exponential_high_modes_damped():
    """
    SphericalFilter2D.exponential_filter(): highest-l mode must be near-zero.

    For a spectral coefficient at l=Ny-1 (highest degree):
    F(l_max) = exp(-alpha * 1^power) = exp(-36) ≈ 2.3e-16.
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    filt = SphericalFilter2D(g)

    u_hat_high = jnp.zeros((Ny, Nx), dtype=jnp.complex128).at[-1, 0].set(1.0)
    u_hat_filt = filt.exponential_filter(
        u_hat_high, alpha=36.0, power=16, spectral=True
    )

    assert float(jnp.abs(u_hat_filt[-1, 0])) < 1e-10, (
        f"SphericalFilter2D: highest-l mode should be near-zero after filtering; "
        f"got {float(jnp.abs(u_hat_filt[-1, 0])):.2e}"
    )


def test_spherical_filter2d_hyperviscosity_dc_preserved():
    """SphericalFilter2D.hyperviscosity(): l=0 (DC) mode must be preserved."""
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    filt = SphericalFilter2D(g)
    u = jnp.ones((Ny, Nx))
    u_filt = filt.hyperviscosity(u, nu_hyper=1e-3, dt=0.1)
    assert jnp.allclose(u_filt, u, atol=1e-12), (
        f"SphericalFilter2D hyperviscosity: constant field changed; "
        f"max diff = {float(jnp.max(jnp.abs(u_filt - u))):.2e}"
    )
