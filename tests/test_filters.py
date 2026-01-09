import jax.numpy as jnp
import pytest

from spectraldiffx._src.filters import (
    SpectralFilter1D,
    SpectralFilter2D,
    SpectralFilter3D,
)
from spectraldiffx._src.grid import FourierGrid1D, FourierGrid2D, FourierGrid3D


@pytest.fixture
def grid1d():
    return FourierGrid1D.from_N_L(64, 2 * jnp.pi)


@pytest.fixture
def grid2d():
    return FourierGrid2D.from_N_L(32, 32, 2 * jnp.pi, 2 * jnp.pi)


@pytest.fixture
def grid3d():
    return FourierGrid3D.from_N_L(16, 16, 16, 2 * jnp.pi, 2 * jnp.pi, 2 * jnp.pi)


def test_filter1d_exponential(grid1d):
    filt = SpectralFilter1D(grid1d)
    # Define ones in spectral space
    u_hat = jnp.ones(grid1d.N, dtype=jnp.complex64)
    u_hat_f = filt.exponential_filter(u_hat, spectral=True)
    # DC mode should be unaffected (filter=1 at k=0)
    assert jnp.isclose(u_hat_f[0], 1.0)
    # Nyquist frequency should be dampened
    assert jnp.abs(u_hat_f[grid1d.N // 2]) < 1e-10


def test_filter2d_hyperviscosity(grid2d):
    filt = SpectralFilter2D(grid2d)
    u_hat = jnp.ones((grid2d.Ny, grid2d.Nx), dtype=jnp.complex64)
    dt, nu = 0.1, 1e-4
    u_hat_f = filt.hyperviscosity(u_hat, nu, dt)
    assert jnp.isclose(u_hat_f[0, 0], 1.0)
    assert jnp.all(jnp.abs(u_hat_f) <= 1.0)


def test_filter3d_shapes(grid3d):
    filt = SpectralFilter3D(grid3d)
    u_hat = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx), dtype=jnp.complex64)
    u_hat_f = filt.exponential_filter(u_hat)
    assert u_hat_f.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)
