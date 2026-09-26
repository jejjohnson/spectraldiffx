import jax.numpy as jnp
import numpy as np
import pytest

from spectraldiffx._src.fourier.filters import (
    SpectralFilter1D,
    SpectralFilter2D,
    SpectralFilter3D,
)
from spectraldiffx._src.fourier.grid import FourierGrid1D, FourierGrid2D, FourierGrid3D


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


def test_filter1d_hyperviscosity_dc_preserved(grid1d):
    """
    Hyperviscosity filter: F(k=0) = exp(-nu * 0^power * dt) = 1.
    The DC mode (k=0) must be exactly preserved.
    """
    filt = SpectralFilter1D(grid1d)
    u_hat = jnp.zeros(grid1d.N, dtype=jnp.complex128).at[0].set(1.0)
    u_hat_f = filt.hyperviscosity(u_hat, nu_hyper=1e-4, dt=0.1, spectral=True)
    assert jnp.isclose(jnp.abs(u_hat_f[0]), 1.0, atol=1e-15), (
        f"DC mode should be preserved by hyperviscosity; got {float(jnp.abs(u_hat_f[0])):.6f}"
    )


def test_filter1d_hyperviscosity_monotone(grid1d):
    """
    Hyperviscosity filter is monotonically decreasing: higher |k| → more damping.

    F(k) = exp(-nu * |k|^p * dt), so higher k → smaller F(k).
    """
    filt = SpectralFilter1D(grid1d)

    amplitudes = []
    for k in [0, 1, 3, 5, 10, 20]:
        u_hat = jnp.zeros(grid1d.N, dtype=jnp.complex128).at[k].set(1.0)
        u_filt = filt.hyperviscosity(u_hat, nu_hyper=1e-4, dt=0.1, spectral=True)
        amplitudes.append(float(jnp.abs(u_filt[k])))

    # All amplitudes should be <= 1.0
    assert all(a <= 1.0 for a in amplitudes), (
        f"Hyperviscosity must not amplify: {amplitudes}"
    )
    # Amplitudes should decrease with k (higher k → more damping → smaller amplitude)
    for i in range(len(amplitudes) - 1):
        assert amplitudes[i + 1] <= amplitudes[i], (
            f"Hyperviscosity not monotone: F(k={[0, 1, 3, 5, 10, 20][i]})={amplitudes[i]:.6f} "
            f"> F(k={[0, 1, 3, 5, 10, 20][i + 1]})={amplitudes[i + 1]:.6f}"
        )


def test_filter3d_exponential_dc_preserved(grid3d):
    """3D exponential filter: F(k=0) = 1 — the DC mode must be unaffected."""
    filt = SpectralFilter3D(grid3d)
    u_hat = (
        jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx), dtype=jnp.complex128)
        .at[0, 0, 0]
        .set(1.0)
    )
    u_hat_f = filt.exponential_filter(u_hat, spectral=True)
    assert jnp.isclose(jnp.abs(u_hat_f[0, 0, 0]), 1.0, atol=1e-15), (
        "3D exponential filter: DC mode must be preserved"
    )


def test_filter3d_exponential_high_modes_damped(grid3d):
    """
    3D exponential filter: higher wavenumber magnitude → more damping.

    Low mode (index 2) should be damped less than high mode (index 8).
    """
    filt = SpectralFilter3D(grid3d)

    u_low = (
        jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx), dtype=jnp.complex128)
        .at[2, 0, 0]
        .set(1.0)
    )
    u_high = (
        jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx), dtype=jnp.complex128)
        .at[8, 0, 0]
        .set(1.0)
    )

    f_low = float(jnp.abs(filt.exponential_filter(u_low, spectral=True)[2, 0, 0]))
    f_high = float(jnp.abs(filt.exponential_filter(u_high, spectral=True)[8, 0, 0]))

    assert f_high < f_low, (
        f"3D filter: high mode (k=8) should be damped more than low mode (k=2), "
        f"got F(2)={f_low:.4f} and F(8)={f_high:.4f}"
    )


def test_filter3d_hyperviscosity_dc_preserved(grid3d):
    """3D hyperviscosity filter: DC mode must be exactly preserved."""
    filt = SpectralFilter3D(grid3d)
    u_hat = (
        jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx), dtype=jnp.complex128)
        .at[0, 0, 0]
        .set(1.0)
    )
    u_hat_f = filt.hyperviscosity(u_hat, nu_hyper=1e-4, dt=0.1, spectral=True)
    assert jnp.isclose(jnp.abs(u_hat_f[0, 0, 0]), 1.0, atol=1e-15), (
        "3D hyperviscosity: DC mode must be preserved"
    )


# ---------------------------------------------------------------------------
# Tensor-product exponential filter (gh-89)
# ---------------------------------------------------------------------------

_ALPHA = 36.0


def _nyquist_checkerboard(grid, axis):
    """(-1)^j along one axis: the pure axis-Nyquist mode."""
    idx = np.indices(np.asarray(grid.X[0]).shape)[axis]
    return jnp.asarray((-1.0) ** idx)


@pytest.mark.parametrize(
    "make",
    [
        lambda: FourierGrid2D.from_N_L(Nx=64, Ny=64, Lx=2 * np.pi, Ly=2 * np.pi),
        lambda: FourierGrid2D.from_N_L(Nx=64, Ny=16, Lx=2 * np.pi, Ly=1.0),
        lambda: FourierGrid3D.from_N_L(
            Nz=32, Ny=32, Nx=32, Lz=2 * np.pi, Ly=2 * np.pi, Lx=2 * np.pi
        ),
    ],
    ids=["64x64", "64x16", "32x32x32"],
)
def test_exponential_filter_damps_every_axis_nyquist(make):
    grid = make()
    filt = (SpectralFilter2D if len(grid.X) == 2 else SpectralFilter3D)(grid=grid)
    for axis in range(len(grid.X)):
        u = _nyquist_checkerboard(grid, axis)
        # F(axis Nyquist) = exp(-alpha): read it off the spectral multiplier.
        u_hat = grid.transform(u)
        out_hat = filt.exponential_filter(u_hat, alpha=_ALPHA, spectral=True)
        peak = np.unravel_index(np.argmax(np.abs(np.asarray(u_hat))), u_hat.shape)
        ratio = complex(out_hat[peak] / u_hat[peak])
        assert abs(ratio - np.exp(-_ALPHA)) < 1e-12
        # A checkerboard along one axis is removed after one application.
        out = filt.exponential_filter(u, alpha=_ALPHA)
        assert float(jnp.linalg.norm(out) / jnp.linalg.norm(u)) < 1e-12


def test_exponential_filter_keeps_mean_and_multiplies():
    grid = FourierGrid2D.from_N_L(Nx=32, Ny=24, Lx=2 * np.pi, Ly=1.0)
    filt = SpectralFilter2D(grid=grid)
    u = jnp.asarray(np.random.default_rng(0).standard_normal((24, 32)))
    once = filt.exponential_filter(u)
    assert abs(float(jnp.mean(once) - jnp.mean(u))) < 1e-14
    # Applying the multiplier twice equals applying F² (alpha doubled).
    twice = filt.exponential_filter(once)
    squared = filt.exponential_filter(u, alpha=2 * 36.0)
    np.testing.assert_allclose(np.asarray(twice), np.asarray(squared), atol=1e-12)
