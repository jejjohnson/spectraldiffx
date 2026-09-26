import jax.numpy as jnp
import numpy as np
import pytest

from spectraldiffx._src.fourier.grid import FourierGrid1D, FourierGrid2D, FourierGrid3D
from spectraldiffx._src.fourier.operators import (
    SpectralDerivative1D,
    SpectralDerivative2D,
)


def test_fourier_grid_1d():
    """Test 1D Fourier grid properties and factory methods."""
    N, L = 16, 2.0 * jnp.pi
    grid = FourierGrid1D.from_N_L(N, L)
    grid.check_consistency()

    # Check shapes
    assert grid.x.shape == (N,)
    assert grid.k.shape == (N,)

    # Check spacing and origin
    assert jnp.isclose(grid.dx, L / N)
    assert grid.k[0] == 0.0

    # Test alternative factory
    grid_dx = FourierGrid1D.from_L_dx(L, L / N)
    assert grid_dx.N == N

    # Check dealiasing filter
    mask = grid.dealias_filter()
    assert mask.shape == (N,)
    assert mask[0] == 1.0  # DC component should always be kept


def test_fourier_grid_2d():
    """Test 2D Fourier grid with 'xy' indexing convention."""
    Nx, Ny = 16, 12
    Lx, Ly = 2.0, 1.0
    grid = FourierGrid2D.from_N_L(Nx, Ny, Lx, Ly)
    grid.check_consistency()

    # Check physical meshgrid shapes (indexing='xy' -> Ny, Nx)
    X, Y = grid.X
    assert X.shape == (Ny, Nx)
    assert Y.shape == (Ny, Nx)

    # Check spectral meshgrid and Laplacian kernel
    KX, KY = grid.KX
    assert KX.shape == (Ny, Nx)
    assert grid.K2.shape == (Ny, Nx)
    assert jnp.allclose(grid.K2, KX**2 + KY**2)
    assert grid.K2[0, 0] == 0.0

    # Test factory from spacing
    grid_sp = FourierGrid2D.from_N_dx(Nx, Ny, Lx / Nx, Ly / Ny)
    assert jnp.isclose(grid_sp.Lx, Lx)


def test_fourier_grid_3d():
    """Test 3D Fourier grid with 'ij' indexing convention."""
    Nz, Ny, Nx = 8, 16, 12
    Lz, Ly, Lx = 1.0, 2.0, 1.5
    grid = FourierGrid3D.from_N_L(Nz, Ny, Nx, Lz, Ly, Lx)
    grid.check_consistency()

    # Check physical meshgrid shapes (indexing='ij' -> Nz, Ny, Nx)
    Z, Y, X = grid.X
    assert Z.shape == (Nz, Ny, Nx)
    assert Y.shape == (Nz, Ny, Nx)
    assert X.shape == (Nz, Ny, Nx)

    # Check 1D wavenumber vectors
    assert grid.kz.shape == (Nz,)
    assert grid.ky.shape == (Ny,)
    assert grid.kx.shape == (Nx,)

    # Check spectral meshgrid and Laplacian
    KZ, KY, KX = grid.KX
    assert KZ.shape == (Nz, Ny, Nx)
    assert grid.K2.shape == (Nz, Ny, Nx)
    assert jnp.allclose(grid.K2, KX**2 + KY**2 + KZ**2)

    # Verify dealiasing filter
    mask = grid.dealias_filter()
    assert mask.shape == (Nz, Ny, Nx)


def test_grid_invalid_n():
    """Ensure non-integer N detection raises error."""
    with pytest.raises(ValueError):
        FourierGrid1D.from_L_dx(L=1.0, dx=0.3)  # N = 3.33...


# --- k_dealias correctness tests ---


def _alias_energy_1d(N):
    """Energy the 2/3 rule lets through at the alias of the largest kept mode.

    For the largest kept mode K, cos(Kx)² has components at 0 and ±2K. On an
    N-point grid, 2K > N/2 aliases to 2K − N. Orszag's rule is alias-free iff
    that alias is removed by the mask, i.e. 3K < N (gh-88, gh-119).
    """
    grid = FourierGrid1D.from_N_L(N, 2 * jnp.pi, dealias="2/3")
    mask = np.asarray(grid.dealias_filter())
    n = np.fft.fftfreq(N, d=1.0 / N)
    K = int(np.abs(n[mask > 0]).max())
    u = jnp.cos(K * grid.x)
    v = SpectralDerivative1D(grid).apply_dealias(u * u)
    alias = (2 * K - N) % N
    return float(np.abs(np.fft.fft(np.asarray(v))[alias]) / N)


@pytest.mark.parametrize("N", [30, 32, 48, 63, 64, 96])
def test_fourier_grid_1d_dealias_is_alias_free(N):
    assert _alias_energy_1d(N) < 1e-14


def test_fourier_grid_1d_k_dealias_matches_filter():
    """k_dealias keeps exactly the modes dealias_filter keeps (N = 48)."""
    grid = FourierGrid1D.from_N_L(48, 2 * jnp.pi, dealias="2/3")
    kept = np.asarray(grid.dealias_filter()) > 0
    k, k_d = np.asarray(grid.k), np.asarray(grid.k_dealias)
    assert np.array_equal(k_d[kept], k[kept])
    assert np.all(k_d[~kept] == 0.0)
    assert kept.sum() == 31  # |n| <= 15: 3 * 16 = 48 is not < 48


def _kept_1d(N):
    n = np.fft.fftfreq(N, d=1.0 / N)
    return 3 * np.abs(n) < N


def test_fourier_grid_2d_dealias_filter_is_per_axis_strict():
    grid = FourierGrid2D.from_N_L(Nx=48, Ny=32, Lx=2 * jnp.pi, Ly=1.0, dealias="2/3")
    expected = np.outer(_kept_1d(32), _kept_1d(48))
    assert np.array_equal(np.asarray(grid.dealias_filter()) > 0, expected)


def test_fourier_grid_3d_dealias_filter_is_per_axis_strict():
    grid = FourierGrid3D.from_N_L(
        Nz=12, Ny=32, Nx=48, Lz=1.0, Ly=2.0, Lx=2 * jnp.pi, dealias="2/3"
    )
    expected = (
        _kept_1d(12)[:, None, None]
        & _kept_1d(32)[None, :, None]
        & _kept_1d(48)[None, None, :]
    )
    assert np.array_equal(np.asarray(grid.dealias_filter()) > 0, expected)


def test_fourier_2d_dealiased_product_is_alias_free():
    """apply_dealias(u·u) on 48 × 30 leaves nothing at the aliases of the
    largest kept modes along each axis."""
    grid = FourierGrid2D.from_N_L(Nx=48, Ny=30, Lx=2 * jnp.pi, Ly=2 * jnp.pi)
    X, Y = grid.X
    Kx, Ky = 15, 9  # largest kept: 3 * 16 = 48, 3 * 10 = 30 are removed
    u = jnp.cos(Kx * X) * jnp.cos(Ky * Y)
    v_hat = np.asarray(grid.transform(SpectralDerivative2D(grid).apply_dealias(u * u)))
    ay, ax = (2 * Ky - 30) % 30, (2 * Kx - 48) % 48
    assert np.abs(v_hat[:, ax]).max() / v_hat.size < 1e-14
    assert np.abs(v_hat[ay, :]).max() / v_hat.size < 1e-14


def test_fourier_grid_1d_k_dealias_none_unchanged():
    """With dealias=None, k_dealias must equal k (no modes zeroed out)."""
    N = 32
    L = 2 * jnp.pi
    grid = FourierGrid1D.from_N_L(N, L, dealias=None)
    k = grid.k
    k_d = grid.k_dealias
    assert jnp.allclose(k_d, k, atol=1e-15), "With dealias=None, k_dealias must equal k"


def test_fourier_grid_1d_k_dealias_dc_always_zero():
    """
    The DC mode (k=0) in k_dealias must always be zero (the 0th wavenumber
    is zero in the FFT convention, so the dealiased value is also zero).
    """
    N = 32
    grid = FourierGrid1D.from_N_L(N, 2 * jnp.pi, dealias="2/3")
    k_d = grid.k_dealias
    assert float(k_d[0]) == 0.0, "k_dealias[0] (DC mode) must be zero"
