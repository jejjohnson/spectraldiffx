import jax.numpy as jnp
import pytest

from spectraldiffx._src.fourier.grid import FourierGrid1D, FourierGrid2D, FourierGrid3D


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


def test_fourier_grid_1d_k_dealias_zeros_above_cutoff():
    """
    FourierGrid1D.k_dealias must set |k| > k_max*2/3 to zero.

    For N=32, k_max = 16 * 2π/L, cutoff = k_max * 2/3 ≈ 10.67 * 2π/L.
    Wavenumbers with |k| > cutoff must be zeroed out.
    """
    N = 32
    L = 2 * jnp.pi
    grid = FourierGrid1D.from_N_L(N, L, dealias="2/3")
    k = grid.k
    k_d = grid.k_dealias

    k_max = float(jnp.max(jnp.abs(k)))
    cutoff = k_max * 2.0 / 3.0

    above_cutoff_mask = jnp.abs(k) > cutoff
    below_or_equal_mask = ~above_cutoff_mask

    # Modes above cutoff must be zeroed
    assert jnp.allclose(k_d[above_cutoff_mask], 0.0, atol=1e-15), (
        "k_dealias: modes above cutoff must be zero"
    )
    # Modes below or equal cutoff must be preserved
    assert jnp.allclose(k_d[below_or_equal_mask], k[below_or_equal_mask], atol=1e-15), (
        "k_dealias: modes at or below cutoff must be unchanged"
    )


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
