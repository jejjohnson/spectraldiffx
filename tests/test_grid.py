import jax.numpy as jnp
import pytest

from spectraldiffx._src.grid import FourierGrid1D, FourierGrid2D, FourierGrid3D


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
