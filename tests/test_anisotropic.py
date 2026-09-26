"""Operators and solvers on non-square, anisotropic grids (gh-119).

Most tests use square grids with equal lengths, which cannot catch a swapped
axis or a length used on the wrong axis. Here every axis has a different
size and length: a 48 × 20 grid with Lx = 2π, Ly = 1, and an 8 × 24 × 40
grid with three different lengths. The test fields are single low Fourier
modes, so spectral derivatives are exact to round-off, and dealiasing
(which only removes high modes) does not affect them.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import spectraldiffx as sdx

LX, LY, LZ = 2 * np.pi, 1.0, 3.0
KX, KY, KZ = 2 * np.pi / LX * 1, 2 * np.pi / LY * 2, 2 * np.pi / LZ * 1


@pytest.fixture
def grid2d():
    return sdx.FourierGrid2D.from_N_L(Nx=48, Ny=20, Lx=LX, Ly=LY)


@pytest.fixture
def grid3d():
    return sdx.FourierGrid3D.from_N_L(Nz=8, Ny=24, Nx=40, Lz=LZ, Ly=LY, Lx=LX)


def _mode2d(grid):
    X, Y = grid.X
    return jnp.sin(KX * X) * jnp.cos(KY * Y), X, Y


def _mode3d(grid):
    Z, Y, X = grid.X  # 3-D grids use (z, y, x) order
    return jnp.sin(KX * X) * jnp.cos(KY * Y) * jnp.sin(KZ * Z), X, Y, Z


def _assert_close(got, expected, rtol=1e-10):
    got, expected = np.real(np.asarray(got)), np.asarray(expected)
    assert got.shape == expected.shape
    assert np.abs(got - expected).max() <= rtol * np.abs(expected).max()


def test_grid_axes_2d(grid2d):
    X, Y = grid2d.X
    assert X.shape == Y.shape == (20, 48)
    assert float(X[0, 1] - X[0, 0]) == pytest.approx(LX / 48)
    assert float(Y[1, 0] - Y[0, 0]) == pytest.approx(LY / 20)


def test_grid_axes_3d(grid3d):
    Z, Y, X = grid3d.X
    assert X.shape == (8, 24, 40)
    assert float(X[0, 0, 1] - X[0, 0, 0]) == pytest.approx(LX / 40)
    assert float(Y[0, 1, 0] - Y[0, 0, 0]) == pytest.approx(LY / 24)
    assert float(Z[1, 0, 0] - Z[0, 0, 0]) == pytest.approx(LZ / 8)


def test_gradient_and_laplacian_2d(grid2d):
    u, X, Y = _mode2d(grid2d)
    deriv = sdx.SpectralDerivative2D(grid2d)
    du_dx, du_dy = deriv.gradient(u)
    _assert_close(du_dx, KX * jnp.cos(KX * X) * jnp.cos(KY * Y))
    _assert_close(du_dy, -KY * jnp.sin(KX * X) * jnp.sin(KY * Y))
    _assert_close(deriv.laplacian(u), -(KX**2 + KY**2) * u)


def test_gradient_and_laplacian_3d(grid3d):
    u, X, Y, Z = _mode3d(grid3d)
    deriv = sdx.SpectralDerivative3D(grid3d)
    grads = deriv.gradient(u)
    expected = {
        "x": KX * jnp.cos(KX * X) * jnp.cos(KY * Y) * jnp.sin(KZ * Z),
        "y": -KY * jnp.sin(KX * X) * jnp.sin(KY * Y) * jnp.sin(KZ * Z),
        "z": KZ * jnp.sin(KX * X) * jnp.cos(KY * Y) * jnp.cos(KZ * Z),
    }
    # Like grid.X, the 3-D gradient is ordered (z, y, x).
    for got, key in zip(grads, "zyx", strict=True):
        _assert_close(got, expected[key])
    _assert_close(deriv.laplacian(u), -(KX**2 + KY**2 + KZ**2) * u)


@pytest.mark.parametrize("alpha", [0.0, 2.5])
def test_helmholtz_solver_2d(grid2d, alpha):
    psi, _, _ = _mode2d(grid2d)
    f = -(KX**2 + KY**2 + alpha) * psi
    _assert_close(sdx.SpectralHelmholtzSolver2D(grid2d).solve(f, alpha=alpha), psi)


@pytest.mark.parametrize("alpha", [0.0, 2.5])
def test_helmholtz_solver_3d(grid3d, alpha):
    psi, *_ = _mode3d(grid3d)
    f = -(KX**2 + KY**2 + KZ**2 + alpha) * psi
    _assert_close(sdx.SpectralHelmholtzSolver3D(grid3d).solve(f, alpha=alpha), psi)


def test_fd2_solver_anisotropic_spacing():
    """solve_helmholtz_fft on a 20 × 48 grid, dx ≠ dy: FD2 eigenfunction."""
    ny, nx, dx, dy = 20, 48, LX / 48, LY / 20
    X, Y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dy)
    psi = np.sin(KX * X) * np.cos(KY * Y)
    eig_x = -4 / dx**2 * np.sin(KX * dx / 2) ** 2
    eig_y = -4 / dy**2 * np.sin(KY * dy / 2) ** 2
    lam = 1.3
    f = (eig_x + eig_y - lam) * psi
    _assert_close(sdx.solve_helmholtz_fft(jnp.asarray(f), dx, dy, lam), psi)
