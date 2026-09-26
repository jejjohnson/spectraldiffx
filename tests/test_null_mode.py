"""Null-mode (zero_mean) policy of the elliptic solvers (gh-92).

One rule everywhere: a mode whose denominator is zero (the constant null
mode at α = 0) is set to zero, never to an arbitrary value. The default
``zero_mean=None`` zeroes the mean only when it is undefined, so a
Helmholtz solve with α > 0 recovers a field with a non-zero mean.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import spectraldiffx as sdx
from spectraldiffx import (
    solve_helmholtz_2d,
    solve_helmholtz_dct1,
    solve_helmholtz_dct1_1d,
)


def _sphere():
    grid = sdx.SphericalGrid2D.from_N_L(Nx=32, Ny=16, Lx=2 * np.pi, Ly=np.pi)
    phi, theta = grid.X
    return grid, theta, phi


# --- A. spherical Helmholtz keeps the mean for alpha > 0 ---------------------


def test_spherical_helmholtz_default_keeps_mean():
    grid, theta, phi = _sphere()
    deriv = sdx.SphericalDerivative2D(grid)
    phi_true = 2.0 + jnp.sin(theta) ** 2 * jnp.cos(2 * phi)  # mean 2
    alpha = 0.7
    f = deriv.laplacian(phi_true) - alpha * phi_true
    got = sdx.SphericalHelmholtzSolver(grid).solve(f, alpha=alpha)
    assert float(jnp.abs(jnp.real(got) - phi_true).max()) < 1e-12


def test_spherical_helmholtz_zero_mean_true_is_explicit():
    grid, theta, phi = _sphere()
    deriv = sdx.SphericalDerivative2D(grid)
    phi_true = 2.0 + jnp.sin(theta) ** 2 * jnp.cos(2 * phi)
    f = deriv.laplacian(phi_true) - 0.7 * phi_true
    got = sdx.SphericalHelmholtzSolver(grid).solve(f, alpha=0.7, zero_mean=True)
    expected = jnp.sin(theta) ** 2 * jnp.cos(2 * phi)
    assert float(jnp.abs(jnp.real(got) - expected).max()) < 1e-12


# --- B. alpha = 0 never returns -mean(f) -------------------------------------


def test_spherical_poisson_rejects_keeping_undefined_mean():
    grid, theta, _ = _sphere()
    with pytest.raises(ValueError, match="undefined"):
        sdx.SphericalPoissonSolver(grid).solve(1.0 + jnp.cos(theta), zero_mean=False)
    with pytest.raises(ValueError, match="undefined"):
        sdx.SphericalHelmholtzSolver(grid).solve(
            jnp.cos(theta), alpha=0.0, zero_mean=False
        )


def test_spherical_poisson_zeroes_undefined_mean():
    grid, theta, _ = _sphere()
    got = sdx.SphericalPoissonSolver(grid).solve(1.0 + jnp.cos(theta))
    w = np.asarray(grid.weights)
    assert abs(float(np.sum(w * np.real(np.asarray(got))))) < 1e-12


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_fourier_helmholtz_classes_null_mode(dim):
    L = 2 * np.pi
    if dim == 1:
        grid = sdx.FourierGrid1D.from_N_L(16, L)
        solver = sdx.SpectralHelmholtzSolver1D(grid)
        x = grid.x
        mode = jnp.sin(x)
        eig = 1.0
    elif dim == 2:
        grid = sdx.FourierGrid2D.from_N_L(Nx=16, Ny=12, Lx=L, Ly=L)
        solver = sdx.SpectralHelmholtzSolver2D(grid)
        X, Y = grid.X
        mode = jnp.sin(X) * jnp.cos(Y)
        eig = 2.0
    else:
        grid = sdx.FourierGrid3D.from_N_L(Nz=8, Ny=10, Nx=12, Lz=L, Ly=L, Lx=L)
        solver = sdx.SpectralHelmholtzSolver3D(grid)
        Z, Y, X = grid.X
        mode = jnp.sin(X) * jnp.cos(Y) * jnp.cos(Z)
        eig = 3.0
    # alpha > 0, default: the mean (0.5) is recovered.
    alpha = 1.5
    psi = 0.5 + mode
    f = -alpha * 0.5 + (-eig - alpha) * mode
    np.testing.assert_allclose(
        np.asarray(solver.solve(f, alpha=alpha)), np.asarray(psi), atol=1e-12
    )
    # alpha = 0: the mean of f is discarded and psi has zero mean (not -mean(f)).
    got = solver.solve(2.0 + (-eig) * mode)
    np.testing.assert_allclose(np.asarray(got), np.asarray(mode), atol=1e-12)
    with pytest.raises(ValueError, match="undefined"):
        solver.solve(mode, zero_mean=False)


# --- C. DCT-I: the gauge is the trapezoid mean --------------------------------


def _dct1_matrix(n, h):
    A = -2.0 * np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1)
    A[0, 1] = A[-1, -2] = 2.0
    return A / h**2


def _trapezoid(n):
    w = np.ones(n)
    w[0] = w[-1] = 0.5
    return w


def test_dct1_1d_trapezoid_gauge_with_asymmetric_rhs():
    n, h = 16, 0.3
    A, w = _dct1_matrix(n, h), _trapezoid(n)
    x_true = np.random.default_rng(0).standard_normal(n)  # asymmetric
    f = A @ x_true  # in the range of A: w·f = 0
    assert abs(w @ f) < 1e-10 * np.abs(f).max()
    psi = np.asarray(solve_helmholtz_dct1_1d(jnp.asarray(f), h))
    assert np.abs(A @ psi - f).max() < 1e-10 * np.abs(f).max()
    assert abs(w @ psi) < 1e-12 * np.abs(psi).max()  # trapezoid gauge
    # An incompatible RHS is replaced by f - (w·f / w·1): documented.
    g = f + 1.0
    psi_g = np.asarray(solve_helmholtz_dct1_1d(jnp.asarray(g), h))
    c = (w @ g) / w.sum()
    assert np.abs(A @ psi_g - (g - c)).max() < 1e-10 * np.abs(g).max()


def test_dct1_2d_trapezoid_gauge_with_asymmetric_rhs():
    ny, nx, dy, dx = 9, 12, 0.4, 0.3
    A = np.kron(_dct1_matrix(ny, dy), np.eye(nx)) + np.kron(
        np.eye(ny), _dct1_matrix(nx, dx)
    )
    w = np.outer(_trapezoid(ny), _trapezoid(nx)).ravel()
    x_true = np.random.default_rng(1).standard_normal(ny * nx)
    f = A @ x_true
    psi = np.asarray(
        solve_helmholtz_dct1(jnp.asarray(f.reshape(ny, nx)), dx, dy)
    ).ravel()
    assert np.abs(A @ psi - f).max() < 1e-10 * np.abs(f).max()
    assert abs(w @ psi) < 1e-12 * np.abs(psi).max()
    # Same gauge through solve_helmholtz_2d with "neumann" on both axes.
    psi2 = np.asarray(
        solve_helmholtz_2d(jnp.asarray(f.reshape(ny, nx)), dx, dy, "neumann", "neumann")
    ).ravel()
    np.testing.assert_allclose(psi2, psi, atol=1e-12 * np.abs(psi).max())
