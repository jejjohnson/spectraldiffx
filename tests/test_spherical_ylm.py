"""Spherical operators on every real spherical harmonic with l ≤ 6 (gh-119).

The other spherical tests use zonal or even-m fields such as cos θ or
sin²θ cos φ, which cannot see errors that only affect odd zonal wavenumber
m (#97). Here every Re Y_l^m with l ≤ 6 and 0 ≤ m ≤ l, from
``scipy.special.sph_harm_y``, is tested at radius R = 1 and at the Earth's
radius. The radius is set through ``Ly = πR``; ``Lx`` stays 2π, and
``grid.X`` is (φ, θ) in radians either way.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import sph_harm_y

import spectraldiffx as sdx

LMAX = 6
MODES = [(l, m) for l in range(LMAX + 1) for m in range(l + 1)]
RADII = [1.0, 6.371e6]


def _mode_id(lm):
    return f"l{lm[0]}m{lm[1]}"


@pytest.fixture(params=RADII, ids=["R=1", "R=earth"])
def sphere(request):
    radius = request.param
    grid = sdx.SphericalGrid2D.from_N_L(Nx=32, Ny=16, Lx=2 * np.pi, Ly=np.pi * radius)
    phi, theta = (np.asarray(a) for a in grid.X)
    return grid, radius, theta, phi


def _ylm(l, m, theta, phi):
    return np.real(sph_harm_y(l, m, theta, phi))


def _assert_close(got, expected, scale, rtol):
    got = np.real(np.asarray(got))
    assert np.abs(got - expected).max() <= rtol * scale


@pytest.mark.parametrize("lm", MODES, ids=_mode_id)
def test_transform_roundtrip(sphere, lm):
    grid, _, theta, phi = sphere
    y = _ylm(*lm, theta, phi)
    u = jnp.asarray(y)
    _assert_close(
        grid.transform(grid.transform(u), inverse=True), y, np.abs(y).max(), 1e-12
    )


@pytest.mark.parametrize("lm", MODES, ids=_mode_id)
def test_laplacian_eigenvalue(sphere, lm):
    grid, radius, theta, phi = sphere
    l, m = lm
    y = _ylm(l, m, theta, phi)
    eig = -l * (l + 1) / radius**2
    lap = sdx.SphericalDerivative2D(grid).laplacian(jnp.asarray(y))
    _assert_close(lap, eig * y, max(abs(eig), 1 / radius**2) * np.abs(y).max(), 1e-10)


@pytest.mark.parametrize("alpha_scale", [0.0, 3.0])
@pytest.mark.parametrize("lm", [lm for lm in MODES if lm[0] > 0], ids=_mode_id)
def test_helmholtz_solver_inverts_laplacian(sphere, lm, alpha_scale):
    grid, radius, theta, phi = sphere
    l, m = lm
    y = _ylm(l, m, theta, phi)
    alpha = alpha_scale / radius**2
    f = (-l * (l + 1) / radius**2 - alpha) * y
    solver = sdx.SphericalHelmholtzSolver(grid)
    _assert_close(solver.solve(jnp.asarray(f), alpha=alpha), y, np.abs(y).max(), 1e-10)


@pytest.mark.parametrize("lm", MODES, ids=_mode_id)
def test_gradient_phi(sphere, lm):
    grid, radius, theta, phi = sphere
    l, m = lm
    y = _ylm(l, m, theta, phi)
    # ∂φ Re(Y) = Re(i m Y); the φ component carries 1 / (R sin θ).
    expected = np.real(1j * m * sph_harm_y(l, m, theta, phi)) / (radius * np.sin(theta))
    _, grad_phi = sdx.SphericalDerivative2D(grid).gradient(jnp.asarray(y))
    _assert_close(grad_phi, expected, max(np.abs(expected).max(), 1 / radius), 1e-10)


@pytest.mark.parametrize(
    "lm",
    [
        pytest.param(
            lm,
            id=_mode_id(lm),
            marks=pytest.mark.xfail(
                lm[1] % 2 == 1, reason="gh-97: ∂θ wrong for odd m", strict=True
            ),
        )
        for lm in MODES
    ],
)
def test_gradient_theta(sphere, lm):
    grid, radius, theta, phi = sphere
    l, m = lm
    y = _ylm(l, m, theta, phi)
    # Central difference of scipy's Y_l^m in θ: truncation ~h² ≈ 1e-12.
    h = 1e-6
    expected = (_ylm(l, m, theta + h, phi) - _ylm(l, m, theta - h, phi)) / (2 * h)
    expected /= radius
    grad_theta, _ = sdx.SphericalDerivative2D(grid).gradient(jnp.asarray(y))
    _assert_close(grad_theta, expected, max(np.abs(expected).max(), 1 / radius), 1e-8)


@pytest.mark.slow
@pytest.mark.xfail(reason="gh-96: Legendre table overflows at T86+", strict=True)
def test_t127_roundtrip_is_finite():
    grid = sdx.SphericalGrid2D.from_N_L(Nx=256, Ny=128, Lx=2 * np.pi, Ly=np.pi)
    phi, theta = (np.asarray(a) for a in grid.X)
    y = _ylm(40, 17, theta, phi) + _ylm(100, 90, theta, phi)
    u_rec = grid.transform(grid.transform(jnp.asarray(y)), inverse=True)
    _assert_close(u_rec, y, np.abs(y).max(), 1e-10)
