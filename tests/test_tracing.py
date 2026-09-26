"""jit / vmap / grad smoke tests across the public API (gh-119).

Each case is a function of one real field ``u``. The three tests check:

* ``jax.jit(f)(u)`` equals ``f(u)``;
* ``jax.vmap(f)`` over a batch of three fields equals three separate calls;
* ``jax.grad(lambda u: sum(f(u)**2))`` matches a central finite difference
  along a random direction.

The directional-derivative tolerance is 1e-6 relative. A step of h = 1e-5 in
float64 leaves an O(h²) truncation error of ~1e-10 plus round-off of
~eps/h ≈ 1e-11, well below the bound.

Grids are small, and modules are closed over rather than passed as arguments,
because the jit-compatibility of module fields is tracked separately (#102).
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import spectraldiffx as sdx


def _real(f):
    return lambda u: jnp.real(f(u))


def _fourier2d():
    return sdx.FourierGrid2D.from_N_L(Nx=16, Ny=12, Lx=2 * np.pi, Ly=1.0)


def _cheb2d():
    return sdx.ChebyshevGrid2D.from_N_L(Nx=10, Ny=8, Lx=1.0, Ly=1.5)


def _sphere2d():
    return sdx.SphericalGrid2D.from_N_L(Nx=16, Ny=8)


def _capacitance():
    j, i = np.mgrid[:12, :10]
    mask = np.hypot(j - 5.5, i - 4.5) < 4.5
    return sdx.build_capacitance_solver(mask, 1.0, 0.8, lambda_=1.0)


# name -> (builder returning f, input shape)
CASES: dict[str, tuple[Callable[[], Callable], tuple[int, ...]]] = {
    "dctn": (lambda: lambda u: sdx.dctn(u, type=2, axes=[0, 1]), (6, 5)),
    "idstn": (lambda: lambda u: sdx.idstn(u, type=4, axes=[0, 1]), (6, 5)),
    "solve_helmholtz_2d-dirichlet-neumann": (
        lambda: lambda u: sdx.solve_helmholtz_2d(
            u, 0.9, 0.6, bc_x="dirichlet", bc_y="neumann_stag", lambda_=0.5
        ),
        (6, 7),
    ),
    "solve_poisson_fft": (
        lambda: lambda u: sdx.solve_poisson_fft(u, 0.9, 0.6),
        (6, 8),
    ),
    "solve_helmholtz_3d": (
        lambda: lambda u: sdx.solve_helmholtz_3d(
            u, 1.0, 0.8, 0.6, bc_x="periodic", bc_y="dirichlet", lambda_=1.0
        ),
        (4, 5, 6),
    ),
    "SpectralDerivative2D.laplacian": (
        lambda: _real(sdx.SpectralDerivative2D(_fourier2d()).laplacian),
        (12, 16),
    ),
    "SpectralDerivative2D.jacobian": (
        lambda: (lambda d: _real(lambda u: d.jacobian(u, jnp.cos(u))))(
            sdx.SpectralDerivative2D(_fourier2d())
        ),
        (12, 16),
    ),
    "SpectralFilter2D.exponential_filter": (
        lambda: _real(sdx.SpectralFilter2D(_fourier2d()).exponential_filter),
        (12, 16),
    ),
    "SpectralHelmholtzSolver2D": (
        lambda: _real(
            lambda u: sdx.SpectralHelmholtzSolver2D(_fourier2d()).solve(u, alpha=1.0)
        ),
        (12, 16),
    ),
    "build_capacitance_solver": (_capacitance, (12, 10)),
    "ChebyshevDerivative2D.laplacian": (
        lambda: sdx.ChebyshevDerivative2D(_cheb2d()).laplacian,
        (9, 11),
    ),
    "ChebyshevHelmholtzSolver2D": (
        lambda: lambda u: sdx.ChebyshevHelmholtzSolver2D(_cheb2d()).solve(u, alpha=2.0),
        (9, 11),
    ),
    "ChebyshevFilter2D.exponential_filter": (
        lambda: sdx.ChebyshevFilter2D(_cheb2d()).exponential_filter,
        (9, 11),
    ),
    "SphericalDerivative2D.laplacian": (
        lambda: _real(sdx.SphericalDerivative2D(_sphere2d()).laplacian),
        (8, 16),
    ),
    "SphericalFilter2D.exponential_filter": (
        lambda: _real(sdx.SphericalFilter2D(_sphere2d()).exponential_filter),
        (8, 16),
    ),
    "SphericalHelmholtzSolver": (
        lambda: _real(
            lambda u: sdx.SphericalHelmholtzSolver(_sphere2d()).solve(u, alpha=1.0)
        ),
        (8, 16),
    ),
    "SphericalVorticityInversionSolver": (
        lambda: lambda u: jnp.real(
            sdx.SphericalVorticityInversionSolver(_sphere2d()).solve(u)[0]
        ),
        (8, 16),
    ),
}


def _case(name):
    make, shape = CASES[name]
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    return make(), jnp.asarray(rng.standard_normal(shape)), rng


@pytest.fixture(params=list(CASES))
def case(request):
    return _case(request.param)


def _close(a, b, rtol=1e-10):
    a, b = np.asarray(a), np.asarray(b)
    return np.abs(a - b).max() <= rtol * max(np.abs(b).max(), 1.0)


# gh-98: the GFD solvers convert a traced array to NumPy inside solve().
_JIT_XFAIL = {"SphericalVorticityInversionSolver"}


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                name in _JIT_XFAIL, reason="gh-98: not jittable", strict=True
            ),
        )
        for name in CASES
    ],
)
def test_jit(name):
    f, u, _ = _case(name)
    # Wrap in a lambda: jax.jit hashes its callable, and a module (or a bound
    # method of one) holding arrays is not hashable.
    assert _close(jax.jit(lambda v: f(v))(u), f(u))


def test_vmap(case):
    f, u, rng = case
    batch = jnp.stack([u, 2.0 * u, jnp.asarray(rng.standard_normal(u.shape))])
    expected = jnp.stack([f(b) for b in batch])
    assert _close(jax.vmap(f)(batch), expected)


def test_grad_matches_finite_difference(case):
    f, u, rng = case

    def loss(v):
        return jnp.sum(f(v) ** 2)

    direction = jnp.asarray(rng.standard_normal(u.shape))
    analytic = float(jnp.vdot(jax.grad(loss)(u), direction))
    h = 1e-5
    numeric = float((loss(u + h * direction) - loss(u - h * direction)) / (2 * h))
    assert abs(analytic - numeric) <= 1e-6 * max(abs(numeric), 1.0)


def test_traced_lambda_in_solve_helmholtz_2d():
    """λ may be a traced value (jit argument, grad variable)."""
    u = jnp.asarray(np.random.default_rng(3).standard_normal((6, 7)))

    def solve(lam):
        return sdx.solve_helmholtz_2d(u, 0.9, 0.6, "dirichlet", "periodic", lam)

    assert _close(jax.jit(solve)(0.5), solve(0.5))
    h = 1e-5
    fd = (jnp.sum(solve(0.5 + h)) - jnp.sum(solve(0.5 - h))) / (2 * h)
    ad = jax.grad(lambda lam: jnp.sum(solve(lam)))(0.5)
    assert abs(float(ad - fd)) <= 1e-6 * max(abs(float(fd)), 1.0)
