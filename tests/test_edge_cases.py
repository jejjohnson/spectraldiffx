"""One test per untested input corner (gh-119, item 8).

Odd N is covered by ``test_solvers_dense.py`` and ``test_transforms_scipy.py``.
Here: N = 1 transforms, complex input, ``zero_mean=False``, λ < 0,
``approximation="spectral"`` under jit, and the ``spectral=True`` input path.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.fft as sf
from test_solvers_dense import dense_operator

import spectraldiffx as sdx

_TRANSFORMS = {
    "dct": (sdx.dct, sf.dct),
    "dst": (sdx.dst, sf.dst),
    "idct": (sdx.idct, sf.idct),
    "idst": (sdx.idst, sf.idst),
}


@pytest.mark.parametrize("type_", [2, 3, 4])
@pytest.mark.parametrize("name", list(_TRANSFORMS))
def test_length_one_matches_scipy(name, type_):
    """N = 1 for types 2-4 (type 1 at N = 1 is gh-94)."""
    ours, ref = _TRANSFORMS[name]
    x = np.array([1.5])
    np.testing.assert_allclose(
        np.asarray(ours(jnp.asarray(x), type=type_)), ref(x, type=type_)
    )


# Complex input is silently wrong on these paths (the imaginary part is
# dropped or mixed in): gh-93.
_COMPLEX_SILENTLY_WRONG = {
    ("dct", 2), ("dct", 3), ("dct", 4), ("dst", 3), ("dst", 4),
    ("idct", 2), ("idct", 3), ("idct", 4), ("idst", 2), ("idst", 4),
}  # fmt: skip


@pytest.mark.parametrize(
    ("name", "type_"),
    [
        pytest.param(
            name,
            t,
            marks=pytest.mark.xfail(
                (name, t) in _COMPLEX_SILENTLY_WRONG,
                reason="gh-93: complex input silently wrong",
                strict=True,
            ),
        )
        for name in _TRANSFORMS
        for t in (1, 2, 3, 4)
    ],
)
def test_complex_input_matches_scipy_or_raises(name, type_):
    ours, ref = _TRANSFORMS[name]
    rng = np.random.default_rng(type_)
    z = rng.standard_normal(6) + 1j * rng.standard_normal(6)
    try:
        got = np.asarray(ours(jnp.asarray(z), type=type_))
    except (TypeError, ValueError):
        return
    np.testing.assert_allclose(got, ref(z, type=type_), atol=1e-12)


def test_zero_mean_false_keeps_the_mean():
    grid = sdx.FourierGrid2D.from_N_L(Nx=16, Ny=12, Lx=2 * np.pi, Ly=2 * np.pi)
    X, Y = grid.X
    alpha = 2.0
    psi = 0.7 + jnp.sin(X) * jnp.cos(2 * Y)
    f = -alpha * 0.7 + (-(1 + 4) - alpha) * jnp.sin(X) * jnp.cos(2 * Y)
    got = sdx.SpectralHelmholtzSolver2D(grid).solve(f, alpha=alpha, zero_mean=False)
    np.testing.assert_allclose(np.real(np.asarray(got)), np.asarray(psi), atol=1e-12)


def test_negative_lambda_matches_dense():
    """(∇² − λ) with λ < 0 is still non-singular away from the eigenvalues."""
    ny, nx, dy, dx, lam = 5, 7, 0.8, 0.6, -0.3
    rng = np.random.default_rng(0)
    f = rng.standard_normal((ny, nx))
    got = sdx.solve_helmholtz_2d(
        jnp.asarray(f), dx, dy, bc_x="dirichlet", bc_y="neumann_stag", lambda_=lam
    )
    A = dense_operator(("neumann_stag", "dirichlet"), (ny, nx), (dy, dx), lam)
    expected = np.linalg.solve(A, f.reshape(-1)).reshape(ny, nx)
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-10, atol=1e-12)


def test_spectral_approximation_under_jit():
    f = jnp.asarray(np.random.default_rng(1).standard_normal((9, 11)))

    def solve(rhs):
        return sdx.solve_helmholtz_dst(rhs, 0.5, 0.4, 1.0, approximation="spectral")

    np.testing.assert_allclose(
        np.asarray(jax.jit(solve)(f)), np.asarray(solve(f)), rtol=1e-12
    )


def test_spectral_input_path():
    grid = sdx.FourierGrid2D.from_N_L(Nx=16, Ny=12, Lx=2 * np.pi, Ly=1.0)
    X, Y = grid.X
    u = jnp.sin(X) * jnp.cos(2 * np.pi * Y)
    deriv = sdx.SpectralDerivative2D(grid)
    from_physical = deriv.laplacian(u)
    from_spectral = deriv.laplacian(grid.transform(u), spectral=True)
    np.testing.assert_allclose(
        np.real(np.asarray(from_spectral)),
        np.real(np.asarray(from_physical)),
        atol=1e-12,
    )
