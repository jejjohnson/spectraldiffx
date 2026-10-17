"""Fourier parameters can be traced (gh-103).

Learnable coefficients, parameter sweeps and jitted solvers pass these
parameters as tracers; a Python ``if`` on them raised
``TracerBoolConversionError``. Concrete invalid values still raise
``ValueError`` eagerly; traced ones raise at run time (``eqx.error_if``).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectraldiffx import (
    FourierGrid1D,
    FourierGrid2D,
    FourierGrid3D,
    SpectralDerivative1D,
    SpectralDerivative2D,
    SpectralDerivative3D,
    modify_rhs_1d,
    solve_helmholtz_dst,
)

# Under jit, eqx.error_if surfaces through XLA's callback as a JaxRuntimeError
# that carries the EquinoxRuntimeError message.
_RUNTIME_ERRORS = (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError)


def _rng_field(shape, seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape))


DERIVS = {
    "1d": (SpectralDerivative1D(FourierGrid1D.from_N_L(16, 2 * np.pi)), (16,)),
    "2d": (SpectralDerivative2D(FourierGrid2D.from_N_L(16, 12, 2.0, 1.5)), (12, 16)),
    "3d": (
        SpectralDerivative3D(FourierGrid3D.from_N_L(8, 6, 4, 1.0, 1.0, 1.0)),
        (8, 6, 4),
    ),
}


@pytest.mark.parametrize("dim", sorted(DERIVS))
def test_hyperviscosity_with_traced_nu(dim):
    d, shape = DERIVS[dim]
    u = _rng_field(shape)
    eager = d.hyperviscosity(u, 1e-3, 2)
    jitted = jax.jit(lambda nu: d.hyperviscosity(u, nu, 2))(1e-3)
    np.testing.assert_allclose(jitted, eager, rtol=1e-12, atol=1e-12)

    nus = jnp.array([1e-3, 2e-3, 5e-3])
    swept = jax.vmap(lambda nu: d.hyperviscosity(u, nu, 2))(nus)
    for i, nu in enumerate(nus):
        np.testing.assert_allclose(
            swept[i], d.hyperviscosity(u, float(nu), 2), rtol=1e-12, atol=1e-12
        )

    def loss(nu):
        return jnp.sum(d.hyperviscosity(u, nu, 2) ** 2)

    ad = jax.jit(jax.grad(loss))(1e-3)
    eps = 1e-6
    fd = (loss(1e-3 + eps) - loss(1e-3 - eps)) / (2 * eps)
    # loss is quadratic in nu: the central difference is exact up to rounding.
    np.testing.assert_allclose(ad, fd, rtol=1e-6)


def test_negative_nu_still_raises():
    d, shape = DERIVS["2d"]
    u = _rng_field(shape)
    with pytest.raises(ValueError, match="nu must be >= 0"):
        d.hyperviscosity(u, -1.0, 2)
    with pytest.raises(_RUNTIME_ERRORS, match="nu must be >= 0"):
        jax.jit(lambda nu: d.hyperviscosity(u, nu, 2))(-1.0)
    with pytest.raises(ValueError, match="order must be >= 1"):
        d.hyperviscosity(u, 1.0, 0)


def test_spectral_eigenvalue_solve_with_traced_dx():
    rhs = _rng_field((10, 12))
    eager = solve_helmholtz_dst(rhs, 0.1, 0.2, 1.0, approximation="spectral")
    jitted = jax.jit(solve_helmholtz_dst, static_argnames="approximation")(
        rhs, 0.1, 0.2, 1.0, approximation="spectral"
    )
    np.testing.assert_allclose(jitted, eager, rtol=1e-12, atol=1e-12)


def test_grid_checks_skip_tracers_and_still_check_concrete_values():
    g = FourierGrid2D.from_N_L(16, 12, 2.0, 1.5)
    assert g.check_consistency()
    # inside jit the grid's floats are concrete Python values: no tracer bool
    jax.jit(lambda x: x * float(g.check_consistency()))(1.0)
    with pytest.raises(ValueError, match="not divisible"):
        FourierGrid1D.from_L_dx(1.0, 0.3)
    assert FourierGrid1D.from_L_dx(1.0, 0.25).N == 4


def test_modify_rhs_1d_with_traced_boundary_values():
    rhs = _rng_field((9,))
    eager = modify_rhs_1d(rhs, "dirichlet", 0.1, (1.0, -2.0))
    jitted = jax.jit(lambda a, b: modify_rhs_1d(rhs, "dirichlet", 0.1, (a, b)))(
        1.0, -2.0
    )
    np.testing.assert_allclose(jitted, eager, rtol=1e-12, atol=1e-12)
    zeros = jax.jit(lambda a, b: modify_rhs_1d(rhs, "dirichlet", 0.1, (a, b)))(0.0, 0.0)
    np.testing.assert_allclose(zeros, rhs, rtol=0, atol=0)
    with pytest.raises(_RUNTIME_ERRORS, match="Periodic"):
        jax.jit(lambda a: modify_rhs_1d(rhs, "periodic", 0.1, (a, 0.0)))(1.0)
