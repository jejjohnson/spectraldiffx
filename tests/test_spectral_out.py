"""``spectral_out`` on ``SpectralDerivative2D``: coefficients out, no inverse.

``op(u, spectral_out=True)`` equals ``transform(op(u))`` and skips the
inverse transform, so a pipeline that stays in spectral space pays no
fft / ifft round trip between operators. The transform counts are read from
the compiled HLO, so a regression back to physical round trips fails here.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectraldiffx import FourierGrid2D, SpectralDerivative2D

GRID = FourierGrid2D.from_N_L(16, 12, 2 * np.pi, 3.0)  # anisotropic
D = SpectralDerivative2D(GRID)


def _field(seed: int) -> jnp.ndarray:
    return jnp.asarray(np.random.default_rng(seed).standard_normal((12, 16)))


def count_transforms(fn, *args) -> dict[str, int]:
    """FFT / IFFT ops in the compiled HLO of ``jax.jit(fn)(*args)``."""
    text = jax.jit(fn).lower(*args).compile().as_text()
    kinds = re.findall(r"fft_type=(\w+)", text)
    return {k: kinds.count(k) for k in ("FFT", "IFFT", "RFFT", "IRFFT") if k in kinds}


# name -> (method, number of field arguments, extra kwargs)
METHODS = {
    "gradient": (D.gradient, 1, {}),
    "divergence": (D.divergence, 2, {}),
    "curl": (D.curl, 2, {}),
    "laplacian": (D.laplacian, 1, {}),
    "biharmonic": (D.biharmonic, 1, {}),
    "hyperviscosity": (D.hyperviscosity, 1, {"nu": 1e-2, "order": 2}),
    "inverse_laplacian": (D.inverse_laplacian, 1, {}),
    "velocity_from_streamfunction": (D.velocity_from_streamfunction, 1, {}),
    "jacobian": (D.jacobian, 2, {}),
}
LINEAR = [m for m in METHODS if m != "jacobian"]


@pytest.mark.parametrize("name", sorted(METHODS))
@pytest.mark.parametrize("spectral_in", [False, True])
def test_spectral_out_is_the_transform_of_the_physical_output(name, spectral_in):
    method, n, kw = METHODS[name]
    xs = [_field(k) for k in range(n)]
    if spectral_in:
        xs = [GRID.transform(x) for x in xs]
    physical = method(*xs, spectral=spectral_in, **kw)
    spectral = method(*xs, spectral=spectral_in, spectral_out=True, **kw)
    for p, s in zip(jax.tree.leaves(physical), jax.tree.leaves(spectral), strict=True):
        assert jnp.iscomplexobj(s)
        np.testing.assert_allclose(GRID.transform(p), s, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("name", LINEAR)
def test_linear_operators_cost_no_transform_in_spectral_space(name):
    method, n, kw = METHODS[name]
    hats = [GRID.transform(_field(k)) for k in range(n)]
    assert (
        count_transforms(
            lambda *h: method(*h, spectral=True, spectral_out=True, **kw), *hats
        )
        == {}
    )


def test_jacobian_costs_four_inverse_and_one_forward_in_spectral_space():
    f_hat, g_hat = GRID.transform(_field(0)), GRID.transform(_field(1))
    counts = count_transforms(
        lambda f, g: D.jacobian(f, g, spectral=True, spectral_out=True), f_hat, g_hat
    )
    assert counts == {"FFT": 1, "IFFT": 4}
    # physical in, physical out (today's default) for comparison: 2 + 4 + 1 + 1
    counts = count_transforms(D.jacobian, _field(0), _field(1))
    assert counts == {"FFT": 3, "IFFT": 5}


def test_advection_scalar_spectral_out():
    vx, vy, q = _field(0), _field(1), _field(2)
    physical = D.advection_scalar(vx, vy, q)
    spectral = D.advection_scalar(vx, vy, q, spectral_out=True)
    np.testing.assert_allclose(GRID.transform(physical), spectral, atol=1e-10)


def test_apply_dealias_keeps_the_input_space_by_default():
    u = _field(0)
    u_hat = GRID.transform(u)
    assert jnp.iscomplexobj(D.apply_dealias(u_hat, spectral=True))
    assert not jnp.iscomplexobj(D.apply_dealias(u))
    np.testing.assert_allclose(
        D.apply_dealias(u, spectral_out=True),
        D.apply_dealias(u_hat, spectral=True),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        D.apply_dealias(u_hat, spectral=True, spectral_out=False),
        D.apply_dealias(u),
        atol=1e-12,
    )


def test_barotropic_vorticity_rhs_costs_five_transforms():
    """dq/dt = -J(psi, q) - nu (-nabla^2)^2 q with psi = nabla^{-2} q, on q_hat.

    psi_hat and the hyperviscous term never leave spectral space; only the
    Jacobian's products need physical fields: 4 inverse + 1 forward.
    """

    def rhs(q_hat):
        psi_hat = D.inverse_laplacian(q_hat, spectral=True, spectral_out=True)
        adv = D.jacobian(psi_hat, q_hat, spectral=True, spectral_out=True)
        visc = D.hyperviscosity(q_hat, 1e-3, 2, spectral=True, spectral_out=True)
        return -adv + visc

    assert count_transforms(rhs, GRID.transform(_field(0))) == {"FFT": 1, "IFFT": 4}
