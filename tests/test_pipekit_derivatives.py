"""Fourier derivative operators of ``spectraldiffx.pipekit``."""

from __future__ import annotations

import pytest

pytest.importorskip("pipekit_jax")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pipekit import Graph, Input

from spectraldiffx import SpectralDerivative2D, pipekit as sp
from spectraldiffx.pipekit.testing import check_operator, count_transforms

SPEC = (16, 12, 2 * np.pi, 3.0, "2/3")
GRID = sp.build_grid(SPEC)
D = SpectralDerivative2D(GRID)


def _field(seed: int, lead: tuple[int, ...] = ()) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((*lead, 12, 16)))


def _spectral(x):
    return sp.Spectral(GRID.transform(x))


# name -> (op factory taking out=, Tier-A physical reference, n inputs,
#          spectral -> spectral budget)
CASES = {
    "Gradient": (lambda out: sp.Gradient(grid=SPEC, out=out), D.gradient, 1, {}),
    "Divergence": (lambda out: sp.Divergence(grid=SPEC, out=out), D.divergence, 2, {}),
    "Curl": (lambda out: sp.Curl(grid=SPEC, out=out), D.curl, 2, {}),
    "Laplacian": (lambda out: sp.Laplacian(grid=SPEC, out=out), D.laplacian, 1, {}),
    "InverseLaplacian": (
        lambda out: sp.InverseLaplacian(grid=SPEC, out=out),
        D.inverse_laplacian,
        1,
        {},
    ),
    "Biharmonic": (lambda out: sp.Biharmonic(grid=SPEC, out=out), D.biharmonic, 1, {}),
    "Hyperviscosity": (
        lambda out: sp.Hyperviscosity(grid=SPEC, nu=0.05, order=2, out=out),
        lambda u: D.hyperviscosity(u, 0.05, 2),
        1,
        {},
    ),
    "VelocityFromStreamfunction": (
        lambda out: sp.VelocityFromStreamfunction(grid=SPEC, out=out),
        D.velocity_from_streamfunction,
        1,
        {},
    ),
    "Jacobian": (
        lambda out: sp.Jacobian(grid=SPEC, out=out),
        D.jacobian,
        2,
        {"FFT": 1, "IFFT": 4},
    ),
    "Dealias": (lambda out: sp.Dealias(grid=SPEC, out=out), D.apply_dealias, 1, {}),
}
NAMES = sorted(CASES)


def _physical(y):
    return sp.ToPhysical(grid=SPEC)(y) if isinstance(y, sp.Spectral) else y


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("carrier", ["physical", "spectral"])
@pytest.mark.parametrize("out", ["physical", "spectral"])
def test_equals_tier_a(name, carrier, out):
    make, ref, n, _ = CASES[name]
    xs = [_field(k) for k in range(n)]
    want = ref(*xs)
    inputs = xs if carrier == "physical" else [_spectral(x) for x in xs]
    got = make(out)(*inputs)
    want_leaves, got_leaves = (
        jax.tree.leaves(want),
        jax.tree.leaves(got, is_leaf=lambda v: isinstance(v, sp.Spectral)),
    )
    for w, g in zip(want_leaves, got_leaves, strict=True):
        assert isinstance(g, sp.Spectral) == (out == "spectral")
        np.testing.assert_allclose(_physical(g), w, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("name", NAMES)
def test_spectral_to_spectral_budget(name):
    make, _, n, budget = CASES[name]
    hats = [_spectral(_field(k)) for k in range(n)]
    assert count_transforms(make("spectral"), *hats) == budget


def test_physical_carrier_pays_the_round_trips():
    """The same Laplacian on the physical carrier: 1 FFT + 1 IFFT."""
    assert count_transforms(sp.Laplacian(grid=SPEC), _field(0)) == {
        "FFT": 1,
        "IFFT": 1,
    }


def test_advection_scalar():
    vx, vy, q = _field(0), _field(1), _field(2)
    want = D.advection_scalar(vx, vy, q)
    for qq in (q, _spectral(q)):
        got = sp.AdvectionScalar(grid=SPEC)(vx, vy, qq)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)
    got = sp.AdvectionScalar(grid=SPEC, out="spectral")(
        _spectral(vx), _spectral(vy), _spectral(q)
    )
    np.testing.assert_allclose(_physical(got), want, rtol=1e-10, atol=1e-10)


def test_hyperviscosity_nu_is_a_learnable_leaf():
    q_hat = _spectral(_field(0))

    def loss(op):
        return jnp.sum(jnp.abs(op(q_hat).hat) ** 2)

    def make(nu):
        return sp.Hyperviscosity(grid=SPEC, nu=nu, out="spectral")

    ad = eqx.filter_jit(eqx.filter_grad(loss))(make(0.05)).nu
    eps = 1e-6
    fd = (loss(make(0.05 + eps)) - loss(make(0.05 - eps))) / (2 * eps)
    np.testing.assert_allclose(ad, fd, rtol=1e-6)  # quadratic in nu


def test_lincomb_and_pick_work_on_both_carriers():
    a, b = _field(0), _field(1)
    np.testing.assert_allclose(sp.LinComb(coeffs=(2.0, -1.0))(a, b), 2 * a - b)
    s = sp.LinComb(coeffs=(2.0, -1.0))(_spectral(a), _spectral(b))
    np.testing.assert_allclose(_physical(s), 2 * a - b, atol=1e-12)
    u, v = sp.Gradient(grid=SPEC)(a)
    np.testing.assert_array_equal(sp.Pick(index=1)((u, v)), v)
    with pytest.raises(ValueError, match="2 coefficients"):
        sp.LinComb(coeffs=(1.0, 2.0))(a)


# ---------------------------------------------------------------- composition
NU = 1e-3


def barotropic_vorticity_rhs() -> Graph:
    """dq/dt = -J(psi, q) - nu (-nabla^2)^2 q,  nabla^2 psi = q, on q_hat."""
    q = Input("q")
    psi = sp.InverseLaplacian(grid=SPEC, out="spectral")(q)
    adv = sp.Jacobian(grid=SPEC, out="spectral")(psi, q)
    visc = sp.Hyperviscosity(grid=SPEC, nu=NU, order=2, out="spectral")(q)
    return Graph(
        inputs={"q": q}, outputs={"q": sp.LinComb(coeffs=(-1.0, 1.0))(adv, visc)}
    )


def _rk4(rhs, s, dt):
    k1 = rhs(q=s)["q"]
    k2 = rhs(q=s + 0.5 * dt * k1)["q"]
    k3 = rhs(q=s + 0.5 * dt * k2)["q"]
    k4 = rhs(q=s + dt * k3)["q"]
    return s + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def test_barotropic_vorticity_graph_costs_five_transforms_per_rhs():
    rhs = barotropic_vorticity_rhs()
    q_hat = _spectral(_field(0))
    assert count_transforms(lambda q: rhs(q=q)["q"], q_hat) == {"FFT": 1, "IFFT": 4}
    step = eqx.filter_jit(lambda s: _rk4(rhs, s, 1e-3))
    assert count_transforms(step, q_hat) == {"FFT": 4, "IFFT": 16}


def test_barotropic_vorticity_graph_equals_tier_a():
    rhs = barotropic_vorticity_rhs()
    q = _field(0)
    got = _physical(rhs(q=_spectral(q))["q"])
    psi = D.inverse_laplacian(q)
    want = -D.jacobian(psi, q) + D.hyperviscosity(q, NU, 2)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


@pytest.mark.slow  # every op compiles several variants (~2 s each)
@pytest.mark.parametrize("name", NAMES)
def test_contract(name):
    make, _, n, budget = CASES[name]
    check_operator(
        make("spectral"), *[_spectral(_field(k)) for k in range(n)], budget=budget
    )
    check_operator(make("physical"), *[_field(k) for k in range(n)])
