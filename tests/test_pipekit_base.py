"""Scaffolding of the ``spectraldiffx.pipekit`` operator layer (Tier B)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("pipekit_jax")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pipekit import Operator

from spectraldiffx import FourierGrid2D
from spectraldiffx.pipekit import (
    Spectral,
    ToPhysical,
    ToSpectral,
    as_grid_spec,
    build_grid,
)
from spectraldiffx.pipekit.testing import check_operator, count_transforms

SPEC = (16, 12, 2 * np.pi, 3.0, "2/3")


def _field(seed: int = 0, lead: tuple[int, ...] = ()) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((*lead, 12, 16)))


class TestGridSpec:
    def test_normalises_lists_grids_and_the_default_dealias(self):
        assert as_grid_spec([16, 12, 2 * np.pi, 3]) == SPEC
        assert as_grid_spec(FourierGrid2D.from_N_L(16, 12, 2 * np.pi, 3.0)) == SPEC
        assert as_grid_spec((16, 12, 1.0, 1.0, None))[-1] is None

    @pytest.mark.parametrize(
        "bad", [(16, 12, 1.0), (0, 12, 1.0, 1.0), (16, 12, 1.0, 1.0, "3/2")]
    )
    def test_rejects_bad_specs(self, bad):
        with pytest.raises(ValueError, match="grid spec"):
            as_grid_spec(bad)

    def test_build_grid_is_cached(self):
        g = build_grid(SPEC)
        assert (g.Nx, g.Ny, g.dealias) == (16, 12, "2/3")
        assert build_grid(SPEC) is g


class TestSpectralCarrier:
    def test_is_a_pytree_that_vmaps_and_scans(self):
        s = Spectral(jnp.fft.fft2(_field(0, (3,))))
        assert len(jax.tree.leaves(s)) == 1
        doubled = jax.vmap(lambda x: x * 2.0)(s)
        np.testing.assert_allclose(doubled.hat, 2 * s.hat)
        carry, _ = jax.lax.scan(lambda c, _: (c + c, None), s, None, length=3)
        np.testing.assert_allclose(carry.hat, 8 * s.hat)

    def test_linear_algebra_stays_on_the_carrier(self):
        a = Spectral(jnp.fft.fft2(_field(0)))
        b = Spectral(jnp.fft.fft2(_field(1)))
        np.testing.assert_allclose((a + b).hat, a.hat + b.hat)
        np.testing.assert_allclose((a - b).hat, a.hat - b.hat)
        np.testing.assert_allclose((-a).hat, -a.hat)
        np.testing.assert_allclose((0.5 * a / 2).hat, a.hat / 4)
        assert sum([a, b]).hat.shape == a.hat.shape
        with pytest.raises(TypeError, match="ToSpectral"):
            a + _field(2)


class TestTransforms:
    def test_round_trip(self):
        x = _field(3, (2,))
        y = ToPhysical(grid=SPEC)(ToSpectral(grid=SPEC)(x))
        np.testing.assert_allclose(y, x, atol=1e-12)

    def test_transforms_are_idempotent_on_their_output_space(self):
        s = ToSpectral(grid=SPEC)(_field(0))
        assert ToSpectral(grid=SPEC)(s) is s or np.array_equal(
            ToSpectral(grid=SPEC)(s).hat, s.hat
        )
        x = _field(1)
        np.testing.assert_array_equal(ToPhysical(grid=SPEC)(x), x)

    def test_budgets(self):
        x = _field(0)
        s = ToSpectral(grid=SPEC)(x)
        assert count_transforms(ToSpectral(grid=SPEC), x) == {"FFT": 1}
        assert count_transforms(ToPhysical(grid=SPEC), s) == {"IFFT": 1}
        assert count_transforms(ToSpectral(grid=SPEC), s) == {}
        both = eqx.filter_jit(lambda v: ToPhysical(grid=SPEC)(ToSpectral(grid=SPEC)(v)))
        assert count_transforms(both, x) == {"FFT": 1, "IFFT": 1}

    def test_rejects_complex_physical_input(self):
        with pytest.raises(TypeError, match="Spectral"):
            ToSpectral(grid=SPEC)(_field(0) + 0j)

    def test_config_round_trip(self):
        op = ToSpectral(grid=list(SPEC))
        rebuilt = Operator.from_state(json.loads(json.dumps(op.state)))
        assert rebuilt.grid == SPEC

    @pytest.mark.slow  # compiles several variants (~3 s)
    @pytest.mark.parametrize("which", ["to_spectral", "to_physical"])
    def test_contract(self, which):
        x = _field(0)
        if which == "to_spectral":
            check_operator(ToSpectral(grid=SPEC), x, budget={"FFT": 1})
        else:
            check_operator(
                ToPhysical(grid=SPEC), ToSpectral(grid=SPEC)(x), budget={"IFFT": 1}
            )


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


def test_import_spectraldiffx_does_not_import_pipekit():
    res = _run(
        "import sys, spectraldiffx; "
        "assert not any(m.split('.')[0] in ('pipekit', 'pipekit_jax') "
        "for m in sys.modules), 'pipekit imported'"
    )
    assert res.returncode == 0, res.stderr


def test_missing_extra_names_it():
    res = _run(
        "import sys; sys.modules['pipekit_jax'] = None; import spectraldiffx.pipekit"
    )
    assert res.returncode != 0
    assert "spectraldiffx[pipekit]" in res.stderr
