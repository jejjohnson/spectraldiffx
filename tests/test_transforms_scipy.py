"""DCT/DST types I-IV against scipy.fft, forward and inverse (gh-119).

The older tests in ``test_fourier_transforms.py`` use powers of two and
``atol=1e-5``, which cannot see a 1e-6 relative error. These cover odd and
prime lengths, both norms, the inverses, the N-D wrappers with negative and
partial axes, and dtype preservation. The implementation agrees with scipy
to ~1e-15, so the relative tolerance is 1e-12.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.fft as sf

import spectraldiffx as sdx

SIZES = [2, 3, 5, 8, 15, 16, 17, 33, 64]
TYPES = [1, 2, 3, 4]
NORMS = [None, "ortho"]

_PAIRS = {
    "dct": (sdx.dct, sf.dct),
    "idct": (sdx.idct, sf.idct),
    "dst": (sdx.dst, sf.dst),
    "idst": (sdx.idst, sf.idst),
}
_PAIRS_N = {
    "dctn": (sdx.dctn, sf.dctn),
    "idctn": (sdx.idctn, sf.idctn),
    "dstn": (sdx.dstn, sf.dstn),
    "idstn": (sdx.idstn, sf.idstn),
}


def _assert_close(got, expected):
    got = np.asarray(got)
    scale = max(np.abs(expected).max(), 1.0)
    assert got.shape == expected.shape
    assert np.abs(got - expected).max() < 1e-12 * scale


@pytest.mark.parametrize("norm", NORMS, ids=["norm=None", "norm=ortho"])
@pytest.mark.parametrize("type_", TYPES)
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("name", list(_PAIRS))
def test_1d_matches_scipy(name, n, type_, norm):
    ours, ref = _PAIRS[name]
    x = np.random.default_rng(n).standard_normal(n)
    _assert_close(
        ours(jnp.asarray(x), type=type_, norm=norm), ref(x, type=type_, norm=norm)
    )


@pytest.mark.parametrize("axes", [[0], [1], [-1], [-2], [0, 2], [-1, 0], None], ids=str)
@pytest.mark.parametrize("type_", TYPES)
@pytest.mark.parametrize("name", list(_PAIRS_N))
def test_nd_axes_match_scipy(name, type_, axes):
    """3-D input, partial and negative axes, both norms."""
    ours, ref = _PAIRS_N[name]
    x = np.random.default_rng(type_).standard_normal((5, 4, 7))
    for norm in NORMS:
        _assert_close(
            ours(jnp.asarray(x), type=type_, axes=axes, norm=norm),
            ref(x, type=type_, axes=axes, norm=norm),
        )


# float32 input comes back float64 on these paths: gh-93.
_PROMOTES_FLOAT32 = {("dst", 3), ("idst", 2)}


@pytest.mark.parametrize(
    ("name", "type_"),
    [
        pytest.param(
            name,
            t,
            marks=pytest.mark.xfail(
                (name, t) in _PROMOTES_FLOAT32,
                reason="gh-93: float32 promoted to float64",
                strict=True,
            ),
        )
        for name in _PAIRS
        for t in TYPES
    ],
)
def test_float32_is_preserved(name, type_):
    ours, _ = _PAIRS[name]
    x = jnp.asarray(np.random.default_rng(0).standard_normal(16), dtype=jnp.float32)
    assert ours(x, type=type_).dtype == jnp.float32
