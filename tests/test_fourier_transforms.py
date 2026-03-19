"""Tests for DST/DCT transforms (types I-IV)."""

import jax.numpy as jnp
import pytest
from scipy.fft import dct as scipy_dct, dst as scipy_dst

from spectraldiffx._src.fourier.transforms import (
    dct,
    dctn,
    dst,
    dstn,
    idct,
    idctn,
    idst,
    idstn,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIZES = [4, 8, 16, 32]
TYPES = [1, 2, 3, 4]


@pytest.fixture(params=SIZES, ids=lambda n: f"N={n}")
def vec(request):
    """Random 1-D vector."""
    return jnp.sin(jnp.linspace(0.1, 3.0, request.param)) + 0.5


# ---------------------------------------------------------------------------
# 1-D: scipy agreement
# ---------------------------------------------------------------------------


class TestDCTScipyAgreement:
    """DCT types I-IV match scipy.fft.dct (unnormalized)."""

    @pytest.mark.parametrize("t", TYPES)
    @pytest.mark.parametrize("N", SIZES)
    def test_dct_matches_scipy(self, t, N):
        x = jnp.sin(jnp.linspace(0.1, 3.0, N)) + 0.5
        got = dct(x, type=t)
        expected = scipy_dct(x, type=t, norm=None)
        assert jnp.allclose(got, expected, atol=1e-5), f"DCT-{t}, N={N}"

    @pytest.mark.parametrize("t", TYPES)
    @pytest.mark.parametrize("N", SIZES)
    def test_dst_matches_scipy(self, t, N):
        x = jnp.sin(jnp.linspace(0.1, 3.0, N)) + 0.5
        got = dst(x, type=t)
        expected = scipy_dst(x, type=t, norm=None)
        assert jnp.allclose(got, expected, atol=1e-5), f"DST-{t}, N={N}"


# ---------------------------------------------------------------------------
# 1-D: round-trip invertibility
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """idct(dct(x)) == x and idst(dst(x)) == x for all types."""

    @pytest.mark.parametrize("t", TYPES)
    def test_dct_roundtrip(self, vec, t):
        reconstructed = idct(dct(vec, type=t), type=t)
        assert jnp.allclose(reconstructed, vec, atol=1e-5)

    @pytest.mark.parametrize("t", TYPES)
    def test_dst_roundtrip(self, vec, t):
        reconstructed = idst(dst(vec, type=t), type=t)
        assert jnp.allclose(reconstructed, vec, atol=1e-5)


# ---------------------------------------------------------------------------
# 1-D: input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """1-D functions reject multi-dimensional input."""

    def test_dct_rejects_2d(self):
        x = jnp.ones((4, 4))
        with pytest.raises(ValueError, match="1-D"):
            dct(x)

    def test_dst_rejects_2d(self):
        x = jnp.ones((4, 4))
        with pytest.raises(ValueError, match="1-D"):
            dst(x)

    def test_idct_rejects_2d(self):
        x = jnp.ones((4, 4))
        with pytest.raises(ValueError, match="1-D"):
            idct(x)

    def test_idst_rejects_2d(self):
        x = jnp.ones((4, 4))
        with pytest.raises(ValueError, match="1-D"):
            idst(x)

    def test_dct_invalid_type(self):
        x = jnp.ones(4)
        with pytest.raises(ValueError, match="DCT type"):
            dct(x, type=5)

    def test_dst_invalid_type(self):
        x = jnp.ones(4)
        with pytest.raises(ValueError, match="DST type"):
            dst(x, type=0)


# ---------------------------------------------------------------------------
# Multi-axis: correctness
# ---------------------------------------------------------------------------


class TestMultiAxis:
    """dctn/dstn match sequential single-axis application."""

    @pytest.mark.parametrize("t", TYPES)
    def test_dctn_2d_matches_sequential(self, t):
        x = jnp.sin(jnp.linspace(0.1, 3.0, 24)).reshape(4, 6)
        # Apply along axis 0, then axis 1 using scipy
        expected = scipy_dct(
            scipy_dct(x, type=t, axis=0, norm=None), type=t, axis=1, norm=None
        )
        got = dctn(x, type=t, axes=[0, 1])
        assert jnp.allclose(got, expected, atol=1e-5)

    @pytest.mark.parametrize("t", TYPES)
    def test_dstn_2d_matches_sequential(self, t):
        x = jnp.sin(jnp.linspace(0.1, 3.0, 24)).reshape(4, 6)
        expected = scipy_dst(
            scipy_dst(x, type=t, axis=0, norm=None), type=t, axis=1, norm=None
        )
        got = dstn(x, type=t, axes=[0, 1])
        assert jnp.allclose(got, expected, atol=1e-5)

    @pytest.mark.parametrize("t", TYPES)
    def test_dctn_roundtrip_2d(self, t):
        x = jnp.sin(jnp.linspace(0.1, 3.0, 24)).reshape(4, 6)
        reconstructed = idctn(dctn(x, type=t, axes=[0, 1]), type=t, axes=[0, 1])
        assert jnp.allclose(reconstructed, x, atol=1e-5)

    @pytest.mark.parametrize("t", TYPES)
    def test_dstn_roundtrip_2d(self, t):
        x = jnp.sin(jnp.linspace(0.1, 3.0, 24)).reshape(4, 6)
        reconstructed = idstn(dstn(x, type=t, axes=[0, 1]), type=t, axes=[0, 1])
        assert jnp.allclose(reconstructed, x, atol=1e-5)

    def test_dctn_axes_none_transforms_all(self):
        x = jnp.sin(jnp.linspace(0.1, 3.0, 24)).reshape(4, 6)
        a = dctn(x, type=2, axes=None)
        b = dctn(x, type=2, axes=[0, 1])
        assert jnp.allclose(a, b, atol=1e-6)

    def test_dctn_single_axis(self):
        """dctn with one axis matches scipy single-axis DCT."""
        x = jnp.sin(jnp.linspace(0.1, 3.0, 24)).reshape(4, 6)
        got = dctn(x, type=2, axes=[1])
        expected = scipy_dct(x, type=2, axis=1, norm=None)
        assert jnp.allclose(got, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: all-zeros, length-1."""

    @pytest.mark.parametrize("t", TYPES)
    def test_dct_zeros(self, t):
        x = jnp.zeros(8)
        assert jnp.allclose(dct(x, type=t), jnp.zeros(8), atol=1e-10)

    @pytest.mark.parametrize("t", TYPES)
    def test_dst_zeros(self, t):
        x = jnp.zeros(8)
        assert jnp.allclose(dst(x, type=t), jnp.zeros(8), atol=1e-10)

    @pytest.mark.parametrize("t", [2, 3, 4])
    def test_dct_length1(self, t):
        """Length-1 DCT should work for types 2, 3, 4."""
        x = jnp.array([3.0])
        roundtrip = idct(dct(x, type=t), type=t)
        assert jnp.allclose(roundtrip, x, atol=1e-5)


# ---------------------------------------------------------------------------
# Ortho normalization
# ---------------------------------------------------------------------------


class TestOrthoScipyAgreement:
    """Ortho-normalized transforms match scipy with norm='ortho'."""

    @pytest.mark.parametrize("t", TYPES)
    @pytest.mark.parametrize("N", SIZES)
    def test_dct_ortho_matches_scipy(self, t, N):
        x = jnp.sin(jnp.linspace(0.1, 3.0, N)) + 0.5
        got = dct(x, type=t, norm="ortho")
        expected = scipy_dct(x, type=t, norm="ortho")
        assert jnp.allclose(got, expected, atol=1e-5), f"DCT-{t} ortho, N={N}"

    @pytest.mark.parametrize("t", TYPES)
    @pytest.mark.parametrize("N", SIZES)
    def test_dst_ortho_matches_scipy(self, t, N):
        x = jnp.sin(jnp.linspace(0.1, 3.0, N)) + 0.5
        got = dst(x, type=t, norm="ortho")
        expected = scipy_dst(x, type=t, norm="ortho")
        assert jnp.allclose(got, expected, atol=1e-5), f"DST-{t} ortho, N={N}"


class TestOrthoRoundTrip:
    """Ortho round-trips: idct(dct(x, norm='ortho'), norm='ortho') == x."""

    @pytest.mark.parametrize("t", TYPES)
    def test_dct_ortho_roundtrip(self, vec, t):
        reconstructed = idct(dct(vec, type=t, norm="ortho"), type=t, norm="ortho")
        assert jnp.allclose(reconstructed, vec, atol=1e-5)

    @pytest.mark.parametrize("t", TYPES)
    def test_dst_ortho_roundtrip(self, vec, t):
        reconstructed = idst(dst(vec, type=t, norm="ortho"), type=t, norm="ortho")
        assert jnp.allclose(reconstructed, vec, atol=1e-5)


class TestOrthoMultiAxis:
    """Ortho round-trips for multi-axis transforms."""

    @pytest.mark.parametrize("t", TYPES)
    def test_dctn_ortho_roundtrip_2d(self, t):
        x = jnp.sin(jnp.linspace(0.1, 3.0, 24)).reshape(4, 6)
        y = dctn(x, type=t, axes=[0, 1], norm="ortho")
        reconstructed = idctn(y, type=t, axes=[0, 1], norm="ortho")
        assert jnp.allclose(reconstructed, x, atol=1e-5)

    @pytest.mark.parametrize("t", TYPES)
    def test_dstn_ortho_roundtrip_2d(self, t):
        x = jnp.sin(jnp.linspace(0.1, 3.0, 24)).reshape(4, 6)
        y = dstn(x, type=t, axes=[0, 1], norm="ortho")
        reconstructed = idstn(y, type=t, axes=[0, 1], norm="ortho")
        assert jnp.allclose(reconstructed, x, atol=1e-5)

    @pytest.mark.parametrize("t", TYPES)
    def test_dctn_ortho_matches_scipy_2d(self, t):
        x = jnp.sin(jnp.linspace(0.1, 3.0, 24)).reshape(4, 6)
        got = dctn(x, type=t, axes=[0, 1], norm="ortho")
        expected = scipy_dct(
            scipy_dct(x, type=t, axis=0, norm="ortho"), type=t, axis=1, norm="ortho"
        )
        assert jnp.allclose(got, expected, atol=1e-5)


class TestOrthoValidation:
    """Invalid norm values are rejected."""

    def test_dct_invalid_norm(self):
        x = jnp.ones(4)
        with pytest.raises(ValueError, match="norm"):
            dct(x, norm="invalid")

    def test_dst_invalid_norm(self):
        x = jnp.ones(4)
        with pytest.raises(ValueError, match="norm"):
            dst(x, norm="backward")
