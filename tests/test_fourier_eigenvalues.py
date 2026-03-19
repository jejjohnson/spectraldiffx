"""Tests for Laplacian eigenvalue helpers."""

import jax.numpy as jnp
import pytest

from spectraldiffx._src.fourier.eigenvalues import (
    dct1_eigenvalues,
    dct2_eigenvalues,
    dct3_eigenvalues,
    dct4_eigenvalues,
    dst1_eigenvalues,
    dst2_eigenvalues,
    dst3_eigenvalues,
    dst4_eigenvalues,
    fft_eigenvalues,
)

SIZES = [4, 8, 16, 32]


class TestDST1Eigenvalues:
    """DST-I eigenvalues (Dirichlet BCs, regular grid)."""

    @pytest.mark.parametrize("N", SIZES)
    def test_shape(self, N):
        eigs = dst1_eigenvalues(N, dx=1.0)
        assert eigs.shape == (N,)

    @pytest.mark.parametrize("N", SIZES)
    def test_all_strictly_negative(self, N):
        eigs = dst1_eigenvalues(N, dx=1.0)
        assert jnp.all(eigs < 0)

    def test_known_values_N2(self):
        """For N=2, dx=1: λ_k = -4 sin²(π(k+1)/6), k=0,1."""
        eigs = dst1_eigenvalues(2, dx=1.0)
        expected = jnp.array(
            [
                -4.0 * jnp.sin(jnp.pi / 6) ** 2,
                -4.0 * jnp.sin(jnp.pi / 3) ** 2,
            ]
        )
        assert jnp.allclose(eigs, expected, atol=1e-6)

    @pytest.mark.parametrize("N", SIZES)
    def test_scales_with_dx(self, N):
        """Eigenvalues scale as 1/dx²."""
        eigs1 = dst1_eigenvalues(N, dx=1.0)
        eigs2 = dst1_eigenvalues(N, dx=2.0)
        assert jnp.allclose(eigs2, eigs1 / 4.0, atol=1e-6)


class TestDST2Eigenvalues:
    """DST-II eigenvalues (Dirichlet BCs, staggered grid)."""

    @pytest.mark.parametrize("N", SIZES)
    def test_shape(self, N):
        eigs = dst2_eigenvalues(N, dx=1.0)
        assert eigs.shape == (N,)

    @pytest.mark.parametrize("N", SIZES)
    def test_all_strictly_negative(self, N):
        eigs = dst2_eigenvalues(N, dx=1.0)
        assert jnp.all(eigs < 0)

    def test_known_values_N2(self):
        """For N=2, dx=1: λ_k = -4 sin²(π(k+1)/4), k=0,1."""
        eigs = dst2_eigenvalues(2, dx=1.0)
        expected = jnp.array(
            [
                -4.0 * jnp.sin(jnp.pi * 1 / 4) ** 2,
                -4.0 * jnp.sin(jnp.pi * 2 / 4) ** 2,
            ]
        )
        assert jnp.allclose(eigs, expected, atol=1e-6)

    @pytest.mark.parametrize("N", SIZES)
    def test_scales_with_dx(self, N):
        eigs1 = dst2_eigenvalues(N, dx=1.0)
        eigs2 = dst2_eigenvalues(N, dx=2.0)
        assert jnp.allclose(eigs2, eigs1 / 4.0, atol=1e-6)

    @pytest.mark.parametrize("N", SIZES)
    def test_differs_from_dst1(self, N):
        """DST-II and DST-I eigenvalues should differ for same N."""
        eigs1 = dst1_eigenvalues(N, dx=1.0)
        eigs2 = dst2_eigenvalues(N, dx=1.0)
        assert not jnp.allclose(eigs1, eigs2, atol=1e-6)


class TestDCT1Eigenvalues:
    """DCT-I eigenvalues (Neumann BCs, regular grid)."""

    @pytest.mark.parametrize("N", SIZES)
    def test_shape(self, N):
        eigs = dct1_eigenvalues(N, dx=1.0)
        assert eigs.shape == (N,)

    @pytest.mark.parametrize("N", SIZES)
    def test_null_mode(self, N):
        """First eigenvalue (k=0) is zero."""
        eigs = dct1_eigenvalues(N, dx=1.0)
        assert jnp.abs(eigs[0]) < 1e-10

    @pytest.mark.parametrize("N", SIZES)
    def test_remaining_negative(self, N):
        eigs = dct1_eigenvalues(N, dx=1.0)
        assert jnp.all(eigs[1:] < 0)

    @pytest.mark.parametrize("N", SIZES)
    def test_scales_with_dx(self, N):
        eigs1 = dct1_eigenvalues(N, dx=1.0)
        eigs2 = dct1_eigenvalues(N, dx=2.0)
        assert jnp.allclose(eigs2, eigs1 / 4.0, atol=1e-6)

    def test_raises_for_n_less_than_2(self):
        with pytest.raises(ValueError, match="N >= 2"):
            dct1_eigenvalues(1, dx=1.0)


class TestDCT2Eigenvalues:
    """DCT-II eigenvalues (Neumann BCs, staggered grid)."""

    @pytest.mark.parametrize("N", SIZES)
    def test_shape(self, N):
        eigs = dct2_eigenvalues(N, dx=1.0)
        assert eigs.shape == (N,)

    @pytest.mark.parametrize("N", SIZES)
    def test_null_mode(self, N):
        """First eigenvalue (k=0) is zero."""
        eigs = dct2_eigenvalues(N, dx=1.0)
        assert jnp.abs(eigs[0]) < 1e-10

    @pytest.mark.parametrize("N", SIZES)
    def test_remaining_negative(self, N):
        eigs = dct2_eigenvalues(N, dx=1.0)
        assert jnp.all(eigs[1:] < 0)


class TestMixedBCEigenvalues:
    """Mixed-BC eigenvalues (DST-III, DCT-III, DST-IV, DCT-IV)."""

    @pytest.mark.parametrize("N", SIZES)
    @pytest.mark.parametrize(
        "eig_fn",
        [dst3_eigenvalues, dct3_eigenvalues, dst4_eigenvalues, dct4_eigenvalues],
    )
    def test_shape(self, N, eig_fn):
        eigs = eig_fn(N, dx=1.0)
        assert eigs.shape == (N,)

    @pytest.mark.parametrize("N", SIZES)
    @pytest.mark.parametrize(
        "eig_fn",
        [dst3_eigenvalues, dct3_eigenvalues, dst4_eigenvalues, dct4_eigenvalues],
    )
    def test_all_strictly_negative(self, N, eig_fn):
        """Mixed-BC eigenvalues have no null mode — all strictly negative."""
        eigs = eig_fn(N, dx=1.0)
        assert jnp.all(eigs < 0)

    @pytest.mark.parametrize("N", SIZES)
    @pytest.mark.parametrize(
        "eig_fn",
        [dst3_eigenvalues, dct3_eigenvalues, dst4_eigenvalues, dct4_eigenvalues],
    )
    def test_scales_with_dx(self, N, eig_fn):
        eigs1 = eig_fn(N, dx=1.0)
        eigs2 = eig_fn(N, dx=2.0)
        assert jnp.allclose(eigs2, eigs1 / 4.0, atol=1e-6)

    def test_all_four_share_formula(self):
        """All four mixed-BC eigenvalue functions produce the same values."""
        N, dx = 16, 1.0
        ref = dst3_eigenvalues(N, dx)
        assert jnp.allclose(ref, dct3_eigenvalues(N, dx))
        assert jnp.allclose(ref, dst4_eigenvalues(N, dx))
        assert jnp.allclose(ref, dct4_eigenvalues(N, dx))


class TestFFTEigenvalues:
    """FFT eigenvalues (periodic BCs)."""

    @pytest.mark.parametrize("N", SIZES)
    def test_shape(self, N):
        eigs = fft_eigenvalues(N, dx=1.0)
        assert eigs.shape == (N,)

    @pytest.mark.parametrize("N", SIZES)
    def test_null_mode(self, N):
        """First eigenvalue (k=0) is zero."""
        eigs = fft_eigenvalues(N, dx=1.0)
        assert jnp.abs(eigs[0]) < 1e-10

    @pytest.mark.parametrize("N", SIZES)
    def test_all_nonpositive(self, N):
        eigs = fft_eigenvalues(N, dx=1.0)
        assert jnp.all(eigs <= 1e-10)
