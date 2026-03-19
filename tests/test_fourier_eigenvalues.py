"""Tests for Laplacian eigenvalue helpers."""

import jax.numpy as jnp
import pytest

from spectraldiffx._src.fourier.eigenvalues import (
    dct2_eigenvalues,
    dst1_eigenvalues,
    fft_eigenvalues,
)

SIZES = [4, 8, 16, 32]


class TestDST1Eigenvalues:
    """DST-I eigenvalues (Dirichlet BCs)."""

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


class TestDCT2Eigenvalues:
    """DCT-II eigenvalues (Neumann BCs)."""

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
