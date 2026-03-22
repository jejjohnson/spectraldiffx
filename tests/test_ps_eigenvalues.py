"""Tests for pseudo-spectral (PS) eigenvalue functions and convergence."""

import jax.numpy as jnp
import pytest

from spectraldiffx._src.fourier.eigenvalues import (
    dct1_eigenvalues,
    dct1_eigenvalues_ps,
    dct2_eigenvalues,
    dct2_eigenvalues_ps,
    dct3_eigenvalues_ps,
    dct4_eigenvalues_ps,
    dst1_eigenvalues,
    dst1_eigenvalues_ps,
    dst2_eigenvalues,
    dst2_eigenvalues_ps,
    dst3_eigenvalues_ps,
    dst4_eigenvalues_ps,
    fft_eigenvalues,
    fft_eigenvalues_ps,
)
from spectraldiffx._src.fourier.solvers import (
    solve_helmholtz_dct,
    solve_helmholtz_dct1,
    solve_helmholtz_dct1_1d,
    solve_helmholtz_dct2_1d,
    solve_helmholtz_dst,
    solve_helmholtz_dst1_1d,
    solve_helmholtz_dst2_1d,
    solve_helmholtz_fft_1d,
)

SIZES = [4, 8, 16, 32]


# ===========================================================================
# PS eigenvalue basic properties
# ===========================================================================


class TestDST1EigenvaluesPS:
    """DST-I PS eigenvalues (Dirichlet, regular)."""

    @pytest.mark.parametrize("N", SIZES)
    def test_shape(self, N):
        eigs = dst1_eigenvalues_ps(N, L=1.0)
        assert eigs.shape == (N,)

    @pytest.mark.parametrize("N", SIZES)
    def test_all_strictly_negative(self, N):
        eigs = dst1_eigenvalues_ps(N, L=1.0)
        assert jnp.all(eigs < 0)

    def test_known_values(self):
        """λ_k = −(π(k+1)/L)² for L=1."""
        eigs = dst1_eigenvalues_ps(4, L=1.0)
        expected = jnp.array([-((jnp.pi * (k + 1)) ** 2) for k in range(4)])
        assert jnp.allclose(eigs, expected, atol=1e-6)

    @pytest.mark.parametrize("N", SIZES)
    def test_scales_with_L(self, N):
        """Eigenvalues scale as 1/L²."""
        eigs1 = dst1_eigenvalues_ps(N, L=1.0)
        eigs2 = dst1_eigenvalues_ps(N, L=2.0)
        assert jnp.allclose(eigs2, eigs1 / 4.0, atol=1e-6)


class TestDST2EigenvaluesPS:
    """DST-II PS eigenvalues (Dirichlet, staggered)."""

    @pytest.mark.parametrize("N", SIZES)
    def test_same_formula_as_dst1_ps(self, N):
        """DST-I and DST-II PS eigenvalues share the same formula."""
        eigs1 = dst1_eigenvalues_ps(N, L=1.0)
        eigs2 = dst2_eigenvalues_ps(N, L=1.0)
        assert jnp.allclose(eigs1, eigs2, atol=1e-10)


class TestDCT1EigenvaluesPS:
    """DCT-I PS eigenvalues (Neumann, regular)."""

    @pytest.mark.parametrize("N", SIZES)
    def test_null_mode(self, N):
        eigs = dct1_eigenvalues_ps(N, L=1.0)
        assert jnp.abs(eigs[0]) < 1e-10

    @pytest.mark.parametrize("N", SIZES)
    def test_remaining_negative(self, N):
        eigs = dct1_eigenvalues_ps(N, L=1.0)
        assert jnp.all(eigs[1:] < 0)

    def test_known_values(self):
        """λ_k = −(πk/L)² for L=1."""
        eigs = dct1_eigenvalues_ps(4, L=1.0)
        expected = jnp.array([-((jnp.pi * k) ** 2) for k in range(4)])
        assert jnp.allclose(eigs, expected, atol=1e-6)


class TestDCT2EigenvaluesPS:
    """DCT-II PS eigenvalues (Neumann, staggered)."""

    @pytest.mark.parametrize("N", SIZES)
    def test_same_formula_as_dct1_ps(self, N):
        eigs1 = dct1_eigenvalues_ps(N, L=1.0)
        eigs2 = dct2_eigenvalues_ps(N, L=1.0)
        assert jnp.allclose(eigs1, eigs2, atol=1e-10)


class TestMixedBCEigenvaluesPS:
    """Mixed-BC PS eigenvalues (DST-III/IV, DCT-III/IV)."""

    @pytest.mark.parametrize("N", SIZES)
    def test_all_strictly_negative(self, N):
        eigs = dst3_eigenvalues_ps(N, L=1.0)
        assert jnp.all(eigs < 0)

    def test_all_four_share_formula(self):
        N, L = 16, 1.0
        ref = dst3_eigenvalues_ps(N, L)
        assert jnp.allclose(ref, dct3_eigenvalues_ps(N, L))
        assert jnp.allclose(ref, dst4_eigenvalues_ps(N, L))
        assert jnp.allclose(ref, dct4_eigenvalues_ps(N, L))


class TestFFTEigenvaluesPS:
    """FFT PS eigenvalues (periodic)."""

    @pytest.mark.parametrize("N", SIZES)
    def test_null_mode(self, N):
        eigs = fft_eigenvalues_ps(N, L=1.0)
        assert jnp.abs(eigs[0]) < 1e-10

    @pytest.mark.parametrize("N", SIZES)
    def test_all_nonpositive(self, N):
        eigs = fft_eigenvalues_ps(N, L=1.0)
        assert jnp.all(eigs <= 1e-10)

    def test_symmetry(self):
        """FFT PS eigenvalues should be symmetric around N/2."""
        N = 8
        eigs = fft_eigenvalues_ps(N, L=1.0)
        # k=1 and k=N-1 should give the same eigenvalue
        assert jnp.allclose(eigs[1], eigs[N - 1], atol=1e-10)

    @pytest.mark.parametrize("N", [5, 7, 9])
    def test_odd_N_symmetry(self, N):
        """FFT PS eigenvalues are symmetric for odd N too."""
        eigs = fft_eigenvalues_ps(N, L=1.0)
        for k in range(1, N):
            assert jnp.allclose(eigs[k], eigs[N - k], atol=1e-10)

    def test_odd_N_middle_mode(self):
        """For odd N=5, k=2 should map to physical wavenumber 2."""
        N = 5
        L = 1.0
        eigs = fft_eigenvalues_ps(N, L)
        # k=2: k_phys = min(2, 5-2) = 2, so lambda = -(2*pi*2/L)^2
        expected = -((2 * jnp.pi * 2 / L) ** 2)
        assert jnp.allclose(eigs[2], expected, atol=1e-10)


# ===========================================================================
# PS vs FD2 agreement at low wavenumbers
# ===========================================================================


class TestPSvsFD2Agreement:
    """PS and FD2 eigenvalues agree for low wavenumbers (k << N)."""

    def test_dst1_low_modes(self):
        N = 128
        dx = 1.0
        L = (N + 1) * dx
        fd2 = dst1_eigenvalues(N, dx)
        ps = dst1_eigenvalues_ps(N, L)
        # First few modes should agree to within ~1%
        assert jnp.allclose(fd2[:4], ps[:4], rtol=0.01)

    def test_dst2_low_modes(self):
        N = 128
        dx = 1.0
        L = N * dx
        fd2 = dst2_eigenvalues(N, dx)
        ps = dst2_eigenvalues_ps(N, L)
        assert jnp.allclose(fd2[:4], ps[:4], rtol=0.01)

    def test_dct1_low_modes(self):
        N = 128
        dx = 1.0
        L = (N - 1) * dx
        fd2 = dct1_eigenvalues(N, dx)
        ps = dct1_eigenvalues_ps(N, L)
        # Skip k=0 (both zero), check k=1..4
        assert jnp.allclose(fd2[1:5], ps[1:5], rtol=0.01)

    def test_dct2_low_modes(self):
        N = 128
        dx = 1.0
        L = N * dx
        fd2 = dct2_eigenvalues(N, dx)
        ps = dct2_eigenvalues_ps(N, L)
        assert jnp.allclose(fd2[1:5], ps[1:5], rtol=0.01)

    def test_fft_low_modes(self):
        N = 128
        dx = 1.0
        L = N * dx
        fd2 = fft_eigenvalues(N, dx)
        ps = fft_eigenvalues_ps(N, L)
        assert jnp.allclose(fd2[1:5], ps[1:5], rtol=0.01)

    def test_diverge_at_nyquist(self):
        """PS and FD2 should diverge at high wavenumbers."""
        N = 32
        dx = 1.0
        L = (N + 1) * dx
        fd2 = dst1_eigenvalues(N, dx)
        ps = dst1_eigenvalues_ps(N, L)
        # At Nyquist (last mode), relative difference should be large
        rel_diff = jnp.abs((fd2[-1] - ps[-1]) / ps[-1])
        assert rel_diff > 0.05  # > 5% difference at Nyquist


# ===========================================================================
# Spectral convergence tests
# ===========================================================================


class TestSpectralConvergence:
    """With PS eigenvalues, smooth test functions converge spectrally."""

    def test_dst1_spectral_convergence(self):
        """Dirichlet (regular): error drops to machine precision for smooth f."""
        errors = []
        for N in [8, 16, 32, 64]:
            dx = 1.0 / (N + 1)
            x = jnp.linspace(dx, 1.0 - dx, N)
            psi_exact = jnp.sin(2 * jnp.pi * x)
            rhs = -((2 * jnp.pi) ** 2) * psi_exact
            psi_got = solve_helmholtz_dst1_1d(rhs, dx, approximation="spectral")
            errors.append(float(jnp.max(jnp.abs(psi_got - psi_exact))))

        assert errors[-1] < 1e-8

    def test_dst2_spectral_convergence(self):
        """Dirichlet (staggered): spectral convergence for smooth f."""
        errors = []
        for N in [8, 16, 32, 64]:
            dx = 1.0 / N
            x = (jnp.arange(N) + 0.5) * dx
            psi_exact = jnp.sin(2 * jnp.pi * x)
            rhs = -((2 * jnp.pi) ** 2) * psi_exact
            psi_got = solve_helmholtz_dst2_1d(rhs, dx, approximation="spectral")
            errors.append(float(jnp.max(jnp.abs(psi_got - psi_exact))))

        assert errors[-1] < 1e-8

    def test_fft_spectral_convergence(self):
        """Periodic: spectral convergence for smooth f."""
        errors = []
        for N in [8, 16, 32, 64]:
            dx = 1.0 / N
            x = jnp.arange(N) * dx
            psi_exact = jnp.sin(2 * jnp.pi * x)
            rhs = -((2 * jnp.pi) ** 2) * psi_exact
            psi_got = solve_helmholtz_fft_1d(rhs, dx, approximation="spectral")
            errors.append(float(jnp.max(jnp.abs(psi_got - psi_exact))))

        assert errors[-1] < 1e-8


class TestFD2ConvergenceRate:
    """With FD2 eigenvalues, convergence should be O(h²)."""

    def test_dst1_second_order(self):
        """Dirichlet (regular): FD2 error decreases as O(N⁻²)."""
        errors = []
        for N in [16, 32, 64, 128]:
            dx = 1.0 / (N + 1)
            x = jnp.linspace(dx, 1.0 - dx, N)
            psi_exact = jnp.sin(2 * jnp.pi * x)
            rhs = -((2 * jnp.pi) ** 2) * psi_exact
            psi_got = solve_helmholtz_dst1_1d(rhs, dx, approximation="fd2")
            errors.append(float(jnp.max(jnp.abs(psi_got - psi_exact))))

        # Check convergence rate: error ratio between successive refinements
        # Should be ~4x for O(h²) when doubling N
        for i in range(len(errors) - 1):
            ratio = errors[i] / errors[i + 1]
            assert ratio > 3.0, f"Expected ~4x reduction, got {ratio:.1f}x"


class TestBackwardsCompatibility:
    """Default approximation='fd2' matches previous behavior exactly."""

    def test_dst1_default_unchanged(self):
        N = 16
        dx = 0.1
        rhs = jnp.ones((N, N))
        # Default (no approximation kwarg) should equal explicit fd2
        result_default = solve_helmholtz_dst(rhs, dx, dx, lambda_=1.0)
        result_fd2 = solve_helmholtz_dst(rhs, dx, dx, lambda_=1.0, approximation="fd2")
        assert jnp.allclose(result_default, result_fd2, atol=1e-12)

    def test_dct_default_unchanged(self):
        N = 16
        dx = 0.1
        rhs = jnp.ones((N, N))
        result_default = solve_helmholtz_dct(rhs, dx, dx, lambda_=1.0)
        result_fd2 = solve_helmholtz_dct(rhs, dx, dx, lambda_=1.0, approximation="fd2")
        assert jnp.allclose(result_default, result_fd2, atol=1e-12)

    def test_dct1_default_unchanged(self):
        N = 16
        dx = 0.1
        rhs = jnp.ones((N, N))
        result_default = solve_helmholtz_dct1(rhs, dx, dx, lambda_=1.0)
        result_fd2 = solve_helmholtz_dct1(rhs, dx, dx, lambda_=1.0, approximation="fd2")
        assert jnp.allclose(result_default, result_fd2, atol=1e-12)


# ===========================================================================
# Input validation
# ===========================================================================


class TestPSValidation:
    """PS eigenvalue functions validate their inputs."""

    @pytest.mark.parametrize(
        "fn",
        [
            dst1_eigenvalues_ps,
            dst2_eigenvalues_ps,
            dct1_eigenvalues_ps,
            dct2_eigenvalues_ps,
            dst3_eigenvalues_ps,
            dct3_eigenvalues_ps,
            dst4_eigenvalues_ps,
            dct4_eigenvalues_ps,
            fft_eigenvalues_ps,
        ],
    )
    def test_rejects_nonpositive_L(self, fn):
        with pytest.raises(ValueError, match="L > 0"):
            fn(8, L=0.0)
        with pytest.raises(ValueError, match="L > 0"):
            fn(8, L=-1.0)

    def test_dct1_ps_rejects_n_less_than_2(self):
        with pytest.raises(ValueError, match="N >= 2"):
            dct1_eigenvalues_ps(1, L=1.0)

    def test_invalid_approximation_raises(self):
        rhs = jnp.ones(16)
        with pytest.raises(ValueError, match="Unknown approximation"):
            solve_helmholtz_dst1_1d(rhs, dx=0.1, approximation="spec")


# ===========================================================================
# DCT spectral convergence (Neumann solvers)
# ===========================================================================


class TestDCTSpectralConvergence:
    """Neumann solvers with PS eigenvalues converge spectrally."""

    def test_dct1_spectral_convergence(self):
        """Neumann (regular / DCT-I): spectral convergence for cos(2*pi*x)."""
        errors = []
        for N in [8, 16, 32, 64]:
            dx = 1.0 / (N - 1)
            x = jnp.linspace(0, 1, N)
            psi_exact = jnp.cos(2 * jnp.pi * x)
            rhs = -((2 * jnp.pi) ** 2) * psi_exact
            psi_got = solve_helmholtz_dct1_1d(rhs, dx, approximation="spectral")
            # Remove mean (null mode)
            psi_got = psi_got - jnp.mean(psi_got)
            psi_exact = psi_exact - jnp.mean(psi_exact)
            errors.append(float(jnp.max(jnp.abs(psi_got - psi_exact))))

        assert errors[-1] < 1e-8

    def test_dct2_spectral_convergence(self):
        """Neumann (staggered / DCT-II): spectral convergence for cos(2*pi*x)."""
        errors = []
        for N in [8, 16, 32, 64]:
            dx = 1.0 / N
            x = (jnp.arange(N) + 0.5) * dx
            psi_exact = jnp.cos(2 * jnp.pi * x)
            rhs = -((2 * jnp.pi) ** 2) * psi_exact
            psi_got = solve_helmholtz_dct2_1d(rhs, dx, approximation="spectral")
            psi_got = psi_got - jnp.mean(psi_got)
            psi_exact = psi_exact - jnp.mean(psi_exact)
            errors.append(float(jnp.max(jnp.abs(psi_got - psi_exact))))

        assert errors[-1] < 1e-8
