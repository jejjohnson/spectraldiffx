"""Normalised associated Legendre tables of the spherical grid (gh-96).

The tables used to be built from ``scipy.special.lpmv``, the *unnormalised*
Pₗᵐ, which overflows float64 at (m, l) = (85, 86): every transform on a grid
at T86 or above returned NaN. They are now built with the normalised
three-term recurrence (``_alp_tables``). These tests pin the convention
against the old construction where it was finite, check orthonormality on
the Gauss rule at T127 and T255, compare against ``scipy.special.sph_harm_y``
at high degree and order, and cover the public ``legendre_table`` accessor.
"""

import numpy as np
import pytest
from scipy.special import gammaln, lpmv, roots_legendre, sph_harm_y

import spectraldiffx as sdx
from spectraldiffx._src.spherical.grid import _alp_tables


def _lpmv_table(m, l_max, mu):
    """The pre-gh-96 construction, finite only for m + l below the overflow."""
    P = np.zeros((l_max + 1, mu.size))
    for l in range(m, l_max + 1):
        log_norm_sq = np.log((2 * l + 1) / 2) + gammaln(l - m + 1) - gammaln(l + m + 1)
        P[l] = np.exp(0.5 * log_norm_sq) * lpmv(m, l, mu)
    return P


def _gram_error(table, weights):
    """max over m of ‖Gₘ − I‖, Gₘ[l, l'] = Σⱼ wⱼ P̃ₗᵐ(μⱼ) P̃ₗ′ᵐ(μⱼ), l, l' ≥ m."""
    n_l = table.shape[1]
    err = 0.0
    for m in range(min(table.shape[0], n_l)):
        rows = table[m, m:]
        gram = (rows * weights) @ rows.T
        err = max(err, np.abs(gram - np.eye(n_l - m)).max())
    return err


def test_matches_previous_construction_below_overflow():
    # T31 grid: lpmv is finite here, so the old tables are the reference for
    # normalisation and phase. Measured difference 6e-14; 1e-12 leaves margin.
    grid = sdx.SphericalGrid2D.from_N_L(Nx=64, Ny=32)
    table = np.asarray(grid.legendre_table)
    mu = np.asarray(grid.cos_theta)
    for m in range(table.shape[0]):
        np.testing.assert_allclose(table[m], _lpmv_table(m, 31, mu), rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "n_lat", [128, pytest.param(256, marks=pytest.mark.slow)], ids=["T127", "T255"]
)
def test_orthonormal_on_gauss_rule(n_lat):
    # Measured 1.8e-12 (T127) and 2.7e-12 (T255); 1e-11 leaves margin.
    grid = sdx.SphericalGrid2D.from_N_L(Nx=2 * n_lat, Ny=n_lat)
    table = np.asarray(grid.legendre_table)
    assert np.isfinite(table).all()
    assert _gram_error(table, np.asarray(grid.weights_y)) < 1e-11


def test_matches_sph_harm_y_at_high_degree():
    # P̃ₗᵐ(cos θ) = √(2π) · Yₗᵐ(θ, φ = 0) for scipy's orthonormal complex Yₗᵐ,
    # both with the Condon–Shortley phase. (l, m) = (200, 150) is far past
    # the old overflow at m ≈ 85. Measured difference 4e-14.
    mu, _ = roots_legendre(256)
    theta = np.arccos(mu)
    P = _alp_tables(np.array([150]), 200, mu)[0, 200]
    expected = np.sqrt(2 * np.pi) * np.real(sph_harm_y(200, 150, theta, 0.0))
    np.testing.assert_allclose(P, expected, rtol=0, atol=1e-10)


def test_legendre_table_accessor():
    grid = sdx.SphericalGrid2D.from_N_L(Nx=16, Ny=8)
    table = np.asarray(grid.legendre_table)
    full = np.asarray(grid._P_lm)
    assert table.shape == (16 // 2 + 1, 8, 8)
    # Row m of the accessor is the FFT-order table for wavenumbers +m and −m.
    for m in range(1, 8):
        np.testing.assert_array_equal(table[m], full[m])
        np.testing.assert_array_equal(table[m], full[-m])
    # Orders above the degree range have all-zero tables.
    assert not table[8].any()


def test_alp_tables_zero_below_order_and_for_large_order():
    P = _alp_tables(np.array([3, 9]), 6, np.linspace(-0.9, 0.9, 5))
    assert not P[0, :3].any()
    assert not P[1].any()
