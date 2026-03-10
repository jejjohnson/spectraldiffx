"""
Tests for SphericalHarmonicTransform.
"""

import jax.numpy as jnp

from spectraldiffx._src.spherical.grid import SphericalGrid2D
from spectraldiffx._src.spherical.harmonics import SphericalHarmonicTransform


def test_sht_to_spectral_from_spectral_roundtrip():
    """
    SphericalHarmonicTransform.to_spectral then from_spectral must recover
    the original field.
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    sht = SphericalHarmonicTransform(g)
    PHI, THETA = g.X
    u = jnp.sin(THETA) * jnp.cos(PHI) + 0.5 * jnp.cos(2 * THETA)

    u_hat = sht.to_spectral(u)
    u_rec = sht.from_spectral(u_hat)

    assert jnp.allclose(u, u_rec, atol=1e-12), (
        f"SHT class roundtrip: max error = {float(jnp.max(jnp.abs(u_rec - u))):.2e}"
    )


def test_sht_to_spectral_1d_from_spectral_1d_roundtrip():
    """
    SphericalHarmonicTransform.to_spectral_1d then from_spectral_1d must
    recover the original 1D (zonal) field.
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    sht = SphericalHarmonicTransform(g)
    theta = g.y
    # Field with polynomial theta-dependence (exact in the Legendre basis)
    u_col = jnp.cos(theta) + 0.3 * (1.0 - jnp.cos(theta) ** 2)

    c = sht.to_spectral_1d(u_col)
    u_rec = sht.from_spectral_1d(c)

    assert jnp.allclose(u_col, u_rec, atol=1e-12), (
        f"SHT 1D roundtrip: max error = {float(jnp.max(jnp.abs(u_rec - u_col))):.2e}"
    )


def test_sht_class_consistency_with_grid_transform():
    """
    SphericalHarmonicTransform.to_spectral/from_spectral must produce the
    same results as SphericalGrid2D.transform (they are thin wrappers).
    """
    Nx, Ny = 32, 16
    g = SphericalGrid2D.from_N_L(Nx, Ny)
    sht = SphericalHarmonicTransform(g)
    PHI, THETA = g.X
    u = jnp.sin(THETA) ** 2 * jnp.cos(PHI)

    u_hat_sht = sht.to_spectral(u)
    u_hat_grid = g.transform(u)
    assert jnp.allclose(u_hat_sht, u_hat_grid, atol=1e-15), (
        "SHT.to_spectral must match SphericalGrid2D.transform"
    )

    u_rec_sht = sht.from_spectral(u_hat_sht)
    u_rec_grid = g.transform(u_hat_grid, inverse=True)
    assert jnp.allclose(u_rec_sht, u_rec_grid, atol=1e-15), (
        "SHT.from_spectral must match SphericalGrid2D.transform(inverse=True)"
    )
