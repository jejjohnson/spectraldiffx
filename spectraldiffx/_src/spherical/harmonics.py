"""
Spherical Harmonic Transform
=============================

Wraps the full 2D Spherical Harmonic Transform with precomputed Associated
Legendre Polynomial matrices.

Normalisation convention
------------------------
All ALPs are *orthonormal* (Schmidt semi-normalised including the extra
√((2l+1)/2) factor), so that for the Gauss–Legendre weights wⱼ:

    Σⱼ wⱼ · P̃ₗᵐ(cos θⱼ) · P̃_{l'}^m(cos θⱼ) = δ_{l, l'}

with

    P̃ₗᵐ(cos θ) = Nₗ,ₘ · Pₗᵐ(cos θ),    Nₗ,ₘ = √((2l+1)/2 · (l−m)! / (l+m)!).

Equivalently, the spherical harmonics Yₗᵐ are 4π-normalised in the real
convention (no Condon–Shortley phase for m > 0 beyond what scipy's
``lpmv`` supplies).  Work out the forward/inverse formulas below.

References:
-----------
[1] Boyd, J. P. (2001). Chebyshev and Fourier Spectral Methods.
[3] Canuto et al. (2006). Spectral Methods: Fundamentals.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Complex, Float

from .grid import SphericalGrid2D

# Array-shape aliases:
#   "Nlat Nlon"  — full lat-lon grid (Nlat = Ny, Nlon = Nx)
#   "Nl Nm"      — spectral (l, m) grid.  In this implementation Nl = Nlat
#                  and Nm = Nlon for triangular truncation T_{Nlat-1}.


class SphericalHarmonicTransform(eqx.Module):
    """Full 2D Spherical Harmonic Transform (SHT).

    Wraps a :class:`SphericalGrid2D` with forward/inverse methods.  The
    Associated Legendre Polynomial matrices are precomputed once at
    construction time (scipy call) and stored as JAX arrays.

    Mathematical Formulation
    ------------------------
    Expansion in orthonormal spherical harmonics:

        u(θ, φ) = Σₗ Σₘ û(l, m) · P̃ₗᵐ(cos θ) · e^{imφ}

    Forward SHT:

        Step 1 (FFT in longitude):
            ûₘ(θⱼ) = Σₖ u(θⱼ, φₖ) · e^{−imφₖ}
        Step 2 (Legendre transform in colatitude):
            û(l, m) = Σⱼ wⱼ · P̃ₗᵐ(cos θⱼ) · ûₘ(θⱼ)

    Inverse SHT:

        Step 1 (synthesis in θ):
            ûₘ(θⱼ) = Σₗ û(l, m) · P̃ₗᵐ(cos θⱼ)
        Step 2 (IFFT in φ):
            u(θⱼ, φₖ) = IFFTₖ(ûₘ(θⱼ))

    Attributes
    ----------
    grid : SphericalGrid2D
        Underlying lat-lon grid carrying the ALP matrices.

    Notes
    -----
    The ``_P_lm`` attribute lives on :attr:`grid`; this class is a thin,
    more-semantic wrapper around :meth:`SphericalGrid2D.transform`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> grid = SphericalGrid2D.from_N_L(Nx=32, Ny=16)
    >>> sht = SphericalHarmonicTransform(grid=grid)
    >>> PHI, THETA = grid.X
    >>> u = jnp.sin(THETA) * jnp.cos(PHI)  # = Re Y₁¹(θ, φ) up to a constant
    >>> u_hat = sht.to_spectral(u)
    >>> u_back = sht.from_spectral(u_hat)  # ≈ u
    """

    grid: SphericalGrid2D

    def to_spectral(self, u: Float[Array, "Nlat Nlon"]) -> Complex[Array, "Nl Nm"]:
        """Forward SHT: physical u(θ, φ) → spectral û(l, m).

            û(l, m) = Σⱼ wⱼ · P̃ₗᵐ(cos θⱼ) · FFTφ{u(θⱼ, ·)}[m]

        Parameters
        ----------
        u : Float[Array, "Nlat Nlon"]
            Real-valued physical field on the (Nlat, Nlon) lat-lon grid.

        Returns
        -------
        Complex[Array, "Nl Nm"]
            Spectral coefficients.  Rows index harmonic degree l;
            columns index zonal wavenumber m in FFT order.
        """
        return self.grid.transform(u, inverse=False)

    def from_spectral(
        self, u_hat: Complex[Array, "Nl Nm"]
    ) -> Float[Array, "Nlat Nlon"]:
        """Inverse SHT: spectral û(l, m) → physical u(θ, φ)."""
        return self.grid.transform(u_hat, inverse=True)

    def to_spectral_1d(self, u_col: Float[Array, "Nlat"]) -> Float[Array, "Nl"]:
        """1D forward Discrete Legendre Transform (zonal mean, m=0)."""
        # m=0 is at fft_idx=0
        P0 = self.grid._P_lm[0]  # (Nl, Nlat)
        w = self.grid._weights_y  # (Nlat,)
        return P0 @ (w * u_col)

    def from_spectral_1d(self, c: Float[Array, "Nl"]) -> Float[Array, "Nlat"]:
        """1D inverse DLT (zonal mean, m=0)."""
        P0 = self.grid._P_lm[0]  # (Nl, Nlat)
        return P0.T @ c
