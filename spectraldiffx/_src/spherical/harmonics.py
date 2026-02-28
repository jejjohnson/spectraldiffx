"""
Spherical Harmonic Transform
=============================

Wraps the full 2D Spherical Harmonic Transform with precomputed Associated
Legendre Polynomial matrices.

References:
-----------
[1] Boyd, J. P. (2001). Chebyshev and Fourier Spectral Methods.
[3] Canuto et al. (2006). Spectral Methods: Fundamentals.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from .grid import SphericalGrid2D


class SphericalHarmonicTransform(eqx.Module):
    """
    Full 2D Spherical Harmonic Transform.

    This class wraps a SphericalGrid2D and provides forward/inverse SHT methods
    with a clean API.  The Associated Legendre Polynomial matrices are
    precomputed at construction time (scipy call) and stored as JAX arrays.

    Mathematical Formulation:
    -------------------------
    The Spherical Harmonic expansion:
        u(theta, phi) = sum_{l=0}^{L} sum_{m=-l}^{l} u_hat(l, m) * Y_l^m(theta, phi)

    where Y_l^m(theta, phi) = P_l^m_norm(cos(theta)) * exp(i*m*phi) are the
    complex spherical harmonics with normalised ALPs.

    Forward SHT:
        Step 1: FFT in phi — u_m(theta_j) = sum_k u(theta_j, phi_k) * exp(-i*m*phi_k)
        Step 2: Legendre transform — u_hat(l, m) = sum_j w_j * P_l^m_norm(cos(theta_j)) * u_m(theta_j)

    Inverse SHT:
        Step 1: Inverse Legendre — u_m(theta_j) = sum_l P_l^m_norm(cos(theta_j)) * u_hat(l, m)
        Step 2: IFFT in phi — u(theta_j, phi_k) = ifft(u_m(theta_j, :))[k]

    Attributes:
    -----------
    grid : SphericalGrid2D
        The underlying lat-lon grid.

    Notes:
    ------
    The _P_lm attribute is inherited from grid._P_lm (shape Nx, Ny, Ny).
    This class exposes a slightly different interface for direct use without
    going through the grid transform method.
    """

    grid: SphericalGrid2D

    def to_spectral(
        self, u: Float[Array, "Ny Nx"]
    ) -> Complex[Array, "Ny Nx"]:
        """
        Forward Spherical Harmonic Transform: u(theta, phi) -> u_hat(l, m).

        Parameters:
        -----------
        u : Float[Array, "Ny Nx"]
            Physical field on the (Ny, Nx) lat-lon grid.

        Returns:
        --------
        u_hat : Complex[Array, "Ny Nx"]
            Spectral coefficients u_hat(l, m_fft_idx).
            Rows index spherical harmonic degree l, columns index m in FFT order.
        """
        return self.grid.transform(u, inverse=False)

    def from_spectral(
        self, u_hat: Complex[Array, "Ny Nx"]
    ) -> Float[Array, "Ny Nx"]:
        """
        Inverse Spherical Harmonic Transform: u_hat(l, m) -> u(theta, phi).

        Parameters:
        -----------
        u_hat : Complex[Array, "Ny Nx"]
            Spectral coefficients as returned by to_spectral().

        Returns:
        --------
        u : Float[Array, "Ny Nx"]
            Reconstructed physical field.
        """
        return self.grid.transform(u_hat, inverse=True)

    def to_spectral_1d(
        self, u_col: Float[Array, "Ny"]
    ) -> Float[Array, "Ny"]:
        """
        1D forward Discrete Legendre Transform for a single column (m=0).

        Uses the Legendre polynomial matrix from the grid's first column (m=0).

        Parameters:
        -----------
        u_col : Float[Array, "Ny"]
            Physical values at the Gauss-Legendre latitudes.

        Returns:
        --------
        c : Float[Array, "Ny"]
            Legendre spectral coefficients c_l.
        """
        # m=0 is at fft_idx=0
        P0 = self.grid._P_lm[0]  # (Ny, Ny)
        w = self.grid._weights_y  # (Ny,)
        return P0 @ (w * u_col)

    def from_spectral_1d(
        self, c: Float[Array, "Ny"]
    ) -> Float[Array, "Ny"]:
        """
        1D inverse Discrete Legendre Transform (m=0 case).

        Parameters:
        -----------
        c : Float[Array, "Ny"]
            Legendre spectral coefficients.

        Returns:
        --------
        u_col : Float[Array, "Ny"]
            Reconstructed physical values.
        """
        P0 = self.grid._P_lm[0]  # (Ny, Ny)
        return P0.T @ c
