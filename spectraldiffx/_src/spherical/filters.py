"""
Spherical Spectral Filters
============================

Spectral filters for smoothing and numerical stabilisation on the sphere,
applied in Legendre/SHT coefficient space.

References:
-----------
[1] Boyd, J. P. (2001). Chebyshev and Fourier Spectral Methods.
[4] Durran, D. R. (2010). Numerical Methods for Fluid Dynamics.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .grid import SphericalGrid1D, SphericalGrid2D


class SphericalFilter1D(eqx.Module):
    """
    1D spectral filter on the Gauss-Legendre latitude grid.

    Filters are applied as multiplicative masks in Legendre coefficient space:
        c_l_filtered = F(l) * c_l

    Attributes:
    -----------
    grid : SphericalGrid1D
        The 1D Gauss-Legendre grid.
    """

    grid: SphericalGrid1D

    def exponential_filter(
        self,
        u: Float[Array, "N"],
        alpha: float = 36.0,
        power: int = 16,
        spectral: bool = False,
    ) -> Float[Array, "N"]:
        """
        Apply exponential filter in Legendre coefficient space.

            F(l) = exp(-alpha * (l / l_max)^power)

        Near unity for low degrees, falls sharply near l_max.

        Parameters:
        -----------
        u : Float[Array, "N"]
            Physical field or Legendre coefficients (if spectral=True).
        alpha : float
            Damping coefficient. Default is 36.0.
        power : int
            Filter sharpness (even integer). Default is 16.
        spectral : bool
            If True, treat u as Legendre coefficients.

        Returns:
        --------
        Array [N]
            Filtered field or spectral coefficients.
        """
        c = u if spectral else self.grid.transform(u)
        l = self.grid.l
        l_max = float(self.grid.N - 1)
        l_max_safe = jnp.where(l_max == 0, 1.0, l_max)
        mask = jnp.exp(-alpha * (l / l_max_safe) ** power)
        c_f = c * mask
        return c_f if spectral else self.grid.transform(c_f, inverse=True)

    def hyperviscosity(
        self,
        u: Float[Array, "N"],
        nu_hyper: float,
        dt: float,
        power: int = 4,
        spectral: bool = False,
    ) -> Float[Array, "N"]:
        """
        Apply hyperviscous damping in Legendre coefficient space.

            F(l) = exp(-nu_hyper * [l*(l+1)]^(power/2) * dt)

        Simulates high-order diffusion: du/dt = (-1)^(p+1) * nu_h * nabla^p u.

        Parameters:
        -----------
        u : Float[Array, "N"]
            Physical field or Legendre coefficients.
        nu_hyper : float
            Hyperviscosity coefficient.
        dt : float
            Time step for the damping.
        power : int
            Order of the Laplacian power (e.g., 4 for biharmonic). Default is 4.
        spectral : bool
            If True, treat u as Legendre coefficients.

        Returns:
        --------
        Array [N]
        """
        c = u if spectral else self.grid.transform(u)
        l = self.grid.l
        eigenval = l * (l + 1)
        mask = jnp.exp(-nu_hyper * eigenval ** (power / 2.0) * dt)
        c_f = c * mask
        return c_f if spectral else self.grid.transform(c_f, inverse=True)


class SphericalFilter2D(eqx.Module):
    """
    2D spectral filter on the full sphere lat-lon grid.

    Filters are applied in SHT coefficient space using the spherical harmonic
    degree l as the spectral index.

    Attributes:
    -----------
    grid : SphericalGrid2D
        The 2D lat-lon grid.
    """

    grid: SphericalGrid2D

    def exponential_filter(
        self,
        u: Float[Array, "Ny Nx"],
        alpha: float = 36.0,
        power: int = 16,
        spectral: bool = False,
    ) -> Float[Array, "Ny Nx"]:
        """
        Apply 2D exponential filter using spherical harmonic degree l.

            F(l, m) = exp(-alpha * (l / l_max)^power)

        Parameters:
        -----------
        u : Float[Array, "Ny Nx"]
            Physical field or SHT coefficients.
        alpha : float
            Damping coefficient. Default is 36.0.
        power : int
            Filter sharpness. Default is 16.
        spectral : bool
            If True, treat u as SHT coefficients.

        Returns:
        --------
        Array [Ny, Nx]
            Filtered field or spectral coefficients.
        """
        u_hat = u if spectral else self.grid.transform(u)
        l = self.grid.l  # (Ny,)
        l_max = float(self.grid.Ny - 1)
        l_max_safe = jnp.where(l_max == 0.0, 1.0, l_max)
        mask = jnp.exp(-alpha * (l / l_max_safe) ** power)  # (Ny,)
        u_hat_f = u_hat * mask[:, None]  # broadcast over m
        return u_hat_f if spectral else self.grid.transform(u_hat_f, inverse=True)

    def hyperviscosity(
        self,
        u: Float[Array, "Ny Nx"],
        nu_hyper: float,
        dt: float,
        power: int = 4,
        spectral: bool = False,
    ) -> Float[Array, "Ny Nx"]:
        """
        Apply 2D hyperviscous damping using l*(l+1) eigenvalues.

            F(l) = exp(-nu_hyper * [l*(l+1)]^(power/2) * dt)

        Parameters:
        -----------
        u : Float[Array, "Ny Nx"]
            Physical field or SHT coefficients.
        nu_hyper : float
            Hyperviscosity coefficient.
        dt : float
            Time step.
        power : int
            Laplacian power order. Default is 4.
        spectral : bool
            If True, treat u as SHT coefficients.

        Returns:
        --------
        Array [Ny, Nx]
        """
        u_hat = u if spectral else self.grid.transform(u)
        l = self.grid.l  # (Ny,)
        eigenval = l * (l + 1)
        mask = jnp.exp(-nu_hyper * eigenval ** (power / 2.0) * dt)  # (Ny,)
        u_hat_f = u_hat * mask[:, None]
        return u_hat_f if spectral else self.grid.transform(u_hat_f, inverse=True)
