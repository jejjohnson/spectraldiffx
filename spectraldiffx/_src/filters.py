# ============================================================================
# Spectral Filters
# ============================================================================


import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .grid import FourierGrid1D, FourierGrid2D, FourierGrid3D


class SpectralFilter1D(eqx.Module):
    """
    1D Spectral filter for smoothing and numerical stabilization.

    Mathematical Formulation:
    -------------------------
    Filters are applied as a multiplicative mask in Fourier space:
        u_hat_filtered(k) = F(k) * u_hat(k)

    where F(k) is the filter kernel.

    Attributes:
    -----------
        grid : FourierGrid1D
            The 1D Fourier grid object containing wavenumbers k [N].
    """

    grid: FourierGrid1D

    def exponential_filter(
        self, u: Array, alpha: float = 36.0, power: int = 16, spectral: bool = False
    ) -> Array:
        """
        Apply 1D exponential filter: F(k) = exp(-alpha * (abs(k) / k_max)^power)

        This filter is near unity for low wavenumbers and falls off sharply
        near the grid scale (k_max), removing poorly resolved high frequencies.

        Parameters:
        -----------
        u : Array [N]
            Physical space field or spectral coefficients of the field.
        alpha : float, optional
            Damping coefficient. Default is 36.0.
        power : int, optional
            Sharpening order (even integer). Default is 16.
        spectral : bool, optional
            If True, u is treated as spectral coefficients (u_hat). Default is False.

        Returns:
        --------
        Array [N]
            Filtered field or spectral coefficients.
        """
        u_hat = u if spectral else self.grid.transform(u)
        k = self.grid.k
        k_max = jnp.abs(k).max()
        k_max_safe = jnp.where(k_max == 0, 1.0, k_max)

        filter_mask = jnp.exp(-alpha * (jnp.abs(k) / k_max_safe) ** power)
        u_hat_f = u_hat * filter_mask
        return u_hat_f if spectral else self.grid.transform(u_hat_f, inverse=True).real

    def hyperviscosity(
        self,
        u: Array,
        nu_hyper: float,
        dt: float,
        power: int = 4,
        spectral: bool = False,
    ) -> Array:
        """
        Apply 1D hyperviscous damping: F(k) = exp(-nu_h * abs(k)^power * dt)

        Simulates the effect of high-order diffusion: du/dt = -nu_h * (-1)^(p/2) * d^p u / dx^p.

        Parameters:
        -----------
        u : Array [N]
            Physical space field or spectral coefficients of the field.
        nu_hyper : float
            Hyperviscosity coefficient.
        dt : float
            Effective time step for the damping.
        power : int, optional
            Order of the Laplacian power (e.g., 4 for biharmonic). Default is 4.
        spectral : bool, optional
            If True, u is treated as spectral coefficients (u_hat). Default is False.
        """
        u_hat = u if spectral else self.grid.transform(u)
        k = self.grid.k
        filter_mask = jnp.exp(-nu_hyper * jnp.abs(k) ** power * dt)
        u_hat_f = u_hat * filter_mask
        return u_hat_f if spectral else self.grid.transform(u_hat_f, inverse=True).real


class SpectralFilter2D(eqx.Module):
    """
    2D Spectral filter for doubly periodic domains.

    Attributes:
    -----------
        grid : FourierGrid2D
            The 2D Fourier grid object [Ny, Nx].
    """

    grid: FourierGrid2D

    def exponential_filter(
        self, u: Array, alpha: float = 36.0, power: int = 16, spectral: bool = False
    ) -> Array:
        """Apply 2D exponential filter based on isotropic wavenumber magnitude."""
        u_hat = u if spectral else self.grid.transform(u)
        K2 = self.grid.K2
        k_mag = jnp.sqrt(K2)
        k_max = k_mag.max()
        k_max_safe = jnp.where(k_max == 0, 1.0, k_max)

        filter_mask = jnp.exp(-alpha * (k_mag / k_max_safe) ** power)
        u_hat_f = u_hat * filter_mask
        return u_hat_f if spectral else self.grid.transform(u_hat_f, inverse=True).real

    def hyperviscosity(
        self,
        u: Array,
        nu_hyper: float,
        dt: float,
        power: int = 4,
        spectral: bool = False,
    ) -> Array:
        """Apply 2D hyperviscosity: F(k) = exp(-nu_h * |k|^power * dt)"""
        u_hat = u if spectral else self.grid.transform(u)
        K2 = self.grid.K2
        filter_mask = jnp.exp(-nu_hyper * K2 ** (power / 2) * dt)
        u_hat_f = u_hat * filter_mask
        return u_hat_f if spectral else self.grid.transform(u_hat_f, inverse=True).real


class SpectralFilter3D(eqx.Module):
    """
    3D Spectral filter for triply periodic domains.

    Attributes:
    -----------
        grid : FourierGrid3D
            The 3D Fourier grid object [Nz, Ny, Nx].
    """

    grid: FourierGrid3D

    def exponential_filter(
        self, u: Array, alpha: float = 36.0, power: int = 16, spectral: bool = False
    ) -> Array:
        """Apply isotropic 3D exponential filter."""
        u_hat = u if spectral else self.grid.transform(u)
        K2 = self.grid.K2
        k_mag = jnp.sqrt(K2)
        k_max = k_mag.max()
        k_max_safe = jnp.where(k_max == 0, 1.0, k_max)

        filter_mask = jnp.exp(-alpha * (k_mag / k_max_safe) ** power)
        u_hat_f = u_hat * filter_mask
        return u_hat_f if spectral else self.grid.transform(u_hat_f, inverse=True).real

    def hyperviscosity(
        self,
        u: Array,
        nu_hyper: float,
        dt: float,
        power: int = 4,
        spectral: bool = False,
    ) -> Array:
        """Apply 3D hyperviscosity: exp(-nu_h * |k|^power * dt)"""
        u_hat = u if spectral else self.grid.transform(u)
        K2 = self.grid.K2
        filter_mask = jnp.exp(-nu_hyper * K2 ** (power / 2) * dt)
        u_hat_f = u_hat * filter_mask
        return u_hat_f if spectral else self.grid.transform(u_hat_f, inverse=True).real
