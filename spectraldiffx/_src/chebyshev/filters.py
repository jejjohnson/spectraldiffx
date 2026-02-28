# ============================================================================
# Chebyshev Spectral Filters
# ============================================================================

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .grid import ChebyshevGrid1D, ChebyshevGrid2D


class ChebyshevFilter1D(eqx.Module):
    """
    1D Chebyshev spectral filter for smoothing and numerical stabilization.

    Mathematical Formulation:
    -------------------------
    Filters are applied as a multiplicative mask in Chebyshev coefficient space:

        a_k_filtered = F(k) * a_k

    where a_k are Chebyshev expansion coefficients and F(k) is the filter kernel.

    Exponential filter:
        F(k) = exp(-alpha * (k/N)^power)

    Hyperviscosity filter:
        F(k) = exp(-nu_h * k^power * dt)

    Attributes:
    -----------
        grid : ChebyshevGrid1D
            The 1D Chebyshev grid for forward/inverse transforms.
    """

    grid: ChebyshevGrid1D

    def exponential_filter(
        self,
        u: Array,
        alpha: float = 36.0,
        power: int = 16,
        spectral: bool = False,
    ) -> Array:
        """
        Apply 1D exponential filter in Chebyshev mode space.

            F(k) = exp(-alpha * (k/N)^power)

        This filter is near unity for low modes and falls off sharply near
        the highest Chebyshev mode k=N, removing poorly-resolved content.

        Parameters:
        -----------
        u : Array [N1]
            Physical-space field or Chebyshev coefficients.
        alpha : float
            Damping strength. Default 36.0 (≈ machine-epsilon damping at k=N).
        power : int
            Sharpening exponent (even integer). Default 16.
        spectral : bool
            If True, u is treated as Chebyshev coefficients. Default False.

        Returns:
        --------
        Array [N1]
            Filtered field (physical if spectral=False, else coefficients).
        """
        a = u if spectral else self.grid.transform(u)
        N = self.grid.N
        n_modes = N + 1 if self.grid.node_type == "gauss-lobatto" else N
        k = jnp.arange(n_modes, dtype=jnp.float32)
        filter_mask = jnp.exp(-alpha * (k / N) ** power)
        a_f = a * filter_mask
        return a_f if spectral else self.grid.transform(a_f, inverse=True)

    def hyperviscosity(
        self,
        u: Array,
        nu_hyper: float,
        dt: float,
        power: int = 4,
        spectral: bool = False,
    ) -> Array:
        """
        Apply 1D hyperviscous damping in Chebyshev mode space.

            F(k) = exp(-nu_h * k^power * dt)

        Simulates high-order diffusion: du/dt = (-1)^{p/2} nu_h d^p u/dx^p.

        Parameters:
        -----------
        u : Array [N1]
            Physical-space field or Chebyshev coefficients.
        nu_hyper : float
            Hyperviscosity coefficient.
        dt : float
            Time step for the damping.
        power : int
            Diffusion order. Default 4 (biharmonic).
        spectral : bool
            If True, u is treated as Chebyshev coefficients. Default False.

        Returns:
        --------
        Array [N1]
        """
        a = u if spectral else self.grid.transform(u)
        N = self.grid.N
        n_modes = N + 1 if self.grid.node_type == "gauss-lobatto" else N
        k = jnp.arange(n_modes, dtype=jnp.float32)
        filter_mask = jnp.exp(-nu_hyper * k**power * dt)
        a_f = a * filter_mask
        return a_f if spectral else self.grid.transform(a_f, inverse=True)


class ChebyshevFilter2D(eqx.Module):
    """
    2D Chebyshev spectral filter for smoothing on [-Lx, Lx] × [-Ly, Ly].

    Applies separable 1D exponential or hyperviscosity filters along each axis.

    Attributes:
    -----------
        grid : ChebyshevGrid2D
            The 2D Chebyshev grid.
    """

    grid: ChebyshevGrid2D

    def exponential_filter(
        self,
        u: Array,
        alpha: float = 36.0,
        power: int = 16,
        spectral: bool = False,
    ) -> Array:
        """
        Apply 2D separable exponential filter.

            F(kx, ky) = exp(-alpha * (kx/Nx)^power) * exp(-alpha * (ky/Ny)^power)

        Parameters:
        -----------
        u : Array [Ny_pts, Nx_pts]
            Physical-space field or Chebyshev coefficients.
        alpha : float
            Damping strength. Default 36.0.
        power : int
            Sharpening exponent. Default 16.
        spectral : bool
            If True, u is treated as spectral coefficients. Default False.

        Returns:
        --------
        Array [Ny_pts, Nx_pts]
        """
        a = u if spectral else self.grid.transform(u)
        Nx, Ny = self.grid.Nx, self.grid.Ny
        nx_modes = Nx + 1 if self.grid.node_type == "gauss-lobatto" else Nx
        ny_modes = Ny + 1 if self.grid.node_type == "gauss-lobatto" else Ny
        kx = jnp.arange(nx_modes, dtype=jnp.float32)
        ky = jnp.arange(ny_modes, dtype=jnp.float32)
        Fx = jnp.exp(-alpha * (kx / Nx) ** power)
        Fy = jnp.exp(-alpha * (ky / Ny) ** power)
        # Apply separable filter: F = Fy[:, None] * Fx[None, :]
        a_f = a * Fy[:, None] * Fx[None, :]
        return a_f if spectral else self.grid.transform(a_f, inverse=True)

    def hyperviscosity(
        self,
        u: Array,
        nu_hyper: float,
        dt: float,
        power: int = 4,
        spectral: bool = False,
    ) -> Array:
        """
        Apply 2D separable hyperviscosity filter.

            F(kx, ky) = exp(-nu_h * (kx^power + ky^power) * dt)

        Parameters:
        -----------
        u : Array [Ny_pts, Nx_pts]
        nu_hyper : float
        dt : float
        power : int
        spectral : bool

        Returns:
        --------
        Array [Ny_pts, Nx_pts]
        """
        a = u if spectral else self.grid.transform(u)
        Nx, Ny = self.grid.Nx, self.grid.Ny
        nx_modes = Nx + 1 if self.grid.node_type == "gauss-lobatto" else Nx
        ny_modes = Ny + 1 if self.grid.node_type == "gauss-lobatto" else Ny
        kx = jnp.arange(nx_modes, dtype=jnp.float32)
        ky = jnp.arange(ny_modes, dtype=jnp.float32)
        # Separable: exp(-nu * kx^p * dt) * exp(-nu * ky^p * dt)
        Fx = jnp.exp(-nu_hyper * kx**power * dt)
        Fy = jnp.exp(-nu_hyper * ky**power * dt)
        a_f = a * Fy[:, None] * Fx[None, :]
        return a_f if spectral else self.grid.transform(a_f, inverse=True)
