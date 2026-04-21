"""
Spherical Spectral Filters
============================

Spectral filters for smoothing and numerical stabilisation on the sphere,
applied in Legendre / SHT coefficient space.  The eigenvalues of the
Laplace–Beltrami operator, λₗ = −l(l+1)/R², appear in the hyperviscosity
kernel so that the damping rate is intrinsic to the sphere geometry.

References
----------
[1] Boyd, J. P. (2001). Chebyshev and Fourier Spectral Methods.
[4] Durran, D. R. (2010). Numerical Methods for Fluid Dynamics.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Num

from .grid import SphericalGrid1D, SphericalGrid2D

# Array-shape aliases:
#   "N"          — 1D GL latitude grid / 1D Legendre spectrum
#   "Nlat Nlon"  — 2D lat-lon grid / SHT coefficients (Nl = Nlat, Nm = Nlon)


class SphericalFilter1D(eqx.Module):
    """1D spectral filter on the Gauss–Legendre latitude grid.

    Filters multiply each Legendre coefficient by a kernel F(l):

        c̃ₗ = F(l) · cₗ

    Attributes
    ----------
    grid : SphericalGrid1D
        Underlying 1D Gauss–Legendre grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> grid = SphericalGrid1D.from_N_L(N=64, L=jnp.pi)
    >>> flt = SphericalFilter1D(grid=grid)
    >>> u = jnp.cos(grid.x) + 1e-3 * jnp.cos(60 * grid.x)
    >>> u_smooth = flt.exponential_filter(u)
    """

    grid: SphericalGrid1D

    def exponential_filter(
        self,
        u: Num[Array, N],
        alpha: float = 36.0,
        power: int = 16,
        spectral: bool = False,
    ) -> Num[Array, N]:
        """Exponential filter in Legendre coefficient space:

            F(l) = exp(−α · (l / lₘₐₓ)ᵖ)

        Near unity for low l, falls sharply near lₘₐₓ = N−1.

        Parameters
        ----------
        u : Num[Array, "N"]
            Physical field or Legendre coefficients (if ``spectral=True``).
        alpha : float
            Damping coefficient (≥ 0).  Default 36.0 (≈ ε_64 damping at lₘₐₓ).
        power : int
            Filter sharpness (> 0, even).  Default 16.
        spectral : bool
            If ``True``, treat ``u`` as Legendre coefficients.

        Returns
        -------
        Num[Array, "N"]
        """
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if power <= 0:
            raise ValueError(f"power must be > 0, got {power}")
        c = u if spectral else self.grid.transform(u)
        l = self.grid.l
        l_max = float(self.grid.N - 1)
        l_max_safe = jnp.where(l_max == 0, 1.0, l_max)
        mask = jnp.exp(-alpha * (l / l_max_safe) ** power)
        c_f = c * mask
        return c_f if spectral else self.grid.transform(c_f, inverse=True)

    def hyperviscosity(
        self,
        u: Num[Array, N],
        nu_hyper: float,
        dt: float,
        power: int = 4,
        spectral: bool = False,
    ) -> Num[Array, N]:
        """Hyperviscous damping driven by the Laplace–Beltrami eigenvalue:

            F(l) = exp(−ν_h · [l(l+1)/R²]^(p/2) · Δt)

        where R = ``grid.L / π`` is the sphere radius.  Simulates
        high-order diffusion ``∂u/∂t = (−1)^{p+1} ν_h ∇^p u`` with the
        damping rate independent of R for fixed ν_h.

        Parameters
        ----------
        u : Num[Array, "N"]
            Physical field or Legendre coefficients.
        nu_hyper : float
            Hyperviscosity coefficient (≥ 0).
        dt : float
            Time step (≥ 0).
        power : int
            Laplacian power p (> 0, even).  Default 4 (biharmonic).
        spectral : bool
            If ``True``, treat ``u`` as Legendre coefficients.

        Returns
        -------
        Num[Array, "N"]
        """
        if nu_hyper < 0:
            raise ValueError(f"nu_hyper must be >= 0, got {nu_hyper}")
        if power <= 0:
            raise ValueError(f"power must be > 0, got {power}")
        c = u if spectral else self.grid.transform(u)
        R = self.grid.L / jnp.pi
        l = self.grid.l
        eigenval = l * (l + 1) / (R**2)
        mask = jnp.exp(-nu_hyper * eigenval ** (power / 2.0) * dt)
        c_f = c * mask
        return c_f if spectral else self.grid.transform(c_f, inverse=True)


class SphericalFilter2D(eqx.Module):
    """2D spectral filter on the full sphere lat-lon grid.

    Applies multiplicative kernels F(l) in spherical-harmonic-coefficient
    space (broadcast across all m for each l).

    Attributes
    ----------
    grid : SphericalGrid2D
        Underlying 2D lat-lon grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> grid = SphericalGrid2D.from_N_L(Nx=64, Ny=32)
    >>> flt = SphericalFilter2D(grid=grid)
    >>> PHI, THETA = grid.X
    >>> u = jnp.sin(THETA) * jnp.cos(4 * PHI)
    >>> u_smooth = flt.exponential_filter(u, alpha=16.0, power=8)
    """

    grid: SphericalGrid2D

    def exponential_filter(
        self,
        u: Num[Array, "Nlat Nlon"],
        alpha: float = 36.0,
        power: int = 16,
        spectral: bool = False,
    ) -> Num[Array, "Nlat Nlon"]:
        """Exponential filter in (l, m) space:

            F(l, m) = exp(−α · (l / lₘₐₓ)ᵖ)

        Parameters
        ----------
        u : Num[Array, "Nlat Nlon"]
            Physical field or SHT coefficients.
        alpha : float
            Damping coefficient (≥ 0).
        power : int
            Filter sharpness (> 0).
        spectral : bool
            If ``True``, treat ``u`` as SHT coefficients.
        """
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if power <= 0:
            raise ValueError(f"power must be > 0, got {power}")
        u_hat = u if spectral else self.grid.transform(u)
        l = self.grid.l  # (Nl,)
        l_max = float(self.grid.Ny - 1)
        l_max_safe = jnp.where(l_max == 0.0, 1.0, l_max)
        mask = jnp.exp(-alpha * (l / l_max_safe) ** power)  # (Nl,)
        u_hat_f = u_hat * mask[:, None]  # broadcast over m
        return u_hat_f if spectral else self.grid.transform(u_hat_f, inverse=True)

    def hyperviscosity(
        self,
        u: Num[Array, "Nlat Nlon"],
        nu_hyper: float,
        dt: float,
        power: int = 4,
        spectral: bool = False,
    ) -> Num[Array, "Nlat Nlon"]:
        """Hyperviscous damping using the Laplace–Beltrami eigenvalue:

            F(l) = exp(−ν_h · [l(l+1)/R²]^(p/2) · Δt)

        where R = ``grid.Ly / π``.  Applied identically to every m at a
        given l (isotropic on the sphere).

        Parameters
        ----------
        u : Num[Array, "Nlat Nlon"]
            Physical field or SHT coefficients.
        nu_hyper : float
            Hyperviscosity coefficient (≥ 0).
        dt : float
            Time step (≥ 0).
        power : int
            Laplacian power p (> 0).  Default 4 (biharmonic damping).
        spectral : bool
            If ``True``, treat ``u`` as SHT coefficients.
        """
        if nu_hyper < 0:
            raise ValueError(f"nu_hyper must be >= 0, got {nu_hyper}")
        if power <= 0:
            raise ValueError(f"power must be > 0, got {power}")
        u_hat = u if spectral else self.grid.transform(u)
        R = self.grid.Ly / jnp.pi
        l = self.grid.l  # (Nl,)
        eigenval = l * (l + 1) / (R**2)
        mask = jnp.exp(-nu_hyper * eigenval ** (power / 2.0) * dt)  # (Nl,)
        u_hat_f = u_hat * mask[:, None]
        return u_hat_f if spectral else self.grid.transform(u_hat_f, inverse=True)
