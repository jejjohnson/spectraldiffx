# ============================================================================
# Spectral Elliptic Solvers (Helmholtz & Poisson)
# ============================================================================


import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .grid import FourierGrid1D, FourierGrid2D, FourierGrid3D


class SpectralHelmholtzSolver1D(eqx.Module):
    """
    1D Spectral Helmholtz/Poisson solver using the Fast Fourier Transform.

    This module solves the 1D elliptic equation:
        (d^2/dx^2 - alpha) * phi = f

    where phi is the unknown potential and f is a given source term.
    Both phi and f are assumed to be periodic on the domain [0, L].

    Spectral Formulation:
    ---------------------
    Applying the Fourier Transform (FT) to the equation:
        FT[d^2/dx^2 * phi - alpha * phi] = FT[f]
        -(k^2 + alpha) * phi_hat = f_hat

    The solution in Fourier space is:
        phi_hat = -f_hat / (k^2 + alpha)

    For alpha = 0, the equation reduces to the Poisson equation:
        d^2/dx^2 * phi = f
        phi_hat = -f_hat / k^2

    Attributes:
    -----------
        grid : FourierGrid1D
            The 1D Fourier grid object containing wavenumbers k.
    """

    grid: FourierGrid1D

    def solve(
        self,
        f: Array,
        alpha: float = 0.0,
        zero_mean: bool = True,
        spectral: bool = False,
    ) -> Array:
        """
        Solves (d^2/dx^2 - alpha) * phi = f in 1D.

        Parameters:
        -----------
        f : Array [N]
            The source term (Right-Hand Side) in physical space.
        alpha : float, optional
            The Helmholtz parameter (alpha >= 0). Default is 0.0 (Poisson).
        zero_mean : bool, optional
            Whether to force the mean (k=0 mode) of the solution to zero.
            Necessary for Poisson (alpha=0) as the operator is singular.
        spectral : bool, optional
            If True, treats input f as already in spectral space.

        Returns:
        --------
        phi : Array [N]
            The solution field in physical space.
        """
        f_hat = f if spectral else self.grid.transform(f)
        k2 = self.grid.k**2
        denom = k2 + alpha

        denom_safe = jnp.where(denom == 0.0, 1.0, denom)
        phi_hat = -f_hat / denom_safe

        if zero_mean:
            phi_hat = jnp.where(k2 == 0.0, 0.0, phi_hat)

        return self.grid.transform(phi_hat, inverse=True).real


class SpectralHelmholtzSolver2D(eqx.Module):
    """
    2D Spectral Helmholtz/Poisson solver for doubly periodic domains.

    Solves the 2D elliptic equation:
        (div(grad(phi)) - alpha) * phi = f
        (Laplacian - alpha) * phi = f

    Spectral Formulation:
    ---------------------
    In Fourier space (kx, ky):
        -(kx^2 + ky^2 + alpha) * phi_hat = f_hat
        phi_hat = -f_hat / (|k|^2 + alpha)

    Attributes:
    -----------
        grid : FourierGrid2D
            The 2D Fourier grid object containing meshgrid wavenumbers KX, KY.
    """

    grid: FourierGrid2D

    def solve(
        self,
        f: Array,
        alpha: float = 0.0,
        zero_mean: bool = True,
        spectral: bool = False,
    ) -> Array:
        """
        Solves (grad^2 - alpha) * phi = f in 2D.

        Parameters:
        -----------
        f : Array [Ny, Nx]
            The 2D source term in physical space.
        alpha : float, optional
            The Helmholtz parameter (alpha >= 0).
        zero_mean : bool, optional
            If True, sets phi_hat(0,0) = 0.
        spectral : bool, optional
            If True, treats input f as already in spectral space.

        Returns:
        --------
        phi : Array [Ny, Nx]
            The 2D solution field in physical space.
        """
        f_hat = f if spectral else self.grid.transform(f)
        K2 = self.grid.K2
        denom = K2 + alpha

        denom_safe = jnp.where(denom == 0.0, 1.0, denom)
        phi_hat = -f_hat / denom_safe

        if zero_mean:
            phi_hat = jnp.where(K2 == 0.0, 0.0, phi_hat)

        return self.grid.transform(phi_hat, inverse=True).real


class SpectralHelmholtzSolver3D(eqx.Module):
    """
    3D Spectral Helmholtz/Poisson solver for triply periodic domains.

    Solves the 3D elliptic equation:
        (Laplacian - alpha) * phi = (div(grad(phi)) - alpha) * phi = f

    Spectral Formulation:
    ---------------------
    In Fourier space (kz, ky, kx):
        -(kz^2 + ky^2 + kx^2 + alpha) * phi_hat = f_hat
        phi_hat = -f_hat / (|k|^3D_mag^2 + alpha)

    Attributes:
    -----------
        grid : FourierGrid3D
            Object containing 3D wavenumbers on (Nz, Ny, Nx) grid.
    """

    grid: FourierGrid3D

    def solve(
        self,
        f: Array,
        alpha: float = 0.0,
        zero_mean: bool = True,
        spectral: bool = False,
    ) -> Array:
        """
        Solves (grad^2 - alpha) * phi = f in 3D.

        Parameters:
        -----------
        f : Array [Nz, Ny, Nx]
            The 3D source term field.
        alpha : float, optional
            Non-negative Helmholtz constant.
        zero_mean : bool, optional
            Removes the global mean of the solution.
        spectral : bool, optional
            If True, treats input f as already in spectral space.

        Returns:
        --------
        phi : Array [Nz, Ny, Nx]
            The 3D potential field.
        """
        f_hat = f if spectral else self.grid.transform(f)
        K2 = self.grid.K2
        denom = K2 + alpha

        denom_safe = jnp.where(denom == 0.0, 1.0, denom)
        phi_hat = -f_hat / denom_safe

        if zero_mean:
            phi_hat = jnp.where(K2 == 0.0, 0.0, phi_hat)

        return self.grid.transform(phi_hat, inverse=True).real
