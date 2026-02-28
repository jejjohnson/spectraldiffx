"""
Spherical Spectral Solvers
============================

Spectral elliptic solvers for PDEs on the sphere using eigenvalue inversion
in spherical harmonic space.

Poisson equation on the sphere:
    nabla^2 phi = f   =>   phi_hat(l, m) = -f_hat(l, m) / [l*(l+1)/R^2]

Helmholtz equation on the sphere:
    (nabla^2 - alpha) phi = f   =>   phi_hat = -f_hat / [l*(l+1)/R^2 + alpha]

References:
-----------
[1] Boyd, J. P. (2001). Chebyshev and Fourier Spectral Methods.
[4] Durran, D. R. (2010). Numerical Methods for Fluid Dynamics.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .grid import SphericalGrid1D


class SphericalPoissonSolver(eqx.Module):
    """
    Spectral Poisson solver on the sphere.

    Solves: nabla^2 phi = f

    In spectral space (SHT coefficients):
        phi_hat(l, m) = -f_hat(l, m) / [l*(l+1)/R^2]

    The l=0 mode is set to zero (mean of phi is undetermined for Poisson).

    Attributes:
    -----------
    grid : SphericalGrid2D or SphericalGrid1D
        The spherical grid.
    """

    grid: object  # SphericalGrid2D or SphericalGrid1D

    def solve(
        self,
        f: Float[Array, "..."],
        zero_mean: bool = True,
        spectral: bool = False,
    ) -> Float[Array, "..."]:
        """
        Solve nabla^2 phi = f on the sphere.

        Parameters:
        -----------
        f : Array
            Source field (physical space or spectral if spectral=True).
            Shape (Ny,) for 1D or (Ny, Nx) for 2D.
        zero_mean : bool
            If True, set the l=0 (global mean) mode to zero.
        spectral : bool
            If True, treat f as SHT/DLT coefficients.

        Returns:
        --------
        phi : Array
            Solution field in physical space.
        """
        R = (
            self.grid.L / jnp.pi
            if isinstance(self.grid, SphericalGrid1D)
            else self.grid.Ly / jnp.pi
        )
        f_hat = f if spectral else self.grid.transform(f)
        l = self.grid.l  # (N,) or (Ny,)
        eigenval = l * (l + 1) / (R**2)

        if isinstance(self.grid, SphericalGrid1D):
            # 1D case
            denom = jnp.where(eigenval == 0.0, 1.0, eigenval)
            phi_hat = -f_hat / denom
            if zero_mean:
                phi_hat = jnp.where(l == 0.0, 0.0, phi_hat)
        else:
            # 2D case: eigenval is (Ny,), broadcast to (Ny, Nx)
            denom = jnp.where(eigenval[:, None] == 0.0, 1.0, eigenval[:, None])
            phi_hat = -f_hat / denom
            if zero_mean:
                phi_hat = jnp.where(l[:, None] == 0.0, 0.0, phi_hat)

        return self.grid.transform(phi_hat, inverse=True)


class SphericalHelmholtzSolver(eqx.Module):
    """
    Spectral Helmholtz solver on the sphere.

    Solves: (nabla^2 - alpha) phi = f

    In spectral space:
        phi_hat(l, m) = -f_hat(l, m) / [l*(l+1)/R^2 + alpha]

    Attributes:
    -----------
    grid : SphericalGrid2D or SphericalGrid1D
        The spherical grid.
    """

    grid: object  # SphericalGrid2D or SphericalGrid1D

    def solve(
        self,
        f: Float[Array, "..."],
        alpha: float = 0.0,
        zero_mean: bool = True,
        spectral: bool = False,
    ) -> Float[Array, "..."]:
        """
        Solve (nabla^2 - alpha) phi = f on the sphere.

        Parameters:
        -----------
        f : Array
            Source field (physical or spectral).
        alpha : float
            Helmholtz parameter (>= 0).  alpha=0 reduces to Poisson.
        zero_mean : bool
            If True, force l=0 (mean) mode to zero.  Relevant when alpha=0.
        spectral : bool
            If True, treat f as SHT/DLT coefficients.

        Returns:
        --------
        phi : Array
            Solution field in physical space.
        """
        R = (
            self.grid.L / jnp.pi
            if isinstance(self.grid, SphericalGrid1D)
            else self.grid.Ly / jnp.pi
        )
        f_hat = f if spectral else self.grid.transform(f)
        l = self.grid.l
        eigenval = l * (l + 1) / (R**2)

        if isinstance(self.grid, SphericalGrid1D):
            denom = eigenval + alpha
            denom_safe = jnp.where(denom == 0.0, 1.0, denom)
            phi_hat = -f_hat / denom_safe
            if zero_mean:
                phi_hat = jnp.where(l == 0.0, 0.0, phi_hat)
        else:
            denom = eigenval[:, None] + alpha
            denom_safe = jnp.where(denom == 0.0, 1.0, denom)
            phi_hat = -f_hat / denom_safe
            if zero_mean:
                phi_hat = jnp.where(l[:, None] == 0.0, 0.0, phi_hat)

        return self.grid.transform(phi_hat, inverse=True)
