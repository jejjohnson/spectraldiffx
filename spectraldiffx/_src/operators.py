# ============================================================================
# Spectral Derivative Operators
# ============================================================================

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Complex, PyTree
from typing import Literal, Optional, Callable, Tuple, Union, Dict
from functools import partial
from .grid import FourierGrid1D, FourierGrid2D, FourierGrid3D


class SpectralDerivative1D(eqx.Module):
    """
    1D Spectral derivative operator using the Fast Fourier Transform (FFT).
    
    This class provides methods to differentiate fields periodically using
    spectral accuracy. It leverages the property that the Fourier Transform (FT)
    turns differentiation into multiplication by the wavenumber vector.
    
    Mathematical Formulation:
    -------------------------
    Given a periodic field u(x) on a domain of length L with N points:
        u(x) = Σ u_hat(k) * exp(i * k * x)
    
    The n-th derivative is:
        d^n u / dx^n = Σ (i * k)^n * u_hat(k) * exp(i * k * x)
    
    where k are the discrete wavenumbers: k = 2 * pi * n / L.
    
    Attributes:
    -----------
        grid : FourierGrid1D
            The 1D grid object containing wavenumbers k [N].
    """
    grid: FourierGrid1D
    
    def __call__(
        self,
        u: Array,
        order: int = 1,
        spectral: bool = False
    ) -> Float[Array, "N"]:
        """
        Compute the n-th derivative of a field.

        Parameters:
        -----------
        u : Array [N]
            The input field. If spectral=False, this is physical space.
            If spectral=True, this is complex Fourier coefficients.
        order : int, optional
            The order of the derivative (1=first, 2=second, etc.). Default is 1.
        spectral : bool, optional
            Whether the input 'u' is already in Fourier space. Default is False.
        
        Returns:
        --------
        du_dx : Array [N]
            The n-th derivative in physical space.
        """
        # 1. Obtain spectral coefficients u_hat [N]
        u_hat = u if spectral else self.grid.transform(u)
        
        # 2. Multiply by (i*k)^order [N]
        # We use k_dealias to zero out high-frequency modes prone to aliasing
        k = self.grid.k_dealias
        du_hat = (1j * k)**order * u_hat
        
        # 3. Inverse Transform back to physical space [N]
        return self.grid.transform(du_hat, inverse=True).real
    
    def gradient(self, u: Array, spectral: bool = False) -> Array:
        """
        Compute the first derivative (gradient) du/dx.
        
        Parameters:
        -----------
        u : Array [N]
            Input field.
        spectral : bool, optional
            If True, treats u as Fourier coefficients.
        """
        return self(u, order=1, spectral=spectral)
    
    def laplacian(self, u: Array, spectral: bool = False) -> Array:
        """
        Compute the second derivative (Laplacian) d^2u/dx^2.
        
        Operation in Fourier Space:
            lap_hat = -(k^2) * u_hat
        """
        u_hat = u if spectral else self.grid.transform(u)
        k = self.grid.k
        dealias = self.grid.dealias_filter()
        lap_hat = -k**2 * u_hat * dealias
        return self.grid.transform(lap_hat, inverse=True).real

    def apply_dealias(self, u: Array, spectral: bool = False) -> Array:
        """
        Apply the 2/3 dealiasing rule mask to a field.
        
        This zeros out the top 1/3 of the spectrum to prevent aliasing
        errors in nonlinear products (e.g., u * du/dx).
        """
        u_hat = u if spectral else self.grid.transform(u)
        mask = self.grid.dealias_filter()
        u_hat_f = u_hat * mask
        return u_hat_f if spectral else self.grid.transform(u_hat_f, inverse=True).real


class SpectralDerivative2D(eqx.Module):
    """
    2D Spectral derivative operators for doubly periodic rectangular domains.
    
    Mathematical Formulation:
    -------------------------
    For a 2D field u(x, y):
        u(x, y) = ΣΣ u_hat(kx, ky) * exp(i * (kx*x + ky*y))
    
    Partial derivatives:
        ∂u/∂x ↔ i*kx * u_hat
        ∂u/∂y ↔ i*ky * u_hat
    
    Gradient vector: ∇u = (∂u/∂x, ∂u/∂y)
    Laplacian: ∇^2 u = ∂^2u/∂x^2 + ∂^2u/∂y^2 ↔ -(kx^2 + ky^2) * u_hat
    """
    grid: FourierGrid2D
    
    def gradient(
        self,
        u: Array,
        spectral: bool = False
    ) -> Tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Compute the gradient vector [du/dx, du/dy]."""
        u_hat = u if spectral else self.grid.transform(u)
        KX, KY = self.grid.KX
        dealias = self.grid.dealias_filter()
        
        du_dx = self.grid.transform(1j * KX * u_hat * dealias, inverse=True).real
        du_dy = self.grid.transform(1j * KY * u_hat * dealias, inverse=True).real
        return du_dx, du_dy
    
    def divergence(self, vx: Array, vy: Array, spectral: bool = False) -> Float[Array, "Ny Nx"]:
        """
        Compute the divergence of a 2D vector field: div(V) = ∂vx/∂x + ∂vy/∂y.
        """
        vx_hat = vx if spectral else self.grid.transform(vx)
        vy_hat = vy if spectral else self.grid.transform(vy)
        KX, KY = self.grid.KX
        dealias = self.grid.dealias_filter()
        
        div_hat = (1j * KX * vx_hat + 1j * KY * vy_hat) * dealias
        return self.grid.transform(div_hat, inverse=True).real

    def curl(self, vx: Array, vy: Array, spectral: bool = False) -> Float[Array, "Ny Nx"]:
        """
        Compute the 2D scalar curl (vorticity): ζ = ∂vy/∂x - ∂vx/∂y.
        """
        vx_hat = vx if spectral else self.grid.transform(vx)
        vy_hat = vy if spectral else self.grid.transform(vy)
        KX, KY = self.grid.KX
        dealias = self.grid.dealias_filter()
        
        curl_hat = (1j * KX * vy_hat - 1j * KY * vx_hat) * dealias
        return self.grid.transform(curl_hat, inverse=True).real

    def laplacian(self, u: Array, spectral: bool = False) -> Float[Array, "Ny Nx"]:
        """
        Compute the 2D Laplacian: ∇^2 u = ∂^2u/∂x^2 + ∂^2u/∂y^2.
        """
        u_hat = u if spectral else self.grid.transform(u)
        K2 = self.grid.K2
        dealias = self.grid.dealias_filter()
        lap_hat = -K2 * u_hat * dealias
        return self.grid.transform(lap_hat, inverse=True).real

    def apply_dealias(self, u: Array, spectral: bool = False) -> Array:
        """Apply the 2D spectral dealiasing filter mask [Ny, Nx]."""
        u_hat = u if spectral else self.grid.transform(u)
        mask = self.grid.dealias_filter()
        u_hat_f = u_hat * mask
        return u_hat_f if spectral else self.grid.transform(u_hat_f, inverse=True).real

    def project_vector(self, vx: Array, vy: Array) -> Tuple[Array, Array]:
        """
        Perform Leray projection to extract the divergence-free component.
        
        Solves the decomposition: V = V_solenoidal + V_irrotational
        where div(V_solenoidal) = 0 and curl(V_irrotational) = 0.
        
        Physics:
        --------
        In incompressible flows, we project the velocity onto the divergence-free
        manifold by solving for a scalar potential φ:
            div(V - grad(φ)) = 0  =>  ∇^2 φ = div(V)
        
        Then: V_solenoidal = V - grad(φ)
        """
        # 1. Transform components to Fourier space [Ny, Nx]
        vx_hat = self.grid.transform(vx)
        vy_hat = self.grid.transform(vy)
        KX, KY = self.grid.KX
        
        # 2. Compute divergence in Fourier space
        div_hat = 1j * (KX * vx_hat + KY * vy_hat)
        
        # 3. Solve Poisson: -|k|^2 * φ_hat = div_hat
        K2 = self.grid.K2
        K2_safe = jnp.where(K2 == 0, 1.0, K2)
        phi_hat = -div_hat / K2_safe
        phi_hat = jnp.where(K2 == 0, 0.0, phi_hat) # Force mean to zero
        
        # 4. Correct velocity: V_sol_hat = V_hat - i*k*φ_hat
        vx_sol_hat = vx_hat - 1j * KX * phi_hat
        vy_sol_hat = vy_hat - 1j * KY * phi_hat
        
        return (self.grid.transform(vx_sol_hat, inverse=True).real,
                self.grid.transform(vy_sol_hat, inverse=True).real)

    def advection_scalar(self, vx: Array, vy: Array, q: Array) -> Float[Array, "Ny Nx"]:
        """
        Compute scalar advection (u·∇)q using the pseudo-spectral method.
        
        This method computes derivatives in Fourier space for accuracy, then
        transforms back to physical space to perform the multiplication.
        
        Out = vx * ∂q/∂x + vy * ∂q/∂y
        """
        q_hat = self.grid.transform(q)
        KX, KY = self.grid.KX
        dealias = self.grid.dealias_filter()
        
        dq_dx = self.grid.transform(1j * KX * q_hat * dealias, inverse=True).real
        dq_dy = self.grid.transform(1j * KY * q_hat * dealias, inverse=True).real
        
        return vx * dq_dx + vy * dq_dy


class SpectralDerivative3D(eqx.Module):
    """
    3D Spectral derivative operators for triply periodic domains.
    
    Mathematical Framework:
    -----------------------
    Field shapes: (Nz, Ny, Nx) following 'ij' indexing.
    Wavenumbers: (kz, ky, kx).
    
    Gradient vector: ∇u = (∂u/∂z, ∂u/∂y, ∂u/∂x)
    """
    grid: FourierGrid3D
    
    def gradient(self, u: Array, spectral: bool = False) -> Tuple[Array, Array, Array]:
        """Compute the 3D gradient vector field."""
        u_hat = u if spectral else self.grid.transform(u)
        KZ, KY, KX = self.grid.KX
        dealias = self.grid.dealias_filter()
        
        du_dz = self.grid.transform(1j * KZ * u_hat * dealias, inverse=True).real
        du_dy = self.grid.transform(1j * KY * u_hat * dealias, inverse=True).real
        du_dx = self.grid.transform(1j * KX * u_hat * dealias, inverse=True).real
        return du_dz, du_dy, du_dx

    def divergence(self, vz: Array, vy: Array, vx: Array, spectral: bool = False) -> Float[Array, "Nz Ny Nx"]:
        """Compute 3D divergence: ∂vz/∂z + ∂vy/∂y + ∂vx/∂x."""
        vz_hat = vz if spectral else self.grid.transform(vz)
        vy_hat = vy if spectral else self.grid.transform(vy)
        vx_hat = vx if spectral else self.grid.transform(vx)
        KZ, KY, KX = self.grid.KX
        dealias = self.grid.dealias_filter()
        
        div_hat = (1j * KZ * vz_hat + 1j * KY * vy_hat + 1j * KX * vx_hat) * dealias
        return self.grid.transform(div_hat, inverse=True).real

    def curl(self, vz: Array, vy: Array, vx: Array, spectral: bool = False) -> Tuple[Array, Array, Array]:
        """
        Compute the 3D curl vector: ω = ∇ x V.
        
        Components:
            ωz = ∂vy/∂x - ∂vx/∂y
            ωy = ∂vx/∂z - ∂vz/∂x
            ωx = ∂vz/∂y - ∂vy/∂z
        """
        vz_hat = vz if spectral else self.grid.transform(vz)
        vy_hat = vy if spectral else self.grid.transform(vy)
        vx_hat = vx if spectral else self.grid.transform(vx)
        KZ, KY, KX = self.grid.KX
        dealias = self.grid.dealias_filter()
        
        wz_hat = (1j * KX * vy_hat - 1j * KY * vx_hat) * dealias
        wy_hat = (1j * KZ * vx_hat - 1j * KX * vz_hat) * dealias
        wx_hat = (1j * KY * vz_hat - 1j * KZ * vy_hat) * dealias
        
        return (self.grid.transform(wz_hat, inverse=True).real,
                self.grid.transform(wy_hat, inverse=True).real,
                self.grid.transform(wx_hat, inverse=True).real)

    def laplacian(self, u: Array, spectral: bool = False) -> Float[Array, "Nz Ny Nx"]:
        """Compute 3D Laplacian: ∇^2 u = ∂^2u/∂z^2 + ∂^2u/∂y^2 + ∂^2u/∂x^2."""
        u_hat = u if spectral else self.grid.transform(u)
        K2 = self.grid.K2
        dealias = self.grid.dealias_filter()
        lap_hat = -K2 * u_hat * dealias
        return self.grid.transform(lap_hat, inverse=True).real

    def apply_dealias(self, u: Array, spectral: bool = False) -> Array:
        """Apply the 3D periodic dealiasing filter mask [Nz, Ny, Nx]."""
        u_hat = u if spectral else self.grid.transform(u)
        mask = self.grid.dealias_filter()
        u_hat_f = u_hat * mask
        return u_hat_f if spectral else self.grid.transform(u_hat_f, inverse=True).real

    def project_vector(self, vz: Array, vy: Array, vx: Array) -> Tuple[Array, Array, Array]:
        """Project a 3D vector field onto its solenoidal (divergence-free) component."""
        vz_hat = self.grid.transform(vz)
        vy_hat = self.grid.transform(vy)
        vx_hat = self.grid.transform(vx)
        KZ, KY, KX = self.grid.KX
        
        div_hat = 1j * (KZ * vz_hat + KY * vy_hat + KX * vx_hat)
        
        K2 = self.grid.K2
        K2_safe = jnp.where(K2 == 0, 1.0, K2)
        phi_hat = -div_hat / K2_safe
        phi_hat = jnp.where(K2 == 0, 0.0, phi_hat)
        
        vz_sol_hat = vz_hat - 1j * KZ * phi_hat
        vy_sol_hat = vy_hat - 1j * KY * phi_hat
        vx_sol_hat = vx_hat - 1j * KX * phi_hat
        
        return (self.grid.transform(vz_sol_hat, inverse=True).real,
                self.grid.transform(vy_sol_hat, inverse=True).real,
                self.grid.transform(vx_sol_hat, inverse=True).real)

    def advection_scalar(self, vz: Array, vy: Array, vx: Array, q: Array) -> Float[Array, "Nz Ny Nx"]:
        """Compute the 3D scalar advection term: (u·∇)q."""
        q_hat = self.grid.transform(q)
        KZ, KY, KX = self.grid.KX
        dealias = self.grid.dealias_filter()
        
        dq_dz = self.grid.transform(1j * KZ * q_hat * dealias, inverse=True).real
        dq_dy = self.grid.transform(1j * KY * q_hat * dealias, inverse=True).real
        dq_dx = self.grid.transform(1j * KX * q_hat * dealias, inverse=True).real
        
        return vz * dq_dz + vy * dq_dy + vx * dq_dx