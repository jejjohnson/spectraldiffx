"""
Pseudo-Spectral Fourier Discretization Module
==============================================

Spectral methods using Fourier basis for solving PDEs on periodic domains.
Combines spectral accuracy in space with time integration for pseudo-spectral methods.

Key Concepts:
-------------
    • Spectral representation: u(x) = Σ û_k·exp(ikx)
    • Derivatives in Fourier space: ∂ⁿu/∂xⁿ ↔ (ik)ⁿ·û_k
    • Dealiasing: 2/3 rule or padding to prevent aliasing
    • Pseudo-spectral: nonlinear terms computed in physical space

Advantages:
-----------
    • Exponential (spectral) accuracy for smooth functions
    • Exact differentiation (up to machine precision)
    • Fast via FFT: O(N log N)
    • Natural for periodic domains

Limitations:
------------
    • Requires periodic boundary conditions
    • Gibbs phenomenon near discontinuities
    • Global basis (non-local)
    • Best for smooth solutions

References:
-----------
[1] Boyd, J. P. (2001). Chebyshev and Fourier Spectral Methods.
[2] Trefethen, L. N. (2000). Spectral Methods in MATLAB.
[3] Canuto et al. (2006). Spectral Methods: Fundamentals.
[4] Durran, D. R. (2010). Numerical Methods for Fluid Dynamics.
"""

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

# ============================================================================
# Fourier Grid and Wavenumbers
# ============================================================================


class FourierGrid1D(eqx.Module):
    """
    Fourier grid setup for periodic domain.

    Mathematical Framework:
    -----------------------
    For periodic domain [0, L]:

        Grid points: x_j = j·Δx = j·L/N, j = 0, 1, ..., N-1

        Wavenumbers: k_n = 2πn/L, n = -N/2+1, ..., N/2

    Discrete Fourier Transform:
        û_k = (1/N)·Σ_{j=0}^{N-1} u_j·exp(-ik·x_j)

        u_j = Σ_{k=-N/2+1}^{N/2} û_k·exp(ik·x_j)

    FFT Ordering:
        k = [0, 1, 2, ..., N/2-1, -N/2, -N/2+1, ..., -1]

    Attributes:
    -----------
        N : int
            Number of grid points
        L : float
            Domain length [m]
        dx : float
            Grid spacing [m]. Must satisfy L = N * dx.
        dealias : str
            Dealiasing method ('2/3', 'padding', None)
    """

    N: int
    L: float
    dx: float
    dealias: Literal["2/3", "padding", None] | None = "2/3"

    def check_consistency(self, rtol: float = 1e-5) -> bool:
        """
        Verifies that the grid attributes (N, L, dx) are mathematically consistent.

        Checks the relationship: L ≈ N * dx

        Parameters:
        -----------
        rtol : float
            Relative tolerance for the floating point comparison.

        Returns:
        --------
        bool
            True if consistent, raises ValueError otherwise.
        """
        expected_L = self.N * self.dx
        if not jnp.isclose(self.L, expected_L, rtol=rtol):
            raise ValueError(
                f"Grid inconsistency detected.\n"
                f"  L (defined): {self.L}\n"
                f"  N * dx     : {expected_L}\n"
                f"  Difference : {abs(self.L - expected_L)}"
            )
        return True

    @classmethod
    def from_N_L(
        cls,
        N: int,
        L: float,
        dealias: Literal["2/3", "padding", None] | None = "2/3",
    ) -> "FourierGrid1D":
        """
        Initialize FourierGrid using Number of points (N) and Length (L).

        Calculates: dx = L / N
        """
        dx = L / N
        return cls(N=N, L=L, dx=dx, dealias=dealias)

    @classmethod
    def from_N_dx(
        cls,
        N: int,
        dx: float,
        dealias: Literal["2/3", "padding", None] | None = "2/3",
    ) -> "FourierGrid1D":
        """
        Initialize FourierGrid using Number of points (N) and Spacing (dx).

        Calculates: L = N * dx
        """
        L = N * dx
        return cls(N=N, L=L, dx=dx, dealias=dealias)

    @classmethod
    def from_L_dx(
        cls,
        L: float,
        dx: float,
        dealias: Literal["2/3", "padding", None] | None = "2/3",
    ) -> "FourierGrid1D":
        """
        Initialize FourierGrid using Length (L) and Spacing (dx).

        Calculates: N = L / dx (Must result in an integer)

        Raises:
        -------
        ValueError: If L is not divisible by dx (N is not an integer).
        """
        N_float = L / dx
        if not jnp.isclose(N_float % 1, 0) and not jnp.isclose(N_float % 1, 1):
            raise ValueError(
                f"L={L} is not divisible by dx={dx}. Resulting N={N_float} is not an integer."
            )

        N = int(round(N_float))
        return cls(N=N, L=L, dx=dx, dealias=dealias)

    @property
    def x(self) -> Float[Array, "N"]:
        """Physical grid points: x_j = j·Δx"""
        # We use linspace for numerical stability, but it should match self.dx * j
        return jnp.linspace(0, self.L, self.N, endpoint=False)

    @property
    def k(self) -> Float[Array, "N"]:
        """
        Wavenumbers in FFT order.

        k = 2π·[0, 1, 2, ..., N/2-1, -N/2, -N/2+1, ..., -1]/L

        Returns:
        --------
        k : Array [N]
            Wavenumbers [rad/m]
        """
        return 2 * jnp.pi * jnp.fft.fftfreq(self.N, self.dx)

    @property
    def k_dealias(self) -> Float[Array, "N"]:
        """
        Dealiased wavenumbers (set high frequencies to zero).

        2/3 rule: Keep |k| ≤ k_max/3
        This prevents aliasing in quadratic nonlinearities.

        Mathematical Justification:
        ---------------------------
        For product w = u·v:
            ŵ_k = Σ_p û_p·v̂_{k-p}

        Maximum frequency in w: k_max(w) = k_max(u) + k_max(v)

        To avoid aliasing: k_max(w) ≤ N/2
        Therefore: 2·k_max(u) ≤ N/2  →  k_max(u) ≤ N/3

        Returns:
        --------
        k : Array [N]
            Dealiased wavenumbers
        """
        k = self.k
        if self.dealias == "2/3":
            k_max = jnp.abs(k).max()
            cutoff = k_max * 2 / 3
            k = jnp.where(jnp.abs(k) > cutoff, 0.0, k)
        return k

    def dealias_filter(self) -> Float[Array, "N"]:
        """
        Dealiasing filter: 1 for kept modes, 0 for removed.

        Returns:
        --------
        filter : Array [N]
            Filter mask (1 or 0)
        """
        if self.dealias == "2/3":
            k = self.k
            k_max = jnp.abs(k).max()
            cutoff = k_max * 2 / 3
            return jnp.where(jnp.abs(k) <= cutoff, 1.0, 0.0)
        else:
            return jnp.ones(self.N)

    def transform(self, u: Array, inverse: bool = False) -> Complex[Array, "N"]:
        """
        Perform 1D Fourier Transform.
        Forward: physical -> spectral [u -> u_hat]
        Inverse: spectral -> physical [u_hat -> u]
        """
        if inverse:
            return jnp.fft.ifft(u)
        return jnp.fft.fft(u)


class FourierGrid2D(eqx.Module):
    """
    2D Fourier grid for doubly periodic domain.

    Mathematical Framework:
    -----------------------
    For domain [0, Lx] × [0, Ly]:

        u(x,y) = ΣΣ û_{kx,ky}·exp(i(kx·x + ky·y))

    2D FFT:
        û_{kx,ky} = (1/(Nx·Ny))·ΣΣ u_{j,l}·exp(-i(kx·x_j + ky·y_l))

    Attributes:
    -----------
        Nx, Ny : int
            Grid points in x, y
        Lx, Ly : float
            Domain lengths [m]
        dx, dy : float
            Grid spacings in x, y [m]. Must satisfy L = N * d.
        dealias : str
            Dealiasing method
    """

    Nx: int
    Ny: int
    Lx: float
    Ly: float
    dx: float
    dy: float
    dealias: Literal["2/3", "padding", None] | None = "2/3"

    def check_consistency(self, rtol: float = 1e-5) -> bool:
        """
        Verifies that the grid attributes are mathematically consistent for both dimensions.

        Checks:
            Lx ≈ Nx * dx
            Ly ≈ Ny * dy

        Parameters:
        -----------
        rtol : float
            Relative tolerance for the floating point comparison.

        Returns:
        --------
        bool
            True if consistent, raises ValueError otherwise.
        """
        expected_Lx = self.Nx * self.dx
        expected_Ly = self.Ny * self.dy

        errors = []
        if not jnp.isclose(self.Lx, expected_Lx, rtol=rtol):
            errors.append(
                f"X-dimension inconsistency: Lx={self.Lx} vs Nx*dx={expected_Lx}"
            )
        if not jnp.isclose(self.Ly, expected_Ly, rtol=rtol):
            errors.append(
                f"Y-dimension inconsistency: Ly={self.Ly} vs Ny*dy={expected_Ly}"
            )

        if errors:
            raise ValueError("Grid inconsistency detected:\n" + "\n".join(errors))

        return True

    @classmethod
    def from_N_L(
        cls,
        Nx: int,
        Ny: int,
        Lx: float,
        Ly: float,
        dealias: Literal["2/3", "padding", None] | None = "2/3",
    ) -> "FourierGrid2D":
        """
        Initialize FourierGrid2D using Number of points (N) and Length (L).

        Calculates:
            dx = Lx / Nx
            dy = Ly / Ny
        """
        dx = Lx / Nx
        dy = Ly / Ny
        return cls(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dx=dx, dy=dy, dealias=dealias)

    @classmethod
    def from_N_dx(
        cls,
        Nx: int,
        Ny: int,
        dx: float,
        dy: float,
        dealias: Literal["2/3", "padding", None] | None = "2/3",
    ) -> "FourierGrid2D":
        """
        Initialize FourierGrid2D using Number of points (N) and Spacing (dx/dy).

        Calculates:
            Lx = Nx * dx
            Ly = Ny * dy
        """
        Lx = Nx * dx
        Ly = Ny * dy
        return cls(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dx=dx, dy=dy, dealias=dealias)

    @classmethod
    def from_L_dx(
        cls,
        Lx: float,
        Ly: float,
        dx: float,
        dy: float,
        dealias: Literal["2/3", "padding", None] | None = "2/3",
    ) -> "FourierGrid2D":
        """
        Initialize FourierGrid2D using Length (L) and Spacing (dx/dy).

        Calculates:
            Nx = Lx / dx
            Ny = Ly / dy

        Raises:
        -------
        ValueError: If L is not divisible by dx (N is not an integer).
        """
        Nx_float = Lx / dx
        Ny_float = Ly / dy

        errors = []
        if not jnp.isclose(Nx_float % 1, 0) and not jnp.isclose(Nx_float % 1, 1):
            errors.append(f"Lx={Lx} not divisible by dx={dx} (Nx={Nx_float})")
        if not jnp.isclose(Ny_float % 1, 0) and not jnp.isclose(Ny_float % 1, 1):
            errors.append(f"Ly={Ly} not divisible by dy={dy} (Ny={Ny_float})")

        if errors:
            raise ValueError("\n".join(errors))

        return cls(
            Nx=int(round(Nx_float)),
            Ny=int(round(Ny_float)),
            Lx=Lx,
            Ly=Ly,
            dx=dx,
            dy=dy,
            dealias=dealias,
        )

    @property
    def x(self) -> Float[Array, "Nx"]:
        """Physical grid points x: [0, dx, ..., Lx-dx]"""
        return jnp.linspace(0, self.Lx, self.Nx, endpoint=False)

    @property
    def y(self) -> Float[Array, "Ny"]:
        """Physical grid points y: [0, dy, ..., Ly-dy]"""
        return jnp.linspace(0, self.Ly, self.Ny, endpoint=False)

    @property
    def X(self) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """2D meshgrid: X, Y"""
        result = jnp.meshgrid(self.x, self.y, indexing="xy")
        return (result[0], result[1])

    @property
    def kx(self) -> Float[Array, "Nx"]:
        """Wavenumbers in x-direction"""
        return 2 * jnp.pi * jnp.fft.fftfreq(self.Nx, self.dx)

    @property
    def ky(self) -> Float[Array, "Ny"]:
        """Wavenumbers in y-direction"""
        return 2 * jnp.pi * jnp.fft.fftfreq(self.Ny, self.dy)

    @property
    def KX(self) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """2D wavenumber meshgrid: KX, KY"""
        result = jnp.meshgrid(self.kx, self.ky, indexing="xy")
        return (result[0], result[1])

    @property
    def K2(self) -> Float[Array, "Ny Nx"]:
        """
        Squared wavenumber magnitude: |k|² = kx² + ky²

        Used for Laplacian: ∇²u ↔ -|k|²·û
        """
        KX, KY = self.KX
        return KX**2 + KY**2

    def dealias_filter(self) -> Float[Array, "Ny Nx"]:
        """2D dealiasing filter."""
        if self.dealias == "2/3":
            KX, KY = self.KX
            kx_max = jnp.abs(self.kx).max()
            ky_max = jnp.abs(self.ky).max()
            cutoff_x = kx_max * 2 / 3
            cutoff_y = ky_max * 2 / 3

            filter_x = jnp.where(jnp.abs(KX) <= cutoff_x, 1.0, 0.0)
            filter_y = jnp.where(jnp.abs(KY) <= cutoff_y, 1.0, 0.0)

            return filter_x * filter_y
        else:
            return jnp.ones((self.Ny, self.Nx))

    def transform(self, u: Array, inverse: bool = False) -> Complex[Array, "Ny Nx"]:
        """Perform 2D Fourier Transform."""
        if inverse:
            return jnp.fft.ifft2(u)
        return jnp.fft.fft2(u)


class FourierGrid3D(eqx.Module):
    """
    3D Fourier grid for triply periodic domain.

    Mathematical Framework:
    -----------------------
    For domain [0, Lz] × [0, Ly] × [0, Lx]:

        u(z,y,x) = ΣΣΣ û_{kz,ky,kx}·exp(i(kz·z + ky·y + kx·x))

    3D FFT:
        û_{kz,ky,kx} = (1/(Nz·Ny·Nx))·ΣΣΣ u_{m,l,j}·exp(-i(kz·z_m + ky·y_l + kx·x_j))

    Grid Shapes (indexing='ij'):
        Physical/Spectral arrays have shape (Nz, Ny, Nx).
        Axis 0: z-direction (Depth)
        Axis 1: y-direction (Height)
        Axis 2: x-direction (Width)

    Attributes:
    -----------
        Nz, Ny, Nx : int
            Grid points in z, y, x
        Lz, Ly, Lx : float
            Domain lengths [m]
        dz, dy, dx : float
            Grid spacings in z, y, x [m]. Must satisfy L = N * d.
        dealias : str
            Dealiasing method
    """

    Nz: int
    Ny: int
    Nx: int
    Lz: float
    Ly: float
    Lx: float
    dz: float
    dy: float
    dx: float
    dealias: Literal["2/3", "padding", None] | None = "2/3"

    def check_consistency(self, rtol: float = 1e-5) -> bool:
        """
        Verifies that the grid attributes are mathematically consistent for all three dimensions.

        Checks:
            Lz ≈ Nz * dz
            Ly ≈ Ny * dy
            Lx ≈ Nx * dx

        Parameters:
        -----------
        rtol : float
            Relative tolerance for the floating point comparison.

        Returns:
        --------
        bool
            True if consistent, raises ValueError otherwise.
        """
        expected_Lz = self.Nz * self.dz
        expected_Ly = self.Ny * self.dy
        expected_Lx = self.Nx * self.dx

        errors = []
        if not jnp.isclose(self.Lz, expected_Lz, rtol=rtol):
            errors.append(
                f"Z-dimension inconsistency: Lz={self.Lz} vs Nz*dz={expected_Lz}"
            )
        if not jnp.isclose(self.Ly, expected_Ly, rtol=rtol):
            errors.append(
                f"Y-dimension inconsistency: Ly={self.Ly} vs Ny*dy={expected_Ly}"
            )
        if not jnp.isclose(self.Lx, expected_Lx, rtol=rtol):
            errors.append(
                f"X-dimension inconsistency: Lx={self.Lx} vs Nx*dx={expected_Lx}"
            )

        if errors:
            raise ValueError("Grid inconsistency detected:\n" + "\n".join(errors))

        return True

    @classmethod
    def from_N_L(
        cls,
        Nz: int,
        Ny: int,
        Nx: int,
        Lz: float,
        Ly: float,
        Lx: float,
        dealias: Literal["2/3", "padding", None] | None = "2/3",
    ) -> "FourierGrid3D":
        """
        Initialize FourierGrid3D using Number of points (N) and Length (L).

        Calculates:
            dz = Lz / Nz
            dy = Ly / Ny
            dx = Lx / Nx
        """
        dz = Lz / Nz
        dy = Ly / Ny
        dx = Lx / Nx
        return cls(
            Nz=Nz,
            Ny=Ny,
            Nx=Nx,
            Lz=Lz,
            Ly=Ly,
            Lx=Lx,
            dz=dz,
            dy=dy,
            dx=dx,
            dealias=dealias,
        )

    @classmethod
    def from_N_dx(
        cls,
        Nz: int,
        Ny: int,
        Nx: int,
        dz: float,
        dy: float,
        dx: float,
        dealias: Literal["2/3", "padding", None] | None = "2/3",
    ) -> "FourierGrid3D":
        """
        Initialize FourierGrid3D using Number of points (N) and Spacing (dz/dy/dx).

        Calculates:
            Lz = Nz * dz
            Ly = Ny * dy
            Lx = Nx * dx
        """
        Lz = Nz * dz
        Ly = Ny * dy
        Lx = Nx * dx
        return cls(
            Nz=Nz,
            Ny=Ny,
            Nx=Nx,
            Lz=Lz,
            Ly=Ly,
            Lx=Lx,
            dz=dz,
            dy=dy,
            dx=dx,
            dealias=dealias,
        )

    @classmethod
    def from_L_dx(
        cls,
        Lz: float,
        Ly: float,
        Lx: float,
        dz: float,
        dy: float,
        dx: float,
        dealias: Literal["2/3", "padding", None] | None = "2/3",
    ) -> "FourierGrid3D":
        """
        Initialize FourierGrid3D using Length (L) and Spacing (dz/dy/dx).

        Calculates:
            Nz = Lz / dz
            Ny = Ly / dy
            Nx = Lx / dx

        Raises:
        -------
        ValueError: If any L is not divisible by its corresponding d.
        """
        Nz_float = Lz / dz
        Ny_float = Ly / dy
        Nx_float = Lx / dx

        errors = []
        if not jnp.isclose(Nz_float % 1, 0) and not jnp.isclose(Nz_float % 1, 1):
            errors.append(f"Lz={Lz} not divisible by dz={dz} (Nz={Nz_float})")
        if not jnp.isclose(Ny_float % 1, 0) and not jnp.isclose(Ny_float % 1, 1):
            errors.append(f"Ly={Ly} not divisible by dy={dy} (Ny={Ny_float})")
        if not jnp.isclose(Nx_float % 1, 0) and not jnp.isclose(Nx_float % 1, 1):
            errors.append(f"Lx={Lx} not divisible by dx={dx} (Nx={Nx_float})")

        if errors:
            raise ValueError("\n".join(errors))

        return cls(
            Nz=int(round(Nz_float)),
            Ny=int(round(Ny_float)),
            Nx=int(round(Nx_float)),
            Lz=Lz,
            Ly=Ly,
            Lx=Lx,
            dz=dz,
            dy=dy,
            dx=dx,
            dealias=dealias,
        )

    @property
    def z(self) -> Float[Array, "Nz"]:
        """Physical grid points z: [0, dz, ..., Lz-dz]"""
        return jnp.linspace(0, self.Lz, self.Nz, endpoint=False)

    @property
    def y(self) -> Float[Array, "Ny"]:
        """Physical grid points y: [0, dy, ..., Ly-dy]"""
        return jnp.linspace(0, self.Ly, self.Ny, endpoint=False)

    @property
    def x(self) -> Float[Array, "Nx"]:
        """Physical grid points x: [0, dx, ..., Lx-dx]"""
        return jnp.linspace(0, self.Lx, self.Nx, endpoint=False)

    @property
    def X(
        self,
    ) -> tuple[
        Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]
    ]:
        """
        3D meshgrid: Z, Y, X

        Shape Note:
        -----------
        Uses indexing='ij' with (z, y, x) input.
        Resulting shape: (Nz, Ny, Nx).
        """
        result = jnp.meshgrid(self.z, self.y, self.x, indexing="ij")
        return (result[0], result[1], result[2])

    @property
    def kz(self) -> Float[Array, "Nz"]:
        """Wavenumbers in z-direction"""
        return 2 * jnp.pi * jnp.fft.fftfreq(self.Nz, self.dz)

    @property
    def ky(self) -> Float[Array, "Ny"]:
        """Wavenumbers in y-direction"""
        return 2 * jnp.pi * jnp.fft.fftfreq(self.Ny, self.dy)

    @property
    def kx(self) -> Float[Array, "Nx"]:
        """Wavenumbers in x-direction"""
        return 2 * jnp.pi * jnp.fft.fftfreq(self.Nx, self.dx)

    @property
    def KX(
        self,
    ) -> tuple[
        Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]
    ]:
        """
        3D wavenumber meshgrid: KZ, KY, KX

        Shape Note:
        -----------
        Uses indexing='ij' with (kz, ky, kx) input.
        Resulting shape: (Nz, Ny, Nx).
        """
        result = jnp.meshgrid(self.kz, self.ky, self.kx, indexing="ij")
        return (result[0], result[1], result[2])

    @property
    def K2(self) -> Float[Array, "Nz Ny Nx"]:
        """
        Squared wavenumber magnitude: |k|² = kz² + ky² + kx²

        Used for Laplacian: ∇²u ↔ -|k|²·û
        """
        KZ, KY, KX = self.KX
        return KZ**2 + KY**2 + KX**2

    def dealias_filter(self) -> Float[Array, "Nz Ny Nx"]:
        """3D dealiasing filter."""
        if self.dealias == "2/3":
            KZ, KY, KX = self.KX
            kz_max = jnp.abs(self.kz).max()
            ky_max = jnp.abs(self.ky).max()
            kx_max = jnp.abs(self.kx).max()

            cutoff_z = kz_max * 2 / 3
            cutoff_y = ky_max * 2 / 3
            cutoff_x = kx_max * 2 / 3

            filter_z = jnp.where(jnp.abs(KZ) <= cutoff_z, 1.0, 0.0)
            filter_y = jnp.where(jnp.abs(KY) <= cutoff_y, 1.0, 0.0)
            filter_x = jnp.where(jnp.abs(KX) <= cutoff_x, 1.0, 0.0)

            return filter_z * filter_y * filter_x
        else:
            return jnp.ones((self.Nz, self.Ny, self.Nx))

    def transform(self, u: Array, inverse: bool = False) -> Complex[Array, "Nz Ny Nx"]:
        """Perform 3D Fourier Transform (Nz, Ny, Nx)."""
        if inverse:
            return jnp.fft.ifftn(u)
        return jnp.fft.fftn(u)
