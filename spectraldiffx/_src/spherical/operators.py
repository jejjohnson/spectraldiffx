"""
Spherical Derivative Operators
================================

Pseudo-spectral derivative operators for fields on the sphere.

Mathematical conventions:
    theta   — colatitude in (0, pi), theta=0 at North Pole.
    phi     — longitude in [0, 2*pi).
    mu      — cos(theta), used as argument for Legendre polynomials.
    R       — sphere radius (inferred as Ly / pi).

Metric factors on the sphere:
    d/d_theta = 1 factor (colatitude derivative)
    d/d_phi   = 1/(R * sin(theta)) factor for physical space operators

References:
-----------
[1] Boyd, J. P. (2001). Chebyshev and Fourier Spectral Methods.
[4] Durran, D. R. (2010). Numerical Methods for Fluid Dynamics.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np

from .grid import SphericalGrid1D, SphericalGrid2D, _alp_matrix


def _gradient_alp_matrix(
    N: int,
    nodes_np: np.ndarray,
) -> np.ndarray:
    """
    Compute the normalised Associated Legendre Polynomial matrix for m=1,
    used in the colatitude-gradient reconstruction.

    Relation used (with scipy's Condon-Shortley phase convention):
        d/d_theta P_l_norm(cos(theta)) = +sqrt(l*(l+1)) * P_l^1_norm(cos(theta))

    Note on sign: scipy's lpmv(1, l, mu) already carries the Condon-Shortley
    phase (-1)^1 = -1, so lpmv(1, l, mu) = -sqrt(1-mu^2) * dP_l/dmu.  Using
    d/d_theta = -sin(theta) * d/dmu with sin(theta) = sqrt(1-mu^2) gives:
        d P_l(cos(theta)) / d_theta = lpmv(1, l, cos(theta))
    which, after normalisation, yields a positive coefficient here.

    where P_l^1_norm[l, j] = N_{l,1} * P_l^1(cos(theta_j))
    and   N_{l,1} = sqrt((2*l+1) / (2*l*(l+1)))  for l >= 1.

    Parameters:
    -----------
    N : int
        Grid size (number of GL nodes).
    nodes_np : ndarray [N]
        Gauss-Legendre nodes (cos(theta)).

    Returns:
    --------
    P1 : ndarray [N, N]
        Normalised m=1 ALP matrix.  Row 0 (l=0) is zero.
    """
    l_values = np.arange(N)
    return _alp_matrix(1, l_values, nodes_np)


# ============================================================================
# SphericalDerivative1D
# ============================================================================


class SphericalDerivative1D(eqx.Module):
    """
    1D spectral derivative operator on Gauss-Legendre latitude grid.

    Supports colatitude (theta) gradient and Laplacian using Legendre polynomial
    spectral coefficients.

    Mathematical Formulation:
    -------------------------
    Expand u in normalised Legendre polynomials:
        u(theta) = sum_l c_l * P_l_norm(cos(theta))

    Colatitude gradient (using recurrence P_l'(cos(theta)) = -P_l^1(cos(theta))):
        du/d_theta = -sum_l c_l * sqrt(l*(l+1)) * P_l^1_norm(cos(theta))

    Laplacian on the unit sphere (eigenvalue relation):
        nabla^2_sphere u = -sum_l c_l * l*(l+1) / R^2 * P_l_norm(cos(theta))

    Attributes:
    -----------
    grid : SphericalGrid1D
        The 1D Gauss-Legendre grid.
    """

    grid: SphericalGrid1D
    # Precomputed m=1 ALP matrix for gradient reconstruction
    _P1_matrix: Float[Array, "N N"]
    # Precomputed gradient coefficients: sqrt(l*(l+1))
    _grad_coeff: Float[Array, "N"]

    def __init__(self, grid: SphericalGrid1D):
        self.grid = grid
        N = grid.N
        nodes_np = np.array(grid._nodes)
        P1_np = _gradient_alp_matrix(N, nodes_np)
        self._P1_matrix = jnp.asarray(P1_np)
        l = np.arange(N, dtype=np.float64)
        self._grad_coeff = jnp.asarray(np.sqrt(l * (l + 1)))

    def to_spectral(self, u: Float[Array, "N"]) -> Float[Array, "N"]:
        """
        Forward Discrete Legendre Transform.

        c_l = sum_j w_j * P_l_norm(cos(theta_j)) * u(theta_j)

        Parameters:
        -----------
        u : Float[Array, "N"]
            Physical field at GL nodes.

        Returns:
        --------
        c : Float[Array, "N"]
            Legendre spectral coefficients.
        """
        return self.grid.transform(u, inverse=False)

    def from_spectral(self, c: Float[Array, "N"]) -> Float[Array, "N"]:
        """
        Inverse Discrete Legendre Transform.

        u(theta_j) ≈ sum_l c_l * P_l_norm(cos(theta_j))

        Parameters:
        -----------
        c : Float[Array, "N"]
            Legendre spectral coefficients.

        Returns:
        --------
        u : Float[Array, "N"]
            Physical field at GL nodes.
        """
        return self.grid.transform(c, inverse=True)

    def gradient(
        self, u: Float[Array, "N"], spectral: bool = False
    ) -> Float[Array, "N"]:
        """
        Colatitude gradient: du/d_theta.

        Algorithm:
            1. Forward DLT: c_l = P_norm_matrix @ (w * u)
            2. Multiply:    c_grad_l = -sqrt(l*(l+1)) * c_l
            3. Reconstruct: du/d_theta = P1_norm_matrix.T @ c_grad

        Parameters:
        -----------
        u : Float[Array, "N"]
            Physical field or spectral coefficients (if spectral=True).
        spectral : bool
            If True, u contains Legendre coefficients.

        Returns:
        --------
        du_dtheta : Float[Array, "N"]
            Colatitude derivative du/d_theta at GL nodes.
        """
        c = u if spectral else self.to_spectral(u)
        # scipy.special.lpmv uses the Condon-Shortley phase convention (-1)^m.
        # For m=1, lpmv(1, l, mu) = -sqrt(1-mu^2) * dP_l/dmu, so the stored
        # P1 matrix already carries the negative sign for nodes with mu > 0.
        # Consequently d P_l_norm / d_theta = +sqrt(l*(l+1)) * P_l^1_norm
        # (no additional minus sign needed here).
        c_grad = self._grad_coeff * c
        return self._P1_matrix.T @ c_grad

    def laplacian(
        self, u: Float[Array, "N"], spectral: bool = False
    ) -> Float[Array, "N"]:
        """
        Spherical Laplacian: (1/sin(theta)) * d/d_theta [sin(theta) * du/d_theta]
        (zonal, m=0 case): nabla^2_sphere u = -l*(l+1)/R^2 * u in spectral space.

        Parameters:
        -----------
        u : Float[Array, "N"]
            Physical field or spectral coefficients.
        spectral : bool
            If True, u contains Legendre coefficients.

        Returns:
        --------
        lap_u : Float[Array, "N"]
            Laplacian at GL nodes.
        """
        R = self.grid.L / jnp.pi  # sphere radius
        c = u if spectral else self.to_spectral(u)
        l = self.grid.l
        c_lap = -(l * (l + 1)) / (R**2) * c
        return self.from_spectral(c_lap)

    def __call__(
        self, u: Float[Array, "N"], order: int = 1, spectral: bool = False
    ) -> Float[Array, "N"]:
        """
        Apply derivative operator.

        Parameters:
        -----------
        u : Float[Array, "N"]
            Physical field or spectral coefficients.
        order : int
            1 = gradient (du/d_theta), 2 = Laplacian.
        spectral : bool
            If True, u contains Legendre coefficients.

        Returns:
        --------
        Array [N]
        """
        if order == 1:
            return self.gradient(u, spectral=spectral)
        elif order == 2:
            return self.laplacian(u, spectral=spectral)
        raise ValueError(f"order must be 1 or 2, got {order}")


# ============================================================================
# SphericalDerivative2D
# ============================================================================


class SphericalDerivative2D(eqx.Module):
    """
    2D pseudo-spectral derivative operators on the sphere.

    Computes physical-space differential operators (gradient, divergence, curl,
    Laplacian) for a field u(theta, phi) on the sphere using the full Spherical
    Harmonic Transform.

    Mathematical Formulation:
    -------------------------
    For a field u(theta, phi) on a sphere of radius R = Ly / pi:

        Gradient (covariant components):
            grad_theta u = (1/R) * du/d_theta
            grad_phi   u = 1 / (R * sin(theta)) * du/d_phi

        Divergence of vector (V_theta, V_phi) (in physical coordinates):
            div V = 1/(R*sin(theta)) * [d(V_theta*sin(theta))/d_theta + d(V_phi)/d_phi]

        Scalar curl (vertical component of curl):
            curl V = 1/(R*sin(theta)) * [d(V_theta)/d_phi - d(V_phi*sin(theta))/d_theta]

        Laplacian (via eigenvalue in spectral space):
            nabla^2 u = sum_{l,m} -l*(l+1)/R^2 * u_hat(l,m) * Y_l^m

    Attributes:
    -----------
    grid : SphericalGrid2D
        The full lat-lon grid.
    deriv_theta : SphericalDerivative1D
        1D operator reused for latitude derivatives.
    """

    grid: SphericalGrid2D
    deriv_theta: SphericalDerivative1D

    def __init__(self, grid: SphericalGrid2D):
        self.grid = grid
        # Build a compatible 1D grid from the latitude parameters
        grid_1d = SphericalGrid1D(
            N=grid.Ny,
            L=grid.Ly,
            dx=grid.dy,
            dealias=grid.dealias,
        )
        self.deriv_theta = SphericalDerivative1D(grid_1d)

    def gradient(
        self, u: Float[Array, "Ny Nx"], spectral: bool = False
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """
        Compute the gradient of u on the sphere.

            grad_theta u = (1/R) * du/d_theta
            grad_phi   u = 1 / (R * sin(theta)) * du/d_phi

        Parameters:
        -----------
        u : Float[Array, "Ny Nx"]
            Physical field.
        spectral : bool
            If True, u is already in spectral (SHT) space.

        Returns:
        --------
        (grad_theta, grad_phi) : tuple of Float[Array, "Ny Nx"]
            Covariant gradient components.
        """
        R = self.grid.Ly / jnp.pi

        # --- theta-component: vmap 1D gradient over longitude columns ---
        u_phys = self.grid.transform(u, inverse=True).real if spectral else u
        du_dtheta = jax.vmap(self.deriv_theta.gradient, in_axes=1, out_axes=1)(
            u_phys
        )  # (Ny, Nx)

        # --- phi-component via FFT ---
        u_hat_fft = jnp.fft.fft(u_phys, axis=-1)  # (Ny, Nx)
        m_phys = 2 * jnp.pi * jnp.fft.fftfreq(self.grid.Nx, self.grid.dx)  # (Nx,)
        du_dphi = jnp.fft.ifft(
            1j * m_phys[None, :] * u_hat_fft, axis=-1
        ).real  # (Ny, Nx)

        # Apply metric factors
        sin_theta = jnp.sin(self.grid.y)[:, None]  # (Ny, 1)
        grad_theta = du_dtheta / R
        grad_phi = du_dphi / (R * sin_theta)
        return grad_theta, grad_phi

    def divergence(
        self,
        v_theta: Float[Array, "Ny Nx"],
        v_phi: Float[Array, "Ny Nx"],
        spectral: bool = False,
    ) -> Float[Array, "Ny Nx"]:
        """
        Divergence of a vector field (V_theta, V_phi) on the sphere.

            div V = 1/(R*sin(theta)) * [d(V_theta*sin(theta))/d_theta + dV_phi/d_phi]

        Parameters:
        -----------
        v_theta : Float[Array, "Ny Nx"]
            Colatitude component (physical space).
        v_phi : Float[Array, "Ny Nx"]
            Longitude component (physical space).
        spectral : bool
            Unused (reserved for API consistency; inputs expected in physical space).

        Returns:
        --------
        div : Float[Array, "Ny Nx"]
            Scalar divergence field.
        """
        R = self.grid.Ly / jnp.pi
        sin_theta = jnp.sin(self.grid.y)[:, None]  # (Ny, 1)

        # d(V_theta * sin(theta))/d_theta
        vs = v_theta * sin_theta  # (Ny, Nx)
        d_vs_dtheta = jax.vmap(self.deriv_theta.gradient, in_axes=1, out_axes=1)(
            vs
        )  # (Ny, Nx)

        # dV_phi/d_phi via FFT
        m_phys = 2 * jnp.pi * jnp.fft.fftfreq(self.grid.Nx, self.grid.dx)  # (Nx,)
        vp_hat = jnp.fft.fft(v_phi, axis=-1)  # (Ny, Nx)
        d_vp_dphi = jnp.fft.ifft(
            1j * m_phys[None, :] * vp_hat, axis=-1
        ).real  # (Ny, Nx)

        return (d_vs_dtheta + d_vp_dphi) / (R * sin_theta)

    def curl(
        self,
        v_theta: Float[Array, "Ny Nx"],
        v_phi: Float[Array, "Ny Nx"],
        spectral: bool = False,
    ) -> Float[Array, "Ny Nx"]:
        """
        Vertical curl (scalar vorticity) of a 2D vector field on the sphere.

            curl V = 1/(R*sin(theta)) * [dV_theta/d_phi - d(V_phi*sin(theta))/d_theta]

        Parameters:
        -----------
        v_theta : Float[Array, "Ny Nx"]
            Colatitude component.
        v_phi : Float[Array, "Ny Nx"]
            Longitude component.
        spectral : bool
            Unused (reserved for API consistency).

        Returns:
        --------
        zeta : Float[Array, "Ny Nx"]
            Scalar vorticity field.
        """
        R = self.grid.Ly / jnp.pi
        sin_theta = jnp.sin(self.grid.y)[:, None]  # (Ny, 1)

        # dV_theta/d_phi via FFT
        m_phys = 2 * jnp.pi * jnp.fft.fftfreq(self.grid.Nx, self.grid.dx)  # (Nx,)
        vt_hat = jnp.fft.fft(v_theta, axis=-1)  # (Ny, Nx)
        d_vt_dphi = jnp.fft.ifft(
            1j * m_phys[None, :] * vt_hat, axis=-1
        ).real  # (Ny, Nx)

        # d(V_phi * sin(theta))/d_theta
        vs = v_phi * sin_theta  # (Ny, Nx)
        d_vs_dtheta = jax.vmap(self.deriv_theta.gradient, in_axes=1, out_axes=1)(
            vs
        )  # (Ny, Nx)

        return (d_vt_dphi - d_vs_dtheta) / (R * sin_theta)

    def laplacian(
        self, u: Float[Array, "Ny Nx"], spectral: bool = False
    ) -> Float[Array, "Ny Nx"]:
        """
        Scalar Laplace-Beltrami operator on the sphere.

        In spectral space:
            nabla^2 u_hat(l, m) = -l*(l+1)/R^2 * u_hat(l, m)

        Parameters:
        -----------
        u : Float[Array, "Ny Nx"]
            Physical field or spectral coefficients.
        spectral : bool
            If True, u is already in spectral space.

        Returns:
        --------
        lap_u : Float[Array, "Ny Nx"]
            Laplacian in physical space.
        """
        R = self.grid.Ly / jnp.pi
        u_hat = u if spectral else self.grid.transform(u)
        l = self.grid.l  # (Ny,)
        eig = -(l * (l + 1)) / (R**2)  # (Ny,)
        lap_hat = eig[:, None] * u_hat  # (Ny, Nx)
        return self.grid.transform(lap_hat, inverse=True)

    def advection_scalar(
        self,
        v_theta: Float[Array, "Ny Nx"],
        v_phi: Float[Array, "Ny Nx"],
        q: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """
        Pseudo-spectral scalar advection: (V · nabla) q.

        Computes the gradient of q spectrally, then multiplies by V in physical space:
            adv = V_theta * grad_theta(q) + V_phi * grad_phi(q)

        Parameters:
        -----------
        v_theta : Float[Array, "Ny Nx"]
            Colatitude velocity component.
        v_phi : Float[Array, "Ny Nx"]
            Longitude velocity component.
        q : Float[Array, "Ny Nx"]
            Scalar field to advect.

        Returns:
        --------
        adv : Float[Array, "Ny Nx"]
            Advection term at each grid point.
        """
        grad_theta_q, grad_phi_q = self.gradient(q)
        return v_theta * grad_theta_q + v_phi * grad_phi_q
