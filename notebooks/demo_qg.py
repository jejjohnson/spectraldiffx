# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Quasi-Geostrophic (QG) Turbulence Simulation
#
# This notebook demonstrates a 2D barotropic quasi-geostrophic turbulence simulation
# using the `spectraldiffx` library for pseudospectral differentiation.
#
# ## Governing Equations
#
# The barotropic QG equation:
# $$
# \partial_t q + \mathbf{u} \cdot \nabla q = \nu \nabla^{2n} q - \mu q
# $$
#
# where:
# - $q = \nabla^2 \psi$ is the relative vorticity
# - $\psi$ is the stream function
# - $u = -\partial_y \psi$ is the zonal velocity
# - $v = \partial_x \psi$ is the meridional velocity
# - $\nu$ is the (hyper)viscosity coefficient
# - $\mu$ is the linear drag coefficient
# - $n$ is the hyperviscosity order

# %%
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

# %%
from spectraldiffx._src.grid import FourierGrid2D
from spectraldiffx._src.operators import SpectralDerivative2D
from spectraldiffx._src.solvers import SpectralHelmholtzSolver2D
from spectraldiffx._src.filters import SpectralFilter2D

# %% [markdown]
# ## Domain Setup

# %%
# Grid parameters
Nx, Ny = 256, 256
Lx, Ly = 2 * math.pi, 2 * math.pi

# Create grid
grid = FourierGrid2D.from_N_L(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dealias="2/3")

# Create operators
deriv = SpectralDerivative2D(grid=grid)
solver = SpectralHelmholtzSolver2D(grid=grid)
filt = SpectralFilter2D(grid=grid)

# Get meshgrid for plotting
X, Y = grid.X

print(f"Grid: {Nx} x {Ny}")
print(f"Domain: [{0}, {Lx}] x [{0}, {Ly}]")
print(f"Resolution: dx={grid.dx:.4f}, dy={grid.dy:.4f}")

# %% [markdown]
# ## Physical Parameters

# %%
class QGParams(NamedTuple):
    """Parameters for QG simulation."""
    nu: float = 1e-4       # viscosity coefficient
    mu: float = 0.0        # linear drag coefficient
    nv: int = 1            # hyperviscosity order (1=Laplacian, 2=biharmonic)


params = QGParams(nu=1e-4, mu=0.0, nv=1)
print(f"Parameters: nu={params.nu}, mu={params.mu}, nv={params.nv}")

# %% [markdown]
# ## Initial Condition
#
# We generate random initial vorticity field with energy concentrated at
# intermediate scales (avoiding very large and very small scales).


# %%
def generate_initial_vorticity(grid: FourierGrid2D, seed: int = 42) -> jnp.ndarray:
    """
    Generate random initial vorticity field.

    Energy is band-passed to intermediate wavenumbers to create
    interesting turbulent dynamics.
    """
    key = jrandom.PRNGKey(seed)

    # Random field in physical space
    q0 = jrandom.normal(key, shape=(grid.Nx, grid.Ny))

    # Transform to spectral space
    q_hat = jnp.fft.fft2(q0)

    # Band-pass filter: keep modes with 3 < |k| < 10
    K2 = grid.K2
    k_mag = jnp.sqrt(K2)
    mask = (k_mag > 3.0) & (k_mag < 10.0)
    q_hat = jnp.where(mask, q_hat, 0.0)

    # Transform back and normalize
    q0 = jnp.real(jnp.fft.ifft2(q_hat))
    q0 = q0 / jnp.std(q0)  # Normalize variance

    return q0


q0 = generate_initial_vorticity(grid, seed=42)

# %% [markdown]
# ## Diagnostic Functions

# %%
def compute_stream_function(q: jnp.ndarray) -> jnp.ndarray:
    """
    Compute stream function from vorticity by solving Poisson equation.

    q = nabla^2 psi  =>  psi = nabla^{-2} q
    """
    return solver.solve(q, alpha=0.0, zero_mean=True)


def compute_velocities(psi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute velocity components from stream function.

    u = -d(psi)/dy
    v =  d(psi)/dx
    """
    dpsi_dx, dpsi_dy = deriv.gradient(psi)
    u = -dpsi_dy
    v = dpsi_dx
    return u, v


# %%
# Compute initial diagnostics
psi0 = compute_stream_function(q0)
u0, v0 = compute_velocities(psi0)

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Vorticity
im = axes[0, 0].pcolormesh(X.T, Y.T, q0.T, cmap="RdBu_r", shading="auto")
axes[0, 0].set_title("Vorticity $q$")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
plt.colorbar(im, ax=axes[0, 0])

# Stream function
im = axes[0, 1].pcolormesh(X.T, Y.T, psi0.T, cmap="viridis", shading="auto")
axes[0, 1].set_title(r"Stream function $\psi$")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")
plt.colorbar(im, ax=axes[0, 1])

# Zonal velocity
im = axes[1, 0].pcolormesh(X.T, Y.T, u0.T, cmap="RdBu_r", shading="auto")
axes[1, 0].set_title("Zonal velocity $u$")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")
plt.colorbar(im, ax=axes[1, 0])

# Meridional velocity
im = axes[1, 1].pcolormesh(X.T, Y.T, v0.T, cmap="RdBu_r", shading="auto")
axes[1, 1].set_title("Meridional velocity $v$")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Right-Hand Side Function
#
# The tendency (time derivative) of vorticity includes:
# 1. Advection: $-\mathbf{u} \cdot \nabla q$
# 2. Diffusion: $\nu \nabla^{2n} q$
# 3. Linear drag: $-\mu q$

# %%
def qg_tendency(q: jnp.ndarray, params: QGParams) -> jnp.ndarray:
    """
    Compute the right-hand side of the QG equation.

    dq/dt = -u * dq/dx - v * dq/dy + nu * nabla^{2n} q - mu * q
    """
    # Compute stream function and velocities
    psi = compute_stream_function(q)
    u, v = compute_velocities(psi)

    # Advection term: -(u * dq/dx + v * dq/dy)
    advection = -deriv.advection_scalar(u, v, q)

    # Initialize RHS with advection
    rhs = advection

    # Diffusion term: nu * nabla^{2n} q
    if params.nu > 0:
        # Apply Laplacian n times
        lap_q = q
        for _ in range(params.nv):
            lap_q = deriv.laplacian(lap_q)
        rhs = rhs + params.nu * lap_q

    # Linear drag: -mu * q
    if params.mu > 0:
        rhs = rhs - params.mu * q

    return rhs


# %% [markdown]
# ## Time Integration
#
# We use a simple 4th-order Runge-Kutta scheme for time integration.

# %%
def rk4_step(q: jnp.ndarray, dt: float, params: QGParams) -> jnp.ndarray:
    """Single RK4 time step."""
    k1 = qg_tendency(q, params)
    k2 = qg_tendency(q + 0.5 * dt * k1, params)
    k3 = qg_tendency(q + 0.5 * dt * k2, params)
    k4 = qg_tendency(q + dt * k3, params)
    return q + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@jax.jit
def integrate_steps(q: jnp.ndarray, dt: float, n_steps: int, params: QGParams) -> jnp.ndarray:
    """Integrate for n_steps using RK4."""
    def body_fn(i, q):
        return rk4_step(q, dt, params)
    return jax.lax.fori_loop(0, n_steps, body_fn, q)


# %% [markdown]
# ## Run Simulation

# %%
# Time stepping parameters
dt = 0.01
t_final = 50.0
n_steps_per_output = 100
n_outputs = int(t_final / (dt * n_steps_per_output))

print(f"dt = {dt}")
print(f"t_final = {t_final}")
print(f"Total steps = {int(t_final / dt)}")
print(f"Outputs every {n_steps_per_output} steps")

# %%
# Storage for outputs
times = [0.0]
q_history = [q0]

# Run simulation
q = q0
for i in range(n_outputs):
    q = integrate_steps(q, dt, n_steps_per_output, params)
    t = (i + 1) * n_steps_per_output * dt
    times.append(t)
    q_history.append(q)

    if (i + 1) % 10 == 0:
        print(f"t = {t:.1f}, max|q| = {jnp.abs(q).max():.3f}")

print("Simulation complete!")

# %% [markdown]
# ## Visualize Results

# %%
# Select times to plot
plot_indices = [0, len(times) // 4, len(times) // 2, -1]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

vmax = max(jnp.abs(q_history[i]).max() for i in plot_indices)

for ax, idx in zip(axes, plot_indices):
    q_plot = q_history[idx]
    t_plot = times[idx]

    im = ax.pcolormesh(X.T, Y.T, q_plot.T, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_title(f"$t = {t_plot:.1f}$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax)

plt.suptitle("Vorticity Evolution", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Final State with Velocity Vectors

# %%
q_final = q_history[-1]
psi_final = compute_stream_function(q_final)
u_final, v_final = compute_velocities(psi_final)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Vorticity with velocity vectors
skip = max(1, Nx // 32)  # Subsample for quiver plot
im = axes[0].pcolormesh(X.T, Y.T, q_final.T, cmap="RdBu_r", shading="auto")
axes[0].quiver(X[::skip, ::skip].T, Y[::skip, ::skip].T,
               u_final[::skip, ::skip].T, v_final[::skip, ::skip].T,
               color='black', alpha=0.7)
axes[0].set_title(f"Vorticity & velocity (t={times[-1]:.1f})")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
plt.colorbar(im, ax=axes[0])

# Stream function with velocity vectors
im = axes[1].pcolormesh(X.T, Y.T, psi_final.T, cmap="viridis", shading="auto")
axes[1].quiver(X[::skip, ::skip].T, Y[::skip, ::skip].T,
               u_final[::skip, ::skip].T, v_final[::skip, ::skip].T,
               color='white', alpha=0.7)
axes[1].set_title(f"Stream function & velocity (t={times[-1]:.1f})")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Energy and Enstrophy Diagnostics

# %%
def compute_energy(q: jnp.ndarray) -> float:
    """Compute kinetic energy: E = 0.5 * integral(u^2 + v^2)."""
    psi = compute_stream_function(q)
    u, v = compute_velocities(psi)
    return 0.5 * jnp.mean(u**2 + v**2)


def compute_enstrophy(q: jnp.ndarray) -> float:
    """Compute enstrophy: Z = 0.5 * integral(q^2)."""
    return 0.5 * jnp.mean(q**2)


# %%
energies = [compute_energy(q) for q in q_history]
enstrophies = [compute_enstrophy(q) for q in q_history]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(times, energies, 'b-', linewidth=2)
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Energy")
axes[0].set_title("Kinetic Energy")
axes[0].grid(True, alpha=0.3)

axes[1].plot(times, enstrophies, 'r-', linewidth=2)
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Enstrophy")
axes[1].set_title("Enstrophy")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Energy: initial={energies[0]:.4f}, final={energies[-1]:.4f}, "
      f"change={100*(energies[-1]/energies[0] - 1):.2f}%")
print(f"Enstrophy: initial={enstrophies[0]:.4f}, final={enstrophies[-1]:.4f}, "
      f"change={100*(enstrophies[-1]/enstrophies[0] - 1):.2f}%")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated:
#
# 1. **Grid setup** with `FourierGrid2D`
# 2. **Spectral derivatives** with `SpectralDerivative2D`:
#    - `gradient()` for velocity from stream function
#    - `advection_scalar()` for nonlinear advection term
#    - `laplacian()` for diffusion
# 3. **Poisson solver** with `SpectralHelmholtzSolver2D` to invert $q = \nabla^2 \psi$
# 4. **Time integration** using RK4 with JAX's `lax.fori_loop`
#
# The spectral method provides:
# - Spectral accuracy for spatial derivatives
# - Efficient FFT-based computations
# - Built-in dealiasing (2/3 rule)
