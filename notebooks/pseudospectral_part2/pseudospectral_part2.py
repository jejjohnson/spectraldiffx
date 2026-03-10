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
# # QG Simulation with Diffrax Integration
#
# This notebook demonstrates a 2D barotropic quasi-geostrophic turbulence simulation
# using the `spectraldiffx` library for pseudospectral differentiation and
# `diffrax` for adaptive time stepping.
#
# ## Key Features Demonstrated:
# - Using diffrax for adaptive ODE integration
# - McWilliams (1984) initial conditions
# - Hyperviscosity for numerical stability
# - Energy and enstrophy diagnostics

# %%
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

# %%
from spectraldiffx._src.grid import FourierGrid2D
from spectraldiffx._src.operators import SpectralDerivative2D
from spectraldiffx._src.solvers import SpectralHelmholtzSolver2D

# %% [markdown]
# ## Domain Setup

# %%
# Grid parameters
Nx, Ny = 128, 128
Lx, Ly = 2 * math.pi, 2 * math.pi

# Create grid and operators
grid = FourierGrid2D.from_N_L(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dealias="2/3")
deriv = SpectralDerivative2D(grid=grid)
solver = SpectralHelmholtzSolver2D(grid=grid)

# Get meshgrid for plotting
X, Y = grid.X

print(f"Grid: {Nx} x {Ny}")
print(f"Domain: [0, {Lx:.4f}] x [0, {Ly:.4f}]")

# %% [markdown]
# ## Physical Parameters

# %%
class QGParams(NamedTuple):
    """Parameters for QG simulation with hyperviscosity."""
    nu: float = 1e-8       # hyperviscosity coefficient
    mu: float = 0.0        # linear drag coefficient
    nv: int = 2            # hyperviscosity order (2=biharmonic)


params = QGParams(nu=1e-8, mu=0.0, nv=2)
print(f"Hyperviscosity: nu={params.nu}, order={params.nv}")

# %% [markdown]
# ## McWilliams (1984) Initial Condition
#
# Generate random initial conditions with a specific energy spectrum,
# as used in the classic decaying 2D turbulence study.


# %%
def mcwilliams_ic(grid: FourierGrid2D, seed: int = 42) -> jnp.ndarray:
    """
    Generate McWilliams (1984) initial condition.

    Creates random vorticity with energy spectrum peaked at intermediate scales.
    """
    rng = np.random.RandomState(seed)

    # Wavenumber magnitudes
    K2 = grid.K2
    wv = np.sqrt(np.array(K2))

    # Energy spectrum: peaked at intermediate wavenumbers
    # ckappa = k * (1 + (k^2/36)^2)^{-1/2}
    fk = wv != 0
    ckappa = np.zeros_like(wv)
    ckappa[fk] = np.sqrt(wv[fk]**2 * (1.0 + (wv[fk]**2 / 36.0)**2))**(-1)

    # Random stream function in spectral space
    psi_hat = (rng.randn(Nx, Ny) + 1j * rng.randn(Nx, Ny)) * ckappa

    # Transform to physical space
    psi = np.fft.ifft2(psi_hat).real
    psi = psi - psi.mean()

    # Compute vorticity q = nabla^2 psi
    psi_hat = np.fft.fft2(psi)
    q_hat = -K2 * psi_hat
    q = np.fft.ifft2(q_hat).real

    # Normalize to unit energy
    q_hat = np.fft.fft2(q)
    energy = np.sum(np.abs(q_hat)**2) / (Nx * Ny)**2
    q = q / np.sqrt(energy)

    return jnp.asarray(q)


q0 = mcwilliams_ic(grid, seed=42)

# %% [markdown]
# ## Diagnostic Functions

# %%
def compute_stream_function(q: jnp.ndarray) -> jnp.ndarray:
    """Compute stream function from vorticity."""
    return solver.solve(q, alpha=0.0, zero_mean=True)


def compute_velocities(psi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute velocity from stream function."""
    dpsi_dx, dpsi_dy = deriv.gradient(psi)
    return -dpsi_dy, dpsi_dx


def compute_energy(q: jnp.ndarray) -> float:
    """Kinetic energy."""
    psi = compute_stream_function(q)
    u, v = compute_velocities(psi)
    return 0.5 * jnp.mean(u**2 + v**2)


def compute_enstrophy(q: jnp.ndarray) -> float:
    """Enstrophy."""
    return 0.5 * jnp.mean(q**2)


# %%
# Visualize initial condition
psi0 = compute_stream_function(q0)
u0, v0 = compute_velocities(psi0)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

im = axes[0].pcolormesh(X.T, Y.T, q0.T, cmap="RdBu_r", shading="auto")
axes[0].set_title("Vorticity $q_0$")
plt.colorbar(im, ax=axes[0])

im = axes[1].pcolormesh(X.T, Y.T, psi0.T, cmap="viridis", shading="auto")
axes[1].set_title(r"Stream function $\psi_0$")
plt.colorbar(im, ax=axes[1])

# Speed
speed = jnp.sqrt(u0**2 + v0**2)
im = axes[2].pcolormesh(X.T, Y.T, speed.T, cmap="magma", shading="auto")
axes[2].set_title("Speed $|\\mathbf{u}|$")
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.show()

print(f"Initial energy: {compute_energy(q0):.6f}")
print(f"Initial enstrophy: {compute_enstrophy(q0):.6f}")

# %% [markdown]
# ## Right-Hand Side with Hyperviscosity

# %%
def qg_tendency(q: jnp.ndarray, params: QGParams) -> jnp.ndarray:
    """
    QG tendency with hyperviscosity.

    dq/dt = -u * dq/dx - v * dq/dy + nu * nabla^{2n} q
    """
    # Stream function and velocities
    psi = compute_stream_function(q)
    u, v = compute_velocities(psi)

    # Advection
    rhs = -deriv.advection_scalar(u, v, q)

    # Hyperviscosity: nu * nabla^{2n} q
    if params.nu > 0:
        lap_q = q
        for _ in range(params.nv):
            lap_q = deriv.laplacian(lap_q)
        rhs = rhs + params.nu * lap_q

    # Linear drag
    if params.mu > 0:
        rhs = rhs - params.mu * q

    return rhs


# %% [markdown]
# ## Diffrax Integration
#
# We use diffrax with adaptive time stepping (Dormand-Prince 5th order).

# %%
try:
    import diffrax as dfx
    HAS_DIFFRAX = True
except ImportError:
    print("diffrax not installed. Using simple RK4 integration instead.")
    HAS_DIFFRAX = False

# %%
if HAS_DIFFRAX:
    class State(NamedTuple):
        q: jnp.ndarray

    def vector_field(t, state: State, args) -> State:
        params = args
        rhs = qg_tendency(state.q, params)
        return State(q=rhs)

    def integrate_diffrax(q0, t0, t1, dt0, params, saveat=None):
        """Integrate using diffrax."""
        solver = dfx.Tsit5()  # 5th order Tsitouras method
        stepsize_controller = dfx.PIDController(rtol=1e-5, atol=1e-5)

        sol = dfx.diffeqsolve(
            terms=dfx.ODETerm(vector_field),
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=State(q=q0),
            saveat=saveat,
            args=params,
            stepsize_controller=stepsize_controller,
            max_steps=50000,
        )
        return sol

# %% [markdown]
# ## Run Simulation

# %%
# Time parameters
t0, t1 = 0.0, 20.0
dt0 = 0.01
n_saves = 50

# %%
if HAS_DIFFRAX:
    # Save at regular intervals
    t_save = np.linspace(t0, t1, n_saves)
    saveat = dfx.SaveAt(ts=t_save)

    print(f"Integrating from t={t0} to t={t1}...")
    sol = integrate_diffrax(q0, t0, t1, dt0, params, saveat=saveat)

    times = sol.ts
    q_history = sol.ys.q
    print(f"Completed! Saved {len(times)} snapshots.")
else:
    # Fallback: simple RK4
    def rk4_step(q, dt, params):
        k1 = qg_tendency(q, params)
        k2 = qg_tendency(q + 0.5*dt*k1, params)
        k3 = qg_tendency(q + 0.5*dt*k2, params)
        k4 = qg_tendency(q + dt*k3, params)
        return q + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    dt = 0.01
    n_steps = int(t1 / dt)
    save_every = n_steps // n_saves

    times = [0.0]
    q_history = [q0]
    q = q0

    for i in range(n_steps):
        q = rk4_step(q, dt, params)
        if (i + 1) % save_every == 0:
            times.append((i + 1) * dt)
            q_history.append(q)

    times = jnp.array(times)
    q_history = jnp.stack(q_history)

# %% [markdown]
# ## Visualize Evolution

# %%
# Select snapshots to plot
n_plots = 4
indices = np.linspace(0, len(times) - 1, n_plots, dtype=int)

fig, axes = plt.subplots(1, n_plots, figsize=(14, 3.5))

vmax = jnp.max(jnp.abs(q_history))

for ax, idx in zip(axes, indices):
    q_plot = q_history[idx]
    t_plot = times[idx]

    im = ax.pcolormesh(X.T, Y.T, q_plot.T, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_title(f"$t = {t_plot:.1f}$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

plt.suptitle("Vorticity Evolution (McWilliams 1984 IC)", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Final State

# %%
q_final = q_history[-1]
psi_final = compute_stream_function(q_final)
u_final, v_final = compute_velocities(psi_final)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Vorticity with velocity vectors
skip = max(1, Nx // 24)
im = axes[0].pcolormesh(X.T, Y.T, q_final.T, cmap="RdBu_r", shading="auto")
axes[0].quiver(X[::skip, ::skip].T, Y[::skip, ::skip].T,
               u_final[::skip, ::skip].T, v_final[::skip, ::skip].T,
               color='black', alpha=0.6, scale=30)
axes[0].set_title(f"Vorticity (t={times[-1]:.1f})")
plt.colorbar(im, ax=axes[0])

# Stream function
im = axes[1].pcolormesh(X.T, Y.T, psi_final.T, cmap="viridis", shading="auto")
axes[1].contour(X.T, Y.T, psi_final.T, colors='white', alpha=0.5, linewidths=0.5)
axes[1].set_title(f"Stream function (t={times[-1]:.1f})")
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Conservation Diagnostics

# %%
energies = [compute_energy(q_history[i]) for i in range(len(times))]
enstrophies = [compute_enstrophy(q_history[i]) for i in range(len(times))]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(times, energies, 'b-', linewidth=2)
axes[0].axhline(energies[0], color='gray', linestyle='--', alpha=0.5, label='Initial')
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Energy")
axes[0].set_title("Kinetic Energy Conservation")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(times, enstrophies, 'r-', linewidth=2)
axes[1].axhline(enstrophies[0], color='gray', linestyle='--', alpha=0.5, label='Initial')
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Enstrophy")
axes[1].set_title("Enstrophy (decays due to hyperviscosity)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
energy_change = (energies[-1] - energies[0]) / energies[0] * 100
enstrophy_change = (enstrophies[-1] - enstrophies[0]) / enstrophies[0] * 100

print(f"Energy change: {energy_change:+.4f}%")
print(f"Enstrophy change: {enstrophy_change:+.4f}%")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated:
#
# 1. **McWilliams (1984) initial conditions** - realistic decaying turbulence setup
# 2. **Hyperviscosity** ($\nabla^4$) for numerical stability at high resolution
# 3. **Diffrax integration** with adaptive time stepping
# 4. **Conservation diagnostics** - energy conservation and enstrophy decay
#
# The spectraldiffx library provides:
# - `FourierGrid2D` for domain setup
# - `SpectralDerivative2D` for gradient, Laplacian, and advection
# - `SpectralHelmholtzSolver2D` for Poisson inversion
