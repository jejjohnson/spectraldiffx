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
# # Decaying 2D Barotropic Quasi-Geostrophic Turbulence
#
# ## Introduction
#
# Two-dimensional quasi-geostrophic (QG) turbulence is a cornerstone model in
# geophysical fluid dynamics. It captures the leading-order dynamics of
# large-scale atmospheric and oceanic flows, where rotation and stratification
# constrain the motion to be nearly two-dimensional.
#
# The central phenomenon is the **dual cascade**:
#
# - **Inverse energy cascade**: kinetic energy flows from small scales
#   (where it is injected or initialized) toward *larger* scales, producing
#   coherent vortices that grow by merging.
# - **Forward enstrophy cascade**: enstrophy (mean-square vorticity) flows
#   toward *smaller* scales, where it is dissipated by viscosity. This
#   generates fine filamentary structures between the vortices.
#
# This dual cascade was first predicted theoretically by Kraichnan (1967) and
# Batchelor (1969). Charney (1971) showed that quasi-geostrophic dynamics
# exhibit the same dual cascade as 2D Navier--Stokes. McWilliams (1984)
# performed landmark numerical simulations demonstrating vortex emergence and
# merging from random initial conditions.
#
# **What this notebook demonstrates:**
#
# - Setting up a doubly-periodic pseudospectral QG solver with `spectraldiffx`
# - `FourierGrid2D` for the spectral domain
# - `SpectralDerivative2D` for gradient, Laplacian, and dealiased advection
# - `SpectralHelmholtzSolver2D` for Poisson inversion ($q = \nabla^2 \psi$)
# - `SpectralFilter2D` for spectral filtering
# - RK4 time integration via `jax.lax.fori_loop`
# - Diagnostics: energy and enstrophy evolution showing the dual cascade

# %% [markdown]
# ## Governing Equations
#
# The barotropic QG vorticity equation on a doubly-periodic domain is:
#
# $$
# \frac{\partial q}{\partial t} + J(\psi, q) = \nu \nabla^{2n} q - \mu q
# $$
#
# where $q$ is the relative vorticity (potential vorticity in the barotropic
# limit) and $\psi$ is the stream function. They are related by the Poisson
# equation:
#
# $$
# q = \nabla^2 \psi
# $$
#
# ### The Jacobian (advection)
#
# The Jacobian $J(\psi, q)$ represents advection of vorticity by the
# geostrophic flow:
#
# $$
# J(\psi, q) = \frac{\partial \psi}{\partial x}\frac{\partial q}{\partial y}
#            - \frac{\partial \psi}{\partial y}\frac{\partial q}{\partial x}
#            = u \frac{\partial q}{\partial x} + v \frac{\partial q}{\partial y}
# $$
#
# where the velocity components are derived from the stream function:
#
# $$
# u = -\frac{\partial \psi}{\partial y}, \qquad v = \frac{\partial \psi}{\partial x}
# $$
#
# This guarantees that the flow is non-divergent: $\nabla \cdot \mathbf{u} = 0$.
#
# ### Hyperviscosity
#
# The term $\nu \nabla^{2n} q$ is hyperviscosity of order $n$. The sign
# convention is:
#
# $$
# \text{dissipation} = (-1)^{n+1} \, \nu \, (\nabla^2)^n \, q
# $$
#
# - For $n=1$ (Laplacian viscosity): $(-1)^2 \nu \nabla^2 q = +\nu \nabla^2 q$.
#   Since $\nabla^2$ has negative eigenvalues ($-|k|^2$), this *dissipates*
#   energy as expected.
# - For $n=2$ (biharmonic): $(-1)^3 \nu \nabla^4 q = -\nu \nabla^4 q$.
#   The eigenvalues of $\nabla^4$ are $|k|^4 > 0$, so the minus sign ensures
#   dissipation.
#
# Higher-order hyperviscosity concentrates dissipation at the smallest resolved
# scales, leaving the large-scale dynamics less contaminated by artificial
# diffusion.
#
# ### Linear drag (Ekman friction)
#
# The term $-\mu q$ models bottom friction (Ekman drag). It removes energy
# at all scales equally, preventing the inverse cascade from piling up energy
# at the domain scale. In this notebook we set $\mu = 0$ (free decay).
#
# ### Conserved quantities (inviscid limit)
#
# When $\nu = \mu = 0$, the QG equation conserves two quadratic invariants:
#
# - **Kinetic energy**: $E = \frac{1}{2} \langle |\mathbf{u}|^2 \rangle
#   = \frac{1}{2} \langle u^2 + v^2 \rangle$
# - **Enstrophy**: $Z = \frac{1}{2} \langle q^2 \rangle$
#
# With viscosity, both decay, but enstrophy decays much faster because it
# is dominated by small-scale contributions that are selectively dissipated.

# %% [markdown]
# ## Pseudospectral Algorithm
#
# The simulation advances vorticity $q$ in time using a pseudospectral method
# with explicit RK4 integration. At each RK4 substep, we evaluate the
# right-hand side as follows:
#
# ```
# Pseudospectral RHS evaluation
# =============================
#
# Input: q(x, y) on the physical grid         shape: [Nx, Ny] real
#
# Step 1.  q_hat = FFT2(q)                    [Nx, Ny] -> [Nx, Ny] complex
#          Transform vorticity to spectral space.
#
# Step 2.  psi_hat = -q_hat / |k|^2           Poisson inversion
#          Solve nabla^2 psi = q in spectral space.
#          (k=0 mode set to zero for zero-mean psi.)
#
# Step 3.  u_hat = -i*ky * psi_hat            Velocity from stream function
#          v_hat = +i*kx * psi_hat
#
# Step 4.  u, v, q = IFFT2(u_hat, v_hat, q_hat)
#          Transform back to physical space.
#
# Step 5.  rhs = -dealiased(u * dq/dx + v * dq/dy)   Advection (2/3 rule)
#              + (-1)^(nv+1) * nu * (nabla^2)^nv q    Hyperviscosity
#              - mu * q                                 Ekman drag
#
# Output: dq/dt on the physical grid           shape: [Nx, Ny] real
# ```
#
# The 2/3 dealiasing rule zeros out the upper third of wavenumbers before
# computing the nonlinear product, preventing aliasing errors from corrupting
# the resolved modes.

# %%
import math
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

jax.config.update("jax_enable_x64", True)

IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "images" / "demo_qg"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %%
import equinox as eqx
from spectraldiffx import (
    FourierGrid2D,
    SpectralDerivative2D,
    SpectralFilter2D,
    SpectralHelmholtzSolver2D,
)

# %% [markdown]
# ## Domain and Parameters
#
# We use a $256 \times 256$ grid on a doubly-periodic domain $[0, 2\pi)^2$.
#
# ### Parameter choices
#
# | Parameter | Value  | Physical meaning |
# |-----------|--------|-----------------|
# | `Nx, Ny`  | 256    | Grid resolution. Gives $k_{\max} = 128$. |
# | `Lx, Ly`  | $2\pi$ | Domain size. Wavenumber spacing $\Delta k = 1$. |
# | `nu`      | 5e-4   | Laplacian viscosity coefficient. |
# | `nv`      | 1      | Viscosity order (Laplacian, $\nabla^2 q$). |
# | `mu`      | 0.0    | No Ekman drag (free decay). |
# | `dt`      | 0.005  | Time step (CFL-limited). |
# | `dealias` | 2/3    | Standard 2/3 dealiasing rule. |
#
# **Why Laplacian viscosity (nv=1) instead of biharmonic (nv=2)?**
#
# With explicit RK4, the stability constraint is approximately:
#
# $$
# \Delta t \cdot \nu \cdot k_{\max}^{2n} < 2.8
# $$
#
# For $n_v = 1$: $0.005 \times 5 \times 10^{-4} \times 128^2 = 0.041 \ll 2.8$ (stable).
#
# For $n_v = 2$ with the same $\nu$: $0.005 \times 5 \times 10^{-4} \times 128^4 = 671$
# (catastrophically unstable). One would need $\nu \sim 10^{-12}$ for $n_v = 2$
# at this resolution with explicit RK4, but then the viscosity would be too
# weak to prevent enstrophy buildup. An adaptive or implicit integrator
# (see `pseudospectral_part2` notebook) is needed for high-order hyperviscosity.

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

print(f"Grid:       {Nx} x {Ny}")
print(f"Domain:     [0, {Lx:.4f}] x [0, {Ly:.4f}]")
print(f"Resolution: dx = {grid.dx:.6f}, dy = {grid.dy:.6f}")
print(f"k_max:      {Nx // 2}")
print(f"X shape:    {X.shape}")
print(f"Y shape:    {Y.shape}")

# %%
class QGParams(NamedTuple):
    """Parameters for the barotropic QG simulation."""
    nu: float = 1e-4   # (hyper)viscosity coefficient
    mu: float = 0.0    # linear drag (Ekman friction) coefficient
    nv: int = 1        # viscosity order: 1=Laplacian, 2=biharmonic, ...


params = QGParams(nu=5e-4, mu=0.0, nv=1)

print(f"Viscosity:       nu = {params.nu}")
print(f"Viscosity order: nv = {params.nv}  (Laplacian)")
print(f"Ekman drag:      mu = {params.mu}")

# Verify RK4 stability
k_max = Nx // 2
dt = 0.005
stability = dt * params.nu * k_max ** (2 * params.nv)
print(f"\nRK4 stability number: dt * nu * k_max^(2*nv) = {stability:.4f}  (must be < 2.8)")

# %% [markdown]
# ## Initial Condition
#
# We generate a random vorticity field with energy concentrated at
# intermediate wavenumbers ($3 < |k| < 10$). This mimics a state where
# energy has been injected at mesoscales and has not yet cascaded to
# either extreme of the spectrum.
#
# The procedure:
# 1. Generate white noise in physical space.
# 2. Transform to spectral space.
# 3. Apply a band-pass filter that retains only modes with
#    $3 < |\mathbf{k}| < 10$.
# 4. Transform back and normalize to unit variance.
#
# The band-pass filter seeds the dual cascade: the inverse cascade will
# move energy to $|k| < 3$ (large vortices), while the forward enstrophy
# cascade will populate $|k| > 10$ (thin filaments).


# %%
def generate_initial_vorticity(grid: FourierGrid2D, seed: int = 42) -> jnp.ndarray:
    """
    Generate random initial vorticity with a band-pass spectral filter.

    Parameters
    ----------
    grid : FourierGrid2D
        Spectral grid providing wavenumber arrays.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    q0 : jnp.ndarray, shape [Nx, Ny]
        Initial vorticity field normalized to unit variance.
    """
    key = jrandom.PRNGKey(seed)

    # White noise in physical space
    q0 = jrandom.normal(key, shape=(grid.Nx, grid.Ny))
    print(f"  White noise:   shape = {q0.shape}, std = {float(jnp.std(q0)):.4f}")

    # Transform to spectral space
    q_hat = jnp.fft.fft2(q0)
    print(f"  Spectral:      shape = {q_hat.shape}, dtype = {q_hat.dtype}")

    # Band-pass filter: retain 3 < |k| < 10
    K2 = grid.K2
    k_mag = jnp.sqrt(K2)
    mask = (k_mag > 3.0) & (k_mag < 10.0)
    n_retained = int(jnp.sum(mask))
    print(f"  Band-pass:     {n_retained} modes retained out of {Nx * Ny}")

    q_hat = jnp.where(mask, q_hat, 0.0)

    # Transform back and normalize
    q0 = jnp.real(jnp.fft.ifft2(q_hat))
    q0 = q0 / jnp.std(q0)
    print(f"  Normalized:    shape = {q0.shape}, std = {float(jnp.std(q0)):.4f}")

    return q0


q0 = generate_initial_vorticity(grid, seed=42)

# %% [markdown]
# ## Diagnostic Functions
#
# We define helper functions to extract physical quantities from the
# vorticity field $q$:
#
# 1. **Stream function** $\psi$: solve $\nabla^2 \psi = q$ (Poisson equation).
# 2. **Velocity** $(u, v)$: $u = -\partial_y \psi$, $v = \partial_x \psi$.
# 3. **Kinetic energy**: $E = \tfrac{1}{2}\langle u^2 + v^2 \rangle$.
# 4. **Enstrophy**: $Z = \tfrac{1}{2}\langle q^2 \rangle$.
#
# Note that energy can also be computed purely from the vorticity spectrum:
# $E = \tfrac{1}{2} \sum_k |\hat{q}_k|^2 / |k|^2$, but the physical-space
# formula is more transparent.

# %%
def compute_stream_function(q: jnp.ndarray) -> jnp.ndarray:
    """
    Solve the Poisson equation nabla^2 psi = q for the stream function.

    Uses SpectralHelmholtzSolver2D with alpha=0 (pure Poisson) and
    zero_mean=True to fix the gauge freedom.

    Parameters
    ----------
    q : jnp.ndarray, shape [Nx, Ny]
        Vorticity field.

    Returns
    -------
    psi : jnp.ndarray, shape [Nx, Ny]
        Stream function with zero spatial mean.
    """
    return solver.solve(q, alpha=0.0, zero_mean=True)


def compute_velocities(psi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute velocity components from stream function.

    u = -dpsi/dy   (zonal, eastward)
    v = +dpsi/dx   (meridional, northward)

    Parameters
    ----------
    psi : jnp.ndarray, shape [Nx, Ny]

    Returns
    -------
    u, v : tuple of jnp.ndarray, each shape [Nx, Ny]
    """
    dpsi_dx, dpsi_dy = deriv.gradient(psi)
    u = -dpsi_dy
    v = dpsi_dx
    return u, v


def compute_energy(q: jnp.ndarray) -> float:
    """
    Kinetic energy: E = (1/2) * <u^2 + v^2>.

    Computed from vorticity via Poisson inversion.
    """
    psi = compute_stream_function(q)
    u, v = compute_velocities(psi)
    return 0.5 * jnp.mean(u**2 + v**2)


def compute_enstrophy(q: jnp.ndarray) -> float:
    """Enstrophy: Z = (1/2) * <q^2>."""
    return 0.5 * jnp.mean(q**2)


# %% [markdown]
# ## Visualize Initial State
#
# We show four fields derived from the initial vorticity:
# vorticity $q$, stream function $\psi$, zonal velocity $u$, and
# meridional velocity $v$.

# %%
# Compute derived fields from initial vorticity
psi0 = compute_stream_function(q0)
u0, v0 = compute_velocities(psi0)

print(f"Initial vorticity:       shape = {q0.shape},   range = [{float(q0.min()):.3f}, {float(q0.max()):.3f}]")
print(f"Initial stream function: shape = {psi0.shape}, range = [{float(psi0.min()):.3f}, {float(psi0.max()):.3f}]")
print(f"Initial u velocity:      shape = {u0.shape},   range = [{float(u0.min()):.3f}, {float(u0.max()):.3f}]")
print(f"Initial v velocity:      shape = {v0.shape},   range = [{float(v0.min()):.3f}, {float(v0.max()):.3f}]")
print(f"\nInitial energy:    E0 = {float(compute_energy(q0)):.6f}")
print(f"Initial enstrophy: Z0 = {float(compute_enstrophy(q0)):.6f}")

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
axes[1, 0].set_title("Zonal velocity $u = -\\partial_y \\psi$")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")
plt.colorbar(im, ax=axes[1, 0])

# Meridional velocity
im = axes[1, 1].pcolormesh(X.T, Y.T, v0.T, cmap="RdBu_r", shading="auto")
axes[1, 1].set_title("Meridional velocity $v = \\partial_x \\psi$")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")
plt.colorbar(im, ax=axes[1, 1])

plt.suptitle("Initial State ($t = 0$)", fontsize=14)
plt.tight_layout()
fig.savefig(IMG_DIR / "initial_state.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Initial state: vorticity, stream function, and velocities](../images/demo_qg/initial_state.png)

# %% [markdown]
# ## Right-Hand Side with Hyperviscosity
#
# The tendency (time derivative) of vorticity consists of three terms:
#
# 1. **Advection**: $-J(\psi, q) = -(u \, \partial_x q + v \, \partial_y q)$.
#    Computed with `deriv.advection_scalar` which applies 2/3 dealiasing.
#
# 2. **Hyperviscosity**: $(-1)^{n_v+1} \, \nu \, (\nabla^2)^{n_v} \, q$.
#    For $n_v = 1$, this is $+\nu \nabla^2 q$ (ordinary Laplacian diffusion).
#    The sign convention ensures energy dissipation regardless of the order.
#
# 3. **Linear drag**: $-\mu \, q$ (Ekman friction, zero in this simulation).
#
# The sign formula $(-1)^{n_v+1}$ deserves careful explanation:
#
# - The Laplacian eigenvalue for wavenumber $k$ is $-|k|^2$.
# - The $n$-th iterated Laplacian eigenvalue is $(-1)^n |k|^{2n}$.
# - For dissipation we need the RHS contribution to be $-\nu |k|^{2n} \hat{q}$.
# - So the correct sign is $(-1)^{n+1} \nu (\nabla^2)^n q$, which gives
#   eigenvalue $(-1)^{n+1} \cdot (-1)^n |k|^{2n} = -|k|^{2n}$ (always negative).

# %%
def qg_tendency(q: jnp.ndarray, params: QGParams) -> jnp.ndarray:
    """
    Compute dq/dt for the barotropic QG equation.

    dq/dt = -J(psi, q) + (-1)^(nv+1) * nu * (nabla^2)^nv q - mu * q

    Parameters
    ----------
    q : jnp.ndarray, shape [Nx, Ny]
        Vorticity field.
    params : QGParams
        Physical parameters (nu, mu, nv).

    Returns
    -------
    rhs : jnp.ndarray, shape [Nx, Ny]
        Time tendency of vorticity.
    """
    # Step 1: Poisson inversion  q = nabla^2 psi  =>  psi = nabla^{-2} q
    psi = compute_stream_function(q)

    # Step 2: Velocity from stream function
    u, v = compute_velocities(psi)

    # Step 3: Advection (dealiased)
    rhs = -deriv.advection_scalar(u, v, q)

    # Step 4: Hyperviscosity  (-1)^(nv+1) * nu * (nabla^2)^nv q
    if params.nu > 0:
        lap_q = q
        for _ in range(params.nv):
            lap_q = deriv.laplacian(lap_q)
        rhs = rhs + (-1) ** (params.nv + 1) * params.nu * lap_q

    # Step 5: Linear drag  -mu * q
    if params.mu > 0:
        rhs = rhs - params.mu * q

    return rhs


# %% [markdown]
# ## Time Integration
#
# We use the classical 4th-order Runge--Kutta method (RK4). Each full time
# step requires **four** evaluations of the right-hand side.
#
# The integration loop is compiled with `jax.lax.fori_loop` inside a
# JIT-compiled function, so the entire inner loop runs as a single fused
# XLA computation without Python overhead.
#
# ```
# RK4 scheme
# ==========
# k1 = f(q_n)
# k2 = f(q_n + dt/2 * k1)
# k3 = f(q_n + dt/2 * k2)
# k4 = f(q_n + dt * k3)
# q_{n+1} = q_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
# ```

# %%
def rk4_step(q: jnp.ndarray, dt: float, params: QGParams) -> jnp.ndarray:
    """Advance vorticity by one RK4 time step."""
    k1 = qg_tendency(q, params)
    k2 = qg_tendency(q + 0.5 * dt * k1, params)
    k3 = qg_tendency(q + 0.5 * dt * k2, params)
    k4 = qg_tendency(q + dt * k3, params)
    return q + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@eqx.filter_jit
def integrate_steps(
    q: jnp.ndarray, dt: float, n_steps: int, params: QGParams
) -> jnp.ndarray:
    """
    Integrate for n_steps using RK4, compiled as a single fused XLA loop.

    Parameters
    ----------
    q : jnp.ndarray, shape [Nx, Ny]
        Initial vorticity.
    dt : float
        Time step size.
    n_steps : int
        Number of RK4 steps to take.
    params : QGParams
        Physical parameters.

    Returns
    -------
    q : jnp.ndarray, shape [Nx, Ny]
        Vorticity after n_steps time steps.
    """
    def body_fn(i, q):
        return rk4_step(q, dt, params)

    return jax.lax.fori_loop(0, n_steps, body_fn, q)


# %% [markdown]
# ## Run Simulation
#
# We integrate from $t = 0$ to $t = 50$ with $\Delta t = 0.005$, giving
# 10,000 total RK4 steps. Snapshots are saved every 100 steps for diagnostics.

# %%
# Time stepping parameters
dt = 0.005
t_final = 50.0
n_steps_per_output = 100
n_outputs = int(t_final / (dt * n_steps_per_output))

print(f"Time step:          dt = {dt}")
print(f"Final time:         t_final = {t_final}")
print(f"Total RK4 steps:    {int(t_final / dt)}")
print(f"Snapshot interval:  every {n_steps_per_output} steps ({n_steps_per_output * dt:.1f} time units)")
print(f"Number of outputs:  {n_outputs}")

# %%
# Storage for time series
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
        E = float(compute_energy(q))
        Z = float(compute_enstrophy(q))
        print(f"  t = {t:5.1f}  |  max|q| = {float(jnp.abs(q).max()):8.3f}  |  E = {E:.6f}  |  Z = {Z:.6f}")

print("\nSimulation complete!")
print(f"Final vorticity: shape = {q.shape}, range = [{float(q.min()):.3f}, {float(q.max()):.3f}]")

# %% [markdown]
# ## Vorticity Evolution
#
# Four snapshots show the hallmarks of 2D turbulence:
#
# - **Early times**: The initial random field develops sharp vorticity gradients
#   and thin filaments (forward enstrophy cascade).
# - **Intermediate times**: Coherent vortices emerge as like-signed vorticity
#   patches merge (inverse energy cascade).
# - **Late times**: A few large, well-separated vortices dominate the domain,
#   connected by thin filamentary structures.

# %%
# Select four times spanning the simulation
plot_indices = [0, len(times) // 4, len(times) // 2, -1]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

vmax = max(float(jnp.abs(q_history[i]).max()) for i in plot_indices)

for ax, idx in zip(axes, plot_indices):
    q_plot = q_history[idx]
    t_plot = times[idx]

    im = ax.pcolormesh(
        X.T, Y.T, q_plot.T, cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, shading="auto"
    )
    ax.set_title(f"$t = {t_plot:.1f}$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax)

plt.suptitle("Vorticity Evolution — Decaying QG Turbulence", fontsize=14)
plt.tight_layout()
fig.savefig(IMG_DIR / "vorticity_evolution.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Vorticity evolution at four time snapshots](../images/demo_qg/vorticity_evolution.png)

# %% [markdown]
# ## Final State with Velocity Vectors
#
# The final state shows large coherent vortices (cyclones and anticyclones)
# with the velocity field (arrows) swirling around each vortex center. The
# stream function contours are closed around each vortex, confirming
# geostrophic balance.

# %%
q_final = q_history[-1]
psi_final = compute_stream_function(q_final)
u_final, v_final = compute_velocities(psi_final)

print(f"Final vorticity:       range = [{float(q_final.min()):.3f}, {float(q_final.max()):.3f}]")
print(f"Final stream function: range = [{float(psi_final.min()):.3f}, {float(psi_final.max()):.3f}]")
print(f"Final max speed:       {float(jnp.sqrt(u_final**2 + v_final**2).max()):.3f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subsample for quiver plot (every 8th point)
skip = max(1, Nx // 32)

# Vorticity with velocity vectors
im = axes[0].pcolormesh(X.T, Y.T, q_final.T, cmap="RdBu_r", shading="auto")
axes[0].quiver(
    X[::skip, ::skip].T, Y[::skip, ::skip].T,
    u_final[::skip, ::skip].T, v_final[::skip, ::skip].T,
    color="black", alpha=0.7,
)
axes[0].set_title(f"Vorticity & velocity ($t = {times[-1]:.1f}$)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
plt.colorbar(im, ax=axes[0])

# Stream function with velocity vectors
im = axes[1].pcolormesh(X.T, Y.T, psi_final.T, cmap="viridis", shading="auto")
axes[1].quiver(
    X[::skip, ::skip].T, Y[::skip, ::skip].T,
    u_final[::skip, ::skip].T, v_final[::skip, ::skip].T,
    color="white", alpha=0.7,
)
axes[1].set_title(f"Stream function & velocity ($t = {times[-1]:.1f}$)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
fig.savefig(IMG_DIR / "final_state.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Final state: vorticity and stream function with velocity vectors](../images/demo_qg/final_state.png)

# %% [markdown]
# ## Conservation Diagnostics
#
# In ideal (inviscid) 2D turbulence, both energy and enstrophy are conserved.
# With Laplacian viscosity:
#
# - **Energy** decays slowly because it resides at large scales where
#   viscous dissipation ($\propto \nu k^2$) is weak.
# - **Enstrophy** decays much faster because it cascades to small scales
#   where viscous dissipation is strong. This selective enstrophy decay
#   is the signature of the forward enstrophy cascade.
#
# The contrast between slow energy decay and rapid enstrophy decay is a
# key diagnostic of 2D turbulence.

# %%
energies = [float(compute_energy(q)) for q in q_history]
enstrophies = [float(compute_enstrophy(q)) for q in q_history]

print(f"Energy:    initial = {energies[0]:.6f},  final = {energies[-1]:.6f}")
print(f"Enstrophy: initial = {enstrophies[0]:.6f},  final = {enstrophies[-1]:.6f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Energy
axes[0].plot(times, energies, "b-", linewidth=2)
axes[0].axhline(energies[0], color="gray", linestyle="--", alpha=0.5, label="Initial")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Energy $E$")
axes[0].set_title("Kinetic Energy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Enstrophy
axes[1].plot(times, enstrophies, "r-", linewidth=2)
axes[1].axhline(enstrophies[0], color="gray", linestyle="--", alpha=0.5, label="Initial")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Enstrophy $Z$")
axes[1].set_title("Enstrophy (forward cascade $\\rightarrow$ dissipation)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(IMG_DIR / "diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Energy and enstrophy diagnostics](../images/demo_qg/diagnostics.png)

# %%
energy_change = 100 * (energies[-1] / energies[0] - 1)
enstrophy_change = 100 * (enstrophies[-1] / enstrophies[0] - 1)

print(f"Energy change:    {energy_change:+.2f}%  (slow decay — inverse cascade protects large scales)")
print(f"Enstrophy change: {enstrophy_change:+.2f}%  (fast decay — forward cascade feeds dissipation)")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated a complete pseudospectral simulation of decaying
# 2D barotropic quasi-geostrophic turbulence using `spectraldiffx`:
#
# 1. **Grid setup** with `FourierGrid2D.from_N_L()` — doubly-periodic domain
#    with 2/3 dealiasing.
# 2. **Spectral derivatives** with `SpectralDerivative2D`:
#    - `gradient()` to compute velocity from the stream function.
#    - `advection_scalar()` for the dealiased nonlinear advection term.
#    - `laplacian()` for viscous diffusion.
# 3. **Poisson inversion** with `SpectralHelmholtzSolver2D.solve()` to recover
#    $\psi$ from $q = \nabla^2 \psi$.
# 4. **RK4 time integration** via `jax.lax.fori_loop` for efficient compiled
#    time stepping.
# 5. **Dual cascade diagnostics**: energy decays slowly (inverse cascade to
#    large scales), enstrophy decays rapidly (forward cascade to small scales).
#
# **Key references:**
#
# - Charney, J.G. (1971). *Geostrophic turbulence.* J. Atmos. Sci., 28, 1087--1095.
# - McWilliams, J.C. (1984). *The emergence of isolated coherent vortices in
#   turbulent flow.* J. Fluid Mech., 146, 21--43.
# - Kraichnan, R.H. (1967). *Inertial ranges in two-dimensional turbulence.*
#   Phys. Fluids, 10, 1417--1423.
