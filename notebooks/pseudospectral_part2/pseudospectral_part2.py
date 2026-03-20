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
# # QG Turbulence with Adaptive Time Stepping (Diffrax)
#
# ## Introduction
#
# This notebook builds on the `demo_qg` tutorial, where we simulated decaying
# 2D barotropic quasi-geostrophic turbulence with fixed-step RK4. Here we
# replace the fixed time stepper with **adaptive ODE integration** via the
# [diffrax](https://docs.kidger.site/diffrax/) library.
#
# **Why adaptive time stepping?**
#
# With a fixed time step, we must choose $\Delta t$ small enough to satisfy the
# CFL condition during the *most demanding* phase of the simulation. But the
# flow intensity varies over time: early on, strong vortex interactions produce
# large velocities and sharp gradients, while later the flow becomes smoother
# as vortices merge. A fixed step wastes computation during the quiescent
# phases.
#
# Adaptive methods (like Tsitouras 5th-order, `Tsit5`) estimate the local
# truncation error at each step and adjust $\Delta t$ automatically:
#
# - **Small steps** when the flow is rapidly changing (early turbulent phase).
# - **Large steps** when the flow is smooth (late coherent-vortex phase).
#
# This can dramatically reduce the total number of RHS evaluations while
# maintaining accuracy.
#
# **What this notebook demonstrates:**
#
# - `spectraldiffx` operators: `FourierGrid2D`, `SpectralDerivative2D`,
#   `SpectralHelmholtzSolver2D`
# - McWilliams (1984) initial conditions for realistic decaying turbulence
# - High-order hyperviscosity ($\nabla^8$) enabled by adaptive stepping
# - `diffrax` integration: `ODETerm`, `Tsit5`, `PIDController`, `SaveAt`
# - Energy and enstrophy conservation diagnostics

# %% [markdown]
# ## Governing Equations
#
# The barotropic QG vorticity equation (see `demo_qg` for full derivation):
#
# $$
# \frac{\partial q}{\partial t} + J(\psi, q) = (-1)^{n_v+1} \, \nu \, (\nabla^2)^{n_v} q - \mu \, q
# $$
#
# with $q = \nabla^2 \psi$ (vorticity--stream function relation) and velocity
# $u = -\partial_y \psi$, $v = \partial_x \psi$.
#
# The key difference from `demo_qg` is the **viscosity order**: we use
# $n_v = 4$ (i.e., $\nabla^8$ hyperviscosity) with a very small coefficient
# $\nu = 10^{-12}$. This concentrates dissipation at the grid scale and
# leaves the resolved dynamics essentially inviscid.
#
# With fixed-step RK4, the explicit stability constraint
# $\Delta t \cdot \nu \cdot k_{\max}^{2 n_v} < 2.8$ would require impossibly
# small time steps for $n_v = 4$. Adaptive stepping sidesteps this: diffrax
# detects the stiffness and reduces $\Delta t$ only when needed.

# %% [markdown]
# ## McWilliams (1984) Initial Condition
#
# McWilliams (1984) introduced a canonical initial condition for decaying 2D
# turbulence simulations. The procedure constructs a random vorticity field
# with a prescribed energy spectrum peaked at intermediate wavenumbers:
#
# 1. **Spectral amplitude**: For each wavenumber $|\mathbf{k}|$, set the
#    amplitude of the stream function as:
#    $$
#    |\hat{\psi}(k)| \propto \frac{1}{k \left(1 + (k^2/k_0^2)^2\right)^{1/2}}
#    $$
#    where $k_0 = 6$ is the peak wavenumber. This gives an energy spectrum
#    $E(k) \propto k \, |\hat{\psi}|^2$ that is broadband with a peak near $k_0$.
#
# 2. **Random phases**: Multiply each spectral coefficient by a random complex
#    phase (uniform on the unit circle).
#
# 3. **Compute vorticity**: $\hat{q} = -|k|^2 \hat{\psi}$ (Poisson relation
#    in spectral space).
#
# 4. **Normalize**: Scale to unit energy.
#
# The result is a field with many small vortices that will merge into fewer,
# larger vortices over time — the hallmark of the inverse energy cascade.

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

IMG_DIR = (
    Path(__file__).resolve().parent.parent / "docs" / "images" / "pseudospectral_part2"
)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %%
from spectraldiffx import (
    FourierGrid2D,
    SpectralDerivative2D,
    SpectralHelmholtzSolver2D,
)

# %% [markdown]
# ## Domain and Parameters
#
# | Parameter | Value    | Physical meaning |
# |-----------|----------|-----------------|
# | `Nx, Ny`  | 128      | Grid resolution. $k_{\max} = 64$. |
# | `Lx, Ly`  | $2\pi$   | Domain size. Wavenumber spacing $\Delta k = 1$. |
# | `nu`      | 1e-12    | $\nabla^8$ hyperviscosity coefficient (very small). |
# | `nv`      | 4        | Viscosity order: $(\nabla^2)^4 = \nabla^8$. |
# | `mu`      | 0.0      | No Ekman drag (free decay). |
# | `dt0`     | 0.01     | Initial step size hint for diffrax. |
# | `t_end`   | 20.0     | Integration horizon. |
# | `dealias` | 2/3      | Standard 2/3 dealiasing rule. |
#
# The $\nabla^8$ operator has spectral eigenvalue $|k|^8$. At $k_{\max} = 64$:
#
# $$
# \nu \cdot k_{\max}^{2 n_v} = 10^{-12} \times 64^8 \approx 0.28
# $$
#
# This is large enough to dissipate enstrophy at the grid scale while being
# negligible at resolved wavenumbers ($k \lesssim k_{\max}/3$). The adaptive
# integrator handles the resulting stiffness automatically.

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

print(f"Grid:       {Nx} x {Ny}")
print(f"Domain:     [0, {Lx:.4f}] x [0, {Ly:.4f}]")
print(f"Resolution: dx = {grid.dx:.6f}, dy = {grid.dy:.6f}")
print(f"k_max:      {Nx // 2}")
print(f"X shape:    {X.shape}")
print(f"Y shape:    {Y.shape}")

# %%
class QGParams(NamedTuple):
    """Parameters for the QG simulation with high-order hyperviscosity."""
    nu: float = 1e-8   # hyperviscosity coefficient
    mu: float = 0.0    # linear drag (Ekman friction)
    nv: int = 2        # viscosity order: (nabla^2)^nv


params = QGParams(nu=1e-12, mu=0.0, nv=4)

print(f"Viscosity:       nu = {params.nu}")
print(f"Viscosity order: nv = {params.nv}  (nabla^{2 * params.nv})")
print(f"Ekman drag:      mu = {params.mu}")

# Effective dissipation at k_max
k_max = Nx // 2
dissipation_kmax = params.nu * k_max ** (2 * params.nv)
print(f"\nnu * k_max^(2*nv) = {dissipation_kmax:.4f}")
print(f"  (dissipation rate at the grid scale)")

# %% [markdown]
# ## McWilliams (1984) Initial Condition — Implementation

# %%
def mcwilliams_ic(grid: FourierGrid2D, seed: int = 42) -> jnp.ndarray:
    """
    Generate McWilliams (1984) initial condition for decaying 2D turbulence.

    The energy spectrum is shaped by:
        |psi_hat(k)| ~ 1 / (k * sqrt(1 + (k^2/36)^2))

    This peaks near k ~ 6 and decays at both large and small scales,
    providing broadband initial turbulence.

    Parameters
    ----------
    grid : FourierGrid2D
        Spectral grid providing wavenumber arrays.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    q : jnp.ndarray, shape [Nx, Ny]
        Vorticity field normalized to unit energy.
    """
    rng = np.random.RandomState(seed)

    # Wavenumber magnitudes |k| = sqrt(kx^2 + ky^2)
    K2 = grid.K2
    wv = np.sqrt(np.array(K2))
    print(f"  Wavenumber grid: shape = {wv.shape}, max|k| = {wv.max():.1f}")

    # Spectral amplitude of stream function:
    #   |psi_hat(k)| ~ 1 / (k * sqrt(1 + (k^2/36)^2))
    # This gives energy spectrum E(k) ~ k |psi_hat|^2 peaked near k = 6
    fk = wv != 0
    ckappa = np.zeros_like(wv)
    ckappa[fk] = np.sqrt(wv[fk] ** 2 * (1.0 + (wv[fk] ** 2 / 36.0) ** 2)) ** (-1)
    print(f"  Spectral envelope: max amplitude = {ckappa.max():.4f}")

    # Random stream function in spectral space (random phase + prescribed amplitude)
    psi_hat = (rng.randn(Nx, Ny) + 1j * rng.randn(Nx, Ny)) * ckappa

    # Transform to physical space and remove mean
    psi = np.fft.ifft2(psi_hat).real
    psi = psi - psi.mean()
    print(f"  Stream function: shape = {psi.shape}, range = [{psi.min():.4f}, {psi.max():.4f}]")

    # Compute vorticity: q = nabla^2 psi  =>  q_hat = -|k|^2 * psi_hat
    psi_hat = np.fft.fft2(psi)
    q_hat = -K2 * psi_hat
    q = np.fft.ifft2(q_hat).real
    print(f"  Vorticity:       shape = {q.shape}, range = [{q.min():.4f}, {q.max():.4f}]")

    # Normalize to unit energy: E = sum|q_hat|^2 / N^2 -> 1
    q_hat = np.fft.fft2(q)
    energy = np.sum(np.abs(q_hat) ** 2) / (Nx * Ny) ** 2
    q = q / np.sqrt(energy)
    print(f"  After normalization: energy proxy = {np.sum(np.abs(np.fft.fft2(q))**2) / (Nx*Ny)**2:.4f}")

    return jnp.asarray(q)


q0 = mcwilliams_ic(grid, seed=42)

# %% [markdown]
# ## Diagnostic Functions
#
# Same diagnostics as `demo_qg`:
#
# - **Stream function** $\psi$: Poisson inversion of $q$.
# - **Velocity** $(u, v)$: from $\psi$ via $u = -\partial_y \psi$, $v = \partial_x \psi$.
# - **Energy**: $E = \tfrac{1}{2}\langle u^2 + v^2 \rangle$.
# - **Enstrophy**: $Z = \tfrac{1}{2}\langle q^2 \rangle$.

# %%
def compute_stream_function(q: jnp.ndarray) -> jnp.ndarray:
    """Solve nabla^2 psi = q for the stream function."""
    return solver.solve(q, alpha=0.0, zero_mean=True)


def compute_velocities(psi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Velocity from stream function: u = -dpsi/dy, v = dpsi/dx."""
    dpsi_dx, dpsi_dy = deriv.gradient(psi)
    return -dpsi_dy, dpsi_dx


def compute_energy(q: jnp.ndarray) -> float:
    """Kinetic energy: E = (1/2) <u^2 + v^2>."""
    psi = compute_stream_function(q)
    u, v = compute_velocities(psi)
    return 0.5 * jnp.mean(u**2 + v**2)


def compute_enstrophy(q: jnp.ndarray) -> float:
    """Enstrophy: Z = (1/2) <q^2>."""
    return 0.5 * jnp.mean(q**2)


# %% [markdown]
# ## Visualize Initial State
#
# The McWilliams initial condition produces broadband vorticity with many
# small-scale structures. The stream function is smoother (Poisson inversion
# suppresses high wavenumbers). The speed field $|\mathbf{u}|$ shows where
# the strongest currents are.

# %%
psi0 = compute_stream_function(q0)
u0, v0 = compute_velocities(psi0)
speed0 = jnp.sqrt(u0**2 + v0**2)

print(f"Initial vorticity:       shape = {q0.shape},     range = [{float(q0.min()):.3f}, {float(q0.max()):.3f}]")
print(f"Initial stream function: shape = {psi0.shape},   range = [{float(psi0.min()):.3f}, {float(psi0.max()):.3f}]")
print(f"Initial speed:           shape = {speed0.shape}, max   = {float(speed0.max()):.3f}")
print(f"\nInitial energy:    E0 = {float(compute_energy(q0)):.6f}")
print(f"Initial enstrophy: Z0 = {float(compute_enstrophy(q0)):.6f}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

im = axes[0].pcolormesh(X.T, Y.T, q0.T, cmap="RdBu_r", shading="auto")
axes[0].set_title("Vorticity $q_0$")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
plt.colorbar(im, ax=axes[0])

im = axes[1].pcolormesh(X.T, Y.T, psi0.T, cmap="viridis", shading="auto")
axes[1].set_title(r"Stream function $\psi_0$")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
plt.colorbar(im, ax=axes[1])

im = axes[2].pcolormesh(X.T, Y.T, speed0.T, cmap="magma", shading="auto")
axes[2].set_title("Speed $|\\mathbf{u}_0|$")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
plt.colorbar(im, ax=axes[2])

plt.suptitle("McWilliams (1984) Initial Condition", fontsize=14)
plt.tight_layout()
fig.savefig(IMG_DIR / "initial_state.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Initial state: vorticity, stream function, and speed](../images/pseudospectral_part2/initial_state.png)

# %% [markdown]
# ## RHS Implementation with Hyperviscosity
#
# The right-hand side is the same as in `demo_qg`, but with $n_v = 4$
# ($\nabla^8$ hyperviscosity):
#
# $$
# \frac{\partial q}{\partial t} = -J(\psi, q)
#   + \underbrace{(-1)^{5} \cdot \nu \cdot (\nabla^2)^4 q}_{= -\nu \nabla^8 q}
# $$
#
# The sign $(-1)^{n_v+1} = (-1)^5 = -1$ ensures dissipation: the eigenvalue
# of $(\nabla^2)^4$ is $|k|^8 > 0$, so the contribution $-\nu |k|^8 \hat{q}$
# damps all modes.

# %%
def qg_tendency(q: jnp.ndarray, params: QGParams) -> jnp.ndarray:
    """
    QG tendency with high-order hyperviscosity.

    dq/dt = -J(psi, q) + (-1)^(nv+1) * nu * (nabla^2)^nv q - mu * q

    Parameters
    ----------
    q : jnp.ndarray, shape [Nx, Ny]
        Vorticity field.
    params : QGParams
        Physical parameters.

    Returns
    -------
    rhs : jnp.ndarray, shape [Nx, Ny]
        Time tendency.
    """
    # Poisson inversion: psi from q
    psi = compute_stream_function(q)

    # Velocity: u = -dpsi/dy, v = dpsi/dx
    u, v = compute_velocities(psi)

    # Advection (dealiased): -J(psi, q) = -(u dq/dx + v dq/dy)
    rhs = -deriv.advection_scalar(u, v, q)

    # Hyperviscosity: (-1)^(nv+1) * nu * (nabla^2)^nv q
    if params.nu > 0:
        lap_q = q
        for _ in range(params.nv):
            lap_q = deriv.laplacian(lap_q)
        rhs = rhs + (-1) ** (params.nv + 1) * params.nu * lap_q

    # Linear drag
    if params.mu > 0:
        rhs = rhs - params.mu * q

    return rhs


# %% [markdown]
# ## Diffrax Integration
#
# We use [diffrax](https://docs.kidger.site/diffrax/) for adaptive ODE
# integration. The key components are:
#
# - **`ODETerm`**: Wraps the RHS function into a diffrax term.
# - **`Tsit5()`**: Tsitouras 5th-order Runge--Kutta method (similar to
#   Dormand--Prince but with optimized coefficients).
# - **`PIDController`**: Adaptive step size controller using a PID
#   (proportional-integral-derivative) algorithm to adjust $\Delta t$ based
#   on the estimated local error.
# - **`SaveAt`**: Specifies which times to save the solution at (without
#   affecting the internal time stepping).
#
# ```
# Diffrax integration pipeline
# =============================
#
#   ODETerm(vector_field)     wraps dq/dt = f(t, q, params)
#         |
#         v
#   Tsit5()                   5th-order adaptive RK method
#         |
#         v
#   PIDController(rtol, atol) adjusts dt to keep error < tol
#         |
#         v
#   diffeqsolve(...)          drives the integration, saves at SaveAt times
#         |
#         v
#   sol.ys.q                  solution snapshots [n_saves, Nx, Ny]
# ```
#
# If diffrax is not available, we fall back to a simple fixed-step RK4.

# %%
try:
    import diffrax as dfx

    HAS_DIFFRAX = True
    print(f"diffrax version: {dfx.__version__}")
except ImportError:
    print("diffrax not installed. Will use fallback RK4 integration.")
    HAS_DIFFRAX = False

# %%
# Time parameters
t0, t_end = 0.0, 20.0
dt0 = 0.01
n_saves = 50

print(f"Integration interval: [{t0}, {t_end}]")
print(f"Initial step size:    dt0 = {dt0}")
print(f"Number of snapshots:  {n_saves}")

# %%
if HAS_DIFFRAX:
    # -- Diffrax integration --

    class State(NamedTuple):
        """ODE state vector (just vorticity)."""
        q: jnp.ndarray

    def vector_field(t, state: State, args) -> State:
        """Diffrax vector field: dq/dt = qg_tendency(q, params)."""
        params = args
        rhs = qg_tendency(state.q, params)
        return State(q=rhs)

    def integrate_diffrax(q0, t0, t1, dt0, params, saveat):
        """
        Integrate the QG equation using diffrax adaptive stepping.

        Parameters
        ----------
        q0 : jnp.ndarray, shape [Nx, Ny]
            Initial vorticity.
        t0, t1 : float
            Start and end times.
        dt0 : float
            Initial step size (hint for the adaptive controller).
        params : QGParams
            Physical parameters.
        saveat : dfx.SaveAt
            Times at which to save the solution.

        Returns
        -------
        sol : diffrax Solution object
            Contains sol.ts (save times) and sol.ys.q (vorticity snapshots).
        """
        ode_term = dfx.ODETerm(vector_field)
        ode_solver = dfx.Tsit5()
        controller = dfx.PIDController(rtol=1e-5, atol=1e-5)

        sol = dfx.diffeqsolve(
            terms=ode_term,
            solver=ode_solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=State(q=q0),
            saveat=saveat,
            args=params,
            stepsize_controller=controller,
            max_steps=50000,
        )
        return sol

    # Run the integration
    t_save = np.linspace(t0, t_end, n_saves)
    saveat = dfx.SaveAt(ts=t_save)

    print(f"Integrating with diffrax (Tsit5 + PID controller)...")
    sol = integrate_diffrax(q0, t0, t_end, dt0, params, saveat=saveat)

    times = sol.ts
    q_history = sol.ys.q
    print(f"Done! Saved {len(times)} snapshots.")
    print(f"  q_history shape: {q_history.shape}")
    print(f"  Time range: [{float(times[0]):.2f}, {float(times[-1]):.2f}]")

else:
    # -- Fallback: fixed-step RK4 --
    import equinox as eqx

    print("Using fallback RK4 integration (no diffrax)...")

    def rk4_step(q, dt, params):
        """Single RK4 time step."""
        k1 = qg_tendency(q, params)
        k2 = qg_tendency(q + 0.5 * dt * k1, params)
        k3 = qg_tendency(q + 0.5 * dt * k2, params)
        k4 = qg_tendency(q + dt * k3, params)
        return q + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    dt_fixed = 0.01
    n_steps = int(t_end / dt_fixed)
    save_every = max(1, n_steps // n_saves)

    times_list = [0.0]
    q_list = [q0]
    q = q0

    for i in range(n_steps):
        q = rk4_step(q, dt_fixed, params)
        if (i + 1) % save_every == 0:
            times_list.append((i + 1) * dt_fixed)
            q_list.append(q)
            if len(times_list) % 10 == 0:
                print(f"  t = {times_list[-1]:.1f}")

    times = jnp.array(times_list)
    q_history = jnp.stack(q_list)

    print(f"Done! Saved {len(times)} snapshots.")
    print(f"  q_history shape: {q_history.shape}")

# %% [markdown]
# ## Vorticity Evolution
#
# Four snapshots show the emergence of coherent vortices from the initially
# random McWilliams field. The $\nabla^8$ hyperviscosity keeps the small-scale
# filaments sharp (less artificial smoothing than Laplacian viscosity), while
# the inverse cascade drives vortex merging at large scales.

# %%
n_plots = 4
indices = np.linspace(0, len(times) - 1, n_plots, dtype=int)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

vmax = float(jnp.max(jnp.abs(q_history)))

for ax, idx in zip(axes, indices):
    q_plot = q_history[idx]
    t_plot = times[idx]

    im = ax.pcolormesh(
        X.T, Y.T, q_plot.T, cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, shading="auto",
    )
    ax.set_title(f"$t = {float(t_plot):.1f}$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax)

plt.suptitle("Vorticity Evolution (McWilliams 1984 IC, $\\nabla^8$ hyperviscosity)", fontsize=12)
plt.tight_layout()
fig.savefig(IMG_DIR / "vorticity_evolution.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Vorticity evolution at four time snapshots](../images/pseudospectral_part2/vorticity_evolution.png)

# %% [markdown]
# ## Final State with Velocity Vectors
#
# The final state shows coherent vortices with velocity vectors indicating
# the flow structure. Compared to the Laplacian viscosity case in `demo_qg`,
# the filaments are thinner and sharper thanks to the selective $\nabla^8$
# dissipation.

# %%
q_final = q_history[-1]
psi_final = compute_stream_function(q_final)
u_final, v_final = compute_velocities(psi_final)

print(f"Final vorticity:       range = [{float(q_final.min()):.3f}, {float(q_final.max()):.3f}]")
print(f"Final stream function: range = [{float(psi_final.min()):.3f}, {float(psi_final.max()):.3f}]")
print(f"Final max speed:       {float(jnp.sqrt(u_final**2 + v_final**2).max()):.3f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Subsample for quiver plot
skip = max(1, Nx // 24)

# Vorticity with velocity vectors
im = axes[0].pcolormesh(X.T, Y.T, q_final.T, cmap="RdBu_r", shading="auto")
axes[0].quiver(
    X[::skip, ::skip].T, Y[::skip, ::skip].T,
    u_final[::skip, ::skip].T, v_final[::skip, ::skip].T,
    color="black", alpha=0.6, scale=30,
)
axes[0].set_title(f"Vorticity & velocity ($t = {float(times[-1]):.1f}$)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
plt.colorbar(im, ax=axes[0])

# Stream function with contours
im = axes[1].pcolormesh(X.T, Y.T, psi_final.T, cmap="viridis", shading="auto")
axes[1].contour(X.T, Y.T, psi_final.T, colors="white", alpha=0.5, linewidths=0.5)
axes[1].set_title(f"Stream function ($t = {float(times[-1]):.1f}$)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
fig.savefig(IMG_DIR / "final_state.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Final state: vorticity and stream function with velocity vectors](../images/pseudospectral_part2/final_state.png)

# %% [markdown]
# ## Conservation Diagnostics
#
# With $\nabla^8$ hyperviscosity ($\nu = 10^{-12}$, $n_v = 4$):
#
# - **Energy** decays by approximately 8%. The inverse cascade transfers
#   energy to large scales where dissipation is negligible, so most energy
#   is preserved. The small decay comes from the energy in small-scale
#   filaments that are dissipated.
#
# - **Enstrophy** decays by approximately 58%. The forward enstrophy cascade
#   efficiently transports enstrophy to the grid scale where hyperviscosity
#   removes it. This large decay — contrasted with the small energy decay —
#   is the hallmark of 2D turbulence.

# %%
energies = [float(compute_energy(q_history[i])) for i in range(len(times))]
enstrophies = [float(compute_enstrophy(q_history[i])) for i in range(len(times))]

E0, E_final = energies[0], energies[-1]
Z0, Z_final = enstrophies[0], enstrophies[-1]

print(f"Energy:    E0 = {E0:.6f},  E_final = {E_final:.6f},  change = {100*(E_final/E0 - 1):+.2f}%")
print(f"Enstrophy: Z0 = {Z0:.6f},  Z_final = {Z_final:.6f},  change = {100*(Z_final/Z0 - 1):+.2f}%")

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
# ![Energy and enstrophy diagnostics](../images/pseudospectral_part2/diagnostics.png)

# %%
energy_change = 100 * (E_final / E0 - 1)
enstrophy_change = 100 * (Z_final / Z0 - 1)

print(f"Energy change:    {energy_change:+.2f}%  (slow decay — large scales protected by inverse cascade)")
print(f"Enstrophy change: {enstrophy_change:+.2f}%  (fast decay — forward cascade feeds grid-scale dissipation)")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated adaptive time stepping for QG turbulence using
# `spectraldiffx` and `diffrax`:
#
# 1. **McWilliams (1984) initial conditions** — broadband random vorticity with
#    controlled spectral shape, the standard benchmark for decaying 2D turbulence.
#
# 2. **High-order hyperviscosity** ($\nabla^8$, $\nu = 10^{-12}$) — concentrates
#    dissipation at the grid scale while leaving resolved dynamics essentially
#    inviscid. This is only feasible with adaptive stepping; fixed-step RK4
#    would violate the explicit stability constraint.
#
# 3. **Diffrax adaptive integration** — `Tsit5` (5th-order Runge--Kutta) with
#    `PIDController` for automatic step size control. The solver takes small
#    steps during the initial turbulent adjustment and large steps once the
#    flow becomes smoother.
#
# 4. **Conservation diagnostics** — energy decays slowly (~8%), enstrophy
#    decays rapidly (~58%), confirming the dual cascade of 2D turbulence.
#
# **Key references:**
#
# - McWilliams, J.C. (1984). *The emergence of isolated coherent vortices in
#   turbulent flow.* J. Fluid Mech., 146, 21--43.
# - Kidger, P. (2022). *On Neural Differential Equations.* PhD thesis.
#   (diffrax documentation: https://docs.kidger.site/diffrax/)
