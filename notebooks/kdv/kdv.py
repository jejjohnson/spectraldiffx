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
# # The Korteweg-de Vries Equation: Soliton Interactions
#
# ## Introduction
#
# The Korteweg-de Vries (KdV) equation is the prototypical nonlinear dispersive
# wave equation. First derived by Boussinesq (1877) and later rediscovered by
# Korteweg and de Vries (1895), it models shallow water waves in a channel where
# nonlinear steepening and linear dispersion are in balance.
#
# The equation is historically significant for several reasons:
#
# - **Solitons**: Zabusky and Kruskal (1965) discovered numerically that KdV
#   solutions behave like particles — localized waves that maintain their shape
#   and survive collisions elastically. They coined the term "soliton."
# - **Inverse Scattering Transform**: Gardner, Greene, Kruskal, and Miura (1967)
#   showed that the KdV equation can be solved exactly via a nonlinear analogue of
#   the Fourier transform, opening an entirely new branch of mathematical physics.
# - **Integrability**: The KdV equation possesses infinitely many conserved
#   quantities and is the first example of a completely integrable infinite-dimensional
#   Hamiltonian system.
#
# In this notebook, we simulate the interaction of two solitons using a
# pseudospectral method from `spectraldiffx`. We will see the hallmark property
# of solitons: after collision, both waves emerge with their original shapes and
# speeds, shifted only in phase.
#
# **spectraldiffx features used**:
# - `FourierGrid1D` for the periodic spectral grid and FFT transforms
# - `SpectralDerivative1D` for computing spatial derivatives in Fourier space
# - Dealiasing via the 2/3 rule to handle quadratic nonlinearity

# %% [markdown]
# ## Mathematical Background
#
# ### The KdV Equation
#
# The canonical form of the KdV equation is:
#
# $$
# \frac{\partial u}{\partial t} + 6u\frac{\partial u}{\partial x}
# + \frac{\partial^3 u}{\partial x^3} = 0
# $$
#
# Each term has a distinct physical role:
#
# | Term | Name | Physical Effect |
# |------|------|-----------------|
# | $\partial_t u$ | Time evolution | Rate of change of the wave profile |
# | $6u \, \partial_x u$ | Nonlinear steepening | Amplitude-dependent wave speed; tall parts travel faster, tending to form shocks |
# | $\partial_x^3 u$ | Linear dispersion | Different wavelengths travel at different speeds, spreading wave packets |
#
# The key insight is that these two effects — steepening and dispersion — can
# exactly balance each other, producing permanent-form traveling waves called
# **solitons**.
#
# ### Deriving the Right-Hand Side
#
# Rearranging for time integration, we write the equation as an ODE in time:
#
# $$
# \frac{\partial u}{\partial t} = -6u\frac{\partial u}{\partial x}
# - \frac{\partial^3 u}{\partial x^3}
# $$
#
# This splits naturally into two parts:
#
# 1. **Nonlinear advection**: $-6u \, \partial_x u$
# 2. **Linear dispersion**: $-\partial_x^3 u$
#
# ### Dispersion Relation
#
# Linearizing around $u = 0$ and substituting a plane wave $u \sim e^{i(kx - \omega t)}$
# into the linear part $\partial_t u + \partial_x^3 u = 0$ gives:
#
# $$
# \omega(k) = -k^3
# $$
#
# The phase velocity $c_p = \omega / k = -k^2$ depends on wavenumber, so short
# waves travel faster (to the left). This is **anomalous dispersion** — it
# prevents shock formation by spreading energy to the left.

# %% [markdown]
# ### Exact Soliton Solutions
#
# The KdV equation admits an exact one-soliton solution:
#
# $$
# u(x, t) = \frac{c}{2} \operatorname{sech}^2\!\left(
# \frac{\sqrt{c}}{2}(x - ct - x_0)\right)
# $$
#
# where $c > 0$ is the soliton speed and $x_0$ is the initial position.
#
# Key properties:
#
# - **Amplitude** $= c/2$: taller solitons travel faster
# - **Width** $\sim 1/\sqrt{c}$: taller solitons are narrower
# - **Speed** $= c$: proportional to amplitude
#
# This amplitude-speed relation is what makes two-soliton collisions interesting:
# a tall, fast soliton can overtake a short, slow one.
#
# ```
# Amplitude
#   ^
#   |   ___
#   |  /   \          c1 = 4 (tall, fast, narrow)
#   | /     \
#   |/       \___
#   |            \     c2 = 1 (short, slow, wide)
#   |         ____\__________________________
#   +-----------------------------------------> x
#        x1           x2
#
#   After time T, the tall soliton has traveled 4T units
#   while the short one has traveled only T units.
# ```
#
# After collision, both solitons emerge unchanged in shape and speed, but each
# acquires a **phase shift** — a spatial displacement compared to where it would
# have been without the interaction.

# %% [markdown]
# ## Pseudospectral Discretization
#
# We use a Fourier pseudospectral method on a periodic domain. The strategy is
# to handle each term where it is most natural:
#
# ```
# ┌─────────────────────────────────────────────────────────┐
# │                    KdV RHS Computation                  │
# │                                                         │
# │  Physical space          Spectral space                 │
# │  ──────────────          ──────────────                 │
# │                                                         │
# │  u(x) ──── FFT ────────> û(k)                          │
# │                            │                            │
# │                            ├──> (ik)³ û  ──> -d³u/dx³  │
# │                            │    [dispersion, exact]     │
# │                            │                            │
# │                            └──> (ik) û ──> du/dx        │
# │                                   │                     │
# │                              IFFT │                     │
# │                                   ▼                     │
# │  6u · du/dx  <──── multiply in physical space           │
# │       │              [avoids convolution cost]          │
# │       │                                                 │
# │  FFT  │                                                 │
# │       ▼                                                 │
# │  Combine in spectral space, apply dealias filter        │
# │       │                                                 │
# │  IFFT ▼                                                 │
# │  du/dt(x)   ──── return to ODE solver                  │
# └─────────────────────────────────────────────────────────┘
# ```
#
# ### Why Dealiasing?
#
# The nonlinear term $u \cdot \partial_x u$ involves a **product** of two
# functions in physical space. In spectral space, multiplication becomes
# convolution, which can fold high-frequency energy back into resolved modes
# (aliasing). The **2/3 rule** zeroes out the upper third of Fourier modes,
# ensuring that the quadratic product does not alias:
#
# - $N$ grid points resolve wavenumbers $|k| \le N/2$
# - Product of two signals with bandwidth $N/2$ has bandwidth $N$
# - Only modes with $|k| \le N/3$ are alias-free after the product
#
# `spectraldiffx` provides `grid.dealias_filter()` which returns the appropriate
# mask.

# %%
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float

matplotlib.use("Agg")

from spectraldiffx import FourierGrid1D, SpectralDerivative1D

# Enable 64-bit precision — essential for long-time soliton propagation
# where phase errors accumulate
jax.config.update("jax_enable_x64", True)

# Output directory for saved figures
IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "images" / "kdv"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Grid and Parameters
#
# We choose the following parameters with care:
#
# | Parameter | Value | Rationale |
# |-----------|-------|-----------|
# | `N` | 512 | Enough modes to resolve the narrow tall soliton ($\text{width} \sim 1/\sqrt{c_1} = 0.5$); needs $\Delta x \ll 0.5$ |
# | `L` | 100.0 | Large domain so solitons do not wrap around and re-enter during the simulation. At $c_1 = 4$, the fast soliton travels $4 \times 5 = 20$ units by $t = 5$. |
# | `c1` | 4.0 | Tall, fast soliton. Amplitude $= c_1/2 = 2.0$, width $\sim 1/\sqrt{4} = 0.5$ |
# | `c2` | 1.0 | Short, slow soliton. Amplitude $= c_2/2 = 0.5$, width $\sim 1/\sqrt{1} = 1.0$ |
# | `t_end` | 5.0 | Long enough for the fast soliton to overtake the slow one and separate again |
# | `dealias` | `"2/3"` | Standard 2/3 dealiasing rule for quadratic nonlinearity |

# %%
# --- Simulation parameters ---
N = 512           # Number of Fourier modes (grid points)
L = 100.0         # Domain length [0, L) — periodic
t_end = 5.0       # Final simulation time
dt0 = 1e-3        # Initial time step for adaptive solver
n_saves = 201     # Number of snapshots to save

# Soliton parameters
c1 = 4.0          # Speed (and 2x amplitude) of the tall soliton
c2 = 1.0          # Speed (and 2x amplitude) of the short soliton
x1_offset = -20.0 # Initial position of soliton 1 (relative to domain center)
x2_offset = -5.0  # Initial position of soliton 2 (relative to domain center)

print(f"Soliton 1: amplitude = {c1/2:.1f}, width ~ {1/jnp.sqrt(c1):.2f}, speed = {c1:.1f}")
print(f"Soliton 2: amplitude = {c2/2:.1f}, width ~ {1/jnp.sqrt(c2):.2f}, speed = {c2:.1f}")
print(f"Grid spacing: dx = L/N = {L/N:.4f}")
print(f"Points per soliton-1 width: ~{1/jnp.sqrt(c1) / (L/N):.0f}")

# %% [markdown]
# ## Setting Up the Spectral Grid
#
# `FourierGrid1D.from_N_L` creates a uniform grid on $[0, L)$ with $N$ points
# and precomputes the wavenumber array $k$ and the dealiasing filter.

# %%
grid = FourierGrid1D.from_N_L(N=N, L=L, dealias="2/3")
deriv = SpectralDerivative1D(grid)

# Shift coordinates so the domain is centered at x = 0: [-L/2, L/2)
# This makes it natural to place solitons symmetrically
x = grid.x - L / 2.0

print(f"Grid shape:          x.shape = {grid.x.shape}")
print(f"Wavenumber shape:    k.shape = {grid.k.shape}")
print(f"Domain:              [{float(x[0]):.1f}, {float(x[-1]):.1f}]")
print(f"Grid spacing:        dx = {float(x[1] - x[0]):.4f}")

# %% [markdown]
# ## State and Parameter Containers
#
# We use `equinox.Module` to bundle the simulation state and parameters into
# JAX-compatible pytree containers. This is required by `diffrax`, which
# expects the state `y` and arguments `args` to be pytrees.

# %%
class Params(eqx.Module):
    """Immutable container for simulation parameters (grid + derivative operator).

    Marked as static so JAX traces through them as constants,
    enabling efficient JIT compilation.
    """
    grid: FourierGrid1D = eqx.field(static=True)
    deriv: SpectralDerivative1D = eqx.field(static=True)


class State(eqx.Module):
    """The dynamical state: wave amplitude u(x) at a single time."""
    u: Float[Array, "N"]


params = Params(grid=grid, deriv=deriv)
print(f"Params fields: {[f.name for f in params.__dataclass_fields__.values()]}")

# %% [markdown]
# ## The KdV Right-Hand Side
#
# The vector field function computes $du/dt = -6u \, \partial_x u - \partial_x^3 u$.
#
# **Step by step**:
#
# 1. Transform $u(x)$ to spectral space: $\hat{u}(k) = \mathcal{F}[u]$
# 2. Compute $\partial_x u$ spectrally: multiply by $ik$, then IFFT back
# 3. Form the nonlinear product $6u \cdot \partial_x u$ in physical space
# 4. Transform the product back to spectral space
# 5. Compute $\partial_x^3 u$ spectrally: multiply $\hat{u}$ by $(ik)^3$
# 6. Combine, apply the dealiasing filter, and IFFT to get $du/dt$
#
# The dealiasing filter is applied to the combined result to remove any
# spurious high-frequency energy introduced by the nonlinear product.

# %%
def kdv_vector_field(t: float, y: State, args: Params) -> State:
    """Compute the RHS of the KdV equation: du/dt = -6u du/dx - d³u/dx³.

    Parameters
    ----------
    t : float
        Current time (unused — the KdV equation is autonomous).
    y : State
        Current state containing u(x).
    args : Params
        Grid and derivative operator.

    Returns
    -------
    State
        Time derivative du/dt.
    """
    del t  # Autonomous equation — no explicit time dependence

    u = y.u
    grid = args.grid
    k = grid.k

    # Forward FFT: u(x) -> û(k)
    u_hat = grid.transform(u)

    # --- Nonlinear term: -6u · du/dx ---
    # Compute du/dx in spectral space with dealiasing
    du_dx_hat = 1j * grid.k_dealias * u_hat
    du_dx = grid.transform(du_dx_hat, inverse=True).real
    nonlinear_phys = -6.0 * u * du_dx
    nonlinear_hat = grid.transform(nonlinear_phys)

    # --- Dispersion term: -d³u/dx³ ---
    # (ik)³ = -ik³, so -d³u/dx³ = -(ik)³ û = ik³ û
    dispersion_hat = -(1j * k) ** 3 * u_hat

    # --- Combine and dealias ---
    rhs_hat = (nonlinear_hat + dispersion_hat) * grid.dealias_filter()
    du_dt = grid.transform(rhs_hat, inverse=True).real

    return State(u=du_dt)

# %% [markdown]
# ## Initial Condition: Two Solitons
#
# We set up two sech$^2$ solitons separated in space. The exact one-soliton
# solution at $t = 0$ is:
#
# $$
# u(x, 0) = \frac{c}{2}\operatorname{sech}^2\!\left(\frac{\sqrt{c}}{2}(x - x_0)\right)
# $$
#
# The tall soliton ($c_1 = 4$) starts to the **left** of the short one ($c_2 = 1$).
# Since it travels faster, it will catch up, pass through, and emerge ahead.

# %%
def soliton(x: Float[Array, "N"], c: float, x0: float) -> Float[Array, "N"]:
    """Exact KdV one-soliton solution at t = 0.

    Parameters
    ----------
    x : array
        Spatial coordinates.
    c : float
        Soliton speed parameter. Amplitude = c/2, width ~ 2/sqrt(c).
    x0 : float
        Initial center position.

    Returns
    -------
    array
        Soliton profile u(x, 0).
    """
    return (c / 2.0) * (1.0 / jnp.cosh(jnp.sqrt(c) / 2.0 * (x - x0))) ** 2


# Construct the initial condition as a superposition
u0 = soliton(x, c1, x1_offset) + soliton(x, c2, x2_offset)
y0 = State(u=u0)

print(f"Initial condition shape: u0.shape = {u0.shape}")
print(f"Max amplitude:           {float(jnp.max(u0)):.4f}")
print(f"L2 norm:                 {float(jnp.sqrt(jnp.sum(u0**2) * (L / N))):.4f}")
print(f"Mass (integral of u):    {float(jnp.sum(u0) * (L / N)):.4f}")

# %% [markdown]
# ## Time Integration with diffrax
#
# We use `diffrax.Tsit5()`, the Tsitouras 5th-order Runge-Kutta method with
# adaptive step-size control. This is a good default for non-stiff ODEs.
#
# The `ODETerm` wraps our vector field, and `PIDController` adjusts the time
# step to maintain the specified tolerances.
#
# **Why adaptive stepping?** The KdV equation has a CFL-like stability
# constraint from the $k^3$ dispersion relation. With $N = 512$ modes on
# a domain of length $L = 100$, the maximum wavenumber is $k_{\max} \approx 16$,
# giving a maximum phase speed of $k_{\max}^2 \approx 256$. An adaptive solver
# automatically finds the right step size without us having to estimate this.

# %%
# Define the ODE problem
term = diffrax.ODETerm(kdv_vector_field)
solver = diffrax.Tsit5()
stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)

# Save at evenly spaced times for visualization
t_saves = jnp.linspace(0, t_end, n_saves)
saveat = diffrax.SaveAt(ts=t_saves)

print(f"Time span:          [0, {t_end}]")
print(f"Initial step size:  dt0 = {dt0}")
print(f"Save times shape:   {t_saves.shape}")
print(f"Solver:             Tsit5 (adaptive 5th-order Runge-Kutta)")
print(f"Tolerances:         rtol = 1e-6, atol = 1e-6")

# %% [markdown]
# ### Running the Simulation
#
# The `diffeqsolve` call integrates the ODE from $t = 0$ to $t = t_{\text{end}}$,
# saving the state at the requested times.

# %%
print("Running simulation...")

sol = diffrax.diffeqsolve(
    term,
    solver,
    t0=0.0,
    t1=t_end,
    dt0=dt0,
    y0=y0,
    args=params,
    saveat=saveat,
    stepsize_controller=stepsize_controller,
    max_steps=16**4,  # 65536 max steps — generous for adaptive solver
)

print(f"Simulation complete!")
print(f"  Final time:       {float(sol.ts[-1]):.4f}")
print(f"  Steps taken:      {sol.stats['num_steps']}")
print(f"  Solution shape:   u.shape = {sol.ys.u.shape}  (n_saves, N)")

# %% [markdown]
# ## Visualization
#
# We create a two-panel figure:
#
# - **Left panel**: Space-time heatmap of $u(x, t)$, showing the trajectories
#   of both solitons. The tall soliton's track has a steeper slope (faster speed).
#   You can see the tracks cross and then separate with a phase shift.
#
# - **Right panel**: Wave profiles at selected times, showing snapshots before,
#   during, and after the collision.

# %%
# Convert to NumPy for plotting
u_data = np.asarray(sol.ys.u)   # shape: (n_saves, N)
t_data = np.asarray(sol.ts)     # shape: (n_saves,)
x_data = np.asarray(x)          # shape: (N,)

print(f"Plotting data shapes:")
print(f"  u_data: {u_data.shape}  (time, space)")
print(f"  t_data: {t_data.shape}")
print(f"  x_data: {x_data.shape}")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Left panel: Space-time heatmap ---
im = ax1.pcolormesh(
    x_data, t_data, u_data,
    cmap="RdYlBu_r", shading="auto",
    vmin=-0.1, vmax=2.2,
)
fig.colorbar(im, ax=ax1, label="$u(x, t)$", shrink=0.85)
ax1.set_xlabel("Position $x$", fontsize=12)
ax1.set_ylabel("Time $t$", fontsize=12)
ax1.set_title("Space-Time Evolution", fontsize=13)
ax1.set_xlim(-35, 25)

# --- Right panel: Wave profiles at selected times ---
# Choose times that capture before, during, and after the collision
snapshot_times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_times)))

for t_snap, color in zip(snapshot_times, colors):
    # Find the closest saved time index
    idx = int(np.argmin(np.abs(t_data - t_snap)))
    ax2.plot(
        x_data, u_data[idx],
        color=color, linewidth=1.5,
        label=f"$t = {t_data[idx]:.1f}$",
    )

ax2.set_xlabel("Position $x$", fontsize=12)
ax2.set_ylabel("Amplitude $u(x)$", fontsize=12)
ax2.set_title("Wave Profiles at Selected Times", fontsize=13)
ax2.set_xlim(-35, 25)
ax2.set_ylim(-0.1, 2.5)
ax2.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax2.grid(True, linestyle="--", alpha=0.4)

fig.suptitle(
    "KdV Equation: Two-Soliton Interaction",
    fontsize=15, fontweight="bold", y=1.02,
)
plt.tight_layout()

# Save the figure
fig.savefig(
    IMG_DIR / "two_soliton_interaction.png",
    dpi=150, bbox_inches="tight",
)
print(f"Figure saved to: {IMG_DIR / 'two_soliton_interaction.png'}")
plt.show()

# %% [markdown]
# ![KdV two-soliton interaction](../images/kdv/two_soliton_interaction.png)

# %% [markdown]
# ## Key Physics: Soliton Phase Shift
#
# The most remarkable feature of this simulation is what happens during and after
# the collision:
#
# 1. **Before collision** ($t < 2$): Two distinct solitons propagate to the right
#    at their respective speeds $c_1 = 4$ and $c_2 = 1$.
#
# 2. **During collision** ($t \approx 2$--$3$): The solitons overlap and interact
#    nonlinearly. The combined profile does **not** simply look like two
#    superimposed sech$^2$ pulses.
#
# 3. **After collision** ($t > 3$): Both solitons re-emerge with exactly their
#    original shapes and speeds. However, each has been displaced:
#    - The **tall soliton** is shifted **forward** (it exits ahead of where it
#      would have been without interaction)
#    - The **short soliton** is shifted **backward**
#
# This **phase shift** is the only lasting effect of the collision. It can be
# computed analytically:
#
# $$
# \Delta x_1 = \frac{1}{\sqrt{c_1}} \ln\!\left(\frac{\sqrt{c_1} + \sqrt{c_2}}
# {\sqrt{c_1} - \sqrt{c_2}}\right), \qquad
# \Delta x_2 = -\frac{1}{\sqrt{c_2}} \ln\!\left(\frac{\sqrt{c_1} + \sqrt{c_2}}
# {\sqrt{c_1} - \sqrt{c_2}}\right)
# $$

# %%
# Compute the theoretical phase shifts
delta_x1 = (1.0 / np.sqrt(c1)) * np.log((np.sqrt(c1) + np.sqrt(c2)) / (np.sqrt(c1) - np.sqrt(c2)))
delta_x2 = -(1.0 / np.sqrt(c2)) * np.log((np.sqrt(c1) + np.sqrt(c2)) / (np.sqrt(c1) - np.sqrt(c2)))

print(f"Theoretical phase shifts:")
print(f"  Tall soliton (c={c1}):  Δx₁ = +{delta_x1:.4f}  (shifted forward)")
print(f"  Short soliton (c={c2}): Δx₂ = {delta_x2:.4f}  (shifted backward)")

# %% [markdown]
# ## Conservation Laws
#
# The KdV equation conserves infinitely many quantities. The first three are:
#
# $$
# I_1 = \int u \, dx, \qquad
# I_2 = \int u^2 \, dx, \qquad
# I_3 = \int \left(u^3 - \tfrac{1}{2}(\partial_x u)^2\right) dx
# $$
#
# Monitoring these provides a check on the accuracy of the numerical integration.

# %%
dx = L / N

# Compute conserved quantities at all saved times
mass = np.sum(u_data, axis=1) * dx          # I_1: mass
energy = np.sum(u_data**2, axis=1) * dx     # I_2: energy (L2 norm squared)

print(f"Conservation check over t = [0, {t_end}]:")
print(f"  Mass   I₁:  initial = {mass[0]:.8f},  final = {mass[-1]:.8f},  "
      f"relative change = {abs(mass[-1] - mass[0]) / abs(mass[0]):.2e}")
print(f"  Energy I₂:  initial = {energy[0]:.8f},  final = {energy[-1]:.8f},  "
      f"relative change = {abs(energy[-1] - energy[0]) / abs(energy[0]):.2e}")

# %% [markdown]
# The relative changes in both conserved quantities should be at the level of
# the solver tolerances ($\sim 10^{-6}$), confirming that the pseudospectral
# method combined with adaptive time stepping is highly accurate for this problem.
