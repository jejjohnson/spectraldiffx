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
# # Korteweg-de Vries (KdV) Equation Simulation
#
# This notebook solves the 1D Korteweg-de Vries (KdV) equation using a
# pseudo-spectral method with the `spectraldiffx` library. The KdV equation is a
# classic model for weakly nonlinear, long-wavelength waves, and it is famous
# for its soliton solutions.
#
# ## Equation
#
# The KdV equation in its canonical form is:
# $$\frac{\partial u}{\partial t} + 6u \frac{\partial u}{\partial x} + \frac{\partial^3 u}{\partial x^3} = 0$$
#
# Rearranged for time-stepping:
# $$\frac{\partial u}{\partial t} = -6u \frac{\partial u}{\partial x} - \frac{\partial^3 u}{\partial x^3}$$
#
# Where:
# - $u(x, t)$ is the wave amplitude.
# - The term $6u \, \partial u/\partial x$ is the nonlinear advection.
# - The term $\partial^3 u/\partial x^3$ is the linear dispersion.
#
# ## Numerical Method
#
# - **Spatial Discretization**: A pseudo-spectral method using FFTs via `spectraldiffx`.
#   The nonlinear term $u \cdot \partial u/\partial x$ is computed in physical space after
#   calculating the derivative in spectral space. The dispersive term is calculated
#   entirely in spectral space by multiplying the transformed field by $-(ik)^3$.
# - **Time Integration**: The `diffrax` library is used with an explicit
#   Runge-Kutta solver (`Tsit5`), as the equation is non-dissipative.
# - **Dealiasing**: The 2/3 rule is applied to the nonlinear term.
#
# ## Initial Condition
#
# The initial condition is a "two-soliton" solution:
# $$u(x, t) = 2c \operatorname{sech}^2\!\left(\sqrt{c}\,(x - 4ct)\right)$$

# %%
import pathlib
from pathlib import Path
from typing import Annotated

import cyclopts
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
from jaxtyping import Array, Float
from loguru import logger

matplotlib.use("Agg")

from spectraldiffx import FourierGrid1D
from spectraldiffx import SpectralDerivative1D

# JAX configuration
jax.config.update("jax_enable_x64", True)

IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "images" / "kdv"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the cyclopts app
app = cyclopts.App()

# %% [markdown]
# ## 1. Parameter and State Management

# %%


class Params(eqx.Module):
    """Simulation parameters."""

    grid: FourierGrid1D = eqx.field(static=True)
    deriv: SpectralDerivative1D = eqx.field(static=True)


class State(eqx.Module):
    """The state of the simulation: the wave amplitude `u`."""

    u: Float[Array, "Nx"]


# %% [markdown]
# ## 2. The Right-Hand-Side (RHS) of the PDE

# %%


def kdv_vector_field(t: float, y: State, args: Params) -> State:
    """
    Computes the RHS of the KdV equation: -6u * du/dx - d^3u/dx^3.
    This is the "vector field" for the ODE solver.
    """
    del t  # The KdV equation is autonomous

    u_hat = args.grid.transform(y.u)
    k = args.grid.k
    k_dealias = args.grid.k_dealias

    # --- Nonlinear Advection Term: -6u * du/dx ---
    # Computed using the pseudo-spectral method.
    du_dx_hat = 1j * k_dealias * u_hat
    du_dx = args.grid.transform(du_dx_hat, inverse=True).real
    advection_phys = -6.0 * y.u * du_dx
    advection_hat = args.grid.transform(advection_phys)

    # --- Linear Dispersion Term: -d^3u/dx^3 ---
    # Computed entirely in spectral space.
    d3u_dx3_hat = (1j * k) ** 3 * u_hat
    dispersion_hat = -d3u_dx3_hat

    # --- Combine terms in spectral space and transform back ---
    # We apply the dealiasing mask to the final result to filter out any
    # high-frequency noise that may have been introduced.
    total_hat = (advection_hat + dispersion_hat) * args.grid.dealias_filter()
    du_dt = args.grid.transform(total_hat, inverse=True).real

    return State(u=du_dt)


# %% [markdown]
# ## 3. Initial Condition

# %%


def two_soliton_initial_condition(
    x: Float[Array, "Nx"], c1: float, x1: float, c2: float, x2: float
) -> Float[Array, "Nx"]:
    """
    Creates a two-soliton initial condition.
    A single soliton is given by u(x) = 2c * sech^2(sqrt(c) * (x - x0)).
    """
    soliton1 = 2 * c1 * (1 / jnp.cosh(jnp.sqrt(c1) * (x - x1))) ** 2
    soliton2 = 2 * c2 * (1 / jnp.cosh(jnp.sqrt(c2) * (x - x2))) ** 2
    return soliton1 + soliton2


# %% [markdown]
# ## 4. Main Simulation Logic

# %%


@app.default
def run_kdv(
    nx: Annotated[
        int, cyclopts.Parameter("--nx", help="Number of grid points (resolution).")
    ] = 512,
    domain_length: Annotated[
        float, cyclopts.Parameter("--length", help="Length of the periodic domain.")
    ] = 100.0,
    t_end: Annotated[
        float, cyclopts.Parameter("--t-end", help="Final simulation time.")
    ] = 20.0,
    dt0: Annotated[
        float, cyclopts.Parameter("--dt0", help="Initial time step for adaptive solver.")
    ] = 1e-3,
    n_saves: Annotated[
        int, cyclopts.Parameter("--n-saves", help="Number of time points to save.")
    ] = 201,
    output_dir: Annotated[
        pathlib.Path | None,
        cyclopts.Parameter("--output-dir", help="Directory to save the output NetCDF."),
    ] = None,
):
    """
    Main function to run the 1D KdV equation simulation.
    """
    logger.info("=" * 60)
    logger.info("1D Korteweg-de Vries (KdV) Equation Simulation")
    logger.info("=" * 60)

    # --- Setup Grid and Operators ---
    logger.info("Setting up grid and operators...")
    # The domain is shifted to be centered at 0 for convenience with the sech initial condition.
    # Note: the FFT grid uses grid.x in [0, L), while x below is shifted to [-L/2, L/2).
    # Soliton positions (x1, x2) below must be given relative to this shifted x coordinate.
    grid = FourierGrid1D.from_N_L(N=nx, L=domain_length, dealias="2/3")
    x = (
        grid.x - domain_length / 2.0
    )  # Center domain at x=0; used only for initial condition
    deriv = SpectralDerivative1D(grid)
    params = Params(grid=grid, deriv=deriv)
    logger.success(
        f"Grid initialized: {nx} points on [-L/2, L/2], L={domain_length:.2f}"
    )

    # --- Initial Condition ---
    logger.info("Setting up two-soliton initial condition...")
    # Two solitons: a tall, fast one (c1) and a short, slow one (c2).
    # The tall one starts behind and will overtake the short one.
    c1, x1 = 2.0, domain_length / 4.0
    c2, x2 = 1.0, domain_length / 2.5
    u0 = two_soliton_initial_condition(x, c1, x1, c2, x2)
    y0 = State(u=u0)
    logger.success(f"Soliton 1: amplitude c1={c1}, position x1={x1:.2f}")
    logger.success(f"Soliton 2: amplitude c2={c2}, position x2={x2:.2f}")

    # --- Time Integration Setup ---
    logger.info("Configuring time integration...")
    term = diffrax.ODETerm(kdv_vector_field)
    solver = diffrax.Tsit5()  # A standard explicit RK solver
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_end, n_saves))
    logger.success(f"Time integration: t=[0, {t_end}], saving {n_saves} snapshots")
    logger.info(
        "Solver: Tsit5 (explicit RK), Adaptive time-stepping with PID controller"
    )

    # --- Run the Simulation ---
    logger.info("Running simulation...")
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=t_end,
        dt0=dt0,
        y0=y0,
        args=params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=16**4,
    )
    logger.success(
        f"Simulation complete! Final time: {sol.ts[-1]:.4f}, "
        f"Steps taken: {sol.stats['num_steps']}"
    )

    # --- Post-processing and output ---
    logger.info("Post-processing results...")
    ds = build_dataset(sol, x, domain_length)

    if output_dir is None:
        output_dir = pathlib.Path("./output/kdv")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "kdv_sim.nc"
    ds.to_netcdf(output_path)
    logger.success(f"Output saved to: {output_path}")

    logger.info("Generating plots...")
    fig = plot_results(ds)
    fig.savefig(IMG_DIR / "two_soliton_interaction.png", dpi=150, bbox_inches="tight")
    logger.success("Plots generated successfully!")
    plt.show()

    return ds


# %% [markdown]
# ## 5. Post-processing with xarray

# %%


def build_dataset(sol, x, domain_length: float) -> xr.Dataset:
    """Assembles the simulation output into an xarray Dataset."""
    ds = xr.Dataset(
        data_vars={
            "u": (("time", "x"), sol.ys.u),
        },
        coords={
            "time": sol.ts,
            "x": x,
        },
        attrs={
            "description": "1D Korteweg-de Vries (KdV) Equation",
            "domain_length": domain_length,
        },
    )
    return ds


# %% [markdown]
# ## 6. Plotting

# %%


def plot_results(ds: xr.Dataset):
    """
    Simple plotting of the KdV simulation results.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot of u(x, t) as a heatmap
    ds["u"].plot(ax=ax1, cmap="viridis")
    ax1.set_title("Wave amplitude u(x, t)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Position (x)")

    # Plot of u(x) at different time snapshots
    times_to_plot = ds.time.values[:: len(ds.time) // 6]
    for t in times_to_plot:
        ds["u"].sel(time=t).plot(ax=ax2, label=f"t={t:.2f}")
    ax2.set_title("Wave profiles at different times")
    ax2.set_xlabel("Position (x)")
    ax2.set_ylabel("Amplitude u")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("KdV Equation - Two-Soliton Interaction")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# %% [markdown]
# ## Run the Simulation
#
# Configure parameters and execute. Adjust `nx`, `t_end`, and `n_saves` as needed.

# %%

# Default parameters for notebook execution
NX = 512
DOMAIN_LENGTH = 100.0
T_END = 20.0
DT0 = 1e-3
N_SAVES = 201
OUTPUT_DIR = pathlib.Path("./output/kdv")

# %%

logger.info("Setting up grid and operators...")
grid = FourierGrid1D.from_N_L(N=NX, L=DOMAIN_LENGTH, dealias="2/3")
x = grid.x - DOMAIN_LENGTH / 2.0
deriv = SpectralDerivative1D(grid)
params = Params(grid=grid, deriv=deriv)

c1, x1 = 2.0, DOMAIN_LENGTH / 4.0
c2, x2 = 1.0, DOMAIN_LENGTH / 2.5
u0 = two_soliton_initial_condition(x, c1, x1, c2, x2)
y0 = State(u=u0)

term = diffrax.ODETerm(kdv_vector_field)
solver_ode = diffrax.Tsit5()
stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)
saveat = diffrax.SaveAt(ts=jnp.linspace(0, T_END, N_SAVES))

logger.info("Running simulation...")
sol = diffrax.diffeqsolve(
    term,
    solver_ode,
    t0=0,
    t1=T_END,
    dt0=DT0,
    y0=y0,
    args=params,
    saveat=saveat,
    stepsize_controller=stepsize_controller,
    max_steps=16**4,
)
logger.success(f"Simulation complete! Steps taken: {sol.stats['num_steps']}")

# %%

ds = build_dataset(sol, x, DOMAIN_LENGTH)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / "kdv_sim.nc"
ds.to_netcdf(output_path)
logger.success(f"Output saved to: {output_path}")

fig = plot_results(ds)
fig.savefig(IMG_DIR / "two_soliton_interaction.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![KdV two-soliton interaction](../images/kdv/two_soliton_interaction.png)

# %%

if __name__ == "__main__":
    app()
