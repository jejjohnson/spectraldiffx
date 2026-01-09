"""
Korteweg-de Vries (KdV) Equation Simulation
=============================================

This script solves the 1D Korteweg-de Vries (KdV) equation using a
pseudo-spectral method with the `spectraldiffx` library. The KdV equation is a
classic model for weakly nonlinear, long-wavelength waves, and it is famous
for its soliton solutions.

Equation:
---------
The KdV equation in its canonical form is:
  du/dt + 6u * du/dx + d^3u/dx^3 = 0

Rearranged for time-stepping:
  du/dt = -6u * du/dx - d^3u/dx^3

Where:
- u(x, t) is the wave amplitude.
- The term `6u * du/dx` is the nonlinear advection.
- The term `d^3u/dx^3` is the linear dispersion.

Numerical Method:
-----------------
- **Spatial Discretization**: A pseudo-spectral method using FFTs via `spectraldiffx`.
  The nonlinear term `u * du/dx` is computed in physical space after
  calculating the derivative in spectral space. The dispersive term `d^3u/dx^3`
  is calculated entirely in spectral space by multiplying the transformed
  field by `-(ik)^3`.

- **Time Integration**: The `diffrax` library is used with an explicit
  Runge-Kutta solver (`Tsit5`), as the equation is non-dissipative and does not
  require an implicit scheme for stability, provided the time step is
  sufficiently small (handled by the adaptive stepsize controller).

- **Dealiasing**: The 2/3 rule is applied to the nonlinear term to prevent
  aliasing errors.

Initial Condition:
------------------
The initial condition is a "two-soliton" solution, which demonstrates the
remarkable property of solitons to pass through each other and emerge unchanged.
A soliton solution has the form: `u(x,t) = 2c * sech^2(sqrt(c) * (x - 4ct))`.

Usage:
------
The script is run from the command line, with parameters controlled by `cyclopts`.

Example:
  python scripts/kdv.py --nx 512 --length 100 --t-end 20.0
"""

import pathlib
from typing import Annotated

import cyclopts
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from loguru import logger
import matplotlib.pyplot as plt
import xarray as xr

from spectraldiffx._src.grid import FourierGrid1D
from spectraldiffx._src.operators import SpectralDerivative1D

# JAX configuration
jax.config.update("jax_enable_x64", True)

# Initialize the cyclopts app
app = cyclopts.App()


# ============================================================================
# 1. Parameter and State Management
# ============================================================================


class Params(eqx.Module):
    """Simulation parameters."""

    grid: FourierGrid1D = eqx.field(static=True)
    deriv: SpectralDerivative1D = eqx.field(static=True)


class State(eqx.Module):
    """The state of the simulation: the wave amplitude `u`."""

    u: Float[Array, "Nx"]


# ============================================================================
# 2. The Right-Hand-Side (RHS) of the PDE
# ============================================================================


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


# ============================================================================
# 3. Initial Condition
# ============================================================================


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


# ============================================================================
# 4. Main Simulation Logic
# ============================================================================


@app.default
def run_kdv(
    nx: Annotated[
        int, cyclopts.Option("--nx", help="Number of grid points (resolution).")
    ] = 512,
    domain_length: Annotated[
        float, cyclopts.Option("--length", help="Length of the periodic domain.")
    ] = 100.0,
    t_end: Annotated[
        float, cyclopts.Option("--t-end", help="Final simulation time.")
    ] = 20.0,
    dt0: Annotated[
        float, cyclopts.Option("--dt0", help="Initial time step for adaptive solver.")
    ] = 1e-3,
    n_saves: Annotated[
        int, cyclopts.Option("--n-saves", help="Number of time points to save.")
    ] = 201,
    output_dir: Annotated[
        pathlib.Path | None,
        cyclopts.Option("--output-dir", help="Directory to save the output NetCDF."),
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
    grid = FourierGrid1D.from_N_L(N=nx, L=domain_length, dealias="2/3")
    x = grid.x - domain_length / 2.0  # Center domain at x=0
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

    # ========================================================================
    # 5. Post-processing with xarray
    # ========================================================================
    logger.info("Post-processing results...")

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
    logger.info(f"Dataset created with shape: time={len(ds.time)}, x={len(ds.x)}")

    if output_dir is None:
        output_dir = pathlib.Path("./output/kdv")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "kdv_sim.nc"
    ds.to_netcdf(output_path)
    logger.success(f"Output saved to: {output_path}")

    # ========================================================================
    # 6. Plotting
    # ========================================================================
    logger.info("Generating plots...")
    plot_results(ds)
    logger.success("Plots generated successfully!")
    plt.show()


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


if __name__ == "__main__":
    app()
