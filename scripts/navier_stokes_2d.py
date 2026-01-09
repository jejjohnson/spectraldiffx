"""
2D Navier-Stokes (Vorticity) Simulation
========================================

This script simulates the 2D incompressible Navier-Stokes equations in their
vorticity-streamfunction formulation, also known as the barotropic vorticity
equation. It demonstrates the evolution of 2D turbulence.

Equation:
---------
The vorticity equation is:
  d(omega)/dt + J(psi, omega) = nu * laplacian^n(omega) + F

Where:
- omega = laplacian(psi) is the vertical component of vorticity.
- psi is the streamfunction.
- The velocity field (u, v) is defined by u = -d(psi)/dy and v = d(psi)/dx.
- J(psi, omega) = (d(psi)/dx)(d(omega)/dy) - (d(psi)/dy)(d(omega)/dx) is the
  Jacobian operator, representing the advection of vorticity, (u dot grad)omega.
- nu is the kinematic viscosity.
- n is the order of the hyperviscosity (n=1 for standard viscosity).
- F is an optional external forcing term.

Numerical Method:
-----------------
- **Spatial Discretization**: A pseudo-spectral method on a 2D periodic domain
  using `spectraldiffx`.
- **Vorticity Inversion**: At each step, the streamfunction psi is recovered from
  the vorticity omega by solving the Poisson equation laplacian(psi) = omega. This
  is done efficiently in Fourier space using `SpectralHelmholtzSolver2D`.
- **Time Integration**: A semi-implicit (IMEX) scheme using `diffrax.KenCarp4`.
  The nonlinear advection term `J(psi, omega)` and forcing `F` are treated
  explicitly, while the stiff diffusion term `nu * laplacian^n(omega)` is treated
  implicitly for numerical stability.

Initial Condition & Forcing:
----------------------------
- The simulation starts from a random vorticity field with energy concentrated
  at intermediate wavenumbers.
- An optional, steady forcing pattern can be applied at a specific wavenumber
  to inject energy and sustain turbulence.

Usage:
------
Example:
  python scripts/navier_stokes_2d.py --nx 256 --viscosity 1e-6 --hyperviscosity-order 2
"""

import math
import pathlib
from typing import Annotated

import cyclopts
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float
from loguru import logger
import matplotlib.pyplot as plt
from tqdm import tqdm
import xarray as xr

from spectraldiffx._src.grid import FourierGrid2D
from spectraldiffx._src.operators import SpectralDerivative2D
from spectraldiffx._src.solvers import SpectralHelmholtzSolver2D

# JAX configuration
jax.config.update("jax_enable_x64", True)

app = cyclopts.App()


# ============================================================================
# 1. Parameter and State Management
# ============================================================================


class Params(eqx.Module):
    """Simulation parameters."""

    nu: float  # Kinematic viscosity (nu)
    nv: int = eqx.field(static=True)  # Hyperviscosity order (n)
    grid: FourierGrid2D = eqx.field(static=True)
    deriv: SpectralDerivative2D = eqx.field(static=True)
    solver: SpectralHelmholtzSolver2D = eqx.field(static=True)
    forcing: Float[Array, "Ny Nx"] | None  # Forcing term F


class State(eqx.Module):
    """The state of the simulation: the vorticity field `omega`."""

    omega: Float[Array, "Ny Nx"]


# ============================================================================
# 2. Helper Functions
# ============================================================================


@eqx.filter_jit
def get_psi_and_uv(omega: Float[Array, "Ny Nx"], p: Params):
    """
    Computes streamfunction and velocity from vorticity.
    1. Solve Poisson equation laplacian(psi) = omega for psi.
    2. Compute u = -d(psi)/dy and v = d(psi)/dx.
    """
    psi = p.solver.solve(omega, alpha=0.0)
    dpsi_dx, dpsi_dy = p.deriv.gradient(psi)
    u, v = -dpsi_dy, dpsi_dx
    return psi, u, v


# ============================================================================
# 3. The Right-Hand-Side (RHS) of the PDE
# ============================================================================


def explicit_term(t: float, y: State, args: Params) -> State:
    """
    Computes the explicit part of the RHS: Advection + Forcing.
    RHS_exp = -J(psi, omega) + F
    """
    del t  # Autonomous equation
    omega = y.omega

    # Get velocity field from vorticity
    _, u, v = get_psi_and_uv(omega, args)

    # Advection: -(u dot grad)omega
    advection = -args.deriv.advection_scalar(u, v, omega)

    # Add forcing if provided
    rhs = advection + args.forcing if args.forcing is not None else advection

    return State(omega=rhs)


def implicit_term(t: float, y: State, args: Params) -> State:
    """
    Computes the implicit part of the RHS: Diffusion.
    RHS_imp = nu * laplacian^n(omega)
    """
    del t  # Autonomous equation
    omega = y.omega

    # Apply Laplacian `nv` times
    lap_omega = omega
    for _ in range(args.nv):
        lap_omega = args.deriv.laplacian(lap_omega)

    diffusion = args.nu * lap_omega
    return State(omega=diffusion)


# ============================================================================
# 4. Initial Condition & Forcing Setup
# ============================================================================


def generate_initial_vorticity(grid: FourierGrid2D, seed: int = 42) -> jnp.ndarray:
    """
    Generates a random initial vorticity field with energy in a specific
    wavenumber band.
    """
    key = jrandom.PRNGKey(seed)
    q0 = jrandom.normal(key, shape=(grid.Ny, grid.Nx))
    q_hat = grid.transform(q0)

    # Band-pass filter to initialize turbulence
    k_mag = jnp.sqrt(grid.K2)
    k_min, k_max = 6, 10
    mask = (k_mag > k_min) & (k_mag < k_max)
    q_hat = jnp.where(mask, q_hat, 0.0)

    # Transform back and normalize
    q0 = grid.transform(q_hat, inverse=True).real
    return q0 / jnp.std(q0)


def create_forcing(grid: FourierGrid2D, k_force: float) -> jnp.ndarray:
    """Creates a sinusoidal forcing pattern at a specific wavenumber."""
    X, Y = grid.X
    return jnp.sin(k_force * Y)


# ============================================================================
# 5. Main Simulation Logic
# ============================================================================


@app.default
def run_navier_stokes(
    nx: Annotated[
        int, cyclopts.Option("--nx", help="Number of grid points in x-direction.")
    ] = 256,
    ny: Annotated[
        int, cyclopts.Option("--ny", help="Number of grid points in y-direction.")
    ] = 256,
    domain_length: Annotated[
        float, cyclopts.Option("--length", help="Length of the square periodic domain.")
    ] = 2.0 * math.pi,
    viscosity: Annotated[
        float, cyclopts.Option("--viscosity", help="Kinematic viscosity (nu).")
    ] = 1e-6,
    hyperviscosity_order: Annotated[
        int,
        cyclopts.Option(
            "--hyperviscosity-order",
            help="Order of hyperviscosity (n in laplacian^n). n=1 is standard viscosity.",
        ),
    ] = 2,
    forcing_wavenumber: Annotated[
        float | None,
        cyclopts.Option(
            "--k-force",
            help="Wavenumber of the sinusoidal forcing. If not set, no forcing.",
        ),
    ] = 4.0,
    t_end: Annotated[
        float, cyclopts.Option("--t-end", help="Final simulation time.")
    ] = 50.0,
    dt0: Annotated[
        float, cyclopts.Option("--dt0", help="Initial time step for adaptive solver.")
    ] = 1e-3,
    n_saves: Annotated[
        int, cyclopts.Option("--n-saves", help="Number of time points to save.")
    ] = 101,
    output_dir: Annotated[
        pathlib.Path | None,
        cyclopts.Option("--output-dir", help="Directory to save the output NetCDF."),
    ] = None,
):
    """
    Main function to run the 2D Navier-Stokes simulation.
    """
    logger.info("=" * 60)
    logger.info("2D Navier-Stokes (Vorticity) Simulation")
    logger.info("=" * 60)

    # --- Setup Grid, Operators, and Forcing ---
    logger.info("Setting up grid and operators...")
    grid = FourierGrid2D.from_N_L(Nx=nx, Ny=ny, Lx=domain_length, Ly=domain_length)
    deriv = SpectralDerivative2D(grid)
    solver_helmholtz = SpectralHelmholtzSolver2D(grid)
    logger.success(
        f"Grid initialized: {nx}x{ny}, Domain: {domain_length:.4f}x{domain_length:.4f}"
    )

    logger.info("Configuring forcing...")
    forcing_field = (
        create_forcing(grid, forcing_wavenumber) if forcing_wavenumber else None
    )
    if forcing_field is not None:
        logger.success(f"Forcing enabled at wavenumber k={forcing_wavenumber}")
    else:
        logger.info("No forcing applied")

    params = Params(
        nu=viscosity,
        nv=hyperviscosity_order,
        grid=grid,
        deriv=deriv,
        solver=solver_helmholtz,
        forcing=forcing_field,
    )
    logger.info(
        f"Viscosity: nu={viscosity:.2e}, Hyperviscosity order: n={hyperviscosity_order}"
    )

    # --- Initial Condition ---
    logger.info("Generating random initial vorticity field...")
    omega0 = generate_initial_vorticity(grid)
    y0 = State(omega=omega0)
    logger.success("Initial vorticity field generated (band-pass filtered noise)")

    # --- Time Integration Setup (IMEX) ---
    logger.info("Configuring time integration...")
    terms = diffrax.MultiTerm(
        diffrax.ODETerm(explicit_term), diffrax.ODETerm(implicit_term)
    )
    solver = diffrax.KenCarp4()
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_end, n_saves))
    logger.success(f"Time integration: t=[0, {t_end}], saving {n_saves} snapshots")
    logger.info("Solver: KenCarp4 (IMEX), Adaptive time-stepping with PID controller")

    # --- Run the Simulation ---
    logger.info("Running simulation...")
    sol = diffrax.diffeqsolve(
        terms,
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
    # 6. Post-processing with xarray
    # ========================================================================
    logger.info("Post-processing results...")
    logger.info("Computing derived fields (psi, u, v) for all time steps...")

    # Compute final fields for the dataset
    # JAX vmap is used to efficiently apply the function over the time dimension
    psi, u, v = jax.vmap(get_psi_and_uv, in_axes=(0, None))(sol.ys.omega, params)

    X, Y = grid.X
    ds = xr.Dataset(
        data_vars={
            "omega": (("time", "y", "x"), sol.ys.omega),
            "psi": (("time", "y", "x"), psi),
            "u": (("time", "y", "x"), u),
            "v": (("time", "y", "x"), v),
        },
        coords={
            "time": sol.ts,
            "x": X[0, :],
            "y": Y[:, 0],
        },
        attrs={
            "description": "2D Incompressible Navier-Stokes (Vorticity Formulation)",
            "viscosity": viscosity,
            "hyperviscosity_order": hyperviscosity_order,
        },
    )
    logger.info(
        f"Dataset created with shape: time={len(ds.time)}, y={len(ds.y)}, x={len(ds.x)}"
    )

    if output_dir is None:
        output_dir = pathlib.Path("./output/navier_stokes_2d")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ns2d_sim.nc"
    ds.to_netcdf(output_path)
    logger.success(f"Output saved to: {output_path}")

    # ========================================================================
    # 7. Plotting
    # ========================================================================
    logger.info("Generating plots...")
    plot_results(ds)
    logger.success("Plots generated successfully!")
    plt.show()


def plot_results(ds: xr.Dataset):
    """Plots the vorticity field at several time points."""
    times_to_plot = ds.time.values[[0, len(ds.time) // 3, 2 * len(ds.time) // 3, -1]]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8.5),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()

    # Find a common color scale
    vmax = ds["omega"].quantile(0.99)
    vmin = ds["omega"].quantile(0.01)

    for i, t in enumerate(tqdm(times_to_plot, desc="Generating plots")):
        ds["omega"].sel(time=t).plot.pcolormesh(
            ax=axes[i],
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            cbar_kwargs={"label": "Vorticity omega"},
        )
        axes[i].set_title(f"Time = {t:.2f}")
        axes[i].set_aspect("equal")

    fig.suptitle("Vorticity Evolution in 2D Turbulence", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])


if __name__ == "__main__":
    app()
