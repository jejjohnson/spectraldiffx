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
# # 2D Navier-Stokes (Vorticity) Simulation
#
# This notebook simulates the 2D incompressible Navier-Stokes equations in their
# vorticity-streamfunction formulation, also known as the barotropic vorticity
# equation. It demonstrates the evolution of 2D turbulence.
#
# ## Equation
#
# The vorticity equation is:
# $$\frac{\partial \omega}{\partial t} + J(\psi, \omega) = \nu \nabla^{2n} \omega + F$$
#
# Where:
# - $\omega = \nabla^2 \psi$ is the vertical component of vorticity.
# - $\psi$ is the streamfunction.
# - The velocity field $(u, v)$ is defined by $u = -\partial\psi/\partial y$ and $v = \partial\psi/\partial x$.
# - $J(\psi, \omega)$ is the Jacobian (advection of vorticity).
# - $\nu$ is the kinematic viscosity, $n$ is the hyperviscosity order.
# - $F$ is an optional external forcing term.
#
# ## Numerical Method
#
# - **Spatial Discretization**: Pseudo-spectral method on a 2D periodic domain using `spectraldiffx`.
# - **Vorticity Inversion**: The streamfunction $\psi$ is recovered from $\omega$ by solving
#   $\nabla^2\psi = \omega$ in Fourier space via `SpectralHelmholtzSolver2D`.
# - **Time Integration**: Semi-implicit (IMEX) scheme using `diffrax.KenCarp4`.
#   The nonlinear advection $J(\psi, \omega)$ is explicit; the stiff diffusion is implicit.

# %%
import math
import pathlib
from typing import Annotated

import cyclopts
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import xarray as xr
from jaxtyping import Array, Float
from loguru import logger
from tqdm import tqdm

from spectraldiffx._src.grid import FourierGrid2D
from spectraldiffx._src.operators import SpectralDerivative2D
from spectraldiffx._src.solvers import SpectralHelmholtzSolver2D

# JAX configuration
jax.config.update("jax_enable_x64", True)

app = cyclopts.App()

# %% [markdown]
# ## 1. Parameter and State Management

# %%


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


# %% [markdown]
# ## 2. Helper Functions

# %%


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


# %% [markdown]
# ## 3. The Right-Hand-Side (RHS) of the PDE

# %%


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

    The Laplacian is computed directly in spectral space WITHOUT dealiasing,
    because the implicit operator must be a clean linear operator for the IMEX
    solver to invert correctly. Applying dealiasing here would cause mode mismatch
    at the dealiased boundary, leading to instability and NaN.
    """
    del t  # Autonomous equation
    omega_hat = args.grid.transform(y.omega)
    K2 = args.grid.K2
    # Apply (-K2)^nv in spectral space for laplacian^nv, without dealiasing
    lap_hat = ((-K2) ** args.nv) * omega_hat
    diffusion = args.grid.transform(lap_hat, inverse=True).real
    return State(omega=args.nu * diffusion)


# %% [markdown]
# ## 4. Initial Condition & Forcing Setup

# %%


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
    _X, Y = grid.X
    return jnp.sin(k_force * Y)


# %% [markdown]
# ## 5. Main Simulation Logic

# %%


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

    return sol, params, grid


# %% [markdown]
# ## 6. Post-processing with xarray

# %%


def build_dataset(sol, params: Params, grid: FourierGrid2D) -> xr.Dataset:
    """Assembles the simulation output into an xarray Dataset."""
    logger.info("Computing derived fields (psi, u, v) for all time steps...")

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
            "viscosity": params.nu,
            "hyperviscosity_order": params.nv,
        },
    )
    return ds


# %% [markdown]
# ## 7. Plotting

# %%


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
    return fig


# %% [markdown]
# ## Run the Simulation
#
# Configure parameters and execute. For interactive exploration, reduce `nx`/`ny`
# (e.g. 64 or 128) and `t_end`.

# %%

# Default parameters for notebook execution
NX = 256
NY = 256
DOMAIN_LENGTH = 2.0 * math.pi
VISCOSITY = 1e-6
HYPERVISCOSITY_ORDER = 2
FORCING_WAVENUMBER = 4.0
T_END = 50.0
DT0 = 1e-3
N_SAVES = 101
OUTPUT_DIR = pathlib.Path("./output/navier_stokes_2d")

# %%

logger.info("Setting up grid and operators...")
grid = FourierGrid2D.from_N_L(Nx=NX, Ny=NY, Lx=DOMAIN_LENGTH, Ly=DOMAIN_LENGTH)
deriv = SpectralDerivative2D(grid)
solver_helmholtz = SpectralHelmholtzSolver2D(grid)

forcing_field = create_forcing(grid, FORCING_WAVENUMBER)
params = Params(
    nu=VISCOSITY,
    nv=HYPERVISCOSITY_ORDER,
    grid=grid,
    deriv=deriv,
    solver=solver_helmholtz,
    forcing=forcing_field,
)

omega0 = generate_initial_vorticity(grid)
y0 = State(omega=omega0)

terms = diffrax.MultiTerm(
    diffrax.ODETerm(explicit_term), diffrax.ODETerm(implicit_term)
)
solver_ode = diffrax.KenCarp4()
stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
saveat = diffrax.SaveAt(ts=jnp.linspace(0, T_END, N_SAVES))

logger.info("Running simulation...")
sol = diffrax.diffeqsolve(
    terms,
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

ds = build_dataset(sol, params, grid)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / "ns2d_sim.nc"
ds.to_netcdf(output_path)
logger.success(f"Output saved to: {output_path}")

fig = plot_results(ds)
plt.show()

# %%

if __name__ == "__main__":
    app()
