"""
1.5-Layer Quasigeostrophic (QG) Model
=======================================

This script simulates the 1.5-layer (or "equivalent barotropic")
Quasigeostrophic (QG) model. This is a fundamental model in geophysical fluid
dynamics for studying the interaction of large-scale ocean or atmospheric
eddies with planetary waves (Rossby waves).

Equation:
---------
The full QG PV equation (in terms of total PV q_total = q' + beta*y) is:
  dq_total/dt + J(psi, q_total) = nu * laplacian^n(q_total) + F

The state variable evolved in this script is the **PV anomaly**:
  q' = q_total - beta*y

Rewriting in terms of q' and expanding J(psi, beta*y) = beta*v gives:
  dq'/dt + J(psi, q') = -beta*v + nu * laplacian^n(q') + F

Where:
- q' is the PV anomaly (the state variable stored and evolved by the solver).
- q_total = q' + beta*y is the full PV (reconstructed for output only).
- psi is the geostrophic streamfunction.
- J(psi, q') is the Jacobian (advection) of PV anomaly by the geostrophic flow.
- -beta*v is the beta-plane term (= J(psi, beta*y) rewritten in doubly-periodic form).
- nu, n, and F are viscosity, hyperviscosity order, and forcing.

The key to the QG model is the PV inversion relationship:
  q' = laplacian(psi) - (1/R^2)*psi

Where:
- laplacian(psi) is the relative vorticity.
- -(1/R^2)*psi represents stretching due to vertical displacement of the water column.
  R is the Rossby radius of deformation.

Numerical Method:
-----------------
- **PV Inversion**: The model evolves the PV anomaly q' = q_total - beta*y. Given
  the PV anomaly `q'`, the streamfunction `psi` is recovered by solving:
    (laplacian - 1/R^2)psi = q'
  This avoids FFT of the non-periodic beta*y ramp (which would cause Gibbs ringing).
  The beta-plane effect is instead incorporated in the explicit RHS as -beta*v.
  This is handled efficiently in Fourier space by `SpectralHelmholtzSolver2D`.

- **Time Integration**: A semi-implicit (IMEX) scheme with `diffrax.KenCarp4`.
  The nonlinear advection `J(psi, q)` and beta term `-beta*v` are explicit, and
  the diffusion `nu * laplacian^n(q)` is implicit (computed in spectral space
  without dealiasing to preserve the linearity of the implicit operator).

Usage:
------
Example:
  python scripts/qg_model.py --nx 128 --beta 10.0 --rossby-radius 0.1
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
    nv: int = eqx.static_field()  # Hyperviscosity order (n)
    beta: float  # Planetary vorticity gradient (beta)
    r_def_inv_sq: float  # Inverse squared Rossby radius of deformation (1/R^2)
    grid: FourierGrid2D = eqx.static_field()
    deriv: SpectralDerivative2D = eqx.static_field()
    solver: SpectralHelmholtzSolver2D = eqx.static_field()
    forcing: Float[Array, "Ny Nx"] | None  # Forcing term F


class State(eqx.Module):
    """The state of the simulation: the potential vorticity field `q`."""

    q: Float[Array, "Ny Nx"]


# ============================================================================
# 2. Helper Functions
# ============================================================================


@eqx.filter_jit
def get_psi_and_uv_from_q(q: Float[Array, "Ny Nx"], p: Params):
    """
    Computes streamfunction and velocity from Potential Vorticity (PV) anomaly.
    1. Solve (laplacian - 1/R^2)psi = q directly (q is the PV anomaly q' = q_total - beta*y).
    2. Compute u = -d(psi)/dy and v = d(psi)/dx.

    Note: The beta*y planetary vorticity is NOT subtracted here because q already
    represents the PV anomaly. Subtracting beta*y (a non-periodic linear ramp) from
    the Helmholtz RHS would introduce Gibbs ringing and cause NaN instability.
    """
    psi = p.solver.solve(q, alpha=p.r_def_inv_sq)
    dpsi_dx, dpsi_dy = p.deriv.gradient(psi)
    u, v = -dpsi_dy, dpsi_dx
    return psi, u, v


# ============================================================================
# 3. The Right-Hand-Side (RHS) of the PDE
# ============================================================================


def explicit_term(t: float, y: State, args: Params) -> State:
    """
    Computes the explicit part of the RHS: Advection + Beta-plane term + Forcing.
    RHS_exp = -J(psi, q) - beta*v + F

    The beta-plane effect is incorporated as -beta*v (= J(psi, beta*y) in the
    doubly-periodic formulation), avoiding direct use of the non-periodic beta*y field.
    """
    del t  # Autonomous equation
    q = y.q

    # Get streamfunction and velocity from PV
    _, u, v = get_psi_and_uv_from_q(q, args)

    # Advection of PV: -(u dot grad)q
    advection = -args.deriv.advection_scalar(u, v, q)

    # Beta-plane term: replaces J(psi, beta*y) = beta * dpsi/dx = beta * v
    beta_term = -args.beta * v

    # Add forcing if provided
    rhs = advection + beta_term
    rhs = rhs + args.forcing if args.forcing is not None else rhs

    return State(q=rhs)


def implicit_term(t: float, y: State, args: Params) -> State:
    """
    Computes the implicit part of the RHS: Diffusion.
    RHS_imp = nu * laplacian^n(q)

    The Laplacian is computed directly in spectral space WITHOUT dealiasing,
    because the implicit operator must be a clean linear operator for the IMEX
    solver to invert correctly. Applying dealiasing here would cause mode mismatch
    at the dealiased boundary, leading to instability and NaN.
    """
    del t  # Autonomous equation
    q_hat = args.grid.transform(y.q)
    K2 = args.grid.K2
    # Apply (-K2)^nv in spectral space for laplacian^nv, without dealiasing
    lap_hat = ((-K2) ** args.nv) * q_hat
    diffusion = args.grid.transform(lap_hat, inverse=True).real
    return State(q=args.nu * diffusion)


# ============================================================================
# 4. Initial Condition & Forcing Setup
# ============================================================================


def generate_initial_pv(grid: FourierGrid2D, seed: int = 42) -> jnp.ndarray:
    """
    Generates a random initial PV field with energy in a specific
    wavenumber band, suitable for triggering baroclinic instability.
    """
    key = jrandom.PRNGKey(seed)
    noise = jrandom.normal(key, shape=(grid.Ny, grid.Nx))
    q_hat = grid.transform(noise)

    # Band-pass filter to initialize turbulence around the deformation scale
    k_mag = jnp.sqrt(grid.K2)
    k_def = 2 * math.pi / (grid.Lx * 0.1)  # Rough deformation wavenumber
    k_min, k_max = k_def * 0.8, k_def * 1.2
    mask = (k_mag > k_min) & (k_mag < k_max)
    q_hat = jnp.where(mask, q_hat, 0.0)

    # Transform back and normalize
    q0 = grid.transform(q_hat, inverse=True).real
    return 1e-1 * q0 / jnp.std(q0)  # Scale to be a small perturbation


# ============================================================================
# 5. Main Simulation Logic
# ============================================================================


@app.default
def run_qg_model(
    nx: Annotated[
        int, cyclopts.Option("--nx", help="Number of grid points in x-direction.")
    ] = 128,
    ny: Annotated[
        int, cyclopts.Option("--ny", help="Number of grid points in y-direction.")
    ] = 128,
    domain_length: Annotated[
        float, cyclopts.Option("--length", help="Length of the square periodic domain.")
    ] = 1.0,
    viscosity: Annotated[
        float, cyclopts.Option("--viscosity", help="Kinematic viscosity (nu).")
    ] = 2e-12,
    hyperviscosity_order: Annotated[
        int,
        cyclopts.Option(
            "--hyperviscosity-order", help="Order of hyperviscosity (n in laplacian^n)."
        ),
    ] = 4,
    beta: Annotated[
        float, cyclopts.Option("--beta", help="Planetary vorticity gradient (beta).")
    ] = 10.0,
    rossby_radius: Annotated[
        float,
        cyclopts.Option("--rossby-radius", help="Rossby radius of deformation (R)."),
    ] = 0.1,
    t_end: Annotated[
        float, cyclopts.Option("--t-end", help="Final simulation time.")
    ] = 50.0,
    dt0: Annotated[
        float, cyclopts.Option("--dt0", help="Initial time step for adaptive solver.")
    ] = 1e-2,
    n_saves: Annotated[
        int, cyclopts.Option("--n-saves", help="Number of time points to save.")
    ] = 101,
    output_dir: Annotated[
        pathlib.Path | None,
        cyclopts.Option("--output-dir", help="Directory to save the output NetCDF."),
    ] = None,
):
    """Main function to run the 1.5-Layer QG simulation."""
    logger.info("=" * 60)
    logger.info("1.5-Layer Quasigeostrophic (QG) Model")
    logger.info("=" * 60)

    # --- Setup Grid, Operators, and Parameters ---
    logger.info("Setting up grid and operators...")
    grid = FourierGrid2D.from_N_L(Nx=nx, Ny=ny, Lx=domain_length, Ly=domain_length)
    deriv = SpectralDerivative2D(grid)
    solver_helmholtz = SpectralHelmholtzSolver2D(grid)
    logger.success(
        f"Grid initialized: {nx}x{ny}, Domain: {domain_length:.4f}x{domain_length:.4f}"
    )

    _, Y = grid.X
    planetary_vort = beta * Y
    r_def_inv_sq = 1.0 / (rossby_radius**2)

    params = Params(
        nu=viscosity,
        nv=hyperviscosity_order,
        beta=beta,
        r_def_inv_sq=r_def_inv_sq,
        grid=grid,
        deriv=deriv,
        solver=solver_helmholtz,
        forcing=None,
    )
    logger.info("Physical parameters:")
    logger.info(f"  - beta (planetary vorticity gradient): {beta}")
    logger.info(f"  - R (Rossby radius of deformation): {rossby_radius}")
    logger.info(f"  - nu (viscosity): {viscosity:.2e}")
    logger.info(f"  - n (hyperviscosity order): {hyperviscosity_order}")

    # --- Initial Condition: small random noise on a zonal jet ---
    logger.info("Generating random initial PV field...")
    q0 = generate_initial_pv(grid)
    y0 = State(q=q0)
    logger.success("Initial PV field generated (band-pass filtered noise)")

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
        max_steps=16**5,
    )
    logger.success(
        f"Simulation complete! Final time: {sol.ts[-1]:.4f}, "
        f"Steps taken: {sol.stats['num_steps']}"
    )

    # ========================================================================
    # 6. Post-processing with xarray
    # ========================================================================
    logger.info("Post-processing results...")
    logger.info("Computing derived fields (psi, omega, u, v) for all time steps...")

    psi, u, v = jax.vmap(get_psi_and_uv_from_q, in_axes=(0, None))(sol.ys.q, params)
    relative_vorticity = jax.vmap(params.deriv.laplacian)(psi)

    X, Y = grid.X
    ds = xr.Dataset(
        data_vars={
            "q": (("time", "y", "x"), sol.ys.q + planetary_vort),
            "psi": (("time", "y", "x"), psi),
            "omega": (("time", "y", "x"), relative_vorticity),
            "u": (("time", "y", "x"), u),
            "v": (("time", "y", "x"), v),
        },
        coords={"time": sol.ts, "x": X[0, :], "y": Y[:, 0]},
        attrs={
            "description": "1.5-Layer Quasigeostrophic Model",
            "beta": beta,
            "rossby_radius": rossby_radius,
            "viscosity": viscosity,
            "hyperviscosity_order": hyperviscosity_order,
        },
    )
    logger.info(
        f"Dataset created with shape: time={len(ds.time)}, y={len(ds.y)}, x={len(ds.x)}"
    )

    if output_dir is None:
        output_dir = pathlib.Path("./output/qg_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "qg_sim.nc"
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
    """Plots the PV and relative vorticity fields."""
    times_to_plot = ds.time.values[[0, len(ds.time) // 3, 2 * len(ds.time) // 3, -1]]

    fig = plt.figure(figsize=(11, 8))
    subfigs = fig.subfigures(1, 2, wspace=0.07)

    # Plot Potential Vorticity
    axes_q = subfigs[0].subplots(2, 2, sharex=True, sharey=True)
    subfigs[0].suptitle("Potential Vorticity (q)", fontsize=14)
    vmax_q = ds["q"].quantile(0.99)
    vmin_q = ds["q"].quantile(0.01)
    for i, t in enumerate(tqdm(times_to_plot, desc="Plotting PV")):
        ax = axes_q.flatten()[i]
        ds["q"].sel(time=t).plot.pcolormesh(
            ax=ax, cmap="RdBu_r", vmin=vmin_q, vmax=vmax_q, cbar_kwargs={"label": "PV"}
        )
        ax.set_title(f"Time = {t:.2f}")
        ax.set_aspect("equal")

    # Plot Relative Vorticity
    axes_om = subfigs[1].subplots(2, 2, sharex=True, sharey=True)
    subfigs[1].suptitle("Relative Vorticity (omega)", fontsize=14)
    vmax_om = ds["omega"].quantile(0.99)
    vmin_om = ds["omega"].quantile(0.01)
    for i, t in enumerate(tqdm(times_to_plot, desc="Plotting omega")):
        ax = axes_om.flatten()[i]
        ds["omega"].sel(time=t).plot.pcolormesh(
            ax=ax,
            cmap="RdBu_r",
            vmin=vmin_om,
            vmax=vmax_om,
            cbar_kwargs={"label": "omega"},
        )
        ax.set_title(f"Time = {t:.2f}")
        ax.set_aspect("equal")

    plt.tight_layout()


if __name__ == "__main__":
    app()
