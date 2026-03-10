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
# # Pseudospectral Differentiation - 1D
#
# This notebook demonstrates how to use the `spectraldiffx` library for
# computing derivatives using the pseudospectral (Fourier) method in 1D.

# %%
import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

jax.config.update("jax_enable_x64", True)
sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)

# %% [markdown]
# ## Define Test Function
#
# We use a simple periodic function:
# $$
# u(x) = \sin(2x) + \frac{1}{2}\cos(5x)
# $$
#
# With analytical derivatives:
# $$
# \frac{du}{dx} = 2\cos(2x) - \frac{5}{2}\sin(5x)
# $$
# $$
# \frac{d^2u}{dx^2} = -4\sin(2x) - \frac{25}{2}\cos(5x)
# $$

# %%
# Define the function and its analytical derivatives
f = lambda x: jnp.sin(2 * x) + 0.5 * jnp.cos(5 * x)
df = lambda x: 2.0 * jnp.cos(2.0 * x) - 2.5 * jnp.sin(5.0 * x)
d2f = lambda x: -4.0 * jnp.sin(2.0 * x) - 12.5 * jnp.cos(5.0 * x)

# %% [markdown]
# ## Setup Grid Using New API
#
# We use `FourierGrid1D` to define the computational domain.

# %%
from spectraldiffx._src.grid import FourierGrid1D
from spectraldiffx._src.operators import SpectralDerivative1D

# Grid parameters
Nx = 32
Lx = 2 * math.pi

# Create grid using the new API
grid = FourierGrid1D.from_N_L(N=Nx, L=Lx, dealias="2/3")

# Get grid points
x_coords = grid.x
x_plot_coords = jnp.linspace(0, Lx, 256, endpoint=False)

# Initialize fields
u = f(x_coords)
u_plot = f(x_plot_coords)

# %%
print(f"Grid points (N): {grid.N}")
print(f"Domain length (L): {grid.L}")
print(f"Grid spacing (dx): {grid.dx}")
print(f"Dealiasing method: {grid.dealias}")

# %%
fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(x_plot_coords, u_plot, linestyle="-", color="black", label="$u(x)$")
ax.scatter(x_coords, u, color="r", marker="*", label="$u_i$", zorder=3)
ax.set(xlabel=r"$x$", ylabel=r"$u(x)$")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## First Derivative
#
# ### Manual Computation (For Understanding)
#
# The pseudospectral derivative works by:
# 1. Transform to Fourier space: $\hat{u}_k = \text{FFT}(u)$
# 2. Multiply by $ik$: $\widehat{du/dx}_k = ik \cdot \hat{u}_k$
# 3. Transform back: $du/dx = \text{IFFT}(\widehat{du/dx})$

# %%
# Manual computation for educational purposes
k = grid.k  # Wavenumbers from grid

# Forward FFT
u_hat = jnp.fft.fft(u)

# Multiply by ik for first derivative
du_hat = 1j * k * u_hat

# Inverse FFT
du_dx_manual = jnp.fft.ifft(du_hat).real

# %% [markdown]
# ### Using SpectralDerivative1D Operator
#
# The `SpectralDerivative1D` class provides a clean API for computing derivatives.

# %%
# Create the derivative operator
deriv = SpectralDerivative1D(grid=grid)

# Compute first derivative
du_dx = deriv(u, order=1)

# Or use the gradient method
du_dx_grad = deriv.gradient(u)

# %%
# Analytical solution for comparison
du_dx_analytical = jax.vmap(df)(x_coords)
du_dx_plot = jax.vmap(df)(x_plot_coords)

# %%
fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(x_plot_coords, du_dx_plot, linestyle="-", color="black", label=r"$du/dx$ (analytical)")
ax.scatter(x_coords, du_dx, color="green", marker=".", label="SpectralDerivative1D", zorder=3)
ax.scatter(x_coords, du_dx_analytical, color="red", marker="x", s=20, label="Analytical", zorder=3)
ax.set(xlabel=r"$x$", ylabel=r"$du/dx$")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Check error
error = jnp.abs(du_dx - du_dx_analytical)
print(f"Max error in first derivative: {error.max():.2e}")

# %%
fig, ax = plt.subplots(figsize=(8, 3))

ax.semilogy(x_coords, error, color="green", marker=".", label="Error")
ax.set(xlabel=r"$x$", ylabel=r"$|du/dx - du/dx_{analytical}|$")
ax.set_title("First Derivative Error (Spectral Method)")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Second Derivative
#
# We can compute higher-order derivatives by specifying the `order` parameter,
# or use the `laplacian` method for the second derivative.

# %%
# Compute second derivative using order parameter
d2u_dx2 = deriv(u, order=2)

# Or use the laplacian method (equivalent for 1D)
d2u_dx2_lap = deriv.laplacian(u)

# Analytical solution
d2u_dx2_analytical = jax.vmap(d2f)(x_coords)
d2u_dx2_plot = jax.vmap(d2f)(x_plot_coords)

# %%
fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(x_plot_coords, d2u_dx2_plot, linestyle="-", color="black", label=r"$d^2u/dx^2$ (analytical)")
ax.scatter(x_coords, d2u_dx2, color="green", marker=".", label="SpectralDerivative1D", zorder=3)
ax.scatter(x_coords, d2u_dx2_analytical, color="red", marker="x", s=20, label="Analytical", zorder=3)
ax.set(xlabel=r"$x$", ylabel=r"$d^2u/dx^2$")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Check error
error_2nd = jnp.abs(d2u_dx2 - d2u_dx2_analytical)
print(f"Max error in second derivative: {error_2nd.max():.2e}")

# %%
fig, ax = plt.subplots(figsize=(8, 3))

ax.semilogy(x_coords, error_2nd, color="green", marker=".", label="Error")
ax.set(xlabel=r"$x$", ylabel=r"$|d^2u/dx^2 - d^2u/dx^2_{analytical}|$")
ax.set_title("Second Derivative Error (Spectral Method)")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Dealiasing
#
# The grid has built-in dealiasing support (2/3 rule by default).
# This is important for nonlinear terms to prevent aliasing errors.

# %%
# View the dealiasing filter
dealias_filter = grid.dealias_filter()

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(grid.k, dealias_filter, ".-")
ax.set(xlabel="Wavenumber $k$", ylabel="Filter value")
ax.set_title("2/3 Dealiasing Filter")
plt.tight_layout()
plt.show()

# %%
# Apply dealiasing to a field
u_dealiased = deriv.apply_dealias(u)

print(f"Original field energy: {jnp.sum(jnp.abs(u)**2):.6f}")
print(f"Dealiased field energy: {jnp.sum(jnp.abs(u_dealiased)**2):.6f}")

# %% [markdown]
# ## Working in Spectral Space
#
# If you're doing multiple operations in Fourier space, you can work directly
# with spectral coefficients using `spectral=True`.

# %%
# Transform to spectral space
u_hat = grid.transform(u)

# Compute derivative in spectral space
du_dx_from_spectral = deriv(u_hat, order=1, spectral=True)

# Compare with physical space computation
print(f"Max difference: {jnp.abs(du_dx - du_dx_from_spectral).max():.2e}")

# %% [markdown]
# ## Summary
#
# The new API provides:
#
# 1. **FourierGrid1D**: Grid setup with automatic wavenumber computation
#    - `from_N_L()`: Create from number of points and domain length
#    - `from_N_dx()`: Create from number of points and grid spacing
#    - `from_L_dx()`: Create from domain length and grid spacing
#    - Properties: `x`, `k`, `k_dealias`, `dealias_filter()`
#    - Methods: `transform()`, `check_consistency()`
#
# 2. **SpectralDerivative1D**: Derivative operator
#    - `__call__(u, order, spectral)`: Compute n-th derivative
#    - `gradient(u)`: First derivative
#    - `laplacian(u)`: Second derivative
#    - `apply_dealias(u)`: Apply dealiasing filter
