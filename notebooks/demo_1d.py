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
# # 1D Fourier Pseudospectral Differentiation
#
# ## What is pseudospectral differentiation?
#
# The pseudospectral method computes derivatives of periodic functions with
# **spectral accuracy** -- errors decrease exponentially with resolution for
# smooth functions, rather than algebraically as with finite-difference methods.
#
# The key idea is remarkably simple: differentiation in physical space becomes
# **multiplication by `ik`** in Fourier space. Given a periodic function
# $u(x)$ expanded as:
#
# $$
# u(x) = \sum_k \hat{u}_k \, e^{ikx}
# $$
#
# its derivative is:
#
# $$
# \frac{du}{dx} = \sum_k (ik) \, \hat{u}_k \, e^{ikx}
# $$
#
# The algorithm is:
# 1. **FFT**: Transform $u$ to spectral space to get $\hat{u}_k$
# 2. **Multiply**: $\widehat{du/dx}_k = ik \cdot \hat{u}_k$
# 3. **IFFT**: Transform back to physical space
#
# This notebook demonstrates 1D pseudospectral differentiation using the
# `spectraldiffx` library, covering grid setup, derivative computation,
# convergence properties, dealiasing, and spectral filtering.

# %%
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from spectraldiffx import FourierGrid1D, SpectralDerivative1D, SpectralFilter1D

matplotlib.use("Agg")

jax.config.update("jax_enable_x64", True)
sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)

IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "images" / "demo_1d"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Grid Setup
#
# We create a `FourierGrid1D` with $N$ uniformly-spaced points on the periodic
# domain $[0, L)$. The grid does **not** include the right endpoint because the
# function is periodic -- the value at $x = L$ equals the value at $x = 0$.
#
# ```
# Periodic grid (N=8 points on [0, L)):
#
# x:  |--o--o--o--o--o--o--o--o--|
#     0  dx 2dx              L-dx  L
#                                  ^
#                            (not included, wraps to x=0)
#
# dx = L / N
# ```
#
# The discrete wavenumbers are:
#
# $$
# k_n = \frac{2\pi n}{L}, \quad n = -N/2, \ldots, N/2 - 1
# $$
#
# The maximum resolvable wavenumber (the **Nyquist frequency**) is
# $k_{\max} = \pi N / L = \pi / dx$.

# %%
# Grid parameters
N = 64   # Number of grid points (power of 2 for efficient FFT)
L = 4.0  # Domain length [0, L)

# Create the Fourier grid with 2/3-rule dealiasing
grid = FourierGrid1D.from_N_L(N=N, L=L, dealias="2/3")

# Physical grid points
x = grid.x  # x: [N] -- uniformly spaced points on [0, L)

# Wavenumber array
k = grid.k  # k: [N] -- discrete wavenumbers (FFT-ordered)

print(f"N = {grid.N} grid points")
print(f"L = {grid.L} domain length")
print(f"dx = {grid.dx:.6f} grid spacing")
print(f"x shape: {x.shape}")
print(f"k shape: {k.shape}")
print(f"Nyquist wavenumber: k_max = pi/dx = {jnp.pi / grid.dx:.4f}")

# %% [markdown]
# ## 2. Test Function
#
# We use a single Fourier mode as our test function:
#
# $$
# u(x) = \cos\!\left(m \cdot \frac{2\pi}{L} x\right)
# $$
#
# where $m$ is the mode number. This is an **exact eigenfunction** of the
# derivative operator, which makes it an ideal test case: the analytical
# derivatives are known exactly.
#
# **Analytical derivatives:**
#
# $$
# \frac{du}{dx} = -m \frac{2\pi}{L} \sin\!\left(m \frac{2\pi}{L} x\right)
# $$
#
# $$
# \frac{d^2u}{dx^2} = -\left(m \frac{2\pi}{L}\right)^2 \cos\!\left(m \frac{2\pi}{L} x\right)
# $$

# %%
# Mode number -- must satisfy m < N/2 to be resolved on the grid
m = 3  # A low mode that is well-resolved by our N=64 grid

# Effective wavenumber for this mode
km = m * 2 * jnp.pi / L
print(f"Mode number m = {m}")
print(f"Effective wavenumber km = m * 2pi/L = {km:.4f}")
print(f"Nyquist wavenumber = {jnp.pi / grid.dx:.4f}")
print(f"Ratio km / k_nyquist = {km / (jnp.pi / grid.dx):.4f} (must be < 1)")

# %%
# Evaluate the test function on the grid
u = jnp.cos(km * x)  # u: [N] -- field values at grid points

# Analytical derivatives (closed-form)
dudx_analytical = -km * jnp.sin(km * x)        # du/dx: [N]
d2udx2_analytical = -(km**2) * jnp.cos(km * x)  # d2u/dx2: [N]

# Dense grid for smooth plotting
x_plot = jnp.linspace(0, L, 512, endpoint=False)
u_plot = jnp.cos(km * x_plot)
dudx_plot = -km * jnp.sin(km * x_plot)
d2udx2_plot = -(km**2) * jnp.cos(km * x_plot)

print(f"u shape: {u.shape}")
print(f"u range: [{float(u.min()):.4f}, {float(u.max()):.4f}]")

# %%
fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(x_plot, u_plot, "-", color="black", label=r"$u(x) = \cos(k_m x)$")
ax.scatter(x, u, color="red", marker="*", s=30, label=f"Grid points (N={N})", zorder=3)
ax.set(xlabel=r"$x$", ylabel=r"$u(x)$", title=f"Test Function: mode $m={m}$ on $[0, {L})$")
ax.legend()
plt.tight_layout()
fig.savefig(IMG_DIR / "test_function.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Test function and grid points](../../images/demo_1d/test_function.png)

# %% [markdown]
# ## 3. First Derivative
#
# The spectral differentiation formula for the first derivative is:
#
# $$
# \widehat{u'}_k = ik \cdot \hat{u}_k
# $$
#
# In code, this is a three-step process:
# 1. `u_hat = FFT(u)` -- transform to spectral space
# 2. `du_hat = 1j * k * u_hat` -- multiply by $ik$
# 3. `du_dx = IFFT(du_hat).real` -- transform back
#
# The `SpectralDerivative1D` operator encapsulates this.

# %%
# Create the derivative operator
deriv = SpectralDerivative1D(grid=grid)

# Compute first derivative: du/dx
dudx_spectral = deriv(u, order=1)  # dudx_spectral: [N]

# Equivalently, using the .gradient() method:
dudx_grad = deriv.gradient(u)  # dudx_grad: [N]

print(f"dudx_spectral shape: {dudx_spectral.shape}")
print(f"gradient() vs __call__(order=1) difference: {jnp.abs(dudx_spectral - dudx_grad).max():.2e}")

# %%
fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(x_plot, dudx_plot, "-", color="black", label=r"$du/dx$ (analytical)")
ax.scatter(x, dudx_spectral, color="green", marker=".", s=40, label="SpectralDerivative1D", zorder=3)
ax.scatter(x, dudx_analytical, color="red", marker="x", s=20, label="Closed-form on grid", zorder=3)
ax.set(xlabel=r"$x$", ylabel=r"$du/dx$", title="First Derivative Comparison")
ax.legend()
plt.tight_layout()
fig.savefig(IMG_DIR / "first_derivative.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![First derivative comparison](../../images/demo_1d/first_derivative.png)

# %%
# Pointwise error -- should be at machine precision (~1e-14 for float64)
error_1st = jnp.abs(dudx_spectral - dudx_analytical)
print(f"Max error in first derivative: {float(error_1st.max()):.2e}")
print(f"Mean error in first derivative: {float(error_1st.mean()):.2e}")
print(f"(Machine epsilon for float64: {jnp.finfo(jnp.float64).eps:.2e})")

# %%
fig, ax = plt.subplots(figsize=(8, 3))

ax.semilogy(x, error_1st, "g.-", label="Pointwise error")
ax.axhline(jnp.finfo(jnp.float64).eps, color="gray", linestyle="--", label="Machine epsilon")
ax.set(xlabel=r"$x$", ylabel=r"$|du/dx_{\mathrm{spectral}} - du/dx_{\mathrm{exact}}|$",
       title="First Derivative Error (Spectral Method)")
ax.legend()
plt.tight_layout()
fig.savefig(IMG_DIR / "first_derivative_error.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![First derivative error](../../images/demo_1d/first_derivative_error.png)

# %% [markdown]
# ## 4. Second Derivative
#
# For the $n$-th derivative, the spectral formula generalizes to:
#
# $$
# \widehat{u^{(n)}}_k = (ik)^n \cdot \hat{u}_k
# $$
#
# For the second derivative ($n=2$): $(ik)^2 = -k^2$, so:
#
# $$
# \widehat{u''}_k = -k^2 \cdot \hat{u}_k
# $$
#
# We can compute this with `deriv(u, order=2)` or equivalently `deriv.laplacian(u)`
# (since the Laplacian reduces to $d^2/dx^2$ in 1D).

# %%
# Compute second derivative -- two equivalent ways
d2udx2_spectral = deriv(u, order=2)     # d2udx2_spectral: [N]
d2udx2_lap = deriv.laplacian(u)          # d2udx2_lap: [N] -- via laplacian method

print(f"d2udx2_spectral shape: {d2udx2_spectral.shape}")
print(f"laplacian() vs __call__(order=2) difference: {jnp.abs(d2udx2_spectral - d2udx2_lap).max():.2e}")

# %%
fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(x_plot, d2udx2_plot, "-", color="black", label=r"$d^2u/dx^2$ (analytical)")
ax.scatter(x, d2udx2_spectral, color="green", marker=".", s=40, label="SpectralDerivative1D", zorder=3)
ax.scatter(x, d2udx2_analytical, color="red", marker="x", s=20, label="Closed-form on grid", zorder=3)
ax.set(xlabel=r"$x$", ylabel=r"$d^2u/dx^2$", title="Second Derivative Comparison")
ax.legend()
plt.tight_layout()
fig.savefig(IMG_DIR / "second_derivative.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Second derivative comparison](../../images/demo_1d/second_derivative.png)

# %%
error_2nd = jnp.abs(d2udx2_spectral - d2udx2_analytical)
print(f"Max error in second derivative: {float(error_2nd.max()):.2e}")
print(f"Mean error in second derivative: {float(error_2nd.mean()):.2e}")

# %%
fig, ax = plt.subplots(figsize=(8, 3))

ax.semilogy(x, error_2nd, "g.-", label="Pointwise error")
ax.axhline(jnp.finfo(jnp.float64).eps, color="gray", linestyle="--", label="Machine epsilon")
ax.set(xlabel=r"$x$", ylabel=r"$|d^2u/dx^2_{\mathrm{spectral}} - d^2u/dx^2_{\mathrm{exact}}|$",
       title="Second Derivative Error (Spectral Method)")
ax.legend()
plt.tight_layout()
fig.savefig(IMG_DIR / "second_derivative_error.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Second derivative error](../../images/demo_1d/second_derivative_error.png)

# %% [markdown]
# ## 5. Convergence with Mode Number
#
# A key strength of spectral methods is that they differentiate **all resolved
# modes** with the same spectral accuracy. However, once the mode number $m$
# exceeds the **aliasing limit** at $N/3$ (when using the 2/3 dealiasing rule)
# or the Nyquist limit at $N/2$, the method breaks down.
#
# We sweep over mode numbers $m = 1, 2, \ldots, N/2 - 1$ and measure the
# maximum error in the first derivative for each mode.
#
# Expected behavior:
# - $m < N/3$: machine-precision errors (fully resolved and dealiased)
# - $N/3 \leq m < N/2$: errors grow because the dealiasing filter zeros
#   these modes
# - $m \geq N/2$: aliased (not representable on the grid)

# %%
max_mode = N // 2 - 1  # Highest non-aliased mode
modes = jnp.arange(1, max_mode + 1)
errors = []

for mi in range(1, max_mode + 1):
    kmi = mi * 2 * jnp.pi / L
    u_test = jnp.cos(kmi * x)
    dudx_test = deriv(u_test, order=1)
    dudx_exact = -kmi * jnp.sin(kmi * x)
    errors.append(float(jnp.abs(dudx_test - dudx_exact).max()))

errors = np.array(errors)

# %%
fig, ax = plt.subplots(figsize=(8, 4))

ax.semilogy(modes, errors, "b.-", markersize=4, label="Max error per mode")
ax.axvline(N / 3, color="red", linestyle="--", label=f"$N/3 = {N/3:.0f}$ (dealias cutoff)")
ax.axvline(N / 2, color="orange", linestyle="--", label=f"$N/2 = {N/2:.0f}$ (Nyquist)")
ax.axhline(jnp.finfo(jnp.float64).eps, color="gray", linestyle=":", label="Machine epsilon")
ax.set(xlabel="Mode number $m$", ylabel="Max error in $du/dx$",
       title="Convergence: Error vs Mode Number")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(IMG_DIR / "convergence_modes.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Convergence with mode number](../../images/demo_1d/convergence_modes.png)
#
# For modes below the $N/3$ dealiasing cutoff, the error is at machine
# precision. Above this cutoff, the dealiasing filter zeros out the mode,
# causing $O(1)$ errors. This is by design -- those modes would produce
# aliasing errors in nonlinear computations.

# %% [markdown]
# ## 6. Dealiasing
#
# When computing nonlinear terms like $u \cdot du/dx$, the product of two
# fields with wavenumbers $k_1$ and $k_2$ generates a mode at $k_1 + k_2$.
# If $k_1 + k_2 > k_{\max}$, this mode **aliases** back into the resolved
# spectrum, corrupting the solution.
#
# The **2/3 rule** prevents this by zeroing out the top 1/3 of the spectrum
# before computing products. The dealiasing filter is:
#
# $$
# D_k = \begin{cases} 1 & \text{if } |k| \leq \frac{2}{3} k_{\max} \\
# 0 & \text{otherwise} \end{cases}
# $$
#
# This ensures that $k_1 + k_2 \leq \frac{2}{3}k_{\max} + \frac{2}{3}k_{\max}
# = \frac{4}{3}k_{\max}$, which after aliasing folds back to at most
# $\frac{4}{3}k_{\max} - k_{\max} = \frac{1}{3}k_{\max}$ -- safely within
# the zeroed-out region.

# %%
# The dealias filter from the grid
dealias_mask = grid.dealias_filter()  # dealias_mask: [N] -- binary mask in spectral space

print(f"Dealias filter shape: {dealias_mask.shape}")
print(f"Number of retained modes: {int(dealias_mask.sum())} / {N}")
print(f"Fraction retained: {float(dealias_mask.sum()) / N:.4f}")

# %%
# Apply dealiasing to a field
u_dealiased = deriv.apply_dealias(u)  # u_dealiased: [N]

print(f"Original field energy:   {float(jnp.sum(jnp.abs(u)**2)):.6f}")
print(f"Dealiased field energy:  {float(jnp.sum(jnp.abs(u_dealiased)**2)):.6f}")
print(f"Energy ratio: {float(jnp.sum(jnp.abs(u_dealiased)**2) / jnp.sum(jnp.abs(u)**2)):.6f}")
print("(Ratio ~ 1.0 because mode m=3 is well below the N/3 cutoff)")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3))

# Left: dealias filter in spectral space
k_sorted_idx = jnp.argsort(k)
axes[0].plot(k[k_sorted_idx], dealias_mask[k_sorted_idx], "b.-", markersize=3)
axes[0].axvline(-2 / 3 * jnp.abs(k).max(), color="red", linestyle="--", alpha=0.7,
                label=r"$\pm \frac{2}{3} k_{\max}$")
axes[0].axvline(2 / 3 * jnp.abs(k).max(), color="red", linestyle="--", alpha=0.7)
axes[0].set(xlabel="Wavenumber $k$", ylabel="Filter value",
            title="2/3 Dealiasing Filter (Spectral Space)")
axes[0].legend(fontsize=9)

# Right: effect on a high-mode signal
m_high = N // 3 + 2  # A mode above the dealias cutoff
km_high = m_high * 2 * jnp.pi / L
u_high = jnp.cos(km_high * x)
u_high_dealiased = deriv.apply_dealias(u_high)

axes[1].plot(x, u_high, "b-", label=f"Original (m={m_high})")
axes[1].plot(x, u_high_dealiased, "r--", label=f"After dealiasing")
axes[1].set(xlabel=r"$x$", ylabel=r"$u(x)$",
            title=f"Dealiasing Removes Mode m={m_high} > N/3={N//3}")
axes[1].legend(fontsize=9)

plt.tight_layout()
fig.savefig(IMG_DIR / "dealiasing.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Dealiasing filter and effect](../../images/demo_1d/dealiasing.png)

# %% [markdown]
# ## 7. Spectral Filter
#
# For time-stepping PDEs, we often need a smoother alternative to the sharp
# dealiasing cutoff. The **exponential filter** provides a smooth rolloff:
#
# $$
# F(k) = \exp\!\left(-\alpha \left(\frac{|k|}{k_{\max}}\right)^p\right)
# $$
#
# where:
# - $\alpha$ controls the damping strength at $k_{\max}$ ($F(k_{\max}) = e^{-\alpha}$)
# - $p$ (the power/order) controls the sharpness of the transition
#
# For large $p$, the filter is nearly 1 for low wavenumbers and drops sharply
# near $k_{\max}$ -- preserving resolved scales while damping grid-scale noise.

# %%
# Create the spectral filter
filt = SpectralFilter1D(grid=grid)

# Apply exponential filter with default parameters (alpha=36, power=16)
u_filtered = filt.exponential_filter(u, alpha=36.0, power=16)

print(f"u_filtered shape: {u_filtered.shape}")
print(f"Max difference from original: {float(jnp.abs(u - u_filtered).max()):.2e}")
print("(Small for low modes because the filter is near 1 for |k| << k_max)")

# %%
# Visualize the filter profile for different parameter choices
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# Left: vary the power parameter
k_max = float(jnp.abs(k).max())
k_normalized = jnp.abs(k[k_sorted_idx]) / k_max

for p in [4, 8, 16, 32]:
    F = jnp.exp(-36.0 * (k_normalized) ** p)
    axes[0].plot(k[k_sorted_idx], F, label=f"power={p}")

axes[0].set(xlabel="Wavenumber $k$", ylabel="Filter $F(k)$",
            title=r"Exponential Filter: $\exp(-36 (|k|/k_{\max})^p)$")
axes[0].legend(fontsize=8)

# Right: vary the alpha parameter (with power=16)
for a in [6, 18, 36, 72]:
    F = jnp.exp(-a * (k_normalized) ** 16)
    axes[1].plot(k[k_sorted_idx], F, label=fr"$\alpha$={a}")

axes[1].set(xlabel="Wavenumber $k$", ylabel="Filter $F(k)$",
            title=r"Exponential Filter: $\exp(-\alpha (|k|/k_{\max})^{16})$")
axes[1].legend(fontsize=8)

plt.tight_layout()
fig.savefig(IMG_DIR / "spectral_filter.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Spectral filter profiles](../../images/demo_1d/spectral_filter.png)

# %% [markdown]
# ## 8. Working in Spectral Space
#
# When performing multiple operations in Fourier space (e.g., inside a
# time-stepping loop), it is more efficient to stay in spectral space and
# avoid redundant FFT/IFFT pairs. The `spectral=True` flag lets you pass
# Fourier coefficients directly.

# %%
# Transform to spectral space once
u_hat = grid.transform(u)  # u_hat: [N] -- complex Fourier coefficients
print(f"u_hat shape: {u_hat.shape}")
print(f"u_hat dtype: {u_hat.dtype}")

# Compute derivative from spectral coefficients
dudx_from_spectral = deriv(u_hat, order=1, spectral=True)

# Verify it matches the physical-space computation
print(f"Max difference (spectral vs physical input): {float(jnp.abs(dudx_spectral - dudx_from_spectral).max()):.2e}")

# %% [markdown]
# ## 9. Parseval's Theorem and Normalization
#
# When using spectral transforms, a natural question is: does the transform
# preserve the "energy" (squared norm) of the signal?
#
# **Parseval's theorem** states that for an orthonormal transform $T$:
#
# $$
# \|T(x)\|^2 = \|x\|^2
# $$
#
# The DCT-II with `norm='ortho'` is orthonormal, so Parseval holds exactly.
# With `norm=None` (the default unnormalized convention), the spectral
# coefficients are scaled by $2N$, so $\|Y\|^2 / \|x\|^2 \propto N$.

# %%
from spectraldiffx import dct

Ns = [8, 16, 32, 64, 128, 256]
ratios_none = []
ratios_ortho = []

for N_p in Ns:
    x_p = jnp.sin(jnp.linspace(0.1, 3.0, N_p)) + 0.5
    energy_x = float(jnp.sum(x_p**2))

    y_none = dct(x_p, type=2, norm=None)
    ratios_none.append(float(jnp.sum(y_none**2)) / energy_x)

    y_ortho = dct(x_p, type=2, norm="ortho")
    ratios_ortho.append(float(jnp.sum(y_ortho**2)) / energy_x)

    print(f"  N={N_p:4d}:  ratio(None)={ratios_none[-1]:8.2f},  ratio(ortho)={ratios_ortho[-1]:.6f}")

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(Ns, ratios_none, "s-", label=r"norm=None: $\|Y\|^2 / \|x\|^2$", color="C3")
ax.plot(Ns, ratios_ortho, "o-", label=r'norm="ortho": $\|Y\|^2 / \|x\|^2$', color="C0")
ax.axhline(1.0, color="k", ls="--", lw=0.8, label="Parseval (ratio = 1)")
ax.set_xlabel("$N$ (vector length)")
ax.set_ylabel(r"$\|DCT(x)\|^2 \,/\, \|x\|^2$")
ax.set_title("DCT-II Energy Ratio: effect of normalization")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(ratios_none) * 1.15)
plt.tight_layout()
fig.savefig(IMG_DIR / "parseval_ortho.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Parseval's theorem](../../images/demo_1d/parseval_ortho.png)
#
# With `norm='ortho'`, the energy ratio is exactly 1 for all $N$ — Parseval's
# theorem holds. With `norm=None`, the ratio grows linearly with $N$ because
# the unnormalized DCT-II scales coefficients by a factor related to $2N$.
#
# **Takeaway**: Use `norm='ortho'` when energy conservation matters (e.g.,
# in Parseval-based spectral energy calculations). Use `norm=None` (the
# default) when you need the raw transform matching SciPy conventions.

# %% [markdown]
# ## Summary
#
# This tutorial demonstrated the core features of the `spectraldiffx` 1D API:
#
# | Class / Method | Purpose |
# |---|---|
# | `FourierGrid1D.from_N_L(N, L, dealias)` | Create periodic grid with wavenumbers |
# | `grid.x`, `grid.k`, `grid.k_dealias` | Physical points, wavenumbers, dealiased wavenumbers |
# | `grid.dealias_filter()` | Binary mask for 2/3-rule dealiasing |
# | `grid.transform(u, inverse=False)` | FFT / IFFT |
# | `SpectralDerivative1D(grid)` | Derivative operator |
# | `deriv(u, order=n)` | $n$-th derivative |
# | `deriv.gradient(u)` | First derivative (alias for `order=1`) |
# | `deriv.laplacian(u)` | Second derivative ($-k^2 \hat{u}$) |
# | `deriv.apply_dealias(u)` | Apply 2/3-rule dealiasing |
# | `SpectralFilter1D(grid)` | Smoothing filters |
# | `filt.exponential_filter(u, alpha, power)` | Exponential spectral filter |
# | `filt.hyperviscosity(u, nu_hyper, dt, power)` | Hyperviscous damping |
#
# Key takeaways:
# - Spectral derivatives achieve **machine-precision** accuracy for resolved modes.
# - The 2/3 dealiasing rule prevents aliasing in nonlinear computations by zeroing the top 1/3 of the spectrum.
# - Exponential filters provide smooth spectral damping for time-stepping stability.
# - The `spectral=True` flag avoids redundant FFT/IFFT pairs when working in Fourier space.
