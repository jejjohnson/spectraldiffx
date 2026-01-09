# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python [conda env:jsom]
#     language: python
#     name: conda-env-jsom-py
# ---

# %% [markdown] id="e8566f8d-87f1-4a4e-91a9-b852d11aa389"
# ---
# title: Pseudospectral Differences (QG)
# ---

# %% id="jYHk2fFCDuLd"
# # !pip install git+https://github.com/jejjohnson/jaxsw@eman

# %% id="686728c7-a3d7-478e-84f6-75bd7c4300dc"
# import autoroot
import math

import equinox as eqx
import jax
import jax.numpy as jnp

# from jaxsw._src.fields.base import Field
import jax.random as jrandom
from jaxsw import Field, SpectralField

# from jaxsw._src.domain.base import Domain
from jaxsw._src.domain.base_v2 import Domain
from jaxsw._src.operators.functional import spectral as F_spectral
from jaxtyping import Array
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

# %load_ext autoreload
# %autoreload 2

# %% [markdown] id="e434fbf5-426f-496c-bdbb-d386008e9221"
# ### Problem Background

# %% [markdown] id="bff1c1d8-3907-47a6-97cb-ab562f2beac8"
# $$
# \partial_t q + u \partial_x q + v \partial_y q = \nu \Delta q
# $$
#
# where
# * $q=\Delta\psi$ - relative vorticity
# * $u=-\partial_u\psi$ - zonal velocity
# * $v=\partial_x\psi$ - meridonal velocity

# %% [markdown] id="1056b012-7c55-4d57-a0ba-c510d08b03ad"
# ### Domain

# %% id="f67696d1-298d-4333-abc8-4da6cb1f6389"
Nx, Ny = 256, 256
Lx, Ly = 3.0*math.pi, 3.0*math.pi
dx, dy = Lx/Nx, Ly/Ny

x_domain = Domain(xmin=0, xmax=Lx-dx, dx=dx, Nx=Nx, Lx=Lx)
y_domain = Domain(xmin=0, xmax=Ly-dy, dx=dy, Nx=Ny, Lx=Ly)

xy_domain = x_domain * y_domain

# %% id="b846af51-c3bc-400c-bb8d-11b598b66f0a"
# construct parameters
key = jrandom.PRNGKey(123)
u0: Array = jrandom.normal(key=key, shape=xy_domain.Nx)
u: Field = SpectralField(values=u0, domain=xy_domain)

# %% colab={"base_uri": "https://localhost:8080/"} id="51d716c9-be43-4c56-a211-906284141be4" outputId="fbf2c789-1261-4b3f-9b25-70556e660bd6"
u.k_vec[0].shape, u.k_vec[1].shape

# %% id="72573d39-6e88-4109-ab18-051eea6c9e6e"
import functools
import typing as tp

import einops


# %% [markdown]
# * kvec vs K
# * aliasing - where
# * real vs complex
# * scaling...

# %% [markdown]
# * u [Nx]
# * kvec [Nfx]
# * aliasing kvec [Nfx]
# * apply fft transform [Nx] --> [Nfx]
# * difference: kvec * uf [Nfx]x[Nfx]
# * inverse fft [Nfx] --> [Nx]
#
# ---

# %% [markdown]
# **Tests**
#
# * simple functions - exponential, cosine
# * 1D, 2D, 3D
# * real consistency
# * shapes - real
# * types - real
# * check operator shapes - K, L, etc
#
#
# **Examples**
#
# * 1D Example
# * 2D Example
# * Mixed! Periodic + Mixed (Spectral + FiniteDifference)
# * Spherical Harmonics - 2D Periodic (lat,lon) + 1D FD (radial)
# * Faster
#     * separate the nonlinear and linear
#     * solver - implicit (linear, division!), explicit (nonlinear)
#     * stability conditional on viscosity - stepsize larger

# %% [markdown]
# **Fields**
#
# ```python
# u: Field = ... [Nx,Ny]
# v: Field = ... [cNx, cNy]
#
# # transform to domain
# u_on_v_domain = Transformation(...)
#
# #
#
# ```

# %% [markdown]
# $$
# |k|^{2n}
# $$

# %%
# # jnp.fft.rfftfreq??

# %%
def calculate_fft_freq(Nx: int, Lx: float = 2.0 * math.pi) -> Array:
    """a helper function to generate 1D FFT frequencies

    Args:
        Nx (int): the number of points for the grid
        Lx (float): the distance for the points along the grid

    Returns:
        freq (Array): the 1D fourier frequencies
    """
    # return jnp.fft.fftfreq(n=Nx, d=Lx / (2.0 * math.pi * Nx))
    return (2 * math.pi / Lx) * jnp.fft.rfftfreq(n=Nx, d=Lx / (Nx * 2.0 * math.pi))

def fft_transform(u: Array, axis: int = -1, inverse: bool = False) -> Array:
    """the FFT transformation (forward and inverse)

    Args:
        u (Array): the input array to be transformed
        axis (int, optional): the axis to do the FFT transformation. Defaults to -1.
        scale (float, optional): the scaler value to rescale the inputs.
            Defaults to 1.0.
        inverse (bool, optional): whether to do the forward or inverse transformation.
            Defaults to False.

    Returns:
        u (Array): the transformation that maybe forward or backwards
    """
    # check if complex or real (save time!!!)
    if inverse:
        return jnp.fft.irfft(a=u, axis=axis)
    else:
        return jnp.fft.rfft(a=u, axis=axis)


def spectral_difference(
    fu: Array, k_vec: Array, axis: int = 0, derivative: int = 1
) -> Array:
    """the difference method in spectral space

    Args:
        fu (Array): the array in spectral space to take the difference
        k_vec (Array): the parameters to take the finite difference
        axis (int, optional): the axis of the multidim array, u, to
            take the finite difference. Defaults to 0.
        derivative (int, optional): the number of derivatives to take.
            Defaults to 1.

    Returns:
        dfu (Array): the finite difference method
    """

    # reshape axis
    fu = jnp.moveaxis(fu, axis, -1)

    # do difference method
    dfu = fu * (1j * k_vec) ** float(derivative)

    # re-reshape axis
    dfu = jnp.moveaxis(dfu, -1, axis)

    return dfu


# %%
# from spectraldiffx._src.operators import elliptical_operator
# from spectraldiffx._src.utils import calculate_aliasing, calculate_fft_freq

# %%
# calculate vector
k_vec = [calculate_fft_freq(Nx,Lx) for Nx, Lx in zip(xy_domain.Nx, xy_domain.Lx, strict=False)]

# calculate FFT transformation
fux = fft_transform(u[:], axis=-2, inverse=False)

# do derivative
dfudx = spectral_difference(fux, k_vec[-2], derivative=1)

# calculate inverse FFT transformation
dudx = fft_transform(dfudx, axis=-2, inverse=True)

dudx

# %%



# %%
u_ = jnp.fft.rfft(a=u[:], axis=-1)
u_.shape

# %%
k_vec[0].shape, k_vec[1].shape

# %%
ksq = elliptical_operator(u.k_vec)

# %%
# generate aliasing condition
cond = [calculate_aliasing(ikvec) for ikvec in u.k_vec]

assert cond[0].shape == u.k_vec[0].shape
assert cond[1].shape == u.k_vec[1].shape

# %%

# create operators
K = einops.repeat(u.k_vec[0], "Nx -> Nx Ny", Ny=Ny)
L = einops.repeat(u.k_vec[1], "Ny -> Nx Ny", Nx=Nx)

# create aliased operators
K_aliased = einops.repeat(u.k_vec[0], "Nx -> Nx Ny", Ny=Ny)
L_aliased = einops.repeat(u.k_vec[1], "Ny -> Nx Ny", Nx=Nx)

# %%
K, K_aliased

# %% id="b42ea636-7fd0-44ff-ba20-77e95b9ff67a"



# create dealiasing mask (for 2/3 rule)
def aliased(waves, ratio: float = 1.0/3.0) -> jax.Array:
    cond = functools.reduce(jnp.logical_or, [jnp.abs(w_i) > (1 - ratio) * jnp.abs(w_i).max() for w_i in waves])
    return cond

k_mask = aliased([K,L])

# create inversion array
invksq = 1.0 / ksq
invksq = invksq.at[0,0].set(1.0)

# %%
k_mask.shape


# %% id="59854358-0b77-442d-825d-86c664507298"
# potential vorticity
def generate_q0(domain, seed: int=42):
    rng = np.random.RandomState(seed)
    q0 = rng.randn(*domain.Nx)
    qh0 = np.fft.fftn(q0)
    qh0 = SpectralField(values=qh0, domain=domain)
    ksq = F_spectral.elliptical_operator_2D(qh0.k_vec)
    qh0 = qh0[:]
    qh0 = np.where(ksq > 10**2, 0.0, qh0)
    qh0 = np.where(ksq < 3**2, 0.0, qh0)
    q0 = np.real(np.fft.ifftn(qh0))
    return jnp.asarray(qh0), jnp.asarray(q0)

qh0, q0 = generate_q0(xy_domain, 42)
K = K.astype(qh0.dtype)
L = L.astype(qh0.dtype)
invksq = invksq.astype(qh0.dtype)
ksq = ksq.astype(qh0.dtype)
# stream function
psih0 = - invksq * qh0
psi0 = jnp.real(jnp.fft.ifftn(psih0))

# velocities
u0 = jnp.real(jnp.fft.ifftn(-1j * L * psih0))
v0 = jnp.real(jnp.fft.ifftn(1j * K * psih0))

# %% colab={"base_uri": "https://localhost:8080/", "height": 248} id="54658d96-7777-49b3-9f7d-fe59d30df8f4" outputId="37657a27-acd2-496e-8d0d-363a63eff16d"
fig, ax = plt.subplots(ncols=4, figsize=(14,3))

X, Y = xy_domain.grid_axis

pts = ax[0].pcolormesh(X, Y, q0, cmap="RdBu_r")
ax[0].set(title="Potential Vorticity")
plt.colorbar(pts)

pts = ax[1].pcolormesh(X, Y, psi0)
ax[1].set(title="Stream Function")
plt.colorbar(pts)

pts = ax[2].pcolormesh(X, Y, u0, cmap="gray")
ax[2].set(title="Zonal Velocity")
plt.colorbar(pts)

pts = ax[3].pcolormesh(X, Y, v0, cmap="gray")
ax[3].set(title="Meridional Velocity")
plt.colorbar(pts)

plt.tight_layout()
plt.show()


# %% id="bbb49ff6-08c6-4ebe-96a0-1e8d78d0fdf6"
def equation_of_motion(qh: Array, nu: float) -> Array:

    # calculate psi
    psih = - invksq * qh

    # calculate velocities
    u = jnp.real(jnp.fft.ifftn(- 1j * L * psih))
    v = jnp.real(jnp.fft.ifftn(1j * K * psih))
    # calculate advection terms: ∂ζ/∂x | ∂ζ/∂y
    dqh_dx = jnp.real(jnp.fft.ifftn(1j * K * qh))
    dqh_dy = jnp.real(jnp.fft.ifftn(1j * L * qh))

    adv_rhs = - jnp.fft.fftn(u*dqh_dx + v*dqh_dy)



    # calculate diffusion term
    diff_rhs = - nu * ksq * qh


    return adv_rhs + diff_rhs

def plot_field(u, name: str=""):

    fig, ax = plt.subplots(figsize=(4,3))

    pts = ax.pcolormesh(X, Y, np.real(u), cmap="RdBu_r")
    plt.colorbar(pts)
    ax.set(title=name)

    plt.tight_layout()
    plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 324} id="009593b8-414c-4794-972d-b0c1c72b0c1a" outputId="95acda42-242c-489d-e656-e36c54b84276"
nu = 3.0e-04
rhs = equation_of_motion(qh0, nu)
print((rhs).dtype, (qh0).dtype)

plot_field(jnp.fft.ifftn(rhs), name="rhs")

# %% colab={"base_uri": "https://localhost:8080/", "height": 307} id="f8c2f9d8-03b1-4d56-a240-611aefac01d4" outputId="dba7fb16-b339-4ef8-c429-003ae4ce5423"
fig, ax = plt.subplots(ncols=2, figsize=(7,3))

X, Y = xy_domain.grid_axis

pts = ax[0].pcolormesh(X, Y, jnp.real(jnp.fft.ifftn(qh0)), cmap="RdBu_r")
ax[0].set(title="Potential Vorticity")
plt.colorbar(pts)

pts = ax[1].pcolormesh(X, Y, jnp.real(jnp.fft.ifftn(rhs)), cmap="RdBu_r")
ax[1].set(title="RHS")
plt.colorbar(pts)

plt.tight_layout()
plt.show()

# %% id="c6137c7c-7053-4676-aca3-4a842e66bcb6"
nu=3.0e-04

dt = 0.025
t1 = 1_000
num_steps = int(t1/dt)

ts = jnp.arange(0, t1, dt)

# %% colab={"base_uri": "https://localhost:8080/", "height": 104, "referenced_widgets": ["2ba26298f8bf42128f7a6a25ad540fa3", "13daaca066554b18b30c66545451b497", "e1ea05fcaa4f420191f94695616f63ac", "6a8a371d883e4bd5875dfe99854318a0", "925d035269984785bcc321ba1781ac09", "1b0b548d3343458099c6566230290729", "ecfc2e356c1e421e9b95e40ee9f5c609", "d0177fa812ee4cbe889def252035b682", "82935afdecc54d49a9c80af90435b9c6", "28ae313237d9483bbecf86f0df02778c", "a7d1c7eee20d4f06bdbe64f07f4a45a3"]} id="89acafe3-f099-4c71-b766-06f7a7be8747" outputId="1e6693d7-2f3a-4d8f-9837-e1fba5432514"
from tqdm.autonotebook import tqdm

qh0, q = generate_q0(xy_domain, 42)

qh = jnp.copy(qh0)

fn = jax.jit(equation_of_motion, static_argnames="nu")

for i in tqdm(ts):

    rhs = fn(qh, nu)

    qh = qh + dt*rhs




# %% colab={"base_uri": "https://localhost:8080/", "height": 307} id="5d51f4f8-9427-42a4-98ef-4337d4048f98" outputId="a7569ccf-0514-456e-9ef3-2761af754452"
fig, ax = plt.subplots(ncols=2, figsize=(7,3))

X, Y = xy_domain.grid_axis

pts = ax[0].pcolormesh(X, Y, jnp.real(jnp.fft.ifftn(qh0)), cmap="RdBu_r")
ax[0].set(title="Potential Vorticity")
plt.colorbar(pts)

pts = ax[1].pcolormesh(X, Y, jnp.real(jnp.fft.ifftn(qh)), cmap="RdBu_r")
ax[1].set(title="RHS")
plt.colorbar(pts)

plt.tight_layout()
plt.show()

# %% id="9c0e3f3d-bd14-42e1-916f-568bc06148ef"
# stream function
psih = - invksq * qh
psi = jnp.real(jnp.fft.ifftn(psih))

# velocities
u = jnp.real(jnp.fft.ifftn(-1j * L * psih))
v = jnp.real(jnp.fft.ifftn(1j * K * psih))

# %% colab={"base_uri": "https://localhost:8080/", "height": 407} id="fbff2e22-0680-4ea8-9724-12ae64f8b04c" outputId="804e8cd9-ae68-4d3d-9bd1-4b3dc43686bf"
fig, ax = plt.subplots(ncols=2, figsize=(8,4))

nc = int(Nx/32)

X, Y = xy_domain.grid_axis

pts = ax[0].pcolormesh(X, Y, q[:], cmap="RdBu_r")
ax[0].quiver(X[::nc,::nc], Y[::nc,::nc], u[::nc,::nc], v[::nc,::nc])
ax[0].set(xlabel="x", ylabel="y", title="q, u, v (t=0)")

pts = ax[1].pcolormesh(X, Y, psi[:], cmap="viridis")
ax[1].quiver(X[::nc,::nc], Y[::nc,::nc], u[::nc,::nc], v[::nc,::nc])
ax[1].set(xlabel="x", ylabel="y", title=r"$\psi$, u, v (t=0)")


plt.colorbar(pts)

plt.tight_layout()
plt.show()


# %% [markdown] id="9010818e-e629-4e11-8d2d-f1d089f3e0cc"
# ### Diffrax Integration Scheme

# %% id="b718ebf0-cf7c-4d8f-bc50-c3b2b55ce1b6"
def equation_of_motion_real(q: Array, nu: float) -> Array:

    qh = jnp.fft.fftn(q)

    # calculate psi
    psih = - invksq * qh

    # calculate velocities
    u = jnp.real(jnp.fft.ifftn(- 1j * L * psih))
    v = jnp.real(jnp.fft.ifftn(1j * K * psih))
    # calculate advection terms: ∂ζ/∂x | ∂ζ/∂y
    dqh_dx = jnp.real(jnp.fft.ifftn(1j * K * qh))
    dqh_dy = jnp.real(jnp.fft.ifftn(1j * L * qh))

    adv_rhs = - jnp.fft.fftn(u*dqh_dx + v*dqh_dy)

    # calculate diffusion term
    diff_rhs = - nu * ksq * qh

    rhs = adv_rhs + diff_rhs

    rhs = jnp.real(jnp.fft.ifftn(rhs))

    return rhs


# %% colab={"base_uri": "https://localhost:8080/"} id="64b1b8e9-7e54-4d89-b3a6-d04f6aeb29d7" outputId="d7b88575-c97f-49e1-8f60-eb5c8ea7b639"
import diffrax as dfx

# %% id="e37fd98c-bf56-4926-9c99-040fa642adcc"
# Euler, Constant StepSize
# solver = dfx.Euler()
# stepsize_controller = dfx.ConstantStepSize()

solver = dfx.Dopri5()
stepsize_controller = dfx.PIDController(rtol=1e-5, atol=1e-5)

# %% id="db4a751e-7ec3-4d8f-adf6-ffc6585f5bc9"
t0, t1 = 0.0, 1_000.0 #1_000.0
t_save = np.linspace(t0, t1, 100)
saveat = dfx.SaveAt(ts=t_save)


# %% id="833d7dc7-2ff8-417d-899e-eb14adaa88d3"
class State(tp.NamedTuple):
    q: Array

def vector_field(t, state: State, args) -> State:
    nu = args
    q = state.q
    rhs = equation_of_motion_real(q=q, nu=nu)
    return State(q=rhs)


# %% id="846faeaa-f951-4073-9975-4f47a83ce3df"
# integration
sol = dfx.diffeqsolve(
    terms=dfx.ODETerm(vector_field),
    solver=solver,
    t0=t0,
    t1=t1,
    dt0=dt,
    y0=State(q=q0),
    saveat=saveat,
    args=nu,
    stepsize_controller=stepsize_controller,
    # max_steps=
)

# %% id="17b512b8-4ffc-46fa-91f0-90e4f21f994e"
# sol.ys

# %% colab={"base_uri": "https://localhost:8080/", "height": 307} id="a4b3b87f-05c8-4d36-bd23-7476d3a3270d" outputId="af147d56-4a2c-487c-b29a-3741be779e56"
fig, ax = plt.subplots(ncols=2, figsize=(7,3))

X, Y = xy_domain.grid_axis
q = sol.ys.q[-1]


pts = ax[0].pcolormesh(X, Y, q0[:], cmap="RdBu_r")
ax[0].set(title="Potential Vorticity (t=0)")
plt.colorbar(pts)

pts = ax[1].pcolormesh(X, Y, q[:], cmap="RdBu_r")
ax[1].set(title=f"Potential Vorticity (t={t1})")
plt.colorbar(pts)

plt.tight_layout()
plt.show()

# %% id="b99a30e4-e120-47fa-9e8a-070896f582c7"
q = SpectralField(q, xy_domain)
psi = F_spectral.elliptical_inversion_2D(q)
u = - F_spectral.difference_field(psi, axis=1, derivative=1, real=True)
v = F_spectral.difference_field(psi, axis=0, derivative=1, real=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 407} id="5b5eb90e-9440-4af0-9197-92070b2d35a7" outputId="27149ba3-7678-4a62-ba15-a380623ebfb5"
fig, ax = plt.subplots(ncols=2, figsize=(8,4))

nc = int(Nx/32)

X, Y = q.domain.grid_axis

pts = ax[0].pcolormesh(X, Y, q[:], cmap="RdBu_r")
ax[0].quiver(X[::nc,::nc], Y[::nc,::nc], u[::nc,::nc], v[::nc,::nc])
ax[0].set(xlabel="x", ylabel="y", title="q, u, v (t=0)")

pts = ax[1].pcolormesh(X, Y, psi[:], cmap="viridis")
ax[1].quiver(X[::nc,::nc], Y[::nc,::nc], u[::nc,::nc], v[::nc,::nc])
ax[1].set(xlabel="x", ylabel="y", title=r"$\psi$, u, v (t=0)")


plt.colorbar(pts)

plt.tight_layout()
plt.show()

# %% [markdown] id="7e23eba0-f707-486a-8fb5-7631dba53b2e"
# ### REFACTORED

# %% id="e31a166c-977a-4910-836d-a742d57407fd"
Nx, Ny = 64, 64
Lx, Ly = 2.0*math.pi, 2.0*math.pi
dx, dy = Lx/Nx, Ly/Ny

x_domain = Domain(xmin=0, xmax=Lx-dx, dx=dx, Nx=Nx, Lx=Lx)
y_domain = Domain(xmin=0, xmax=Ly-dy, dx=dy, Nx=Ny, Lx=Ly)

xy_domain = x_domain * y_domain

# %% id="f3f8bc16-5342-4a99-8f22-072ee261ade0"
qh0, q0 = generate_q0(xy_domain, 42)


# %% colab={"base_uri": "https://localhost:8080/"} id="77e59fc1-fd7b-405a-b2fa-3fa3d9cfb2ed" outputId="8c4d4137-588a-4dd8-d495-dd65346569b9"
class Params(eqx.Module):
    nu: float = eqx.static_field() # viscosity coefficient
    mu: float = eqx.static_field() # linear drag
    beta: float = eqx.static_field() # Planetary vorticity y-gradient
    nv: int = eqx.static_field() # hyperviscous order

l_unit = 5_040e3 / math.pi
t_unit = 1.2e6

params = Params(
    nu=3.0e-04, # 352 / l_unit**2 / t_unit
    mu=0.0, #1.24e-8 / t_unit**(-1),
    beta=0.0, #0.5,
    nv=1, # hyperviscous order
)
params

# %% id="03dae87d-d2d1-4ce0-96f2-13e43ba5e1bd"
from jaxsw._src.operators.constant import Constant


def equation_of_motion_field(q: Field, params: Params) -> Array:

    # ==============================
    # calculate psi (Elliptical inversion)
    # ----------------
    # q = ∇²ψ
    # q̂ = (|k|² + 1/ℓ²)ψ̂
    # ψ̂ = q / (|k|² + 1/ℓ²)
    # ==============================
    psi = F_spectral.elliptical_inversion_2D(q)

    # ###############################
    # Advection Term
    # ----------------
    # ∇⋅(uq) = u ∂x q + v ∂y q
    #        = u ikx q̂ + iky q̂
    # where
    #  u = - ∂yψ = - iky ψ̂
    #  v =   ∂xψ =   ikx ψ̂
    # ###############################

    # calculate velocities
    u = - F_spectral.difference_field(psi, axis=1)
    v = F_spectral.difference_field(psi, axis=0)

    # calculate advection terms: ∂ζ/∂x | ∂ζ/∂y
    dqh_dx = F_spectral.difference_field(q, axis=0)
    dqh_dy = F_spectral.difference_field(q, axis=1)

    rhs = - (u * dqh_dx + v * dqh_dy)

    # ###############################
    # Linear Terms
    # ###############################

    # ==============================
    # bottom drag
    # -----------
    # − μ q
    # ==============================
    if params.mu > 0.0:
        rhs += - Constant(params.mu) * q

    # ===================================
    # (hyper-) viscosity term
    # -----------------------
    # - ν|k|²ⁿ q
    # ===================================
    if params.nu > 0.0:
        rhs += - Constant(params.nu) * F_spectral.laplacian_field(q, order=params.nv)
    # ===================================
    # beta plane term
    # ---------------
    # - ∂xψ = - iβkx ψ̂
    #       = - β (ikx ψ̂)
    #       = - β v
    # =======================================
    if params.beta > 0.0:
        rhs += - Constant(params.beta) * v

    return rhs


# %% id="5d3a7fa6-9b87-4af0-963f-fdd2ea4899c0"
# # Euler, Constant StepSize
# solver = dfx.Euler()
# stepsize_controller = dfx.ConstantStepSize()

solver = dfx.Tsit5()
stepsize_controller = dfx.PIDController(rtol=1e-4, atol=1e-4)

# %% id="9d8c71e7-b9ce-4ce1-9b04-e3ea46da4528"
t0, t1 = 0.0, 1_000.0 #1_000.0
t_save = np.linspace(t0, t1, 100)
saveat = dfx.SaveAt(ts=t_save)


# %% id="c707cea0-db18-4e9b-ae14-91c9297562b8"
class State(tp.NamedTuple):
    q: Field

def vector_field(t, state: State, args) -> State:
    params = args
    q = state.q
    rhs = equation_of_motion_field(q=q, params=params)
    return State(q=rhs)



# %% colab={"base_uri": "https://localhost:8080/"} id="105bc26d-19cb-4c05-b4b4-45ef8be5ba76" outputId="090aee06-a064-4b3c-bb69-90b4df091675"
# %%time

# integration
sol = dfx.diffeqsolve(
    terms=dfx.ODETerm(vector_field),
    solver=solver,
    t0=t0,
    t1=t1,
    dt0=dt,
    y0=State(q=SpectralField(q0, xy_domain)),
    saveat=saveat,
    args=params,
    stepsize_controller=stepsize_controller,
    max_steps=6_000
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 307} id="22fb9650-affe-434f-9eac-5a97bac7be02" outputId="ace3f221-cf59-4c13-bd9e-36c9a3ec7fdf"
fig, ax = plt.subplots(ncols=2, figsize=(7,3))

X, Y = xy_domain.grid_axis
q_plot = sol.ys.q[-1]


pts = ax[0].pcolormesh(X, Y, q0[:], cmap="RdBu_r")
ax[0].set(title="Potential Vorticity (t=0)")
plt.colorbar(pts)

pts = ax[1].pcolormesh(X, Y, q_plot[:], cmap="RdBu_r")
ax[1].set(title=f"Potential Vorticity (t={t1})")
plt.colorbar(pts)

plt.tight_layout()
plt.show()

# %% id="744ff431-e928-45ca-8b58-2b87c4e6f88c"
# q = SpectralField(q, xy_domain)
psi = F_spectral.elliptical_inversion_2D(SpectralField(sol.ys.q[-1], xy_domain))
u = - F_spectral.difference_field(psi, axis=1, derivative=1, real=True)
v = F_spectral.difference_field(psi, axis=0, derivative=1, real=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 407} id="b980b459-1e19-4ffe-bf62-23ee1864e602" outputId="a40d8c20-8d4d-44cc-da9e-a56815af4d1d"
fig, ax = plt.subplots(ncols=2, figsize=(9,4))

nc = int(Nx/32)

X, Y = sol.ys.q.domain.grid_axis
# q_plot = sol.ys.q[-2]

pts = ax[0].pcolormesh(X, Y, sol.ys.q[-1][:], cmap="RdBu_r")
ax[0].quiver(X[::nc,::nc], Y[::nc,::nc], u[::nc,::nc], v[::nc,::nc])
ax[0].set(xlabel="x", ylabel="y", title=f"q, u, v  (t={t1})")
plt.colorbar(pts)

pts = ax[1].pcolormesh(X, Y, psi[:], cmap="viridis")
ax[1].quiver(X[::nc,::nc], Y[::nc,::nc], u[::nc,::nc], v[::nc,::nc])
ax[1].set(xlabel="x", ylabel="y", title=rf"$\psi$, u, v  (t={t1})")

plt.colorbar(pts)

plt.tight_layout()
plt.show()

# %% [markdown] id="3oU1r7BsMVnQ"
# # J. C. Mcwilliams (1984)

# %% id="ZTuP4t2Ybnwu"
# generate McWilliams 1984 initial condition
dl = 2.*math.pi/Ly
dk = 2.*math.pi/Lx
nl = Ny
nk = int(Nx/2+1)

ll = dl*np.append( np.arange(0.,Nx/2),
            np.arange(-Nx/2,0.) )
kk = dk*np.arange(0.,nk)

k, l = np.meshgrid(kk, ll)
wv2 = k**2+l**2
wv = jnp.sqrt(wv2)

fk = wv != 0
ckappa = np.zeros_like(wv2)
ckappa[fk] = np.sqrt( wv2[fk]*(1. + (wv2[fk]/36.)**2) )**-1

nhx,nhy = wv2.shape

Pi_hat = np.random.randn(nhx,nhy)*ckappa +1j*np.random.randn(nhx,nhy)*ckappa

Pi = np.fft.irfftn( Pi_hat )
Pi = Pi - Pi.mean()
Pi_hat = np.fft.rfftn( Pi )

def spec_var(ph):
    M = Nx**2 + Ny**2
    var_dens = 2. * np.abs(ph)**2 / M**2
    # only half of coefs [0] and [nx/2+1] due to symmetry in real fft2
    var_dens[...,0] = var_dens[...,0]/2.
    var_dens[...,-1] = var_dens[...,-1]/2.
    return var_dens.sum()

KEaux = spec_var( wv*Pi_hat )

pih = ( Pi_hat/np.sqrt(KEaux) )
qih = -wv2*pih
qi = np.fft.irfft2(qih)*0.5

# %% colab={"base_uri": "https://localhost:8080/", "height": 452} id="jvpCguiSUraG" outputId="90499a52-2ecb-4bcf-96d5-3609012bebd5"
plt.imshow(qi, cmap="RdBu_r",vmax=40,vmin=-40)
plt.colorbar()

# %% id="tzVozEFQtYu3"
# # Euler, Constant StepSize
#solver = dfx.Euler()
stepsize_controller = dfx.ConstantStepSize()

solver = dfx.Tsit5()
#stepsize_controller = dfx.PIDController(rtol=1e-4, atol=1e-4)

# %% id="ozr8ns2Mw4Jx"
dt=0.001

# %% id="eyPZrzN6tYu3"
t0, t1 = 0.0, dt*100
t_save = np.linspace(t0, t1, 10)
saveat = dfx.SaveAt(ts=t_save)

# %% colab={"base_uri": "https://localhost:8080/"} id="YXlGay_ZtYu3" outputId="e4bb6dad-9932-47f7-dd4c-5265054fa274"
# %%time

# integration
sol = dfx.diffeqsolve(
    terms=dfx.ODETerm(vector_field),
    solver=solver,
    t0=t0,
    t1=t1,
    dt0=dt,
    y0=State(q=SpectralField(jnp.asarray(qi), xy_domain)),
    saveat=saveat,
    args=params,
    stepsize_controller=stepsize_controller,
    max_steps=6_000
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 307} id="WTd16Z0itYu3" outputId="40dfed86-6e75-4c58-bbba-5bbac3957e97"
fig, ax = plt.subplots(ncols=2, figsize=(7,3))

X, Y = xy_domain.grid_axis
q_plot = sol.ys.q[-1]


pts = ax[0].pcolormesh(X, Y, qi[:], cmap="RdBu_r",vmax=40,vmin=-40)
ax[0].set(title="Potential Vorticity (t=0)")
plt.colorbar(pts)

pts = ax[1].pcolormesh(X, Y, q_plot[:], cmap="RdBu_r",vmax=40,vmin=-40)
ax[1].set(title=f"Potential Vorticity (t={t1})")
plt.colorbar(pts)

plt.tight_layout()
plt.show()

# %% id="b-NF8d12tYu4"
# q = SpectralField(q, xy_domain)
psi = F_spectral.elliptical_inversion_2D(SpectralField(sol.ys.q[-1], xy_domain))
u = - F_spectral.difference_field(psi, axis=1, derivative=1, real=True)
v = F_spectral.difference_field(psi, axis=0, derivative=1, real=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 407} id="8Inx1TA2tYu4" outputId="2125b6b5-8ba5-4a2c-e015-c71ca2873498"
fig, ax = plt.subplots(ncols=2, figsize=(9,4))

nc = int(Nx/32)

X, Y = sol.ys.q.domain.grid_axis
# q_plot = sol.ys.q[-2]

pts = ax[0].pcolormesh(X, Y, sol.ys.q[-1][:], cmap="RdBu_r")
ax[0].quiver(X[::nc,::nc], Y[::nc,::nc], u[::nc,::nc], v[::nc,::nc])
ax[0].set(xlabel="x", ylabel="y", title=f"q, u, v  (t={t1})")
plt.colorbar(pts)

pts = ax[1].pcolormesh(X, Y, psi[:], cmap="viridis")
ax[1].quiver(X[::nc,::nc], Y[::nc,::nc], u[::nc,::nc], v[::nc,::nc])
ax[1].set(xlabel="x", ylabel="y", title=rf"$\psi$, u, v  (t={t1})")

plt.colorbar(pts)

plt.tight_layout()
plt.show()


# %% id="ELbg9x694N-1"
def equation_of_motion(qh: Array, nu: float) -> Array:

    # calculate psi
    psih = - invksq * qh

    # calculate velocities
    u = jnp.real(jnp.fft.ifftn(- 1j * L * psih))
    v = jnp.real(jnp.fft.ifftn(1j * K * psih))
    # calculate advection terms: ∂ζ/∂x | ∂ζ/∂y
    dqh_dx = jnp.real(jnp.fft.ifftn(1j * K * qh))
    dqh_dy = jnp.real(jnp.fft.ifftn(1j * L * qh))

    adv_rhs = - jnp.fft.fftn(u*dqh_dx + v*dqh_dy)

    # calculate diffusion term
    diff_rhs = - nu * ksq**2 * qh


    return adv_rhs + diff_rhs


# %% id="THvB2tB44UF-"
nu=3.125e-08

dt = 0.001
t1 = 100*dt
num_steps = int(t1/dt)

ts = jnp.arange(0, t1, dt)

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["fcac11eeea6d420dbf7deb9ea731ef7c", "f804c88e7f42411a9d5133296961587c", "ca4041b3918443858036656e347b4080", "faf29c60252849b1baa4c7870838a749", "6ce3ac6ca7e5467a9aec12c3e02d780b", "69ad5f943e634667bea76349c163de77", "08b531f742d44411a25e899bf980503f", "b7750372fa054919909141d41058230b", "71ee3d021b7c41f489c76a04d1e3c5bc", "ba1ec53013d24e50a97d058351a63b4f", "606950266a29467e8f6f5af4bc79406a"]} id="R6kzrvvV4UGM" outputId="7d2671a3-390a-468b-9634-2892179d0b81"
qh0 = jnp.asarray(np.fft.fftn(qi))
qh = jnp.copy(qh0)

fn = jax.jit(equation_of_motion, static_argnames="nu")

for i in tqdm(ts):

    rhs = fn(qh, nu)

    qh = qh + dt*rhs

# %% colab={"base_uri": "https://localhost:8080/", "height": 307} id="JJ0ET0sW4UGM" outputId="00600d58-15be-4b24-c694-2ac2aeb949b0"
fig, ax = plt.subplots(ncols=2, figsize=(7,3))

X, Y = xy_domain.grid_axis

pts = ax[0].pcolormesh(X, Y, jnp.real(jnp.fft.ifftn(qh0)), cmap="RdBu_r")
ax[0].set(title="Potential Vorticity")
plt.colorbar(pts)

pts = ax[1].pcolormesh(X, Y, jnp.real(jnp.fft.ifftn(qh)), cmap="RdBu_r")
ax[1].set(title="RHS")
plt.colorbar(pts)

plt.tight_layout()
plt.show()

# %% id="122xnjGp4qws"
