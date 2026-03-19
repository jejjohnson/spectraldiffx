# Spectral Elliptic Solvers

Helmholtz, Poisson, and Laplace solvers for rectangular domains.
See the [theory page](../../theory/elliptic_solvers.md) for the mathematical background.

## Layer 0 — Pure Functions

### Periodic (FFT)

::: spectraldiffx.solve_helmholtz_fft_1d

::: spectraldiffx.solve_poisson_fft_1d

::: spectraldiffx.solve_helmholtz_fft

::: spectraldiffx.solve_poisson_fft

### Dirichlet (DST-I)

::: spectraldiffx.solve_helmholtz_dst

::: spectraldiffx.solve_poisson_dst

### Neumann (DCT-II)

::: spectraldiffx.solve_helmholtz_dct

::: spectraldiffx.solve_poisson_dct

## Layer 1 — Module Classes

### Periodic (FFT)

::: spectraldiffx.SpectralHelmholtzSolver1D

::: spectraldiffx.SpectralHelmholtzSolver2D

::: spectraldiffx.SpectralHelmholtzSolver3D

### Dirichlet (DST-I)

::: spectraldiffx.DirichletHelmholtzSolver2D

### Neumann (DCT-II)

::: spectraldiffx.NeumannHelmholtzSolver2D
