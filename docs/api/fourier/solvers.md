# Spectral Elliptic Solvers

Helmholtz, Poisson, and Laplace solvers for rectangular domains.
See the [theory page](../../theory/elliptic_solvers.md) for the mathematical background.

## Layer 0 — Pure Functions

### Periodic (FFT)

#### 1D

::: spectraldiffx.solve_helmholtz_fft_1d

::: spectraldiffx.solve_poisson_fft_1d

#### 2D

::: spectraldiffx.solve_helmholtz_fft

::: spectraldiffx.solve_poisson_fft

#### 3D

::: spectraldiffx.solve_helmholtz_fft_3d

::: spectraldiffx.solve_poisson_fft_3d

### Dirichlet, Regular Grid (DST-I)

#### 1D

::: spectraldiffx.solve_helmholtz_dst1_1d

::: spectraldiffx.solve_poisson_dst1_1d

#### 2D

::: spectraldiffx.solve_helmholtz_dst

::: spectraldiffx.solve_poisson_dst

#### 3D

::: spectraldiffx.solve_helmholtz_dst1_3d

::: spectraldiffx.solve_poisson_dst1_3d

### Dirichlet, Staggered Grid (DST-II)

#### 1D

::: spectraldiffx.solve_helmholtz_dst2_1d

::: spectraldiffx.solve_poisson_dst2_1d

#### 2D

::: spectraldiffx.solve_helmholtz_dst2

::: spectraldiffx.solve_poisson_dst2

#### 3D

::: spectraldiffx.solve_helmholtz_dst2_3d

::: spectraldiffx.solve_poisson_dst2_3d

### Neumann, Regular Grid (DCT-I)

#### 1D

::: spectraldiffx.solve_helmholtz_dct1_1d

::: spectraldiffx.solve_poisson_dct1_1d

#### 2D

::: spectraldiffx.solve_helmholtz_dct1

::: spectraldiffx.solve_poisson_dct1

#### 3D

::: spectraldiffx.solve_helmholtz_dct1_3d

::: spectraldiffx.solve_poisson_dct1_3d

### Neumann, Staggered Grid (DCT-II)

#### 1D

::: spectraldiffx.solve_helmholtz_dct2_1d

::: spectraldiffx.solve_poisson_dct2_1d

#### 2D

::: spectraldiffx.solve_helmholtz_dct

::: spectraldiffx.solve_poisson_dct

#### 3D

::: spectraldiffx.solve_helmholtz_dct2_3d

::: spectraldiffx.solve_poisson_dct2_3d

## Layer 1 — Module Classes

### Periodic (FFT)

::: spectraldiffx.SpectralHelmholtzSolver1D

::: spectraldiffx.SpectralHelmholtzSolver2D

::: spectraldiffx.SpectralHelmholtzSolver3D

### Dirichlet, Regular Grid (DST-I)

::: spectraldiffx.DirichletHelmholtzSolver2D

### Dirichlet, Staggered Grid (DST-II)

::: spectraldiffx.StaggeredDirichletHelmholtzSolver2D

### Neumann, Staggered Grid (DCT-II)

::: spectraldiffx.NeumannHelmholtzSolver2D

### Neumann, Regular Grid (DCT-I)

::: spectraldiffx.RegularNeumannHelmholtzSolver2D
