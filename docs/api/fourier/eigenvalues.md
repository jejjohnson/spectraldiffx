# Eigenvalue Helpers

1-D Laplacian eigenvalues for spectral elliptic solvers.
See the [theory page](../../theory/spectral_transforms.md#5-finite-difference-eigenvalues) for derivations.

## FD2 Eigenvalues (Finite-Difference)

Exact inverses of the 3-point FD Laplacian stencil. Take grid spacing `dx` as parameter.

### Dirichlet

::: spectraldiffx.dst1_eigenvalues

::: spectraldiffx.dst2_eigenvalues

### Neumann

::: spectraldiffx.dct1_eigenvalues

::: spectraldiffx.dct2_eigenvalues

### Periodic

::: spectraldiffx.fft_eigenvalues

### Mixed-BC

::: spectraldiffx.dst3_eigenvalues

::: spectraldiffx.dct3_eigenvalues

::: spectraldiffx.dst4_eigenvalues

::: spectraldiffx.dct4_eigenvalues

## PS Eigenvalues (Pseudo-Spectral)

Continuous Laplacian eigenvalues. Take domain length `L` as parameter.
Spectral accuracy for smooth solutions but not the exact inverse of the FD stencil.

### Dirichlet

::: spectraldiffx.dst1_eigenvalues_ps

::: spectraldiffx.dst2_eigenvalues_ps

### Neumann

::: spectraldiffx.dct1_eigenvalues_ps

::: spectraldiffx.dct2_eigenvalues_ps

### Periodic

::: spectraldiffx.fft_eigenvalues_ps

### Mixed-BC

::: spectraldiffx.dst3_eigenvalues_ps

::: spectraldiffx.dct3_eigenvalues_ps

::: spectraldiffx.dst4_eigenvalues_ps

::: spectraldiffx.dct4_eigenvalues_ps
