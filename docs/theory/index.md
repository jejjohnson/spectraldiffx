# Theory Overview

SpectralDiffX implements three families of pseudospectral methods. Each exploits the idea that **derivatives become algebraic operations in the appropriate transform space**, enabling spectral (exponential) accuracy for smooth functions.

---

## Methods at a Glance

| Method | Domain | Basis Functions | Transform |
|--------|--------|----------------|-----------|
| [Fourier](fourier.md) | Periodic $[0, L)$ | $e^{ikx}$ | FFT |
| [Chebyshev](chebyshev.md) | Non-periodic $[-1, 1]$ | $T_n(\cos\theta)$ | DCT |
| [Spherical Harmonics](spherical.md) | Sphere $S^2$ | $Y_\ell^m(\theta,\phi)$ | SHT |

---

## Spectral Accuracy

The hallmark of spectral methods is **spectral (exponential) convergence** for smooth functions:

$$\|u - u_N\|_\infty \sim C \, e^{-\alpha N}$$

compared to the algebraic convergence of finite difference methods ($O(h^p)$ for order $p$).

This means that doubling the resolution squares the accuracy — rather than merely doubling it.

---

## Explore the Theory

- **[Fourier Pseudospectral Methods](fourier.md)** — DFT, spectral derivatives, aliasing, dealiasing, and spectral filters for periodic domains.
- **[Chebyshev Pseudospectral Methods](chebyshev.md)** — Chebyshev polynomials, Gauss-Lobatto nodes, differentiation matrices, and boundary conditions for non-periodic problems.
- **[Spherical Harmonic Methods](spherical.md)** — Spherical harmonics, Gauss-Legendre quadrature, and spectral methods on the sphere for geophysical applications.
