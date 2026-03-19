# Fourier Pseudospectral Methods

Fourier pseudospectral methods are the workhorse of computational fluid dynamics for problems with **periodic boundary conditions**. They achieve **spectral (exponential) accuracy** for smooth, periodic functions and are among the most efficient methods available per degree of freedom.

---

## 1. Mathematical Setup

Consider a $2\pi$-periodic function $u : [0, 2\pi) \to \mathbb{R}$. The Fourier series representation is:

$$u(x) = \sum_{k=-\infty}^{\infty} \hat{u}_k \, e^{ikx}$$

where the Fourier coefficients are:

$$\hat{u}_k = \frac{1}{2\pi} \int_0^{2\pi} u(x) \, e^{-ikx} \, dx$$

For a domain $[0, L)$, the wavenumbers are rescaled: $k \to 2\pi k / L$.

---

## 2. The Discrete Fourier Transform (DFT)

Given $N$ uniformly spaced grid points $x_j = j \Delta x$, $j = 0, \ldots, N-1$, with $\Delta x = L/N$, the **Discrete Fourier Transform** is:

$$\hat{u}_k = \frac{1}{N} \sum_{j=0}^{N-1} u_j \, e^{-2\pi i k j / N}, \quad k = -N/2+1, \ldots, N/2$$

and the **Inverse DFT** is:

$$u_j = \sum_{k=-N/2+1}^{N/2} \hat{u}_k \, e^{2\pi i k j / N}$$

The DFT can be computed in $O(N \log N)$ time via the **Fast Fourier Transform** (FFT).

### Wavenumbers

On a grid with $N$ points and domain length $L$, the resolved wavenumbers are:

$$k_n = \frac{2\pi n}{L}, \quad n = 0, 1, \ldots, \frac{N}{2}-1, -\frac{N}{2}, \ldots, -1$$

The **Nyquist wavenumber** $k_{N/2} = \pi N / L$ is the highest wavenumber that can be resolved on a grid of $N$ points.

!!! note "FFT output ordering"
    The mathematical index range $k = -N/2+1, \ldots, N/2$ (centered at zero) differs from
    the **FFT output ordering** used by `numpy` and `jax.numpy`, which places positive frequencies
    first:

    $$n_{\text{fft}} = 0, 1, \ldots, \tfrac{N}{2}-1, -\tfrac{N}{2}, \ldots, -1$$

    In SpectralDiffX, `FourierGrid1D.k` stores the wavenumbers $k_n = 2\pi n_{\text{fft}} / L$ in
    this FFT order (as returned by `jnp.fft.fftfreq(N, d=L/N) * 2*pi`). When computing derivatives
    or inspecting the spectrum, use `jnp.fft.fftshift` to reorder to the centered representation.

---

## 3. Spectral Differentiation

The key insight of spectral differentiation is that **differentiation in physical space is equivalent to multiplication in spectral space**. Formally, if:

$$u(x) = \sum_k \hat{u}_k \, e^{ikx}$$

then:

$$\frac{d^n u}{dx^n}(x) = \sum_k (ik)^n \hat{u}_k \, e^{ikx}$$

In matrix-free notation, the $n$-th derivative is computed as:

$$\widehat{\left(\frac{d^n u}{dx^n}\right)}_k = (ik)^n \hat{u}_k$$

**Algorithm:**

1. Compute $\hat{u}_k = \text{FFT}(u_j)$
2. Multiply: $\hat{v}_k = (ik)^n \hat{u}_k$
3. Invert: $v_j = \text{IFFT}(\hat{v}_k)$

This is exact up to machine precision for band-limited functions.

---

## 4. Higher-Order Derivatives

The second, third, and fourth derivatives follow directly:

$$\frac{d^2 u}{dx^2} \longleftrightarrow (ik)^2 \hat{u}_k = -k^2 \hat{u}_k$$

$$\frac{d^3 u}{dx^3} \longleftrightarrow (ik)^3 \hat{u}_k = -ik^3 \hat{u}_k$$

$$\frac{d^4 u}{dx^4} \longleftrightarrow (ik)^4 \hat{u}_k = k^4 \hat{u}_k$$

The **Laplacian** in $d$ dimensions is:

$$\nabla^2 u \longleftrightarrow -(k_1^2 + k_2^2 + \cdots + k_d^2) \hat{u}_\mathbf{k} = -|\mathbf{k}|^2 \hat{u}_\mathbf{k}$$

### Nyquist Mode Treatment

At the Nyquist wavenumber $k = N/2$, the derivative of a real-valued function is ambiguous. The standard convention is to **zero out** the Nyquist mode before differentiating. For odd-order derivatives this is necessary to maintain a real output; for even orders it is optional but recommended.

---

## 5. Aliasing

**Aliasing** arises when the product of two band-limited signals creates energy at frequencies above the Nyquist limit. Those modes are then misidentified (aliased) as lower-frequency modes.

Consider the product $w = u \cdot v$ where $u$ and $v$ have $N/2$ non-zero modes each. The product has up to $N$ non-zero modes, but our grid can only represent $N/2$ — the excess folds back onto lower modes.

Concretely, on a grid of $N$ points, the wavenumber $k$ and $k + N$ are **indistinguishable**. A mode at $k = 3N/4$ will masquerade as $k = -N/4$.

This introduces an **aliasing error**:

$$\hat{w}_k^{\text{aliased}} = \hat{w}_k + \hat{w}_{k+N} + \hat{w}_{k-N} + \cdots$$

---

## 6. Dealiasing — The 2/3 Rule

The standard remedy is the **2/3 rule** (Orszag, 1971): before computing a nonlinear product, truncate the spectrum so that only the lower $2N/3$ modes are retained. The upper $N/3$ modes are set to zero.

**Why 2/3?** If both operands have at most $M = 2N/3$ non-zero modes, their product has at most $2M = 4N/3$ non-zero modes. Since $4N/3 \leq 2N$, the $N$-point grid resolves all product modes without aliasing.

The dealiasing filter is:

$$\mathcal{D}(\hat{u}_k) = \begin{cases} \hat{u}_k & |k| \leq k_{\max}/3 \times 2 \\ 0 & \text{otherwise} \end{cases}$$

**Padding approach:** An equivalent formulation zero-pads the signal to $3N/2$ points before the nonlinear operation and then truncates back to $N$ points afterwards. This is mathematically identical to the 2/3 rule but avoids an explicit dealiasing step.

---

## 7. Spectral Filters

Beyond dealiasing, **spectral filters** are used to control numerical instability in under-resolved simulations. Rather than a sharp cutoff, they apply a smooth roll-off to high-frequency modes.

### Exponential Filter

$$\sigma(k) = \exp\!\left(-\alpha \left(\frac{|k|}{k_{\max}}\right)^{2p}\right)$$

where $\alpha$ controls the strength and $p$ is the order. Higher $p$ gives a sharper roll-off. A typical choice is $\alpha = \ln(\epsilon_{\text{mach}}) \approx -36$ and $p = 4$.

### Raised-Cosine Filter

$$\sigma(k) = \begin{cases} 1 & |k| \leq k_c \\ \frac{1}{2}\left[1 + \cos\!\left(\pi \frac{|k| - k_c}{k_{\max} - k_c}\right)\right] & k_c < |k| \leq k_{\max} \end{cases}$$

This provides a smooth transition from passband to stopband.

---

## 8. The Helmholtz/Poisson Equation in Spectral Space

The **Helmholtz equation** is:

$$(\nabla^2 - \alpha) u = f$$

In spectral space, this becomes a **diagonal system**:

$$(-|\mathbf{k}|^2 - \alpha) \hat{u}_\mathbf{k} = \hat{f}_\mathbf{k}$$

Solving:

$$\hat{u}_\mathbf{k} = \frac{\hat{f}_\mathbf{k}}{-|\mathbf{k}|^2 - \alpha}$$

The $\mathbf{k} = 0$ mode (mean) is singular for the **Poisson equation** ($\alpha = 0$). The standard treatment is to fix $\hat{u}_0 = 0$ (zero mean solution).

**Algorithm:**

1. $\hat{f}_\mathbf{k} = \text{FFT}(f)$
2. $\hat{u}_\mathbf{k} = \hat{f}_\mathbf{k} / (-|\mathbf{k}|^2 - \alpha)$, with $\hat{u}_0 = 0$
3. $u = \text{IFFT}(\hat{u}_\mathbf{k})$

This is the most efficient spectral solver — it requires only two FFTs and a pointwise division.

!!! note "Non-periodic boundary conditions"
    The FFT-based solver above assumes **periodic BCs**. For Dirichlet ($\psi = 0$) or
    Neumann ($\partial\psi/\partial n = 0$) boundaries, the DST and DCT transforms play
    the same role. See [Spectral Transforms: DST & DCT](spectral_transforms.md) for the
    transform definitions and [Spectral Elliptic Solvers](elliptic_solvers.md) for the
    full solver algorithms. For irregular (masked) domains, see the
    [Capacitance Matrix Method](capacitance.md).

---

## 9. Numerical Convergence and Spectral Accuracy

For a smooth, periodic function $u \in C^\infty([0, L))$, the Fourier approximation error converges **faster than any algebraic power of $N$**:

$$\|u - u_N\|_\infty \leq C_s N^{-s} \quad \text{for all } s > 0$$

Equivalently, if $u$ is analytic (holomorphic in a strip), the error decays exponentially:

$$\|u - u_N\|_\infty \leq C \, e^{-\beta N}$$

This **spectral accuracy** is the defining property of spectral methods. The practical implication: for smooth problems, spectral methods achieve the same accuracy as finite difference methods of order $p$ with far fewer grid points — the ratio grows as $N^p / e^{\beta N}$.

**Comparison of convergence rates** (schematic):

| Method | Error scaling |
|--------|--------------|
| FD order 2 | $O(N^{-2})$ |
| FD order 4 | $O(N^{-4})$ |
| FD order 8 | $O(N^{-8})$ |
| Spectral (analytic $u$) | $O(e^{-\beta N})$ |

For non-smooth or discontinuous functions, spectral methods exhibit the **Gibbs phenomenon** — persistent oscillations near discontinuities — and algebraic convergence at best. In such cases, filtering, spectral mollification, or non-oscillatory reconstructions are required.
