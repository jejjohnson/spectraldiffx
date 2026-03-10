# Chebyshev Pseudospectral Methods

Chebyshev pseudospectral methods extend the spectral approach to **non-periodic problems** on bounded intervals, achieving the same exponential accuracy as Fourier methods without requiring periodicity.

---

## 1. Why Chebyshev?

Fourier methods require periodic boundary conditions. For problems on bounded domains with Dirichlet or Neumann boundary conditions, a naive periodic extension introduces discontinuities and destroys spectral accuracy.

Chebyshev methods solve this by using a basis of **Chebyshev polynomials** that are naturally suited to $[-1, 1]$. They inherit exponential convergence for smooth functions and offer a DCT-based transform with $O(N \log N)$ complexity.

**Key advantages over Fourier for non-periodic problems:**

- No periodic extension required
- Naturally incorporates boundary conditions
- Gauss-Lobatto nodes include the endpoints $\pm 1$
- Spectral accuracy maintained throughout

---

## 2. Chebyshev Polynomials

The Chebyshev polynomials of the first kind $T_n : [-1,1] \to [-1,1]$ are defined by the recurrence:

$$T_0(x) = 1, \quad T_1(x) = x, \quad T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)$$

or equivalently, using the trigonometric representation:

$$T_n(x) = \cos(n \arccos x), \quad x \in [-1, 1]$$

The first few polynomials are:

$$T_0 = 1, \quad T_1 = x, \quad T_2 = 2x^2 - 1, \quad T_3 = 4x^3 - 3x, \quad T_4 = 8x^4 - 8x^2 + 1$$

### Orthogonality

Chebyshev polynomials are orthogonal with respect to the weight $w(x) = (1-x^2)^{-1/2}$:

$$\int_{-1}^{1} T_m(x) T_n(x) \frac{dx}{\sqrt{1-x^2}} = \begin{cases} 0 & m \neq n \\ \pi/2 & m = n \neq 0 \\ \pi & m = n = 0 \end{cases}$$

---

## 3. Chebyshev-Gauss-Lobatto Nodes

The **Chebyshev-Gauss-Lobatto (CGL)** nodes are the extrema of $T_N$ plus the endpoints:

$$x_j = \cos\!\left(\frac{j\pi}{N}\right), \quad j = 0, 1, \ldots, N$$

These $N+1$ points lie in $[-1, 1]$ and **include both endpoints** $x_0 = +1$ and $x_N = -1$, which is essential for imposing boundary conditions.

### Node Clustering at Boundaries

The CGL nodes cluster quadratically near the boundaries:

$$x_j \approx 1 - \frac{j^2\pi^2}{2N^2} \quad (j \text{ small}), \quad \Delta x_{\min} = O(N^{-2})$$

This clustering is the Chebyshev method's mechanism for controlling the **Runge phenomenon** — the oscillations near boundaries that plague high-degree polynomial interpolation on uniform grids. The boundary clustering means that the condition number of the differentiation matrix grows as $O(N^2)$ rather than exponentially.

---

## 4. The Chebyshev Transform

A function $u$ expanded in Chebyshev polynomials:

$$u(x) = \sum_{n=0}^{N} \hat{u}_n T_n(x)$$

The **Chebyshev coefficients** can be computed via the **Discrete Cosine Transform (DCT)**. On the CGL nodes:

$$\hat{u}_n = \frac{2}{N c_n} \sum_{j=0}^{N} \frac{u(x_j)}{c_j} \cos\!\left(\frac{n j \pi}{N}\right)$$

where $c_0 = c_N = 2$, $c_j = 1$ otherwise. This is precisely the DCT-I and can be computed in $O(N \log N)$ using FFT.

---

## 5. Spectral Differentiation via the Differentiation Matrix

The Chebyshev differentiation matrix $\mathbf{D}$ maps function values at CGL nodes to derivative values:

$$\mathbf{u}' = \mathbf{D} \mathbf{u}$$

where $\mathbf{u} = [u(x_0), \ldots, u(x_N)]^T$.

The entries of $\mathbf{D}$ are given by (Trefethen, 2000):

$$D_{jk} = \begin{cases}
\displaystyle\frac{c_j}{c_k} \frac{(-1)^{j+k}}{x_j - x_k} & j \neq k \\[6pt]
\displaystyle -\frac{x_j}{2(1 - x_j^2)} & 0 < j < N \\[6pt]
\displaystyle\frac{2N^2 + 1}{6} & j = k = 0 \\[6pt]
\displaystyle-\frac{2N^2 + 1}{6} & j = k = N
\end{cases}$$

Higher-order derivatives are computed by matrix powers: $\mathbf{u}^{(n)} = \mathbf{D}^n \mathbf{u}$.

The condition number of $\mathbf{D}$ scales as $O(N^2)$ and of $\mathbf{D}^n$ as $O(N^{2n})$.

---

## 6. Chebyshev Derivative via Recurrence Relation

An alternative to the differentiation matrix uses the **recurrence relation on Chebyshev coefficients**:

$$c_{n-1} \hat{u}'_{n-1} = \hat{u}'_{n+1} + 2n \hat{u}_n, \quad n = N, N-1, \ldots, 1$$

starting from $\hat{u}'_N = 0$ and $\hat{u}'_{N+1} = 0$. This avoids matrix-vector products and runs in $O(N)$ operations after the initial DCT.

**Full algorithm:**

1. $\hat{u}_n \leftarrow \text{DCT}(u_j)$
2. Apply recurrence to get $\hat{u}'_n$
3. $u'_j \leftarrow \text{IDCT}(\hat{u}'_n)$

---

## 7. Boundary Conditions

The CGL grid includes the endpoints, so boundary conditions are imposed by **replacing the first or last row** of the differentiation matrix system.

### Dirichlet Boundary Conditions

For $u(-1) = a$ and $u(1) = b$:

Replace the rows for $j=0$ and $j=N$ in the system $\mathbf{D}^2 \mathbf{u} = \mathbf{f}$ with:

$$u_0 = b, \quad u_N = a$$

In matrix form, this sets the first and last rows of the coefficient matrix to $[1, 0, \ldots, 0]$ and $[0, \ldots, 0, 1]$ respectively, with the right-hand side set to $b$ and $a$.

### Neumann Boundary Conditions

For $u'(-1) = a$ and $u'(1) = b$:

Replace the rows with the corresponding rows from $\mathbf{D}$.

---

## 8. The Chebyshev Helmholtz Solver

The 1D Helmholtz equation with Dirichlet BCs:

$$u'' - \alpha u = f, \quad u(-1) = a, \quad u(1) = b$$

In Chebyshev pseudospectral form, this becomes a banded linear system:

$$(\mathbf{D}^2 - \alpha \mathbf{I})\mathbf{u} = \mathbf{f}$$

with rows $0$ and $N$ replaced by the boundary conditions. The system is solved once (or LU-factored for repeated solves with different $f$).

For $\alpha = 0$ this reduces to the **Chebyshev Poisson solver**.

---

## 9. Extension to 2D via Tensor Products

For 2D problems on $[-1,1]^2$, the Chebyshev basis extends naturally via **tensor products**:

$$u(x, y) = \sum_{m=0}^{M} \sum_{n=0}^{N} \hat{u}_{mn} T_m(x) T_n(y)$$

The 2D differentiation matrix for $\partial/\partial x$ is:

$$\mathbf{D}_x = \mathbf{D} \otimes \mathbf{I}_N$$

and for $\partial/\partial y$:

$$\mathbf{D}_y = \mathbf{I}_M \otimes \mathbf{D}$$

The 2D Laplacian is:

$$\nabla^2 = \mathbf{D}_x^2 + \mathbf{D}_y^2$$

For separable problems (e.g., rectangular domains with separable BCs), the 2D Helmholtz equation reduces to a sequence of 1D solves via the **matrix diagonalisation method**, reducing cost from $O(N^4)$ to $O(N^3)$.

---

## References

- Trefethen, L.N. (2000). *Spectral Methods in MATLAB*. SIAM.
- Boyd, J.P. (2001). *Chebyshev and Fourier Spectral Methods*. Dover.
- Canuto, C., et al. (2006). *Spectral Methods: Fundamentals in Single Domains*. Springer.
