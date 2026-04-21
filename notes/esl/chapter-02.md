# 2.3 RSS and K-NN

---

## 2.3.1 Matrix Differentiation of RSS

> Derivation of the gradient of RSS with respect to $W$, where $y = WX$

---

### Derivative of a Linear Term

$$\frac{\partial}{\partial W}(AW) = A^T$$

#### Derivation

For $f = AW$, differentiating element-wise gives:

$$\frac{\partial (AW)_{ij}}{\partial W_{kl}} = A_{ik} \delta_{jl}$$

Collecting into matrix form:

$$\frac{\partial}{\partial W}(AW) = A^T$$

Similarly:

$$\frac{\partial}{\partial W}(W^T A) = A$$

#### Example — Applied to the Linear Term of RSS

From $-2(WX)^T y = -2 X^T W^T y$:

$$\frac{\partial}{\partial W}\left(-2(WX)^T y\right) = -2Xy$$

---

### Derivative of a Quadratic Form

$$\frac{\partial}{\partial W}(W^T A W) = (A + A^T)W$$

When $A$ is **symmetric** $(A = A^T)$:

$$\frac{\partial}{\partial W}(W^T A W) = 2AW$$

#### Derivation

Treating $f = W^T A W$ as a scalar and differentiating with respect to the $i$-th element $w_i$ of $W$:

$$\frac{\partial f}{\partial w_i} = (A + A^T)_{i\cdot} W$$

Collecting into matrix form:

$$\frac{\partial}{\partial W}(W^T A W) = (A + A^T)W$$

#### Example — Applied to the Quadratic Term of RSS

From $(WX)^T(WX) = X^T W^T W X$, since $W^T W$ is symmetric:

$$\frac{\partial}{\partial W}(X^T W^T W X) = 2WXX^T$$

---

### Full Derivation of $\frac{\partial RSS}{\partial W}$

#### Expanding RSS

$$RSS = (y - WX)^T(y - WX) = y^T y - 2X^T W^T y + X^T W^T W X$$

#### Differentiating Each Term

| Term | Derivative | Rule Applied |
|---|---|---|
| $y^T y$ | $0$ | Independent of $W$ |
| $-2X^T W^T y$ | $-2Xy$ | Linear term |
| $X^T W^T W X$ | $2WXX^T$ | Quadratic form |

#### Result

$$\frac{\partial RSS}{\partial W} = 0 - 2Xy + 2WXX^T$$

$$\boxed{\frac{\partial RSS}{\partial W} = -2(y - WX)X^T}$$

#### Optimal Solution

Setting $\frac{\partial RSS}{\partial W} = 0$ and solving:

$$yX^T = WXX^T$$

Right-multiplying both sides by $(XX^T)^{-1}$:

$$\boxed{W^* = y X^T (X X^T)^{-1}}$$

This is the **Least Squares Solution**.

---

### Key Formula Summary

| Form | Derivative |
|---|---|
| $\frac{\partial}{\partial W}(AW)$ | $A^T$ |
| $\frac{\partial}{\partial W}(W^T A W)$, $A = A^T$ | $2AW$ |
| $\frac{\partial}{\partial W}\|y - WX\|^2$ | $-2(y - WX)X^T$ |
| Optimal solution $W^*$ | $yX^T(XX^T)^{-1}$ |

---

## 2.3.2 K-Nearest Neighbors (K-NN)

$$\hat{Y}(x) = \frac{1}{k} \sum_{x_i \in N_k(x)} y_i$$

$N_k(x)$ is the neighborhood of $x$, defined by the $k$ closest training points $x_i$.

### Voronoi Tessellation

Given $n$ seed points $p_1, p_2, \ldots, p_n$ in the plane, the Voronoi cell $V_i$ for each point $p_i$ is defined as:

$$V_i = \{ x \in \mathbb{R}^2 \mid d(x, p_i) \leq d(x, p_j), \; \forall j \neq i \}$$

The set of all points closer to $p_i$ than to any other seed — the cells tile the entire plane without gaps. This partitioning is the **Voronoi tessellation**.

- **Voronoi cell**: the region closest to a given seed point
- **Voronoi edge**: the boundary between two cells; equidistant from two seeds
- **Voronoi vertex**: the point where three cells meet; equidistant from three seeds

<iframe src="esl/voronoi_tessellation.html" width="100%" height="500" style="border:1px solid #30363d; border-radius:8px;"></iframe>

---

## 2.4 Statistical Decision Theory

### Expected Prediction Error (EPE)

$$\text{EPE}(f) = \mathbb{E}_{(x,y)}\left[(y - f(x))^2\right] = \int [y-f(x)]^2 \, \Pr(dx,dy)$$

The expected squared error between the predicted value $f(x)$ and the true value $y$, averaged over all possible $(x, y)$ pairs under the joint distribution $\Pr(X, Y)$.

#### Conditioning on $X$

Applying the law of total expectation:

$$
\begin{align}
\text{EPE}(f) &= \int [y - f(x)]^2 \, \Pr(dx, dy) \\
              &= \int\int [y - f(x)]^2 \, p(y \mid x) \, p(x) \, dx \, dy \\
              &= \mathbb{E}_x\left[\mathbb{E}_{y \mid x}\left[(y - f(x))^2 \mid x\right]\right]
\end{align}
$$

<details>
<summary>Law of Total Expectation (used above)</summary>

$$
\begin{align*}
\mathbb{E}_{(x,y)}[Z] &= \int\int Z \cdot p(x, y) \, dx \, dy \\
                       &= \int\int Z \cdot p(y \mid x) \, p(x) \, dx \, dy \\
                       &= \mathbb{E}_x\left[\mathbb{E}_{y \mid x}[Z \mid x]\right]
\end{align*}
$$

</details>

---

### Eq. 2.11 → 2.12 → 2.13 Derivation

#### Eq. 2.11 — Decomposition via Conditional Expectation

$$\text{EPE}(f) = E_X \left[ E_{Y|X}\left[ (Y - f(X))^2 \mid X \right] \right] \tag{2.11}$$

Using the factorization $\Pr(X,Y) = \Pr(Y \mid X) \cdot \Pr(X)$, EPE is split into a nested expectation.
Since the outer $E_X$ and inner $E_{Y|X}$ are separated, the global minimization reduces to **independent pointwise minimization at each $x$**.

#### Eq. 2.11 → 2.12 — Pointwise Minimization

To minimize $E_X[h(X)]$ where $h(x) \geq 0$, it suffices to minimize $h(x)$ independently at each $x$.

With $X = x$ fixed, $f(X)$ is no longer a function but a **single constant** $c$:

| Symbol | Role |
|---|---|
| $x$ | **Constant** — fixed input point |
| $Y$ | **Random variable** — still random after fixing $X = x$ |
| $c$ | **Variable** — the candidate prediction to optimize |

The optimization at each $x$ becomes:

$$f(x) = \underset{c}{\arg\min} \; E_{Y|X}\left[(Y - c)^2 \mid X = x\right] \tag{2.12}$$

#### Eq. 2.12 → 2.13 — Solving for the Minimum

Differentiating the objective with respect to $c$:

$$\frac{\partial}{\partial c}\; E_{Y|X}\left[(Y-c)^2 \mid X=x\right]
= E_{Y|X}\left[-2(Y-c) \mid X=x\right]
= -2\left(E[Y \mid X=x] - c\right) = 0$$

Solving for $c$:

$$\boxed{f(x) = E(Y \mid X = x)} \tag{2.13}$$

The optimal predictor at any point $x$ is the **conditional expectation of $Y$ given $X = x$**.
