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
<summary>Law of Total Expectation</summary>

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

---

## 2.5 Local Methods in High Dimensions

### Median Nearest-Neighbor Distance (Eq. 2.24)

For $N$ uniformly distributed points in a $p$-dimensional unit hypercube, the expected distance to the nearest neighbor is:

$$d(p, N) = \left(1 - \left(\frac{1}{2}\right)^{1/N}\right)^{1/p}$$

#### Derivation

$$
\begin{align*}
&\text{Step 1: Probability that one point falls within distance } d \\
&\quad P(\text{one point} \leq d) = d^p \\[6pt]
&\text{Step 2: Probability that all } N \text{ points fall outside } d \\
&\quad P(\text{nearest neighbor} > d) = (1 - d^p)^N \\[6pt]
&\text{Step 3: Median condition} \\
&\quad (1 - d^p)^N = \frac{1}{2} \\[6pt]
&\text{Step 4: Solve for } d \\
&\quad d^p = 1 - \left(\frac{1}{2}\right)^{1/N}
\implies d(p,N) = \left(1 - \left(\frac{1}{2}\right)^{1/N}\right)^{1/p}
\end{align*}
$$

As $p$ increases, $d(p, N) \to 1$ — the nearest neighbor moves toward the boundary regardless of how many training points are available. This is the **curse of dimensionality**.

<details>
<summary>d(p, N) Table</summary>

| $p$ \ $N$ | 10 | 50 | 100 | 500 | 1000 | 10000 |
|:---------:|:------:|:------:|:------:|:------:|:------:|:------:|
| **1**   | 0.0670 | 0.0138 | 0.0069 | 0.0014 | 0.0007 | 0.0001 |
| **2**   | 0.2588 | 0.1173 | 0.0831 | 0.0372 | 0.0263 | 0.0083 |
| **3**   | 0.4061 | 0.2397 | 0.1904 | 0.1115 | 0.0885 | 0.0411 |
| **5**   | 0.5823 | 0.4244 | 0.3697 | 0.2681 | 0.2334 | 0.1473 |
| **10**  | 0.7631 | 0.6515 | 0.6080 | 0.5178 | 0.4831 | 0.3838 |
| **20**  | 0.8736 | 0.8071 | 0.7798 | 0.7196 | 0.6951 | 0.6195 |
| **50**  | 0.9474 | 0.9179 | 0.9053 | 0.8767 | 0.8646 | 0.8257 |
| **100** | 0.9733 | 0.9581 | 0.9515 | 0.9363 | 0.9298 | 0.9087 |
| **200** | 0.9866 | 0.9788 | 0.9754 | 0.9676 | 0.9643 | 0.9532 |
| **500** | 0.9946 | 0.9915 | 0.9901 | 0.9869 | 0.9856 | 0.9810 |

> Values close to **1.0** mean most data points lie near the **boundary** — the curse of dimensionality in action.

---

### Key Observations

| Scenario | $d(p, N)$ | Meaning |
|----------|:---------:|---------|
| $p=1$, $N=500$ | 0.0014 | Closest point is essentially at the origin — very local |
| $p=10$, $N=500$ | 0.5178 | Closest point is already **halfway to the boundary** |
| $p=10$, $N=10$ | 0.7631 | With few data, closest point is 76% of the way out |
| $p=50$, $N=500$ | 0.8767 | Even with 500 points, data is near the edge |
| $p=500$, $N=10000$ | 0.9810 | Boundary effect dominates regardless of $N$ |

</details>

---

## 2.8 Classes of Restricted Estimators

### Nadaraya-Watson Kernel Estimator

A weighted average of nearby $y$ values — points closer to $x_0$ receive higher weight.

$$\hat{f}(x_0) = \frac{\sum_{i=1}^{N} K_\lambda(x_0, x_i)\, y_i}{\sum_{i=1}^{N} K_\lambda(x_0, x_i)} = \sum_{i=1}^{N} w_i(x_0)\, y_i$$

where the weights are:

$$w_i(x_0) = \frac{K_\lambda(x_0, x_i)}{\sum_{j=1}^{N} K_\lambda(x_0, x_j)}, \qquad w_i \geq 0, \quad \sum_{i=1}^{N} w_i = 1$$

The prediction at $x_0$ is a **smooth local average** of the $y$ values in the neighborhood of $x_0$.

#### Gaussian Kernel (Eq. 2.40)

$$K_\lambda(x_0, x) = \frac{1}{\lambda}\exp\left(-\frac{\|x - x_0\|^2}{2\lambda}\right)$$

#### Derivation from Local Constant Fitting (Eq. 2.41 from 2.42)

Minimizing the locally weighted sum of squares with $f_\theta = \theta_0$:

$$\frac{\partial}{\partial \theta_0} \sum_{i=1}^{N} K_\lambda(x_0, x_i)(y_i - \theta_0)^2 = 0
\implies \theta_0 = \frac{\sum_i K_\lambda(x_0, x_i)\,y_i}{\sum_i K_\lambda(x_0, x_i)}$$

---

### Basis Functions

A set of **building-block functions** used to represent a complex target function.

#### Analogy — RGB Color Mixing

$$\text{any color} = r \cdot \text{Red} + g \cdot \text{Green} + b \cdot \text{Blue}$$

Red, Green, and Blue are the **basis**. Any color can be expressed as a linear combination of these three.

#### Connection to Eq. 2.43

$$f(x) = \sum_{m=1}^{M} \theta_m h_m(x) = \theta_1 h_1(x) + \theta_2 h_2(x) + \cdots + \theta_M h_M(x) \tag{2.43}$$

| Symbol | RGB Analogy | Meaning |
|---|---|---|
| $h_m(x)$ | Red, Green, Blue | Pre-defined **basis functions** (ingredients) |
| $\theta_m$ | $r, g, b$ values | **How much** of each ingredient to mix (coefficients) |
| $f(x)$ | Final color | The resulting **prediction function** |

#### Example — Cubic Spline Basis (2 knots: $t_1, t_2$)

| $h_m(x)$ | Role | Active Region |
|---|---|---|
| $1$ | Global constant | All $x$ |
| $x$ | Global slope | All $x$ |
| $x^2$ | Global curvature | All $x$ |
| $x^3$ | Global cubic term | All $x$ |
| $(x - t_1)^3_+$ | Additional curvature after $t_1$ | $x \geq t_1$ only |
| $(x - t_2)^3_+$ | Additional curvature after $t_2$ | $x \geq t_2$ only |

The first four basis functions are **global**; the last two are **local** — they activate only past their respective knots.

$$\boxed{f(x) = \underbrace{\theta_1 \cdot 1}_{\text{constant}} + \underbrace{\theta_2 \cdot x}_{\text{slope}} + \underbrace{\theta_3 x^2 + \theta_4 x^3}_{\text{curvature}} + \underbrace{\theta_5 (x-t_1)^3_+ + \theta_6 (x-t_2)^3_+}_{\text{active past knots only}}}$$

---

### Polynomial Splines

A spline partitions the domain into intervals at **knot** points, fits a separate polynomial on each interval, and enforces **continuity conditions** at the knots.

$$\text{Spline} = \text{Piecewise Polynomial} + \text{Continuity at Knots}$$
