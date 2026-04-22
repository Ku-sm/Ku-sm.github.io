# 3.2 Linear Regression and RSS

---

## Computing Fitted Values (Eq. 3.7)

$$\hat{\mathbf{y}} = \mathbf{X}\hat{\beta} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} \tag{3.7}$$

Define the **Hat Matrix** $H$:

$$H = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \qquad \Rightarrow \qquad \hat{\mathbf{y}} = H\mathbf{y}$$

## Properties of the Hat Matrix

| Property | Formula | Meaning |
|---|---|---|
| Symmetry | $H^T = H$ | — |
| Idempotency | $H^2 = H$ | Key property of a projection matrix |
| Projection | $\hat{\mathbf{y}} = H\mathbf{y}$ | Projects $\mathbf{y}$ onto the column space of $\mathbf{X}$ |
| Residual | $(I-H)\mathbf{y}$ | Component of $\mathbf{y}$ orthogonal to the column space of $\mathbf{X}$ |

> **Name origin:** $H$ puts a "hat (^)" on $\mathbf{y}$ to produce $\hat{\mathbf{y}}$.

---

## Variance-Covariance Matrix of $\hat{\beta}$ (Eq. 3.8)

### Assumptions

- $y_i$ are **mutually independent** with **equal variance** $\sigma^2$
- $\mathbf{x}_i$ are **fixed** (non-random)
- i.e. $\text{Var}(\mathbf{y}) = \sigma^2 I$

### Derivation

$$\hat{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

Let $A = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$, so $\hat{\beta} = A\mathbf{y}$. Applying the linear transformation variance formula $\text{Var}(A\mathbf{y}) = A\,\text{Var}(\mathbf{y})\,A^T$:

$$\text{Var}(\hat{\beta}) = A\,\sigma^2 I\, A^T = \sigma^2 A A^T$$

$$= \sigma^2 (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \cdot \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}$$

$$= \sigma^2 (\mathbf{X}^T\mathbf{X})^{-1} \underbrace{(\mathbf{X}^T\mathbf{X})(\mathbf{X}^T\mathbf{X})^{-1}}_{= I}$$

$$\boxed{\text{Var}(\hat{\beta}) = (\mathbf{X}^T\mathbf{X})^{-1}\sigma^2} \tag{3.8}$$

When $\hat{\beta}$ is a scalar this reduces to a scalar variance; when it is a vector, this gives the full covariance matrix.

### Unbiased Estimator of $\sigma^2$

$$\hat{\sigma}^2 = \frac{1}{N-p-1}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

> The denominator is $N-p-1$, not $N$, because estimating the $(p+1)$ coefficients $\beta_0, \ldots, \beta_p$ consumes $(p+1)$ degrees of freedom. This correction ensures $E(\hat{\sigma}^2) = \sigma^2$ (unbiasedness).

---

## Eq. 3.11 — Chi-Squared Distribution

### Background

Adding the Gaussian error assumption to Eq. 3.9:

$$Y = \beta_0 + \sum_{j=1}^{p} X_j\beta_j + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2) \tag{3.9}$$

Under this assumption:

$$\hat{\beta} \sim \mathcal{N}\!\left(\beta,\; (\mathbf{X}^T\mathbf{X})^{-1}\sigma^2\right) \tag{3.10}$$

$$\boxed{(N-p-1)\hat{\sigma}^2 \sim \sigma^2 \chi^2_{N-p-1}} \tag{3.11}$$

### Chi-Squared Distribution

**Definition:** The sum of squares of $k$ independent standard normal variables $Z_1, \ldots, Z_k \overset{\text{iid}}{\sim} \mathcal{N}(0,1)$:

$$\chi^2_k = Z_1^2 + Z_2^2 + \cdots + Z_k^2$$

| Property | Value |
|---|---|
| Degrees of freedom | $k$ |
| Mean | $k$ |
| Variance | $2k$ |
| Support | $[0, \infty)$ |
| Shape | Right-skewed (non-negative) |

### Why Eq. 3.11 Holds

Analyzing the residual vector:

$$\mathbf{y} - \hat{\mathbf{y}} = (I - H)\mathbf{y}$$

Since $I - H$ is idempotent with $\text{rank}(I-H) = N - p - 1$, under the Gaussian assumption the standardized residuals are independent standard normals, giving:

$$\frac{\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}{\sigma^2} = \frac{(N-p-1)\hat{\sigma}^2}{\sigma^2} \sim \chi^2_{N-p-1}$$

> **Degrees of freedom $N-p-1$:** There are $N$ residuals, but estimating $(p+1)$ coefficients imposes $(p+1)$ constraints, leaving $N - (p+1) = N-p-1$ free dimensions.

### Where Eq. 3.11 Is Used

$$z_j = \frac{\hat{\beta}_j}{\hat{\sigma}\sqrt{v_j}} \sim t_{N-p-1} \tag{3.12}$$

where $v_j$ is the $j$-th diagonal element of $(\mathbf{X}^T\mathbf{X})^{-1}$. The $t$-distribution is a normal divided by the square root of a chi-squared, so **Eq. 3.10 and 3.11 together** form the basis of the $t$-test:

$$t_k = \frac{Z}{\sqrt{\chi^2_k / k}}, \quad Z \sim \mathcal{N}(0,1), \quad \chi^2_k \perp Z$$

---

## Summary: Full Derivation Chain

$$\text{RSS}(\beta) = (\mathbf{y} - \mathbf{X}\beta)^T(\mathbf{y} - \mathbf{X}\beta)$$

$$\xrightarrow{\partial/\partial\beta = 0} \quad \mathbf{X}^T\mathbf{X}\beta = \mathbf{X}^T\mathbf{y} \tag{3.5}$$

$$\xrightarrow{(\mathbf{X}^T\mathbf{X})^{-1}} \quad \hat{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} \tag{3.6}$$

$$\xrightarrow{\hat{\mathbf{y}} = \mathbf{X}\hat{\beta}} \quad \hat{\mathbf{y}} = \underbrace{\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T}_{H}\,\mathbf{y} \tag{3.7}$$

$$\xrightarrow{\text{Var}(A\mathbf{y}) = A\,\text{Var}(\mathbf{y})\,A^T} \quad \text{Var}(\hat{\beta}) = (\mathbf{X}^T\mathbf{X})^{-1}\sigma^2 \tag{3.8}$$

$$\xrightarrow{+\;\varepsilon \sim \mathcal{N}(0,\sigma^2)} \quad (N-p-1)\hat{\sigma}^2 \sim \sigma^2\chi^2_{N-p-1} \tag{3.11}$$

---

## 3.2.2 Gauss-Markov Theorem

### Core Claim

> **The least squares estimator $\hat{\beta}$ has the smallest variance among all linear unbiased estimators.**

---

### Setup and Notation

The target is not $\beta$ itself, but any **linear combination**:

$$\theta = a^T\beta$$

For example, the prediction at a new point $x_0$, i.e. $f(x_0) = x_0^T\beta$, is of this form.

| Symbol | Meaning |
|---|---|
| $a$ | Any fixed vector |
| $\theta = a^T\beta$ | Target quantity (scalar) |
| $\hat{\theta} = a^T\hat{\beta}$ | Least squares estimator |
| $\tilde{\theta} = c^T y$ | Any other linear estimator |

<details>
<summary>Details on each parameter</summary>

$a$ specifies **which linear combination of $\beta$** we care about — fixed in advance, independent of the data.

**Examples** with $\beta = (\beta_0, \beta_1, \beta_2)^T$:

| Target | $a$ | $a^T\beta$ |
|---|---|---|
| $\beta_1$ alone | $(0, 1, 0)^T$ | $\beta_1$ |
| $\beta_1 + \beta_2$ | $(0, 1, 1)^T$ | $\beta_1 + \beta_2$ |
| Prediction at $x_0 = (1,2,3)$ | $(1, 2, 3)^T$ | $\beta_0 + 2\beta_1 + 3\beta_2$ |

$\theta = a^T\beta$ is the **true quantity we want to know** — unknowable in practice, only estimable from data.

| Component | Meaning |
|---|---|
| $\beta$ | True coefficients set by nature — unknown |
| $\theta$ | A specific linear combination of those true coefficients — also unknown |
| What we can do | **Estimate** from data |

$\hat{\theta} = a^T\hat{\beta} = a^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T y$ is a random variable — it changes with the data.

**Unbiasedness check:**

$$E[a^T\hat{\beta}] = a^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \cdot \underbrace{E[y]}_{\mathbf{X}\beta} = a^T \underbrace{(\mathbf{X}^T\mathbf{X})^{-1}(\mathbf{X}^T\mathbf{X})}_{=I}\beta = a^T\beta = \theta$$

$\tilde{\theta} = c^T y$ is any alternative estimator — the least squares estimator is a special case with $c_0 = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}a$.

| Estimator | $c$ | Description |
|---|---|---|
| Least squares $\hat{\theta}$ | $c_0 = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}a$ | Special case |
| Simple average | $(1/N, \ldots, 1/N)^T$ | Equal weights |
| Arbitrary weighted average | $(w_1, \ldots, w_N)^T$ | Any linear combination |

Gauss-Markov states: $\text{Var}(a^T\hat{\beta}) \leq \text{Var}(c^T y)$ for any unbiased $\tilde{\theta}$.

</details>

---

### Limitations of Gauss-Markov — MSE Perspective

$$\text{MSE}(\tilde{\theta}) = \text{Var}(\tilde{\theta}) + \text{Bias}^2(\tilde{\theta})$$

Gauss-Markov guarantees optimality only within the class of **unbiased** estimators. Allowing bias can yield a smaller MSE.

| Estimator | Bias | Variance | MSE |
|---|---|---|---|
| Least squares $\hat{\theta}$ | 0 | Minimum (Gauss-Markov) | Variance only |
| Ridge, Lasso, etc. | Non-zero | Can be smaller | **Can be smaller** |

---

### Unbiasedness of the Least Squares Estimator (Eq. 3.17 → 3.18)

$$\hat{\theta} = a^T\hat{\beta} = a^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T y \tag{3.17}$$

Since $E[y] = \mathbf{X}\beta$:

$$E[a^T\hat{\beta}] = a^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \cdot \mathbf{X}\beta = a^T \underbrace{(\mathbf{X}^T\mathbf{X})^{-1}(\mathbf{X}^T\mathbf{X})}_{= I} \beta = a^T\beta \tag{3.18}$$

$$\therefore \quad E[\hat{\theta}] = \theta \quad \Rightarrow \quad \text{unbiased estimator}$$

---

### Gauss-Markov Theorem (Eq. 3.19)

For any other linear unbiased estimator $\tilde{\theta} = c^T y$ satisfying $E[c^T y] = a^T\beta$:

$$\boxed{\text{Var}(a^T\hat{\beta}) \leq \text{Var}(c^T y)} \tag{3.19}$$

### Proof

**Step 1.** The unbiasedness condition $E[c^T y] = a^T\beta$ requires $c^T\mathbf{X} = a^T$ for all $\beta$.

**Step 2.** Decompose $c$ as:

$$c = \underbrace{(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T a}_{c_0} + d$$

By $(*)$, $d = c - c_0$ satisfies $d^T\mathbf{X} = \mathbf{0}$.

**Step 3.** Compute $\text{Var}(c^T y)$:

$$\text{Var}(c^T y) = \sigma^2 c^T c = \sigma^2 (c_0 + d)^T(c_0 + d) = \sigma^2(c_0^T c_0 + 2c_0^T d + d^T d)$$

**Step 4.** The cross term vanishes:

$$c_0^T d = a^T(\mathbf{X}^T\mathbf{X})^{-1} \underbrace{\mathbf{X}^T d}_{\mathbf{0}} = 0$$

**Step 5.** Final comparison:

$$\text{Var}(c^T y) = \underbrace{\sigma^2 c_0^T c_0}_{\text{Var}(a^T\hat{\beta})} + \underbrace{\sigma^2 d^T d}_{\geq\; 0} \geq \text{Var}(a^T\hat{\beta}) \quad \blacksquare$$

> Equality holds if and only if $d = 0$, i.e. $c = c_0$, i.e. $\tilde{\theta}$ is the least squares estimator.

---

### MSE Decomposition (Eq. 3.20)

$$\text{MSE}(\tilde{\theta}) = E(\tilde{\theta} - \theta)^2 = \text{Var}(\tilde{\theta}) + \underbrace{[E(\tilde{\theta}) - \theta]^2}_{\text{Bias}^2} \tag{3.20}$$

| Estimator | Bias | Variance | MSE |
|---|---|---|---|
| Least squares ($\hat{\theta}$) | 0 | Minimum (Gauss-Markov) | Variance only |
| Biased estimators (Ridge, etc.) | Non-zero | Can be smaller | **Can be smaller** |

> **Key insight:** Accepting a small bias can substantially reduce variance, yielding a lower total MSE. This is the motivation for Ridge regression, Lasso, and other **regularized estimators**.

---

### Prediction Error and MSE (Eq. 3.22)

For a new observation $Y_0 = f(x_0) + \varepsilon_0$ and prediction $\tilde{f}(x_0) = x_0^T\tilde{\beta}$:

$$E(Y_0 - \tilde{f}(x_0))^2 = E(f(x_0) + \varepsilon_0 - \tilde{f}(x_0))^2$$

Since $\varepsilon_0 \perp \tilde{f}(x_0)$, the cross term vanishes:

$$\boxed{E(Y_0 - \tilde{f}(x_0))^2 = \sigma^2 + \text{MSE}(\tilde{f}(x_0))} \tag{3.22}$$

> **Interpretation:** Prediction error = irreducible noise $\sigma^2$ + model MSE. To reduce prediction error, we must reduce $\text{MSE} = \text{Bias}^2 + \text{Variance}$.

---

### Full Chain Summary

$$\hat{\theta} = a^T\hat{\beta} \quad \xrightarrow{\text{unbiasedness}} \quad E[\hat{\theta}] = \theta \tag{3.18}$$

$$\xrightarrow{\text{Gauss-Markov}} \quad \text{Var}(\hat{\theta}) \leq \text{Var}(\tilde{\theta}) \quad \forall \text{ linear unbiased } \tilde{\theta} \tag{3.19}$$

$$\xrightarrow{\text{MSE decomposition}} \quad \text{MSE}(\tilde{\theta}) = \text{Var}(\tilde{\theta}) + \text{Bias}^2(\tilde{\theta}) \tag{3.20}$$

$$\xrightarrow{\text{prediction error}} \quad E(Y_0 - \tilde{f}(x_0))^2 = \sigma^2 + \text{MSE}(\tilde{f}(x_0)) \tag{3.22}$$

$$\Downarrow$$

$$\text{Allowing bias can reduce MSE} \quad \Rightarrow \quad \text{Motivation for Ridge, Lasso, and regularization}$$


## 3.4 Shrinkage Methods

Gauss-Markov guarantees that least squares is optimal among **unbiased** estimators. But by allowing a controlled amount of bias, we can reduce variance enough to lower total MSE. Shrinkage methods do exactly this: they penalize the size of the coefficients.

---

### 3.4.1 Ridge Regression — L2 Regularization

Ridge adds an $L_2$ penalty on the coefficients to shrink them toward zero:

$$\hat{\beta}^{\text{ridge}} = \underset{\beta}{\arg\min} \left\{ \underbrace{\sum_{i=1}^{N} \left( y_i - \beta_0 - \sum_{j=1}^{p} x_{ij}\beta_j \right)^2}_{\text{RSS}} + \underbrace{\lambda \sum_{j=1}^{p} \beta_j^2}_{L_2 \text{ penalty}} \right\} \tag{3.41}$$

The intercept $\beta_0$ is excluded from the penalty (centering the inputs first removes $\beta_0$ from the problem).

#### Closed-Form Solution (Eq. 3.44)

The $L_2$ penalty makes $\mathbf{X}^T\mathbf{X} + \lambda I$ strictly positive definite, guaranteeing invertibility even when $\mathbf{X}^T\mathbf{X}$ is singular:

$$\boxed{\hat{\beta}^{\text{ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda I)^{-1}\mathbf{X}^T y} \tag{3.44}$$

#### Effect of $\lambda$

| $\lambda$ | Effect |
|---|---|
| $\lambda = 0$ | Recovers ordinary least squares |
| $\lambda \to \infty$ | All coefficients shrink to 0 |
| $\lambda > 0$ | Introduces bias, reduces variance |

#### Ridge Shrinkage in the Orthonormal Case

When the columns of $\mathbf{X}$ are orthonormal ($\mathbf{X}^T\mathbf{X} = I$), the ridge and least squares estimates relate simply:

$$\hat{\beta}_j^{\text{ridge}} = \frac{\hat{\beta}_j^{\text{ls}}}{1 + \lambda}$$

Each coefficient is **scaled down by the same factor** $\frac{1}{1+\lambda}$ — it never reaches exactly zero.

---

### 3.4.2 The Lasso — L1 Regularization

Lasso replaces the $L_2$ penalty with an $L_1$ penalty:

$$\hat{\beta}^{\text{lasso}} = \underset{\beta}{\arg\min} \left\{ \frac{1}{2}\sum_{i=1}^{N} \left( y_i - \beta_0 - \sum_{j=1}^{p} x_{ij}\beta_j \right)^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right\} \tag{3.52}$$

Unlike Ridge, there is no closed-form solution in general — the absolute value makes the problem non-differentiable at zero.

#### Soft-Thresholding (Orthonormal Case)

In the orthonormal case the lasso solution is **soft-thresholding**:

$$\boxed{\hat{\beta}_j^{\text{lasso}} = \text{sign}(\hat{\beta}_j^{\text{ls}})\left(|\hat{\beta}_j^{\text{ls}}| - \lambda\right)_{+}}$$

where $(z)_+ = \max(z, 0)$. Coefficients smaller than $\lambda$ in magnitude are set **exactly to zero** — this is the key difference from Ridge.

#### Ridge vs. Lasso — Shrinkage Behavior

$$\underbrace{\hat{\beta}_j^{\text{ridge}} = \frac{\hat{\beta}_j^{\text{ls}}}{1 + \lambda}}_{\text{proportional shrinkage — never reaches zero}}
\qquad
\underbrace{\hat{\beta}_j^{\text{lasso}} = \text{sign}(\hat{\beta}_j^{\text{ls}})\left(|\hat{\beta}_j^{\text{ls}}| - \lambda\right)_{+}}_{\text{constant shift — reaches exactly zero}}$$

| | Ridge ($L_2$) | Lasso ($L_1$) |
|---|---|---|
| Penalty | $\lambda\sum\beta_j^2$ | $\lambda\sum|\beta_j|$ |
| Solution | Closed form | Iterative (coordinate descent, etc.) |
| Shrinkage | Proportional scaling | Soft-thresholding |
| Variable selection | No — all coefficients nonzero | Yes — sparse solutions |
| Geometry | Circular constraint region | Diamond constraint region |

---

### General $L_q$ Penalty (Bridge Regression)

Ridge and Lasso are special cases of a general penalized criterion:

$$\hat{\beta} = \underset{\beta}{\arg\min} \left\{ \sum_{i=1}^{N}\left(y_i - \beta_0 - \sum_{j=1}^{p} x_{ij}\beta_j\right)^2 + \lambda \sum_{j=1}^{p} |\beta_j|^q \right\}$$

| $q$ | Penalty | Behavior |
|---|---|---|
| $q = 2$ | Ridge ($L_2$) | Smooth, never exact zero |
| $q = 1$ | Lasso ($L_1$) | Corners at axes → sparse solutions |
| $q < 1$ | Bridge | Non-convex; even sparser, harder to optimize |

The geometry explains sparsity: for $q \leq 1$ the constraint region has **corners on the axes**, where the RSS contours tend to make contact — forcing coefficients to zero.

---

### Elastic Net

Elastic Net combines both penalties, inheriting Ridge's grouping effect and Lasso's sparsity:

$$\hat{\beta}^{\text{enet}} = \underset{\beta}{\arg\min} \left\{ \sum_{i=1}^{N}\left(y_i - \beta_0 - \sum_{j=1}^{p} x_{ij}\beta_j\right)^2 + \lambda \sum_{j=1}^{p} \left( \alpha\beta_j^2 + (1-\alpha)|\beta_j| \right) \right\}$$

| $\alpha$ | Behavior |
|---|---|
| $\alpha = 1$ | Ridge |
| $\alpha = 0$ | Lasso |
| $0 < \alpha < 1$ | Elastic Net — variable selection + correlated-predictor grouping |
