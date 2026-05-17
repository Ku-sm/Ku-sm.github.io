# 4 Linear Methods for Classification

---

## 4.1 Introduction

### Linear Classifier for Each Class

For class $k$, the linear model predicts:

$$\hat{f}_k(x) = \hat{\beta}_{k0} + \hat{\beta}_k^T x$$

### Decision Boundary Between Two Classes

The decision boundary between classes $k$ and $l$ is where $\hat{f}_k(x) = \hat{f}_l(x)$:

$$\{ x : (\hat{\beta}_{k0} - \hat{\beta}_{l0}) + (\hat{\beta}_k - \hat{\beta}_l)^T x = 0 \}$$

This is an **affine set** (hyperplane) in $\mathbb{R}^p$ — the set of all points equidistant from the two class centroids, forming a linear decision boundary. The function $\delta_k(x)$ that defines each region is called the **discriminant function**.

---

### Indicator Response Matrix

When the class label $\mathcal{G} \in \{1, 2, \ldots, K\}$, each class is encoded as an indicator variable:

$$Y_k = \begin{cases} 1 & \text{if } G = k \\ 0 & \text{otherwise} \end{cases}$$

Stacking all $K$ indicators gives an $(N \times K)$ indicator matrix $\mathbf{Y}$:

$$\mathbf{Y} = \underbrace{\begin{bmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ \vdots & & & \vdots \\ 0 & 0 & 0 & 1 \end{bmatrix}}_{K \text{ columns}}$$

Each row has exactly one entry equal to 1, corresponding to the true class of that observation.

---

### Classification Rule

$$\hat{G}(x) = \underset{k \in \mathcal{G}}{\arg\max}\; \hat{f}_k(x)$$

Equivalently, using squared Euclidean distance to class target vectors $t_k$ (the $k$-th column of $I_K$):

$$\hat{G}(x) = \underset{k}{\arg\min}\; \|\hat{f}(x) - t_k\|^2$$

#### Masking Problem (K ≥ 3)

When $K \geq 3$, the rigid structure of linear regression can cause one class to be **masked** — its predicted region is entirely swallowed by neighboring classes. To avoid this, polynomial terms of degree up to $K-1$ in $x$ may be needed.

---

## 4.3 Linear Discriminant Analysis

### Motivation

From Section 2.4, optimal classification requires knowing the **posterior probability** $\Pr(G \mid X)$. By Bayes' theorem:

$$\Pr(G = k \mid X = x) = \frac{f_k(x)\,\pi_k}{\displaystyle\sum_{l=1}^{K} f_l(x)\,\pi_l}$$

where $f_k(x)$ is the class-conditional density for class $k$, and $\pi_k$ is the prior probability with $\sum_{k=1}^K \pi_k = 1$.

---

### Gaussian Assumption with Shared Covariance

Assume each class density is multivariate Gaussian with a **common covariance matrix** $\Sigma_k = \Sigma$ for all $k$:

$$f_k(x) = \frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}} \exp\!\left(-\frac{1}{2}(x - \mu_k)^T \Sigma^{-1}(x - \mu_k)\right)$$

---

### What Is Log-Odds?

The **log-odds** (logit) measures how much more likely class $k$ is compared to class $l$, on a log scale:

$$\text{log-odds}_{k,l}(x) = \log \frac{\Pr(G=k \mid X=x)}{\Pr(G=l \mid X=x)}$$

| Value | Meaning |
|---|---|
| $> 0$ | Class $k$ is more probable than class $l$ |
| $= 0$ | Equal probability — the decision boundary |
| $< 0$ | Class $l$ is more probable than class $k$ |

Using the log scale has two advantages over raw probabilities:
1. **Unbounded** — probabilities live in $[0,1]$ but log-odds range over $(-\infty, +\infty)$, making them easier to model linearly.
2. **Ratio form** — the normalizing denominator $\sum_l f_l(x)\pi_l$ cancels when taking the ratio of two posteriors, so we only need the class-conditional densities and priors.

Expanding via Bayes' theorem:

$$\log \frac{\Pr(G=k \mid X=x)}{\Pr(G=l \mid X=x)} = \log \frac{f_k(x)}{f_l(x)} + \log \frac{\pi_k}{\pi_l}$$

The decision boundary is exactly where this equals zero.

---

### Gaussian Assumption and Why the Boundary Is Linear

Assume each class density is multivariate Gaussian with a **common covariance matrix** $\Sigma_k = \Sigma$ for all $k$:

$$f_k(x) = \frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}} \exp\!\left(-\frac{1}{2}(x - \mu_k)^T \Sigma^{-1}(x - \mu_k)\right)$$

Taking the log-ratio of two class densities, the normalizing constants $(2\pi)^{p/2}|\Sigma|^{1/2}$ cancel (same for both classes under shared $\Sigma$):

$$\log \frac{f_k(x)}{f_l(x)} = -\frac{1}{2}(x-\mu_k)^T\Sigma^{-1}(x-\mu_k) + \frac{1}{2}(x-\mu_l)^T\Sigma^{-1}(x-\mu_l)$$

#### Expanding the Quadratic Terms

Each exponent expands as:

$$(x - \mu_k)^T\Sigma^{-1}(x-\mu_k) = x^T\Sigma^{-1}x - 2x^T\Sigma^{-1}\mu_k + \mu_k^T\Sigma^{-1}\mu_k$$

Taking the difference:

$$\log \frac{f_k(x)}{f_l(x)} = \underbrace{-\frac{1}{2}x^T\Sigma^{-1}x + \frac{1}{2}x^T\Sigma^{-1}x}_{= 0,\;\text{quadratic terms cancel}} + x^T\Sigma^{-1}(\mu_k-\mu_l) - \frac{1}{2}(\mu_k^T\Sigma^{-1}\mu_k - \mu_l^T\Sigma^{-1}\mu_l)$$

The $x^T\Sigma^{-1}x$ term is **identical** for both classes and cancels. What remains is:

$$= x^T\Sigma^{-1}(\mu_k - \mu_l) - \frac{1}{2}(\mu_k + \mu_l)^T\Sigma^{-1}(\mu_k - \mu_l)$$

Adding the prior term gives the full log-odds:

$$\boxed{\log \frac{\Pr(G=k \mid X=x)}{\Pr(G=l \mid X=x)} = \log\frac{\pi_k}{\pi_l} - \frac{1}{2}(\mu_k + \mu_l)^T \Sigma^{-1}(\mu_k - \mu_l) + x^T \Sigma^{-1}(\mu_k - \mu_l)}$$

> **Only the last term depends on $x$, and it is linear** — so the decision boundary (where this equals zero) is a hyperplane.

---

### Linear Discriminant Function (Eq. 4.10)

The log-odds linearity means we can assign class labels by maximizing a **discriminant function** $\delta_k(x)$:

$$\boxed{\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2}\mu_k^T \Sigma^{-1} \mu_k + \log \pi_k}$$

$$\hat{G}(x) = \underset{k}{\arg\max}\; \delta_k(x)$$

| Term | Role |
|---|---|
| $x^T \Sigma^{-1} \mu_k$ | Similarity of $x$ to class centroid $\mu_k$ (Mahalanobis inner product) |
| $-\frac{1}{2}\mu_k^T \Sigma^{-1} \mu_k$ | Offset correcting for the centroid's distance from the origin |
| $\log \pi_k$ | Prior log-probability of class $k$ |

The decision boundary between classes $k$ and $l$ is where $\delta_k(x) = \delta_l(x)$, which is an affine hyperplane in $\mathbb{R}^p$.

---

### LDA vs. QDA

The quadratic cancellation only holds under **shared covariance**. If each class has its own $\Sigma_k$, the $x^T\Sigma_k^{-1}x$ terms differ and do not cancel, leaving a quadratic term in $x$.

| Method | Covariance assumption | $x^T(\cdot)x$ cancels? | Decision boundary |
|---|---|---|---|
| LDA | $\Sigma_k = \Sigma$ (shared) | Yes | Linear (hyperplane) |
| QDA | $\Sigma_k$ free per class | No | Quadratic (conic section) |




## 4.4 Logistic Regression

### Motivation

Logistic regression models the **posterior probabilities** of $K$ classes via linear functions of $x$, while ensuring the probabilities are valid: they sum to 1 and lie in $[0, 1]$.

---

### The Model

Using class $K$ as the reference category, the log-odds of each class $k$ relative to class $K$ are modeled as linear functions of $x$:

$$\log \frac{\Pr(G=k \mid X=x)}{\Pr(G=K \mid X=x)} = \beta_{k0} + \beta_k^T x, \quad k = 1, \ldots, K-1$$

This gives $K-1$ log-odds equations, each with its own intercept $\beta_{k0}$ and coefficient vector $\beta_k \in \mathbb{R}^p$.

---

### Posterior Probabilities

Exponentiating and normalizing the log-odds gives the posterior probabilities explicitly:

$$\Pr(G = k \mid X = x) = \frac{\exp(\beta_{k0} + \beta_k^T x)}{1 + \sum_{l=1}^{K-1} \exp(\beta_{l0} + \beta_l^T x)}, \quad k = 1, \ldots, K-1$$

$$\Pr(G = K \mid X = x) = \frac{1}{1 + \sum_{l=1}^{K-1} \exp(\beta_{l0} + \beta_l^T x)}$$

By construction these $K$ probabilities sum to 1 and each lies in $(0, 1)$.

---

### Binary Case ($K = 2$)

With two classes, there is a single log-odds equation. Writing $p(x) = \Pr(G=1 \mid X=x)$:

$$\log \frac{p(x)}{1 - p(x)} = \beta_0 + \beta^T x$$

Solving for $p(x)$ yields the **sigmoid** (logistic) function:

$$p(x) = \frac{1}{1 + e^{-(\beta_0 + \beta^T x)}} = \sigma(\beta_0 + \beta^T x)$$

The decision boundary $p(x) = 0.5$ is where $\beta_0 + \beta^T x = 0$ — a hyperplane in $\mathbb{R}^p$.

---

### Fitting via Maximum Likelihood

Let $\theta = \{\beta_{10}, \beta_1, \ldots, \beta_{(K-1)0}, \beta_{K-1}\}$ denote all parameters. For $N$ observations, the log-likelihood is:

$$\ell(\theta) = \sum_{i=1}^{N} \log \Pr(G = g_i \mid X = x_i; \theta)$$

where $g_i$ is the true class of observation $i$. There is no closed-form solution, so $\ell(\theta)$ is maximized numerically.

#### Newton–Raphson (IRLS) for the Binary Case

The score equations $\partial \ell / \partial \beta = 0$ have no closed form. Newton's method iterates:

$$\beta^{\text{new}} = \beta^{\text{old}} - \left(\frac{\partial^2 \ell}{\partial \beta \, \partial \beta^T}\right)^{-1} \frac{\partial \ell}{\partial \beta}$$

In matrix form with $\mathbf{X}$ the $(N \times (p+1))$ design matrix, $\mathbf{p}$ the vector of fitted probabilities, and $\mathbf{W} = \operatorname{diag}(p_i(1-p_i))$:

$$\frac{\partial \ell}{\partial \beta} = \mathbf{X}^T(\mathbf{y} - \mathbf{p}), \qquad \frac{\partial^2 \ell}{\partial \beta \, \partial \beta^T} = -\mathbf{X}^T \mathbf{W} \mathbf{X}$$

Each Newton step solves a **weighted least squares** problem:

$$\beta^{\text{new}} = (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{z}, \qquad \mathbf{z} = \mathbf{X}\beta^{\text{old}} + \mathbf{W}^{-1}(\mathbf{y} - \mathbf{p})$$

where $\mathbf{z}$ is called the **adjusted response**. This procedure is known as **Iteratively Reweighted Least Squares (IRLS)**.

---

### LDA vs. Logistic Regression

Both produce linear decision boundaries, but differ in how parameters are estimated.

| | LDA | Logistic Regression |
|---|---|---|
| Assumption | Gaussian class densities, shared $\Sigma$ | No distributional assumption on $X$ |
| Fitting | Parameters from class means, covariance, priors | Maximum likelihood via IRLS |
| Extra info used | Marginal density $\Pr(X)$ | Only $\Pr(G \mid X)$ |
| Robustness | Sensitive to outliers and non-Gaussian features | More robust, safer in practice |

> LDA uses more structure (the full joint distribution) so it can be more efficient when assumptions hold. Logistic regression makes fewer assumptions and is generally preferred when the Gaussian assumption is doubtful.



## 4.5 Separating Hyperplanes

### Motivation

Instead of modeling class probabilities, we can directly construct a **linear decision boundary** that explicitly separates the data into different classes as cleanly as possible.

---

### The Hyperplane $L$

Define the affine hyperplane $L$ by:

$$f(x) = \beta_0 + \beta^T x = 0$$

In $\mathbb{R}^2$ this is a line; in $\mathbb{R}^p$ it is a $(p-1)$-dimensional hyperplane. Let $\beta^* = \beta / \|\beta\|$ be the unit normal to $L$.

---

### Geometric Properties

**Property 1 — $\beta$ is normal to $L$.**

For any two points $x_1, x_2$ on $L$:

$$\beta^T(x_1 - x_2) = 0$$

so $\beta$ (and $\beta^*$) is orthogonal to every vector lying in $L$.

**Property 2 — Points on $L$ satisfy a fixed inner product.**

For any point $x_0 \in L$:

$$\beta^T x_0 = -\beta_0$$

**Property 3 — Signed distance from any point $x$ to $L$ (Eq. 4.40).**

$$\boxed{\beta^{*T} x + \frac{\beta_0}{\|\beta\|} = \frac{f(x)}{\|\beta\|}}$$

The sign indicates which side of $L$ the point lies on:

| Sign of $f(x)$ | Location |
|---|---|
| $f(x) > 0$ | Positive side of $L$ |
| $f(x) = 0$ | On $L$ (the boundary) |
| $f(x) < 0$ | Negative side of $L$ |

The magnitude $|f(x)| / \|\beta\|$ is the Euclidean distance from $x$ to $L$.

---

### Classification Rule

Assign class labels based on which side of $L$ a point falls on:

$$\hat{G}(x) = \operatorname{sign}(\beta_0 + \beta^T x)$$

The goal is to find $\beta_0, \beta$ such that $f(x_i)$ has the correct sign for all (or most) training points $x_i$.

