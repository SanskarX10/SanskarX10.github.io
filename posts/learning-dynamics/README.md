#### 2025-06-17
# Learning dynamics of LLM finetuning

[paper link](https://arxiv.org/abs/2407.10490)

*   analyzing the step-wise decomposition of how influence accumulates among different potential responses
*   propose a hypothetical explanation of why specific types of hallucination are strengthened after finetuning
*   for each tokens prediction, -ve gradient will push down models prediction on almost all possible output labels

$$
\begin{align*}
\Delta\theta &\triangleq \theta^{t+1} - \theta^t = -\eta \cdot \nabla_{\theta} L(f_{\theta}(\mathbf{x}_{u}), \mathbf{y}_{u}) \\[1em]
\Delta f(\mathbf{x}_{o}) &\triangleq f_{\theta^{t+1}}(\mathbf{x}_{o}) - f_{\theta^t}(\mathbf{x}_{o})
\end{align*}
$$

After an GD update on $\mathbf{x}_{u}$, how does the model's prediction on $\mathbf{x}_{o}$ change?
for instance, model learns to map $\mathbf{x}$ to preds $\mathbf{y} = \{y_1, \dots, y_L\}$ for all.

### Part 1: Supervised Learning Setup

**Model Definition:**
$$ h_{\theta}: \mathcal{X} \to \mathbb{R}^{V \times L} $$

**Logits Generation:**
$$ \mathbf{z} = h_{\theta}(\mathbf{x}) \in \mathbb{R}^{V \times L} $$

**Output Prediction:**
$$ \mathbf{y} = \{y_1, \dots, y_L\} \in \mathcal{V}^L $$
where $\mathcal{V}$ is the vocabulary of size $V$.

**Probability Distribution:**
$$ \pi_{\theta}(\mathbf{y} \mid \mathbf{x}) = \text{Softmax}(\mathbf{z}) \text{ (applied column-wise)} $$

**Model Confidence Metric:**
$$ \log \pi_{\theta}(\mathbf{y} \mid \mathbf{x}) $$

### Part 2: Per-step Influence Decomposition

**Quantity of Interest:**
$$ \Delta \log \pi^t(\mathbf{y} \mid \mathbf{x}_{o}) $$

**Definition:**
$$ \Delta \log \pi^t(\mathbf{y} \mid \mathbf{x}_{o}) \triangleq \log \pi_{\theta_{t+1}}(\mathbf{y} \mid \mathbf{x}_{o}) - \log \pi_{\theta_t}(\mathbf{y} \mid \mathbf{x}_{o}) $$
Where:
*   $\theta_t$ is the model parameters at training step $t$.
*   $\theta_{t+1}$ is the model parameters after one training step.
*   $\mathbf{x}_{o}$ is an "observation" input (data point whose confidence is being tracked).

observation is change in log probability after one step of training
one step learning dynamic can be written as
$$
\Delta \log \pi^t(y \mid \mathbf{x}_{o}) = -\eta A^t(\mathbf{x}_{o}) K^t(\mathbf{x}_{o}, \mathbf{x}_{u}) G^t(\mathbf{x}_{u}, \mathbf{y}_{u}) + O(\eta^2 |\nabla_{\theta} z(\mathbf{x}_{u})|^{2}_{\text{op}})
$$

*   **G** is the energy term, gradient of loss wrt logits for ($\mathbf{x}_{u}$, $\mathbf{y}_{u}$) if model is very wrong on a class, then this will be large
*   **K** is the similarity term, or Empirical Neural Tangent Kernel {appendix below}, how aligned gradients for $\mathbf{x}_{o}$ and $\mathbf{x}_{u}$ are
*   **A** is the gradient of log- probability wrt to logits for $\mathbf{x}_{o}$ (observable example)

change in confidence on $\mathbf{x}_{o}$ is proportional to error on training example $\mathbf{x}_{u}$, multiplied by how similar the model thinks $\mathbf{x}_{o}$ and $\mathbf{x}_{u}$ are which the acts on the current state of the $\mathbf{x}_{o}$ prediction

## Neural Tangent Kernel

- linear map: transform vectors while keeping the relationship
- linear maps in 2d : parallel lines stay parallel , even spaces are preserved , origin is fixed
- for example: two points , (1, 0) , (0, 1) if we linearly map them , (1.33 , -0.73) , (1.17, 0.75)
- they can be represented in a matrix , which captures essence of linear map
- areas scale by same factor , that factor is determinant
- since the linear map is a matrix, determinant can be calculated
- in 1d, linear maps just scale by an integer
- for example, $f(x) = x^2$ , numbers of number lines will be 3 times (approximate ) apart from each other , [3] will be the Jacobian matrix for it and derivate is $f'(a)$
-*but this depends on a, both Jacobian and derivative can change with a's position in 2d, for example $f(x, y) = (x^2 - y^2 , 3xy)$ and Jacobian will depend on a, b*
- Jacobian is matrix represents best linear map approximation of f near (a, b) and Jacobian matrix is a matrix composed of the first-order partial derivatives of a multivariable function.
- JM will have as many rows as vector components and num columns will match number of variables

Find the Jacobian matrix at the point (1,2) of the following function:
$$
f(x, y) = (x^4 + 3y^2 x,\ 5y^2 - 2xy + 1)
$$
First of all, we calculate all the first-order partial derivatives of the function:
$$
\begin{align*}
\frac{\partial f_{1}}{\partial x} &= 4x^3 + 3y^2 \\
\frac{\partial f_{1}}{\partial y} &= 6yx \\
\frac{\partial f_{2}}{\partial x} &= -2y \\
\frac{\partial f_{2}}{\partial y} &= 10y - 2x
\end{align*}
$$
Now we apply the formula of the Jacobian matrix. In this case, the function has two variables and two vector components, so the Jacobian matrix will be a $2 \times 2$ matrix:
$$
J_{f}(x, y) =
\begin{pmatrix}
4x^3 + 3y^2 & 6yx \\
-2y & 10y - 2x
\end{pmatrix}
$$
Once we have found the expression of the Jacobian matrix, we evaluate it at point (1,2):
$$
J_{f}(1,2) =
\begin{pmatrix}
4 \cdot 1^3 + 3 \cdot 2^2 & 6 \cdot 2 \cdot 1 \\
-2 \cdot 2 & 10 \cdot 2 - 2 \cdot 1
\end{pmatrix}
$$
And finally, we perform the operations:
$$
J_{f}(1,2) =
\begin{pmatrix}
16 & 12 \\
-4 & 18
\end{pmatrix}
$$

a kernel is a similarity function between two data points , some kernels can be decomposed into two feature maps
*to be continued..*
