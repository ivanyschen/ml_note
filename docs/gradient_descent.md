# Gradient Descent

## Concept

### Main Concept

Gradient Descent is a way to minimize a function (usually a cost function in machine learning problems).

### Intuition

Slope of a function tends to point to a minimum of a function. Hence, by searching along the direction guided by slope, we could end up at a minimum of a function.

### Algorithm

*Given a function $J(\Theta)$, find $\Theta^{\ast}$ such that $J(\Theta^{\ast})$ is a minimum of $J$. (The most ideal case is to find the global minimum)*

**Vector Form**

Repeat until converge

- Calculate $\nabla J(\Theta)$
- Update $\Theta$ with
    $$\Theta := \Theta - \alpha \nabla J(\Theta) \text{, where }\alpha \text{ is learning rate}$$

**Scaler Form**

Repeat until converge

- for i = 1 to n
    - Calculate $\frac{\partial J(\Theta)}{\partial \theta_i}$
    - Update $\theta_i$ with
            $$\theta_i := \theta_i - \alpha \frac{\partial J(\Theta)}{\partial \theta_i} \text{, where }\alpha \text{ is learning rate}$$


## Three Forms in Terms of Size of Data Used in Each Update
**Batch Gradient Gradient Descent**: Using full training set at each iteration.

**Mini-Batch Gradient Descent**: Randomly divide training set into small batches and use one small batch at each iteration.

**Stochastic Gradient Descent**: Randomly select and use one instance from training set at each iteration

## Partial Derivative for Frequent Used Functions
**MSE**:
$$J(\Theta) = \frac{1}{m} (\mathbf Y - \mathbf X \cdot \Theta)^2$$

$$\nabla J(\Theta) = \frac{2}{m} \mathbf X^T \cdot (\mathbf X \cdot \Theta - \mathbf Y) $$

**Softmax Function**:
$$p_i = \frac{e^{a_i}}{\sum_{k=1}^N e_k^a}$$

$$\frac{\partial p_i}{\partial a_j} =
\begin{cases}
  p_i(1-p_i) \text{ if } i = j,\\    
  -p_jp_i \text{ if } i \ne j.
\end{cases}$$

**Cross Entropy**:
$$J = -\sum_{k=1}^K y_k^{(i)} \log(\hat y_k^{(i)})$$

$$\frac{\partial J}{\partial a_i} = - \sum y_k \frac{1}{p_k} \cdot \frac{\partial p_k}{\partial a_i}$$

**Input -> Softmax -> Cross Entropy**:

$$\frac{\partial J}{\partial o_i} = p_i - y_i$$


## Gradient Explode/Vanish Problem

### Problem

### Solution
