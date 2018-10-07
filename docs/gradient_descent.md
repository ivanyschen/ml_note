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

**stochastic Gradient Descent**: Randomly select and use one instance from training set at each iteration

## Partial Derivative for Frequent Used Functions
**MSE**:
$$J(\Theta) = \frac{1}{m} (\mathbf Y - \mathbf X \cdot \Theta)^2$$

$$\nabla J(\Theta) = \frac{2}{m} \mathbf X^T \cdot (\mathbf X \cdot \Theta - \mathbf Y) $$

**Cross Entropy for Binary Classes**:
