# Gradient Descent

## Concept

### Main Concept

Gradient Descent is a way to minimize a function (usually a cost function in machine learning problems).

### Intuition

Slope of a function tends to point to a minimum of a function. Hence, by searching along the direction guided by slope, we could end up at a minimum of a function.

### Algorithm

*Given a function $J(\Theta)$, find $\Theta^{\ast}$ such that $J(\Theta^{\ast})$ is a minimum of $J$.*

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


## 3 Forms in Terms of Frequency of Parameter Update

### Batch Gradient Gradient Descent

### Mini-Batch Gradient Descent

### stochastic Gradient Descent

## Other Variations

## Partial Derivative for Frequent Used Functions
