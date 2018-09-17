# Linear Models

## Regression

### Linear Regression

#### Model:    
$$\begin{align}
\hat y & = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n \nonumber\\
 & = \theta^T \cdot \mathbf x \nonumber
\end{align}$$

#### Cost Function: Mean Squared Error   
$$MSE(\theta) = \frac{1}{m} \sum_{i=1}^m (\theta^T \cdot \mathbf x^{(i)} - y^{(i)})^2$$

#### Training:    

* **The Normal Equation:**
$$\hat \theta = (\mathbf X^T \cdot \mathbf X)^{-1} \cdot \mathbf X^T \cdot \mathbf Y$$

* **Gradient Descent:**

    Refer to [Gradient Descent](/gradient_descent/)

* The Normal Equation v.s. Gradient Descent    

<table>
    <tbody>
        <tr>
            <th></th>
            <th>Normal Equation</th>
            <th>Gradient Descent</th>
        </tr>
        <tr>
            <td>Pros</td>
            <td>
                <ul>
                    <li>No need to choose $\alpha$</li>
                    <li>No need to iterate</li>
                </ul>
            </td>
            <td>
            <ul>
                <li>Work well even $n$ is large</li>
            </ul>
            </td>
        </tr>
            <td>Cons</td>
            <td>
                <ul>
                    <li>Need to calculate $(\mathbf X^T \mathbf X)^{-1}$</li>
                    <li>Slow if $n$ is very large</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>Need to choose $\alpha$</li>
                    <li>Need to iterate</li>
                </ul>
            </td>
        <tr></tr>
    </tbody>
</table>
#### Code

Linear Regression with Normal Equation:
```python
from sklearn.linear_model import LlinearRegression


lin_reg = LlinearRegression()
lin_reg.fit(X_train, Y_train)
lin_reg.predict(X_new)
```

Linear Regression with Gradient Descent
```python
from sklearn.linear_model import SGDRegressor


sgd_reg = SGDRegressor(n_iter=n_epochs, penalty=None, eta0=learning_rate)
sgd_reg.fit(X_train, Y_train)
```

### Ridge Regression

Add **L2** regularization to Linear Regression's Cost function.

#### Model:    

the same as linear regression

#### Cost Function:    

$$J(\theta) = MSE(\theta) + \alpha\sum_{i=1}^n \theta_i^2$$

#### Training:

* **The Normal Equation (Cholesky):**
$$\hat \theta = (\mathbf X^T \cdot \mathbf X + \alpha \mathbf A)^{-1} \cdot \mathbf X^T \cdot \mathbf Y$$
<div style="text-align: right">where $\mathbf A$ is an identify matrix</div>

* **Gradient Descent:**

    Refer to [Gradient Descent](/gradient_descent/)

#### Code

Ridge Regression with Cholesky Equation:
```python
from sklearn.linear_model import Ridge


ridge_reg = Ridge(alpha=1, solver='cholesky')
ridge_reg.fit(X_train, Y_train)
ridge_reg.predict(X_new)
```

Ridge Regression with Gradient Descent:
```python
from sklearn.linear_model import SGDRegressor


sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X_train, Y_train)
```
### Lasso Regression

Add **L1** regularization to Linear Regression's Cost function.

#### Model:    

the same as linear regression

#### Cost Function:    

$$J(\theta) = MSE(\theta) + \sum_{i=1}^n \vert \theta_i \vert$$

#### Training:

* **[Gradient Descent](/gradient_descent/)**

The Lasso cost function is not differentiable at $\theta_i = 0$, but Gradient Descent still works fine if subgradient vector is used when $\theta_i = 0$

#### code
```python
from sklearn.linear_model import Lasso


lasso_reg = Lasso()
lasso_reg.fit(X_train, Y_train)
lasso_reg.predict(X_new)
```

```python
from sklearn.linear_model import SGDRegressor


sgd_reg = SGDRegressor(penalty='l1')
sgd_reg.fit(X_train, Y_train)
sgd_reg.predict(X_new)
```

### Elastic Net

Add a mix of **L1** and **L2** regularization into Linear Regression's cost function

#### Model:

the same as linear regression

#### Cost function

$$J(\theta) = MSE(\theta) + r\alpha\sum_{i=1}^n\vert\theta_i\vert + \frac{1-r}{2}\alpha\sum_{i=1}^n\theta_i^2$$

#### Training

* **[Gradient Descent](/gradient_descent/)**

#### Code
```python
from sklearn.linear_model import ElasticNet


elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train_, Y_train)
elastic_net.predict(X_new)
```

### Comparison Between L1 regularization and L2 regularization

|                        |L1 Regularization| L2 Regularization|
|------------------------|-----------------|------------------|
|solution uniqueness     |No               |Yes               |
|sparsity                |Yes              |No                |
|feature selection       |Yes              |No                |
|computational efficiency|Low (No analytical solution)|High   |

### How to choose between regression Models

* It is always preferable to have some regularization; thus, avoid plain Linear Regression models.
* **Ridge Regression** is a good *default*.
* If you suspect only a few features are actually useful, use either Lasso Regression or Elastic Net.
* Generally speaking, **Elastic Net** is more ideal than Lasso since Lasso may behave erratically (1) when the number of features ($n$) is greater the number of training instances ($m$) (2) when several features are strongly correlated.


## Classification

### Logistic Regression (Binary Classes)

#### Model

Estimate Probabilities:
$$\begin{align}
& p = \sigma(\theta^T \cdot x) \nonumber\\
& \text{ where } \sigma(t) = \frac{1}{1 + e^{-t}} \nonumber
\end{align}$$

Prediction:

$$\hat y =
\begin{cases}
  0 \text{ if } \hat p < 0.5,\\    
  1 \text{ if } \hat p >= 0.5.
\end{cases}$$

#### Cost function

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} y^{(i)}log(\hat p^{(i)}) + (1 - y^{(i)})log(1 - \hat p^{(i)})$$

#### Training

The function above is convex so gradient descent is guaranteed to find the global minimum.

#### Code
```python
from sklearn.linear_model import LogisticRegression


log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)
log_reg.predict(X_new)
```

### Logistic Regression (Multiple Classes)

#### Models

Compute score for each class:

$$s_k(\mathbf x) = (\theta^{(k)})^T \cdot \mathbf x$$

Note: Each class has its own set of $\theta_k$

Normalize score with Softmax function:

$$\hat p_k = \frac {exp(s_k(\mathbf x))}{\sum_{i=1}^{k} exp(s_i(\mathbf x))}$$

Prediction:

$$\hat y = argmax_{k} \,\sigma(s(\mathbf x))_k = argmax_k \,s_k(\mathbf x)$$

#### Cost Function

$$J(\theta) = -\frac {1}{m} \sum_{i=1}^{m} \sum_{j=1}^{k} y_k^{(i)}log(\hat p_k^{(i)})$$

#### Training

* **[Gradient Descent](/gradient_descent/)**

#### Code
```python
from sklearn.linear_model import LogisticRegression


multi_log_reg = LogisticRegression(multi_class='multinomial', solver='sag', C=10)
multi_log_reg.fit(X_train, Y_train)
multi_log_reg.predict(X_train)
```
