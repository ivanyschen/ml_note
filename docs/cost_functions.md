# Cost Functions

## Mean Squared Error (MSE)

$$MSE = \frac{1}{m} \sum_{i=1}^m (y^{(i)} - \hat y^{(i)})^2$$

## Root Mean Squared Error (RMSE)

$$RMSE = \sqrt{\frac{1}{m} \sum_{i=1}^m (y^{(i)} - \hat y^{(i)})^2}$$

## Mean Absolute Error (MAE)

$$MAE = \frac{1}{m}\sum_{i=1}^m \vert y^{(i)} - \hat y^{(i)} \vert$$

## Comparisons

### MSE v.s. MAE

|                        |MSE              |MAE               |
|------------------------|-----------------|------------------|
|Robustness to outliers  |Low              |High              |
|Stability of solution   |High             |Low               |
|Number of solutions     |Unique           |multiple          |
