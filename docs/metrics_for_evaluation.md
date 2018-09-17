# Metrics for Evaluation

## For Regression Models

### Mean Absolute Error (MAE)

$$MAE = \frac{1}{m} \sum_{i=1}^m \vert y^{(i)} - \hat y^{(i)} \vert$$

### Mean Squared Error (MSE)

$$MSE = \frac{1}{m} \sum_{i=1}^m (y^{(i)} - \hat y^{(i)})^2$$

### Root Mean Squared Error (RMSE)

$$RMSE = \sqrt{\frac{1}{m} \sum_{i=1}^m (y^{(i)} - \hat y^{(i)})^2}$$

### Relative Absolute Error (RAE)

$$RAE = \frac{\sum_{i=1}^m \vert y^{(i)} - \hat y^{(i)} \vert}{\sum_{i=1}^m \vert y^{(i)} - \bar y^{(i)} \vert}$$

### Relative Squared Error (RSE)

$$RAE = \frac{\sum_{i=1}^m (y^{(i)} - \hat y^{(i)})^2}{\sum_{i=1}^m (y^{(i)} - \bar y^{(i)})^2}$$

### Squared Value ($R^2$)

$$R^2 = 1 - RSE$$
