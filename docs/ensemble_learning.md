# Ensemble Learning
## Voting
### Model
Aggregate the predictions of several different models and predict the class that gets the most votes.

Due to the *law of large number*, even if each classifier were a weak learner (classifiers that perform only a little better than simply guessing), the ensemble could still be a strong learner.

Voting algorithm works best when the predictions are as independent from one another as possible.

- Hard Voting
> Predicts the class that gets the most votes from the individual models

- Soft Voting
> Predict the class with the highest probability, averaged over all the individual classifiers. Soft Voting is available when all the individual classifiers have `predict_proba()` method

### Code
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression


log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf)],
    voting='hard'   # for soft voting, assign `soft`
)
voting_clf.fit(X_train, Y_train)
```

## Bagging & Pasting

### Model
Train the same based model on different random subsets of the training set. The final predictions is typically the most frequent predictions of from the individual models or average for regression.

**Bagging**: sampling process is performed *with* replacement

**Pasting**: sampling process is performed *without* replacement

**Random Patches**: sampling both training instances and features.

**Random Subspaces**: sampling features but keeping all training instances

**Why the net result is generally better?**

Each individual model has a higher bias than if it were trained on the original training set. However, the ensemble model will achieve a similar bias but *lower variance* than a single model trained on the original training set.

**Comparison between Bagging and Pasting**:

Without replacement during the sampling process, pasting results in a slightly lower bias than bagging because each based model is trained on a more diverse subset. However, each based model also ends up being less correlated; therefore, the variance of the ensemble is reduced.



### Out-of-Bag Evaluation
Use the instances in the training set which is never sampled and used to train models to evaluate the performance of the ensemble model so that the ensemble model could be evaluated without a separate validation set.


### Code
```python
# If a based model has `predict_proba` method,
# BaggingClassifier automatically performs soft voting
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,
    bootstrap=True,    # for bagging, bootstrap=True; for pasting, bootstrap=False
    )
bag_clf.fit(X_train, Y_train)
```

Out-of-bag evaluation
```python
bag_clf = BaggingClassifier(
    n_estimators=500,
    max_samples=100,
    bootstrap=True,    # for bagging, bootstrap=True; for pasting, bootstrap=False
    oob_score=True,
    )
bag_clf.fit(X_train, Y_train)
print(bag_clf.oob_score_)
```

## Random Forests
### Model
Random Forests is an ensemble of Decision Trees, trained via the bagging method.

The algorithm introduces randomness when growing trees; instead of searching for the very best feature when splitting a node, it searches for the best feature among a random subset of features.

### Code
```python
from sklearn.ensemble import RandomForestClassifier


rnd_clf = RandomForestClassifier()
rnd_clf.fit(X_train, Y_train)
```

## Boosting
###AdaBoost
**Training (Classification)**

- Initialize $w^{(i)} = \frac{1}{m} \text{for i = 1 ... m}$
- For t = 1 ... T
    - train $model_t$ and compute error on the training set
        $$r_t = \frac{\sum_{\hat y^{(i)} \neq y^{(i)}}w^{(i)}}{\sum_{i=1}^m w^{(i)}}$$
    - Compute $model_t$'s weight $\alpha_t$. $\eta$ is learning rate
        $$\alpha_t = \eta \log{\frac{1 - r_t}{r_t}}$$
    - Update $w^{(i)}$ for $i = 1,...,m$
        $$w^{(i)} =
        \begin{cases}
        w^{(i)}  & \text{if } \hat y_t^{(i)} = y^{(i)} \\
        w^{(i)}exp(\alpha_t) & \text{if } \hat y_t^{(i)} \neq y^{(i)}
        \end{cases}$$

**Model Prediction (Classification)**
$$\hat y = {argmax}_k {\sum_{t=1}^T}_{\hat y_t = k} \alpha_t$$

**Code (Classification)**
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    algorithm='SAMME.R',
    learning_rate=0.5,
    )
ada_clf.fit(X_train, Y_train)
```

###Gradient Boosting
**Training (Regression)**

For t (step) = 1...T

- if t == 1:
    - fit a model_t with the training dataset
- else:
    - fit a a model_t with $\epsilon^{(i)}$
- calculate residual error
    $$\epsilon^{(i)} = y^{(i)} - \hat y_t^{(i)}$$

**Model prediction (Regression)**

$$\hat y = \sum_{t=1}^T model_t(x)$$

**Code (Regression)**
```python
from sklearn.ensemble import GradientBoostingRegressor


bg_reg = GradientBoostingRegressor()
bg_reg.fit
```
