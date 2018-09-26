# Decision Trees

## Classification
### Training
**CART algorithm**

- Find each feature's best split
    - For each continuous features, sort its values and find a value such that it achieves the best split
    - For each categorial feature, find a category such that it achieves the best split

- Among the best splits in the previous step, select the one minimizing the cost function

- Split the node with the feature found in the previous step.


**Cost function to minimize**

- Gini impurity Based

$$J(k, t_k) = \frac{m_{left}}{m}G_{left} + \frac{m_{right}}{m}G_{right}$$
$$\text{where }G_i = 1 - \sum_{k=1}^n p_{i, k}^2$$


- Entropy Based

$$J(k, t_k) = \frac{m_{left}}{m}H_{left} + \frac{m_{right}}{m}H_{right}$$
$$\text{where }H_i = - \sum_{k=1}^n p_{i, k} \log_2(p_{i, k})$$


- Comparison between Gini impurity and Entropy
> Most of the times, it does not make a difference choosing either of them. However, Gini impurity tends to isolate the most frequent class in its own branch of the tree while entropy tends to produce slightly more balanced trees.

**Time complexity**

Training time complexity is $O(n\text{ }m\text{ }log(m))$.

**Regularization**

- maximum depth of decision tree (**max_depth**)
- minimum number of sample split: the minimum number of samples a node must have before it can be split. (**min_samples_split**)
- minimum number of samples a leaf node must have (**min_samples_leaf**)
- maximum number of leaf nodes (**max_leaf_nodes**)
- maximum number of features to evaluate for split (**max_features**)


### Code
```python
from sklearn.tree import DecisionTreeClassifier


tree_clf = DecisionTreeClassifier()
# use "criterion" to choose between gini and entropy
tree_clf.fit(X_train, Y_train)
```


## Regression
### Model
In Decision Regression models, the predicted value is the mean of the trained samples in the leaf node that predicted instance ends up in.

$$\hat y_{node} = \frac{1}{m_{node}} \sum_{i \in node} y^{(i)}$$


### Training
**Cost function to minimize**
$$J(k, t_k) = \frac{m_{left}}{m}MSE_{left} + \frac{m_{right}}{m}MSE_{right}$$


### Code
```python
from sklearn.tree import DecisionTreeRegressor


tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, Y_train)
```


## Instability of Decision Tree
- Decision trees splits perpendicularly to an axis which makes them sensitive to rotation of dataset. Therefore, applying PCA is a good idea before feeding training data into a decision tree model.

- Decision Trees are very sensitive to small variations in the training data.

- The training algorithm used by Scikit-Learn is stochastic since the algorithm selects the sets of features to evaluate at each node *randomly*.
