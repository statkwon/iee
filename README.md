# Interventional Effect Explanations

[**Abstract**] Numerous methods have been proposed recently to elucidate the operations of machine learning models. Among these, Shapley value-based methods have become particularly renowned for their capability to provide individualized interpretations across different models. However, the practical use of Shapley value is often limited by its exponential computational costs, which scale with the number of features. Additionally, the application of Shapley value as indicators of feature importance has faced criticism due to their allowance for indirect feature influences, which can lead to paradoxical results. Addressing these challenges, this paper presents an alternative approach that assesses feature importance through the interventional effects of each feature. This novel method not only offers more intuitive results but also requires affordable computational resources while maintaining model agnosticity and the ability to create local interpretations. We demonstrate the competitiveness of our approach by validating it on OpenXAI benchmark, which is for evaluations of post hoc explanation methods.

## Example
```py
from iee import Explainer
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train a RF model
model = RandomForestRegressor(random_state=0)
model.fit(X_train, y_train)

# Explain the predicted results using IEE
explainer = iee.Explainer(model)
iee_values = explainer(X_test)
```

## Benchmark with OpenXAI
You can compare the performance of IEE using [OpenXAI](https://github.com/AI4LIFE-GROUP/OpenXAI).


