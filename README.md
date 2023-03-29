# Interventional Effect Explanations

[Interpreting ML Predictions based on Interventional Effect](http://www.riss.kr/link?id=T16626970)

[**Abstract**] Recently, many approaches have been proposed to reveal the magic of machine learning models. Among them, Shapley Additive exPlanations (SHAP) has gained the most fame for its capacity to provide individualized interpretations regardless of the model being used. However, one may hesitate to use Kernel SHAP, which is the most generalized approach, to estimate SHAP values, since it requires an exponential computational cost depending on the number of the features. Besides, using Shapley value as the measure for feature importance has been criticized as being paradoxical in that it allows an indirect influence of features. Thus, this paper proposes an alternative for this game-theoretic formulation by measuring feature importance based on the interventional effect of each feature. This new method can provide more intuitive results while spending less time for computation. We proved the competitiveness of our method using five different datasets from Scikit-learn and the UCI machine learning repository.

## Example
```py
import iee
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

diabetes = load_diabetes(scaled=False)
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
