# Source: https://fairlearn.org/v0.11/quickstart.html

# pip install fairlearn
# pip install matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset containing hospital re-admissions of diabetic patients.
# We will take a look at how racial disparities impact health care resource allocation in the US.
from fairlearn.datasets import fetch_diabetes_hospital

data = fetch_diabetes_hospital(as_frame=True)
X = data.data.copy()
X.drop(columns=["readmitted", "readmit_binary"], inplace=True)
y = data.target
X_ohe = pd.get_dummies(X)
race = X['race']

# Look at the amount of time each race is represented in the dataset.
print("Race value counts: \n", race.value_counts())

# Next we evaluate metrics for subgroups:
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42)  # set seed for consistent results
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X_ohe, y, race, random_state=123)
classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
print(classifier.fit(X_train, y_train))

y_pred = (classifier.predict_proba(X_test)[:,1] >= 0.1)
mf = MetricFrame(metrics=accuracy_score, y_true=y_test, y_pred=y_pred, sensitive_features=A_test)

# We print the metrics per group, which indicate model's accuracy per racial group.
print("Accuracy score per group: \n", mf.by_group)

# The threshold is set based on the risk of readmission. 
# The false positive rate captures the individuals who in reality would be admitted to the hospital, but the model does not predict that outcome.
from fairlearn.metrics import false_negative_rate
mf = MetricFrame(metrics=false_negative_rate, y_true=y_test, y_pred=y_pred, sensitive_features=A_test)

print("False negative rate per group: \n", mf.by_group)

# We can plot these findings using fiarlearn.metrics.MetricsFrame
from fairlearn.metrics import false_positive_rate, selection_rate, count

metrics = {
    "accuracy": accuracy_score,
    #"precision": zero_div_precision_score,
    "false positive rate": false_positive_rate,
    "false negative rate": false_negative_rate,
    "selection rate": selection_rate,
    "count": count,
}
metric_frame = MetricFrame(
    metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=A_test
)
metric_frame.by_group.plot.bar(
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    title="Show all metrics",
)

print(metrics)

# Create a new model with fairness constraint, to mitigate disparities.
from fairlearn.reductions import ErrorRate, EqualizedOdds, ExponentiatedGradient
objective = ErrorRate(costs={'fp': 0.1, 'fn': 0.9})
constraint = EqualizedOdds(difference_bound=0.01)
classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
mitigator = ExponentiatedGradient(classifier, constraint, objective=objective)
mitigator.fit(X_train, y_train, sensitive_features=A_train)

y_pred_mitigated = mitigator.predict(X_test)
mf_mitigated = MetricFrame(metrics=accuracy_score, y_true=y_test, y_pred=y_pred_mitigated, sensitive_features=A_test)

print("Mitigation per group: \n", mf_mitigated.by_group)

# More can be found in the User guide: https://fairlearn.org/v0.8/user_guide/index.html#user-guide