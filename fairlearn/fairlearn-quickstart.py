# Source: https://fairlearn.org/v0.8/quickstart.html

# pip install fairlearn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Predict whether a person makes more (label1) or less (0) than $50.000 a year, using 'sex' as a sensitive feature
data = fetch_openml(data_id=1590, as_frame=True)
X = pd.get_dummies(data.data)
y_true = (data.target == '>50K') * 1
sex = data.data['sex']

print(sex.value_counts())

# Evaluate metrics for subgroup 'sex'
# Fairlearn MetricFrame: https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.MetricFrame.html
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
classifier.fit(X, y_true)
y_pred = classifier.predict(X)
metrics = accuracy_score
gm = MetricFrame(metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sex)

print(gm.overall)
print(gm.by_group)

# Selection rate: % of population which have '1' as their label
from fairlearn.metrics import selection_rate

metrics = selection_rate
sr = MetricFrame(metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sex)

print(sr.overall)
print(sr.by_group)

# More can be found in the User guide: https://fairlearn.org/v0.8/user_guide/index.html#user-guide