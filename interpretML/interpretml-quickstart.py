## NOTE 20/11/2024: 'pip install interpret' is not working due to runtime errors. Created .gitignore and optionally pick up later.




# Source: https://interpret.ml/docs/index.html

# pip install interpret
# If this fails, check: https://interpret.ml/docs/installation-guide.html

# Download and prepare data, using a dataset on adults from UCI Machine Learning Repository
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    header=None)
df.columns = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
]
train_cols = df.columns[0:-1]
label = df.columns[-1]
X = df[train_cols]
y = df[label]

seed = 42
np.random.seed(seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

# Train a glassbox model: models that are designed to be completely interpretable.
# Easier to remember: a glassbox model is the opposite of a black box model ;)

from interpret.glassbox import ExplainableBoostingClassifier
ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train) # Result here is: "ExplainableBoostingClassifier()". How do I interpret this?

# Interpreting the glassbox using global explanations
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

from interpret import show
ebm_global = ebm.explain_global()
show(ebm_global)
# If you're using VScode, right-click your code 'Run in interactive terminal' to display the image.
# Some text with how to interpret the results is included in the image (awesome!)

# BLACKBOX PIPELINE
# Blackbox interpretability models can extract explanations from any machine learning (ML) pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# We have to transform categorical variables to use sklearn models
X_enc = pd.get_dummies(X, prefix_sep='.')
feature_names = list(X_enc.columns)
y = df[label].apply(lambda x: 0 if x == " <=50K" else 1)  # Turning response into 0 and 1
X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.20, random_state=seed)

#Blackbox system can include preprocessing, not just a classifier!
pca = PCA()
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
blackbox_model = Pipeline([('pca', pca), ('rf', rf)])
blackbox_model.fit(X_train, y_train)

# Explaining and interpreting the blackbox
from interpret.blackbox import LimeTabular
from interpret import show

lime = LimeTabular(blackbox_model.predict_proba, X_train, random_state=seed)
lime_local = lime.explain_local(X_test[:5], y_test[:5])

show(lime_local, 0)
# If you're using VScode, right-click your code 'Run in interactive terminal' to display the image.