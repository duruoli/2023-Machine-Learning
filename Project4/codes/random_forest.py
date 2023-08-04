import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import statsmodels.api as sm
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from skorch import NeuralNetRegressor
import warnings
warnings.filterwarnings('ignore')
import lime
import lime.lime_tabular



### 1. Diabete prediction
## 1.1 data preprocessing
df1 = pd.read_csv('./diabetes.csv')
len(df1[df1['Outcome']==1])/len(df1)
len(df1[df1['Outcome']==0])/len(df1)

X = df1.iloc[:, :8]
y = df1['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
## 1.2 Lasso linear model
lasso_cv = LassoCV(cv=5, random_state=1).fit(X_train, y_train)
print("Best alpha:", lasso_cv.alpha_)
a_l = lasso_cv.alpha_ # too large 50+

l1_model = Lasso(alpha=a_l)
l1_model.fit(X_train, y_train)

l1_weights = pd.DataFrame(l1_model.coef_, columns=['weight'], index=X_train.columns)
l1_weights = l1_weights.iloc[l1_weights['weight'].abs().argsort()[::-1]]
print(l1_weights.to_latex())

## 1.3 Random Forest
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

importances = rf_classifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_classifier.estimators_], axis=0)
forest_importances = pd.DataFrame(importances, index=X_train.columns, columns=['Feature importance'])
forest_importances['std'] = std
forest_importances = forest_importances.sort_values('Feature importance', ascending=False)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr='std', ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
fig.tight_layout()
plt.show()

## 1.4 RandomForest-LIME
# LIME has one explainer for all the models
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns.values.tolist(), class_names=y_train.unique(), verbose=True, mode='classification')

j = 5
exp = explainer.explain_instance(X_test.values[j], rf_classifier.predict_proba, num_features=len(X_train.columns))
exp.show_in_notebook(show_table=True)
exp.as_list()

j = 12
exp = explainer.explain_instance(X_test.values[j], rf_classifier.predict_proba, num_features=len(X_train.columns))
exp.show_in_notebook(show_table=True)