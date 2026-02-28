import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

X = pd.read_csv("features.csv")
y = pd.read_csv("labels.csv")

params = {"C": [0.1, 1, 10]}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid=params,
    cv=3
)

grid.fit(X, y.values.ravel())

print("Best Parameters:", grid.best_params_)

print("Best Score:", grid.best_score_)

