import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

X = pd.read_csv("features.csv")
y = pd.read_csv("labels.csv")

model = LogisticRegression(max_iter=1000)

scores = cross_val_score(model, X, y.values.ravel(), cv=5)

print("Cross-Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())