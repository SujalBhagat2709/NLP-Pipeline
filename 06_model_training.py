import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train.values.ravel())

joblib.dump(model, "models/nlp_model.pkl")

print("Model Training Completed")