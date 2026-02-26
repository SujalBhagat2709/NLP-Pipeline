import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

model = joblib.load("models/nlp_model.pkl")

predictions = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))