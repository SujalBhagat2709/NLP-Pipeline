import pandas as pd
import joblib

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

model = joblib.load("models/nlp_model.pkl")

predictions = model.predict(X_test)

errors = pd.DataFrame({
    "Actual": y_test.values.ravel(),
    "Predicted": predictions
})

print("Misclassified Examples:")
print(errors[errors["Actual"] != errors["Predicted"]])