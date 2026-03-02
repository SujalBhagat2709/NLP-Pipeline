import joblib

model = joblib.load("models/nlp_model.pkl")

joblib.dump(model, "models/final_nlp_model.pkl")

print("Final Model Saved Successfully")