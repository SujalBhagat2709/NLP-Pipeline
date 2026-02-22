import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

df = pd.read_csv("step3_cleaned.csv")

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=5000,
    min_df=1
)

X = vectorizer.fit_transform(df["clean_text"])

joblib.dump(vectorizer, "models/vectorizer.pkl")

pd.DataFrame(X.toarray()).to_csv("features.csv", index=False)
df["label"].to_csv("labels.csv", index=False)

print("Feature Engineering Completed")