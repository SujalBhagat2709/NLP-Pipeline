import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

vectorizer = joblib.load("models/vectorizer.pkl")
model = joblib.load("models/final_nlp_model.pkl")

text = "This product is absolutely wonderful"

cleaned = clean_text(text)
vector = vectorizer.transform([cleaned])

prediction = model.predict(vector)

print("Prediction:", prediction[0])