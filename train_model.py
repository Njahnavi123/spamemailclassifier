import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load dataset
data = pd.read_csv("dataset/spam.csv")

# Convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Features & labels
X = data['text']
y = data['label']

# Vectorization
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Save model and vectorizer
pickle.dump(model, open("model/spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model trained successfully")
