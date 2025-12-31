import pickle

# Load model and vectorizer
model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def predict_spam(email):
    email_vec = vectorizer.transform([email])
    prediction = model.predict(email_vec)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Test
if __name__ == "__main__":
    email = input("Enter email text: ")
    print("Result:", predict_spam(email))
