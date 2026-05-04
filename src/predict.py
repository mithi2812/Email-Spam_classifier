import pandas as pd
import re
import joblib

# Load your large unlabeled dataset
data = pd.read_csv("data/spam.csv", encoding="latin-1")

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

data['clean_text'] = data['message'].apply(clean_text)

# Load trained model + vectorizer
model = joblib.load("src/spam_model.pkl")
vectorizer = joblib.load("src/vectorizer.pkl")

# Transform text
X = vectorizer.transform(data['clean_text'])

# Predict spam/ham
predictions = model.predict(X)

# Add predictions to dataframe
data['predicted_label'] = predictions
data['predicted_label'] = data['predicted_label'].map({0:'ham', 1:'spam'})

# Save results
data.to_csv("data/spam_predictions.csv", index=False)

print("Predictions saved to data/spam_predictions.csv")
