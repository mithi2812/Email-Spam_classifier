import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
data = pd.read_csv("data/sms_spam.csv", encoding="latin-1")

# Show actual columns
print("Original columns:", data.columns)

# Keep only the first two columns (label + message)
data = data.iloc[:, :2]
data.columns = ['label','text']

print("Renamed columns:", data.columns)

data['label'] = data['label'].map({'ham':0, 'spam':1})
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

data['clean_text'] = data['text'].apply(clean_text)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['clean_text'])
y = data['label']
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

import joblib
joblib.dump(model, "src/spam_model.pkl")
joblib.dump(vectorizer, "src/vectorizer.pkl")
print("Model and vectorizer saved successfully!")


