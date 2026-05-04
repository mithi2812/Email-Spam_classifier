import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer


data = pd.read_csv("data/spam.csv", encoding="latin-1")
data = data[['file','message']]
data.columns = ['label','text']

data['label'] = data['label'].map({'ham':0, 'spam':1})


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove punctuation
    return text

data['clean_text'] = data['text'].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['clean_text'])
y = data['label']

print("Shape of features:", X.shape)
print("First 5 labels:", y.head())
print(data.head(10))



