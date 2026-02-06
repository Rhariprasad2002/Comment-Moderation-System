import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ✅ Load SPAM dataset
df = pd.read_csv(r"C:\spam\data\spam.csv", encoding="latin-1")

# Keep correct columns
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['text'] = df['text'].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test)))

# Save
joblib.dump(model, r"C:\spam\models\spam_model.pkl")
joblib.dump(vectorizer, r"C:\spam\models\spam_vectorizer.pkl")

print("✅ Spam model trained successfully")
