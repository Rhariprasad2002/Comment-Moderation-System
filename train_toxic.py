import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv(r"C:\spam\data\data.csv")
df.columns = ['text', 'label']
df['label'] = df['label'].map({
    'Bullying': 1,
    'Non-Bullying': 0
})
df.dropna(inplace=True)

# -----------------------------
# 2. CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df['text'] = df['text'].apply(clean_text)

# -----------------------------
# 3. TF-IDF FEATURES
# -----------------------------
vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3,5),
    min_df=3,
    max_df=0.9
)

X = vectorizer.fit_transform(df['text'])
y = df['label']

# -----------------------------
# 4. TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 5. MODELS TO COMPARE
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight='balanced'
    ),
    "Linear SVM": LinearSVC(
        class_weight='balanced'
    ),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42
    )
}

# -----------------------------
# 6. TRAIN & EVALUATE
# -----------------------------
for name, model in models.items():
    print(f"\n================ {name} =================")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
