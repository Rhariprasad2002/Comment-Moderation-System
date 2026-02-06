import streamlit as st
import joblib
import re

# -----------------------------
# LOAD MODELS
# -----------------------------
spam_model = joblib.load(r"C:\spam\models\spam_model.pkl")
spam_vectorizer = joblib.load(r"C:\spam\models\spam_vectorizer.pkl")

toxic_model = joblib.load(r"C:\spam\models\toxic_model.pkl")
toxic_vectorizer = joblib.load(r"C:\spam\models\toxic_vectorizer.pkl")

# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

# -----------------------------
# PREDICTION LOGIC
# -----------------------------
def moderate_text(text):
    text = clean_text(text)

    # 1️⃣ Spam check
    spam_pred = spam_model.predict(
        spam_vectorizer.transform([text])
    )[0]

    if spam_pred == 1:
        return "🚫 SPAM MESSAGE"

    # 2️⃣ Toxic check
    toxic_pred = toxic_model.predict(
        toxic_vectorizer.transform([text])
    )[0]

    if toxic_pred == 1:
        return "⚠️ INAPPROPRIATE / BULLYING"

    return "✅ CLEAN MESSAGE"

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(
    page_title="ML Comment Moderation",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Comment Moderation System")
st.write("Detect **Spam** and **Inappropriate comments** using Machine Learning")

user_text = st.text_area("Enter a comment:", height=120)

if st.button("Analyze"):
    if user_text.strip() == "":
        st.warning("Please enter a comment")
    else:
        result = moderate_text(user_text)
        if "CLEAN" in result:
            st.success(result)
        else:
            st.error(result)
