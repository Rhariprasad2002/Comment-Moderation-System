# 🛡️ Machine Learning–Based Comment Moderation System

A Machine Learning project that automatically detects **Spam** and **Inappropriate (Bullying/Toxic)** comments using **Natural Language Processing (NLP)** techniques.  
The system is deployed as an interactive **Streamlit web application**.

---

## 📌 Project Motivation

Online platforms often face issues such as:
- Spam messages (ads, scams, promotions)
- Abusive or inappropriate comments (bullying, insults)

Manual moderation is time-consuming and not scalable.  
This project uses **Machine Learning** to automatically moderate user-generated text.

---

## 🎯 Project Objective

To build a supervised ML-based text classification system that categorizes user comments as:
- 🚫 **Spam**
- ⚠️ **Inappropriate / Bullying**
- ✅ **Clean**

---

## 🧠 Machine Learning Approach

- **Type:** Supervised Machine Learning  
- **Domain:** Natural Language Processing (NLP)  
- **Feature Extraction:** TF-IDF  
- **Models Used:**
  - **Spam Detection:** Multinomial Naive Bayes
  - **Toxic/Bullying Detection:** Logistic Regression / Linear SVM
- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-score

---

## 📂 Datasets Used

### 1️⃣ Spam Dataset
- **Name:** SMS Spam Collection Dataset  
- **Labels:** `spam`, `ham`  
- **Purpose:** Detect promotional or scam messages

### 2️⃣ Toxic / Bullying Dataset
- **Type:** Cyberbullying / Inappropriate Comments  
- **Labels:** `Bullying`, `Non-Bullying`  
- **Purpose:** Detect abusive or offensive language

---

## 🛠️ Project Structure
comment_moderation_ml/
│
├── data/
│ ├── spam.csv
│ └── data.csv
│
├── models/
│ ├── spam_model.pkl
│ ├── spam_vectorizer.pkl
│ ├── toxic_model.pkl
│ └── toxic_vectorizer.pkl
│
├── train_spam.py
├── train_toxic.py
├── app.py
├── requirements.txt
