 📧 Spam Detection Web App

Overview

This project is a **Spam Detection System** built using **Machine Learning** and presented through an interactive **Streamlit web application**. It classifies SMS/email messages as either **spam** or **not spam (ham)** using a trained Naive Bayes model on TF-IDF vectorized text data.

This system is part of a broader study to tackle modern spam threats and enhance communication security using lightweight and interpretable ML models.


🔍 Features

- 🧠 **Naive Bayes-based classifier** trained on SMS spam data.
- ✨ **Interactive Streamlit UI** for message classification.
- 📊 View processed text, top influential features, and prediction confidence.
- 📂 Upload your own dataset to **train a custom model**.
- 🔐 Helps defend against phishing, scams, and spam content in emails/SMS.

🛠️ Technologies Used

- Python 3.x
- Streamlit
- Scikit-learn
- NLTK
- Pandas, NumPy
- TF-IDF Vectorizer
- Pickle (for model serialization)

---

📁 File Structure

 app.py # Streamlit app for spam detection
 
train_model.py # Script to train the model

spam.csv # Dataset containing SMS labeled as spam/ham

spam_fil_model.pkl # Pre-trained model (Multinomial Naive Bayes)

tfidf_vectorizer.pkl # Pre-fitted TF-IDF vectorizer

Advanced Spam Defence Technology.ipynb # Jupyter Notebook for analysis

Cutting-Edge Spam Defense Technology Project Report.pdf

spam filtering_base paper.pdf

1️⃣pip install -r requirements.txt


2️⃣streamlit run app.py

📊 Dataset
The project uses a labeled dataset (spam.csv) containing SMS messages marked as:
spam (1)
ham (0)

Each message is preprocessed using:
Lowercasing
Tokenization
Stopword removal
Stemming
TF-IDF vectorization

🧪 Model Performance

The Multinomial Naive Bayes model was evaluated using:

Accuracy

Precision

Recall

F1-score

Typically achieves 95–98% accuracy on standard SMS spam datasets.
