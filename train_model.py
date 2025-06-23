import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    """Preprocess text by removing special characters, stopwords, and applying stemming."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def train_model(data_path="E:\\PROJECT_SPAM\\spam.csv"):
    """Train a spam detection model and save the model and vectorizer."""
    # Load dataset
    data = pd.read_csv(data_path, encoding="latin-1")
    
    # Rename columns to match expected format
    data.rename(columns={'Category': 'label', 'Message': 'text'}, inplace=True)
    
    # Convert labels to binary (spam=1, ham=0)
    data['label'] = data['label'].map({'spam': 1, 'ham': 0})
    
    # Preprocess the text
    data['processed_text'] = data['text'].apply(preprocess_text)
    
    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['processed_text'])
    y = data['label']
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("Model Training Complete. Performance Report:\n", report)
    
    # Save the model and vectorizer
    with open('spam_fil_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    with open('tfidf_vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)
    
    print("Model and vectorizer saved as 'spam_fil_model.pkl' and 'tfidf_vectorizer.pkl'")

if __name__ == "__main__":
    train_model()
