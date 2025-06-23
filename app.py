import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt_tab')

st.set_page_config(
        page_title="Spam Filter",
        page_icon="🛡️",
        layout="wide"
    )

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    except:
        st.warning("Could not download NLTK resources. Some features may not work correctly.")

download_nltk_resources()

# Text preprocessing functions
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Join tokens back into text
    return ' '.join(tokens)

# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        with open('spam_fil_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('tfidf_vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please make sure 'spam_fil_model.pkl' and 'tfidf_vectorizer.pkl' exist.")
        return None, None

# Function to make prediction
def predict_spam(text, model, vectorizer):
    if model is None or vectorizer is None:
        return None
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Transform using vectorizer
    text_vector = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    return prediction, probability

# Function to train a new model
def train_model(data):
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import classification_report
    
    # Preprocess the data
    data['processed_text'] = data['text'].apply(preprocess_text)
    
    # Create vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['processed_text'])
    y = data['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save the model and vectorizer
    with open('spam_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    with open('tfidf_vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)
    
    return model, vectorizer, report

# Main app
def main():

    
    # Custom styling
    st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 20px;
    }
    .spam-tag {
        background-color: #ef4444;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .ham-tag {
        background-color: #10b981;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .spam-result {
        background-color: rgba(239, 68, 68, 0.1);
        border: 2px solid #ef4444;
    }
    .ham-result {
        background-color: rgba(16, 185, 129, 0.1);
        border: 2px solid #10b981;
    }
    .prediction-text {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .info-text {
        font-size: 1rem;
        color: #6b7280;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown("<h1 class='main-title'>📧 Spam Filter</h1>", unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Spam Detection", "Train Model", "About"])
    
    with tab1:
        # Load model and vectorizer
        model, vectorizer = load_model()
        
        # Text input
        st.subheader("Enter a message to check if it's spam")
        text_input = st.text_area("", height=150, placeholder="Type or paste your message here...")
        
        # Example messages
        with st.expander("Show example messages"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<span class='spam-tag'>SPAM EXAMPLES</span>", unsafe_allow_html=True)
                spam_examples = [
                    "URGENT: You have won a $1,000 Walmart gift card. Go to http://claim-prize.com to claim now!",
                    "Congratulations! You've been selected for a FREE iPhone 15. Click here to claim your prize now!",
                    "LAST CHANCE: Your account will be suspended. Update your information at secure-banking-verify.com"
                ]
                for example in spam_examples:
                    if st.button(f"Try: {example[:50]}...", key=f"spam_{hash(example)}"):
                        text_input = example
                        st.session_state.text_input = example
            
            with col2:
                st.markdown("<span class='ham-tag'>HAM EXAMPLES</span>", unsafe_allow_html=True)
                ham_examples = [
                    "Hi Sarah, can we reschedule our meeting to 3pm tomorrow? Thanks!",
                    "The quarterly report is ready for your review. Please let me know if you have any questions.",
                    "Don't forget to bring your laptop to the workshop on Friday."
                ]
                for example in ham_examples:
                    if st.button(f"Try: {example[:50]}...", key=f"ham_{hash(example)}"):
                        text_input = example
                        st.session_state.text_input = example
        
        # Make prediction
        if st.button("Check Message", use_container_width=True) and text_input:
            with st.spinner("Analyzing message..."):
                result = predict_spam(text_input, model, vectorizer)
                
                if result:
                    prediction, probability = result
                    
                    # Display result
                    if prediction == 1:  # Spam
                        st.markdown(f"""
                        <div class='result-box spam-result'>
                            <p class='prediction-text'>This message is likely <span class='spam-tag'>SPAM</span></p>
                            <p class='info-text'>Confidence: {probability[1]:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.warning("⚠️ Why this might be spam:")
                        
                        # Simple explanation
                        if re.search(r'(free|won|winner|prize|claim|urgent|limited|offer)', text_input.lower()):
                            st.markdown("- Contains promotional language (free, won, prize, urgent)")
                        if re.search(r'(click|http|www|\.com|\.net|\.org)', text_input.lower()):
                            st.markdown("- Contains URLs or click instructions")
                        if re.search(r'(\$|money|cash|credit|loan|payment)', text_input.lower()):
                            st.markdown("- References money or financial terms")
                        if re.search(r'(password|account|login|verify|security|update)', text_input.lower()):
                            st.markdown("- Requests account verification or updates")
                    else:  # Ham
                        st.markdown(f"""
                        <div class='result-box ham-result'>
                            <p class='prediction-text'>This message is likely <span class='ham-tag'>NOT SPAM</span></p>
                            <p class='info-text'>Confidence: {probability[0]:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show text features
                with st.expander("View text analysis"):
                    st.write("**Processed text:**")
                    st.code(preprocess_text(text_input))
                    
                    if model and vectorizer:
                        # Show top features
                        processed = preprocess_text(text_input)
                        feature_vector = vectorizer.transform([processed])
                        feature_names = vectorizer.get_feature_names_out()
                        
                        # Get non-zero features
                        non_zero = feature_vector.nonzero()[1]
                        
                        # Create a DataFrame of features and their values
                        features_df = pd.DataFrame({
                            'Feature': [feature_names[i] for i in non_zero],
                            'TF-IDF Score': [feature_vector[0, i] for i in non_zero]
                        }).sort_values('TF-IDF Score', ascending=False).head(10)
                        
                        st.write("**Top features in this message:**")
                        st.dataframe(features_df)
        
    with tab2:
        st.subheader("Train your own spam filter")
        st.write("""
        You can train a custom spam filter by uploading a CSV file with two columns:
        - `text`: The message text
        - `label`: 1 for spam, 0 for ham (not spam)
        """)
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                
                # Check if the data has the required columns
                if 'text' in data.columns and 'label' in data.columns:
                    st.write(f"Data loaded successfully: {len(data)} rows")
                    
                    # Show sample data
                    st.write("Sample data:")
                    st.dataframe(data.head())
                    
                    # Train model button
                    if st.button("Train Model", use_container_width=True):
                        with st.spinner("Training model..."):
                            model, vectorizer, report = train_model(data)
                            
                            # Show model performance
                            st.success("Model trained successfully!")
                            
                            # Display classification report
                            st.write("**Model Performance:**")
                            
                            # Convert classification report to DataFrame
                            report_df = pd.DataFrame(report).transpose()
                            
                            # Only show relevant metrics
                            if 'accuracy' in report:
                                st.metric("Accuracy", f"{report['accuracy']:.2%}")
                            
                            metrics_df = report_df.loc[['0', '1'], ['precision', 'recall', 'f1-score', 'support']]
                            metrics_df.index = ['Not Spam (Ham)', 'Spam']
                            st.dataframe(metrics_df)
                            
                            # Show confusion matrix
                            st.write("Your new model is ready to use in the 'Spam Detection' tab.")
                else:
                    st.error("The uploaded file must contain 'text' and 'label' columns.")
            except Exception as e:
                st.error(f"Error loading data: {e}")
    
    with tab3:
        st.subheader("About this Spam Filter")
        st.write("""
        This spam filter uses machine learning to classify messages as spam or not spam (ham).
        
        ### How it works:
        
        1. **Text Preprocessing**: The message is cleaned by removing special characters, converting to lowercase, 
           removing stopwords, and applying stemming.
        
        2. **Feature Extraction**: The processed text is converted into numerical features using TF-IDF vectorization.
        
        3. **Classification**: A machine learning model (Naive Bayes) predicts whether the message is spam or not.
        
        ### Model Details:
        
        - The default model is trained on a dataset of SMS messages.
        - You can train your own model with custom data using the 'Train Model' tab.
        - The model uses a Multinomial Naive Bayes classifier, which is effective for text classification.
        
        ### Common Spam Indicators:
        
        - Urgency language (urgent, limited time, act now)
        - Too-good-to-be-true offers (free, win, prize)
        - Requests for personal information
        - Unusual sender addresses
        - Poor grammar and spelling
        - Suspicious links
        """)
        
        st.info("""
        **Note**: This application is for educational purposes. While it can detect many spam messages, 
        no spam filter is 100% accurate. Always use caution when dealing with suspicious messages.
        """)

if __name__ == '__main__':
    main()