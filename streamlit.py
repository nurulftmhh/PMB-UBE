import streamlit as st
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

# Download stopwords and wordnet for lemmatization
nltk.download('stopwords')
nltk.download('wordnet')

# Load your trained model
# Make sure to save your model after training and load it here
# For example, you can use joblib or pickle to save and load your model
import joblib
model = joblib.load('intent_classifier.pkl')

# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit app
st.title("Chatbot Intent Prediction")

user_input = st.text_input("Ask your question:")

if st.button("Submit"):
    if user_input:
        processed_input = preprocess_text(user_input)
        prediction = model.predict([processed_input])
        st.write(f'Predicted Intent: {prediction[0]}')
    else:
        st.write("Please enter a question.")
