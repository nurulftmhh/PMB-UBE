import streamlit as st
import joblib

# Load the trained model
model_path = 'intent_classifier.pkl'
model = joblib.load(model_path)

# Preprocessing function
def preprocess_text(text):
    import string
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    import nltk

    nltk.download('stopwords')
    nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 3. Tokenize, remove stopwords, and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # 4. Join the words back into a single string
    return ' '.join(words)

# Streamlit app for chatbot
def main():
    st.title("Chatbot Intent Classifier")
    st.write("Ask a question or type a statement, and the chatbot will predict its intent.")

    user_input = st.text_input("Your Message:", "")

    if st.button("Send"):
        if user_input.strip():
            # Preprocess the input
            processed_input = preprocess_text(user_input)

            # Predict intent
            prediction = model.predict([processed_input])

            # Display the result as a chatbot response
            st.write(f"**Chatbot:** I think your intent is '{prediction[0]}'.")
        else:
            st.write("**Chatbot:** Please type a message to get a response.")

if __name__ == "__main__":
    main()