import streamlit as st
import joblib

# Load the trained model
model_path = 'intent_classifier.pkl'
model = joblib.load(model_path)

def predict_intent(user_input):
    # Predict the intent of the user input
    prediction = model.predict([user_input])
    return prediction[0]

# Streamlit app
st.title("Intent Classification Chatbot")

st.sidebar.header("About")
st.sidebar.write(
    "This chatbot is powered by a Random Forest Classifier trained to classify intents based on text input."
)

# Input box for user query
user_input = st.text_input("Ask a question:")

if user_input:
    # Get the predicted intent
    response = predict_intent(user_input)

    # Display the response
    st.write(f"**Predicted Intent:** {response}")
