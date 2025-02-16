import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import string

# Cache the model loading
@st.cache_resource
def load_resources():
    try:
        # Load the LSTM model
        model = load_model("model_lstm.h5")
        
        # Load the label encoder
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
            
        # Load the text vectorization layer
        with open("text_vectorization.pkl", "rb") as f:
            text_vectorizer = pickle.load(f)
            
        # Load the training data for response mapping
        train_df = pd.read_csv('Data Train.csv')
        intent_response_mapping = dict(zip(train_df['Intent'], train_df['Respon']))
        
        return model, label_encoder, text_vectorizer, intent_response_mapping
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None, None, None, None

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Prediction function
def predict_intent_and_response(user_input, model, label_encoder, text_vectorizer, intent_response_mapping):
    try:
        # Preprocess the input
        processed_input = preprocess_text(user_input)
        
        # Vectorize the text
        input_seq = text_vectorizer([processed_input])
        
        # Make prediction
        prediction = model.predict(input_seq)
        predicted_class_index = np.argmax(prediction)
        
        # Get the predicted intent
        predicted_intent = label_encoder.inverse_transform([predicted_class_index])[0]
        
        # Get the corresponding response
        response = intent_response_mapping.get(predicted_intent, 
            "Maaf, saya tidak dapat memahami pertanyaan Anda. Mohon ajukan pertanyaan dengan cara yang berbeda.")
        
        return predicted_intent, response
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, "Terjadi kesalahan dalam memproses pertanyaan Anda. Silakan coba lagi."

def local_css():
    st.markdown("""
    <style>
    .chat-container {
        padding: 20px;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
    }
    
    .message-content {
        display: flex;
        align-items: flex-start;
        gap: 10px;
    }
    
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
    }
    
    .bot-name {
        font-weight: bold;
        margin-bottom: 5px;
        color: #2E4053;
    }
    
    .message-bubble {
        padding: 12px 16px;
        border-radius: 15px;
        max-width: 80%;
        line-height: 1.4;
    }
    
    .user-message {
        background-color: #E9ECEF;
        margin-left: auto;
        margin-right: 2%;
    }
    
    .bot-message {
        background-color: #007AFF;
        color: white;
        margin-right: auto;
        margin-left: 2%;
    }
    
    .chat-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 20px;
        background-color: white;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    
    .header-container {
        padding: 20px;
        text-align: center;
        border-bottom: 1px solid #E9ECEF;
        margin-bottom: 30px;
        background-color: white;
    }
    
    .stButton button {
        background-color: #007AFF;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #0056b3;
    }
    
    .stTextInput input {
        border-radius: 20px;
        padding: 0.8rem 1rem;
        border: 1px solid #E9ECEF;
        font-size: 16px;
    }
    
    .main-content {
        margin-bottom: 100px;
        padding: 0 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def display_message(message, is_user=True):
    bot_avatar = "https://miro.medium.com/v2/resize:fit:828/format:webp/1*I9KrlBSL9cZmpQU3T2nq-A.jpeg"
    
    if is_user:
        st.markdown(f"""
        <div class="chat-message">
            <div class="message-bubble user-message">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message">
            <div class="message-content">
                <img src="{bot_avatar}" class="avatar" alt="EduBot">
                <div style="flex-grow: 1;">
                    <div class="bot-name">EduBot</div>
                    <div class="message-bubble bot-message">
                        {message}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="EduBot - Asisten Pendaftaran Mahasiswa",
        page_icon="🎓",
        layout="wide"
    )
    
    local_css()
    
    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    # Load resources
    model, label_encoder, text_vectorizer, intent_response_mapping = load_resources()
    
    if not all([model, label_encoder, text_vectorizer, intent_response_mapping]):
        st.error("Gagal memuat sumber daya yang diperlukan. Silakan refresh halaman.")
        return
    
    # Header
    st.markdown("""
    <div class="header-container">
        <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*I9KrlBSL9cZmpQU3T2nq-A.jpeg" 
             style="width: 80px; height: 80px; border-radius: 50%; margin-bottom: 10px;">
        <h1 style="margin: 10px 0; font-size: 28px;">EduBot</h1>
        <p style="color: #666; font-size: 16px;">Asisten Pendaftaran Mahasiswa Baru</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Display conversation history
    for message in st.session_state.conversation:
        display_message(message['text'], message['is_user'])
    
    # Chat input
    with st.container():
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input(
                "",
                placeholder="Ketik pertanyaan Anda di sini...",
                key="user_input"
            )
            
            col1, col2, col3 = st.columns([4, 1, 4])
            with col2:
                submit_button = st.form_submit_button("Kirim")
    
    # Handle user input
    if submit_button and user_input:
        # Add user message to conversation
        st.session_state.conversation.append({
            'text': user_input,
            'is_user': True
        })
        
        # Get bot response
        intent, response = predict_intent_and_response(
            user_input, model, label_encoder, text_vectorizer, intent_response_mapping
        )
        
        # Add bot response to conversation
        st.session_state.conversation.append({
            'text': response,
            'is_user': False
        })
        
        # Rerun to update the display
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
