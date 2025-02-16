import streamlit as st
import tensorflow as tf
import joblib
import pandas as pd
from datetime import datetime
import base64

def local_css():
    st.markdown("""
    <style>
    .chat-container {
        padding: 10px;
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
        padding: 10px;
        border-radius: 15px;
        max-width: 80%;
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
        width: 100%;
        padding: 20px;
        background-color: white;
    }
    
    .main-container {
        margin-bottom: 100px;
    }

    .stButton button {
        background-color: white;
        color: #007AFF;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        border: 1px solid #007AFF;
    }

    .stTextInput input {
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: 1px solid #E9ECEF;
    }

    .header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin: 20px auto 40px auto;  /* Added more bottom margin */
        padding-bottom: 20px;  /* Added padding at bottom */
        border-bottom: 1px solid #E9ECEF;  /* Added separator line */
        max-width: 600px;  /* Limit width for larger screens */
    }

    .header-content {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 10px;
    }

    .chat-container {
        margin-top: 30px;  /* Added space below header */
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

BOT_AVATAR = "https://miro.medium.com/v2/resize:fit:828/format:webp/1*I9KrlBSL9cZmpQU3T2nq-A.jpeg"

def display_message(message, is_user=True):
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
                <img src="{BOT_AVATAR}" class="avatar" alt="EduBot">
                <div style="flex-grow: 1;">
                    <div class="bot-name">EduBot</div>
                    <div class="message-bubble bot-message">
                        {message}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_lstm.h5")  
    return model

@st.cache_data
def load_dataset():
    train_df = pd.read_csv('Data Train.csv')
    return train_df

def preprocess_text(text):
    return text.lower()

def predict_intent_and_response(user_input, model, intent_response_mapping):
    processed_input = preprocess_text(user_input)
    prediction = model.predict([processed_input])[0]
    response = intent_response_mapping.get(prediction, "Maaf, saya tidak memahami pertanyaan Anda.")
    return prediction, response

def main():
    st.set_page_config(page_title="BotEdu Chat", layout="wide")
    local_css()
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    model = load_model()
    train_df = load_dataset()
    intent_response_mapping = dict(zip(train_df['Intent'], train_df['Respon']))
    
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Centered header with more spacing
    st.markdown(f"""
    <div class="header-container">
        <div class="header-content">
            <img src="{BOT_AVATAR}" style="width: 60px; height: 60px; border-radius: 50%;">
            <div>
                <h2 style="margin: 0; font-size: 24px;">EduBot</h2>
                <p style="margin: 5px 0 0 0; color: #666;">Asisten Pendaftaran Mahasiswa BotEdu</p>
            </div>
        </div>
    </div>
    <div class="chat-container">
    """, unsafe_allow_html=True)
    
    for message in st.session_state.conversation:
        display_message(message['text'], message['is_user'])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns([6, 1])
        with col1:
            with st.form(key='chat_form', clear_on_submit=True):
                user_input = st.text_input("", placeholder="Ketik pertanyaan Anda di sini...")
                submit_button = st.form_submit_button("Kirim")
    
        if submit_button and user_input:
            st.session_state.conversation.append({
                'text': user_input,
                'is_user': True
            })
            
            _, bot_response = predict_intent_and_response(user_input, model, intent_response_mapping)
            
            st.session_state.conversation.append({
                'text': bot_response,
                'is_user': False
            })
            
            st.rerun()

if __name__ == "__main__":
    main()
