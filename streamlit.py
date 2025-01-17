import streamlit as st
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
    
    .message-time {
        font-size: 0.8rem;
        color: #888;
        margin-top: 5px;
        text-align: right;
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
        background-color: #007AFF;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        border: none;
    }

    .stTextInput input {
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: 1px solid #E9ECEF;
    }
    </style>
    """, unsafe_allow_html=True)

# Base64 encoded robot avatar (you can replace this with your own image)
BOT_AVATAR = "https://miro.medium.com/v2/resize:fit:828/format:webp/1*I9KrlBSL9cZmpQU3T2nq-A.jpeg"

# Fungsi untuk menampilkan pesan dalam format bubble
def display_message(message, is_user=True):
    current_time = datetime.now().strftime("%H:%M")
    
    if is_user:
        st.markdown(f"""
        <div class="chat-message">
            <div class="message-bubble user-message">
                {message}
                <div class="message-time">{current_time}</div>
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
                        <div class="message-time">{current_time}</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Muat model dan dataset
@st.cache_resource
def load_model():
    model = joblib.load('intent_classifier.pkl')
    return model

@st.cache_data
def load_dataset():
    train_df = pd.read_csv('Intent Pendaftaran Mahasiswa BotEdu - Data Train.csv')
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
    
    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    # Load model and dataset
    model = load_model()
    train_df = load_dataset()
    intent_response_mapping = dict(zip(train_df['Intent'], train_df['Respon']))
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header with bot info
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin: 20px 0;">
            <img src="{BOT_AVATAR}" style="width: 50px; height: 50px; border-radius: 50%;">
            <div>
                <h2 style="margin: 0;">EduBot</h2>
                <p style="margin: 0; color: #666;">Asisten Pendaftaran Mahasiswa BotEdu</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display conversation history
    for message in st.session_state.conversation:
        display_message(message['text'], message['is_user'])
    
    # Chat input container
    with st.container():
        col1, col2 = st.columns([6, 1])
        with col1:
            # Using a form to handle input clearing properly
            with st.form(key='chat_form', clear_on_submit=True):
                user_input = st.text_input("", placeholder="Ketik pesan Anda di sini...")
                submit_button = st.form_submit_button("Kirim")
    
        if submit_button and user_input:
            # Add user message to conversation
            st.session_state.conversation.append({
                'text': user_input,
                'is_user': True
            })
            
            # Get bot response
            _, bot_response = predict_intent_and_response(user_input, model, intent_response_mapping)
            
            # Add bot response to conversation
            st.session_state.conversation.append({
                'text': bot_response,
                'is_user': False
            })
            
            # Rerun to update chat
            st.rerun()

if __name__ == "__main__":
    main()
