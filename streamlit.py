import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Custom CSS untuk chat bubbles
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
    
    .user-message {
        background-color: #E9ECEF;
        margin-left: 20%;
        margin-right: 2%;
    }
    
    .bot-message {
        background-color: #007AFF;
        color: white;
        margin-right: 20%;
        margin-left: 2%;
    }
    
    .message-time {
        font-size: 0.8rem;
        color: #888;
        margin-top: 5px;
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
    </style>
    """, unsafe_allow_html=True)

# Fungsi untuk menampilkan pesan dalam format bubble
def display_message(message, is_user=True):
    current_time = datetime.now().strftime("%H:%M")
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            {message}
            <div class="message-time">{current_time}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            {message}
            <div class="message-time">{current_time}</div>
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
    
    # Header
    st.title("Pendaftaran Mahasiswa BotEdu")
    
    # Display conversation history
    for message in st.session_state.conversation:
        display_message(message['text'], message['is_user'])
    
    # Chat input container
    with st.container():
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input("", placeholder="Ketik pesan Anda di sini...", key="user_input")
        with col2:
            send_button = st.button("Kirim")
    
    if send_button and user_input:
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
        
        # Clear input
        st.session_state.user_input = ""
        
        # Rerun to update chat
        st.rerun()

if __name__ == "__main__":
    main()
