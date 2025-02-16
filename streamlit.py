import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import string
import csv
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path

class ChatBot:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.train_df = None
        self.intent_response_mapping = None
        self.slangwords_dict = self._load_slang_dictionary()
        self.initialize_components()

    def _load_slang_dictionary(self):
        # Default slang dictionary
        base_dict = {
            "mhs": "mahasiswa",
            "maba": "mahasiswa baru",
            "pkkmb": "pengenalan kehidupan kampus bagi mahasiswa baru"
        }
        
        # Try to load additional slangwords from CSV if available
        try:
            slang_path = Path('Slangword-indonesian.csv')
            if slang_path.exists():
                with open(slang_path, mode='r', encoding='utf-8', newline='') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        if len(row) >= 2:
                            base_dict[row[0].strip()] = row[1].strip()
        except Exception as e:
            st.warning(f"Could not load slangword dictionary: {str(e)}")
        
        return base_dict

    @st.cache_resource
    def initialize_components(self):
        try:
            # Load the saved model
            self.model = tf.keras.models.load_model('model_lstm.h5')
            
            # Load the text vectorizer
            with open('text_vectorization.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load the label encoder
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Load training data and create intent-response mapping
            self.train_df = pd.read_csv('Data Train.csv')
            self.intent_response_mapping = dict(zip(
                self.train_df['Intent'], 
                self.train_df['Respon']
            ))
        except Exception as e:
            st.error(f"Error loading model components: {str(e)}")
            raise

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Fix slang words
        words = text.split()
        fixed_words = [
            self.slangwords_dict.get(word.lower(), word) 
            for word in words
        ]
        return ' '.join(fixed_words)

    def predict_intent_and_response(self, user_input):
        # Preprocess the input
        processed_input = self.preprocess_text(user_input)
        
        # Vectorize the input
        input_seq = self.vectorizer([processed_input])
        
        # Make prediction
        prediction = self.model.predict(input_seq, verbose=0)
        predicted_class_index = np.argmax(prediction)
        
        # Get the predicted intent
        predicted_intent = self.label_encoder.inverse_transform([predicted_class_index])[0]
        
        # Get the corresponding response
        response = self.intent_response_mapping.get(
            predicted_intent,
            "Maaf, saya tidak memahami pertanyaan Anda."
        )
        
        return predicted_intent, response

def setup_page_config():
    st.set_page_config(
        page_title="EduBot - PMB Assistant",
        page_icon="🎓",
        layout="wide"
    )

def load_css():
    st.markdown("""
    <style>
    .chat-container {
        padding: 20px;
        margin-bottom: 100px;
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
        width: 45px;
        height: 45px;
        border-radius: 50%;
        object-fit: cover;
    }
    
    .message-bubble {
        padding: 12px 16px;
        border-radius: 15px;
        max-width: 80%;
    }
    
    .user-message {
        background-color: #F0F2F6;
        margin-left: auto;
    }
    
    .bot-message {
        background-color: #2E86C1;
        color: white;
        margin-right: auto;
    }
    
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 20px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    
    .stTextInput input {
        border-radius: 25px;
        border: 2px solid #2E86C1;
        padding: 10px 20px;
    }
    
    .stButton button {
        border-radius: 25px;
        background-color: #2E86C1;
        color: white;
        border: none;
        padding: 10px 25px;
    }
    </style>
    """, unsafe_allow_html=True)

def display_message(message, is_user=True):
    avatar_url = "https://miro.medium.com/v2/resize:fit:828/format:webp/1*I9KrlBSL9cZmpQU3T2nq-A.jpeg"  
    
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
                <img src="{avatar_url}" class="avatar" alt="EduBot">
                <div class="message-bubble bot-message">
                    {message}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    setup_page_config()
    load_css()
    
    # Initialize chatbot
    try:
        chatbot = ChatBot()
    except Exception as e:
        st.error("Failed to initialize chatbot. Please check if all required files are present.")
        return

    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1>EduBot - Asisten PMB</h1>
        <p>Selamat datang! Saya siap membantu Anda dengan informasi seputar Penerimaan Mahasiswa Baru.</p>
    </div>
    """, unsafe_allow_html=True)

    # Display conversation history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.conversation:
        display_message(message['text'], message['is_user'])
    st.markdown('</div>', unsafe_allow_html=True)

    # Input form
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input(
                "",
                placeholder="Ketik pertanyaan Anda di sini...",
                key="user_input"
            )
            submit_button = st.form_submit_button("Kirim")

        if submit_button and user_input:
            # Add user message to conversation
            st.session_state.conversation.append({
                'text': user_input,
                'is_user': True
            })
            
            try:
                # Get bot response
                _, bot_response = chatbot.predict_intent_and_response(user_input)
                
                # Add bot response to conversation
                st.session_state.conversation.append({
                    'text': bot_response,
                    'is_user': False
                })
            except Exception as e:
                st.error(f"Error processing message: {str(e)}")
            
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
