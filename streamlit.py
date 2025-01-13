import streamlit as st
import joblib
import pandas as pd

# Muat model yang sudah dilatih
model_path = 'intent_classifier.pkl'
model = joblib.load(model_path)

# Muat dataset dari CSV yang berisi intent dan respons
train_df = pd.read_csv('Intent Pendaftaran Mahasiswa BotEdu - Data Train.csv')

# Membuat mapping antara intent dan respons
intent_response_mapping = dict(zip(train_df['Intent'], train_df['Respon']))

# Fungsi untuk memproses input sebelum prediksi (misalnya pembersihan teks, tokenisasi, dll.)
def preprocess_text(text):
    # Tambahkan proses pembersihan teks sesuai kebutuhan
    return text.lower()  # Contoh sederhana, bisa disesuaikan dengan preprocessing yang lebih kompleks

def predict_intent_and_response(user_input):
    # Memproses input dari user
    processed_input = preprocess_text(user_input)
    
    # Prediksi intent menggunakan model yang sudah dilatih
    prediction = model.predict([processed_input])[0]
    
    # Ambil respons berdasarkan intent yang diprediksi
    response = intent_response_mapping.get(prediction, "Intent tidak ditemukan.")
    
    return prediction, response

# Streamlit app
st.title("Pendaftaran Mahasiswa BotEdu")

st.sidebar.header("Tentang")
st.sidebar.write(
    "Chatbot ini membantu Anda dengan pertanyaan terkait pendaftaran mahasiswa di BotEdu."
)

# State untuk menyimpan percakapan
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Input box untuk user query
user_input = st.text_input("Tanyakan sesuatu tentang pendaftaran:")

if user_input:
    # Mendapatkan intent dan respons
    predicted_intent, predicted_response = predict_intent_and_response(user_input)
    
    # Menyimpan pesan dan respons dalam session_state untuk melacak percakapan
    st.session_state['messages'].append(('user', user_input))
    st.session_state['messages'].append(('bot', predicted_response))
    
    # Kosongkan kolom input setelah pengguna mengirimkan pesan
    st.text_input("Tanyakan sesuatu tentang pendaftaran:", key="input_box", value="")

# Menampilkan percakapan dengan tampilan bubble chat
for message_type, message in st.session_state['messages']:
    if message_type == 'user':
        st.markdown(f'<div style="background-color:#DCF8C6; padding:10px; border-radius:15px; margin-bottom:10px;">'
                    f'<strong>User:</strong> {message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background-color:#E8E8E8; padding:10px; border-radius:15px; margin-bottom:10px;">'
                    f'<strong>Bot:</strong> {message}</div>', unsafe_allow_html=True)
