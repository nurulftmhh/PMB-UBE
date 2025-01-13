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

# Input box untuk user query
user_input = st.text_input("Tanyakan sesuatu tentang pendaftaran:")

if user_input:
    # Mendapatkan intent dan respons
    predicted_intent, predicted_response = predict_intent_and_response(user_input)
    
    # Menampilkan hasil
    st.write(f"**Predicted Intent:** {predicted_intent}")
    st.write(f"**Response:** {predicted_response}")
