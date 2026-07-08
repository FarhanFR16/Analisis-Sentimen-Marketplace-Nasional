import streamlit as st
import requests
import json
import time

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Analisis Sentimen Marketplace",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ Deteksi Sentimen E-Commerce Indonesia")
st.markdown("""
Aplikasi ini menggunakan **Support Vector Machine (SVM)** dan **TF-IDF** untuk menganalisis sentimen ulasan pelanggan di marketplace secara otomatis. 
Masukkan ulasan Anda di bawah ini!
""")

API_URL = "http://localhost:8000/predict"

with st.form("sentiment_form"):
    user_input = st.text_area("Masukkan Ulasan:", placeholder="Contoh: Barangnya bagus banget, pengiriman super cepat!", height=150)
    submitted = st.form_submit_button("Analisis Sentimen")

if submitted:
    if not user_input.strip():
        st.warning("⚠️ Mohon masukkan teks ulasan terlebih dahulu.")
    else:
        # Loading Spinner untuk User Feedback
        with st.spinner("Menganalisis sentimen dengan model AI (Sastrawi + TF-IDF + SVM)..."):
            try:
                # Simulasi sedikit delay agar spinner terlihat jika API terlalu cepat
                time.sleep(0.5) 
                
                payload = {"text": user_input}
                response = requests.post(API_URL, json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    sentiment = result.get("sentiment")
                    confidence = result.get("confidence", 0.0)
                    
                    st.success("✅ Analisis Selesai!")
                    
                    # Tampilkan hasil dengan metrik UI yang rapi
                    col1, col2 = st.columns(2)
                    with col1:
                        if sentiment == "Low Cortisol":
                            st.metric(label="Prediksi Sentimen", value=f"😊 {sentiment} (Positif)")
                        elif sentiment == "High Cortisol":
                            st.metric(label="Prediksi Sentimen", value=f"😠 {sentiment} (Negatif)")
                        elif sentiment == "No Cortisol":
                            st.metric(label="Prediksi Sentimen", value=f"😐 {sentiment} (Netral)")
                        else:
                            st.metric(label="Prediksi Sentimen", value=f"{sentiment}")
                    with col2:
                        st.metric(label="Confidence Score", value=f"{confidence * 100:.1f}%")
                        
                    with st.expander("Lihat Detail Respons JSON API"):
                        st.json(result)
                        
                elif response.status_code == 400:
                    error_msg = response.json().get("detail", "Input tidak valid.")
                    st.error(f"❌ Kesalahan Input: {error_msg}")
                else:
                    st.error(f"❌ Terjadi kesalahan pada server (Status {response.status_code}). Pastikan backend API berjalan.")
            
            except requests.exceptions.ConnectionError:
                st.error("🔌 Gagal terhubung ke API. Pastikan server FastAPI sudah berjalan di http://localhost:8000.")
            except Exception as e:
                st.error(f"⚠️ Terjadi kesalahan tidak terduga: {e}")

st.markdown("---")
st.markdown("💡 *Arsitektur: Input Layer (Streamlit) -> Preprocessing (Tokenization & Sastrawi) -> Inference (SVM Model) -> Output (Sentiment Result)*")
