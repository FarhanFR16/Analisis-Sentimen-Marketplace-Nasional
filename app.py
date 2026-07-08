import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Adjust import path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.preprocess import TextPreprocessor, load_preprocessor, SLANG_DICT, refine_sentiment

# ----------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ----------------------------------------------------
st.set_page_config(
    page_title="Analisis Sentimen Marketplace Nasional",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B2B 0%, #FF416C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border-left: 5px solid #FF4B2B;
    }
    
    .metric-card {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    
    .sentiment-positive {
        color: #2ecc71;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    .sentiment-neutral {
        color: #f1c40f;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    .sentiment-negative {
        color: #e74c3c;
        font-weight: bold;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. MODEL LOADERS (WITH CACHING & ERROR HANDLING)
# ----------------------------------------------------
@st.cache_resource
def load_ml_components():
    models_dir = r"c:\git\UAS-AI\Analisis-Sentimen-Marketplace-Nasional\models"
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    model_path = os.path.join(models_dir, "svm_model.pkl")
    preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
    metrics_path = os.path.join(models_dir, "evaluation_metrics.pkl")
    
    try:
        if not (os.path.exists(vectorizer_path) and os.path.exists(model_path)):
            return None, None, None, None, "Model file tidak ditemukan. Harap jalankan script training terlebih dahulu."
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load preprocessor if exists, otherwise create new
        if os.path.exists(preprocessor_path):
            preprocessor = load_preprocessor(preprocessor_path)
        else:
            preprocessor = TextPreprocessor(use_stemmer=True)
            
        metrics = None
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                metrics = pickle.load(f)
                
        return vectorizer, model, preprocessor, metrics, None
    except Exception as e:
        return None, None, None, None, f"Terjadi kesalahan saat memuat model: {str(e)}"

vectorizer, model, preprocessor, metrics, err_msg = load_ml_components()

# ----------------------------------------------------
# 3. HELPER FUNCTIONS
# ----------------------------------------------------
def detect_sarcasm(text):
    """
    Simple heuristic rule to flag potential sarcasm/contradiction.
    """
    text_lower = text.lower()
    positive_words = ['bagus', 'mantap', 'oke', 'original', 'ori', 'cepat', 'cpt', 'baik', 'puas', 'suka']
    negative_words = ['mati', 'rusak', 'pecah', 'cacat', 'kecewa', 'nyesel', 'sesal', 'lambat', 'lemot', 'buruk']
    
    has_pos = any(w in text_lower for w in positive_words)
    has_neg = any(w in text_lower for w in negative_words)
    
    return has_pos and has_neg

# ----------------------------------------------------
# 4. SIDEBAR
# ----------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/online-shopping.png", width=150)
    st.markdown("### 🛍️ Analisis Sentimen Marketplace")
    st.markdown("Aplikasi analisis sentimen ulasan pelanggan berbasis **TF-IDF** dan **Support Vector Machine (SVM)**.")
    
    st.markdown("---")
    st.markdown("### 👥 Kelompok 4 (Tema 6):")
    st.markdown("- **Farhan Fathurrahman**")
    st.markdown("- **Muhamad Aziz Sukandar**")
    st.markdown("- **Muhammad Aden Fikri Darmawan**")
    
    st.markdown("---")
    st.markdown("### 🛠️ Fitur:")
    st.markdown("- Prapemrosesan NLP (Sastrawi Stemmer)")
    st.markdown("- Prediksi Sentimen Tunggal")
    st.markdown("- Prediksi Batch via Unggah CSV")
    st.markdown("- Evaluasi Metrik & Matriks Kebingungan")

# ----------------------------------------------------
# 5. MAIN CONTENT
# ----------------------------------------------------
st.markdown("<div class='main-title'>Analisis Sentimen & Tingkat Cortisol</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Klasifikasi Sentimen Ulasan (Low vs High Cortisol) Menggunakan Pipeline NLP Sastrawi & Support Vector Machine (SVM)</div>", unsafe_allow_html=True)

if err_msg:
    st.error(err_msg)
    st.warning("Pastikan Anda sudah menjalankan pipeline latihan dengan perintah: `python src/train.py`")
else:
    # Create Tabs
    tab_single, tab_batch, tab_metrics = st.tabs([
        "🔍 Analisis Ulasan Tunggal", 
        "📁 Analisis Batch (CSV)", 
        "📈 Performa Model & Diagnostics"
    ])
    
    # ----------------------------------------------------
    # TAB 1: SINGLE PREDICTION
    # ----------------------------------------------------
    with tab_single:
        st.markdown("### Masukkan Ulasan Pelanggan")
        
        user_input = st.text_area(
            "Tulis ulasan produk di sini (Bahasa Indonesia):",
            placeholder="Contoh: Barang bagus banget! Pengiriman super cepat dan kurir ramah.",
            height=120
        )
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            if st.button("Analisis Sentimen", type="primary"):
                if not user_input.strip():
                    st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
                else:
                    with st.spinner("Memproses teks dan melakukan prediksi..."):
                        try:
                            # Step-by-step preprocessing logs for Orchestration demo
                            steps = {}
                            
                            # Step 1: Clean
                            steps['1. Case Folding & Cleaning'] = preprocessor.clean_text(user_input)
                            # Step 2: Normalization
                            steps['2. Normalisasi Slang/Singkatan'] = preprocessor.normalize_slang(steps['1. Case Folding & Cleaning'])
                            # Step 3: Stopwords
                            steps['3. Penghapusan Stopwords'] = preprocessor.remove_stopwords(steps['2. Normalisasi Slang/Singkatan'])
                            # Step 4: Stemming
                            steps['4. Stemming (Sastrawi)'] = preprocessor.stem_text(steps['3. Penghapusan Stopwords'])
                            
                            final_text = steps['4. Stemming (Sastrawi)']
                            
                            # Vectorize
                            tfidf_feats = vectorizer.transform([final_text])
                            
                            # Predict
                            pred_class_raw = model.predict(tfidf_feats)[0]
                            pred_probs = model.predict_proba(tfidf_feats)[0]
                            class_idx = list(model.classes_).index(pred_class_raw)
                            confidence_raw = pred_probs[class_idx]
                            
                            # Refine prediction based on negation and sarcasm rules
                            pred_class, confidence, rule_triggered, rule_msg = refine_sentiment(
                                user_input, pred_class_raw, confidence_raw
                            )
                            
                            # Show Result Card
                            st.markdown("#### Hasil Analisis")
                            
                            # Color coding based on sentiment with Cortisol Theme
                            if pred_class == 'Positive':
                                sentiment_html = f"<span class='sentiment-positive'>🟢 Low Cortisol (Relaxed / Stress-Free)</span>"
                            elif pred_class == 'Neutral':
                                sentiment_html = f"<span class='sentiment-neutral'>🟡 Normal Cortisol (Steady)</span>"
                            else:
                                sentiment_html = f"<span class='sentiment-negative'>🔴 High Cortisol (Stressed / Alert)</span>"
                                
                            rule_html = ""
                            if rule_triggered:
                                rule_html = f"<p style='color:#e67e22; font-size:0.9rem; margin-top:0.5rem;'>⚠️ <b>Rule-Based Refiner</b>: {rule_msg}</p>"
                                
                            st.markdown(f"""<div class='card'>
<h5>Hasil Analisis Cortisol:</h5>
{sentiment_html}
{rule_html}
<br>
<h5>Skor Keyakinan (Confidence Score):</h5>
<h3>{confidence * 100:.2f}%</h3>
</div>""", unsafe_allow_html=True)
                            
                            # Video rendering (Discord MP4 Proxies with Autoplay & Loop)
                            if pred_class == 'Positive':
                                st.markdown("##### Agnes Tachyon wishes you Low Cortisol 🐎")
                                st.video(
                                    "https://images-ext-1.discordapp.net/external/l7KB5jD97ZZELMXPDOnGaRxbnnBeOY6c7HYI20qFBSI/https/static.klipy.com/ii/6b3a6ab13999af763b4db87b91ee67d0/28/2f/eAzsnFzVPianrKxOF.mp4",
                                    autoplay=True, loop=True, muted=True
                                )
                            elif pred_class == 'Neutral':
                                st.markdown("##### Asa Mitaka maintains Medium Cortisol 👩")
                                st.video(
                                    "https://images-ext-1.discordapp.net/external/VcM-y_VaayrQ7rieQNjJBoEEC-FjI1_19myQKqsmVIY/https/static.klipy.com/ii/4493325008d34b7bf8cd6813cd5c1619/a9/a3/vVz6B1NI5PGOrsP.mp4",
                                    autoplay=True, loop=True, muted=True
                                )
                            else:
                                st.warning("🚨 Cortisol Anda terdeteksi sangat tinggi! Redakan stres Anda dengan menonton tarian Tamamo Cross berikut:")
                                st.video(
                                    "https://images-ext-1.discordapp.net/external/4C1NfuWBSxmd0G-Rn03Yqb2SlQsRZxKx62SGyGh3TN0/https/static.klipy.com/ii/4493325008d34b7bf8cd6813cd5c1619/9c/26/cZuQPPiyiT17ueqYeWX.mp4",
                                    autoplay=True, loop=True, muted=True
                                )
                            
                            # Sarcasm Check
                            if detect_sarcasm(user_input):
                                st.info("⚠️ **Deteksi Potensi Sarkasme/Kontradiksi**: Teks ulasan mengandung kombinasi kata positif dan negatif. Mesin mungkin kesulitan menginterpretasikan konteks semantik secara akurat.")
                                
                            # Preprocessing Pipeline Steps Visualizer
                            with st.expander("🛠️ Lihat Aliran Preprocessing NLP (Orkestrasi)"):
                                for step_name, step_val in steps.items():
                                    st.markdown(f"**{step_name}**")
                                    st.code(step_val if step_val else "[Hasil Kosong]")
                                    
                        except Exception as ex:
                            st.error(f"Terjadi kesalahan saat memproses input: {str(ex)}")
                            
        with col2:
            st.markdown("#### Kamus Singkatan & Slang (Contoh)")
            st.markdown("Aplikasi menormalisasi bahasa tidak baku sebelum analisis:")
            
            # Show a slice of slang dictionary
            slang_sample = pd.DataFrame(
                list(SLANG_DICT.items())[:10], 
                columns=['Kata Slang/Singkatan', 'Kata Baku']
            )
            st.table(slang_sample)
            st.caption("Dan ratusan kata slang/singkatan pasar marketplace lainnya.")
            
    # ----------------------------------------------------
    # TAB 2: BATCH PREDICTION
    # ----------------------------------------------------
    with tab_batch:
        st.markdown("### Unggah File CSV untuk Analisis Batch")
        st.write("File CSV harus memiliki kolom teks ulasan (misal: `content`).")
        
        uploaded_file = st.file_uploader("Pilih file CSV:", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df_batch = pd.read_csv(uploaded_file)
                st.write("Pratinjau File yang Diunggah:")
                st.dataframe(df_batch.head(5))
                
                columns = df_batch.columns.tolist()
                text_col = st.selectbox("Pilih kolom yang berisi ulasan:", columns, index=0)
                
                if st.button("Mulai Analisis Batch", type="primary"):
                    with st.spinner("Memproses analisis ulasan batch (bisa memakan waktu tergantung jumlah baris)..."):
                        start_batch = time.time()
                        
                        # Process texts (without stemmer to make batch fast on large files if they wish, or use the preprocessor cache)
                        # We will use preprocessor.preprocess to do it robustly
                        progress_bar = st.progress(0)
                        
                        clean_texts = []
                        total_rows = len(df_batch)
                        
                        # Preprocess text
                        for idx, text in enumerate(df_batch[text_col]):
                            clean_t = preprocessor.preprocess(str(text))
                            clean_texts.append(clean_t)
                            progress_bar.progress((idx + 1) / total_rows)
                        
                        # Extract TF-IDF
                        tfidf_batch = vectorizer.transform(clean_texts)
                        
                        # Predict
                        raw_predictions = model.predict(tfidf_batch)
                        probabilities = model.predict_proba(tfidf_batch)
                         
                        predictions = []
                        max_probs = []
                         
                        # Apply Refiner on Batch predictions
                        for i, orig_text in enumerate(df_batch[text_col]):
                            raw_pred = raw_predictions[i]
                            class_idx = list(model.classes_).index(raw_pred)
                            raw_conf = probabilities[i][class_idx]
                             
                            refined_pred, refined_conf, _, _ = refine_sentiment(
                                str(orig_text), raw_pred, raw_conf
                            )
                            predictions.append(refined_pred)
                            max_probs.append(refined_conf)
                         
                        df_batch['Cleaned_Text'] = clean_texts
                        df_batch['Predicted_Sentiment'] = predictions
                        df_batch['Confidence_Score'] = max_probs
                        
                        st.success(f"Analisis Batch selesai dalam {time.time() - start_batch:.2f} detik!")
                        
                        # Results Distribution Chart
                        st.markdown("#### Distribusi Sentimen Prediksi")
                        fig, ax = plt.subplots(figsize=(6, 3))
                        sns.countplot(x='Predicted_Sentiment', data=df_batch, palette='viridis', ax=ax)
                        ax.set_title("Jumlah Ulasan per Sentimen")
                        st.pyplot(fig)
                        
                        # Display Predictions
                        st.dataframe(df_batch[[text_col, 'Cleaned_Text', 'Predicted_Sentiment', 'Confidence_Score']].head(20))
                        
                        # Download Link
                        csv_data = df_batch.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Unduh Hasil Prediksi CSV",
                            data=csv_data,
                            file_name="hasil_prediksi_sentimen.csv",
                            mime="text/csv"
                        )
            except Exception as e_batch:
                st.error(f"Kesalahan memproses file CSV: {str(e_batch)}")
                
    # ----------------------------------------------------
    # TAB 3: DIAGNOSTICS & METRICS
    # ----------------------------------------------------
    with tab_metrics:
        st.markdown("### Performa Evaluasi Model SVM + TF-IDF")
        
        if metrics:
            st.markdown(f"Evaluasi dilakukan pada data uji berukuran **{metrics['test_samples']} sampel** (dari total dataset {metrics['train_samples'] + metrics['test_samples']} ulasan).")
            
            # Metric Cards
            acc = metrics['accuracy']
            rep = metrics['classification_report']
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h5 style='color:#7f8c8d;margin:0;'>Accuracy</h5>
                    <h2 style='color:#FF4B2B;margin:0;font-size:2.2rem;'>{acc * 100:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h5 style='color:#7f8c8d;margin:0;'>Precision (Pos)</h5>
                    <h2 style='color:#2ecc71;margin:0;font-size:2.2rem;'>{rep['Positive']['precision'] * 100:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            with col_m3:
                st.markdown(f"""
                <div class='metric-card'>
                    <h5 style='color:#7f8c8d;margin:0;'>Recall (Pos)</h5>
                    <h2 style='color:#3498db;margin:0;font-size:2.2rem;'>{rep['Positive']['recall'] * 100:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            with col_m4:
                st.markdown(f"""
                <div class='metric-card'>
                    <h5 style='color:#7f8c8d;margin:0;'>F1-Score (W. Avg)</h5>
                    <h2 style='color:#9b59b6;margin:0;font-size:2.2rem;'>{rep['weighted avg']['f1-score'] * 100:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            
            # Detailed Class Performance Table
            st.markdown("#### Detail Metrik Per Kelas")
            classes = ['Negative', 'Neutral', 'Positive']
            class_data = []
            for c in classes:
                class_data.append({
                    'Kelas': c,
                    'Precision': f"{rep[c]['precision'] * 100:.1f}%",
                    'Recall': f"{rep[c]['recall'] * 100:.1f}%",
                    'F1-Score': f"{rep[c]['f1-score'] * 100:.1f}%",
                    'Support': rep[c]['support']
                })
            st.table(pd.DataFrame(class_data))
            
            # Confusion Matrix Plot
            st.markdown("#### Matriks Kebingungan (Confusion Matrix)")
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            cm = np.array(metrics['confusion_matrix'])
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax_cm
            )
            ax_cm.set_xlabel("Prediksi")
            ax_cm.set_ylabel("Sebenarnya")
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm)
            
        else:
            st.info("File metrik evaluasi tidak ditemukan. Harap pastikan training menghasilkan evaluation_metrics.pkl.")
