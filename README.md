# Analisis Sentimen Marketplace Nasional

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E.svg?logo=scikit-learn)

Proyek ini merupakan implementasi sistem *End-to-End Artificial Intelligence* untuk melakukan Analisis Sentimen terhadap ulasan pelanggan di marketplace nasional (seperti Shopee dan Tokopedia).

Sistem ini dikembangkan menggunakan **Support Vector Machine (SVM)** dengan ekstraksi fitur **TF-IDF**, dan melalui tahapan pra-pemrosesan Natural Language Processing (NLP) yang mencakup tokenisasi dan stemming menggunakan **Sastrawi**.

## Architecture Diagram

```mermaid
graph LR
    A[User / Streamlit UI] -->|Input Teks (JSON POST)| B(FastAPI Backend)
    B --> C{NLP Preprocessing}
    C -->|Case Folding, Sastrawi| D(TF-IDF Vectorizer)
    D --> E((SVM Classifier))
    E -->|Sentimen & Confidence| B
    B -->|Response JSON| A
```

## Struktur Repositori

```text
📦 Analisis-Sentimen-Marketplace-Nasional
 ┣ 📂 data/
 ┃ ┣ 📂 raw/              # Dataset mentah dari Kaggle
 ┃ ┗ 📂 processed/        # (Opsional) Dataset setelah preprocessing
 ┣ 📂 models/             # File model (.pkl)
 ┣ 📂 notebooks/          # Eksperimen awal (EDA, Tuning)
 ┣ 📂 src/
 ┃ ┣ 📜 setup_data.py     # Script unduh dataset
 ┃ ┣ 📜 preprocess.py     # Modul pipeline NLP (Sastrawi)
 ┃ ┣ 📜 train.py          # Script pelatihan model SVM & ekspor
 ┃ ┣ 📜 api.py            # FastAPI REST Endpoint
 ┃ ┗ 📜 app.py            # Streamlit Frontend
 ┣ 📜 README.md
 ┗ 📜 requirements.txt
```

## Setup Guide

### 1. Instalasi Kebutuhan

Pastikan Python telah terinstal, lalu jalankan:

```bash
git checkout dev/aden
pip install -r requirements.txt
```

### 2. Akuisisi Dataset

Unduh dataset dari link berikut ke folder `data/raw/`:
[Dataset Kaggle: e-commerce-sampled-reviews-in-bahasa-indonesia](https://www.kaggle.com/datasets/satyaahb/e-commerce-sampled-reviews-in-bahasa-indonesia)

Atau gunakan script (membutuhkan `kaggle.json` yang sudah dikonfigurasi):
```bash
python src/setup_data.py
```

### 3. Melatih Model

Jalankan script untuk memproses data dan melatih model SVM:
```bash
python src/train.py
```
*(Script ini akan menyimpan model di `models/svm_tfidf_pipeline.pkl`)*

### 4. Menjalankan REST API (Backend)

Jalankan server FastAPI:
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```
API akan berjalan di `http://localhost:8000`. Dokumentasi Swagger UI tersedia di `http://localhost:8000/docs`.

### 5. Menjalankan Streamlit App (Frontend)

Buka terminal baru dan jalankan Streamlit:
```bash
streamlit run src/app.py
```
Aplikasi web akan terbuka otomatis di browser Anda.

## API Documentation

Endpoint utama untuk prediksi sentimen:
- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`

### JSON Request Format (Input)
```json
{
  "text": "Barangnya bagus banget, pengiriman super cepat!"
}
```

### JSON Response Format - Success (200 OK)
```json
{
  "sentiment": "Positif",
  "confidence": 0.8924,
  "status": "success"
}
```

### JSON Response Format - Error (400 Bad Request)
```json
{
  "detail": "Teks tidak boleh kosong."
}
```
*(Containment Unit 1 telah diimplementasikan untuk mencegah server crash saat menerima payload tidak valid)*
