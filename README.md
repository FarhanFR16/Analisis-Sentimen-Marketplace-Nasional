# 🛍️ Analisis Sentimen Marketplace Nasional (Tema 6)

[![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B2B.svg)](https://streamlit.io/)
[![FastAPI API](https://img.shields.io/badge/FastAPI-REST_API-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Repositori ini berisi implementasi lengkap untuk **Analisis Sentimen Ulasan Marketplace Nasional (Shopee)** menggunakan kombinasi **NLP Preprocessing (Sastrawi Stemmer & Slang Normalizer)** dan **Support Vector Machine (SVM)** dengan pembobotan **TF-IDF**. 

Proyek ini dibangun sebagai bagian dari penilaian **UAS Kecerdasan Buatan (AI-03)** oleh **Kelompok 4**:
* **Farhan Fathurrahman**
* **Muhamad Aziz Sukandar**
* **Muhammad Aden Fikri Darmawan**

---

## 📐 Arsitektur Sistem (Pipeline AI)

Berikut adalah diagram alir dari prapemrosesan ulasan hingga menghasilkan hasil klasifikasi sentimen dan skor keyakinan:

```mermaid
graph TD
    A[Input Ulasan / Payload JSON] --> B[NLP Preprocessing]
    subgraph B [NLP Preprocessing Pipeline]
        B1[Case Folding & Cleaning] --> B2[Slang Normalization]
        B2 --> B3[Stopwords Removal (Negation Preserved)]
        B3 --> B4[Stemming (Sastrawi Stemmer)]
    end
    B4 --> C[TF-IDF Vectorization]
    C --> D[SVM Classifier Inference]
    D --> E[Output Result]
    subgraph E [Output Layer]
        E1[Predicted Sentiment Label: Positive/Neutral/Negative]
        E2[Confidence Score %]
    end
```

---

## 🚀 Panduan Memulai (Setup Guide)

### 1. Prasyarat
Pastikan Anda sudah menginstal Python 3.10+ di komputer Anda.

### 2. Kloning Repositori
```bash
git clone https://github.com/FarhanFR16/Analisis-Sentimen-Marketplace-Nasional.git
cd Analisis-Sentimen-Marketplace-Nasional
```

### 3. Instalasi Dependensi
Instal semua pustaka yang dibutuhkan menggunakan `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Menjalankan Model Training Pipeline
Untuk memproses dataset Shopee, melatih model SVM, dan menyimpan bobot model (`.pkl`), jalankan perintah:
```bash
python src/train.py
```
*Catatan: Pipeline prapemrosesan menggunakan optimasi kamus kata unik dan multiprocessing, sehingga hanya membutuhkan waktu sekitar ~2 menit dibandingkan ~17 menit jika berjalan normal.*

### 5. Menjalankan Aplikasi Web (Streamlit)
Untuk membuka dasbor antarmuka interaktif:
```bash
streamlit run app.py
```

### 6. Menjalankan Server REST API (FastAPI)
Untuk menjalankan server API lokal untuk diintegrasikan dengan aplikasi lain:
```bash
uvicorn src.api:app --reload
```
Server akan berjalan di `http://127.0.0.1:8000`. Dokumentasi Swagger interaktif dapat diakses di `http://127.0.0.1:8000/docs`.

---

## 🧪 Pengujian Unit (Unit Testing)
Kami telah menyediakan rangkaian tes otomatis untuk memverifikasi fungsionalitas sistem:
```bash
python -m unittest src/test_pipeline.py
```

---

## 🔌 Dokumentasi REST API

### **1. Health Check**
Memeriksa status kesehatan server dan kesiapan model AI.
* **Method & URL**: `GET /health`
* **Response (200 OK)**:
  ```json
  {
    "status": "healthy",
    "message": "Service is ready to predict."
  }
  ```
* **Response (503 Service Unavailable)**:
  ```json
  {
    "status": "error",
    "message": "ML Models are not loaded."
  }
  ```

### **2. Prediksi Sentimen**
Melakukan klasifikasi sentimen terhadap ulasan masukan.
* **Method & URL**: `POST /predict`
* **Headers**: `Content-Type: application/json`
* **Request Body (JSON)**:
  ```json
  {
    "text": "Barang sangat bagus sekali! Pengiriman super cepat dan kurirnya ramah."
  }
  ```
* **Response Sukses (200 OK)**:
  ```json
  {
    "status": "success",
    "prediction": "Positive",
    "confidence": 0.9388120392,
    "clean_text": "barang sangat bagus sekali kirim super cepat kurir ramah"
  }
  ```
* **Response Request Salah (400 Bad Request)**:
  ```json
  {
    "detail": {
      "status": "error",
      "message": "Input teks tidak boleh kosong atau hanya berupa spasi."
    }
  }
  ```
* **Response Kesalahan Server (500 Internal Server Error)**:
  ```json
  {
    "detail": {
      "status": "error",
      "message": "Terjadi kesalahan internal: [deskripsi error]"
    }
  }
  ```

---

## 📈 Metrik Evaluasi Model

Model divalidasi pada data uji terpisah (20% split) dan mendapatkan hasil berikut:
- **Akurasi Keseluruhan (Accuracy)**: **67.2%**
- **Precision (Positive)**: **78.0%**
- **Recall (Positive)**: **76.0%**
- **F1-Score (Positive)**: **77.0%**

*Model ini menunjukkan performa terbaik dalam membedakan ulasan Positif dan Negatif secara konsisten, namun mengalami keterbatasan pada ulasan Neutral karena jumlah data yang lebih sedikit dan karakteristik ulasan netral yang ambigu.*
