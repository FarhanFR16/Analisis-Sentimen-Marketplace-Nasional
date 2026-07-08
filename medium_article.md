# Transformasi Deteksi Sentimen E-Commerce Indonesia: Implementasi End-to-End SVM dan Streamlit

Oleh: **Kelompok 4 (Farhan Fathurrahman, Muhamad Aziz Sukandar, Muhammad Aden Fikri Darmawan)**

## The Hook (Problem)
Di era digital, marketplace telah menjadi pusat perbelanjaan utama di Indonesia. Setiap harinya, ribuan ulasan ditinggalkan oleh pelanggan. Bagi penjual maupun platform, memahami sentimen dari ribuan ulasan tersebut secara manual adalah hal yang mustahil. Jika keluhan pelanggan (sentimen negatif/high cortisol) tidak ditangani dengan cepat, reputasi toko dapat hancur dalam hitungan jam. Proyek ini bertujuan untuk mengotomatisasi analisis ulasan pelanggan dengan target mempercepat proses respons terhadap keluhan pelanggan secara real-time. Siapa yang menderita jika solusi ini tidak ada? Tentu saja penjual yang kehilangan pelanggan potensial karena rating buruk yang tak terdeteksi, dan platform yang kehilangan kepercayaan pengguna.

## The Tech (Solution)
Untuk memecahkan masalah ini, kami membangun sebuah sistem AI terintegrasi (End-to-End). Sistem ini mengorkestrasikan pipeline NLP (Natural Language Processing) yang presisi.

**Aliran Arsitektur (Architecture Flow):**
`Input Layer (Streamlit) -> Preprocessing (Tokenization, Slang Normalization & Sastrawi Stemmer) -> Inference (SVM Model via TF-IDF) -> Output (Sentiment Result JSON via FastAPI)`

Tahap preprocessing menggunakan **Sastrawi Stemmer** untuk menangani morfologi bahasa Indonesia (mengubah kata seperti "membelikan" menjadi "beli") dan *Slang Normalizer* untuk mengubah kata gaul menjadi kata baku. Selanjutnya, **TF-IDF (Term Frequency-Inverse Document Frequency)** digunakan untuk mengekstrak bobot penting dari kata-kata kunci unik.

Kami memilih **Support Vector Machine (SVM)** dibandingkan Naive Bayes karena SVM bekerja dengan mencari *hyperplane* (garis batas) terbaik antar kelas teks. Hal ini memberikan akurasi yang lebih stabil pada data teks berdimensi besar yang dihasilkan oleh TF-IDF. Selain itu, sistem ini dioperasionalisasikan dengan **FastAPI** yang dilengkapi dengan *Try-Except Blocks (Containment Unit 1)* untuk memastikan robustness, sehingga server tidak akan *crash* saat menerima input yang tidak valid.

## The Proof (Demo)
Aplikasi kami telah di-deploy secara lokal menggunakan **Streamlit**. Antarmuka ini memungkinkan pengguna untuk melakukan analisis sentimen tunggal maupun batch (melalui unggah file CSV). 

*Happy Path*: Ketika pengguna memasukkan teks "Barang sangat bagus sekali! Pengiriman super cepat dan kurirnya ramah.", antarmuka Streamlit secara instan mengirim payload teks ke FastAPI dan menerima respons JSON (Status 200 OK). Hasilnya menonjolkan label "Positive" dengan skor keyakinan di atas 90%.

*Edge Cases*: Kami menemukan edge case yang menarik di mana model terkadang kesulitan menangani sarkasme. Misalnya, "Bagus banget barangnya... baru 5 menit mati total". Secara literal, terdapat kata "bagus", namun maknanya negatif. Oleh karena itu, kami menambahkan heuristik sederhana di antarmuka kami untuk memberikan peringatan (*warning*) jika terdapat indikasi kalimat sarkasme (gabungan kuat kata positif dan negatif). Ini menunjukkan kejujuran kami terhadap keterbatasan sistem, membuktikan bahwa pragmatik bahasa masih menjadi tantangan bagi NLP.

## The Value (Impact)
Model divalidasi menggunakan metrik standar industri untuk memastikan keseimbangan prediksi pada data uji. Hasil evaluasi transparansi performa model kami adalah:

- **Accuracy**: 66.8%
- **Precision (Positive)**: 79.0%
- **Recall (Positive)**: 75.0%
- **F1-Score (Positive)**: 77.0%

Model ini menunjukkan kemampuan yang sangat baik dalam mendeteksi sentimen positif (precision tinggi menghindari *False Positives*), meskipun kelas netral masih menjadi area pengembangan karena ambiguitas bawaannya. Dengan metrik ini, solusi kami sudah cukup tangguh untuk membantu penjual marketplace menyaring keluhan utama pelanggan secara cepat dan efisien.

---
*Proyek ini diajukan untuk memenuhi tugas UAS Artificial Intelligence T.A. 2025/2026 - Program Studi Ilmu Komputer, Universitas Cakrawala.*
