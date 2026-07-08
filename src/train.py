import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle
import os
import glob
from preprocess import preprocess_pipeline

def train_model():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    # Temukan file csv di data/raw
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print("Data CSV tidak ditemukan di data/raw. Harap download dataset Kaggle dan ekstrak ke sana.")
        # Buat dummy data jika tidak ada (untuk memastikan pipeline berjalan)
        print("Membuat synthetic dummy data untuk keperluan pengujian pipeline...")
        df = pd.DataFrame({
            'text': ["Barangnya bagus banget", "Pengiriman cepat, seller ramah", "Barang rusak, sangat mengecewakan", "Biasa saja, sesuai harga", "Tidak direkomendasikan, pengiriman lambat", "Cukup untuk sementara", "Standar, tidak ada yang spesial"],
            'label': ["Low Cortisol", "Low Cortisol", "High Cortisol", "No Cortisol", "High Cortisol", "No Cortisol", "No Cortisol"]
        })
    else:
        # Gunakan csv pertama yang ditemukan
        print(f"Membaca dataset dari {csv_files[0]}")
        df = pd.read_csv(csv_files[0])
        # Coba deteksi kolom teks dan label
        text_cols = [c for c in df.columns if 'text' in c.lower() or 'review' in c.lower()]
        label_cols = [c for c in df.columns if 'label' in c.lower() or 'sentiment' in c.lower() or 'rating' in c.lower()]
        
        if text_cols and label_cols:
            df = df.rename(columns={text_cols[0]: 'text', label_cols[0]: 'label'})
        else:
            print("Peringatan: Tidak dapat mendeteksi kolom text dan label secara otomatis. Menggunakan 2 kolom pertama.")
            df = df.rename(columns={df.columns[0]: 'text', df.columns[1]: 'label'})
            
        # Drop missing values
        df = df.dropna(subset=['text', 'label'])
        
        # Mapping kelas sentimen ke: Low Cortisol (Positif), High Cortisol (Negatif), No Cortisol (Netral)
        if df['label'].dtype == 'object' or isinstance(df['label'].iloc[0], str):
            df['label'] = df['label'].astype(str).str.lower()
            conditions = [
                df['label'].str.contains('pos|1|baik|bagus'),
                df['label'].str.contains('neg|0|jelek|buruk|rusak|kecewa')
            ]
            choices = ['Low Cortisol', 'High Cortisol']
            df['label'] = np.select(conditions, choices, default='No Cortisol')
        else:
            # Jika dataset memiliki rating numerik (misal 1-5)
            conditions = [
                df['label'] >= 4,
                df['label'] <= 2
            ]
            choices = ['Low Cortisol', 'High Cortisol']
            df['label'] = np.select(conditions, choices, default='No Cortisol')
            
    print(f"Total data: {len(df)}")
    print("Distribusi Kelas:")
    print(df['label'].value_counts())
    
    # Preprocessing
    print("Memulai preprocessing (Sastrawi Stemming dll)...")
    X = preprocess_pipeline(df['text'].tolist())
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pipeline: TF-IDF + SVM
    print("Melatih model SVM dengan TF-IDF (3 Kelas)...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('svm', SVC(kernel='linear', probability=True, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluasi
    y_pred = pipeline.predict(X_test)
    print("\n--- Hasil Evaluasi ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    # Menggunakan macro avg untuk multiclass classification
    print(f"Precision (Macro): {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall (Macro):    {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1-Score (Macro):  {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    
    # Export Model
    model_path = os.path.join(models_dir, "svm_tfidf_pipeline.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Model berhasil disimpan di {model_path}")

if __name__ == "__main__":
    train_model()
