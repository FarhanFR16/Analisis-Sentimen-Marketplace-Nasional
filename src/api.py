from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import os
import sys

# Tambahkan src ke system path untuk import preprocess
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_pipeline

app = FastAPI(
    title="API Analisis Sentimen",
    description="REST API untuk memprediksi sentimen teks ulasan e-commerce menggunakan SVM dan TF-IDF",
    version="1.0.0"
)

# Load Model
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "svm_tfidf_pipeline.pkl")
pipeline = None

@app.on_event("startup")
def load_model():
    global pipeline
    try:
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
        print("Model berhasil dimuat.")
    except FileNotFoundError:
        print(f"Peringatan: File model tidak ditemukan di {model_path}. Harap jalankan src/train.py terlebih dahulu.")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    status: str

@app.post("/predict", response_model=PredictResponse)
async def predict_sentiment(request: PredictRequest):
    # Containment Unit 1: Try-Except Blocks for Robustness
    try:
        text = request.text
        if not text or len(text.strip()) == 0:
            raise ValueError("Teks tidak boleh kosong.")
            
        if pipeline is None:
            raise RuntimeError("Model belum dimuat di server.")
            
        # Prediksi langsung dengan pipeline (pipeline akan melakukan tf-idf)
        # Tapi kita perlu melakukan preprocessing NLP manual dulu (Sastrawi dll)
        preprocessed_texts = preprocess_pipeline([text])
        cleaned_text = preprocessed_texts[0]
        
        # Inference
        prediction = pipeline.predict([cleaned_text])[0]
        probabilities = pipeline.predict_proba([cleaned_text])[0]
        
        confidence = float(max(probabilities))
        label = str(prediction)
        
        return PredictResponse(
            sentiment=label,
            confidence=confidence,
            status="success"
        )
        
    except ValueError as ve:
        # Human-readable error message (Bad Request)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Human-readable error message (Internal Server Error)
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal pada server saat memproses prediksi.")

@app.get("/")
def read_root():
    return {"message": "API Analisis Sentimen Marketplace aktif. Gunakan endpoint POST /predict"}
