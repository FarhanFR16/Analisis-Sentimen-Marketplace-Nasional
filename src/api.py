from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import pickle
import os
import sys

# Add src to python path
sys.path.append(os.path.dirname(__file__))
from preprocess import TextPreprocessor, load_preprocessor, refine_sentiment

app = FastAPI(
    title="Sentiment Analysis API",
    description="REST API standard untuk deteksi sentimen ulasan marketplace Indonesia menggunakan TF-IDF dan SVM.",
    version="1.0.0"
)

# Global variables for models
model = None
vectorizer = None
preprocessor = None

MODELS_DIR = r"c:\git\UAS-AI\Analisis-Sentimen-Marketplace-Nasional\models"

# ----------------------------------------------------
# Request and Response Models
# ----------------------------------------------------
class SentimentRequest(BaseModel):
    text: str = Field(..., example="Barang bagus banget! Pengiriman cepat.", description="Teks ulasan produk.")

class SentimentResponse(BaseModel):
    status: str
    prediction: str
    confidence: float
    clean_text: str

class ErrorResponse(BaseModel):
    status: str
    message: str

# ----------------------------------------------------
# Startup Event: Load ML models
# ----------------------------------------------------
@app.on_event("startup")
def startup_event():
    global model, vectorizer, preprocessor
    
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    model_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    
    try:
        if not (os.path.exists(vectorizer_path) and os.path.exists(model_path)):
            print("WARNING: Model files not found. Run training script first.")
            return
            
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        if os.path.exists(preprocessor_path):
            preprocessor = load_preprocessor(preprocessor_path)
        else:
            preprocessor = TextPreprocessor(use_stemmer=True)
            
        print("ML Models loaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to load models during startup: {str(e)}")

# ----------------------------------------------------
# API ENDPOINTS
# ----------------------------------------------------
@app.get("/health", response_model=ErrorResponse)
def health_check():
    """
    Check if the service is healthy and models are loaded.
    """
    if model is None or vectorizer is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "unhealthy", "message": "ML Models are not loaded."}
        )
    return {"status": "healthy", "message": "Service is ready to predict."}

@app.post(
    "/predict", 
    response_model=SentimentResponse, 
    responses={
        400: {"model": ErrorResponse, "description": "Invalid Input"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Model Not Ready"}
    }
)
def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment for a given marketplace review text.
    """
    # 1. Validation & Try-Except Containment Unit 1
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"status": "error", "message": "Input teks tidak boleh kosong atau hanya berupa spasi."}
        )
    
    if model is None or vectorizer is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "error", "message": "Model belum dimuat. Hubungi administrator."}
        )
        
    try:
        # 2. NLP Pipeline Preprocessing
        clean_txt = preprocessor.preprocess(request.text)
        
        if not clean_txt.strip():
            # If preprocessing removed all words, e.g., input was only punctuation/stopwords
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"status": "error", "message": "Prapemrosesan menghasilkan teks kosong (teks input tidak mengandung kata bermakna)."}
            )
            
        # 3. Vectorization & Inference
        tfidf_feats = vectorizer.transform([clean_txt])
        pred_class_raw = model.predict(tfidf_feats)[0]
        
        # Calculate confidence score
        probabilities = model.predict_proba(tfidf_feats)[0]
        class_idx = list(model.classes_).index(pred_class_raw)
        confidence_raw = float(probabilities[class_idx])
        
        # Refine prediction based on negation and sarcasm rules
        prediction, confidence, _, _ = refine_sentiment(
            request.text, pred_class_raw, confidence_raw
        )
        
        return {
            "status": "success",
            "prediction": prediction,
            "confidence": confidence,
            "clean_text": clean_txt
        }
        
    except HTTPException as http_ex:
        # Re-raise HTTP exceptions
        raise http_ex
    except Exception as ex:
        # Catch unexpected exceptions to avoid crash
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "error", "message": f"Terjadi kesalahan internal: {str(ex)}"}
        )
