import unittest
import sys
import os
import pickle

# Add src directory to path
sys.path.append(os.path.dirname(__file__))
from preprocess import TextPreprocessor, refine_sentiment
from api import app
from fastapi.testclient import TestClient

class TestSentimentPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.preprocessor = TextPreprocessor(use_stemmer=True)
        cls.client = TestClient(app)
        
        # Load vectorizer and model to verify they exist
        models_dir = r"c:\git\UAS-AI\Analisis-Sentimen-Marketplace-Nasional\models"
        cls.vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
        cls.model_path = os.path.join(models_dir, "svm_model.pkl")

    def test_text_cleaning(self):
        """Test that cleaning removes punctuation, usernames, links, and repeated characters."""
        raw_text = "Wah!!! Sangat bagussss @user https://shopee.co.id/test #mantap"
        expected = "wah sangat bagussss user mantap"
        
        cleaned = self.preprocessor.clean_text(raw_text)
        # Check basic cleanup
        self.assertIn("wah", cleaned)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("@user", cleaned)

    def test_slang_normalization(self):
        """Test that slang is correctly mapped to standard Indonesian."""
        slang_text = "yg trs tp g lemot udh bgt"
        expected = "yang terus tapi tidak lambat sudah banget"
        normalized = self.preprocessor.normalize_slang(slang_text)
        self.assertEqual(normalized, expected)

    def test_negation_words_preservation(self):
        """Test that negation words are NOT removed as stopwords."""
        text = "tidak bagus tapi lambat"
        # "tidak" and "tapi" should remain
        no_stopwords = self.preprocessor.remove_stopwords(text)
        self.assertIn("tidak", no_stopwords)
        self.assertIn("tapi", no_stopwords)

    def test_sastrawi_stemming(self):
        """Test that Sastrawi stemming runs correctly on a single word."""
        word = "menyelesaikan"
        stemmed = self.preprocessor.stem_word(word)
        self.assertEqual(stemmed, "selesai")

    def test_model_files_exist(self):
        """Verify that training generated the model and vectorizer pkl files."""
        self.assertTrue(os.path.exists(self.vectorizer_path), "TF-IDF Vectorizer file missing!")
        self.assertTrue(os.path.exists(self.model_path), "SVM Model file missing!")

    def test_api_health(self):
        """Test FastAPI health check endpoint."""
        with TestClient(app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "healthy")

    def test_api_predict_positive(self):
        """Test FastAPI prediction for a positive review."""
        payload = {"text": "Barang sangat bagus sekali, pengiriman super cepat dan packing rapi!"}
        with TestClient(app) as client:
            response = client.post("/predict", json=payload)
            self.assertEqual(response.status_code, 200)
            json_data = response.json()
            self.assertEqual(json_data["status"], "success")
            self.assertIn("prediction", json_data)
            self.assertIn("confidence", json_data)
            self.assertIn("clean_text", json_data)

    def test_api_predict_empty_error(self):
        """Test FastAPI validation handles empty text with 400 Bad Request."""
        payload = {"text": ""}
        with TestClient(app) as client:
            response = client.post("/predict", json=payload)
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["detail"]["status"], "error")

    def test_refine_sentiment_negation(self):
        """Test that refine_sentiment overrides positive SVM predictions when negations are present."""
        # e.g., SVM predicts Positive but text has 'tidak membantu'
        pred, conf, triggered, msg = refine_sentiment("barang ini sangat tidak membantu", "Positive", 0.85)
        self.assertEqual(pred, "Negative")
        self.assertTrue(triggered)
        self.assertIn("tidak membantu", msg.lower())

    def test_refine_sentiment_sarcasm(self):
        """Test that refine_sentiment overrides positive SVM predictions when sarcasm/contradiction is present."""
        # e.g., SVM predicts Positive but text has 'bagus tapi langsung mati'
        pred, conf, triggered, msg = refine_sentiment("bagus sih tapi sayang layarnya mati total", "Positive", 0.90)
        self.assertEqual(pred, "Negative")
        self.assertTrue(triggered)
        self.assertIn("sarkasme", msg.lower())

if __name__ == "__main__":
    unittest.main()
