import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Case folding
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_pipeline(texts):
    print("Mulai membersihkan teks...")
    cleaned_texts = [clean_text(t) for t in texts]
    
    print("Mulai stemming dengan Sastrawi (ini mungkin memakan waktu)...")
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    stemmed_texts = [stemmer.stem(t) for t in cleaned_texts]
    return stemmed_texts

if __name__ == "__main__":
    # Test script
    sample = ["Bagus banget barangnya... baru 5 menit mati total", "Pengiriman super cepat dan seller ramah!"]
    print("Original:", sample)
    print("Processed:", preprocess_pipeline(sample))
