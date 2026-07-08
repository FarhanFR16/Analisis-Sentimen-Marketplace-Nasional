import os
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import TextPreprocessor, save_preprocessor

def map_score_to_sentiment(score):
    if score in [4, 5]:
        return 'Positive'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Negative'

def main():
    print("=== Sentiment Analysis SVM Training Pipeline ===")
    
    # Define paths
    data_dir = r"c:\git\UAS-AI\Analisis-Sentimen-Marketplace-Nasional\data"
    models_dir = r"c:\git\UAS-AI\Analisis-Sentimen-Marketplace-Nasional\models"
    os.makedirs(models_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, "Shopee_Sampled_Reviews.csv")
    preprocessed_csv_path = os.path.join(data_dir, "preprocessed_reviews.csv")
    
    # 1. Load Data
    print(f"Loading raw data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Map score to sentiment
    df['sentiment'] = df['score'].apply(map_score_to_sentiment)
    print("Class distribution after mapping:")
    print(df['sentiment'].value_counts())
    
    # Initialize Preprocessor
    preprocessor = TextPreprocessor(use_stemmer=True)
    
    # 2. Preprocess or Load Preprocessed Data
    if os.path.exists(preprocessed_csv_path):
        print(f"Preprocessed data found at {preprocessed_csv_path}. Loading cached version...")
        df_processed = pd.read_csv(preprocessed_csv_path)
        # Ensure we fill NaN in case any review was completely cleaned to empty
        df_processed['clean_content'] = df_processed['clean_content'].fillna("")
    else:
        print("Preprocessed data not found. Running prapemrosesan pipeline (this may take 1-2 minutes)...")
        start_time = time.time()
        
        # Run parallel preprocessing
        cleaned_texts = preprocessor.preprocess_parallel(df['content'].tolist(), n_jobs=4)
        
        df['clean_content'] = cleaned_texts
        df_processed = df[['reviewId', 'content', 'clean_content', 'score', 'sentiment']]
        
        # Save to csv for future runs
        df_processed.to_csv(preprocessed_csv_path, index=False)
        print(f"Prapemrosesan selesai dalam {time.time() - start_time:.2f} detik. Saved to: {preprocessed_csv_path}")
    
    # 3. Train-Test Split
    X = df_processed['clean_content']
    y = df_processed['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training size: {len(X_train)}, Testing size: {len(X_test)}")
    
    # 4. Feature Extraction: TF-IDF
    print("Extracting features using TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 5. Model Training: Support Vector Machine (SVM)
    print("Training Support Vector Machine (SVM) classifier...")
    svm_start = time.time()
    # SVC probability=True allows prediction probability scores
    model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    model.fit(X_train_tfidf, y_train)
    print(f"SVM training completed in {time.time() - svm_start:.2f} seconds.")
    
    # 6. Evaluation
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['Negative', 'Neutral', 'Positive'])
    
    print("\n" + "="*40)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("="*40)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(conf_matrix)
    print("="*40 + "\n")
    
    # 7. Save Models and Objects
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    model_path = os.path.join(models_dir, "svm_model.pkl")
    metrics_path = os.path.join(models_dir, "evaluation_metrics.pkl")
    preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
    
    print(f"Saving TF-IDF Vectorizer to: {vectorizer_path}")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print(f"Saving SVM Model to: {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    # Save training metrics for app display
    metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'test_samples': len(X_test),
        'train_samples': len(X_train),
        'class_distribution': df_processed['sentiment'].value_counts().to_dict()
    }
    print(f"Saving evaluation metrics to: {metrics_path}")
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
        
    print(f"Saving Preprocessor instance to: {preprocessor_path}")
    save_preprocessor(preprocessor, preprocessor_path)
    
    print("Training pipeline finished successfully!")

if __name__ == "__main__":
    main()
