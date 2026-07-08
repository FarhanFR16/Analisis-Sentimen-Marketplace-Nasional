import os
import zipfile

try:
    import kaggle
except OSError as e:
    print(f"Kaggle API error: {e}")
    print("Please set up Kaggle credentials or KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
    exit(1)

def download_dataset():
    dataset_name = "satyaahb/e-commerce-sampled-reviews-in-bahasa-indonesia"
    download_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    base_path = os.path.dirname(os.path.dirname(__file__))
    
    os.makedirs(download_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, "data", "processed"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "notebooks"), exist_ok=True)
    
    print(f"Downloading dataset {dataset_name} to {download_path}...")
    try:
        kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        print("Dataset downloaded and extracted successfully.")
        
        # List downloaded files
        for f in os.listdir(download_path):
            print(f"Downloaded file: {f}")
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure your Kaggle API credentials are set up correctly.")

if __name__ == "__main__":
    download_dataset()
