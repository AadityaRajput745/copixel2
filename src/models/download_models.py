import os
import sys
import argparse
import logging
import requests
from tqdm import tqdm
import hashlib
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model URLs - in a production environment, these would point to actual pretrained models
MODEL_URLS = {
    "deepfake_model": {
        "url": "https://github.com/COPixel/models/releases/download/v1.0/deepfake_model.h5",  # Replace with actual URL
        "md5": "0123456789abcdef0123456789abcdef",  # Replace with actual MD5
        "size": 120 * 1024 * 1024,  # Approximate size in bytes
        "description": "EfficientNet + LSTM model for detecting video deepfakes"
    },
    "document_forgery_model": {
        "url": "https://github.com/COPixel/models/releases/download/v1.0/document_forgery_model.h5",  # Replace with actual URL
        "md5": "fedcba9876543210fedcba9876543210",  # Replace with actual MD5
        "size": 90 * 1024 * 1024,  # Approximate size in bytes
        "description": "ResNet-based model for detecting document forgery with multi-task output"
    },
    "signature_forgery_model": {
        "url": "https://github.com/COPixel/models/releases/download/v1.0/signature_forgery_model.h5",  # Replace with actual URL
        "md5": "abcdef0123456789abcdef0123456789",  # Replace with actual MD5
        "size": 70 * 1024 * 1024,  # Approximate size in bytes
        "description": "Siamese network model for comparing and authenticating signatures"
    }
}

# Additional optional models for specialized detection
ADDITIONAL_MODELS = {
    "face_landmark_model": {
        "url": "https://github.com/COPixel/models/releases/download/v1.0/face_landmark_model.h5",
        "md5": "1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p",
        "size": 50 * 1024 * 1024,
        "description": "Model for detecting facial landmarks in deepfake analysis"
    },
    "text_analysis_model": {
        "url": "https://github.com/COPixel/models/releases/download/v1.0/text_analysis_model.h5",
        "md5": "p6o5n4m3l2k1j0i9h8g7f6e5d4c3b2a1",
        "size": 40 * 1024 * 1024,
        "description": "NLP model for analyzing text content in documents"
    }
}

def calculate_md5(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, destination, expected_md5=None, file_size=None):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0)) or file_size or 0
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download with progress bar
        with open(destination, 'wb') as f, tqdm(
                desc=os.path.basename(destination),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        
        # Verify MD5 if provided
        if expected_md5:
            actual_md5 = calculate_md5(destination)
            if actual_md5 != expected_md5:
                logger.warning(f"MD5 verification failed for {destination}")
                logger.warning(f"Expected: {expected_md5}")
                logger.warning(f"Actual: {actual_md5}")
                return False
                
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        # If URL fetch failed and we're in development mode, create placeholder file
        if "--dev" in sys.argv:
            logger.info(f"Creating placeholder model file for development: {destination}")
            create_placeholder_model(destination)
            return True
        return False

def create_placeholder_model(filepath):
    """Create a placeholder model file for development"""
    try:
        # Create a minimal h5 file that can be loaded but not used for predictions
        with h5py.File(filepath, 'w') as f:
            f.attrs['placeholder'] = True
            f.create_dataset('placeholder', data=np.zeros((1, 1)))
        logger.info(f"Created placeholder model at {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error creating placeholder model: {str(e)}")
        # Fallback to empty file
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(b'PLACEHOLDER_MODEL')
        return False

def download_models(models_dir, model_names=None, force=False, include_additional=False):
    """Download the specified models"""
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Combine model dictionaries if needed
    all_models = MODEL_URLS.copy()
    if include_additional:
        all_models.update(ADDITIONAL_MODELS)
    
    # Determine which models to download
    if not model_names:
        model_names = list(MODEL_URLS.keys())  # Only download core models by default
        if include_additional:
            model_names.extend(ADDITIONAL_MODELS.keys())
    else:
        # Validate model names
        for name in model_names:
            if name not in all_models:
                logger.error(f"Unknown model: {name}")
                return False
    
    # Download models
    success = True
    for name in model_names:
        if name in all_models:
            model_info = all_models[name]
            model_path = os.path.join(models_dir, f"{name}.h5")
            
            # Check if model already exists
            if os.path.exists(model_path) and not force:
                logger.info(f"Model {name} already exists at {model_path}")
                if model_info.get("md5"):
                    actual_md5 = calculate_md5(model_path)
                    if actual_md5 == model_info["md5"]:
                        logger.info(f"MD5 verification passed for existing model: {name}")
                        continue
                    else:
                        logger.warning(f"MD5 verification failed for existing model: {name}")
                        logger.warning(f"Expected: {model_info['md5']}")
                        logger.warning(f"Actual: {actual_md5}")
                        if not force:
                            logger.info(f"Redownloading model: {name}")
                else:
                    continue
                    
            # Download model
            logger.info(f"Downloading model: {name} - {model_info.get('description', '')}")
            result = download_file(
                model_info["url"],
                model_path,
                expected_md5=model_info.get("md5"),
                file_size=model_info.get("size")
            )
            
            if not result:
                logger.error(f"Failed to download model: {name}")
                success = False
                
    return success

def list_available_models():
    """List all available models with descriptions"""
    logger.info("Available core models:")
    for name, info in MODEL_URLS.items():
        logger.info(f"  - {name}: {info.get('description', 'No description')}")
    
    logger.info("\nAvailable additional models:")
    for name, info in ADDITIONAL_MODELS.items():
        logger.info(f"  - {name}: {info.get('description', 'No description')}")

def main():
    parser = argparse.ArgumentParser(description="Download pre-trained models for AI detection system")
    parser.add_argument("--models", nargs="+", help="Specific models to download")
    parser.add_argument("--force", action="store_true", help="Force download even if models exist")
    parser.add_argument("--dir", help="Directory to store models", default=None)
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--all", action="store_true", help="Download all models including additional ones")
    parser.add_argument("--dev", action="store_true", help="Development mode - create placeholders if download fails")
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    # Determine models directory
    if args.dir:
        models_dir = args.dir
    else:
        # Default to the directory where this script is located
        models_dir = os.path.dirname(os.path.abspath(__file__))
    
    logger.info(f"Downloading models to: {models_dir}")
    
    # Download models
    success = download_models(models_dir, args.models, args.force, args.all)
    
    if success:
        logger.info("All models downloaded successfully")
    else:
        logger.error("Some models failed to download")
        sys.exit(1)

if __name__ == "__main__":
    main() 