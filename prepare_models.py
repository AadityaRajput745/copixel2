"""
A utility script to create trained model files without actually training.
This script creates model files with random weights for testing purposes.
"""

import os
import sys
import tensorflow as tf
import logging
from deepfake_model import DeepfakeModel
from document_forgery_model import DocumentForgeryModel
from signature_forgery_model import SignatureForgeryModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_models(output_dir):
    """Create model files with random weights for testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create deepfake detection model
    logger.info("Creating deepfake model...")
    deepfake_path = os.path.join(output_dir, "deepfake_model.h5")
    deepfake_model = DeepfakeModel()
    logger.info(f"Saving deepfake model to {deepfake_path}")
    deepfake_model.save(deepfake_path)
    
    # Create document forgery detection model
    logger.info("Creating document forgery model...")
    document_path = os.path.join(output_dir, "document_forgery_model.h5")
    document_model = DocumentForgeryModel()
    logger.info(f"Saving document forgery model to {document_path}")
    document_model.save(document_path)
    
    # Create signature forgery detection model
    logger.info("Creating signature forgery model...")
    signature_path = os.path.join(output_dir, "signature_forgery_model.h5")
    signature_model = SignatureForgeryModel()
    logger.info(f"Saving signature forgery model to {signature_path}")
    signature_model.save(signature_path)
    
    logger.info("All models created successfully!")
    return True

if __name__ == "__main__":
    # Default output directory
    output_dir = "../../models"
    
    # Check if another output directory is specified
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    
    # Create the models
    create_models(output_dir) 