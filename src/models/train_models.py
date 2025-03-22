import os
import sys
import argparse
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import cv2
import random
from tqdm import tqdm
import json

from deepfake_model import DeepfakeModel
from document_forgery_model import DocumentForgeryModel
from signature_forgery_model import SignatureForgeryModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoFrameSequence(Sequence):
    """Sequence generator for video frames to train deepfake model"""
    
    def __init__(self, video_paths, labels, batch_size=8, sequence_length=16, input_shape=(224, 224, 3)):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.input_shape = input_shape
        self.indices = np.arange(len(video_paths))
        
    def __len__(self):
        return len(self.video_paths) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_videos = [self.video_paths[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]
        
        # Initialize batch arrays
        X = np.zeros((self.batch_size, self.sequence_length, *self.input_shape))
        y = np.array(batch_labels)
        
        # Load video frames for each video in batch
        for i, video_path in enumerate(batch_videos):
            frames = self._extract_frames(video_path)
            X[i] = frames
            
        return X, y
    
    def on_epoch_end(self):
        # Shuffle indices at the end of each epoch
        np.random.shuffle(self.indices)
    
    def _extract_frames(self, video_path):
        """Extract frames from a video file"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Count frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            # Fallback for when frame count is not available
            logger.warning(f"Could not get frame count for {video_path}")
            return np.zeros((self.sequence_length, *self.input_shape))
        
        # Choose frames to extract
        if total_frames <= self.sequence_length:
            # If not enough frames, duplicate frames
            frame_indices = list(range(total_frames)) * (self.sequence_length // total_frames + 1)
            frame_indices = frame_indices[:self.sequence_length]
        else:
            # Sample frames evenly
            frame_indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
        
        # Read selected frames
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize and normalize
                frame = cv2.resize(frame, self.input_shape[:2])
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            else:
                # If frame reading fails, add a blank frame
                frames.append(np.zeros(self.input_shape))
        
        cap.release()
        return np.array(frames)

class DocumentDataGenerator(Sequence):
    """Data generator for document forgery detection"""
    
    def __init__(self, image_paths, labels, forgery_types, batch_size=16, input_shape=(512, 512, 3)):
        self.image_paths = image_paths
        self.labels = labels  # Binary labels (real/forged)
        self.forgery_types = forgery_types  # Multi-class labels for forgery type
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.indices = np.arange(len(image_paths))
        
    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_images = [self.image_paths[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]
        batch_types = [self.forgery_types[i] for i in batch_indices]
        
        # Initialize batch arrays
        X = np.zeros((self.batch_size, *self.input_shape))
        y_label = np.array(batch_labels)
        y_type = np.array(batch_types)
        
        # Load and preprocess images
        for i, img_path in enumerate(batch_images):
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Could not read image {img_path}")
                continue
                
            # Resize and normalize
            img = cv2.resize(img, self.input_shape[:2])
            img = img.astype(np.float32) / 255.0
            X[i] = img
            
        return X, [y_label, y_type]
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

class SignatureDataGenerator(Sequence):
    """Data generator for signature comparison and forgery detection"""
    
    def __init__(self, genuine_paths, forged_paths, batch_size=16, input_shape=(256, 256, 1)):
        self.genuine_paths = genuine_paths
        self.forged_paths = forged_paths
        self.batch_size = batch_size
        self.input_shape = input_shape
        
        # Generate pairs: (reference_signature, query_signature, label)
        # label: 0 = genuine, 1 = forged
        self.pairs = []
        
        # Genuine pairs (same signature)
        for path in genuine_paths:
            self.pairs.append((path, path, 0))  # Same signature is genuine
        
        # Forged pairs
        for ref_path in genuine_paths:
            for forged_path in forged_paths:
                self.pairs.append((ref_path, forged_path, 1))  # Forged signature
        
        # Shuffle pairs
        random.shuffle(self.pairs)
        
    def __len__(self):
        return len(self.pairs) // self.batch_size
    
    def __getitem__(self, idx):
        batch_pairs = self.pairs[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        # Initialize batch arrays
        X1 = np.zeros((len(batch_pairs), *self.input_shape))  # Reference signatures
        X2 = np.zeros((len(batch_pairs), *self.input_shape))  # Query signatures
        y = np.zeros((len(batch_pairs), 1))  # Labels
        
        # Load and preprocess signature pairs
        for i, (ref_path, query_path, label) in enumerate(batch_pairs):
            # Load reference signature
            ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            if ref_img is None:
                logger.warning(f"Could not read image {ref_path}")
                continue
                
            # Load query signature
            query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
            if query_img is None:
                logger.warning(f"Could not read image {query_path}")
                continue
                
            # Resize and normalize
            ref_img = cv2.resize(ref_img, self.input_shape[:2])
            query_img = cv2.resize(query_img, self.input_shape[:2])
            
            # Add channel dimension
            ref_img = np.expand_dims(ref_img, axis=-1)
            query_img = np.expand_dims(query_img, axis=-1)
            
            # Normalize pixel values
            ref_img = ref_img.astype(np.float32) / 255.0
            query_img = query_img.astype(np.float32) / 255.0
            
            # Store in batch arrays
            X1[i] = ref_img
            X2[i] = query_img
            y[i] = label
            
        return [X1, X2], y
    
    def on_epoch_end(self):
        # Shuffle pairs at the end of each epoch
        random.shuffle(self.pairs)

def train_deepfake_model(data_dir, output_dir, epochs=20, batch_size=8):
    """Train the deepfake detection model"""
    logger.info("Starting deepfake model training")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "deepfake_model.h5")
    
    # Load and prepare data
    # In a real implementation, you would load the actual video data
    train_dir = os.path.join(data_dir, "deepfake", "train")
    val_dir = os.path.join(data_dir, "deepfake", "val")
    
    # Check if data exists
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        logger.error(f"Training data not found at {train_dir} or {val_dir}")
        return False
    
    # Load video paths and labels from directories or metadata file
    # This is a placeholder - in a real implementation, load actual data
    # For demo purposes, we're creating dummy data
    try:
        # Load or create training data
        train_videos = []
        train_labels = []
        val_videos = []
        val_labels = []
        
        # Try to load metadata if available
        train_meta_file = os.path.join(train_dir, "metadata.json")
        if os.path.exists(train_meta_file):
            with open(train_meta_file, 'r') as f:
                train_meta = json.load(f)
                train_videos = [os.path.join(train_dir, item["path"]) for item in train_meta]
                train_labels = [int(item["label"]) for item in train_meta]
        
        val_meta_file = os.path.join(val_dir, "metadata.json")
        if os.path.exists(val_meta_file):
            with open(val_meta_file, 'r') as f:
                val_meta = json.load(f)
                val_videos = [os.path.join(val_dir, item["path"]) for item in val_meta]
                val_labels = [int(item["label"]) for item in val_meta]
        
        # If no real data found, create synthetic data for demonstration
        if not train_videos:
            logger.warning("No real training data found, using synthetic data for demonstration")
            
            # For demonstration purposes, create synthetic data
            # Create a model instance
            model = DeepfakeModel(model_path=model_path)
            
            # Create synthetic data
            sequence_length = 16
            input_shape = (224, 224, 3)
            num_samples = 100
            
            # Synthetic training data
            X_train = np.random.rand(num_samples, sequence_length, *input_shape).astype(np.float32)
            y_train = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)
            
            # Synthetic validation data
            X_val = np.random.rand(num_samples//5, sequence_length, *input_shape).astype(np.float32)
            y_val = np.random.randint(0, 2, size=(num_samples//5, 1)).astype(np.float32)
            
            # Train the model with synthetic data
            try:
                logger.info(f"Starting deepfake model training for {epochs} epochs")
                history = model.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size
                )
                model.save(model_path)
                return True
            except Exception as e:
                logger.error(f"Error training deepfake model: {str(e)}")
                return False
        else:
            # Create data generators
            train_seq = VideoFrameSequence(train_videos, train_labels, batch_size=batch_size)
            val_seq = VideoFrameSequence(val_videos, val_labels, batch_size=batch_size)
            
            # Create and train the model
            model = DeepfakeModel(model_path=model_path)
            history = model.train(train_seq, val_seq, epochs=epochs, batch_size=batch_size)
            return True
    except Exception as e:
        logger.error(f"Error in deepfake model training: {str(e)}")
        return False

def train_document_model(data_dir, output_dir, epochs=30, batch_size=16):
    """Train the document forgery detection model"""
    logger.info("Starting document forgery model training")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "document_forgery_model.h5")
    
    # Load and prepare data
    train_dir = os.path.join(data_dir, "document", "train")
    val_dir = os.path.join(data_dir, "document", "val")
    
    # Check if data exists
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        logger.error(f"Training data not found at {train_dir} or {val_dir}")
        return False
    
    try:
        # Load training data from directories or metadata
        train_images = []
        train_labels = []
        train_types = []
        val_images = []
        val_labels = []
        val_types = []
        
        # Try to load metadata if available
        train_meta_file = os.path.join(train_dir, "metadata.json")
        if os.path.exists(train_meta_file):
            with open(train_meta_file, 'r') as f:
                train_meta = json.load(f)
                train_images = [os.path.join(train_dir, item["path"]) for item in train_meta]
                train_labels = [int(item["label"]) for item in train_meta]
                train_types = [[int(t) for t in item["type"]] for item in train_meta]
        
        val_meta_file = os.path.join(val_dir, "metadata.json")
        if os.path.exists(val_meta_file):
            with open(val_meta_file, 'r') as f:
                val_meta = json.load(f)
                val_images = [os.path.join(val_dir, item["path"]) for item in val_meta]
                val_labels = [int(item["label"]) for item in val_meta]
                val_types = [[int(t) for t in item["type"]] for item in val_meta]
        
        # If no real data found, create synthetic data for demonstration
        if not train_images:
            logger.warning("No real training data found, using synthetic data for demonstration")
            
            # Create a model instance
            model = DocumentForgeryModel(model_path=model_path)
            
            # Create synthetic data
            input_shape = (512, 512, 3)
            num_samples = 100
            
            # Synthetic training data
            X_train = np.random.rand(num_samples, *input_shape).astype(np.float32)
            y_train_binary = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)
            y_train_type = np.random.randint(0, 4, size=(num_samples, 4)).astype(np.float32)
            
            # Synthetic validation data
            X_val = np.random.rand(num_samples//5, *input_shape).astype(np.float32)
            y_val_binary = np.random.randint(0, 2, size=(num_samples//5, 1)).astype(np.float32)
            y_val_type = np.random.randint(0, 4, size=(num_samples//5, 4)).astype(np.float32)
            
            # Train the model with synthetic data
            try:
                logger.info(f"Starting document forgery model training for {epochs} epochs")
                history = model.model.fit(
                    X_train, 
                    {"binary_output": y_train_binary, "type_output": y_train_type},
                    validation_data=(X_val, {"binary_output": y_val_binary, "type_output": y_val_type}),
                    epochs=epochs,
                    batch_size=batch_size
                )
                model.save(model_path)
                return True
            except Exception as e:
                logger.error(f"Error training document forgery model: {str(e)}")
                return False
        else:
            # Create data generators
            train_gen = DocumentDataGenerator(train_images, train_labels, train_types, batch_size=batch_size)
            val_gen = DocumentDataGenerator(val_images, val_labels, val_types, batch_size=batch_size)
            
            # Create and train the model
            model = DocumentForgeryModel(model_path=model_path)
            history = model.train(train_gen, val_gen, epochs=epochs, batch_size=batch_size)
            return True
    except Exception as e:
        logger.error(f"Error in document forgery model training: {str(e)}")
        return False

def train_signature_model(data_dir, output_dir, epochs=50, batch_size=16):
    """Train the signature forgery detection model"""
    logger.info("Starting signature forgery model training")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "signature_forgery_model.h5")
    
    # Load and prepare data
    genuine_dir = os.path.join(data_dir, "signature", "genuine")
    forged_dir = os.path.join(data_dir, "signature", "forged")
    
    # Check if data exists
    if not os.path.exists(genuine_dir) or not os.path.exists(forged_dir):
        logger.error(f"Training data not found at {genuine_dir} or {forged_dir}")
        return False
    
    try:
        # Load signature images from directories
        genuine_paths = []
        forged_paths = []
        
        # Try to load from directories
        if os.path.exists(genuine_dir):
            for filename in os.listdir(genuine_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    genuine_paths.append(os.path.join(genuine_dir, filename))
        
        if os.path.exists(forged_dir):
            for filename in os.listdir(forged_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    forged_paths.append(os.path.join(forged_dir, filename))
        
        # If no real data found, create synthetic data for demonstration
        if not genuine_paths and not forged_paths:
            logger.warning("No real training data found, using synthetic data for demonstration")
            
            # Create a model instance
            model = SignatureForgeryModel(model_path=model_path)
            
            # Create synthetic data
            input_shape = (256, 256, 1)
            num_samples = 100
            
            # Synthetic training data for signature pairs
            X1_train = np.random.rand(num_samples, *input_shape).astype(np.float32)
            X2_train = np.random.rand(num_samples, *input_shape).astype(np.float32)
            y_train = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)
            
            # Synthetic validation data
            X1_val = np.random.rand(num_samples//5, *input_shape).astype(np.float32)
            X2_val = np.random.rand(num_samples//5, *input_shape).astype(np.float32)
            y_val = np.random.randint(0, 2, size=(num_samples//5, 1)).astype(np.float32)
            
            # Train the model with synthetic data
            try:
                logger.info(f"Starting signature forgery model training for {epochs} epochs")
                history = model.model.fit(
                    [X1_train, X2_train], 
                    y_train,
                    validation_data=([X1_val, X2_val], y_val),
                    epochs=epochs,
                    batch_size=batch_size
                )
                model.save(model_path)
                return True
            except Exception as e:
                logger.error(f"Error training signature forgery model: {str(e)}")
                return False
        else:
            # Create data generator
            data_gen = SignatureDataGenerator(genuine_paths, forged_paths, batch_size=batch_size)
            
            # Create train/validation split
            total_samples = len(data_gen) * batch_size
            train_size = int(0.8 * total_samples)
            train_steps = train_size // batch_size
            val_steps = (total_samples - train_size) // batch_size
            
            # Create and train the model
            model = SignatureForgeryModel(model_path=model_path)
            history = model.train(data_gen, validation_data=data_gen, 
                                 epochs=epochs, batch_size=batch_size,
                                 train_steps=train_steps, val_steps=val_steps)
            return True
    except Exception as e:
        logger.error(f"Error in signature forgery model training: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train AI detection models")
    parser.add_argument("--data-dir", required=True, help="Path to training data directory")
    parser.add_argument("--output-dir", required=True, help="Path to save trained models")
    parser.add_argument("--models", nargs="+", choices=["deepfake", "document", "signature", "all"], 
                        default=["all"], help="Models to train")
    parser.add_argument("--epochs", type=int, default=0, help="Number of training epochs (0 for default per model)")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size (0 for default per model)")
    args = parser.parse_args()
    
    # Ensure directories exist
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        return 1
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which models to train
    models_to_train = args.models
    if "all" in models_to_train:
        models_to_train = ["deepfake", "document", "signature"]
    
    # Train selected models
    success = True
    
    if "deepfake" in models_to_train:
        logger.info("Training deepfake detection model")
        deepfake_epochs = args.epochs if args.epochs > 0 else 20
        deepfake_batch = args.batch_size if args.batch_size > 0 else 8
        if not train_deepfake_model(args.data_dir, args.output_dir, deepfake_epochs, deepfake_batch):
            success = False
    
    if "document" in models_to_train:
        logger.info("Training document forgery detection model")
        document_epochs = args.epochs if args.epochs > 0 else 30
        document_batch = args.batch_size if args.batch_size > 0 else 16
        if not train_document_model(args.data_dir, args.output_dir, document_epochs, document_batch):
            success = False
    
    if "signature" in models_to_train:
        logger.info("Training signature forgery detection model")
        signature_epochs = args.epochs if args.epochs > 0 else 50
        signature_batch = args.batch_size if args.batch_size > 0 else 16
        if not train_signature_model(args.data_dir, args.output_dir, signature_epochs, signature_batch):
            success = False
    
    if success:
        logger.info("All requested models trained successfully")
        return 0
    else:
        logger.error("Some models failed to train")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 