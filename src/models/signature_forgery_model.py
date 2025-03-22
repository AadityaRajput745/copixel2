import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import GaussianNoise, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import logging

logger = logging.getLogger(__name__)

class SignatureForgeryModel:
    """
    Model for detecting forged signatures by analyzing stroke patterns,
    pressure variations, and other signature-specific characteristics.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the signature forgery detection model.
        
        Args:
            model_path (str, optional): Path to a saved model. If None, a new model will be created.
        """
        self.input_shape = (256, 256, 1)  # Grayscale images are sufficient for signatures
        
        # Default model path if none provided
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'signature_forgery_model.h5'
        )
        
        # Try to load existing model, or create a new one
        if model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                logger.info(f"Loaded signature forgery detection model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {str(e)}")
                self.model = self._build_model()
        else:
            self.model = self._build_model()
    
    def _build_model(self):
        """
        Build the signature forgery detection model.
        
        Returns:
            tf.keras.Model: Signature forgery detection model
        """
        # Siamese network approach - for comparing signatures
        input_shape = self.input_shape
        
        # Base CNN for feature extraction
        def build_base_network():
            model = Sequential([
                # Initial noise layer to improve robustness
                GaussianNoise(0.1),
                
                # First convolutional block
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.2),
                
                # Second convolutional block
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.3),
                
                # Third convolutional block - specialized for signature patterns
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.4),
                
                # Features extraction
                Conv2D(256, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                GlobalAveragePooling2D(),
                Dense(256, activation='relu'),
                Dropout(0.5),
            ])
            return model
        
        # Build the siamese network components
        base_network = build_base_network()
        
        # Input for reference signature (known authentic)
        reference_input = Input(shape=input_shape)
        
        # Input for query signature (to be verified)
        query_input = Input(shape=input_shape)
        
        # Process both inputs through the same network
        reference_features = base_network(reference_input)
        query_features = base_network(query_input)
        
        # Concatenate features - could also use subtraction or other comparison
        x = tf.keras.layers.concatenate([reference_features, query_features])
        
        # Decision layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        
        # Output: 0 = genuine, 1 = forged
        output = Dense(1, activation='sigmoid')(x)
        
        # Create and compile the model
        model = Model(inputs=[reference_input, query_input], outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Built new signature forgery detection model")
        return model
    
    def train(self, train_data, validation_data=None, epochs=50, batch_size=16):
        """
        Train the signature forgery detection model.
        
        Args:
            train_data: Training data generator or tuple ([x_ref, x_query], y)
            validation_data: Validation data generator or tuple ([x_ref_val, x_query_val], y_val)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            History object with training metrics
        """
        # Set up callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                self.model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        logger.info(f"Starting signature forgery model training for {epochs} epochs")
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        logger.info("Signature forgery model training completed")
        return history
    
    def predict(self, reference_signature, query_signature):
        """
        Predict whether a query signature is a forgery compared to a reference signature.
        
        Args:
            reference_signature (np.ndarray): Known authentic signature
            query_signature (np.ndarray): Signature to verify
            
        Returns:
            dict: Detection results with forgery confidence
        """
        # Preprocess signatures
        reference_processed = self._preprocess_signature(reference_signature)
        query_processed = self._preprocess_signature(query_signature)
        
        # Add batch dimension if needed
        if len(reference_processed.shape) == 3:
            reference_processed = np.expand_dims(reference_processed, axis=0)
        if len(query_processed.shape) == 3:
            query_processed = np.expand_dims(query_processed, axis=0)
        
        # Get prediction
        prediction = self.model.predict([reference_processed, query_processed])
        confidence = float(prediction[0][0])
        
        # Return results
        return {
            "is_forged": bool(confidence > 0.5),
            "confidence": confidence if confidence > 0.5 else 1.0 - confidence,
            "forgery_score": confidence  # Raw score from 0-1
        }
    
    def _preprocess_signature(self, signature_image):
        """
        Preprocess a signature image for model input.
        
        Args:
            signature_image (np.ndarray): Raw signature image
            
        Returns:
            np.ndarray: Preprocessed signature image
        """
        # Convert to grayscale if color
        if len(signature_image.shape) == 3 and signature_image.shape[2] == 3:
            grayscale = np.mean(signature_image, axis=2)
        else:
            grayscale = signature_image
            
        # Resize to expected input shape
        resized = tf.image.resize(
            np.expand_dims(grayscale, axis=-1),
            self.input_shape[:2],
            method=tf.image.ResizeMethod.BILINEAR
        ).numpy()
        
        # Normalize pixel values to [0, 1]
        normalized = resized / 255.0
        
        # Invert if necessary (make signature white on black background)
        if np.mean(normalized) > 0.5:
            normalized = 1.0 - normalized
            
        return normalized
    
    def save(self, filepath=None):
        """
        Save the model to disk.
        
        Args:
            filepath (str, optional): Path to save the model. If None, uses the default path.
        """
        save_path = filepath or self.model_path
        try:
            self.model.save(save_path)
            logger.info(f"Saved signature forgery detection model to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model to {save_path}: {str(e)}")
            return False
    
    def evaluate(self, test_data):
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data generator or tuple ([x_ref_test, x_query_test], y_test)
            
        Returns:
            tuple: (loss, accuracy) on test data
        """
        logger.info("Evaluating signature forgery detection model")
        return self.model.evaluate(test_data)
    
    def extract_signature_features(self, signature_image):
        """
        Extract features from a signature for analysis.
        
        Args:
            signature_image (np.ndarray): Signature image
            
        Returns:
            np.ndarray: Extracted feature vector
        """
        # Preprocess signature
        processed = self._preprocess_signature(signature_image)
        if len(processed.shape) == 3:
            processed = np.expand_dims(processed, axis=0)
            
        # Create a feature extraction model from base network
        # Get the base network from the Siamese model
        base_network = self.model.layers[2]  # This assumes the base network is the 3rd layer
        
        # Use base network to extract features
        features = base_network.predict(processed)
        return features[0]
    
    def compare_signatures(self, signature1, signature2):
        """
        Compare two signatures and calculate similarity metrics.
        
        Args:
            signature1 (np.ndarray): First signature
            signature2 (np.ndarray): Second signature
            
        Returns:
            dict: Comparison metrics
        """
        # Extract features from both signatures
        features1 = self.extract_signature_features(signature1)
        features2 = self.extract_signature_features(signature2)
        
        # Calculate Euclidean distance
        euclidean_distance = np.linalg.norm(features1 - features2)
        
        # Calculate cosine similarity
        cosine_similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        
        # Make a prediction
        prediction_result = self.predict(signature1, signature2)
        
        # Return comparison metrics
        return {
            "euclidean_distance": float(euclidean_distance),
            "cosine_similarity": float(cosine_similarity),
            "is_forged": prediction_result["is_forged"],
            "confidence": prediction_result["confidence"]
        } 