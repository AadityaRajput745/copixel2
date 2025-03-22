import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D, concatenate
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import logging

logger = logging.getLogger(__name__)

class DocumentForgeryModel:
    """
    Model for detecting forged documents by analyzing visual inconsistencies,
    artifacts, and unusual patterns in document images.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the document forgery detection model.
        
        Args:
            model_path (str, optional): Path to a saved model. If None, a new model will be created.
        """
        self.input_shape = (512, 512, 3)  # Higher resolution for document details
        
        # Default model path if none provided
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'document_forgery_model.h5'
        )
        
        # Try to load existing model, or create a new one
        if model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                logger.info(f"Loaded document forgery detection model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {str(e)}")
                self.model = self._build_model()
        else:
            self.model = self._build_model()
    
    def _build_edge_detection_branch(self, input_tensor):
        """
        Build a branch for edge detection and analysis.
        
        Args:
            input_tensor: Input tensor for this branch
            
        Returns:
            Tensor: Output of the edge detection branch
        """
        # Edge detection branch (focus on document boundaries and artifacts)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        return x
    
    def _build_texture_analysis_branch(self, input_tensor):
        """
        Build a branch for texture analysis to detect inconsistencies.
        
        Args:
            input_tensor: Input tensor for this branch
            
        Returns:
            Tensor: Output of the texture analysis branch
        """
        # Texture analysis branch (focus on paper texture, ink distribution)
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor
        )
        
        # Freeze early layers
        for layer in base_model.layers[:100]:
            layer.trainable = False
            
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        return x
    
    def _build_model(self):
        """
        Build the complete document forgery detection model with multiple analysis branches.
        
        Returns:
            tf.keras.Model: Complete document forgery detection model
        """
        # Input layer
        input_img = Input(shape=self.input_shape)
        
        # Edge detection branch
        edge_branch = self._build_edge_detection_branch(input_img)
        
        # Texture analysis branch
        texture_branch = self._build_texture_analysis_branch(input_img)
        
        # Merge branches
        merged = concatenate([edge_branch, texture_branch])
        
        # Final classification layers
        x = Dense(128, activation='relu')(merged)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        
        # Output for multi-task learning: 
        # - Document forgery (binary)
        # - Type of forgery (multiclass)
        forgery_output = Dense(1, activation='sigmoid', name='forgery_detection')(x)
        forgery_type_output = Dense(4, activation='softmax', name='forgery_type')(x)
        
        # Create model with multiple outputs
        model = Model(
            inputs=input_img, 
            outputs=[forgery_output, forgery_type_output]
        )
        
        # Compile with appropriate losses for each output
        model.compile(
            optimizer='adam',
            loss={
                'forgery_detection': 'binary_crossentropy',
                'forgery_type': 'categorical_crossentropy'
            },
            metrics={
                'forgery_detection': ['accuracy'],
                'forgery_type': ['accuracy']
            },
            loss_weights={
                'forgery_detection': 1.0,
                'forgery_type': 0.5
            }
        )
        
        logger.info("Built new document forgery detection model")
        return model
    
    def train(self, train_data, validation_data=None, epochs=30, batch_size=16):
        """
        Train the document forgery detection model.
        
        Args:
            train_data: Training data generator or tuple (x_train, [y_train_forgery, y_train_type])
            validation_data: Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            History object with training metrics
        """
        # Set up callbacks
        callbacks = [
            EarlyStopping(monitor='val_forgery_detection_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                self.model_path,
                monitor='val_forgery_detection_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        logger.info(f"Starting document forgery model training for {epochs} epochs")
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        logger.info("Document forgery model training completed")
        return history
    
    def predict(self, document_image):
        """
        Predict whether a document image contains forgery.
        
        Args:
            document_image: Numpy array of shape (height, width, channels)
            
        Returns:
            dict: Detection results including forgery confidence and type
        """
        # Ensure image has correct shape
        if len(document_image.shape) == 3:
            # Add batch dimension if not present
            document_image = np.expand_dims(document_image, axis=0)
        
        # Resize image if needed
        if document_image.shape[1:3] != self.input_shape[0:2]:
            resized_image = tf.image.resize(
                document_image, 
                self.input_shape[0:2], 
                method=tf.image.ResizeMethod.BILINEAR
            )
            document_image = resized_image.numpy()
        
        # Normalize pixel values to [0, 1]
        document_image = document_image.astype(np.float32) / 255.0
        
        # Get predictions
        forgery_pred, forgery_type_pred = self.model.predict(document_image)
        
        # Map forgery types to human-readable labels
        forgery_types = ["content_manipulation", "signature_forgery", "digital_splicing", "ai_generated"]
        predicted_type_idx = np.argmax(forgery_type_pred[0])
        
        # Return prediction results
        return {
            "is_forged": bool(forgery_pred[0][0] > 0.5),
            "forgery_confidence": float(forgery_pred[0][0]),
            "forgery_type": forgery_types[predicted_type_idx],
            "forgery_type_confidence": float(forgery_type_pred[0][predicted_type_idx])
        }
    
    def save(self, filepath=None):
        """
        Save the model to disk.
        
        Args:
            filepath (str, optional): Path to save the model. If None, uses the default path.
        """
        save_path = filepath or self.model_path
        try:
            self.model.save(save_path)
            logger.info(f"Saved document forgery detection model to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model to {save_path}: {str(e)}")
            return False
    
    def evaluate(self, test_data):
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data generator or tuple (x_test, [y_test_forgery, y_test_type])
            
        Returns:
            dict: Evaluation metrics for each output
        """
        logger.info("Evaluating document forgery detection model")
        results = self.model.evaluate(test_data)
        
        # Format results as dictionary
        metrics = {
            'loss': results[0],
            'forgery_detection_loss': results[1],
            'forgery_type_loss': results[2],
            'forgery_detection_accuracy': results[3],
            'forgery_type_accuracy': results[4]
        }
        
        return metrics
    
    def extract_document_features(self, document_image):
        """
        Extract features from a document image for further analysis.
        
        Args:
            document_image: Document image as numpy array
            
        Returns:
            np.ndarray: Extracted features
        """
        # Create a feature extraction model from the main model
        # by taking outputs from an intermediate layer
        feature_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(index=-3).output  # The layer before final classification
        )
        
        # Preprocess image
        if len(document_image.shape) == 3:
            document_image = np.expand_dims(document_image, axis=0)
            
        # Resize if needed
        if document_image.shape[1:3] != self.input_shape[0:2]:
            resized_image = tf.image.resize(
                document_image, 
                self.input_shape[0:2], 
                method=tf.image.ResizeMethod.BILINEAR
            )
            document_image = resized_image.numpy()
            
        # Normalize
        document_image = document_image.astype(np.float32) / 255.0
        
        # Extract features
        features = feature_model.predict(document_image)
        return features[0]  # Remove batch dimension 