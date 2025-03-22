import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LSTM, TimeDistributed
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import logging

logger = logging.getLogger(__name__)

class DeepfakeModel:
    """
    Model for detecting deepfake videos by analyzing facial inconsistencies
    and temporal anomalies across video frames.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the deepfake detection model.
        
        Args:
            model_path (str, optional): Path to a saved model. If None, a new model will be created.
        """
        self.input_shape = (224, 224, 3)  # Standard input size for face images
        self.sequence_length = 16  # Number of frames to analyze in sequence
        
        # Default model path if none provided
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'deepfake_model.h5'
        )
        
        # Try to load existing model, or create a new one
        if model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                logger.info(f"Loaded deepfake detection model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {str(e)}")
                self.model = self._build_model()
        else:
            self.model = self._build_model()
    
    def _build_frame_feature_extractor(self):
        """
        Build a CNN model for extracting features from individual frames.
        
        Returns:
            tf.keras.Model: Frame feature extractor model
        """
        # Use EfficientNet as base model (good balance of accuracy and efficiency)
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze initial layers to prevent overfitting
        for layer in base_model.layers[:100]:
            layer.trainable = False
            
        # Add custom layers on top of base model
        x = base_model.output
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        
        # Create model
        frame_model = Model(inputs=base_model.input, outputs=x)
        return frame_model
    
    def _build_model(self):
        """
        Build the complete deepfake detection model with temporal analysis.
        
        Returns:
            tf.keras.Model: Complete deepfake detection model
        """
        # Build frame feature extractor
        frame_extractor = self._build_frame_feature_extractor()
        
        # Create input for sequence of frames
        sequence_input = Input(shape=(self.sequence_length, *self.input_shape))
        
        # Apply the frame extractor to each frame in the sequence
        encoded_frames = TimeDistributed(frame_extractor)(sequence_input)
        
        # Add LSTM for temporal analysis
        x = LSTM(64, return_sequences=True)(encoded_frames)
        x = LSTM(64)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Output layer (0 = real, 1 = fake)
        output = Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = Model(inputs=sequence_input, outputs=output)
        
        # Compile with binary crossentropy (fake/real is binary classification)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Built new deepfake detection model")
        return model
    
    def train(self, train_data, validation_data=None, epochs=20, batch_size=8):
        """
        Train the deepfake detection model.
        
        Args:
            train_data: Training data generator or tuple (x_train, y_train)
            validation_data: Validation data generator or tuple (x_val, y_val)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            History object with training metrics
        """
        # Set up callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(
                self.model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        logger.info(f"Starting deepfake model training for {epochs} epochs")
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        logger.info("Deepfake model training completed")
        return history
    
    def predict(self, frame_sequence):
        """
        Predict whether a sequence of video frames contains deepfake content.
        
        Args:
            frame_sequence: Numpy array of shape (sequence_length, height, width, channels)
            
        Returns:
            float: Confidence score (0-1) where higher values indicate deepfake
        """
        # Ensure frame sequence has correct shape
        if len(frame_sequence.shape) == 4:
            # Add batch dimension if not present
            frame_sequence = np.expand_dims(frame_sequence, axis=0)
        
        # Process normalized frames in batch
        predictions = self.model.predict(frame_sequence)
        
        # Return the prediction confidence
        return float(predictions[0][0])
    
    def save(self, filepath=None):
        """
        Save the model to disk.
        
        Args:
            filepath (str, optional): Path to save the model. If None, uses the default path.
        """
        save_path = filepath or self.model_path
        try:
            self.model.save(save_path)
            logger.info(f"Saved deepfake detection model to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model to {save_path}: {str(e)}")
            return False
    
    def evaluate(self, test_data):
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data generator or tuple (x_test, y_test)
            
        Returns:
            tuple: (loss, accuracy) on test data
        """
        logger.info("Evaluating deepfake detection model")
        return self.model.evaluate(test_data) 