import os
import cv2
import numpy as np
import uuid
import logging
from datetime import datetime
import tensorflow as tf
import torch
from facenet_pytorch import MTCNN
import json
from ..models.deepfake_model import DeepfakeModel

logger = logging.getLogger(__name__)

class VideoDeepfakeDetector:
    def __init__(self, model_path=None):
        self.model_path = model_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                   '..', 'models', 'deepfake_model.h5')
        
        # Initialize face detector
        self.face_detector = MTCNN(
            keep_all=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Load deepfake detection model
        try:
            self.model = DeepfakeModel(self.model_path)
            logger.info(f"Loaded deepfake detection model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
    
    def preprocess_frame(self, frame):
        """Preprocess a video frame for the detection model"""
        # Resize frame to model input size
        processed_frame = cv2.resize(frame, (224, 224))
        # Convert to RGB if needed
        if len(processed_frame.shape) == 2:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
        # Normalize pixel values
        processed_frame = processed_frame.astype(np.float32) / 255.0
        return processed_frame
    
    def extract_faces(self, frame):
        """Extract faces from a frame using MTCNN"""
        try:
            # Convert BGR to RGB (facenet expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detect faces
            boxes, probs = self.face_detector.detect(rgb_frame)
            faces = []
            
            if boxes is not None:
                for box in boxes:
                    box = box.astype(int)
                    x1, y1, x2, y2 = box
                    # Extract face with some margin
                    face = rgb_frame[max(0, y1):min(y2, frame.shape[0]), 
                                   max(0, x1):min(x2, frame.shape[1])]
                    # Only add if face is valid
                    if face.size > 0 and face.shape[0] > 0 and face.shape[1] > 0:
                        faces.append(face)
            
            return faces
        except Exception as e:
            logger.error(f"Error extracting faces: {str(e)}")
            return []
    
    def analyze_facial_inconsistencies(self, faces):
        """Analyze facial inconsistencies across frames"""
        # Basic analysis for demo purposes
        # In a real implementation, this would use more sophisticated techniques
        inconsistencies = {}
        
        if len(faces) == 0:
            return {"no_faces_detected": True}
        
        # Check for unrealistic facial features (placeholder)
        inconsistencies["unrealistic_features"] = False
        
        # Check for temporal inconsistencies (placeholder)
        inconsistencies["temporal_inconsistencies"] = False
        
        return inconsistencies
    
    def detect_with_model(self, faces):
        """Use the loaded deepfake detection model"""
        if self.model is None:
            return {"model_prediction": "unavailable"}
        
        predictions = []
        for face in faces:
            try:
                # Resize and preprocess face for model
                processed_face = cv2.resize(face, (224, 224))
                processed_face = processed_face.astype(np.float32) / 255.0
                
                # Get prediction from our DeepfakeModel
                prediction = self.model.predict(processed_face)
                predictions.append(float(prediction))
            except Exception as e:
                logger.error(f"Error in model prediction: {str(e)}")
        
        # Average predictions across all faces
        if predictions:
            avg_prediction = sum(predictions) / len(predictions)
            return {
                "model_prediction": avg_prediction,
                "is_fake": avg_prediction > 0.5,  # Threshold can be adjusted
                "confidence": avg_prediction if avg_prediction > 0.5 else 1 - avg_prediction
            }
        else:
            return {"model_prediction": "failed", "is_fake": False, "confidence": 0}
    
    def detect(self, video_path):
        """Main detection method for a video file"""
        logger.info(f"Starting deepfake detection on video: {video_path}")
        
        # Generate a unique detection ID
        detection_id = str(uuid.uuid4())
        
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {
                    "success": False,
                    "error": "Failed to open video file",
                    "detection_id": detection_id
                }
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames (process every 10th frame to save time)
            sample_interval = max(1, int(fps / 2))  # 2 frames per second
            
            all_faces = []
            frame_results = []
            frames_processed = 0
            
            # Process video frames
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process every nth frame
                if frames_processed % sample_interval == 0:
                    # Extract faces
                    faces = self.extract_faces(frame)
                    if faces:
                        all_faces.extend(faces)
                    
                    # If using a model, get prediction for this frame
                    if self.model is not None and faces:
                        frame_result = self.detect_with_model(faces)
                        frame_result["frame_number"] = frames_processed
                        frame_results.append(frame_result)
                
                frames_processed += 1
                
                # For very long videos, limit the processing
                if frames_processed >= 300:  # Process max 5 minutes at 1 fps
                    break
            
            # Release the video
            cap.release()
            
            # Analyze facial inconsistencies
            inconsistency_results = self.analyze_facial_inconsistencies(all_faces)
            
            # Get overall result
            if self.model is not None and frame_results:
                # Calculate average prediction across frames
                fake_frames = sum(1 for r in frame_results if r.get("is_fake", False))
                fake_ratio = fake_frames / len(frame_results) if frame_results else 0
                
                overall_fake = fake_ratio > 0.3  # If more than 30% of frames are fake
                confidence = sum(r.get("confidence", 0) for r in frame_results) / len(frame_results)
            else:
                # Fallback method if no model is available
                overall_fake = inconsistency_results.get("unrealistic_features", False) or \
                              inconsistency_results.get("temporal_inconsistencies", False)
                confidence = 0.5
            
            result = {
                "success": True,
                "detection_id": detection_id,
                "timestamp": datetime.now().isoformat(),
                "filename": os.path.basename(video_path),
                "is_fake": overall_fake,
                "confidence": confidence,
                "details": {
                    "frames_processed": frames_processed,
                    "faces_detected": len(all_faces),
                    "inconsistencies": inconsistency_results,
                }
            }
            
            # Save detection result to file
            result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                    '..', '..', 'data', 'results')
            os.makedirs(result_dir, exist_ok=True)
            result_path = os.path.join(result_dir, f"{detection_id}.json")
            
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
                
            logger.info(f"Detection completed. Result saved to {result_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error in video detection: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "detection_id": detection_id
            } 