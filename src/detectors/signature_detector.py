import os
import numpy as np
import uuid
import logging
from datetime import datetime
import cv2
import json
from PIL import Image
from ..models.signature_forgery_model import SignatureForgeryModel

logger = logging.getLogger(__name__)

class SignatureDetector:
    def __init__(self, model_path=None):
        self.model_path = model_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                    '..', 'models', 'signature_forgery_model.h5')
        
        # Load signature forgery model
        try:
            self.model = SignatureForgeryModel(self.model_path)
            logger.info(f"Loaded signature forgery model from {self.model_path}")
            self.model_available = True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model_available = False
            
        # Reference signature path (in a real app, this would come from a database of verified signatures)
        self.reference_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                          '..', '..', 'data', 'reference_signatures')
        os.makedirs(self.reference_path, exist_ok=True)
    
    def preprocess_signature(self, signature_path):
        """Preprocess the signature image for analysis"""
        try:
            # Load image
            img = cv2.imread(signature_path)
            if img is None:
                raise ValueError(f"Could not read image file: {signature_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to segment signature
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours to identify signature region
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If no contours found, return the original image
            if not contours:
                return gray
            
            # Find the largest contour (assumed to be the signature)
            max_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box of the signature
            x, y, w, h = cv2.boundingRect(max_contour)
            
            # Extract signature region with some margin
            margin = 10
            x_min = max(0, x - margin)
            y_min = max(0, y - margin)
            x_max = min(gray.shape[1], x + w + margin)
            y_max = min(gray.shape[0], y + h + margin)
            
            # Crop the signature
            signature = gray[y_min:y_max, x_min:x_max]
            
            # Resize to a standard size for model (224x224 is common)
            signature_resized = cv2.resize(signature, (224, 224))
            
            return signature_resized
        except Exception as e:
            logger.error(f"Error preprocessing signature: {str(e)}")
            raise
    
    def extract_features(self, signature_img):
        """Extract features from signature for traditional analysis"""
        features = {}
        
        try:
            # Compute histogram of oriented gradients (HOG)
            # This is a simplified version - real implementation would use proper HOG
            edges = cv2.Canny(signature_img, 50, 150)
            gradient_x = cv2.Sobel(signature_img, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(signature_img, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude and orientation
            magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            angle = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
            
            # Basic statistics as features
            features["mean_gradient_magnitude"] = np.mean(magnitude)
            features["std_gradient_magnitude"] = np.std(magnitude)
            features["edge_density"] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Calculate pixel density (ratio of signature pixels to background)
            _, binary = cv2.threshold(signature_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            features["pixel_density"] = np.sum(binary > 0) / (binary.shape[0] * binary.shape[1])
            
            # Waviness/smoothness metrics
            # Calculate number of direction changes in signature contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                # Approximate contour and count corners
                epsilon = 0.02 * cv2.arcLength(max_contour, True)
                approx = cv2.approxPolyDP(max_contour, epsilon, True)
                features["contour_complexity"] = len(approx) / cv2.arcLength(max_contour, True)
            else:
                features["contour_complexity"] = 0
                
            return features
        except Exception as e:
            logger.error(f"Error extracting signature features: {str(e)}")
            return {"error": str(e)}
    
    def analyze_signature_consistency(self, features):
        """Analyze signature for signs of forgery using traditional metrics"""
        # This would be more sophisticated in a real implementation
        result = {
            "unusual_stroke_pattern": False,
            "inconsistent_pressure": False,
            "tremor_signs": False,
            "ai_generation_signs": False,
        }
        
        # Check for signs of AI generation or inconsistency
        # These are placeholder tests - real implementation would use more sophisticated methods
        
        # High edge density might indicate digital creation
        if features.get("edge_density", 0) > 0.5:  # Arbitrary threshold
            result["ai_generation_signs"] = True
            
        # Very smooth contours might indicate AI generation
        if features.get("contour_complexity", 0) < 0.01:  # Arbitrary threshold
            result["unusual_stroke_pattern"] = True
            
        # Add more sophisticated analysis here in a real implementation
            
        return result
    
    def detect_with_model(self, signature_img, reference_signature=None):
        """Use the signature forgery detection model if available"""
        if not self.model_available:
            return {"model_analysis": "unavailable"}
        
        try:
            # If reference signature not provided, use placeholder
            # In a real application, you would retrieve the correct reference signature
            if reference_signature is None:
                # Create a dummy reference as placeholder
                # Ideally, we'd have a database of verified signatures to compare against
                logger.warning("No reference signature provided, analysis may be limited")
                reference_signature = signature_img.copy()  # This is a simplification
            
            # Use our SignatureForgeryModel for prediction
            result = self.model.predict(reference_signature, signature_img)
            
            return {
                "is_fake": result["is_forged"],
                "confidence": result["confidence"],
                "forgery_score": result["forgery_score"]
            }
            
        except Exception as e:
            logger.error(f"Error in model prediction: {str(e)}")
            return {"model_analysis": f"error: {str(e)}"}
    
    def detect(self, signature_path):
        """Main detection method for a signature file"""
        logger.info(f"Starting signature forgery detection on: {signature_path}")
        
        # Generate a unique detection ID
        detection_id = str(uuid.uuid4())
        
        try:
            # Preprocess signature
            signature_img = self.preprocess_signature(signature_path)
            
            # Extract features for traditional analysis
            features = self.extract_features(signature_img)
            
            # Analyze signature consistency
            consistency_analysis = self.analyze_signature_consistency(features)
            
            # Use model for detection if available
            model_result = self.detect_with_model(signature_img)
            
            # Determine if fake based on all analyses
            is_fake = False
            confidence = 0.5
            
            if self.model_available and isinstance(model_result, dict) and "is_fake" in model_result:
                # If we have a model, use its prediction
                is_fake = model_result["is_fake"]
                confidence = model_result["confidence"]
            else:
                # Otherwise use rule-based detection from analyses
                is_fake = (
                    consistency_analysis.get("unusual_stroke_pattern", False) or
                    consistency_analysis.get("ai_generation_signs", False)
                )
                
                # Simple confidence calculation
                confidence_factors = 0
                confidence_total = 0
                
                for key, value in consistency_analysis.items():
                    if isinstance(value, bool):
                        confidence_total += 1
                        if value:
                            confidence_factors += 1
                            
                if confidence_total > 0:
                    confidence = max(0.5, confidence_factors / confidence_total)
            
            # Put together the result
            result = {
                "success": True,
                "detection_id": detection_id,
                "timestamp": datetime.now().isoformat(),
                "filename": os.path.basename(signature_path),
                "is_fake": is_fake,
                "confidence": confidence,
                "details": {
                    "feature_analysis": features,
                    "consistency_analysis": consistency_analysis,
                    "model_analysis": model_result if self.model_available else "unavailable"
                }
            }
            
            # Save detection result to file
            result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     '..', '..', 'data', 'results')
            os.makedirs(result_dir, exist_ok=True)
            result_path = os.path.join(result_dir, f"{detection_id}.json")
            
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
                
            logger.info(f"Signature detection completed. Result saved to {result_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error in signature detection: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "detection_id": detection_id
            } 