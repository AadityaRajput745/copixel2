import os
import numpy as np
import uuid
import logging
from datetime import datetime
import cv2
import json
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from ..models.document_forgery_model import DocumentForgeryModel

logger = logging.getLogger(__name__)

class DocumentForgeryDetector:
    def __init__(self, model_path=None):
        self.model_path = model_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                    '..', 'models', 'document_forgery_model.h5')
        
        # Initialize OCR engine
        try:
            # Tesseract path may need to be set explicitly on Windows
            # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            self._test_ocr_engine()
            self.ocr_available = True
        except Exception as e:
            logger.error(f"OCR engine not available: {str(e)}")
            self.ocr_available = False
            
        # Load document forgery model
        try:
            self.model = DocumentForgeryModel(self.model_path)
            logger.info(f"Loaded document forgery model from {self.model_path}")
            self.model_available = True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model_available = False
    
    def _test_ocr_engine(self):
        """Test if OCR engine is working properly"""
        # Create a simple test image
        test_img = np.zeros((50, 200), dtype=np.uint8)
        cv2.putText(test_img, "TEST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        # Try to OCR the test image
        test_pil = Image.fromarray(test_img)
        text = pytesseract.image_to_string(test_pil).strip()
        if not text:
            logger.warning("OCR test returned no text, may not be working correctly")
            
    def preprocess_document(self, document_path):
        """Preprocess the document for analysis"""
        images = []
        
        # Handle different file types
        file_extension = os.path.splitext(document_path)[1].lower()
        
        try:
            if file_extension == '.pdf':
                # Convert PDF to images
                pages = convert_from_path(document_path, 300)  # 300 DPI
                for page in pages:
                    # Convert PIL Image to numpy array
                    img = np.array(page)
                    # Convert to grayscale if needed
                    if len(img.shape) == 3:
                        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    else:
                        img_gray = img
                    images.append(img_gray)
            else:
                # Load image file
                img = cv2.imread(document_path)
                if img is None:
                    raise ValueError(f"Could not read image file: {document_path}")
                # Convert to grayscale
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(img_gray)
                
            return images
        except Exception as e:
            logger.error(f"Error preprocessing document: {str(e)}")
            raise
    
    def extract_text(self, images):
        """Extract text from document images using OCR"""
        if not self.ocr_available:
            return {"error": "OCR engine not available"}
        
        all_text = ""
        
        try:
            for img in images:
                # Convert numpy array to PIL Image
                pil_img = Image.fromarray(img)
                # Extract text
                text = pytesseract.image_to_string(pil_img)
                all_text += text + "\n\n"
                
            return all_text.strip()
        except Exception as e:
            logger.error(f"Error in OCR: {str(e)}")
            return {"error": f"OCR failed: {str(e)}"}
    
    def analyze_text_inconsistencies(self, text):
        """Analyze text for inconsistencies that might indicate AI generation"""
        if isinstance(text, dict) and "error" in text:
            return {"text_analysis_failed": True}
        
        # Simple analysis for demo purposes
        # In a real implementation, use NLP techniques for deeper analysis
        result = {
            "text_length": len(text),
            "suspicious_patterns": False,
            "unusual_formatting": False,
            "suspicious_words": [],
        }
        
        # Check for common AI-generated content markers (placeholder)
        suspicious_phrases = ["AI generated", "language model", "as an AI", "I'm an AI"]
        for phrase in suspicious_phrases:
            if phrase.lower() in text.lower():
                result["suspicious_patterns"] = True
                result["suspicious_words"].append(phrase)
                
        return result
    
    def analyze_visual_inconsistencies(self, images):
        """Analyze document images for visual inconsistencies"""
        result = {
            "inconsistent_fonts": False,
            "unusual_spacing": False,
            "unusual_alignment": False,
            "digital_artifacts": False,
            "signature_anomalies": False,
        }
        
        # In a real implementation, these functions would contain sophisticated
        # computer vision algorithms to detect specific types of forgeries
        
        # Basic placeholder implementations
        for img in images:
            # Check for digital artifacts (simple edge detection as placeholder)
            edges = cv2.Canny(img, 100, 200)
            if np.mean(edges) > 10:  # Arbitrary threshold
                result["digital_artifacts"] = True
                
            # Other analyses would be added here
                
        return result
    
    def detect_with_model(self, images):
        """Use the document forgery detection model if available"""
        if not self.model_available:
            return {"model_analysis": "unavailable"}
        
        # Process the first image (or multiple if needed)
        try:
            # For documents with multiple pages, analyze each page separately
            results = []
            for img in images:
                # Use our DocumentForgeryModel for prediction
                result = self.model.predict(img)
                results.append(result)
            
            # Aggregate results from multiple pages if needed
            if len(results) > 1:
                # If any page is forged, consider the document forged
                is_fake = any(result["is_forged"] for result in results)
                
                # Use the highest confidence result
                confidence = max(result["forgery_confidence"] for result in results)
                
                # Get the most frequent forgery type
                forgery_types = [result["forgery_type"] for result in results]
                from collections import Counter
                forgery_type = Counter(forgery_types).most_common(1)[0][0]
                
                return {
                    "is_fake": is_fake,
                    "confidence": confidence,
                    "forgery_type": forgery_type,
                    "page_results": results
                }
            else:
                # Return result for a single page document
                return results[0]
        
        except Exception as e:
            logger.error(f"Error in model prediction: {str(e)}")
            return {"model_analysis": f"error: {str(e)}"}
    
    def detect(self, document_path):
        """Main detection method for a document file"""
        logger.info(f"Starting document forgery detection on: {document_path}")
        
        # Generate a unique detection ID
        detection_id = str(uuid.uuid4())
        
        try:
            # Preprocess document
            images = self.preprocess_document(document_path)
            
            # Extract text if OCR is available
            text = self.extract_text(images) if self.ocr_available else {"error": "OCR not available"}
            
            # Analyze text for inconsistencies
            text_analysis = self.analyze_text_inconsistencies(text)
            
            # Analyze visual elements
            visual_analysis = self.analyze_visual_inconsistencies(images)
            
            # Use model for detection if available
            model_result = self.detect_with_model(images)
            
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
                    text_analysis.get("suspicious_patterns", False) or
                    visual_analysis.get("digital_artifacts", False) or
                    visual_analysis.get("signature_anomalies", False)
                )
                
                # Simple confidence calculation
                confidence_factors = 0
                confidence_total = 0
                
                for key, value in visual_analysis.items():
                    if isinstance(value, bool):
                        confidence_total += 1
                        if value:
                            confidence_factors += 1
                            
                if "suspicious_patterns" in text_analysis:
                    confidence_total += 1
                    if text_analysis["suspicious_patterns"]:
                        confidence_factors += 1
                        
                if confidence_total > 0:
                    confidence = max(0.5, confidence_factors / confidence_total)
            
            # Put together the result
            result = {
                "success": True,
                "detection_id": detection_id,
                "timestamp": datetime.now().isoformat(),
                "filename": os.path.basename(document_path),
                "is_fake": is_fake,
                "confidence": confidence,
                "details": {
                    "text_analysis": text_analysis,
                    "visual_analysis": visual_analysis,
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
                
            logger.info(f"Document detection completed. Result saved to {result_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error in document detection: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "detection_id": detection_id
            } 