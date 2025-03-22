import os
import sys
import unittest
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detectors.video_detector import VideoDeepfakeDetector
from src.detectors.document_detector import DocumentForgeryDetector
from src.detectors.signature_detector import SignatureDetector
from src.utils.reporting import ReportingSystem

class TestDetectors(unittest.TestCase):
    """Test cases for the detection system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'samples')
        
        # Create sample directory if it doesn't exist
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # Initialize detectors
        self.video_detector = VideoDeepfakeDetector()
        self.document_detector = DocumentForgeryDetector()
        self.signature_detector = SignatureDetector()
        self.reporting_system = ReportingSystem()
    
    def test_detector_initialization(self):
        """Test that detectors initialize correctly"""
        self.assertIsNotNone(self.video_detector)
        self.assertIsNotNone(self.document_detector)
        self.assertIsNotNone(self.signature_detector)
        self.assertIsNotNone(self.reporting_system)
    
    def test_video_detection_output_format(self):
        """Test that video detector returns the expected JSON structure even when no video is provided"""
        # This just tests the API shape, not actual detection
        try:
            # Use a non-existent file to test error handling
            result = self.video_detector.detect(os.path.join(self.sample_dir, 'non_existent_video.mp4'))
            
            # Check that the result is a dictionary with expected keys
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            self.assertIn('detection_id', result)
            
            # This should fail since the file doesn't exist
            self.assertFalse(result['success'])
            self.assertIn('error', result)
        except Exception as e:
            self.fail(f"video_detector.detect raised an exception: {str(e)}")
    
    def test_document_detection_output_format(self):
        """Test that document detector returns the expected JSON structure"""
        try:
            # Use a non-existent file to test error handling
            result = self.document_detector.detect(os.path.join(self.sample_dir, 'non_existent_doc.pdf'))
            
            # Check that the result is a dictionary with expected keys
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            self.assertIn('detection_id', result)
            
            # This should fail since the file doesn't exist
            self.assertFalse(result['success'])
            self.assertIn('error', result)
        except Exception as e:
            self.fail(f"document_detector.detect raised an exception: {str(e)}")
    
    def test_signature_detection_output_format(self):
        """Test that signature detector returns the expected JSON structure"""
        try:
            # Use a non-existent file to test error handling
            result = self.signature_detector.detect(os.path.join(self.sample_dir, 'non_existent_sig.png'))
            
            # Check that the result is a dictionary with expected keys
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            self.assertIn('detection_id', result)
            
            # This should fail since the file doesn't exist
            self.assertFalse(result['success'])
            self.assertIn('error', result)
        except Exception as e:
            self.fail(f"signature_detector.detect raised an exception: {str(e)}")
    
    def test_reporting_system(self):
        """Test that reporting system handles invalid detection IDs gracefully"""
        try:
            # Try to report a non-existent detection
            result = self.reporting_system.report({
                'detection_id': 'non_existent_id',
                'reporter_name': 'Test User',
                'reporter_contact': 'test@example.com'
            })
            
            # Check that the result is a dictionary with expected keys
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            
            # This should fail since the detection doesn't exist
            self.assertFalse(result['success'])
            self.assertIn('error', result)
        except Exception as e:
            self.fail(f"reporting_system.report raised an exception: {str(e)}")

if __name__ == '__main__':
    unittest.main() 