#!/usr/bin/env python
"""
Command-line script for COPixel detection system.

Usage:
    python detect.py --mode deepfake --input <video_file>
    python detect.py --mode document --input <document_file>
    python detect.py --mode signature --input <query_signature> --reference <reference_signature>
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the detector
from src.detection.copixel_detector import COPixelDetector

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='COPixel Detection System')
    
    parser.add_argument('--mode', type=str, required=True, choices=['deepfake', 'document', 'signature'],
                        help='Detection mode: deepfake, document, or signature')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input file (video for deepfake, document image for document, query signature for signature)')
    
    parser.add_argument('--reference', type=str, required=False,
                        help='Path to the reference signature (only for signature mode)')
    
    parser.add_argument('--models-dir', type=str, default=None,
                        help='Directory containing model files (optional)')
    
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file (optional, defaults to stdout)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input):
        parser.error(f"Input file not found: {args.input}")
    
    if args.mode == 'signature' and not args.reference:
        parser.error("Reference signature is required for signature mode")
    
    if args.mode == 'signature' and not os.path.exists(args.reference):
        parser.error(f"Reference signature file not found: {args.reference}")
    
    if args.models_dir and not os.path.exists(args.models_dir):
        parser.error(f"Models directory not found: {args.models_dir}")
    
    return args

def main():
    """Main function for the detection script."""
    # Parse command line arguments
    args = parse_args()
    
    # Initialize the detector
    detector = COPixelDetector(models_dir=args.models_dir)
    
    try:
        # Perform detection based on mode
        if args.mode == 'deepfake':
            logger.info(f"Performing deepfake detection on {args.input}")
            result = detector.detect_deepfake(args.input)
        
        elif args.mode == 'document':
            logger.info(f"Performing document forgery detection on {args.input}")
            result = detector.detect_document_forgery(args.input)
        
        elif args.mode == 'signature':
            logger.info(f"Performing signature forgery detection")
            result = detector.detect_signature_forgery(args.input, args.reference)
        
        # Add input file information to result
        result['input_file'] = args.input
        if args.mode == 'signature':
            result['reference_file'] = args.reference
        
        # Output the result
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Result saved to {args.output}")
        else:
            # Print result to stdout
            print(json.dumps(result, indent=2))
        
        # Print a summary
        if args.mode == 'deepfake':
            is_positive = result.get('is_deepfake', False)
            confidence = result.get('confidence', 0)
            print(f"\nDeepfake detection result: {'DEEPFAKE DETECTED' if is_positive else 'NO DEEPFAKE DETECTED'}")
            print(f"Confidence: {confidence:.2f}\n")
        
        elif args.mode == 'document':
            is_positive = result.get('is_forged', False)
            confidence = result.get('forgery_confidence', 0)
            forgery_type = result.get('forgery_type', 'unknown')
            print(f"\nDocument forgery detection result: {'FORGED' if is_positive else 'AUTHENTIC'}")
            print(f"Forgery confidence: {confidence:.2f}")
            if is_positive:
                print(f"Forgery type: {forgery_type}")
                print(f"Type confidence: {result.get('type_confidence', 0):.2f}\n")
        
        elif args.mode == 'signature':
            is_positive = result.get('is_forged', False)
            confidence = result.get('confidence', 0)
            print(f"\nSignature forgery detection result: {'FORGED' if is_positive else 'AUTHENTIC'}")
            print(f"Confidence: {confidence:.2f}\n")
    
    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 