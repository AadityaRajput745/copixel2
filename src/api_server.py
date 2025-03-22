"""
Flask API server for COPixel Detection System.
This script provides API endpoints for the frontend to interact with the detection models.
"""

import os
import sys
import json
import logging
import tempfile
import uuid
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add the project root directory to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the detector
from src.detection.copixel_detector import COPixelDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
ALLOWED_EXTENSIONS = {
    'deepfake': {'mp4', 'avi', 'mov', 'wmv', 'webm'},
    'document': {'jpg', 'jpeg', 'png', 'pdf', 'tiff'},
    'signature': {'jpg', 'jpeg', 'png'}
}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the detector
detector = COPixelDetector()

def allowed_file(filename, detection_type):
    """Check if file extension is allowed for the given detection type"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS.get(detection_type, {})

def save_uploaded_file(file, detection_type):
    """Save the uploaded file to a temporary location and return the path"""
    if file and allowed_file(file.filename, detection_type):
        filename = secure_filename(file.filename)
        # Create a unique filename to avoid collisions
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        return file_path
    return None

def get_timestamp():
    """Get current timestamp in ISO format"""
    return datetime.datetime.now().isoformat()

@app.route('/api/detect/deepfake', methods=['POST'])
def detect_deepfake():
    """API endpoint for deepfake detection"""
    logger.info("Received deepfake detection request")
    
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file provided',
            'timestamp': get_timestamp()
        }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'error': 'No file selected',
            'timestamp': get_timestamp()
        }), 400
    
    file_path = save_uploaded_file(file, 'deepfake')
    if not file_path:
        return jsonify({
            'error': 'Invalid file type',
            'timestamp': get_timestamp()
        }), 400
    
    try:
        # Process the video for deepfake detection
        result = detector.detect_deepfake(file_path)
        
        # Clean up the temporary file after processing
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {file_path}: {e}")
        
        # Add timestamp to the response
        result['timestamp'] = get_timestamp()
        result['source'] = file.filename
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in deepfake detection: {e}")
        return jsonify({
            'error': f'Detection failed: {str(e)}',
            'timestamp': get_timestamp()
        }), 500

@app.route('/api/detect/document', methods=['POST'])
def detect_document_forgery():
    """API endpoint for document forgery detection"""
    logger.info("Received document forgery detection request")
    
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file provided',
            'timestamp': get_timestamp()
        }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'error': 'No file selected',
            'timestamp': get_timestamp()
        }), 400
    
    file_path = save_uploaded_file(file, 'document')
    if not file_path:
        return jsonify({
            'error': 'Invalid file type',
            'timestamp': get_timestamp()
        }), 400
    
    try:
        # Process the document for forgery detection
        result = detector.detect_document_forgery(file_path)
        
        # Clean up the temporary file after processing
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {file_path}: {e}")
        
        # Add timestamp to the response
        result['timestamp'] = get_timestamp()
        result['source'] = file.filename
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in document forgery detection: {e}")
        return jsonify({
            'error': f'Detection failed: {str(e)}',
            'timestamp': get_timestamp()
        }), 500

@app.route('/api/detect/signature', methods=['POST'])
def detect_signature_forgery():
    """API endpoint for signature forgery detection"""
    logger.info("Received signature forgery detection request")
    
    if 'file' not in request.files:
        return jsonify({
            'error': 'No query signature provided',
            'timestamp': get_timestamp()
        }), 400
    
    if 'reference' not in request.files:
        return jsonify({
            'error': 'No reference signature provided',
            'timestamp': get_timestamp()
        }), 400
    
    query_file = request.files['file']
    reference_file = request.files['reference']
    
    if query_file.filename == '' or reference_file.filename == '':
        return jsonify({
            'error': 'One or more files not selected',
            'timestamp': get_timestamp()
        }), 400
    
    query_path = save_uploaded_file(query_file, 'signature')
    reference_path = save_uploaded_file(reference_file, 'signature')
    
    if not query_path or not reference_path:
        # Clean up any files that were successfully saved
        if query_path and os.path.exists(query_path):
            os.remove(query_path)
        if reference_path and os.path.exists(reference_path):
            os.remove(reference_path)
            
        return jsonify({
            'error': 'Invalid file type for one or more files',
            'timestamp': get_timestamp()
        }), 400
    
    try:
        # Process the signatures for forgery detection
        result = detector.detect_signature_forgery(query_path, reference_path)
        
        # Clean up the temporary files after processing
        try:
            os.remove(query_path)
            os.remove(reference_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary files: {e}")
        
        # Add timestamp and file info to the response
        result['timestamp'] = get_timestamp()
        result['query'] = query_file.filename
        result['reference'] = reference_file.filename
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in signature forgery detection: {e}")
        return jsonify({
            'error': f'Detection failed: {str(e)}',
            'timestamp': get_timestamp()
        }), 500

@app.route('/api/report', methods=['POST'])
def submit_report():
    """API endpoint for submitting reports"""
    logger.info("Received report submission")
    
    data = request.json
    if not data:
        return jsonify({
            'error': 'No report data provided',
            'timestamp': get_timestamp()
        }), 400
    
    try:
        # Generate a unique report ID
        report_id = str(uuid.uuid4())
        
        # Store the report data (in a real application, this would go to a database)
        report_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reports')
        os.makedirs(report_folder, exist_ok=True)
        
        report_file = os.path.join(report_folder, f"report_{report_id}.json")
        with open(report_file, 'w') as f:
            json.dump({
                'id': report_id,
                'timestamp': get_timestamp(),
                'data': data
            }, f, indent=2)
        
        return jsonify({
            'success': True,
            'id': report_id,
            'message': 'Report submitted successfully',
            'timestamp': get_timestamp()
        })
    
    except Exception as e:
        logger.error(f"Error in report submission: {e}")
        return jsonify({
            'error': f'Report submission failed: {str(e)}',
            'timestamp': get_timestamp()
        }), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """API endpoint for detection statistics"""
    logger.info("Received statistics request")
    
    try:
        # In a real application, this would retrieve data from a database
        # For demo purposes, return some mock statistics
        return jsonify({
            'total_detections': 4567,
            'deepfake_detections': 2134,
            'document_forgery_detections': 1623,
            'signature_forgery_detections': 810,
            'detection_accuracy': 95,
            'timestamp': get_timestamp()
        })
    
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        return jsonify({
            'error': f'Failed to retrieve statistics: {str(e)}',
            'timestamp': get_timestamp()
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'API server is running',
        'timestamp': get_timestamp()
    })

if __name__ == '__main__':
    # Add command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='COPixel API Server')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    logger.info(f"Starting COPixel API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug) 