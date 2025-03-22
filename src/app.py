import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import logging

# Import detectors
from detectors.video_detector import VideoDeepfakeDetector
from detectors.document_detector import DocumentForgeryDetector
from detectors.signature_detector import SignatureDetector
from utils.reporting import ReportingSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='frontend/static', template_folder='frontend/templates')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize detectors
video_detector = VideoDeepfakeDetector()
document_detector = DocumentForgeryDetector()
signature_detector = SignatureDetector()
reporting_system = ReportingSystem()

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'video': {'mp4', 'avi', 'mov', 'mkv'},
    'document': {'pdf', 'jpg', 'jpeg', 'png'},
    'signature': {'jpg', 'jpeg', 'png', 'pdf'}
}

def allowed_file(filename, file_type):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect/video', methods=['POST'])
def detect_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename, 'video'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = video_detector.detect(filepath)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error in video detection: {str(e)}")
            return jsonify({'error': f'Detection failed: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/detect/document', methods=['POST'])
def detect_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename, 'document'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = document_detector.detect(filepath)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error in document detection: {str(e)}")
            return jsonify({'error': f'Detection failed: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/detect/signature', methods=['POST'])
def detect_signature():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename, 'signature'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = signature_detector.detect(filepath)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error in signature detection: {str(e)}")
            return jsonify({'error': f'Detection failed: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/report', methods=['POST'])
def report_to_authorities():
    data = request.json
    
    if not data or 'detection_id' not in data:
        return jsonify({'error': 'Invalid report data'}), 400
    
    try:
        report_result = reporting_system.report(data)
        return jsonify(report_result)
    except Exception as e:
        logger.error(f"Error in reporting: {str(e)}")
        return jsonify({'error': f'Reporting failed: {str(e)}'}), 500

if __name__ == '__main__':
    logger.info("Starting AI Detection System")
    app.run(debug=True, host='0.0.0.0', port=5000) 