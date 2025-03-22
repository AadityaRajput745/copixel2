# Quick Start Guide

This guide will help you set up and run the DeepFake & AI-Generated Content Detection System.

## Prerequisites

- Python 3.9+ installed
- pip package manager
- [Optional] Tesseract OCR installed for document detection
- [Optional] CUDA-capable GPU for faster processing

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ai-detection-system.git
   cd ai-detection-system
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment configuration:
   ```
   cp .env.example .env
   ```
   Edit the `.env` file with your configuration settings.

4. Download pre-trained models:
   ```
   python src/models/download_models.py
   ```
   Note: This will download large model files (approx. 240 MB total).

## Running the System

1. Start the web application:
   ```
   python src/app.py
   ```

2. Access the web interface:
   Open a browser and go to `http://localhost:5000`

## Testing the System

To run the test suite:
```
python -m unittest discover tests
```

## Using the API Directly

The system provides RESTful API endpoints for integration with other applications:

### Video Detection
```
POST /api/detect/video
Content-Type: multipart/form-data
Body: file=@path/to/video.mp4
```

### Document Detection
```
POST /api/detect/document
Content-Type: multipart/form-data
Body: file=@path/to/document.pdf
```

### Signature Detection
```
POST /api/detect/signature
Content-Type: multipart/form-data
Body: file=@path/to/signature.png
```

### Report to Authorities
```
POST /api/report
Content-Type: application/json
Body: {
  "detection_id": "uuid-from-detection-result",
  "reporter_name": "Your Name",
  "reporter_contact": "your.email@example.com",
  "reporter_organization": "Your Organization",
  "notes": "Additional information about the report"
}
```

## Troubleshooting

### Model Loading Issues
If you encounter errors when loading models, make sure:
- Models are downloaded correctly via the download script
- You have sufficient system memory (at least 4GB RAM recommended)
- CUDA configuration is correct if using GPU

### Document Detection Issues
If OCR functionality is not working:
- Install Tesseract OCR on your system
- Set the path in the `.env` file for Windows:
  ```
  TESSERACT_PATH=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
  ```

### Web Application Issues
- Check the console output for error messages
- Ensure port 5000 is not in use by another application
- Verify that all dependencies were installed correctly

## Additional Resources

- See the `README.md` file for a more detailed overview of the system
- Check the `docs` directory for detailed documentation (if available)
- Review sample workflows in the `examples` directory (if available) 