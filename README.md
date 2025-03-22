# COPixel AI Detection System

COPixel is an advanced AI-powered detection system for identifying manipulated digital content, including deepfake videos, forged documents, and signature forgeries.

## Features

- **Video Deepfake Detection**: Detect synthetic or manipulated faces in videos using facial inconsistency analysis
- **Document Forgery Detection**: Identify altered or fabricated documents using content analysis algorithms
- **Signature Forgery Detection**: Compare signatures to determine authenticity with high confidence
- **Modern Web Interface**: User-friendly React frontend for easy content upload and analysis
- **Detailed Result Analysis**: Get confidence scores and detailed insights on detection results

## Project Structure

```
COPixel/
├── models/                   # Directory for pre-trained models
├── src/
│   ├── models/               # Model definitions and training scripts
│   │   ├── deepfake_model.py          # Deepfake detection model implementation
│   │   ├── document_forgery_model.py  # Document forgery detection model
│   │   ├── signature_forgery_model.py # Signature forgery detection model
│   │   ├── download_models.py         # Script to download pre-trained models
│   │   ├── train_models.py            # Script to train models from scratch
│   │   └── prepare_models.py          # Script to prepare model files
│   ├── detection/            # Detection module
│   │   ├── copixel_detector.py        # Main detector class
│   │   └── __init__.py                # Package initialization
│   ├── api_server.py         # Flask API server
│   └── detect.py             # Command-line detection script
├── run_api_server.py         # Script to run the API server
├── uploads/                  # Temporary upload directory (created automatically)
├── reports/                  # Directory for saved reports (created automatically)
└── frontend/
    └── ai-detection-app/     # React frontend application
        ├── src/
        │   ├── components/   # React components
        │   ├── pages/        # Page components
        │   ├── utils/        # Utility functions
        │   └── assets/       # Images and styles
        └── ...               # Frontend configuration
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Node.js 14.x or higher (for frontend)
- CUDA-compatible GPU recommended but not required

### Installation

1. **Clone the repository**:
   ```
   git clone https://github.com/yourusername/COPixel.git
   cd COPixel
   ```

2. **Create and activate a virtual environment**:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Download pre-trained models**:
   ```
   python src/models/download_models.py --all
   ```

5. **Install frontend dependencies**:
   ```
   cd frontend/ai-detection-app
   npm install
   ```

### Usage

#### Command Line Detection

You can use the detection system directly from the command line:

```bash
# For deepfake detection
python src/detect.py --mode deepfake --input path/to/video.mp4

# For document forgery detection
python src/detect.py --mode document --input path/to/document.jpg

# For signature forgery detection
python src/detect.py --mode signature --input path/to/query_signature.jpg --reference path/to/reference_signature.jpg
```

#### Running the Web Application

1. **Start the API server** (in a separate terminal):
   ```bash
   # From the project root
   python run_api_server.py --debug
   ```
   The API server will start at `http://localhost:5000`

2. **Start the frontend development server** (in another terminal):
   ```bash
   cd frontend/ai-detection-app
   npm run dev
   ```

3. **Access the web application** at `http://localhost:5173`

4. **Demo Mode**: The frontend includes a demo mode that works without the backend API server. To use it:
   - Open `frontend/ai-detection-app/src/utils/api.js`
   - Set `DEMO_MODE = true` (it's true by default)
   - This allows testing the UI with simulated responses

## Model Training

If you want to train the models using your own data:

1. Prepare your training data
2. Run the training script:
   ```
   python src/models/train_models.py --model [deepfake|document|signature]
   ```

## Technical Details

### Deepfake Detection

The deepfake detection model analyzes facial features, temporal inconsistencies, and visual 
artifacts to identify manipulated videos. It uses a combination of CNN and LSTM layers to capture 
both spatial and temporal patterns.

### Document Forgery Detection

The document forgery system examines document images for:
- Digital manipulation traces
- Inconsistent text or typography
- Seal or watermark authenticity
- Metadata tampering signs

### Signature Forgery Detection

The signature verification compares a query signature against a reference using a Siamese neural 
network architecture to determine similarity and detect potential forgeries.

## Web Frontend

The React-based frontend provides an intuitive interface for:
- Uploading content for analysis
- Visualizing detection results
- Managing detection history
- Learning about AI detection technology

## API Endpoints

The backend provides the following API endpoints:

- `POST /api/detect/deepfake`: Upload and analyze a video for deepfake detection
- `POST /api/detect/document`: Upload and analyze a document for forgery detection
- `POST /api/detect/signature`: Upload and compare signatures for forgery detection
- `POST /api/report`: Submit a detection report
- `GET /api/statistics`: Get usage statistics
- `GET /api/health`: Health check endpoint

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The project uses TensorFlow and PyTorch for deep learning models
- Frontend built with React and Bootstrap 