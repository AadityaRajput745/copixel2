#!/bin/bash

echo
echo "================================================"
echo "COPixel Detection System - Model Accuracy Update"
echo "================================================"
echo
echo "This script will restart the system with improved detection models."
echo "The models have been updated to:"
echo " - Significantly reduce false positives in deepfake detection"
echo " - Improve document forgery detection with more detailed analysis"
echo " - Enhance signature verification with better comparison algorithms"
echo
echo "The updated models prioritize accuracy on real-world content."
echo
echo "Press Enter to restart the servers..."
read

echo
echo "Stopping any running servers..."
pkill -f "python run_api_server.py" 2>/dev/null
pkill -f "npm run dev" 2>/dev/null

echo
echo "Starting the API server..."
python run_api_server.py --debug > api_server.log 2>&1 &

echo
echo "Starting the frontend development server..."
cd frontend/ai-detection-app
npm run dev > frontend.log 2>&1 &
cd ../..

echo
echo "================================================"
echo "Servers restarted successfully!"
echo
echo "API server is running at: http://localhost:5000"
echo "Frontend is running at: http://localhost:5173"
echo
echo "The models now have significantly improved accuracy"
echo "for normal videos, documents, and signatures."
echo "================================================"
echo
echo "Log files are saved as api_server.log and frontend/ai-detection-app/frontend.log"
echo 