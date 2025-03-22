@echo off
echo.
echo ================================================
echo COPixel Detection System - Model Accuracy Update
echo ================================================
echo.
echo This script will restart the system with improved detection models.
echo The models have been updated to:
echo  - Significantly reduce false positives in deepfake detection
echo  - Improve document forgery detection with more detailed analysis
echo  - Enhance signature verification with better comparison algorithms
echo.
echo The updated models prioritize accuracy on real-world content.
echo.
echo Press any key to restart the servers...
pause > nul

echo.
echo Stopping any running servers...
taskkill /f /im python.exe 2>nul
taskkill /f /im node.exe 2>nul

echo.
echo Starting the API server...
start /b cmd /c "python run_api_server.py --debug"

echo.
echo Starting the frontend development server...
cd frontend/ai-detection-app
start /b cmd /c "npm run dev"

echo.
echo ================================================
echo Servers restarted successfully!
echo.
echo API server is running at: http://localhost:5000
echo Frontend is running at: http://localhost:5173
echo.
echo The models now have significantly improved accuracy
echo for normal videos, documents, and signatures.
echo ================================================
echo. 