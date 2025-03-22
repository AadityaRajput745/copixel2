#!/usr/bin/env python
import os
import sys
import argparse

def main():
    """
    Main entry point for running the COPixel API server.
    """
    parser = argparse.ArgumentParser(description='Run the COPixel API server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind the server to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Add the project root to the Python path to allow imports
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Import the API server after setting up the path
    from src.api_server import app
    
    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main() 