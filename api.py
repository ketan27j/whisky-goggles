"""API for the whisky bottle recognition system."""

from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from pathlib import Path
import json

from src.models.recognizer import WhiskyBottleRecognizer
from config import DATABASE_PATH, IMAGES_DIR, API_HOST, API_PORT

class WhiskyRecognitionAPI:
    def __init__(self, recognizer=None):
        """
        Initialize the API.
        
        Args:
            recognizer: WhiskyBottleRecognizer instance (created if None)
        """
        self.app = Flask(__name__)
        
        # Create recognizer if not provided
        if recognizer is None:
            print("Initializing recognizer...")
            self.recognizer = WhiskyBottleRecognizer(DATABASE_PATH, IMAGES_DIR)
        else:
            self.recognizer = recognizer
        
        # Define routes
        self.app.route('/identify', methods=['POST'])(self.identify_bottle)
        self.app.route('/health', methods=['GET'])(self.health_check)
        
    def identify_bottle(self):
        """API endpoint to identify a whisky bottle from an image."""
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        image_file = request.files['image']
        file_path = 'uploads/' + image_file.filename
        image_file.save(file_path)
        # image_bytes = image_file.read()
        image_bytes = open(file_path, 'rb').read()        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get top N parameter (default to 5)
        top_n = int(request.args.get('top_n', 5))
        
        # Identify bottle
        try:
            results = self.recognizer.identify_bottle(image, top_n=top_n, input_image=file_path)
            return jsonify({
                'matches': results
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def health_check(self):
        """API endpoint to check if the service is running."""
        return jsonify({'status': 'ok'})
    
    def run(self, host=API_HOST, port=API_PORT):
        """Run the Flask application."""
        self.app.run(host=host, port=port)

def create_app(recognizer=None):
    """Create and return the Flask app instance."""
    api = WhiskyRecognitionAPI(recognizer)
    return api.app

if __name__ == "__main__":
    # Run the API
    api = WhiskyRecognitionAPI()
    api.run()