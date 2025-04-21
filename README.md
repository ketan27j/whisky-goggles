# Whisky Goggles - Whisky Bottle Recognition System

A computer vision system that identifies whisky bottles from images using feature extraction, OCR text recognition, and similarity matching.

## Features

- Detects and identifies whisky bottles from images
- Extracts text and visual features for matching
- Detects ABV percentage and age statements
- Provides confidence scores for matches
- REST API for integration with other applications
- Visualization tools for matches

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/whisky-recognition-system.git
cd whisky-recognition-system
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Install Tesseract OCR:
   - On Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - On macOS: `brew install tesseract`
   - On Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki

4. Prepare the database:
   - Place your whisky database CSV file at `data/501_Bottle_Dataset.csv`
   - The CSV should contain at minimum these columns: id, name, brand_id, abv, spirit_type, image_url

## Usage

### Run the Demo

```
python run.py demo --image path/to/bottle_image.jpg
```

The demo will:
1. Load the test image
2. Identify potential matches from the database
3. Display and save top matches with confidence scores
4. Generate visualization of matches
5. Save results to JSON

### Run the API Server

```
python run.py api --port 5000
```

The API server provides:
- `POST /identify` - Upload an image to identify a whisky bottle
- `GET /health` - Check if the service is running

Example API usage:
```python
import requests

# Identify a bottle
with open('bottle.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/identify', 
                            files={'image': f},
                            params={'top_n': 3})
    
# Process results
results = response.json()
for match in results['matches']:
    print(f"{match['name']} - Confidence: {match['confidence']}")
```

## Project Structure

```
whisky_recognition_system/
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── config.py               # Configuration settings
├── run.py                  # Main entry point
├── api.py                  # REST API implementation
├── demo.py                 # Demo application
├── tests/                  # Unit tests
│   ├── __init__.py
│   └── test_recognizer.py
└── src/                    # Source code
    ├── __init__.py
    ├── data/               # Data handling
    │   ├── __init__.py
    │   └── database.py
    ├── models/             # Recognition models
    │   ├── __init__.py
    │   └── recognizer.py
    ├── utils/              # Utility functions
    │   ├── __init__.py
    │   ├── image_processing.py
    │   └── text_extraction.py
    └── visualization/      # Visualization tools
        ├── __init__.py
        └── visualizer.py
```

## License

MIT License