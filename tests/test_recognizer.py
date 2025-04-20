"""Tests for the whisky bottle recognizer."""

import unittest
import os
import cv2
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import pandas as pd

from src.models.recognizer import WhiskyBottleRecognizer
from src.utils.image_processing import preprocess_image, detect_label
from src.utils.text_extraction import extract_text, extract_metadata

class TestImageProcessing(unittest.TestCase):
    """Test image processing utilities."""
    
    def test_preprocess_image(self):
        """Test image preprocessing function."""
        # Create a simple test image
        test_image = np.ones((300, 200), dtype=np.uint8) * 128
        
        # Process the image
        processed = preprocess_image(test_image, max_dimension=800)
        
        # Check that image dimensions are preserved (since already below max_dimension)
        self.assertEqual(processed.shape, test_image.shape)
        
        # Test with larger image
        large_image = np.ones((1200, 900), dtype=np.uint8) * 128
        processed = preprocess_image(large_image, max_dimension=800)
        
        # Check that image was resized while maintaining aspect ratio
        self.assertEqual(processed.shape[0], 800)
        self.assertEqual(processed.shape[1], 600)
    
    def test_detect_label(self):
        """Test label detection."""
        # Create a test image with a simulated label
        background = np.zeros((500, 300), dtype=np.uint8)
        # Add a rectangular "label" in the middle
        label_region = background.copy()
        label_region[150:350, 75:225] = 255
        
        # Detect label region
        result = detect_label(label_region)
        
        # Since this is a simplified test, we just check that we get an image back
        self.assertIsInstance(result, np.ndarray)

class TestTextExtraction(unittest.TestCase):
    """Test text extraction utilities."""
    
    @patch('pytesseract.image_to_string')
    def test_extract_text(self, mock_tesseract):
        """Test text extraction from image."""
        # Mock the tesseract OCR function to return test text
        mock_tesseract.return_value = "Glenlivet 12 Years 40% ABV"
        
        # Create a simple test image
        test_image = np.ones((100, 300), dtype=np.uint8) * 255
        
        # Extract text
        text = extract_text(test_image)
        
        # Check that text was properly extracted and normalized
        self.assertEqual(text, "glenlivet 12 years 40 abv")
    
    def test_extract_metadata(self):
        """Test metadata extraction from text."""
        # Test ABV extraction
        text_with_abv = "single malt scotch whisky 43% abv"
        metadata = extract_metadata(text_with_abv)
        self.assertIn('abv', metadata)
        self.assertEqual(metadata['abv'], 43.0)
        
        # Test age extraction
        text_with_age = "highland park 12 year old"
        metadata = extract_metadata(text_with_age)
        self.assertIn('age', metadata)
        self.assertEqual(metadata['age'], 12)
        
        # Test both together
        text_with_both = "macallan 18 year old 43.5% alc./vol."
        metadata = extract_metadata(text_with_both)
        self.assertEqual(metadata['abv'], 43.5)
        self.assertEqual(metadata['age'], 18)

class TestWhiskyBottleRecognizer(unittest.TestCase):
    """Test the main recognizer class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a mock database file
        self.db_path = os.path.join(self.temp_dir.name, "test_db.csv")
        self.images_dir = os.path.join(self.temp_dir.name, "images")
        
        # Create a sample database
        df = pd.DataFrame({
            'id': [1, 2],
            'name': ['Test Whisky 1', 'Test Whisky 2'],
            'brand_id': [10, 20],
            'abv': [40.0, 43.0],
            'spirit_type': ['scotch', 'bourbon'],
            'image_url': ['http://example.com/img1.jpg', 'http://example.com/img2.jpg']
        })
        
        df.to_csv(self.db_path, index=False)
        
        # Create directory for images
        os.makedirs(self.images_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()
    
    @patch('requests.get')
    @patch('cv2.imread')
    @patch('src.models.recognizer.extract_features')
    @patch('src.models.recognizer.extract_text')
    def test_prepare_reference_data(self, mock_extract_text, mock_extract_features, 
                                   mock_imread, mock_requests_get):
        """Test preparation of reference data."""
        # Mock the requests.get response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'fake_image_data'
        mock_requests_get.return_value = mock_response
        
        # Mock image loading
        mock_image = np.ones((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        
        # Mock feature extraction
        mock_keypoints = [MagicMock()]
        mock_descriptors = np.ones((10, 32), dtype=np.uint8)
        mock_extract_features.return_value = (mock_keypoints, mock_descriptors)
        
        # Mock text extraction
        mock_extract_text.return_value = "test whisky 40% abv"
        
        # Initialize recognizer with mocked functions
        with patch('src.models.recognizer.preprocess_image', return_value=mock_image):
            recognizer = WhiskyBottleRecognizer(self.db_path, self.images_dir)
            
            # Check if reference features were properly prepared
            self.assertEqual(len(recognizer.reference_features), 2)
            self.assertIn(1, recognizer.reference_features)
            self.assertIn(2, recognizer.reference_features)
    
    @patch('src.models.recognizer.load_image')
    @patch('src.models.recognizer.detect_label')
    @patch('src.models.recognizer.preprocess_image')
    @patch('src.models.recognizer.extract_features')
    @patch('src.models.recognizer.extract_text')
    @patch('src.models.recognizer.extract_metadata')
    def test_identify_bottle(self, mock_extract_metadata, mock_extract_text, 
                            mock_extract_features, mock_preprocess, 
                            mock_detect_label, mock_load_image):
        """Test bottle identification flow."""
        # Set up mocks for identification process
        mock_image = np.ones((100, 100, 3), dtype=np.uint8)
        mock_load_image.return_value = mock_image
        mock_detect_label.return_value = mock_image
        mock_preprocess.return_value = mock_image
        
        mock_keypoints = [MagicMock()]
        mock_descriptors = np.ones((10, 32), dtype=np.uint8)
        mock_extract_features.return_value = (mock_keypoints, mock_descriptors)
        
        mock_extract_text.return_value = "test whisky 40% abv"
        mock_extract_metadata.return_value = {'abv': 40.0}
        
        # Create a recognizer with mock reference data
        with patch.object(WhiskyBottleRecognizer, '_prepare_reference_data'):
            recognizer = WhiskyBottleRecognizer(self.db_path, self.images_dir)
            
            # Set up mock reference features
            recognizer.reference_features = {
                1: {
                    'descriptors': np.ones((10, 32), dtype=np.uint8),
                    'text': 'test whisky 40% abv',
                    'name': 'Test Whisky 1',
                    'brand_id': 10,
                    'abv': 40.0,
                    'spirit_type': 'scotch'
                },
                2: {
                    'descriptors': np.ones((10, 32), dtype=np.uint8) * 2,
                    'text': 'test whisky 2 43% abv',
                    'name': 'Test Whisky 2',
                    'brand_id': 20,
                    'abv': 43.0,
                    'spirit_type': 'bourbon'
                }
            }
            
            # Mock similarity calculation methods
            with patch.object(recognizer, '_calculate_visual_similarity', return_value=0.8), \
                 patch.object(recognizer, '_calculate_metadata_similarity', return_value=0.9):
                
                # Test identification
                results = recognizer.identify_bottle('test_image.jpg', top_n=2)
                
                # Check results structure
                self.assertEqual(len(results), 2)
                self.assertIn('id', results[0])
                self.assertIn('name', results[0])
                self.assertIn('confidence', results[0])
                self.assertIn('visual_score', results[0])
                self.assertIn('text_score', results[0])
                self.assertIn('metadata_score', results[0])

if __name__ == '__main__':
    unittest.main()