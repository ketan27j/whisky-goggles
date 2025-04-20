"""Text extraction utilities for whisky bottle recognition."""

import re
import pytesseract
from PIL import Image
import numpy as np
from typing import Dict, Any

def extract_text(image: np.ndarray) -> str:
    """
    Extract text from image using OCR.
    
    Args:
        image: Input preprocessed image
        
    Returns:
        Extracted text
    """
    # Convert OpenCV image to PIL image for Tesseract
    pil_image = Image.fromarray(image)
    
    # Extract text using Tesseract OCR
    text = pytesseract.image_to_string(pil_image)
    
    # Clean and normalize text
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    
    return text

def extract_metadata(text: str) -> Dict[str, Any]:
    """
    Extract metadata like ABV from OCR text.
    
    Args:
        text: OCR extracted text
        
    Returns:
        Dictionary of extracted metadata
    """
    metadata = {}
    
    # Extract ABV (typically shown as XX% or XX% ABV or XX% alc./vol.)
    abv_pattern = r'(\d{1,2}\.?\d*)[\s]*%[\s]*(abv|alc|alcohol|vol)'
    abv_match = re.search(abv_pattern, text.lower())
    if abv_match:
        metadata['abv'] = float(abv_match.group(1))
    
    # Extract age statement (X year or X years)
    age_pattern = r'(\d{1,2})[\s]*(year|yr)'
    age_match = re.search(age_pattern, text.lower())
    if age_match:
        metadata['age'] = int(age_match.group(1))
    
    return metadata

def calculate_text_similarity(query_text: str, ref_text: str) -> float:
    """
    Calculate text similarity between query and reference text.
    
    Args:
        query_text: Text extracted from query image
        ref_text: Text extracted from reference image
        
    Returns:
        Similarity score
    """
    from fuzzywuzzy import fuzz
    
    # Use fuzzy string matching for text comparison
    ratio = fuzz.token_set_ratio(query_text, ref_text) / 100.0
    return ratio