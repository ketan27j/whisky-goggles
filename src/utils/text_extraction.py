"""Text extraction utilities for whisky bottle recognition."""

import re
import time
import pytesseract
from PIL import Image
import numpy as np
from typing import Dict, Any
from google import genai
from google.genai import types
import requests
import os
from dotenv import load_dotenv 

load_dotenv()

# def optimize_tesseract_config():
#     """
#     Returns optimized Tesseract configuration for code extraction
#     """
#     # Configuration options for Tesseract
#     config = (
#         '--psm 11 --oem 3'
#     )
#     return config

# def extract_text(image: np.ndarray) -> str:
#     """
#     Extract text from image using OCR.
    
#     Args:
#         image: Input preprocessed image
        
#     Returns:
#         Extracted text
#     """
#     config = optimize_tesseract_config()

#     # Convert OpenCV image to PIL image for Tesseract
#     pil_image = Image.fromarray(image)
    
#     # Extract text using Tesseract OCR
#     text = pytesseract.image_to_string(pil_image,config=config)
    
#     # Clean and normalize text
#     text = text.lower()
#     text = re.sub(r'[^\w\s]', '', text)  # Remove punctuatio    n
#     text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    
#     return text

def extract_text_llm(image_path: str = None, image_url: str = None) -> str:
    """
    Extract text from image using LLM.

    Args:
        image: Input preprocessed image

    Returns:
        Extracted text
    """
    # TODO: Implement LLM-based text extraction
    print(image_path)
    response = ''
    prompt = """
        Analyze the provided image of a liquor bottle or box and extract all textual information present in the image correctly. Provide only text from image separated by space.
        """
    
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    if(image_path):
        my_file = client.files.upload(file=image_path, config={'mime_type': 'image/jpeg'})        
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[my_file,prompt])
    if(image_url):
        image_bytes = requests.get(image_url).content
        image = types.Part.from_bytes(
            data=image_bytes, mime_type="image/jpeg"
        )
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[prompt,image])

    return response.text


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

    # Extract proof statement (100 proof)
    proof_pattern = r'(\d{1,2})[\s]*(proof)'
    proof_match = re.search(proof_pattern, text.lower())
    if proof_match:
        metadata['proof'] = int(proof_match.group(1))
    
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