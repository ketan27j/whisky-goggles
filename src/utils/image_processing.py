"""Image processing utilities for whisky bottle recognition."""

import cv2
import numpy as np
from typing import Tuple, Union, List

def preprocess_image(image: np.ndarray, max_dimension: int = 800, for_ocr: bool = False) -> np.ndarray:
    """
    Preprocess the image for better feature extraction.
    
    Args:
        image: Input image
        max_dimension: Maximum dimension for resizing
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Resize to standard size while maintaining aspect ratio
    height, width = gray.shape
    if height > max_dimension or width > max_dimension:
        scale = max_dimension / max(height, width)
        new_height, new_width = int(height * scale), int(width * scale)
        gray = cv2.resize(gray, (new_width, new_height))

    if for_ocr:
        # OCR-specific preprocessing
        # Apply binary thresholding with Otsu's method
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Dilate to make text thicker and more readable
        dilated = cv2.dilate(opening, kernel, iterations=1)
        
        # Sharpen the image
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(dilated, -1, sharpen_kernel)
        
        return sharpened    
    else:
        # Apply adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        return blurred

def detect_label(image: np.ndarray) -> np.ndarray:
    """
    Attempt to detect and isolate the label region from the bottle image.
    
    Args:
        image: Input image
        
    Returns:
        Cropped image containing the label, or original image if detection fails
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Apply threshold to create binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply morphological operations to enhance label regions
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image  # Return original image if no contours found
    
    # Find the largest contour, which might be the label
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Check if the contour is large enough to be a label (at least 20% of the image)
    img_area = gray.shape[0] * gray.shape[1]
    contour_area = w * h
    
    if contour_area > 0.2 * img_area:
        # Crop the image to the bounding rectangle
        label_region = image[y:y+h, x:x+w]
        return label_region
    
    return image  # Return original image if no suitable label region found

def extract_features(image: np.ndarray, orb=None) -> Tuple[List, np.ndarray]:
    """
    Extract ORB features from the image.
    
    Args:
        image: Input preprocessed image
        orb: ORB feature detector (created if None)
        
    Returns:
        Tuple of (keypoints, descriptors)
    """
    if orb is None:
        orb = cv2.ORB_create(nfeatures=1000)
        
    keypoints = orb.detect(image, None)
    keypoints, descriptors = orb.compute(image, keypoints)
    return keypoints, descriptors

def load_image(image_path: Union[str, np.ndarray]) -> np.ndarray:
    """
    Load image from path or return the image if already a numpy array.
    
    Args:
        image_path: Path to image file or numpy array of the image
        
    Returns:
        Loaded image as numpy array
    """
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
    else:
        image = image_path
    
    return image