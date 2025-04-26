"""Whisky bottle recognizer model."""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any

from src.data.database import WhiskyDatabase
from src.utils.image_processing import preprocess_image, detect_label, extract_features, load_image
# from src.utils.text_extraction import extract_text
from src.utils.text_extraction import extract_metadata, calculate_text_similarity, extract_text_llm
from config import (
    VISUAL_SIMILARITY_WEIGHT,
    TEXT_SIMILARITY_WEIGHT,
    METADATA_SIMILARITY_WEIGHT,
    MAX_MATCHES,
    MAX_DISTANCE
)

class WhiskyBottleRecognizer:
    def __init__(self, database_path: str, images_dir: str):
        """
        Initialize the whisky bottle recognizer.
        
        Args:
            database_path: Path to the CSV file containing whisky bottle database
            images_dir: Directory to store downloaded bottle images
        """
        self.database = WhiskyDatabase(database_path, images_dir)
        self.images_dir = Path(images_dir)
        
        # Initialize feature extraction tools
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Initialize image and feature databases
        self.reference_images = {}
        self.reference_features = {}
        self.image_descriptors = {}
        
        # Download and prepare reference images and features
        self._prepare_reference_data()
    
    def _prepare_reference_data(self):
        """Download bottle images and extract features for the reference database."""
        print("Preparing reference database...")
        for idx, row in self.database.get_all_bottles().iterrows():
            bottle_id = row['id']
            image_url = row['image_url']
            text_features = str(row['text_features'])
            
            # Download image
            image_path = self.database.download_bottle_image(bottle_id, image_url)
            if image_path is None:
                continue
            
            # Load and preprocess image
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image for bottle ID {bottle_id}")
                    continue
                
                # Store reference image
                self.reference_images[bottle_id] = image
                
                # Extract features
                processed_image = preprocess_image(image)
                keypoints, descriptors = extract_features(processed_image, self.orb)
                
                # processed_image_ocr = preprocess_image(image,for_ocr=True)

                if descriptors is not None:
                    self.image_descriptors[bottle_id] = descriptors
                    # text_features = ''
                    # Extract text using OCR
                    # processed_images = extract_text(processed_image_ocr)
                    # for name, img in processed_image_ocr.items():
                    #     # tempText = extract_text(img)
                    #     # if tempText not in text_features:
                    #     #     text_features += ' ' + tempText
                    #     if text_features.strip() == '':
                    #         text_features = extract_text(img)
                    #         # print(name+': '+text_features)
                    max_length = 0
                        
                    # for name, img in processed_image_ocr.items():
                    #     current_text = extract_text(img)
                    #     if len(current_text) > max_length:
                    #         max_length = len(current_text)
                    #         text_features = current_text
                    # print(f"Extracted text: {text_features} for bottle ID {bottle_id}")
                    print(f"text_features: {text_features.strip()}")
                    if text_features.strip() == "nan" or text_features.strip() == "":       
                        print(f"image url{image_url}")                 
                        text_features = extract_text_llm(image_url=image_url)
                        self.database.save_text_feature(bottle_id,text_features)
                        print(f"Extracted text: {text_features} for bottle ID {bottle_id}")
                    
                    # Store combined features
                    self.reference_features[bottle_id] = {
                        'descriptors': descriptors,
                        'text': text_features,
                        'name': row['name'],
                        'brand_id': row['brand_id'],
                        'abv': row['abv'],
                        'spirit_type': row['spirit_type']
                    }
                    # print(self.reference_features[bottle_id])
            except Exception as e:
                print(f"Error processing image for bottle ID {bottle_id}: {e}")
        
        print(f"Prepared reference database with {len(self.reference_features)} bottles")
    
    def _calculate_visual_similarity(self, query_descriptors, ref_descriptors):
        """
        Calculate visual similarity between query and reference descriptors.
        
        Args:
            query_descriptors: Descriptors of the query image
            ref_descriptors: Descriptors of the reference image
            
        Returns:
            Similarity score
        """
        if query_descriptors is None or ref_descriptors is None:
            return 0.0
            
        # Match descriptors
        try:
            matches = self.bf_matcher.match(query_descriptors, ref_descriptors)
            
            # Sort matches by distance (lower distance is better match)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Use the top matches or all if less
            good_matches = matches[:min(MAX_MATCHES, len(matches))]
            
            if len(good_matches) == 0:
                return 0.0
                
            # Calculate similarity score based on average distance
            avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
            
            # Convert distance to similarity score (0-1)
            similarity = max(0, 1 - (avg_distance / MAX_DISTANCE))
            
            return similarity
        except:
            return 0.0
    
    def _calculate_metadata_similarity(self, extracted_info, reference_info):
        """
        Calculate similarity based on extracted metadata (ABV, age, etc.)
        
        Args:
            extracted_info: Information extracted from query image
            reference_info: Information from reference database
            
        Returns:
            Similarity score
        """
        score = 0.0
        count = 0
        
        # Check for ABV match if available
        if 'abv' in extracted_info and extracted_info['abv'] is not None:
            try:
                abv_diff = abs(float(extracted_info['abv']) - float(reference_info['abv']))
                abv_score = max(0, 1 - (abv_diff / 10))  # Normalize difference
                score += abv_score
                count += 1
            except:
                pass
        
        # Add more metadata checks here as needed
        
        return score / max(1, count) if count > 0 else 0.0
    
    def identify_bottle(self, image_path: Union[str, np.ndarray], top_n: int = 5, input_image = None) -> List[Dict]:
        """
        Identify the whisky bottle from an image.
        
        Args:
            image_path: Path to image file or numpy array of the image
            top_n: Number of top matches to return
            
        Returns:
            List of dictionaries containing matched bottles with confidence scores
        """
        # Load image
        image = load_image(image_path)
        
        # Detect and crop label region
        label_image = detect_label(image)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Extract features
        keypoints, descriptors = extract_features(processed_image, self.orb)
        
        # Extract text
        text = ''
        processed_image_ocr = preprocess_image(image,for_ocr=True)
        max_length = 0
            
        # for name, img in processed_image_ocr.items():
        #     current_text = extract_text(img)
        #     if len(current_text) > max_length:
        #         max_length = len(current_text)
        #         text = current_text
        #         print(text)        
        text = extract_text_llm(image_path= input_image)
        print(f"text: {text}")
        # Extract metadata from text
        extracted_metadata = extract_metadata(text)
        print(f"extracted_metadata: {extracted_metadata}")
        # Calculate similarity scores for all reference bottles
        scores = []
        for bottle_id, features in self.reference_features.items():
            # Calculate visual similarity
            visual_sim = self._calculate_visual_similarity(descriptors, features['descriptors'])
            
            # Calculate text similarity
            text_sim = calculate_text_similarity(text, features['text'])
            
            # Calculate metadata similarity
            metadata_sim = self._calculate_metadata_similarity(extracted_metadata, features)
            
            # Calculate weighted total similarity
            total_sim = (VISUAL_SIMILARITY_WEIGHT * visual_sim) + \
                        (TEXT_SIMILARITY_WEIGHT * text_sim) + \
                        (METADATA_SIMILARITY_WEIGHT * metadata_sim)
            
            # Add to scores list
            scores.append({
                'id': bottle_id,
                'name': features['name'],
                'confidence': total_sim,
                'visual_score': visual_sim,
                'text_score': text_sim,
                'metadata_score': metadata_sim
            })
        
        # Sort by confidence score (descending)
        scores.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Return top N matches
        top_matches = scores[:top_n]
        
        return top_matches
    
    def get_reference_image(self, bottle_id: int) -> np.ndarray:
        """Get reference image for a specific bottle ID."""
        if bottle_id not in self.reference_images:
            raise ValueError(f"Bottle ID {bottle_id} not found in reference database")
        return self.reference_images[bottle_id]
    
    def save_results(self, results: List[Dict], output_path: str):
        """
        Save matching results to a JSON file.
        
        Args:
            results: List of matching results
            output_path: Path to save the JSON file
        """
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_path}")