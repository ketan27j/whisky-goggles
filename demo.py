"""Demo application for whisky bottle recognition."""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path

from src.models.recognizer import WhiskyBottleRecognizer
from src.visualization.visualizer import visualize_match, visualize_results
from config import DATABASE_PATH, IMAGES_DIR

def run_demo(image_path):
    """
    Run the recognition demo with the specified image.
    
    Args:
        image_path: Path to the test image
    """
    print("Initializing the whisky bottle recognizer...")
    recognizer = WhiskyBottleRecognizer(DATABASE_PATH, IMAGES_DIR)
    
    if os.path.exists(image_path):
        print(f"Identifying bottle in {image_path}...")
        
        # Identify the bottle
        results = recognizer.identify_bottle(image_path)
        
        # Print results
        print("\nTop matches:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['name']} (Confidence: {result['confidence']:.2f})")
            print(f"   Visual score: {result['visual_score']:.2f}")
            print(f"   Text score: {result['text_score']:.2f}")
            print(f"   Metadata score: {result['metadata_score']:.2f}")
        
        # Visualize the top match
        if results:
            top_match_id = results[0]['id']
            top_match_name = results[0]['name']
            ref_image = recognizer.get_reference_image(top_match_id)
            
            vis_image = visualize_match(image_path, ref_image, top_match_name)
            cv2.imwrite('match_visualization.jpg', vis_image)
            print("\nVisualization saved to match_visualization.jpg")
            
            # Create visualization with multiple matches
            visualize_results(results, image_path, recognizer)
        
        # Save results to JSON
        output_path = 'recognition_results.json'
        recognizer.save_results(results, output_path)
        print(f"Results saved to {output_path}")
    else:
        print(f"Error: Test image not found at {image_path}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Whisky Bottle Recognition Demo')
    parser.add_argument('--image', type=str, default='test_image.jpg', 
                        help='Path to test image')
    args = parser.parse_args()
    
    # Run demo
    run_demo(args.image)