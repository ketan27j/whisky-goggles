"""Visualization utilities for whisky bottle recognition."""

import cv2
import numpy as np
from typing import Union

from src.utils.image_processing import load_image

def visualize_match(query_image: Union[str, np.ndarray], ref_image: np.ndarray, match_name: str) -> np.ndarray:
    """
    Visualize the matching between query image and top match.
    
    Args:
        query_image: Path to image file or numpy array of the query image
        ref_image: Numpy array of the reference image
        match_name: Name of the matched bottle
        
    Returns:
        Visualization image
    """
    # Load query image if path is provided
    query_image = load_image(query_image)
            
    # Resize images to have the same height
    height = 400
    query_aspect = query_image.shape[1] / query_image.shape[0]
    ref_aspect = ref_image.shape[1] / ref_image.shape[0]
    
    query_resized = cv2.resize(query_image, (int(height * query_aspect), height))
    ref_resized = cv2.resize(ref_image, (int(height * ref_aspect), height))
    
    # Create a blank image to put both images side by side
    vis_image = np.zeros((height, query_resized.shape[1] + ref_resized.shape[1] + 10, 3), dtype=np.uint8)
    
    # Put images side by side
    vis_image[:, :query_resized.shape[1]] = query_resized
    vis_image[:, query_resized.shape[1]+10:] = ref_resized
    
    # Add labels
    cv2.putText(vis_image, "Query Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(vis_image, f"Match: {match_name}", 
               (query_resized.shape[1]+20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return vis_image

def visualize_results(results, query_image, recognizer, output_path="visualization.jpg"):
    """
    Visualize multiple top matches.
    
    Args:
        results: List of result dictionaries
        query_image: Query image path or array
        recognizer: WhiskyBottleRecognizer instance
        output_path: Path to save the visualization image
    """
    import matplotlib.pyplot as plt
    
    query_img = load_image(query_image)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, min(4, len(results) + 1), figsize=(15, 5))
    
    # Plot query image
    axes[0].imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Query Image")
    axes[0].axis('off')
    
    # Plot top matches
    for i, result in enumerate(results[:3], 1):
        bottle_id = result['id']
        ref_image = recognizer.get_reference_image(bottle_id)
        
        axes[i].imshow(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"{result['name']}\nScore: {result['confidence']:.2f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")