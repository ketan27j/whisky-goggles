"""Streamlit frontend for the whisky recognition system."""

import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
import io

from src.models.recognizer import WhiskyBottleRecognizer
from src.visualization.visualizer import visualize_match, visualize_results
from config import DATABASE_PATH, IMAGES_DIR

def main():
    st.set_page_config(
        page_title="Whisky Goggles",
        page_icon="🥃",
        layout="wide"
    )
    
    st.title("🥃 Whisky Goggles")
    st.subheader("Upload an image of a whisky bottle to identify it")
    
    # Initialize the recognizer
    @st.cache_resource
    def load_recognizer():
        with st.spinner("Loading whisky recognition model..."):
            return WhiskyBottleRecognizer(DATABASE_PATH, IMAGES_DIR)
    
    recognizer = load_recognizer()
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Camera input option
    use_camera = st.checkbox("Or use camera input")
    camera_image = None
    
    if use_camera:
        camera_image = st.camera_input("Take a picture")
    
    # Process the image
    if uploaded_file is not None or camera_image is not None:
        input_image = camera_image if uploaded_file is None else uploaded_file
        
        # Convert to OpenCV format
        image_bytes = input_image.getvalue()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Display the uploaded image
        st.image(input_image, caption="Uploaded Image", use_column_width=True)
        
        # Add a button to trigger recognition
        if st.button("Identify Whisky"):
            with st.spinner("Analyzing image..."):
                # Get top 5 matches
                results = recognizer.identify_bottle(image, top_n=5)
                
                if results:
                    st.success(f"Top match: {results[0]['name']} (Confidence: {results[0]['confidence']:.2f})")
                    
                    # Display results in columns
                    st.subheader("Top Matches")
                    cols = st.columns(min(3, len(results)))
                    
                    for i, (col, result) in enumerate(zip(cols, results)):
                        bottle_id = result['id']
                        ref_image = recognizer.get_reference_image(bottle_id)
                        
                        # Convert OpenCV BGR to RGB for display
                        ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
                        
                        col.image(ref_image_rgb, caption=f"{result['name']}")
                        col.metric("Confidence", f"{result['confidence']:.2f}")
                        col.metric("visual_score", f"{result['visual_score']:.2f}")
                        col.metric("text_score", f"{result['text_score']:.2f}")
                        col.metric("metadata_score", f"{result['metadata_score']:.2f}")
                        
                        # Display additional info if available
                        if 'details' in result:
                            with col.expander("Details"):
                                st.write(result['details'])
                    
                    # Create visualization of the top match
                    if len(results) > 0:
                        st.subheader("Comparison with Top Match")
                        top_match = results[0]
                        ref_image = recognizer.get_reference_image(top_match['id'])
                        
                        vis_image = visualize_match(image, ref_image, top_match['name'])
                        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                        
                        st.image(vis_image_rgb, use_column_width=True)
                else:
                    st.error("No matches found. Try with a different image.")
    
    # Add information about the app
    with st.sidebar:
        st.header("About")
        st.info(
            "Whisky Goggles helps you identify whisky bottles from images. "
            "Upload a photo or take a picture to get started."
        )
        
        st.header("How it works")
        st.write(
            "1. Upload or take a photo of a whisky bottle\n"
            "2. Click 'Identify Whisky'\n"
            "3. View the top matches and details"
        )

if __name__ == "__main__":
    main()
