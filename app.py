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
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS to improve appearance and fix image sizes
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stImage img {
            width: 300px !important;
            height: 300px !important;
            object-fit: contain;
        }
        .result-card {
            border: 1px solid #ddd;
            background-color: #f8f9fa;
            height: 100%;
        }
        .stMetric {
            background-color: #ffffff;
            border-radius: 5px;
            padding: 2px 5px;
            margin: 2px 0;
        }
        .stButton button {
            width: 100%;
            background-color: red; /* Bootstrap primary color */
            color: white; /* Text color */
            border: none; /* Remove border */
            border-radius: 5px; /* Rounded corners */
            padding: 10px; /* Padding for better touch target */
            font-size: 16px; /* Increase font size */
            cursor: pointer; /* Pointer cursor on hover */
            transition: background-color 0.3s; /* Smooth transition */
        }
        .stButton button:hover {
            background-color: #f5a99a; /* Darker shade on hover */
            color: white;
        }
        .stHeader {
            background-color: #1e1e1e;
            padding: 1rem;
            border-radius: 5px;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with logo and title
    col1,col2 = st.columns([1,2])
    with col1:
        st.title("🥃 Whisky Goggles")
        st.markdown("<p style='font-size: 1.2rem;'>Upload an image of a whisky bottle to identify it</p>", unsafe_allow_html=True)
    
    # Initialize the recognizer
    @st.cache_resource
    def load_recognizer():
        with st.spinner("Loading whisky recognition model..."):
            return WhiskyBottleRecognizer(DATABASE_PATH, IMAGES_DIR)
    
    recognizer = load_recognizer()
    
    # Create a layout with two columns
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("Upload Image")
        
        # Image upload
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        # Camera input option
        use_camera = st.checkbox("Or use camera input")
        camera_image = None
        
        if use_camera:
            camera_image = st.camera_input("Take a picture", key="camera")
        
        # Process button
        identify_button = st.button("🔍 Identify Whisky", use_container_width=True)
        
        # About section
        st.markdown("### About")
        st.info(
            "Whisky Goggles helps you identify whisky bottles from images. "
            "Upload a photo or take a picture to get started."
        )
        
        st.markdown("### How it works")
        st.markdown(
            "1. Upload or take a photo of a whisky bottle\n"
            "2. Click 'Identify Whisky'\n"
            "3. View the top matches and details"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with right_col:
        # Display the uploaded image and results
        if uploaded_file is not None or camera_image is not None:
            input_image = camera_image if uploaded_file is None else uploaded_file
            
            # Convert to OpenCV format
            image_bytes = input_image.getvalue()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Display the uploaded image
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("Your Bottle")
            st.image(input_image, width=300)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Process image when button is clicked
            if identify_button:
                with st.spinner("Analyzing image..."):
                    # Get top 5 matches
                    results = recognizer.identify_bottle(image, top_n=5, input_image=input_image)
                    
                    if results:
                        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                        st.success(f"Top match: {results[0]['name']} (Confidence: {results[0]['confidence']:.2f})")
                        
                        # Display results in a grid
                        st.subheader("Top Matches")
                        
                        # Create a grid for results
                        result_cols = st.columns(min(3, len(results)))
                        
                        for i, (col, result) in enumerate(zip(result_cols, results)):
                            bottle_id = result['id']
                            ref_image = recognizer.get_reference_image(bottle_id)
                            
                            # Convert OpenCV BGR to RGB for display
                            ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
                            
                            with col:
                                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                                st.image(ref_image_rgb, width=200, caption=f"{result['name']}")
                                st.metric("Confidence", f"{result['confidence']:.2f}")
                                
                                with st.expander("Details"):
                                    st.metric("Visual Score", f"{result['visual_score']:.2f}")
                                    st.metric("Text Score", f"{result['text_score']:.2f}")
                                    st.metric("Metadata Score", f"{result['metadata_score']:.2f}")
                                    
                                    # Display additional info if available
                                    if 'details' in result:
                                        st.write(result['details'])
                                st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
