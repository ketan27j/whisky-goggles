"""Configuration settings for the whisky recognition system."""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data paths
DATABASE_PATH = os.path.join(BASE_DIR, "data", "501_Bottle_Dataset.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "data", "whisky_images")

# Image processing parameters
MAX_IMAGE_DIMENSION = 800
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
GAUSSIAN_KERNEL_SIZE = (5, 5)

# Feature extraction parameters
ORB_FEATURES = 1000
MAX_MATCHES = 50
MAX_DISTANCE = 100  # Maximum possible distance for similarity calculation

# API settings
API_HOST = "0.0.0.0"
API_PORT = 5000

# Similarity weights
VISUAL_SIMILARITY_WEIGHT = 0.5
TEXT_SIMILARITY_WEIGHT = 0.4
METADATA_SIMILARITY_WEIGHT = 0.1