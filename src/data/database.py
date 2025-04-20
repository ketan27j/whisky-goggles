"""Database handling for the whisky recognition system."""

import os
import pandas as pd
import requests
from pathlib import Path
from typing import Dict, Any

class WhiskyDatabase:
    """Class to handle whisky bottle database operations."""

    def __init__(self, database_path: str, images_dir: str):
        """
        Initialize the whisky database.
        
        Args:
            database_path: Path to the CSV file containing whisky bottle database
            images_dir: Directory to store downloaded bottle images
        """
        self.database_path = database_path
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(exist_ok=True, parents=True)
        
        self._load_database()
    
    def _load_database(self):
        """Load the whisky database from CSV file."""
        self.database = pd.read_csv(self.database_path)
        print(f"Loaded database with {len(self.database)} entries")
    
    def get_all_bottles(self):
        """Return all bottles in the database."""
        return self.database
    
    def get_bottle_by_id(self, bottle_id: int) -> Dict[str, Any]:
        """
        Get bottle information by ID.
        
        Args:
            bottle_id: The ID of the bottle to retrieve
            
        Returns:
            Dictionary with bottle information
        """
        bottle = self.database[self.database['id'] == bottle_id]
        if len(bottle) == 0:
            return None
        return bottle.iloc[0].to_dict()
    
    def download_bottle_image(self, bottle_id: int, image_url: str) -> str:
        """
        Download bottle image if it doesn't exist.
        
        Args:
            bottle_id: The ID of the bottle
            image_url: URL of the bottle image
            
        Returns:
            Path to the downloaded image
        """
        image_path = self.images_dir / f"{bottle_id}.jpg"
        
        # Download image if it doesn't exist
        if not image_path.exists():
            try:
                response = requests.get(image_url)
                if response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded image for bottle ID {bottle_id}")
                else:
                    print(f"Failed to download image for bottle ID {bottle_id}")
                    return None
            except Exception as e:
                print(f"Error downloading image for bottle ID {bottle_id}: {e}")
                return None
        
        return str(image_path)