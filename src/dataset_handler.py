"""
Lane Detection Dataset Handler
Attempts to download real data, falls back to synthetic data generation
"""
import os
import numpy as np
import cv2
from pathlib import Path
import urllib.request
import json

class LaneDatasetDownloader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.mask_dir = self.data_dir / "masks"
        
    def setup_directories(self):
        """Create necessary directories"""
        for dir_path in [self.image_dir, self.mask_dir, 
                        self.data_dir / "train" / "images", self.data_dir / "train" / "masks",
                        self.data_dir / "val" / "images", self.data_dir / "val" / "masks"]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_tusimple_sample(self):
        """Try to download TuSimple sample data from public mirror"""
        urls = [
            ("https://s3.us-east-2.amazonaws.com/tusimple-benchmark/dataset/tusimple_benchmark.zip", "tusimple.zip"),
        ]
        
        for url, filename in urls:
            try:
                print(f"Attempting to download from {url}...")
                filepath = self.data_dir / filename
                urllib.request.urlretrieve(url, filepath)
                print(f"Downloaded {filename}")
                return True
            except Exception as e:
                print(f"Failed to download from {url}: {e}")
                continue
        return False
    
    def generate_synthetic_lanes(self, num_samples=100, image_size=(640, 480)):
        """Generate synthetic lane detection data for demonstration"""
        print(f"Generating {num_samples} synthetic lane images...")
        
        np.random.seed(42)
        
        for i in range(num_samples):
            # Create road background
            img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 80
            
            # Add road surface variation
            noise = np.random.randint(-20, 20, (image_size[1], image_size[0], 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 40, 120).astype(np.uint8)
            
            # Create lane mask
            mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
            
            # Define lane parameters
            center_x = image_size[0] // 2 + np.random.randint(-30, 30)
            lane_width_base = 60 + np.random.randint(-10, 10)
            perspective_factor = 0.6
            
            # Generate left and right lanes
            for y in range(image_size[1]):
                # Perspective effect - lanes converge at horizon
                progress = y / image_size[1]
                lane_spacing = lane_width_base * (0.3 + 0.7 * progress)
                
                # Add curve variation
                curve_offset = int(30 * np.sin(progress * np.pi * 2 + i * 0.1) * progress)
                
                left_x = int(center_x - lane_spacing + curve_offset)
                right_x = int(center_x + lane_spacing + curve_offset)
                
                # Draw lanes on mask
                lane_thickness = max(2, int(5 * progress))
                
                if 0 <= left_x < image_size[0]:
                    cv2.line(mask, (left_x - lane_thickness//2, y), 
                            (left_x + lane_thickness//2, y), 255, lane_thickness)
                if 0 <= right_x < image_size[0]:
                    cv2.line(mask, (right_x - lane_thickness//2, y), 
                            (right_x + lane_thickness//2, y), 255, lane_thickness)
                
                # Draw lane markings on image
                if 0 <= left_x < image_size[0]:
                    color_var = np.random.randint(-15, 15)
                    cv2.line(img, (left_x - lane_thickness//2, y), 
                            (left_x + lane_thickness//2, y), 
                            (220 + color_var, 220 + color_var, 200 + color_var), lane_thickness)
                if 0 <= right_x < image_size[0]:
                    color_var = np.random.randint(-15, 15)
                    cv2.line(img, (right_x - lane_thickness//2, y), 
                            (right_x + lane_thickness//2, y), 
                            (220 + color_var, 220 + color_var, 200 + color_var), lane_thickness)
            
            # Add some noise and texture
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
            # Save image and mask
            split = "train" if i < int(num_samples * 0.8) else "val"
            cv2.imwrite(str(self.data_dir / split / "images" / f"lane_{i:04d}.jpg"), img)
            cv2.imwrite(str(self.data_dir / split / "masks" / f"lane_{i:04d}.png"), mask)
        
        print(f"Generated {num_samples} synthetic samples")
        print(f"  - Train: {int(num_samples * 0.8)}")
        print(f"  - Val: {num_samples - int(num_samples * 0.8)}")
        
        return True
    
    def prepare_dataset(self, num_synthetic=200):
        """Prepare dataset - try download first, then generate synthetic"""
        self.setup_directories()
        
        # Check if data already exists
        train_images = list((self.data_dir / "train" / "images").glob("*"))
        if len(train_images) > 0:
            print(f"Dataset already exists with {len(train_images)} training images")
            return True
        
        # Try to download real data
        if self.download_tusimple_sample():
            print("Downloaded real dataset")
            return True
        
        # Fall back to synthetic data
        print("Using synthetic lane data for demonstration")
        return self.generate_synthetic_lanes(num_synthetic)

if __name__ == "__main__":
    downloader = LaneDatasetDownloader()
    downloader.prepare_dataset()
