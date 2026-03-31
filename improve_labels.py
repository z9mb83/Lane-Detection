"""
Improved pseudo-labeling using color-based lane detection
White and yellow lane markings detection for better training labels
"""
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def detect_lanes_color_based(image):
    """
    Detect lane markings using color filtering
    Optimized for white and yellow lane markings on gray asphalt
    """
    # Convert to HLS for better color separation
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    
    # White color mask
    lower_white = np.array([0, 200, 0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    
    # Yellow color mask
    lower_yellow = np.array([15, 30, 100])
    upper_yellow = np.array([35, 204, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    
    # Combine masks
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # Apply ROI mask (lower half of image)
    h, w = image.shape[:2]
    roi_mask = np.zeros_like(color_mask)
    
    # Trapezoid ROI for lane detection
    vertices = np.array([
        [(w//2 - 50, h//2 + 30),
         (w//2 + 50, h//2 + 30),
         (w, h - 30),
         (0, h - 30)]
    ], dtype=np.int32)
    
    cv2.fillPoly(roi_mask, vertices, 255)
    masked = cv2.bitwise_and(color_mask, roi_mask)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    masked = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel, iterations=1)
    masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return masked

def create_lane_mask(image):
    """Create binary lane mask from image"""
    # Detect lanes using color
    lane_mask = detect_lanes_color_based(image)
    
    # Apply edge detection to refine
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Combine color and edge information
    combined = cv2.bitwise_or(lane_mask, edges)
    
    # Dilate to make lanes thicker
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.dilate(combined, kernel, iterations=1)
    
    return combined

def regenerate_kaggle_labels():
    """Regenerate labels for Kaggle video frames using improved method"""
    kaggle_dir = Path("data/kaggle_frames")
    
    if not kaggle_dir.exists():
        print("Kaggle frames directory not found!")
        return
    
    frame_files = list(kaggle_dir.glob("kaggle_*.jpg"))
    print(f"Regenerating labels for {len(frame_files)} frames...")
    
    improved_count = 0
    
    for frame_path in tqdm(frame_files, desc="Generating labels"):
        # Load frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        
        # Generate improved mask
        mask = create_lane_mask(frame)
        
        # Save mask
        mask_path = kaggle_dir / (frame_path.stem + "_mask.png")
        cv2.imwrite(str(mask_path), mask)
        
        # Check if we detected any lanes
        if np.sum(mask > 0) > 100:  # At least some pixels detected
            improved_count += 1
    
    print(f"✓ Regenerated {len(frame_files)} labels")
    print(f"  {improved_count} frames have lane detections")

if __name__ == "__main__":
    regenerate_kaggle_labels()
    print("\nImproved labels generated. Ready for retraining.")
