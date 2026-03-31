"""
Video Lane Dataset Handler
Generates synthetic video clips with lane labels (TuSimple-like format)
"""
import os
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm


class VideoLaneDatasetGenerator:
    """Generate synthetic video lane detection dataset similar to TuSimple format"""
    
    def __init__(self, data_dir="data/tusimple_video", num_clips=50, frames_per_clip=20):
        self.data_dir = Path(data_dir)
        self.clips_dir = self.data_dir / "clips"
        self.label_file = self.data_dir / "label_data.json"
        self.num_clips = num_clips
        self.frames_per_clip = frames_per_clip
        self.image_size = (1280, 720)  # TuSimple-like resolution
        
    def setup_directories(self):
        """Create necessary directories"""
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {self.clips_dir}")
        
    def generate_lane_parameters(self, seed):
        """Generate lane parameters for a clip"""
        np.random.seed(seed)
        
        # Base parameters
        center_x = self.image_size[0] // 2 + np.random.randint(-100, 100)
        lane_width_base = 300 + np.random.randint(-50, 50)
        
        # Add curve variation
        curve_amplitude = np.random.uniform(20, 80)
        curve_frequency = np.random.uniform(0.5, 2.0)
        
        return {
            'center_x': center_x,
            'lane_width_base': lane_width_base,
            'curve_amplitude': curve_amplitude,
            'curve_frequency': curve_frequency,
            'speed': np.random.uniform(5, 15)  # Forward movement simulation
        }
    
    def generate_frame(self, frame_idx, params, total_frames):
        """Generate a single frame with lanes"""
        h, w = self.image_size[1], self.image_size[0]
        
        # Create road background (asphalt color)
        img = np.ones((h, w, 3), dtype=np.uint8) * 60
        
        # Add road texture
        noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 30, 90).astype(np.uint8)
        
        # Simulate forward motion by shifting curve phase
        motion_phase = frame_idx / total_frames * params['curve_frequency'] * 2 * np.pi
        
        # Lane parameters with perspective
        center_x = params['center_x']
        lane_width = params['lane_width_base']
        
        # Store lane points for labels
        lane_points_left = []
        lane_points_right = []
        
        # Draw lanes
        for y in range(h // 2, h, 5):  # Start from horizon
            progress = (y - h // 2) / (h // 2)
            
            # Perspective: lanes get closer together at horizon
            lane_spacing = lane_width * (0.2 + 0.8 * progress)
            
            # Curve with motion
            curve = params['curve_amplitude'] * np.sin(progress * np.pi * params['curve_frequency'] + motion_phase)
            
            left_x = int(center_x - lane_spacing + curve)
            right_x = int(center_x + lane_spacing + curve)
            
            # Draw lane markings
            lane_thickness = max(3, int(8 * progress))
            
            # Left lane
            if 0 <= left_x < w:
                cv2.line(img, (left_x - lane_thickness//2, y), 
                        (left_x + lane_thickness//2, y), 
                        (200, 200, 180), lane_thickness)
                if y % 20 == 0:  # Sample points for labels
                    lane_points_left.append([left_x, y])
            
            # Right lane
            if 0 <= right_x < w:
                cv2.line(img, (right_x - lane_thickness//2, y), 
                        (right_x + lane_thickness//2, y), 
                        (200, 200, 180), lane_thickness)
                if y % 20 == 0:  # Sample points for labels
                    lane_points_right.append([right_x, y])
        
        # Add sky
        sky_color = np.array([135, 206, 235])  # Light blue
        img[:h//2, :] = sky_color
        
        # Add some blur for realism
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        return img, lane_points_left, lane_points_right
    
    def generate_clip(self, clip_idx):
        """Generate a video clip with multiple frames"""
        clip_name = f"clip_{clip_idx:04d}"
        clip_folder = self.clips_dir / clip_name
        clip_folder.mkdir(exist_ok=True)
        
        params = self.generate_lane_parameters(seed=clip_idx + 42)
        
        label_entry = {
            "raw_file": f"clips/{clip_name}/{self.frames_per_clip-1}.jpg",
            "lanes": [[], []],
            "h_samples": []
        }
        
        # Generate frames
        for frame_idx in range(self.frames_per_clip):
            frame, left_points, right_points = self.generate_frame(
                frame_idx, params, self.frames_per_clip
            )
            
            # Save frame
            frame_path = clip_folder / f"{frame_idx}.jpg"
            cv2.imwrite(str(frame_path), frame)
            
            # Store label for last frame (TuSimple format)
            if frame_idx == self.frames_per_clip - 1:
                # Convert points to lane format
                left_x = [p[0] for p in left_points]
                right_x = [p[0] for p in right_points]
                y_samples = [p[1] for p in left_points]
                
                label_entry["lanes"] = [left_x, right_x]
                label_entry["h_samples"] = y_samples
        
        return label_entry
    
    def generate_dataset(self):
        """Generate complete video dataset"""
        self.setup_directories()
        
        # Check if already exists
        if self.label_file.exists():
            print(f"Video dataset already exists at {self.data_dir}")
            with open(self.label_file, 'r') as f:
                labels = [json.loads(line) for line in f.readlines()]
            print(f"Found {len(labels)} clips")
            return True
        
        print(f"\nGenerating synthetic video dataset...")
        print(f"  - Clips: {self.num_clips}")
        print(f"  - Frames per clip: {self.frames_per_clip}")
        print(f"  - Resolution: {self.image_size}")
        
        labels = []
        for clip_idx in tqdm(range(self.num_clips), desc="Generating clips"):
            label_entry = self.generate_clip(clip_idx)
            labels.append(label_entry)
        
        # Save labels (TuSimple format - one JSON per line)
        with open(self.label_file, 'w') as f:
            for label in labels:
                f.write(json.dumps(label) + '\n')
        
        print(f"\nDataset generated successfully!")
        print(f"  Location: {self.data_dir}")
        print(f"  Clips: {self.num_clips}")
        print(f"  Total frames: {self.num_clips * self.frames_per_clip}")
        print(f"  Label file: {self.label_file}")
        
        return True


if __name__ == "__main__":
    generator = VideoLaneDatasetGenerator(num_clips=50, frames_per_clip=20)
    generator.generate_dataset()
