"""
Download sample road video and run lane detection inference
Uses our trained model on a real-world driving video
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import urllib.request
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')

print("=" * 70)
print("DOWNLOAD VIDEO & RUN LANE DETECTION")
print("=" * 70)

# Download a sample driving video
video_url = "https://github.com/udacity/CarND-LaneLines-P1/raw/master/test_videos/solidWhiteRight.mp4"
video_path = "kaggle_sample_video.mp4"

print(f"\nDownloading sample driving video...")
print(f"URL: {video_url}")

try:
    urllib.request.urlretrieve(video_url, video_path)
    print(f"✓ Downloaded: {video_path}")
except Exception as e:
    print(f"✗ Download failed: {e}")
    # Create a synthetic video as fallback
    print("Creating synthetic road video instead...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30, (960, 540))
    
    for i in range(90):  # 3 seconds at 30fps
        # Create road frame
        frame = np.ones((540, 960, 3), dtype=np.uint8) * 80
        
        # Add road
        cv2.rectangle(frame, (0, 270), (960, 540), (60, 60, 60), -1)
        
        # Add lanes with slight curve
        center_x = 480 + int(20 * np.sin(i * 0.1))
        left_x = center_x - 150
        right_x = center_x + 150
        
        for y in range(270, 540, 10):
            progress = (y - 270) / 270
            offset = int(30 * progress * np.sin(i * 0.05))
            cv2.line(frame, (left_x + offset - 5, y), (left_x + offset + 5, y), (255, 255, 200), 3)
            cv2.line(frame, (right_x + offset - 5, y), (right_x + offset + 5, y), (255, 255, 200), 3)
        
        out.write(frame)
    
    out.release()
    print(f"✓ Created synthetic video: {video_path}")

# Load model
print("\n" + "=" * 70)
print("LOADING TRAINED MODEL")
print("=" * 70)

import segmentation_models_pytorch as smp

class LaneDetectionModel(nn.Module):
    def __init__(self, encoder_name="resnet18", encoder_weights=None, 
                 video_mode=True, sequence_length=5):
        super().__init__()
        self.video_mode = video_mode
        self.sequence_length = sequence_length
        
        self.base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3 * sequence_length if video_mode else 3,
            classes=1,
            activation=None
        )
    
    def forward(self, x):
        if self.video_mode and x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B, T * C, H, W)
        return self.base_model(x)

# Use image mode for single video frame processing
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LaneDetectionModel(
    encoder_name="resnet18",
    encoder_weights=None,
    video_mode=False,  # Use image mode for simplicity
    sequence_length=1
).to(DEVICE)

# Load weights from trained model
if os.path.exists('models/best_video_model.pth'):
    checkpoint = torch.load('models/best_video_model.pth', map_location=DEVICE)
    # Adapt weights for image mode
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['model_state_dict']
    
    # Filter out temporal conv weights, use first 3 channels
    for key in list(pretrained_dict.keys()):
        if 'encoder.conv1.weight' in key:
            # Average temporal channels to get 3-channel input
            weight = pretrained_dict[key]
            if weight.shape[1] == 15:  # 5 frames * 3 channels
                new_weight = torch.zeros(64, 3, 7, 7)
                for i in range(5):
                    new_weight += weight[:, i*3:(i+1)*3, :, :]
                new_weight /= 5
                pretrained_dict[key] = new_weight
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"✓ Loaded trained model (IoU: {checkpoint.get('best_iou', 'N/A'):.4f})")
else:
    print("! No trained model found, using fresh model")

model.eval()

# Process video
print("\n" + "=" * 70)
print("PROCESSING VIDEO WITH LANE DETECTION")
print("=" * 70)

def preprocess_frame(frame):
    """Preprocess frame for model input"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (384, 256))
    frame_norm = frame_resized / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame_norm = (frame_norm - mean) / std
    return torch.from_numpy(frame_norm.transpose(2, 0, 1)).float()

def detect_and_draw_lanes(frame, model, device):
    """Detect lanes and draw on frame"""
    h, w = frame.shape[:2]
    
    # Preprocess
    input_tensor = preprocess_frame(frame).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        pred = model(input_tensor)
        pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
    
    # Resize mask to original size
    pred_mask = cv2.resize(pred_mask, (w, h))
    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    
    # Find lane contours
    contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Fit polynomials to lanes
    left_lane = None
    right_lane = None
    
    if len(contours) >= 1:
        all_points = []
        for cnt in contours:
            if len(cnt) >= 10:
                for point in cnt:
                    all_points.append(point[0])
        
        if len(all_points) >= 20:
            all_points = np.array(all_points)
            center_x = w // 2
            left_points = all_points[all_points[:, 0] < center_x]
            right_points = all_points[all_points[:, 0] >= center_x]
            
            if len(left_points) >= 10:
                left_lane = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
            if len(right_points) >= 10:
                right_lane = np.polyfit(right_points[:, 1], right_points[:, 0], 2)
    
    # Draw lanes
    result = frame.copy()
    y_vals = np.linspace(h//2, h-1, 50, dtype=int)
    
    if left_lane is not None:
        left_x = np.polyval(left_lane, y_vals).astype(int)
        left_points = np.column_stack((left_x, y_vals))
        left_points = left_points[(left_points[:, 0] >= 0) & (left_points[:, 0] < w)]
        if len(left_points) > 1:
            cv2.polylines(result, [left_points], False, (0, 255, 0), 6)
    
    if right_lane is not None:
        right_x = np.polyval(right_lane, y_vals).astype(int)
        right_points = np.column_stack((right_x, y_vals))
        right_points = right_points[(right_points[:, 0] >= 0) & (right_points[:, 0] < w)]
        if len(right_points) > 1:
            cv2.polylines(result, [right_points], False, (0, 0, 255), 6)
    
    # Fill lane area
    if left_lane is not None and right_lane is not None:
        left_x = np.polyval(left_lane, y_vals).astype(int)
        right_x = np.polyval(right_lane, y_vals).astype(int)
        pts_left = np.column_stack((left_x, y_vals))
        pts_right = np.column_stack((right_x, y_vals))[::-1]
        pts = np.vstack((pts_left, pts_right))
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w)]
        
        if len(pts) >= 3:
            overlay = result.copy()
            cv2.fillPoly(overlay, [pts], (0, 200, 0))
            result = cv2.addWeighted(result, 0.8, overlay, 0.2, 0)
    
    return result, pred_mask

# Process video
output_path = "kaggle_output_video.mp4"
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input video: {width}x{height} @ {fps}fps, {total_frames} frames")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
print(f"\nProcessing...")

with tqdm(total=total_frames, desc="Frame") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result, mask = detect_and_draw_lanes(frame, model, DEVICE)
        out.write(result)
        
        frame_count += 1
        pbar.update(1)

cap.release()
out.release()

print(f"\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print(f"\nInput:  {video_path}")
print(f"Output: {output_path}")
print(f"Frames processed: {frame_count}")

# Show file sizes
for f in [video_path, output_path]:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024
        print(f"  {f}: {size:.1f} KB")

print("\n" + "=" * 70)
