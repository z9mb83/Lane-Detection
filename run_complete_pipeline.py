"""
Complete Video Lane Detection Pipeline - Run Everything
End-to-end execution: dataset generation -> training -> inference
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, 'src')

print("=" * 70)
print("COMPLETE VIDEO LANE DETECTION PIPELINE")
print("=" * 70)

# Configuration
VIDEO_MODE = True
SEQUENCE_LENGTH = 5
EPOCHS = 10
BATCH_SIZE = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\nConfiguration:")
print(f"  Mode: {'VIDEO' if VIDEO_MODE else 'IMAGE'}")
print(f"  Sequence Length: {SEQUENCE_LENGTH}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Device: {DEVICE}")

# ============================================================================
# STEP 1: Generate Video Dataset
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: Generating Video Dataset")
print("=" * 70)

from video_dataset_generator import VideoLaneDatasetGenerator

generator = VideoLaneDatasetGenerator(
    data_dir="data/tusimple_video",
    num_clips=50,
    frames_per_clip=20
)
generator.generate_dataset()

# ============================================================================
# STEP 2: Setup Dataset and Model
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Loading Dataset and Building Model")
print("=" * 70)

from video_dataset import VideoLaneDataset, get_video_train_transforms, get_video_val_transforms

train_dataset = VideoLaneDataset(
    "data/tusimple_video",
    sequence_length=SEQUENCE_LENGTH,
    transform=get_video_train_transforms(),
    split='train'
)

val_dataset = VideoLaneDataset(
    "data/tusimple_video",
    sequence_length=SEQUENCE_LENGTH,
    transform=get_video_val_transforms(),
    split='val'
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# Model
class LaneDetectionModel(nn.Module):
    def __init__(self, encoder_name="resnet18", encoder_weights="imagenet", 
                 video_mode=False, sequence_length=5):
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

model = LaneDetectionModel(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    video_mode=VIDEO_MODE,
    sequence_length=SEQUENCE_LENGTH
).to(DEVICE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# STEP 3: Training
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Training Model")
print("=" * 70)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(pred, target)

def calculate_iou(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_iou = 0, 0
    for data, masks in tqdm(loader, desc="Training"):
        data, masks = data.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_iou += calculate_iou(outputs, masks)
    return total_loss / len(loader), total_iou / len(loader)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou = 0, 0
    with torch.no_grad():
        for data, masks in tqdm(loader, desc="Validation"):
            data, masks = data.to(device), masks.to(device)
            outputs = model(data)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            total_iou += calculate_iou(outputs, masks)
    return total_loss / len(loader), total_iou / len(loader)

criterion = CombinedLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

history = {'train_loss': [], 'train_iou': [], 'val_loss': [], 'val_iou': []}
best_iou = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 50)
    
    train_loss, train_iou = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss, val_iou = validate_epoch(model, val_loader, criterion, DEVICE)
    scheduler.step(val_iou)
    
    history['train_loss'].append(train_loss)
    history['train_iou'].append(train_iou)
    history['val_loss'].append(val_loss)
    history['val_iou'].append(val_iou)
    
    print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}")
    
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_iou': best_iou,
            'video_mode': VIDEO_MODE,
            'sequence_length': SEQUENCE_LENGTH
        }, 'models/best_video_model.pth')
        print(f"✓ Saved best model (IoU: {best_iou:.4f})")

print(f"\nTraining complete! Best Val IoU: {best_iou:.4f}")

# ============================================================================
# STEP 4: Generate Training Curves
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Generating Training Curves")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs = range(1, len(history['train_loss']) + 1)

axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, history['train_iou'], 'b-', label='Train IoU')
axes[1].plot(epochs, history['val_iou'], 'r-', label='Val IoU')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('IoU')
axes[1].set_title('Training & Validation IoU')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('video_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: video_training_curves.png")

# ============================================================================
# STEP 5: Visualize Predictions
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Generating Prediction Visualizations")
print("=" * 70)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
model.eval()

for i in range(min(4, len(val_dataset))):
    frames, mask = val_dataset[i]
    with torch.no_grad():
        pred = model(frames.unsqueeze(0).to(DEVICE))
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()
    
    # Show middle frame
    mid_frame = frames[SEQUENCE_LENGTH // 2]
    img_np = mid_frame.permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    
    mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
    mask_np = mask_np.squeeze()  # Remove channel dimension for visualization
    
    axes[0, i].imshow(img_np)
    axes[0, i].set_title(f"Frame {i+1}")
    axes[0, i].axis('off')
    
    axes[1, i].imshow(mask_np, cmap='gray')
    axes[1, i].set_title(f"Ground Truth {i+1}")
    axes[1, i].axis('off')
    
    axes[2, i].imshow(pred, cmap='gray')
    axes[2, i].set_title(f"Prediction {i+1}")
    axes[2, i].axis('off')

plt.tight_layout()
plt.savefig('video_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: video_predictions.png")

# ============================================================================
# STEP 6: Video Inference Demo
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Running Video Inference Demo")
print("=" * 70)

class VideoLaneDetector:
    def __init__(self, model, device, sequence_length=5):
        self.model = model
        self.device = device
        self.sequence_length = sequence_length
        self.frame_buffer = []
    
    def preprocess_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (384, 256))
        frame_norm = frame_resized / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame_norm = (frame_norm - mean) / std
        return torch.from_numpy(frame_norm.transpose(2, 0, 1)).float()
    
    def get_sequence(self, current_frame):
        self.frame_buffer.append(current_frame)
        if len(self.frame_buffer) > self.sequence_length:
            self.frame_buffer.pop(0)
        while len(self.frame_buffer) < self.sequence_length:
            self.frame_buffer.insert(0, self.frame_buffer[0])
        return torch.stack(self.frame_buffer)
    
    def process_frame(self, frame):
        h, w = frame.shape[:2]
        frame_tensor = self.preprocess_frame(frame)
        sequence = self.get_sequence(frame_tensor)
        input_tensor = sequence.unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            pred = self.model(input_tensor)
            pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
        
        # Resize and overlay
        pred_mask = cv2.resize(pred_mask, (w, h))
        pred_binary = (pred_mask > 0.5).astype(np.uint8)
        
        overlay = frame.copy()
        overlay[pred_binary > 0] = [0, 255, 0]
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        return result, pred_mask

# Load best model
checkpoint = torch.load('models/best_video_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded model from epoch {checkpoint['epoch']+1} (IoU: {checkpoint['best_iou']:.4f})")

# Create sample input video
sample_clip = list(Path("data/tusimple_video/clips").glob("clip_*"))[0]
frames = sorted([f for f in sample_clip.glob("*.jpg")], key=lambda x: int(x.stem))

input_video = "demo_input.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(input_video, fourcc, 10, (1280, 720))
for frame_path in frames:
    frame = cv2.imread(str(frame_path))
    out.write(frame)
out.release()
print(f"Created input video: {input_video}")

# Process video
detector = VideoLaneDetector(model, DEVICE, sequence_length=SEQUENCE_LENGTH)

output_video = "video_output.mp4"
cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Processing {total_frames} frames...")
with tqdm(total=total_frames, desc="Video Inference") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result, _ = detector.process_frame(frame)
        out.write(result)
        frame_count += 1
        pbar.update(1)

cap.release()
out.release()
print(f"Saved output video: {output_video}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("VIDEO LANE DETECTION PIPELINE COMPLETE")
print("=" * 70)
print(f"\nResults:")
print(f"  Best Validation IoU: {best_iou:.4f}")
print(f"  Total Epochs: {EPOCHS}")
print(f"  Device: {DEVICE}")

print("\nGenerated Files:")
files = [
    'models/best_video_model.pth',
    'video_training_curves.png',
    'video_predictions.png',
    'demo_input.mp4',
    'video_output.mp4'
]
for f in files:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024
        print(f"  ✓ {f} ({size:.1f} KB)")

print("\nTo run on your own video:")
print("  python -c \"from src.video_inference import process_video; process_video('input.mp4', 'output.mp4')\"")
print("\nTo use webcam:")
print("  python -c \"from src.video_inference import webcam_inference; webcam_inference()\"")
print("=" * 70)
