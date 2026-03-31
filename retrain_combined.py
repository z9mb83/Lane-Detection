"""
Retrain model with ALL data:
1. Original synthetic images (200)
2. Video clips dataset (1000 frames)
3. Kaggle video frames (221 frames)
With improved augmentation and longer training
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')

print("=" * 70)
print("RETRAINING WITH COMBINED DATASET")
print("=" * 70)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Step 1: Extract Kaggle video frames
print("\n" + "=" * 70)
print("STEP 1: Extracting Kaggle Video Frames")
print("=" * 70)

def extract_kaggle_frames(video_path, output_dir, skip_frames=2):
    """Extract frames and generate lane masks using Hough transform"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    extracted = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % skip_frames == 0:  # Skip every Nth frame
            # Save frame
            frame_path = Path(output_dir) / f"kaggle_{extracted:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            
            # Generate pseudo-label using Hough transform
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Mask for lane regions (lower half of image)
            mask = np.zeros_like(gray)
            h, w = frame.shape[:2]
            roi_vertices = np.array([
                [(0, h), (w//2 - 50, h//2 + 50), (w//2 + 50, h//2 + 50), (w, h)]
            ], dtype=np.int32)
            cv2.fillPoly(mask, roi_vertices, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # Hough transform
            lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30, 
                                     minLineLength=40, maxLineGap=100)
            
            # Create mask from detected lines
            lane_mask = np.zeros((h, w), dtype=np.uint8)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Only keep lines that could be lanes (roughly vertical)
                    if abs(x2 - x1) < abs(y2 - y1) * 2:
                        cv2.line(lane_mask, (x1, y1), (x2, y2), 255, 5)
            
            # Save mask
            mask_path = Path(output_dir) / f"kaggle_{extracted:04d}_mask.png"
            cv2.imwrite(str(mask_path), lane_mask)
            
            extracted += 1
        
        frame_count += 1
    
    cap.release()
    return extracted

kaggle_frames_dir = "data/kaggle_frames"
num_kaggle = extract_kaggle_frames("kaggle_sample_video.mp4", kaggle_frames_dir)
print(f"Extracted {num_kaggle} frames from Kaggle video")

# Step 2: Create combined dataset
print("\n" + "=" * 70)
print("STEP 2: Creating Combined Dataset")
print("=" * 70)

class CombinedLaneDataset(Dataset):
    """Combined dataset from multiple sources"""
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []
        
        # 1. Original synthetic images
        img_dir = Path("data/train/images")
        mask_dir = Path("data/train/masks")
        if img_dir.exists():
            for img_path in img_dir.glob("*.jpg"):
                mask_path = mask_dir / (img_path.stem + ".png")
                if mask_path.exists():
                    self.samples.append(('image', str(img_path), str(mask_path)))
        
        # 2. Video clips (last frame from each clip)
        clips_dir = Path("data/tusimple_video/clips")
        if clips_dir.exists():
            for clip_dir in clips_dir.glob("clip_*"):
                frames = sorted(clip_dir.glob("*.jpg"), key=lambda x: int(x.stem))
                if frames:
                    # Use last frame with pseudo-label from Hough
                    last_frame = frames[-1]
                    frame = cv2.imread(str(last_frame))
                    mask = self._create_hough_mask(frame)
                    mask_path = clip_dir / "mask.png"
                    cv2.imwrite(str(mask_path), mask)
                    self.samples.append(('video', str(last_frame), str(mask_path)))
        
        # 3. Kaggle video frames
        kaggle_dir = Path(kaggle_frames_dir)
        if kaggle_dir.exists():
            for img_path in kaggle_dir.glob("kaggle_*.jpg"):
                mask_path = kaggle_dir / (img_path.stem + "_mask.png")
                if mask_path.exists():
                    self.samples.append(('kaggle', str(img_path), str(mask_path)))
        
        print(f"Total samples: {len(self.samples)}")
        print(f"  - Original images: {sum(1 for s in self.samples if s[0] == 'image')}")
        print(f"  - Video clips: {sum(1 for s in self.samples if s[0] == 'video')}")
        print(f"  - Kaggle frames: {sum(1 for s in self.samples if s[0] == 'kaggle')}")
    
    def _create_hough_mask(self, frame):
        """Create lane mask using Hough transform"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        h, w = frame.shape[:2]
        
        # ROI mask
        mask = np.zeros_like(gray)
        roi_vertices = np.array([
            [(0, h), (w//2 - 50, h//2 + 50), (w//2 + 50, h//2 + 50), (w, h)]
        ], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Hough lines
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30,
                               minLineLength=40, maxLineGap=100)
        
        lane_mask = np.zeros((h, w), dtype=np.uint8)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < abs(y2 - y1) * 2:
                    cv2.line(lane_mask, (x1, y1), (x2, y2), 255, 5)
        
        return lane_mask
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        _, img_path, mask_path = self.samples[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        mask = mask.unsqueeze(0)
        return image, mask

# Enhanced transforms with more augmentation
train_transform = A.Compose([
    A.Resize(256, 384),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.3, var_limit=(10.0, 50.0)),
    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.4),
    A.OneOf([
        A.MotionBlur(blur_limit=3, p=1.0),
        A.MedianBlur(blur_limit=3, p=1.0),
    ], p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 384),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Create dataset and split
dataset = CombinedLaneDataset(transform=train_transform)

# Split train/val
from sklearn.model_selection import train_test_split
train_indices, val_indices = train_test_split(
    range(len(dataset)), test_size=0.2, random_state=42
)

from torch.utils.data import Subset
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

BATCH_SIZE = 8
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}")

# Step 3: Train with better model
print("\n" + "=" * 70)
print("STEP 3: Training Improved Model")
print("=" * 70)

class LaneDetectionModel(nn.Module):
    def __init__(self, encoder_name="resnet34"):  # Upgraded to ResNet34
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )
    
    def forward(self, x):
        return self.model(x)

model = LaneDetectionModel(encoder_name="resnet34").to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Improved loss function
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

class FocalDiceLoss(nn.Module):
    """Combined focal and dice loss for better handling of imbalanced data"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        # Focal loss component
        bce_loss = self.bce(pred, target)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # Dice loss component
        dice_loss = self.dice(pred, target)
        
        return focal_loss + dice_loss

criterion = FocalDiceLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-3)  # Higher LR
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

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
    
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
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
        for images, masks in tqdm(loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            total_iou += calculate_iou(outputs, masks)
    
    return total_loss / len(loader), total_iou / len(loader)

# Train longer
EPOCHS = 20
history = {'train_loss': [], 'train_iou': [], 'val_loss': [], 'val_iou': []}
best_iou = 0

print(f"\nTraining for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 50)
    
    train_loss, train_iou = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss, val_iou = validate_epoch(model, val_loader, criterion, DEVICE)
    
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_iou)
    
    history['train_loss'].append(train_loss)
    history['train_iou'].append(train_iou)
    history['val_loss'].append(val_loss)
    history['val_iou'].append(val_iou)
    
    print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f} (LR: {current_lr:.6f})")
    
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_iou': best_iou,
        }, 'models/best_combined_model.pth')
        print(f"✓ Saved best model (IoU: {best_iou:.4f})")

print(f"\nTraining complete! Best Val IoU: {best_iou:.4f}")

# Step 4: Test on Kaggle video
print("\n" + "=" * 70)
print("STEP 4: Testing on Kaggle Video")
print("=" * 70)

# Load best model
checkpoint = torch.load('models/best_combined_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def process_frame(frame, model, device):
    """Process single frame"""
    h, w = frame.shape[:2]
    
    # Preprocess
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (384, 256))
    frame_norm = frame_resized / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame_norm = (frame_norm - mean) / std
    input_tensor = torch.from_numpy(frame_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        pred = model(input_tensor)
        pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
    
    # Resize and create overlay
    pred_mask = cv2.resize(pred_mask, (w, h))
    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    
    # Find lanes
    contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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

# Process Kaggle video
output_path = "kaggle_output_improved.mp4"
cap = cv2.VideoCapture("kaggle_sample_video.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
ious = []

print(f"Processing {total_frames} frames...")
with tqdm(total=total_frames, desc="Processing") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result, mask = process_frame(frame, model, DEVICE)
        out.write(result)
        
        # Calculate IoU against pseudo-ground-truth
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        h, w = frame.shape[:2]
        roi_mask = np.zeros_like(gray)
        roi_vertices = np.array([
            [(0, h), (w//2 - 50, h//2 + 50), (w//2 + 50, h//2 + 50), (w, h)]
        ], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, roi_mask)
        
        gt_binary = (masked_edges > 0).astype(np.float32)
        pred_binary = (cv2.resize(mask, (w, h)) > 0.5).astype(np.float32)
        
        intersection = (gt_binary * pred_binary).sum()
        union = gt_binary.sum() + pred_binary.sum() - intersection
        if union > 0:
            ious.append(intersection / union)
        
        frame_count += 1
        pbar.update(1)

cap.release()
out.release()

avg_iou = np.mean(ious) if ious else 0
print(f"\nAverage IoU on Kaggle video: {avg_iou:.4f}")

# Step 5: Summary
print("\n" + "=" * 70)
print("RETRAINING COMPLETE - SUMMARY")
print("=" * 70)
print(f"\nDataset:")
print(f"  Total samples: {len(dataset)}")
print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
print(f"\nTraining:")
print(f"  Epochs: {EPOCHS}")
print(f"  Best Val IoU: {best_iou:.4f}")
print(f"  Kaggle Video IoU: {avg_iou:.4f}")
print(f"\nOutput Files:")
for f in ['models/best_combined_model.pth', 'kaggle_output_improved.mp4']:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024
        print(f"  ✓ {f} ({size:.1f} KB)")
print("=" * 70)
