"""
Fine-tune on real Kaggle video with iterative self-improvement
Use the trained model to generate better pseudo-labels, then retrain
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')

print("=" * 70)
print("FINE-TUNING ON REAL-WORLD DATA WITH SELF-TRAINING")
print("=" * 70)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Generate high-quality pseudo-labels using ensemble of techniques
print("\n" + "=" * 70)
print("STEP 1: Generating High-Quality Pseudo-Labels")
print("=" * 70)

class LaneLabelGenerator:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def detect_lanes_multi(self, image):
        """Combine multiple detection methods"""
        h, w = image.shape[:2]
        
        # Method 1: Color-based (white/yellow)
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        
        # White lanes
        white_mask = cv2.inRange(hls, np.array([0, 180, 0]), np.array([255, 255, 255]))
        
        # Yellow lanes  
        yellow_mask = cv2.inRange(hls, np.array([15, 30, 100]), np.array([35, 204, 255]))
        
        color_lanes = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Method 2: Edge detection with morphological filtering
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 15, -2)
        
        # Canny edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Method 3: Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(255 * sobel / sobel.max())
        sobel = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)[1]
        
        # Combine all methods
        combined = np.zeros_like(gray)
        combined = cv2.bitwise_or(combined, color_lanes)
        combined = cv2.bitwise_or(combined, edges)
        combined = cv2.bitwise_or(combined, sobel)
        
        # Apply ROI
        roi_mask = np.zeros_like(combined)
        vertices = np.array([
            [(w//2 - 40, h//2 + 40),
             (w//2 + 40, h//2 + 40),
             (w, h - 20),
             (0, h - 20)]
        ], dtype=np.int32)
        cv2.fillPoly(roi_mask, vertices, 255)
        
        masked = cv2.bitwise_and(combined, roi_mask)
        
        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        masked = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel, iterations=1)
        masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return masked
    
    def fit_polynomial_lanes(self, image, binary_mask):
        """Fit smooth polynomial curves to detected lane points"""
        h, w = image.shape[:2]
        
        # Find all lane points
        y_coords, x_coords = np.where(binary_mask > 0)
        
        if len(y_coords) < 50:
            return binary_mask
        
        # Separate left and right lanes
        center_x = w // 2
        left_mask = x_coords < center_x
        right_mask = x_coords >= center_x
        
        result_mask = np.zeros_like(binary_mask)
        
        # Fit left lane
        if np.sum(left_mask) > 30:
            left_y = y_coords[left_mask]
            left_x = x_coords[left_mask]
            
            # Sort by y
            sort_idx = np.argsort(left_y)
            left_y = left_y[sort_idx]
            left_x = left_x[sort_idx]
            
            # Fit polynomial
            try:
                coeffs = np.polyfit(left_y, left_x, 2)
                y_samples = np.linspace(left_y.min(), left_y.max(), 100)
                x_samples = np.polyval(coeffs, y_samples)
                
                # Draw smooth lane
                for i in range(len(y_samples) - 1):
                    y1, y2 = int(y_samples[i]), int(y_samples[i+1])
                    x1, x2 = int(x_samples[i]), int(x_samples[i+1])
                    cv2.line(result_mask, (x1, y1), (x2, y2), 255, 8)
            except:
                pass
        
        # Fit right lane
        if np.sum(right_mask) > 30:
            right_y = y_coords[right_mask]
            right_x = x_coords[right_mask]
            
            sort_idx = np.argsort(right_y)
            right_y = right_y[sort_idx]
            right_x = right_x[sort_idx]
            
            try:
                coeffs = np.polyfit(right_y, right_x, 2)
                y_samples = np.linspace(right_y.min(), right_y.max(), 100)
                x_samples = np.polyval(coeffs, y_samples)
                
                for i in range(len(y_samples) - 1):
                    y1, y2 = int(y_samples[i]), int(y_samples[i+1])
                    x1, x2 = int(x_samples[i]), int(x_samples[i+1])
                    cv2.line(result_mask, (x1, y1), (x2, y2), 255, 8)
            except:
                pass
        
        return result_mask

# Generate improved labels
generator = LaneLabelGenerator()
kaggle_dir = Path("data/kaggle_frames")

print(f"Generating labels for {len(list(kaggle_dir.glob('*.jpg')))} frames...")

for img_path in tqdm(list(kaggle_dir.glob("kaggle_*.jpg")), desc="Generating"):
    frame = cv2.imread(str(img_path))
    if frame is None:
        continue
    
    # Generate multi-method mask
    mask = generator.detect_lanes_multi(frame)
    
    # Fit smooth polynomials
    mask = generator.fit_polynomial_lanes(frame, mask)
    
    # Save
    mask_path = kaggle_dir / (img_path.stem + "_mask.png")
    cv2.imwrite(str(mask_path), mask)

print("✓ Labels generated with polynomial fitting")

# Step 2: Create dataset with heavy augmentation for real-world data
print("\n" + "=" * 70)
print("STEP 2: Creating Real-World Training Dataset")
print("=" * 70)

class RealWorldLaneDataset(Dataset):
    """Dataset specifically for real-world road images"""
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []
        
        # Add Kaggle video frames
        kaggle_dir = Path("data/kaggle_frames")
        if kaggle_dir.exists():
            for img_path in kaggle_dir.glob("kaggle_*.jpg"):
                mask_path = kaggle_dir / (img_path.stem + "_mask.png")
                if mask_path.exists():
                    self.samples.append((str(img_path), str(mask_path)))
        
        # Add original video dataset frames (they're more realistic)
        clips_dir = Path("data/tusimple_video/clips")
        if clips_dir.exists():
            for clip_dir in clips_dir.glob("clip_*"):
                # Use multiple frames from each clip
                frames = sorted(clip_dir.glob("*.jpg"), key=lambda x: int(x.stem))
                for frame_path in frames[::2]:  # Every other frame
                    mask_path = clip_dir / (frame_path.stem + "_mask.png")
                    if mask_path.exists():
                        self.samples.append((str(frame_path), str(mask_path)))
                    else:
                        # Generate mask on the fly
                        frame = cv2.imread(str(frame_path))
                        if frame is not None:
                            mask = generator.detect_lanes_multi(frame)
                            mask = generator.fit_polynomial_lanes(frame, mask)
                            cv2.imwrite(str(mask_path), mask)
                            self.samples.append((str(frame_path), str(mask_path)))
        
        print(f"Total real-world samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
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

# Heavy augmentation for domain adaptation
real_world_transform = A.Compose([
    A.Resize(256, 384),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.GaussNoise(var_limit=(20, 80), p=0.4),
    A.ISONoise(intensity=(0.1, 0.3), p=0.3),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.4),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, 
                   shadow_dimension=5, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 384),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Create dataset
dataset = RealWorldLaneDataset(transform=real_world_transform)

# Use Kaggle video frames as validation
val_samples = [(str(p), str(p.parent / (p.stem + "_mask.png"))) 
               for p in Path("data/kaggle_frames").glob("kaggle_*.jpg")]
val_samples = val_samples[:20]  # First 20 as validation

# Remove val samples from train
train_samples = [s for s in dataset.samples if s not in val_samples]
dataset.samples = train_samples

print(f"Train: {len(dataset)}, Val: {len(val_samples)}")

# Create dataloaders
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

# Validation dataset
class ValDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
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

val_dataset = ValDataset(val_samples, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

# Step 3: Fine-tune model
print("\n" + "=" * 70)
print("STEP 3: Fine-Tuning Model on Real-World Data")
print("=" * 70)

class LaneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )
    
    def forward(self, x):
        return self.model(x)

model = LaneModel().to(DEVICE)

# Load pre-trained weights if available
if os.path.exists('models/best_combined_model.pth'):
    checkpoint = torch.load('models/best_combined_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded pre-trained weights")

# Loss and optimizer
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

class TverskyLoss(nn.Module):
    """Tversky loss for better recall on imbalanced data"""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky

# Use Tversky loss for better recall (detecting lanes)
criterion = TverskyLoss(alpha=0.3, beta=0.7)

# Lower learning rate for fine-tuning
optimizer = optim.Adam(model.parameters(), lr=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

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

# Train
EPOCHS = 30
best_iou = 0

print(f"\nFine-tuning for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 50)
    
    train_loss, train_iou = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss, val_iou = validate_epoch(model, val_loader, criterion, DEVICE)
    scheduler.step()
    
    print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}")
    
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_iou': best_iou,
        }, 'models/best_realworld_model.pth')
        print(f"✓ Saved best model (IoU: {best_iou:.4f})")

print(f"\nFine-tuning complete! Best Val IoU: {best_iou:.4f}")

# Step 4: Test on full Kaggle video
print("\n" + "=" * 70)
print("STEP 4: Testing on Full Kaggle Video")
print("=" * 70)

# Load best model
checkpoint = torch.load('models/best_realworld_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def process_frame(frame, model, device):
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
    
    # Resize
    pred_mask = cv2.resize(pred_mask, (w, h))
    
    # Post-processing: smooth the mask
    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_CLOSE, kernel)
    
    # Find and draw lanes
    contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = frame.copy()
    
    if len(contours) >= 1:
        all_points = []
        for cnt in contours:
            if len(cnt) >= 5:
                for point in cnt:
                    all_points.append(point[0])
        
        if len(all_points) >= 10:
            all_points = np.array(all_points)
            center_x = w // 2
            left_points = all_points[all_points[:, 0] < center_x]
            right_points = all_points[all_points[:, 0] >= center_x]
            
            y_vals = np.linspace(h//2, h-1, 50, dtype=int)
            
            # Fit and draw left lane
            if len(left_points) >= 5:
                try:
                    coeffs = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
                    left_x = np.polyval(coeffs, y_vals).astype(int)
                    pts = np.column_stack((left_x, y_vals))
                    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w)]
                    if len(pts) > 1:
                        cv2.polylines(result, [pts], False, (0, 255, 0), 8)
                except:
                    pass
            
            # Fit and draw right lane
            if len(right_points) >= 5:
                try:
                    coeffs = np.polyfit(right_points[:, 1], right_points[:, 0], 2)
                    right_x = np.polyval(coeffs, y_vals).astype(int)
                    pts = np.column_stack((right_x, y_vals))
                    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w)]
                    if len(pts) > 1:
                        cv2.polylines(result, [pts], False, (0, 0, 255), 8)
                except:
                    pass
    
    return result, pred_mask

# Process video
output_path = "kaggle_output_final.mp4"
cap = cv2.VideoCapture("kaggle_sample_video.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

ious = []
print(f"Processing {total_frames} frames...")

with tqdm(total=total_frames, desc="Processing") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result, mask = process_frame(frame, model, DEVICE)
        out.write(result)
        
        # Calculate IoU against improved labels
        h, w = frame.shape[:2]
        mask_resized = cv2.resize(mask, (w, h))
        pred_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # Generate GT
        gt_mask = generator.detect_lanes_multi(frame)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        if union > 0:
            ious.append(intersection / union)
        
        pbar.update(1)

cap.release()
out.release()

avg_iou = np.mean(ious) if ious else 0

# Summary
print("\n" + "=" * 70)
print("FINE-TUNING COMPLETE - SUMMARY")
print("=" * 70)
print(f"\nDataset:")
print(f"  Real-world samples: {len(dataset)}")
print(f"  Validation samples: {len(val_samples)}")
print(f"\nTraining:")
print(f"  Epochs: {EPOCHS}")
print(f"  Best Val IoU: {best_iou:.4f}")
print(f"\nKaggle Video Results:")
print(f"  Average IoU: {avg_iou:.4f}")
print(f"  Output: {output_path}")
print(f"\nGenerated Files:")
for f in ['models/best_realworld_model.pth', output_path]:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024
        print(f"  ✓ {f} ({size:.1f} KB)")
print("=" * 70)
