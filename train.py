"""
Lane Detection - Complete Training Pipeline
Run this script to train the model and generate all outputs
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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Dataset
class LaneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.images = sorted(list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        mask = (mask > 127).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        mask = mask.unsqueeze(0)
        return image, mask

# Transforms
train_transform = A.Compose([
    A.Resize(256, 384),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 384),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Load datasets
print("Loading datasets...")
train_dataset = LaneDataset("data/train/images", "data/train/masks", transform=train_transform)
val_dataset = LaneDataset("data/val/images", "data/val/masks", transform=val_transform)

BATCH_SIZE = 8
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Model
class LaneDetectionModel(nn.Module):
    def __init__(self, encoder_name="resnet18", encoder_weights="imagenet"):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, x):
        return self.model(x)

model = LaneDetectionModel().to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and metrics
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

criterion = CombinedLoss()

# Training functions
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_iou = 0
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
    total_loss = 0
    total_iou = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            total_iou += calculate_iou(outputs, masks)
    return total_loss / len(loader), total_iou / len(loader)

# Training loop
EPOCHS = 15
LR = 1e-3
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

history = {'train_loss': [], 'train_iou': [], 'val_loss': [], 'val_iou': []}
best_iou = 0

print(f"\nTraining for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
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
            'optimizer_state_dict': optimizer.state_dict(),
            'best_iou': best_iou,
        }, 'models/best_lane_model.pth')
        print(f"Saved best model (IoU: {best_iou:.4f})")

print(f"\nTraining complete! Best Val IoU: {best_iou:.4f}")

# Plot training curves
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
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
print("Saved training_curves.png")

# Visualize predictions
print("\nGenerating prediction visualizations...")
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
model.eval()

for i in range(4):
    image, mask = val_dataset[i]
    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(DEVICE))
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()

    img_np = image.permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    axes[0, i].imshow(img_np)
    axes[0, i].set_title(f"Image {i+1}")
    axes[0, i].axis('off')

    axes[1, i].imshow(mask.squeeze().numpy(), cmap='gray')
    axes[1, i].set_title(f"Ground Truth {i+1}")
    axes[1, i].axis('off')

    axes[2, i].imshow(pred, cmap='gray')
    axes[2, i].set_title(f"Prediction {i+1}")
    axes[2, i].axis('off')

plt.tight_layout()
plt.savefig('predictions_comparison.png', dpi=150, bbox_inches='tight')
print("Saved predictions_comparison.png")

# Sample inference with overlay
print("\nGenerating sample lane detection result...")
sample_img_path = list(Path("data/val/images").glob("*.jpg"))[0]
image = cv2.imread(str(sample_img_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original = image.copy()
h, w = image.shape[:2]

# Preprocess
image_resized = cv2.resize(image, (384, 256))
image_norm = image_resized / 255.0
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image_norm = (image_norm - mean) / std
image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).float().unsqueeze(0)

# Predict
model.eval()
with torch.no_grad():
    pred = model(image_tensor.to(DEVICE))
    pred = torch.sigmoid(pred).squeeze().cpu().numpy()

pred_mask = cv2.resize(pred, (w, h))
pred_binary = (pred_mask > 0.5).astype(np.uint8)

# Draw overlay
overlay = original.copy()
overlay[pred_binary > 0] = [0, 255, 0]
result = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(pred_mask, cmap='gray')
axes[1].set_title("Predicted Mask")
axes[1].axis('off')

axes[2].imshow(result)
axes[2].set_title("Lane Detection Result")
axes[2].axis('off')

plt.tight_layout()
plt.savefig('lane_detection_result.png', dpi=150, bbox_inches='tight')
print("Saved lane_detection_result.png")

# Save output image
cv2.imwrite('output_lanes.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
print("Saved output_lanes.jpg")

# Final summary
print("\n" + "=" * 60)
print("LANE DETECTION PROJECT COMPLETE")
print("=" * 60)
print(f"\nBest Validation IoU: {best_iou:.4f}")
print(f"Model saved to: models/best_lane_model.pth")
print(f"Device used: {DEVICE}")
print("\nGenerated files:")
for f in ['training_curves.png', 'predictions_comparison.png',
          'lane_detection_result.png', 'output_lanes.jpg']:
    if os.path.exists(f):
        print(f"  ✓ {f}")
print("\n" + "=" * 60)
