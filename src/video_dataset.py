"""
Video Lane Dataset PyTorch Classes
Handles video sequences for lane detection with temporal information
"""
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import json
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class VideoLaneDataset(Dataset):
    """
    Dataset for video lane detection
    Loads sequences of frames and corresponding lane labels
    """
    def __init__(self, data_dir, sequence_length=5, transform=None, 
                 target_frame_idx=None, split='train', train_split=0.8):
        """
        Args:
            data_dir: Path to dataset directory (e.g., data/tusimple_video)
            sequence_length: Number of consecutive frames to load
            transform: Albumentations transforms
            target_frame_idx: Which frame in sequence has labels (None = last frame)
            split: 'train' or 'val'
            train_split: Fraction of data for training
        """
        self.data_dir = Path(data_dir)
        self.clips_dir = self.data_dir / "clips"
        self.label_file = self.data_dir / "label_data.json"
        self.sequence_length = sequence_length
        self.transform = transform
        self.target_frame_idx = target_frame_idx if target_frame_idx is not None else sequence_length - 1
        
        # Load labels
        self.samples = []
        self._load_labels(split, train_split)
        
    def _load_labels(self, split, train_split):
        """Load and split labels"""
        with open(self.label_file, 'r') as f:
            all_labels = [json.loads(line) for line in f.readlines()]
        
        # Split train/val
        n_total = len(all_labels)
        n_train = int(n_total * train_split)
        
        if split == 'train':
            labels = all_labels[:n_train]
        else:
            labels = all_labels[n_train:]
        
        # Each clip becomes a sample
        for label in labels:
            clip_path = self.data_dir / label['raw_file']
            clip_dir = clip_path.parent
            
            # Get all frames in clip
            frames = sorted([f for f in clip_dir.glob("*.jpg")], 
                          key=lambda x: int(x.stem))
            
            if len(frames) >= self.sequence_length:
                self.samples.append({
                    'clip_dir': clip_dir,
                    'frames': frames,
                    'lanes': label['lanes'],
                    'h_samples': label['h_samples']
                })
        
        print(f"VideoLaneDataset [{split}]: Loaded {len(self.samples)} clips")
        
    def __len__(self):
        return len(self.samples)
    
    def _load_frame(self, frame_path):
        """Load and preprocess a single frame"""
        frame = cv2.imread(str(frame_path))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def _create_mask(self, lanes, h_samples, height, width):
        """Create segmentation mask from lane labels"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for lane_x in lanes:
            if len(lane_x) != len(h_samples):
                continue
            points = []
            for x, y in zip(lane_x, h_samples):
                if x >= 0 and x < width:
                    points.append([x, y])
            
            if len(points) > 1:
                points = np.array(points, dtype=np.int32)
                cv2.polylines(mask, [points], False, 255, thickness=5)
        
        return mask
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = sample['frames']
        
        # Take last N frames (or fewer if clip is shorter)
        start_idx = max(0, len(frames) - self.sequence_length)
        frame_paths = frames[start_idx:start_idx + self.sequence_length]
        
        # Pad if needed
        while len(frame_paths) < self.sequence_length:
            frame_paths.insert(0, frame_paths[0])
        
        # Load frames
        frames_list = []
        for fp in frame_paths:
            frame = self._load_frame(fp)
            frames_list.append(frame)
        
        # Target frame is the one with labels
        target_frame = frames_list[self.target_frame_idx]
        
        # Create mask for target frame
        height, width = target_frame.shape[:2]
        mask = self._create_mask(sample['lanes'], sample['h_samples'], height, width)
        mask = (mask > 127).astype(np.float32)
        
        # Apply transforms to all frames
        if self.transform:
            # Transform target frame and mask
            transformed = self.transform(image=target_frame, mask=mask)
            target_frame = transformed['image']
            mask = transformed['mask']
            
            # Transform other frames (without mask)
            transformed_frames = []
            for i, frame in enumerate(frames_list):
                if i == self.target_frame_idx:
                    transformed_frames.append(target_frame)
                else:
                    tf = self.transform(image=frame)
                    transformed_frames.append(tf['image'])
            frames_list = transformed_frames
        else:
            # Simple normalize without albumentations
            target_frame = torch.from_numpy(target_frame.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)
            frames_list = [torch.from_numpy(f.transpose(2, 0, 1)).float() / 255.0 
                          for f in frames_list]
        
        # Stack frames: [sequence_length, C, H, W]
        frames_tensor = torch.stack(frames_list)
        
        # Ensure mask has channel dimension [1, H, W]
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.from_numpy(mask).unsqueeze(0)
        
        return frames_tensor, mask


class TemporalFrameStacker:
    """
    Stacks multiple frames as input channels for temporal processing
    Converts [T, C, H, W] -> [T*C, H, W]
    """
    @staticmethod
    def stack(frames_tensor):
        """
        Args:
            frames_tensor: [B, T, C, H, W] or [T, C, H, W]
        Returns:
            stacked: [B, T*C, H, W] or [T*C, H, W]
        """
        if frames_tensor.dim() == 5:
            B, T, C, H, W = frames_tensor.shape
            return frames_tensor.view(B, T * C, H, W)
        elif frames_tensor.dim() == 4:
            T, C, H, W = frames_tensor.shape
            return frames_tensor.view(T * C, H, W)
        else:
            raise ValueError(f"Unexpected tensor shape: {frames_tensor.shape}")


class TemporalConvBlock(torch.nn.Module):
    """
    Simple temporal convolution block for video lane detection
    Applies 3D conv to capture temporal patterns
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3)):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=kernel_size,
            padding=(kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2)
        )
        self.bn = torch.nn.BatchNorm3d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        
    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TemporalLaneModel(torch.nn.Module):
    """
    Lane detection model with temporal processing
    Uses frame stacking approach (concatenate temporal frames as channels)
    """
    def __init__(self, base_model, sequence_length=5):
        super().__init__()
        self.sequence_length = sequence_length
        self.base_model = base_model
        
        # Modify first conv to accept temporal input
        # Original model expects 3 channels, now we have 3 * sequence_length
        in_channels = 3 * sequence_length
        
        # Get the encoder from base model
        if hasattr(base_model, 'model'):
            encoder = base_model.model.encoder
        else:
            encoder = base_model.encoder
            
        # Modify first conv layer
        first_conv = encoder.conv1
        self.temporal_conv = torch.nn.Conv2d(
            in_channels, 
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )
        
        # Copy weights - repeat across temporal dimension
        with torch.no_grad():
            # Average the original weights across input channels
            orig_weight = first_conv.weight.data
            # Repeat for temporal frames
            new_weight = orig_weight.repeat(1, sequence_length, 1, 1) / sequence_length
            self.temporal_conv.weight.data = new_weight
        
        # Replace the first conv in encoder
        encoder.conv1 = self.temporal_conv
        
    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] - batch of frame sequences
        Returns:
            output: [B, 1, H, W] - segmentation mask
        """
        # Stack temporal frames as channels: [B, T, C, H, W] -> [B, T*C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)
        
        # Pass through base model
        output = self.base_model(x)
        return output


# Transforms for video data
def get_video_train_transforms():
    return A.Compose([
        A.Resize(256, 384),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_video_val_transforms():
    return A.Compose([
        A.Resize(256, 384),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


if __name__ == "__main__":
    # Test the dataset
    dataset = VideoLaneDataset(
        "data/tusimple_video",
        sequence_length=5,
        transform=get_video_val_transforms(),
        split='train'
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        frames, mask = dataset[0]
        print(f"Frames shape: {frames.shape}")
        print(f"Mask shape: {mask.shape}")
        
        # Test temporal stacking
        stacked = TemporalFrameStacker.stack(frames)
        print(f"Stacked shape: {stacked.shape}")
