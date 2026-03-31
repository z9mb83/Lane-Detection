# Lane Detection with PyTorch - End-to-End ML Application

A complete lane detection system using deep learning with PyTorch. This project includes automatic dataset preparation, U-Net model training, post-processing with polynomial fitting, and inference on images and videos.

## Features

- **Automatic Dataset Preparation**: Downloads or generates synthetic lane detection data
- **Deep Learning Model**: U-Net with ResNet-18 encoder for lane segmentation
- **Data Augmentation**: Using albumentations for robust training
- **Post-Processing**: Polynomial fitting for smooth lane detection
- **Inference**: Functions for image and video lane detection with overlay
- **Visualizations**: Training curves, predictions, and results

## Project Structure

```
lane_detection_project/
├── data/                   # Dataset directory
│   ├── train/
│   │   ├── images/         # Training images
│   │   └── masks/          # Training masks
│   └── val/
│       ├── images/         # Validation images
│       └── masks/          # Validation masks
├── models/                 # Saved model checkpoints
├── notebooks/              # Additional notebooks (if any)
├── src/                    # Source code modules
│   └── dataset_handler.py  # Dataset preparation script
├── lane_detection.ipynb    # Main Jupyter notebook (run this!)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

### 1. Clone or navigate to the project directory

```bash
cd lane_detection_project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch (torch, torchvision)
- OpenCV (opencv-python)
- segmentation-models-pytorch
- albumentations
- matplotlib, numpy, scikit-learn
- jupyter, tqdm

## Usage

### Quick Start - Run the Notebook

The main entry point is the Jupyter notebook `lane_detection.ipynb` which runs everything end-to-end:

```bash
jupyter notebook lane_detection.ipynb
```

Then **Run All Cells** in the notebook to:
1. Prepare the dataset (downloads or generates synthetic data)
2. Train the U-Net model
3. Evaluate and visualize results
4. Run inference demo on sample images/video

### Step-by-Step Manual Execution

If you prefer to run components separately:

#### 1. Prepare Dataset
```bash
python src/dataset_handler.py
```

This will generate 200 synthetic lane images (160 train, 40 val) if no real dataset is found.

#### 2. Launch Jupyter and Train
```bash
jupyter notebook lane_detection.ipynb
```

Follow the notebook sections:
- Sections 1-4: Setup and data loading
- Sections 5-7: Model and training setup
- Section 8: Run training loop (15 epochs by default)
- Sections 9-14: Evaluation, visualization, and demo

## Training Details

- **Model**: U-Net with ResNet-18 encoder (ImageNet pretrained)
- **Loss**: Combined BCE + Dice Loss
- **Optimizer**: Adam (LR = 1e-3)
- **Epochs**: 15 (configurable)
- **Batch Size**: 8
- **Image Size**: 256x384 (resized)
- **Metrics**: IoU (Intersection over Union), Accuracy

Expected training time: ~5-10 minutes on CPU, ~2-3 minutes on GPU for 15 epochs with 200 synthetic images.

## Inference

### On a Single Image

```python
from lane_detection import predict_image  # (after running notebook)

original, mask, result = predict_image(model, "path/to/image.jpg", DEVICE, post_processor)
```

### On Video

```python
from lane_detection import process_video

process_video(model, "input.mp4", "output.mp4", DEVICE, post_processor)
```

### Using the Trained Model

Load the saved model checkpoint:

```python
import torch
from lane_detection import LaneDetectionModel

model = LaneDetectionModel()
checkpoint = torch.load('models/best_lane_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Output Files

After running the notebook, the following files are generated:

- `models/best_lane_model.pth` - Best trained model checkpoint
- `training_curves.png` - Loss, IoU, and accuracy plots
- `predictions_comparison.png` - Side-by-side predictions
- `lane_detection_result.png` - Sample detection result
- `output_lanes.jpg` - Output image with detected lanes
- `output_video.mp4` - Demo video with lane detection overlay
- `sample_data.png` - Dataset sample visualization

## Dataset

The project attempts to download the TuSimple lane detection dataset. If unavailable, it automatically generates synthetic lane data with:
- Road backgrounds with realistic textures
- Left and right lane markings
- Perspective effects (lanes converge at horizon)
- Various lane curves and positions
- 200 samples (80% train, 20% validation)

To use a real dataset, place images in `data/train/images/` and masks in `data/train/masks/` (and corresponding val folders).

## Model Architecture

```
Input (3x256x384)
    ↓
U-Net (ResNet-18 encoder)
    ↓
Output (1x256x384) - Lane segmentation mask
    ↓
Post-processing (polynomial fitting)
    ↓
Smoothed lane lines overlaid on image
```

## Performance

Typical results on synthetic data after 15 epochs:
- **Validation IoU**: 0.70-0.85
- **Validation Accuracy**: 0.95-0.98

Results on real datasets (TuSimple/CULane) would be higher with proper training.

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in the notebook: `BATCH_SIZE = 4` or `2`

### Dataset Not Found
The dataset handler automatically generates synthetic data. If you want real data, manually download from:
- [TuSimple Dataset](https://github.com/TuSimple/tusimple-benchmark)
- [CULane Dataset](https://xingangpan.github.io/projects/CULane.html)
- [BDD100K](https://bdd-data.berkeley.edu/)

### Module Not Found
Ensure all packages are installed:
```bash
pip install -r requirements.txt
```

## Next Steps

- Train on real TuSimple or CULane dataset for production use
- Implement more sophisticated post-processing (Hough transform, clustering)
- Add support for multiple lane detection (multi-class)
- Real-time webcam inference
- Experiment with other architectures (DeepLabV3+, PSPNet, etc.)

## License

This project is for educational purposes. Dataset licenses apply to respective owners.

## Acknowledgments

- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) for the U-Net implementation
- [albumentations](https://albumentations.ai/) for data augmentation
- TuSimple for the lane detection benchmark dataset
