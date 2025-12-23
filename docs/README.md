# Industrial Quality Control System (PatchCore/FAULTLESS)

A complete unsupervised anomaly detection system for assembly line quality inspection using PatchCore algorithm with WideResNet50 backbone.

## System Overview

This system provides autonomous real-time quality inspection on assembly lines with:

- **PatchCore Anomaly Detection**: State-of-the-art unsupervised method using memory bank and coreset sampling
- **WideResNet50 Backbone**: Pre-trained feature extractor for robust representation
- **Raspberry Pi HQ Camera**: 12.3 megapixel Sony IMX477 sensor with superior image quality
- **Real-time Dashboard**: Web-based visualization of inspection results with heatmaps
- **Relay Control**: Physical OK/NOK signaling for production line integration
- **FTP Image Archival**: Automatic upload of inspection images

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Windows Training PC                          │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │ Good Images │──│ PatchCore    │──│ Save Model (.pth)   │    │
│  │             │  │ Training     │  │ + Threshold         │    │
│  └─────────────┘  └──────────────┘  └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Deploy Model
┌─────────────────────────────────────────────────────────────────┐
│                  Raspberry Pi 5 (CPU Inference)                 │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────┐       │
│  │ HQ Camera│──▶│ Capture Image│──▶│ PatchCore Detect  │       │
│  └──────────┘   └──────────────┘   └───────────────────┘       │
│                                             │                   │
│                    ┌────────────────────────┼──────────┐       │
│                    ▼                        ▼          ▼       │
│              ┌──────────┐           ┌──────────┐ ┌──────────┐  │
│              │ Relay K1 │ OK        │ Dashboard│ │   FTP    │  │
│              │ Relay K3 │ NOK       │  :8080   │ │  Upload  │  │
│              └──────────┘           └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
uns/
├── config/                 # Configuration files
│   ├── __init__.py
│   └── config.yaml        # Main configuration (PatchCore settings)
├── training/              # Windows training scripts
│   ├── __init__.py
│   └── train_patchcore.py # PatchCore model training
├── deployment/            # Raspberry Pi deployment
│   ├── __init__.py
│   ├── automation.py      # Main automation script
│   ├── camera_controller.py  # HQ Camera control
│   ├── gpio_controller.py    # Relay board control
│   └── ftp_uploader.py    # Async FTP upload
├── inference/             # AI inference engine
│   ├── __init__.py
│   ├── faultless.py       # PatchCore implementation
│   └── anomaly_detector.py  # Detection wrapper
├── dashboard/             # Web dashboard
│   ├── __init__.py
│   └── app.py
├── utils/                 # Shared utilities
│   ├── __init__.py
│   ├── logging_utils.py
│   └── image_utils.py
├── models/                # Trained models (deploy here)
├── docs/                  # Documentation
├── test.py                # Model testing script
└── tests/                 # Test scripts
```

## Quick Start

### 1. Training (Windows PC with GPU)

```bash
# Install dependencies
pip install -r requirements-windows.txt
# Or manually:
pip install torch torchvision timm opencv-python numpy pillow pyyaml scikit-learn matplotlib

# Prepare dataset structure:
# Training/
# └── <class_name>/
#     ├── train/
#     │   └── good/
#     │       ├── image1.png
#     │       └── image2.png
#     └── test/
#         ├── good/
#         │   └── normal_images...
#         └── defect/
#             └── anomaly_images...

# Train model
python training/train_patchcore.py \
    --class bottle \
    --data ./Training \
    --output ./models \
    --f-coreset 0.1

# Test model
python test.py \
    --model models/patchcore_bottle.pth \
    --test_dir ./test_images \
    --save_heatmaps
```

### 2. Deployment (Raspberry Pi 5)

```bash
# Install system dependencies
sudo apt update
sudo apt install python3-pip python3-picamera2 python3-gpiod

# Install Python dependencies (CPU version of PyTorch)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install timm flask flask-cors numpy opencv-python-headless pyyaml scikit-learn matplotlib

# Copy model to Raspberry Pi
# Place in: /home/pi/qc_system/models/patchcore_model.pth

# Run system
python3 deployment/automation.py --config config/config.yaml
```

### 3. Access Dashboard

Open browser: `http://<raspberry-pi-ip>:8080`

## PatchCore Algorithm

PatchCore is a state-of-the-art unsupervised anomaly detection method:

1. **Feature Extraction**: Uses WideResNet50 pre-trained on ImageNet to extract deep features
2. **Memory Bank**: Stores representative feature patches from training images
3. **Coreset Sampling**: Reduces memory bank size while maintaining coverage (default 10%)
4. **Anomaly Scoring**: Computes distance to nearest neighbor in memory bank
5. **Heatmap Generation**: Visualizes anomaly locations in the image

### Key Parameters

```yaml
model:
  patchcore:
    f_coreset: 0.1      # Keep 10% of patches (smaller = faster, less accurate)
    coreset_eps: 0.90   # Random projection epsilon
    n_reweight: 3       # Neighbors for score reweighting
    backbone: "wideresnet50"
```

## Hardware Requirements

### Raspberry Pi Setup

- **Raspberry Pi 5** (8GB recommended)
- **Raspberry Pi HQ Camera** (Sony IMX477 12.3MP sensor)
- **C-mount or CS-mount lens** (12mm recommended for typical inspection)
- **Relay board** (3-channel, optocoupler isolated)
- **Industrial lighting** (LED ring or bar for consistent illumination)

### Wiring Diagram

```
Raspberry Pi 5 GPIO Header:
┌────────────────────────────────────────┐
│ Pin 11 (GPIO 17) ◄── Trigger Input     │
│ Pin 37 (GPIO 26) ──► Relay K1 (OK)     │
│ Pin 38 (GPIO 20) ──► Relay K2 (Light)  │
│ Pin 40 (GPIO 21) ──► Relay K3 (NOK)    │
│ Pin 39 (GND)     ──► Relay GND         │
└────────────────────────────────────────┘

Note: Relays are active-LOW (0=ON, 1=OFF)
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# PatchCore model settings
model:
  paths:
    patchcore_model: "patchcore_model.pth"
    patchcore_threshold: "patchcore_threshold.json"

  inference:
    input_size: [512, 512]
    use_gpu: true  # false for Raspberry Pi
    generate_heatmap: true

  thresholds:
    anomaly_score: 0.0  # Auto-computed during training

  patchcore:
    f_coreset: 0.1
    coreset_eps: 0.90
    backbone: "wideresnet50"

# Camera settings
camera:
  resolution:
    capture_width: 2028
    capture_height: 1520
  exposure:
    mode: manual
    exposure_time_us: 10000

# GPIO relay settings
gpio:
  relays:
    k1_ok:
      pulse_duration_ms: 1000
    k3_nok:
      pulse_duration_ms: 3000
```

## Training Best Practices

### Data Collection

1. Collect 100-500 images of "good" products
2. Include variations in:
   - Slight position differences
   - Minor lighting changes
   - Manufacturing tolerances (within spec)
3. Use consistent camera settings

### Model Tuning

1. Start with `f_coreset=0.1` (10% of patches)
2. For better accuracy, increase to `0.25` or higher
3. After training, optimal threshold is computed automatically
4. Test with both good and defective samples

### Threshold Adjustment

The training script computes optimal threshold using ROC analysis:
- **Higher threshold**: Fewer false NOK (may miss defects)
- **Lower threshold**: Fewer false OK (more false alarms)

## Python API

```python
from inference.anomaly_detector import AnomalyDetector, create_model, load_model

# For inference
detector = AnomalyDetector('models/patchcore_model.pth', 'models/threshold.json')
result = detector.detect('image.jpg')
print(result)
# {'is_normal': True, 'anomaly_score': 0.15, 'confidence': 0.85, ...}

# For training
from inference.faultless import create_model, train_model, MVTecDataset

model = create_model(f_coreset=0.1, device="cuda")
dataset = MVTecDataset("product", source="Training")
train_loader, test_loader = dataset.get_dataloaders()
train_model(model, train_loader)
model.save("models/patchcore_product.pth")
```

## CLI Commands

```bash
# Train new model
python training/train_patchcore.py --class product --data Training --output models

# Test model on images
python test.py --model models/patchcore_product.pth --test_dir test_images

# Test with heatmap visualization
python test.py --model models/patchcore_product.pth --test_dir test_images --save_heatmaps

# Run automation system
python deployment/automation.py --config config/config.yaml

# Run single inspection
python deployment/automation.py --config config/config.yaml --single

# Run inference daemon
python -m inference.anomaly_detector --model models/patchcore_model.pth
```

## Systemd Service

Create `/etc/systemd/system/qc-system.service`:

```ini
[Unit]
Description=Quality Control System (PatchCore)
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/qc_system
ExecStart=/usr/bin/python3 deployment/automation.py --config config/config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable qc-system
sudo systemctl start qc-system
```

## Troubleshooting

### PyTorch/CUDA Issues

```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# For Raspberry Pi, use CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Model Loading Errors

```bash
# Ensure timm is installed
pip install timm

# Check model file exists
ls -la models/patchcore_model.pth
```

### Camera Not Found

```bash
# Check camera connection
libcamera-hello

# Check permissions
sudo usermod -aG video $USER
```

### Slow Inference

1. Reduce `f_coreset` (e.g., 0.05 instead of 0.1)
2. Use smaller input size (e.g., 224x224 instead of 512x512)
3. For Raspberry Pi, ensure using CPU-optimized PyTorch

## API Reference

### Dashboard API

- `GET /api/status` - System status
- `GET /api/latest` - Latest inspection result with image
- `GET /api/history` - Inspection history
- `GET /api/stats` - Statistics

### Detection Result Format

```json
{
  "is_normal": true,
  "anomaly_score": 0.15,
  "confidence": 0.85,
  "inference_time_ms": 150,
  "threshold": 0.5,
  "image_path": "/path/to/image.jpg"
}
```

## License

This project is provided for industrial automation use.

## Support

For issues and questions, please refer to the documentation or contact the system administrator.
