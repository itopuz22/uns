# Industrial Quality Control System

A complete unsupervised anomaly detection system for assembly line quality inspection using Raspberry Pi 5 with AI HAT and HQ Camera.

## System Overview

This system provides autonomous real-time quality inspection on assembly lines with:

- **Unsupervised Anomaly Detection**: Autoencoder-based model that learns from "good" samples only
- **Raspberry Pi HQ Camera**: 12.3 megapixel Sony IMX477 sensor with superior image quality
- **AI HAT Acceleration**: Hardware-accelerated inference on Raspberry Pi 5
- **Real-time Dashboard**: Web-based visualization of inspection results
- **Relay Control**: Physical OK/NOK signaling for production line integration
- **FTP Image Archival**: Automatic upload of inspection images

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Windows Training PC                          │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │ Good Images │──│ Train Model  │──│ Export TFLite/ONNX  │    │
│  └─────────────┘  └──────────────┘  └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Deploy Model
┌─────────────────────────────────────────────────────────────────┐
│                  Raspberry Pi 5 + AI HAT                        │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────┐       │
│  │ HQ Camera│──▶│ Capture Image│──▶│ Anomaly Detection │       │
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
│   └── config.yaml        # Main configuration
├── training/              # Windows training scripts
│   ├── __init__.py
│   ├── data_loader.py     # Dataset loading and augmentation
│   ├── train_autoencoder.py  # Model training
│   └── export_model.py    # Model export and optimization
├── deployment/            # Raspberry Pi deployment
│   ├── __init__.py
│   ├── automation.py      # Main automation script
│   ├── camera_controller.py  # HQ Camera control
│   ├── gpio_controller.py    # Relay board control
│   └── ftp_uploader.py    # Async FTP upload
├── inference/             # AI inference engine
│   ├── __init__.py
│   └── anomaly_detector.py  # TFLite inference with AI HAT
├── dashboard/             # Web dashboard
│   ├── __init__.py
│   └── app.py
├── utils/                 # Shared utilities
│   ├── __init__.py
│   ├── logging_utils.py
│   └── image_utils.py
├── models/                # Trained models (deploy here)
├── docs/                  # Documentation
└── tests/                 # Test scripts
```

## Quick Start

### 1. Training (Windows PC)

```bash
# Install dependencies
pip install tensorflow opencv-python numpy pillow pyyaml

# Prepare dataset
# Place "good" product images in: data/good/

# Train model
python -m training.train_autoencoder \
    --data_dir ./data \
    --output_dir ./models \
    --epochs 100 \
    --batch_size 32

# Export for Raspberry Pi
python -m training.export_model \
    --model_dir ./models \
    --output_dir ./models/exported \
    --quantization int8 \
    --calibration_dir ./data
```

### 2. Deployment (Raspberry Pi 5)

```bash
# Install system dependencies
sudo apt update
sudo apt install python3-pip python3-picamera2 python3-gpiod

# Install Python dependencies
pip3 install tflite-runtime flask flask-cors numpy opencv-python-headless pyyaml

# Copy model and config to Raspberry Pi
# Place in: /home/pi/qc_system/models/

# Run system
python3 -m deployment.automation --config config/config.yaml
```

### 3. Access Dashboard

Open browser: `http://<raspberry-pi-ip>:8080`

## Hardware Requirements

### Raspberry Pi Setup

- **Raspberry Pi 5** (8GB recommended)
- **Raspberry Pi AI HAT** (Hailo-8L for accelerated inference)
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

- Camera resolution and exposure settings
- GPIO pin assignments
- Anomaly detection thresholds
- FTP server credentials
- Dashboard settings

Key configuration options:

```yaml
camera:
  resolution:
    capture_width: 2028   # Half resolution for speed
    capture_height: 1520
  exposure:
    mode: manual
    exposure_time_us: 10000

model:
  thresholds:
    reconstruction_error: 0.05  # Adjust based on validation

gpio:
  relays:
    k1_ok:
      pulse_duration_ms: 1000
    k3_nok:
      pulse_duration_ms: 3000
```

## Camera Calibration

### Focus Adjustment (Manual)

1. Place a reference product at the inspection position
2. Rotate the HQ Camera focus ring until image is sharp
3. Lock the focus ring position with the locking screw

### Exposure Tuning

1. Start with auto exposure: `camera.set_exposure(auto=True)`
2. Capture sample images and note the auto values
3. Switch to manual with optimal values:
   ```python
   camera.set_exposure(
       exposure_time_us=10000,  # 10ms
       gain=1.0,
       auto=False
   )
   ```

### White Balance

For consistent color under industrial lighting:

```python
camera.set_white_balance(
    red_gain=1.5,
    blue_gain=1.5,
    auto=False
)
```

## Training Best Practices

### Data Collection

1. Collect 500+ images of "good" products
2. Include variations in:
   - Slight position differences
   - Minor lighting changes
   - Manufacturing tolerances (within spec)
3. Use consistent camera settings

### Model Tuning

1. Start with default hyperparameters
2. Monitor validation loss during training
3. Adjust threshold based on false positive/negative rates:
   - Higher threshold: Fewer false NOK (may miss defects)
   - Lower threshold: Fewer false OK (more false alarms)

### Validation

Before production deployment:

1. Test with known good samples (expect OK)
2. Test with known defective samples (expect NOK)
3. Calculate precision and recall
4. Adjust threshold to meet production requirements

## Systemd Service

Create `/etc/systemd/system/qc-system.service`:

```ini
[Unit]
Description=Quality Control System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/qc_system
ExecStart=/usr/bin/python3 -m deployment.automation --config config/config.yaml
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

### Camera Not Found

```bash
# Check camera connection
libcamera-hello

# Check permissions
sudo usermod -aG video $USER
```

### GPIO Permission Denied

```bash
# Add user to gpio group
sudo usermod -aG gpio $USER

# Or use gpiod with proper permissions
sudo chmod 666 /dev/gpiochip4
```

### Model Inference Slow

1. Ensure AI HAT is properly installed
2. Check for thermal throttling: `vcgencmd measure_temp`
3. Use INT8 quantized model
4. Reduce input resolution if acceptable

### FTP Upload Failures

1. Verify network connectivity: `ping <ftp-server>`
2. Check credentials in config
3. Ensure FTP server is accessible from Pi
4. Check firewall rules

## API Reference

### Dashboard API

- `GET /api/status` - System status
- `GET /api/latest` - Latest inspection result
- `GET /api/history` - Inspection history
- `GET /api/stats` - Statistics

### Python API

```python
from deployment.automation import QualityControlSystem

# Initialize system
system = QualityControlSystem('config/config.yaml')
system.initialize()

# Run single inspection
result = system.run_single()
print(result)
# {'result': 'OK', 'confidence': 0.95, 'processing_time_ms': 150}

# Get status
status = system.get_status()
```

## License

This project is provided for industrial automation use.

## Support

For issues and questions, please refer to the documentation or contact the system administrator.
