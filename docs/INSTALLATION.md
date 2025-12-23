# Installation Guide (PatchCore/FAULTLESS)

Complete installation instructions for both Windows training environment and Raspberry Pi deployment using the PatchCore anomaly detection system.

## Part 1: Windows Training Environment

### System Requirements

- Windows 10/11 (64-bit)
- Python 3.9 or later
- NVIDIA GPU with CUDA support (recommended for faster training)
- 16GB+ RAM recommended
- 20GB+ free disk space

### Step 1: Install Python

1. Download Python from https://www.python.org/downloads/
2. Run installer with "Add Python to PATH" checked
3. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

### Step 2: Create Virtual Environment

```cmd
# Create project directory
mkdir C:\QualityControl
cd C:\QualityControl

# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate
```

### Step 3: Install Dependencies

```cmd
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (GPU version if NVIDIA GPU available)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Or for CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install timm numpy opencv-python pillow pyyaml scikit-learn matplotlib
```

### Step 4: Copy Project Files

Copy the following directories to your project folder:
- `config/`
- `training/`
- `inference/`
- `utils/`
- `test.py`

### Step 5: Verify Installation

```cmd
# Test PyTorch
python -c "import torch; print('PyTorch:', torch.__version__)"

# Test GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Test timm (WideResNet50 backbone)
python -c "import timm; print('timm:', timm.__version__)"
```

### Step 6: Prepare Training Data

PatchCore uses MVTec-style data structure:

```cmd
# Create data directory structure
mkdir Training\product\train\good
mkdir Training\product\test\good
mkdir Training\product\test\defect

# Copy images:
# - "Good" training images -> Training\product\train\good\
# - "Good" test images -> Training\product\test\good\
# - "Defect" test images -> Training\product\test\defect\
#
# Recommended: 100-500 good images for training
# Use consistent lighting and camera position
```

### Step 7: Train Model

```cmd
# Using the dedicated PatchCore training script
python training/train_patchcore.py ^
    --class product ^
    --data Training ^
    --output models ^
    --f-coreset 0.1

# Or using the legacy script name (same functionality)
python training/train_autoencoder.py ^
    --data_dir Training ^
    --class_name product ^
    --output_dir models ^
    --f_coreset 0.1
```

Training parameters:
- `--f-coreset 0.1`: Keep 10% of feature patches (default, good balance)
- `--f-coreset 0.25`: Keep 25% for better accuracy (slower inference)
- `--f-coreset 0.05`: Keep 5% for faster inference (less accurate)

### Step 8: Verify Model

```cmd
# Verify model can be loaded
python training/export_model.py ^
    --model models/patchcore_product.pth ^
    --verify

# Get model information
python training/export_model.py ^
    --model models/patchcore_product.pth ^
    --info
```

Note: PatchCore models are saved as PyTorch .pth files and don't require conversion to TFLite or ONNX. Deploy the .pth file directly.

### Step 9: Test Model

```cmd
# Test on images
python test.py ^
    --model models/patchcore_product.pth ^
    --test_dir test_images

# Test with heatmap visualization
python test.py ^
    --model models/patchcore_product.pth ^
    --test_dir test_images ^
    --save_heatmaps
```

---

## Part 2: Raspberry Pi Deployment

### Hardware Requirements

- Raspberry Pi 5 (8GB recommended)
- Raspberry Pi HQ Camera with lens (12mm recommended)
- MicroSD card (32GB+ recommended)
- Quality power supply (27W USB-C PD)
- Relay board (3-channel, optocoupler isolated)
- Industrial LED lighting

Note: AI HAT is NOT required. PatchCore runs on CPU using PyTorch.

### Step 1: Install Raspberry Pi OS

1. Download Raspberry Pi Imager
2. Flash Raspberry Pi OS (64-bit, Bookworm)
3. Configure WiFi and SSH during setup
4. Boot and connect via SSH

### Step 2: System Updates

```bash
sudo apt update
sudo apt upgrade -y
sudo reboot
```

### Step 3: Enable Camera

```bash
# Camera should be auto-detected on Pi 5
# Verify with:
libcamera-hello

# If not working, check cable connection
# Add user to video group if needed:
sudo usermod -aG video $USER
```

### Step 4: Install Python Dependencies

```bash
# System packages
sudo apt install -y python3-pip python3-venv python3-picamera2 \
    python3-gpiod python3-yaml

# Create project directory
mkdir -p ~/qc_system
cd ~/qc_system

# Create virtual environment
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Install PyTorch (CPU version for Raspberry Pi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install timm flask flask-cors numpy opencv-python-headless \
    pillow scikit-learn matplotlib
```

### Step 5: Copy Project Files

```bash
# Copy from development machine using SCP or SFTP:
# - config/
# - deployment/
# - inference/
# - dashboard/
# - utils/
# - models/patchcore_*.pth  (trained model files)
# - models/patchcore_*_threshold.json  (threshold files)
```

Example using SCP:
```bash
# From Windows (in project directory):
scp -r config inference deployment dashboard utils pi@<raspberry-pi-ip>:~/qc_system/
scp models/patchcore_*.pth pi@<raspberry-pi-ip>:~/qc_system/models/
scp models/patchcore_*_threshold.json pi@<raspberry-pi-ip>:~/qc_system/models/
```

### Step 6: Configure System

Edit `config/config.yaml`:

```bash
cd ~/qc_system
nano config/config.yaml
```

Update these settings:
```yaml
model:
  paths:
    patchcore_model: "patchcore_product.pth"
    patchcore_threshold: "patchcore_product_threshold.json"
  inference:
    input_size: [512, 512]
    use_gpu: false  # Use CPU on Raspberry Pi
    generate_heatmap: true

camera:
  resolution:
    capture_width: 2028
    capture_height: 1520

gpio:
  trigger_input:
    pin: 17
  relays:
    k1_ok:
      pin: 26
      pulse_duration_ms: 1000
    k3_nok:
      pin: 21
      pulse_duration_ms: 3000
```

### Step 7: Test Components

```bash
cd ~/qc_system
source venv/bin/activate

# Test camera
python3 -c "
from deployment.camera_controller import CameraController
cam = CameraController()
cam.initialize()
cam.start()
print('Capturing test image...')
path = cam.capture()
print(f'Saved to: {path}')
cam.close()
"

# Test GPIO
python3 -c "
from deployment.gpio_controller import GPIOController
gpio = GPIOController()
gpio.initialize()
print(f'Trigger state: {gpio.read_trigger()}')
print('Pulsing K2 (light)...')
gpio.pulse_relay('k2', 1000)
gpio.cleanup()
"

# Test PatchCore inference
python3 -c "
from inference.anomaly_detector import create_detector
detector = create_detector(
    model_path='models/patchcore_product.pth',
    threshold_path='models/patchcore_product_threshold.json',
    use_gpu=False
)
print('Model loaded successfully!')
print(f'Threshold: {detector.threshold}')
"

# Test on a sample image
python3 test.py \
    --model models/patchcore_product.pth \
    --test_dir /tmp/test_images \
    --threshold_file models/patchcore_product_threshold.json
```

### Step 8: Run System

```bash
cd ~/qc_system
source venv/bin/activate

# Interactive mode (for testing)
python3 deployment/automation.py --config config/config.yaml

# Single inspection test
python3 deployment/automation.py --config config/config.yaml --single
```

### Step 9: Setup Systemd Service

```bash
# Create service file
sudo nano /etc/systemd/system/qc-system.service
```

Add content:
```ini
[Unit]
Description=Quality Control System (PatchCore)
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/qc_system
Environment=PATH=/home/pi/qc_system/venv/bin:/usr/bin
ExecStart=/home/pi/qc_system/venv/bin/python deployment/automation.py --config config/config.yaml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable qc-system
sudo systemctl start qc-system

# Check status
sudo systemctl status qc-system

# View logs
sudo journalctl -u qc-system -f
```

### Step 10: Dashboard Setup

The dashboard starts automatically with the main service on port 8080.

Access from any device on the network:
```
http://<raspberry-pi-ip>:8080
```

### Step 11: Production Hardening

```bash
# Set up log rotation
sudo nano /etc/logrotate.d/qc-system
```

Add:
```
/var/log/qc_system/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 pi pi
}
```

```bash
# Set up watchdog
sudo apt install watchdog
sudo systemctl enable watchdog
```

---

## Verification Checklist

### Training Environment
- [ ] Python 3.9+ installed
- [ ] PyTorch installed and working
- [ ] timm library installed
- [ ] GPU detected (if applicable)
- [ ] Training data prepared (MVTec structure)
- [ ] Model trained successfully (.pth file created)
- [ ] Threshold file generated (.json)
- [ ] Model verified with test images

### Deployment Environment
- [ ] Raspberry Pi OS 64-bit installed
- [ ] Camera working (`libcamera-hello`)
- [ ] GPIO permissions configured
- [ ] PyTorch CPU version installed
- [ ] timm library installed
- [ ] Model files copied (.pth and threshold.json)
- [ ] Configuration updated
- [ ] Component tests passing
- [ ] Systemd service running
- [ ] Dashboard accessible
- [ ] FTP connection working (if enabled)

---

## Troubleshooting

### Training Issues

**"CUDA out of memory"**
- Reduce coreset fraction: `--f-coreset 0.05`
- Use smaller batch size: `--batch-size 1`
- Use smaller image size: `--image-size 224`

**"No GPU found"**
- Install CUDA and cuDNN
- Install correct PyTorch version for your CUDA: https://pytorch.org/get-started/locally/
- Check NVIDIA drivers: `nvidia-smi`

**"timm model not found"**
- Update timm: `pip install --upgrade timm`
- Check network connection for downloading pre-trained weights

### Deployment Issues

**"Camera not found"**
- Check ribbon cable connection
- Verify camera with `libcamera-hello --list-cameras`
- Add user to video group: `sudo usermod -aG video $USER`

**"Permission denied: gpiochip4"**
- Add user to gpio group: `sudo usermod -aG gpio $USER`
- Logout and login again

**"Model file not found"**
- Verify model path in config
- Check file permissions: `ls -la models/`
- Ensure .pth file was copied correctly

**"Slow inference"**
- Use smaller coreset fraction model (train with `--f-coreset 0.05`)
- Reduce input image size in config
- Ensure using CPU-optimized PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

**"Memory error on Raspberry Pi"**
- Close other applications
- Use swap: `sudo dphys-swapfile swapoff && sudo nano /etc/dphys-swapfile` (set CONF_SWAPSIZE=2048)
- Train with smaller coreset fraction

**"Module not found: inference.faultless"**
- Ensure all project files are copied
- Check Python path: `export PYTHONPATH=/home/pi/qc_system:$PYTHONPATH`
