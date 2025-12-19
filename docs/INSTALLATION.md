# Installation Guide

Complete installation instructions for both Windows training environment and Raspberry Pi deployment.

## Part 1: Windows Training Environment

### System Requirements

- Windows 10/11 (64-bit)
- Python 3.9 or later
- NVIDIA GPU with CUDA support (recommended for faster training)
- 16GB+ RAM recommended
- 50GB+ free disk space

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

# Install TensorFlow (GPU version if NVIDIA GPU available)
pip install tensorflow

# Or for CPU only:
# pip install tensorflow-cpu

# Install other dependencies
pip install numpy opencv-python pillow pyyaml scikit-learn matplotlib

# For ONNX export support
pip install tf2onnx onnx
```

### Step 4: Copy Project Files

Copy the following directories to your project folder:
- `config/`
- `training/`
- `utils/`

### Step 5: Verify Installation

```cmd
# Test TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# Test GPU availability
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

### Step 6: Prepare Training Data

```cmd
# Create data directory structure
mkdir data\good

# Copy "good" product images to data\good\
# Recommended: 500+ images, JPG format, consistent resolution
```

### Step 7: Train Model

```cmd
python -m training.train_autoencoder ^
    --data_dir data ^
    --output_dir models ^
    --epochs 100 ^
    --batch_size 32 ^
    --input_size 224
```

### Step 8: Export Model

```cmd
python -m training.export_model ^
    --model_dir models ^
    --output_dir models\exported ^
    --quantization int8 ^
    --calibration_dir data
```

---

## Part 2: Raspberry Pi Deployment

### Hardware Requirements

- Raspberry Pi 5 (8GB recommended)
- Raspberry Pi AI HAT (Hailo-8L)
- Raspberry Pi HQ Camera with lens
- MicroSD card (32GB+ recommended)
- Quality power supply (27W USB-C PD)
- Relay board (3-channel)

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

# If not working, check cable connection and boot config
```

### Step 4: Install AI HAT Drivers

```bash
# Install Hailo runtime (follow official instructions)
# https://www.raspberrypi.com/documentation/accessories/ai-hat.html

# For basic setup:
sudo apt install hailo-all

# Verify installation
hailortcli scan
```

### Step 5: Install Python Dependencies

```bash
# System packages
sudo apt install -y python3-pip python3-venv python3-picamera2 \
    python3-gpiod python3-opencv python3-numpy python3-yaml

# Create project directory
mkdir -p ~/qc_system
cd ~/qc_system

# Create virtual environment
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Install additional packages
pip install flask flask-cors tflite-runtime pillow
```

### Step 6: Copy Project Files

```bash
# Copy from development machine using SCP or SFTP:
# - config/
# - deployment/
# - inference/
# - dashboard/
# - utils/
# - models/exported/  (trained model files)
```

### Step 7: Configure System

Edit `config/config.yaml`:

```bash
nano config/config.yaml
```

Update these settings:
- Camera resolution (based on inspection requirements)
- GPIO pin numbers (match your wiring)
- FTP server credentials
- Anomaly threshold (from training validation)

### Step 8: Test Components

```bash
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

# Test inference
python3 -c "
from inference.anomaly_detector import AnomalyDetector
detector = AnomalyDetector('models/exported/anomaly_detector.tflite')
result = detector.detect('/tmp/test_image.jpg')
print(f'Result: {result}')
"
```

### Step 9: Run System

```bash
# Interactive mode (for testing)
python3 -m deployment.automation --config config/config.yaml

# Single inspection test
python3 -m deployment.automation --config config/config.yaml --single
```

### Step 10: Setup Systemd Service

```bash
# Create service file
sudo nano /etc/systemd/system/qc-system.service
```

Add content:
```ini
[Unit]
Description=Quality Control System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/qc_system
Environment=PATH=/home/pi/qc_system/venv/bin:/usr/bin
ExecStart=/home/pi/qc_system/venv/bin/python -m deployment.automation --config config/config.yaml
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

### Step 11: Dashboard Setup

The dashboard starts automatically with the main service on port 8080.

Access from any device on the network:
```
http://<raspberry-pi-ip>:8080
```

### Step 12: Production Hardening

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
- [ ] TensorFlow installed and working
- [ ] GPU detected (if applicable)
- [ ] Training data prepared
- [ ] Model trained successfully
- [ ] Model exported to TFLite

### Deployment Environment
- [ ] Raspberry Pi OS 64-bit installed
- [ ] Camera working (`libcamera-hello`)
- [ ] AI HAT detected (`hailortcli scan`)
- [ ] GPIO permissions configured
- [ ] Model files copied
- [ ] Configuration updated
- [ ] Component tests passing
- [ ] Systemd service running
- [ ] Dashboard accessible
- [ ] FTP connection working

---

## Troubleshooting

### Training Issues

**"CUDA out of memory"**
- Reduce batch size: `--batch_size 16`
- Use smaller input size: `--input_size 160`

**"No GPU found"**
- Install CUDA and cuDNN
- Install tensorflow-gpu
- Check NVIDIA drivers: `nvidia-smi`

### Deployment Issues

**"Camera not found"**
- Check ribbon cable connection
- Verify camera enabled in raspi-config
- Try `libcamera-hello --list-cameras`

**"Permission denied: gpiochip4"**
- Add user to gpio group: `sudo usermod -aG gpio $USER`
- Logout and login again

**"Model file not found"**
- Verify model path in config
- Check file permissions: `ls -la models/`

**"AI HAT not detected"**
- Verify HAT is properly seated
- Check for kernel modules: `lsmod | grep hailo`
- Update firmware: `sudo rpi-eeprom-update`
