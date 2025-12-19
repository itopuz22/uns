# Raspberry Pi HQ Camera Setup Guide

Complete guide for setting up the Raspberry Pi HQ Camera (Sony IMX477) for industrial quality inspection.

## Camera Specifications

| Feature | Specification |
|---------|--------------|
| Sensor | Sony IMX477 |
| Resolution | 12.3 megapixels (4056 × 3040) |
| Sensor Size | 7.9mm diagonal (Type 1/2.3) |
| Pixel Size | 1.55μm × 1.55μm |
| Sensor Type | Back-illuminated |
| Mount | C/CS-mount (C-mount with included adapter) |
| Focus | Manual (lens-dependent) |
| Exposure | 1μs to 230s |

## Hardware Setup

### Connecting the Camera

1. **Power off** the Raspberry Pi
2. Locate the camera connector on Pi 5 (between USB and HDMI ports)
3. Gently pull up the connector latch
4. Insert the ribbon cable with contacts facing HDMI ports
5. Push down the latch to secure
6. Power on and verify with `libcamera-hello`

### Mounting the Camera

For industrial inspection:

1. Use a rigid mounting bracket
2. Position camera perpendicular to inspection surface
3. Ensure vibration isolation from production line
4. Allow access for lens adjustment

### Lens Selection

| Lens Type | Focal Length | Use Case |
|-----------|-------------|----------|
| Wide angle | 6mm | Large parts, close distance |
| Standard | 12mm | General inspection |
| Telephoto | 25mm+ | Small parts, far distance |

**Recommended**: 12mm C-mount lens for typical PCB/small part inspection

### Working Distance Calculation

```
Working Distance = Focal Length × (Field of View / Sensor Size)
```

For a 12mm lens viewing a 100mm × 75mm area:
- Horizontal: 12mm × (100mm / 6.29mm) ≈ 190mm
- Working distance: ~19cm from lens to object

## Camera Configuration

### Resolution Modes

| Mode | Resolution | Aspect | FPS | Use Case |
|------|------------|--------|-----|----------|
| Full | 4056×3040 | 4:3 | 10 | Maximum detail |
| Half | 2028×1520 | 4:3 | 30 | Balanced |
| 1080p | 1920×1080 | 16:9 | 30 | Fast inspection |
| 720p | 1280×720 | 16:9 | 60 | High speed |

**Recommended**: 2028×1520 for balance of quality and speed

### Exposure Settings

For industrial lighting:

```yaml
camera:
  exposure:
    mode: manual
    exposure_time_us: 10000  # 10ms (adjust based on lighting)
    analogue_gain: 1.0       # Keep low to reduce noise
```

#### Exposure Guidelines

| Lighting Condition | Exposure Time | Gain |
|-------------------|---------------|------|
| Bright LED | 5,000μs | 1.0 |
| Standard LED | 10,000μs | 1.0 |
| Dim lighting | 20,000μs | 2.0 |
| Very dim | 50,000μs | 4.0 |

### White Balance

For consistent color under LED lighting:

```yaml
camera:
  white_balance:
    mode: manual
    red_gain: 1.5    # Adjust for your LED color temperature
    blue_gain: 1.5
```

To determine optimal values:
1. Capture with auto white balance
2. Note the values used
3. Configure as manual with those values

## Focus Adjustment

The HQ Camera uses **manual focus** via the lens focus ring.

### Focus Procedure

1. Place a reference object at the inspection position
2. Open live preview:
   ```bash
   libcamera-hello -t 0
   ```
3. Rotate the focus ring slowly
4. Stop when image is sharpest
5. Lock the focus ring (if equipped with locking screw)

### Focus Tips

- Use a focus target with fine lines or text
- Focus on the plane of the object surface
- Check focus at edges of field of view
- Re-check after thermal stabilization (10-15 min warm-up)

### Depth of Field

Depth of field depends on aperture:

| Aperture | DOF | Light | Note |
|----------|-----|-------|------|
| f/1.4 | Very shallow | Maximum | May need precise focus |
| f/2.8 | Moderate | Good | Recommended |
| f/5.6 | Deep | Adequate | May need more light |
| f/8.0 | Very deep | Low | Requires bright lighting |

For flat objects (PCBs, labels): f/2.8 is typically sufficient
For 3D objects: Use f/5.6 or smaller for adequate DOF

## Lighting Setup

### Lighting Types

| Type | Pros | Cons | Use Case |
|------|------|------|----------|
| Ring light | Even illumination | May cause glare | General inspection |
| Diffuse dome | No shadows | Bulky | Reflective surfaces |
| Bar lights | Directional | Shadows | Surface defects |
| Backlight | High contrast | Silhouette only | Dimension check |

### Lighting Recommendations

1. **Color**: White LEDs (5000-6000K) for natural color
2. **Intensity**: Adjustable brightness (PWM control)
3. **Uniformity**: <10% variation across field of view
4. **Stability**: Flicker-free (DC-driven LEDs)

### Reducing Glare

For reflective surfaces:
- Use polarizing filters on camera and lights
- Use diffuse dome lighting
- Angle lights at 45° to surface
- Use dark field (low-angle) illumination

## Calibration

### Geometric Calibration

For accurate measurements, calibrate with a checkerboard pattern:

```python
import cv2
import numpy as np

# Capture multiple checkerboard images at different angles
# Use cv2.calibrateCamera() to get camera matrix
```

### Color Calibration

For consistent color across sessions:

1. Capture a color reference chart
2. Calculate correction matrix
3. Apply to all captured images

## Troubleshooting

### "Camera not detected"

```bash
# Check if camera is recognized
libcamera-hello --list-cameras

# Check cable connection
# Try reseating the ribbon cable

# Check for kernel messages
dmesg | grep -i camera
```

### "Image is blurry"

- Check focus adjustment
- Increase shutter speed (reduce motion blur)
- Clean lens with microfiber cloth
- Check for vibration

### "Image is too dark/bright"

- Adjust exposure time
- Adjust analogue gain
- Check lighting intensity
- Verify aperture setting on lens

### "Colors are wrong"

- Adjust white balance gains
- Check LED color temperature
- Ensure consistent lighting

### "Image has noise"

- Reduce analogue gain
- Increase lighting
- Use longer exposure (if no motion)
- Ensure proper grounding (electrical noise)

## Sample Code

### Basic Capture

```python
from picamera2 import Picamera2

# Initialize
camera = Picamera2()
config = camera.create_still_configuration(
    main={"size": (2028, 1520)}
)
camera.configure(config)

# Start
camera.start()

# Manual settings
camera.set_controls({
    "ExposureTime": 10000,
    "AnalogueGain": 1.0,
    "AwbEnable": False,
    "ColourGains": (1.5, 1.5)
})

# Capture
camera.capture_file("test.jpg")
camera.stop()
```

### Continuous Preview

```python
from picamera2 import Picamera2
import time

camera = Picamera2()
camera.start_preview()
camera.start()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass

camera.stop()
```

### Capture with Metadata

```python
from picamera2 import Picamera2

camera = Picamera2()
camera.start()

# Capture with metadata
metadata = camera.capture_metadata()
print(f"Exposure: {metadata['ExposureTime']}μs")
print(f"Gain: {metadata['AnalogueGain']}")
print(f"Color gains: {metadata['ColourGains']}")

camera.stop()
```

## Maintenance

### Regular Checks

- [ ] Weekly: Clean lens with microfiber cloth
- [ ] Weekly: Check cable connections
- [ ] Monthly: Verify focus accuracy
- [ ] Monthly: Check lighting uniformity
- [ ] Quarterly: Recalibrate if needed

### Environment

- Operating temperature: 0-50°C
- Storage temperature: -20-60°C
- Humidity: 20-80% non-condensing
- Protect from dust and debris
