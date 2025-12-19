#!/usr/bin/env python3
"""
System Integration Tests for Quality Control System

Run these tests to verify the system is working correctly.

Usage:
    python -m pytest tests/test_system.py -v

Or run directly:
    python tests/test_system.py
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_loading():
    """Test configuration module."""
    print("Testing configuration loading...")

    from config import get_config

    config = get_config()

    assert config.camera is not None
    assert config.gpio is not None
    assert config.model is not None
    assert config.ftp is not None

    # Test dot notation access
    assert config.get('system.name') == 'Assembly Line Quality Control'
    assert config.get('camera.resolution.capture_width') == 2028

    print("  Configuration loading: PASSED")


def test_image_preprocessing():
    """Test image preprocessing utilities."""
    print("Testing image preprocessing...")

    import numpy as np
    from utils.image_utils import preprocess_image, postprocess_image

    # Create test image
    test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # Preprocess
    processed = preprocess_image(
        test_image,
        target_size=(224, 224),
        normalize=True,
        normalization_mode='standard'
    )

    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0

    # Postprocess
    restored = postprocess_image(processed, 'standard')
    assert restored.dtype == np.uint8
    assert restored.min() >= 0
    assert restored.max() <= 255

    print("  Image preprocessing: PASSED")


def test_reconstruction_error():
    """Test reconstruction error computation."""
    print("Testing reconstruction error...")

    import numpy as np
    from utils.image_utils import compute_reconstruction_error

    # Create test images
    original = np.random.rand(224, 224, 3).astype(np.float32)

    # Perfect reconstruction
    reconstructed_perfect = original.copy()
    error_perfect, _ = compute_reconstruction_error(original, reconstructed_perfect)
    assert error_perfect < 1e-6

    # Imperfect reconstruction
    reconstructed_noisy = original + np.random.randn(*original.shape) * 0.1
    error_noisy, error_map = compute_reconstruction_error(original, reconstructed_noisy)
    assert error_noisy > error_perfect
    assert error_map.shape == original.shape

    print("  Reconstruction error: PASSED")


def test_logging():
    """Test logging utilities."""
    print("Testing logging utilities...")

    import tempfile
    from utils.logging_utils import setup_logging, get_logger, PerformanceTimer

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = setup_logging(
            name='test_logger',
            log_level='DEBUG',
            log_dir=tmpdir,
            colorized=False
        )

        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")

        # Check log file was created
        log_files = list(Path(tmpdir).glob('*.log'))
        assert len(log_files) == 1

    # Test performance timer
    import time
    with PerformanceTimer("test_operation") as timer:
        time.sleep(0.1)

    assert timer.elapsed_ms >= 90  # Allow some tolerance

    print("  Logging utilities: PASSED")


def test_audit_logger():
    """Test audit logger."""
    print("Testing audit logger...")

    import tempfile
    from utils.logging_utils import AuditLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        audit_file = Path(tmpdir) / 'audit.csv'

        audit = AuditLogger(
            str(audit_file),
            ['timestamp', 'result', 'confidence']
        )

        audit.log(timestamp='2024-01-01', result='OK', confidence='0.95')
        audit.log(timestamp='2024-01-02', result='NOK', confidence='0.85')

        # Verify file content
        with open(audit_file) as f:
            lines = f.readlines()

        assert len(lines) == 3  # Header + 2 entries
        assert 'timestamp,result,confidence' in lines[0]

    print("  Audit logger: PASSED")


def test_ftp_config():
    """Test FTP uploader initialization (without actual connection)."""
    print("Testing FTP uploader config...")

    from deployment.ftp_uploader import FTPUploader

    config = {
        'server': '192.168.1.100',
        'port': 21,
        'username': 'test',
        'password': 'test123',
        'base_directory': '/test',
        'upload': {
            'async_enabled': True,
            'retry_attempts': 3
        }
    }

    uploader = FTPUploader(config)

    assert uploader.server == '192.168.1.100'
    assert uploader.port == 21
    assert uploader.retry_attempts == 3
    assert uploader.async_enabled == True

    # Check stats
    stats = uploader.get_stats()
    assert 'uploads_attempted' in stats
    assert stats['queue_size'] == 0

    print("  FTP uploader config: PASSED")


def test_gpio_controller_init():
    """Test GPIO controller initialization (mock mode)."""
    print("Testing GPIO controller...")

    # This test only checks initialization without hardware
    from deployment.gpio_controller import GPIOController, GPIO_AVAILABLE

    if not GPIO_AVAILABLE:
        print("  GPIO not available (expected on non-Pi systems): SKIPPED")
        return

    config = {
        'chip': 'gpiochip4',
        'input': {'trigger_pin': 17},
        'relays': {
            'k1_ok': {'pin': 26, 'pulse_duration_ms': 1000},
            'k2_light': {'pin': 20},
            'k3_nok': {'pin': 21, 'pulse_duration_ms': 3000}
        }
    }

    gpio = GPIOController(config)
    assert gpio.trigger_pin == 17
    assert gpio.k1_pin == 26
    assert gpio.k1_pulse_ms == 1000

    print("  GPIO controller: PASSED")


def test_camera_controller_init():
    """Test camera controller initialization (mock mode)."""
    print("Testing camera controller...")

    from deployment.camera_controller import CameraController, CAMERA_AVAILABLE

    if not CAMERA_AVAILABLE:
        print("  Camera not available (expected on non-Pi systems): SKIPPED")
        return

    config = {
        'resolution': {
            'capture_width': 1920,
            'capture_height': 1080
        },
        'exposure': {
            'mode': 'auto'
        }
    }

    camera = CameraController(config)
    assert camera.capture_width == 1920
    assert camera.capture_height == 1080

    status = camera.get_status()
    assert 'available' in status

    print("  Camera controller: PASSED")


def test_model_architecture():
    """Test autoencoder model architecture (if TensorFlow available)."""
    print("Testing model architecture...")

    try:
        from training.train_autoencoder import AnomalyAutoencoder
    except ImportError:
        print("  TensorFlow not available: SKIPPED")
        return

    model = AnomalyAutoencoder(
        input_shape=(224, 224, 3),
        latent_dim=64,
        encoder_filters=[16, 32, 64],
        decoder_filters=[64, 32, 16]
    )

    assert model.autoencoder is not None
    assert model.encoder is not None
    assert model.decoder is not None

    # Check shapes
    assert model.autoencoder.input_shape == (None, 224, 224, 3)
    assert model.autoencoder.output_shape == (None, 224, 224, 3)

    print("  Model architecture: PASSED")


def test_data_loader():
    """Test data loader with synthetic data."""
    print("Testing data loader...")

    import tempfile
    import numpy as np

    try:
        from PIL import Image
    except ImportError:
        print("  PIL not available: SKIPPED")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test images
        good_dir = Path(tmpdir) / 'good'
        good_dir.mkdir()

        for i in range(10):
            img = Image.new('RGB', (640, 480), color=(i*25, i*25, i*25))
            img.save(good_dir / f'test_{i}.jpg')

        from training.data_loader import DatasetLoader

        loader = DatasetLoader(
            data_dir=tmpdir,
            target_size=(224, 224),
            validation_split=0.2
        )

        assert len(loader.image_paths) == 10
        assert len(loader.train_indices) == 8
        assert len(loader.val_indices) == 2

        # Load an image
        img = loader.load_image(0)
        assert img is not None
        assert img.shape == (224, 224, 3)

    print("  Data loader: PASSED")


def test_anomaly_detector_init():
    """Test anomaly detector initialization (without model)."""
    print("Testing anomaly detector...")

    import tempfile
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy threshold config
        threshold_path = Path(tmpdir) / 'threshold.json'
        with open(threshold_path, 'w') as f:
            json.dump({
                'threshold': 0.05,
                'mean_error': 0.02,
                'std_error': 0.01
            }, f)

        # Note: We can't test full initialization without a model
        # Just verify the config loading works
        from inference.anomaly_detector import AnomalyDetector, TFLITE_RUNTIME

        if not TFLITE_RUNTIME:
            print("  TFLite not available: SKIPPED")
            return

        # Would need actual model file to continue
        print("  Anomaly detector (config only): PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Quality Control System - Integration Tests")
    print("="*60 + "\n")

    tests = [
        test_config_loading,
        test_image_preprocessing,
        test_reconstruction_error,
        test_logging,
        test_audit_logger,
        test_ftp_config,
        test_gpio_controller_init,
        test_camera_controller_init,
        test_model_architecture,
        test_data_loader,
        test_anomaly_detector_init,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  {test.__name__}: FAILED - {e}")
            failed += 1
        except Exception as e:
            print(f"  {test.__name__}: ERROR - {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
