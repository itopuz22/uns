"""
Camera Controller for Raspberry Pi HQ Camera (Sony IMX477)

Optimized for industrial quality control applications with:
- Full control over exposure, gain, and white balance
- Support for high-resolution capture (up to 4056x3040)
- Automatic reconnection on camera failure
- Configurable image quality settings
"""

import os
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

# Raspberry Pi camera imports
try:
    from picamera2 import Picamera2
    from picamera2.controls import Controls
    from libcamera import controls
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("Warning: picamera2 not available. Camera functionality disabled.")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_utils import get_logger

logger = get_logger('camera_controller')


class CameraController:
    """
    Controller for Raspberry Pi HQ Camera (Sony IMX477 12.3MP sensor).

    Features:
    - High-resolution capture optimized for industrial inspection
    - Manual control of exposure, gain, and white balance
    - Automatic camera reconnection
    - Thread-safe operation
    """

    # Sony IMX477 sensor specifications
    SENSOR_RESOLUTIONS = {
        'full': (4056, 3040),      # 12.3MP - Full resolution
        'high': (2028, 1520),       # 3MP - Half resolution
        'medium': (1920, 1080),     # 2MP - 1080p
        'low': (1280, 720),         # 0.9MP - 720p
        'preview': (640, 480),      # Preview resolution
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize camera controller.

        Args:
            config: Camera configuration dictionary
        """
        self.config = config or {}
        self.camera: Optional[Picamera2] = None
        self.is_started = False
        self.lock = threading.Lock()

        # Configuration with defaults
        self.capture_width = self.config.get('resolution', {}).get('capture_width', 2028)
        self.capture_height = self.config.get('resolution', {}).get('capture_height', 1520)
        self.preview_width = self.config.get('resolution', {}).get('preview_width', 640)
        self.preview_height = self.config.get('resolution', {}).get('preview_height', 480)

        # Exposure settings
        self.exposure_mode = self.config.get('exposure', {}).get('mode', 'auto')
        self.exposure_time = self.config.get('exposure', {}).get('exposure_time_us', 10000)
        self.analogue_gain = self.config.get('exposure', {}).get('analogue_gain', 1.0)

        # White balance
        self.wb_mode = self.config.get('white_balance', {}).get('mode', 'auto')
        self.wb_red_gain = self.config.get('white_balance', {}).get('red_gain', 1.5)
        self.wb_blue_gain = self.config.get('white_balance', {}).get('blue_gain', 1.5)

        # Quality settings
        self.jpeg_quality = self.config.get('quality', {}).get('jpeg_quality', 95)

        # Recovery settings
        self.warmup_time = self.config.get('warmup_time_seconds', 2.0)
        self.max_reconnect_attempts = self.config.get('reconnect_attempts', 3)
        self.reconnect_delay = self.config.get('reconnect_delay_seconds', 5.0)

        # Capture directory
        self.capture_dir = Path(self.config.get('capture_directory', '/tmp/qc_images'))
        self.capture_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.capture_count = 0
        self.last_capture_time = None
        self.failed_captures = 0

    def initialize(self) -> bool:
        """
        Initialize camera hardware.

        Returns:
            True if initialization successful
        """
        if not CAMERA_AVAILABLE:
            logger.error("Camera library not available")
            return False

        with self.lock:
            try:
                logger.info("Initializing Raspberry Pi HQ Camera...")

                # Create camera instance
                self.camera = Picamera2()

                # Create configuration for still capture
                # Main stream: high resolution for analysis
                # Lores stream: lower resolution for preview
                config = self.camera.create_still_configuration(
                    main={
                        "size": (self.capture_width, self.capture_height),
                        "format": "RGB888"
                    },
                    lores={
                        "size": (self.preview_width, self.preview_height),
                        "format": "YUV420"
                    },
                    display="lores"
                )

                self.camera.configure(config)

                logger.info(f"Camera configured: {self.capture_width}x{self.capture_height}")
                logger.info(f"Preview: {self.preview_width}x{self.preview_height}")

                return True

            except Exception as e:
                logger.error(f"Camera initialization failed: {e}")
                self.camera = None
                return False

    def start(self) -> bool:
        """
        Start camera streaming.

        Returns:
            True if camera started successfully
        """
        if not self.camera:
            if not self.initialize():
                return False

        with self.lock:
            if self.is_started:
                logger.debug("Camera already started")
                return True

            try:
                self.camera.start()
                self.is_started = True

                # Apply camera controls
                self._apply_controls()

                # Wait for camera warm-up
                logger.info(f"Waiting {self.warmup_time}s for camera warm-up...")
                time.sleep(self.warmup_time)

                logger.info("Camera started successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to start camera: {e}")
                self.is_started = False
                return False

    def stop(self) -> None:
        """Stop camera streaming."""
        with self.lock:
            if self.camera and self.is_started:
                try:
                    self.camera.stop()
                    logger.info("Camera stopped")
                except Exception as e:
                    logger.error(f"Error stopping camera: {e}")
                finally:
                    self.is_started = False

    def close(self) -> None:
        """Close camera and release resources."""
        self.stop()
        with self.lock:
            if self.camera:
                try:
                    self.camera.close()
                    logger.info("Camera closed")
                except Exception as e:
                    logger.error(f"Error closing camera: {e}")
                finally:
                    self.camera = None

    def _apply_controls(self) -> None:
        """Apply camera control settings."""
        if not self.camera or not self.is_started:
            return

        try:
            ctrl = {}

            # Exposure control
            if self.exposure_mode == 'manual':
                ctrl['ExposureTime'] = self.exposure_time
                ctrl['AnalogueGain'] = self.analogue_gain
                ctrl['AeEnable'] = False
            else:
                ctrl['AeEnable'] = True

            # White balance
            if self.wb_mode == 'manual':
                ctrl['AwbEnable'] = False
                ctrl['ColourGains'] = (self.wb_red_gain, self.wb_blue_gain)
            else:
                ctrl['AwbEnable'] = True

            self.camera.set_controls(ctrl)
            logger.debug(f"Camera controls applied: {ctrl}")

        except Exception as e:
            logger.warning(f"Failed to apply some camera controls: {e}")

    def capture(
        self,
        filename: Optional[str] = None,
        return_array: bool = False
    ) -> Optional[str]:
        """
        Capture an image.

        Args:
            filename: Optional filename (auto-generated if not provided)
            return_array: If True, also return image as numpy array

        Returns:
            Path to captured image file, or None on failure
        """
        if not self.is_started:
            if not self.start():
                return None

        with self.lock:
            try:
                # Generate filename if not provided
                if filename is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"capture_{timestamp}.jpg"

                filepath = self.capture_dir / filename

                # Capture image
                capture_start = time.perf_counter()

                if return_array:
                    # Capture to array and file
                    array = self.camera.capture_array("main")
                    self.camera.capture_file(str(filepath))
                else:
                    self.camera.capture_file(str(filepath))

                capture_time = (time.perf_counter() - capture_start) * 1000

                # Update statistics
                self.capture_count += 1
                self.last_capture_time = datetime.now()

                logger.info(f"Image captured: {filepath} ({capture_time:.1f}ms)")

                if return_array:
                    return str(filepath), array
                return str(filepath)

            except Exception as e:
                logger.error(f"Capture failed: {e}")
                self.failed_captures += 1

                # Attempt recovery
                if self._attempt_recovery():
                    # Retry capture once after recovery
                    return self.capture(filename, return_array)

                return None

    def capture_to_array(self) -> Optional[tuple]:
        """
        Capture image directly to numpy array.

        Returns:
            Tuple of (numpy array, metadata) or None on failure
        """
        if not self.is_started:
            if not self.start():
                return None

        with self.lock:
            try:
                # Capture to array
                array = self.camera.capture_array("main")
                metadata = self.camera.capture_metadata()

                self.capture_count += 1
                self.last_capture_time = datetime.now()

                return array, metadata

            except Exception as e:
                logger.error(f"Array capture failed: {e}")
                self.failed_captures += 1
                return None

    def _attempt_recovery(self) -> bool:
        """
        Attempt to recover from camera failure.

        Returns:
            True if recovery successful
        """
        logger.warning("Attempting camera recovery...")

        for attempt in range(self.max_reconnect_attempts):
            try:
                # Close and reinitialize
                self.is_started = False
                if self.camera:
                    try:
                        self.camera.stop()
                        self.camera.close()
                    except Exception:
                        pass
                    self.camera = None

                time.sleep(self.reconnect_delay)

                if self.initialize() and self.start():
                    logger.info(f"Camera recovery successful (attempt {attempt + 1})")
                    return True

            except Exception as e:
                logger.error(f"Recovery attempt {attempt + 1} failed: {e}")

            time.sleep(self.reconnect_delay)

        logger.error("Camera recovery failed after all attempts")
        return False

    def set_exposure(
        self,
        exposure_time_us: Optional[int] = None,
        gain: Optional[float] = None,
        auto: bool = False
    ) -> None:
        """
        Update exposure settings.

        Args:
            exposure_time_us: Exposure time in microseconds
            gain: Analogue gain
            auto: Enable auto exposure
        """
        if auto:
            self.exposure_mode = 'auto'
        else:
            self.exposure_mode = 'manual'
            if exposure_time_us is not None:
                self.exposure_time = exposure_time_us
            if gain is not None:
                self.analogue_gain = gain

        self._apply_controls()

    def set_white_balance(
        self,
        red_gain: Optional[float] = None,
        blue_gain: Optional[float] = None,
        auto: bool = False
    ) -> None:
        """
        Update white balance settings.

        Args:
            red_gain: Red channel gain
            blue_gain: Blue channel gain
            auto: Enable auto white balance
        """
        if auto:
            self.wb_mode = 'auto'
        else:
            self.wb_mode = 'manual'
            if red_gain is not None:
                self.wb_red_gain = red_gain
            if blue_gain is not None:
                self.wb_blue_gain = blue_gain

        self._apply_controls()

    def get_status(self) -> Dict[str, Any]:
        """
        Get camera status information.

        Returns:
            Dictionary with camera status
        """
        return {
            'available': CAMERA_AVAILABLE,
            'initialized': self.camera is not None,
            'started': self.is_started,
            'capture_resolution': (self.capture_width, self.capture_height),
            'preview_resolution': (self.preview_width, self.preview_height),
            'exposure_mode': self.exposure_mode,
            'capture_count': self.capture_count,
            'failed_captures': self.failed_captures,
            'last_capture': self.last_capture_time.isoformat() if self.last_capture_time else None
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
