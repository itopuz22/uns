#!/usr/bin/env python3
"""
Industrial Quality Control Automation System (PatchCore/FAULTLESS)

Main automation script for Raspberry Pi 5 with AI HAT and HQ Camera.
Performs real-time anomaly detection using PatchCore algorithm.

Features:
- PatchCore-based anomaly detection with WideResNet50 backbone
- Raspberry Pi HQ Camera optimization
- Real-time dashboard support
- Comprehensive error handling

Usage:
    python automation.py [--config config.yaml]
"""

import os
import sys
import json
import time
import signal
import argparse
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import configuration
import yaml
from config import get_config

# Import components
from deployment.camera_controller import CameraController
from deployment.gpio_controller import GPIOController
from deployment.ftp_uploader import FTPUploader
from utils.logging_utils import setup_logging, get_logger, AuditLogger, PerformanceTimer

# Initialize logger
logger = None

# Global state
shutdown_event = threading.Event()


class QualityControlSystem:
    """
    Main quality control automation system using PatchCore.

    Coordinates:
    - Camera capture
    - PatchCore anomaly detection
    - Relay control for OK/NOK signals
    - FTP image upload
    - Dashboard updates
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize quality control system.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = get_config(Path(config_path) if config_path else None)

        # Setup logging
        global logger
        log_config = self.config.get('logging.file', {})
        logger = setup_logging(
            name='qc_system',
            log_level=self.config.get('system.log_level', 'INFO'),
            log_dir=log_config.get('directory', '/var/log/qc_system'),
            colorized=True
        )

        # Initialize components
        self.camera: Optional[CameraController] = None
        self.gpio: Optional[GPIOController] = None
        self.ftp: Optional[FTPUploader] = None
        self.inference_engine = None

        # Audit logging
        audit_config = self.config.get('logging.audit', {})
        if audit_config.get('enabled', True):
            self.audit = AuditLogger(
                audit_config.get('file', '/var/log/qc_system/classifications.csv'),
                audit_config.get('fields', [
                    'timestamp', 'image_path', 'result',
                    'confidence', 'anomaly_score', 'processing_time_ms'
                ])
            )
        else:
            self.audit = None

        # IPC for external AI daemon
        self.ipc_dir = Path(self.config.get('ipc.directory', '/home/ai2/yolo_ipc'))
        self.ipc_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            'inspections': 0,
            'ok_count': 0,
            'nok_count': 0,
            'error_count': 0,
            'start_time': None,
            'last_inspection': None
        }

        logger.info("Quality Control System (PatchCore) initialized")

    def initialize(self) -> bool:
        """
        Initialize all system components.

        Returns:
            True if all components initialized successfully
        """
        logger.info("Initializing system components...")

        # Initialize camera
        try:
            self.camera = CameraController(self.config.camera)
            if not self.camera.initialize():
                logger.error("Camera initialization failed")
                return False
            logger.info("Camera initialized")
        except Exception as e:
            logger.error(f"Camera setup error: {e}")
            return False

        # Initialize GPIO
        try:
            self.gpio = GPIOController(self.config.gpio)
            if not self.gpio.initialize():
                logger.warning("GPIO initialization failed - continuing without GPIO")
                self.gpio = None
        except Exception as e:
            logger.warning(f"GPIO setup error: {e}")
            self.gpio = None

        # Initialize FTP uploader
        try:
            self.ftp = FTPUploader(self.config.ftp)
            if self.ftp.test_connection():
                self.ftp.start()
                logger.info("FTP uploader started")
            else:
                logger.warning("FTP connection failed - continuing without FTP")
        except Exception as e:
            logger.warning(f"FTP setup error: {e}")

        # Initialize PatchCore inference engine
        self._init_inference_engine()

        self.stats['start_time'] = datetime.now()
        logger.info("System initialization complete")
        return True

    def _init_inference_engine(self) -> None:
        """Initialize PatchCore inference engine."""
        try:
            from inference.anomaly_detector import create_detector

            model_config = self.config.model
            patchcore_config = model_config.get('patchcore', {})

            # Get model path
            model_path = Path('models') / model_config.get('paths', {}).get(
                'patchcore_model', 'patchcore_model.pth'
            )
            threshold_path = Path('models') / model_config.get('paths', {}).get(
                'patchcore_threshold', 'patchcore_threshold.json'
            )

            if model_path.exists():
                self.inference_engine = create_detector(
                    model_path=str(model_path),
                    threshold_path=str(threshold_path) if threshold_path.exists() else None,
                    use_gpu=model_config.get('inference', {}).get('use_gpu', True)
                )
                logger.info(f"PatchCore inference engine initialized: {model_path}")
            else:
                logger.warning(f"No model found at {model_path} - using IPC-based inference")

        except ImportError as e:
            logger.warning(f"PatchCore not available: {e}")
            logger.info("Using IPC-based inference instead")
        except Exception as e:
            logger.error(f"Inference engine init error: {e}")

    def shutdown(self) -> None:
        """Shutdown all system components."""
        logger.info("Shutting down system...")

        if self.gpio:
            self.gpio.cleanup()

        if self.camera:
            self.camera.close()

        if self.ftp:
            self.ftp.stop(wait=True)

        logger.info("System shutdown complete")

    def inspect(self) -> Dict[str, Any]:
        """
        Perform single inspection cycle.

        Returns:
            Inspection result dictionary
        """
        inspection_start = time.perf_counter()
        result = {
            'timestamp': datetime.now().isoformat(),
            'result': None,
            'confidence': 0.0,
            'image_path': None,
            'error': None,
            'processing_time_ms': 0
        }

        try:
            # Turn on light
            if self.gpio:
                self.gpio.light_on()
                time.sleep(self.gpio.k2_delay_ms / 1000.0)

            # Capture image
            with PerformanceTimer("capture", logger):
                image_path = self.camera.capture()

            # Turn off light
            if self.gpio:
                self.gpio.light_off()

            if not image_path:
                result['error'] = "Capture failed"
                result['result'] = 'ERROR'
                return result

            result['image_path'] = image_path

            # Perform analysis
            with PerformanceTimer("inference", logger):
                if self.inference_engine:
                    analysis = self._local_inference(image_path)
                else:
                    analysis = self._ipc_inference(image_path)

            result['result'] = analysis.get('result', 'ERROR')
            result['confidence'] = analysis.get('confidence', 0.0)

            # Signal result via relay
            if self.gpio and result['result'] in ['OK', 'NOK']:
                if result['result'] == 'OK':
                    self.gpio.signal_ok(blocking=False)
                else:
                    self.gpio.signal_nok(blocking=False)

            # Queue for FTP upload
            if self.ftp and image_path:
                self.ftp.upload_file(image_path)

                # Upload annotated result if exists
                annotated_path = image_path.replace('.jpg', '_result.jpg')
                if os.path.exists(annotated_path):
                    self.ftp.upload_file(annotated_path)

            # Update statistics
            self.stats['inspections'] += 1
            if result['result'] == 'OK':
                self.stats['ok_count'] += 1
            elif result['result'] == 'NOK':
                self.stats['nok_count'] += 1
            else:
                self.stats['error_count'] += 1
            self.stats['last_inspection'] = result['timestamp']

        except Exception as e:
            logger.error(f"Inspection error: {e}")
            result['error'] = str(e)
            result['result'] = 'ERROR'
            self.stats['error_count'] += 1

        # Calculate total processing time
        result['processing_time_ms'] = (time.perf_counter() - inspection_start) * 1000

        # Audit log
        if self.audit and result['result']:
            self.audit.log(
                timestamp=result['timestamp'],
                image_path=result.get('image_path', ''),
                result=result['result'],
                confidence=f"{result['confidence']:.4f}",
                processing_time_ms=f"{result['processing_time_ms']:.1f}"
            )

        logger.info(
            f"Inspection complete: {result['result']} "
            f"(confidence: {result['confidence']:.2%}, "
            f"time: {result['processing_time_ms']:.1f}ms)"
        )

        return result

    def _local_inference(self, image_path: str) -> Dict[str, Any]:
        """
        Perform local model inference.

        Args:
            image_path: Path to image file

        Returns:
            Analysis result dictionary
        """
        try:
            result = self.inference_engine.detect(image_path)
            return {
                'result': 'OK' if result['is_normal'] else 'NOK',
                'confidence': 1.0 - result['anomaly_score'],
                'reconstruction_error': result.get('reconstruction_error', 0),
                'processing_time': result.get('inference_time_ms', 0)
            }
        except Exception as e:
            logger.error(f"Local inference error: {e}")
            return {'result': 'ERROR', 'error': str(e)}

    def _ipc_inference(self, image_path: str) -> Dict[str, Any]:
        """
        Perform inference via IPC (inter-process communication).

        Compatible with existing yolo_ai_daemon service.

        Args:
            image_path: Path to image file

        Returns:
            Analysis result dictionary
        """
        import uuid

        request_id = str(uuid.uuid4())
        timeout = self.config.get('ipc.timeout_seconds', 120)

        # Create request file
        request_data = {
            'request_id': request_id,
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }

        request_file = self.ipc_dir / f"request_{request_id}.json"

        try:
            with open(request_file, 'w') as f:
                json.dump(request_data, f, indent=2)

            logger.debug(f"IPC request created: {request_id}")

            # Wait for result
            result_file = self.ipc_dir / f"result_{request_id}.json"
            start_time = time.time()

            while time.time() - start_time < timeout:
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            result_data = json.load(f)

                        # Cleanup
                        result_file.unlink(missing_ok=True)
                        request_file.unlink(missing_ok=True)

                        return {
                            'result': result_data.get('result', 'ERROR'),
                            'confidence': 1.0 - result_data.get('defect_count', 0) * 0.1,
                            'defect_count': result_data.get('defect_count', 0),
                            'processing_time': result_data.get('processing_time', 0)
                        }

                    except Exception as e:
                        logger.error(f"Error reading IPC result: {e}")
                        return {'result': 'ERROR', 'error': str(e)}

                time.sleep(0.1)

            # Timeout
            request_file.unlink(missing_ok=True)
            logger.error(f"IPC timeout after {timeout}s")
            return {'result': 'TIMEOUT', 'error': f'Timeout after {timeout}s'}

        except Exception as e:
            logger.error(f"IPC error: {e}")
            return {'result': 'ERROR', 'error': str(e)}

    def run_continuous(self) -> None:
        """
        Run continuous inspection loop.

        Waits for GPIO trigger signals and performs inspections.
        """
        logger.info("Starting continuous inspection mode...")

        if not self.gpio:
            logger.error("GPIO not available - cannot run in trigger mode")
            return

        # Start camera
        if not self.camera.start():
            logger.error("Failed to start camera")
            return

        logger.info("Ready. Waiting for trigger on GPIO 17...")

        try:
            while not shutdown_event.is_set():
                # Wait for trigger
                if self.gpio.wait_for_trigger(timeout=1.0):
                    logger.info("Trigger received - starting inspection")

                    # Perform inspection
                    result = self.inspect()

                    # Wait for trigger release before next cycle
                    self.gpio.wait_for_trigger_release(timeout=10.0)

                # Periodic maintenance
                if self.gpio:
                    self.gpio.check_relay_timeouts()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

    def run_single(self) -> Dict[str, Any]:
        """
        Run single inspection (for testing).

        Returns:
            Inspection result
        """
        if not self.camera.start():
            return {'result': 'ERROR', 'error': 'Camera start failed'}

        return self.inspect()

    def get_status(self) -> Dict[str, Any]:
        """
        Get system status.

        Returns:
            Status dictionary
        """
        uptime = None
        if self.stats['start_time']:
            uptime = (datetime.now() - self.stats['start_time']).total_seconds()

        return {
            'system': {
                'station_id': self.config.get('system.station_id', 'unknown'),
                'uptime_seconds': uptime,
                'version': self.config.get('system.version', '2.0.0')
            },
            'statistics': self.stats,
            'camera': self.camera.get_status() if self.camera else None,
            'gpio': self.gpio.get_status() if self.gpio else None,
            'ftp': self.ftp.get_stats() if self.ftp else None
        }


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}")
    shutdown_event.set()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Industrial Quality Control System'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--single',
        action='store_true',
        help='Run single inspection and exit'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Print system status and exit'
    )

    args = parser.parse_args()

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and initialize system
    system = QualityControlSystem(args.config)

    try:
        if not system.initialize():
            print("System initialization failed")
            sys.exit(1)

        if args.status:
            status = system.get_status()
            print(json.dumps(status, indent=2, default=str))
            sys.exit(0)

        if args.single:
            result = system.run_single()
            print(json.dumps(result, indent=2))
            sys.exit(0 if result['result'] != 'ERROR' else 1)

        # Run continuous mode
        system.run_continuous()

    except Exception as e:
        if logger:
            logger.error(f"Fatal error: {e}")
        else:
            print(f"Fatal error: {e}")
        sys.exit(1)

    finally:
        system.shutdown()


if __name__ == '__main__':
    main()
