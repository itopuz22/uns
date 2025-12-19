#!/usr/bin/env python3
"""
Anomaly Detection Inference Engine for Raspberry Pi

Optimized for Raspberry Pi 5 with AI HAT:
- TensorFlow Lite runtime with hardware acceleration
- Efficient image preprocessing
- Real-time anomaly scoring

Usage:
    from inference.anomaly_detector import AnomalyDetector

    detector = AnomalyDetector('model.tflite', 'threshold.json')
    result = detector.detect('image.jpg')
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_utils import load_image, preprocess_image, compute_reconstruction_error, create_heatmap
from utils.logging_utils import get_logger

logger = get_logger('anomaly_detector')

# TensorFlow Lite runtime
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_RUNTIME = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_RUNTIME = True
    except ImportError:
        TFLITE_RUNTIME = False
        logger.warning("TensorFlow Lite runtime not available")

# AI HAT delegate (Hailo)
try:
    # Hailo runtime for AI HAT acceleration
    from hailo_platform import (
        HailoRTException,
        ConfigureParams,
        InputVStreamParams,
        OutputVStreamParams,
        InferVStreams,
        HEF,
        Device
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    logger.info("Hailo runtime not available - using CPU inference")


class AnomalyDetector:
    """
    Anomaly detection using autoencoder reconstruction error.

    Supports:
    - TensorFlow Lite models (with optional AI HAT acceleration)
    - Configurable anomaly thresholds
    - Reconstruction error heatmap generation
    """

    def __init__(
        self,
        model_path: str,
        threshold_path: Optional[str] = None,
        preprocessing_config_path: Optional[str] = None,
        use_ai_hat: bool = True,
        num_threads: int = 4
    ):
        """
        Initialize anomaly detector.

        Args:
            model_path: Path to TFLite model
            threshold_path: Path to threshold configuration JSON
            preprocessing_config_path: Path to preprocessing config JSON
            use_ai_hat: Attempt to use AI HAT acceleration
            num_threads: Number of CPU threads for inference
        """
        self.model_path = Path(model_path)
        self.use_ai_hat = use_ai_hat and HAILO_AVAILABLE
        self.num_threads = num_threads

        # Load threshold configuration
        self.threshold = 0.05  # Default threshold
        self.threshold_config = {}
        if threshold_path and Path(threshold_path).exists():
            self._load_threshold_config(threshold_path)

        # Load preprocessing configuration
        self.input_size = (224, 224)
        self.normalization_mode = 'standard'
        if preprocessing_config_path and Path(preprocessing_config_path).exists():
            self._load_preprocessing_config(preprocessing_config_path)
        else:
            # Try to find preprocessing config next to model
            auto_config = self.model_path.parent / 'preprocessing_config.json'
            if auto_config.exists():
                self._load_preprocessing_config(str(auto_config))

        # Initialize interpreter
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        self._init_interpreter()

        # Statistics
        self.inference_count = 0
        self.total_inference_time = 0

    def _load_threshold_config(self, path: str) -> None:
        """Load anomaly threshold configuration."""
        try:
            with open(path, 'r') as f:
                self.threshold_config = json.load(f)
            self.threshold = self.threshold_config.get('threshold', 0.05)
            logger.info(f"Loaded threshold config: threshold={self.threshold}")
        except Exception as e:
            logger.warning(f"Failed to load threshold config: {e}")

    def _load_preprocessing_config(self, path: str) -> None:
        """Load preprocessing configuration."""
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            self.input_size = tuple(config.get('target_size', [224, 224]))
            self.normalization_mode = config.get('normalization_mode', 'standard')
            logger.info(f"Loaded preprocessing config: size={self.input_size}")
        except Exception as e:
            logger.warning(f"Failed to load preprocessing config: {e}")

    def _init_interpreter(self) -> None:
        """Initialize TensorFlow Lite interpreter."""
        if not TFLITE_RUNTIME:
            raise RuntimeError("TensorFlow Lite runtime not available")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info(f"Loading model: {self.model_path}")

        # Try AI HAT acceleration first
        if self.use_ai_hat:
            try:
                self._init_hailo()
                logger.info("Using AI HAT (Hailo) acceleration")
                return
            except Exception as e:
                logger.warning(f"AI HAT init failed, falling back to CPU: {e}")

        # Fallback to CPU
        try:
            self.interpreter = tflite.Interpreter(
                model_path=str(self.model_path),
                num_threads=self.num_threads
            )
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Log model info
            input_shape = self.input_details[0]['shape']
            input_dtype = self.input_details[0]['dtype']
            logger.info(f"Model loaded: input={input_shape}, dtype={input_dtype}")

            # Update input size from model
            if len(input_shape) == 4:
                self.input_size = (input_shape[2], input_shape[1])  # (width, height)

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _init_hailo(self) -> None:
        """Initialize Hailo AI HAT runtime."""
        if not HAILO_AVAILABLE:
            raise RuntimeError("Hailo runtime not available")

        # Find HEF file (Hailo Executable Format)
        hef_path = self.model_path.with_suffix('.hef')
        if not hef_path.exists():
            # Convert TFLite to HEF would happen during deployment
            raise FileNotFoundError(f"HEF model not found: {hef_path}")

        # Initialize Hailo device
        self.hailo_device = Device()
        self.hailo_hef = HEF(str(hef_path))

        # Configure network
        configure_params = ConfigureParams.create_from_hef(
            self.hailo_hef,
            interface=self.hailo_device
        )
        self.hailo_network = self.hailo_device.configure(
            self.hailo_hef,
            configure_params
        )[0]

        # Setup VStreams
        self.hailo_input_vstream_info = self.hailo_hef.get_input_vstream_infos()[0]
        self.hailo_output_vstream_info = self.hailo_hef.get_output_vstream_infos()[0]

    def preprocess(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for inference.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (preprocessed batch, original image)
        """
        # Load image
        original = load_image(image_path, color_mode='RGB')
        if original is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Preprocess
        processed = preprocess_image(
            original,
            target_size=self.input_size,
            normalize=True,
            normalization_mode=self.normalization_mode
        )

        # Add batch dimension
        batch = np.expand_dims(processed, axis=0)

        # Handle quantized models
        if self.input_details and self.input_details[0]['dtype'] == np.uint8:
            # Scale to 0-255 for uint8 models
            batch = (batch * 255).astype(np.uint8)

        return batch, original

    def infer(self, input_batch: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed input.

        Args:
            input_batch: Preprocessed image batch

        Returns:
            Model output (reconstructed image)
        """
        start_time = time.perf_counter()

        if hasattr(self, 'hailo_network') and self.hailo_network:
            # Hailo inference
            output = self._hailo_infer(input_batch)
        else:
            # TFLite inference
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                input_batch.astype(self.input_details[0]['dtype'])
            )
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Handle quantized output
        if output.dtype == np.uint8:
            output = output.astype(np.float32) / 255.0

        inference_time = (time.perf_counter() - start_time) * 1000
        self.inference_count += 1
        self.total_inference_time += inference_time

        return output

    def _hailo_infer(self, input_batch: np.ndarray) -> np.ndarray:
        """Run Hailo accelerated inference."""
        with InferVStreams(
            self.hailo_network,
            InputVStreamParams.make_from_network_group(self.hailo_network),
            OutputVStreamParams.make_from_network_group(self.hailo_network)
        ) as infer_pipeline:
            input_data = {self.hailo_input_vstream_info.name: input_batch}
            output_data = infer_pipeline.infer(input_data)
            return output_data[self.hailo_output_vstream_info.name]

    def detect(
        self,
        image_path: str,
        return_heatmap: bool = False
    ) -> Dict[str, Any]:
        """
        Detect anomalies in image.

        Args:
            image_path: Path to image file
            return_heatmap: Generate reconstruction error heatmap

        Returns:
            Detection result dictionary with:
            - is_normal: True if image is normal (no anomaly)
            - anomaly_score: Normalized anomaly score (0-1)
            - reconstruction_error: Raw reconstruction error
            - confidence: Detection confidence
            - inference_time_ms: Inference time in milliseconds
            - heatmap: Error heatmap (if requested)
        """
        result = {
            'is_normal': True,
            'anomaly_score': 0.0,
            'reconstruction_error': 0.0,
            'confidence': 1.0,
            'inference_time_ms': 0,
            'threshold': self.threshold,
            'image_path': image_path
        }

        try:
            # Preprocess
            input_batch, original = self.preprocess(image_path)

            # Store input for error calculation
            input_normalized = input_batch.astype(np.float32)
            if input_batch.dtype == np.uint8:
                input_normalized = input_normalized / 255.0

            # Run inference
            start_time = time.perf_counter()
            output = self.infer(input_batch)
            inference_time = (time.perf_counter() - start_time) * 1000

            result['inference_time_ms'] = inference_time

            # Compute reconstruction error
            error_value, error_map = compute_reconstruction_error(
                input_normalized[0],
                output[0],
                method='mse'
            )

            result['reconstruction_error'] = error_value

            # Normalize to anomaly score (0-1)
            # Higher error = higher anomaly score
            mean_error = self.threshold_config.get('mean_error', self.threshold / 2)
            std_error = self.threshold_config.get('std_error', self.threshold / 4)

            # Z-score based normalization
            z_score = (error_value - mean_error) / (std_error + 1e-8)
            anomaly_score = 1 / (1 + np.exp(-z_score))  # Sigmoid
            result['anomaly_score'] = float(anomaly_score)

            # Classification
            result['is_normal'] = error_value < self.threshold
            result['confidence'] = abs(error_value - self.threshold) / self.threshold
            result['confidence'] = min(1.0, result['confidence'])

            # Generate heatmap if requested
            if return_heatmap:
                try:
                    heatmap = create_heatmap(
                        error_map,
                        original_size=(original.shape[1], original.shape[0])
                    )
                    result['heatmap'] = heatmap
                except Exception as e:
                    logger.warning(f"Failed to create heatmap: {e}")

            logger.debug(
                f"Detection: error={error_value:.6f}, "
                f"threshold={self.threshold:.6f}, "
                f"normal={result['is_normal']}"
            )

        except Exception as e:
            logger.error(f"Detection error: {e}")
            result['error'] = str(e)
            result['is_normal'] = False
            result['confidence'] = 0.0

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        avg_time = 0
        if self.inference_count > 0:
            avg_time = self.total_inference_time / self.inference_count

        return {
            'inference_count': self.inference_count,
            'avg_inference_time_ms': avg_time,
            'total_inference_time_ms': self.total_inference_time,
            'using_ai_hat': hasattr(self, 'hailo_network') and self.hailo_network is not None,
            'input_size': self.input_size,
            'threshold': self.threshold
        }


class AnomalyDetectorDaemon:
    """
    Daemon service for anomaly detection.

    Monitors IPC directory for analysis requests and produces results.
    Compatible with the original automation.py IPC protocol.
    """

    def __init__(
        self,
        model_path: str,
        ipc_dir: str = '/home/ai2/yolo_ipc',
        threshold_path: Optional[str] = None
    ):
        """
        Initialize daemon.

        Args:
            model_path: Path to TFLite model
            ipc_dir: IPC directory for request/result files
            threshold_path: Path to threshold configuration
        """
        self.ipc_dir = Path(ipc_dir)
        self.ipc_dir.mkdir(parents=True, exist_ok=True)

        self.detector = AnomalyDetector(
            model_path=model_path,
            threshold_path=threshold_path
        )

        self.running = False
        logger.info(f"Daemon initialized, watching: {self.ipc_dir}")

    def process_request(self, request_file: Path) -> None:
        """Process a single analysis request."""
        try:
            # Read request
            with open(request_file, 'r') as f:
                request = json.load(f)

            request_id = request.get('request_id')
            image_path = request.get('image_path')

            if not request_id or not image_path:
                logger.warning(f"Invalid request: {request_file}")
                return

            logger.info(f"Processing request: {request_id}")

            # Run detection
            start_time = time.perf_counter()
            detection = self.detector.detect(image_path)
            processing_time = time.perf_counter() - start_time

            # Create result
            result = {
                'request_id': request_id,
                'result': 'OK' if detection['is_normal'] else 'NOK',
                'defect_count': 0 if detection['is_normal'] else 1,
                'confidence': detection['confidence'],
                'reconstruction_error': detection['reconstruction_error'],
                'threshold': detection['threshold'],
                'processing_time': processing_time,
                'inference_time': detection['inference_time_ms'] / 1000
            }

            if not detection['is_normal']:
                result['defects'] = {
                    'anomaly': {
                        'score': detection['anomaly_score'],
                        'error': detection['reconstruction_error']
                    }
                }

            # Write result
            result_file = self.ipc_dir / f"result_{request_id}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)

            # Remove request file
            request_file.unlink(missing_ok=True)

            logger.info(
                f"Request {request_id} complete: {result['result']} "
                f"({processing_time:.2f}s)"
            )

        except Exception as e:
            logger.error(f"Error processing request {request_file}: {e}")

    def run(self) -> None:
        """Run daemon main loop."""
        logger.info("Starting anomaly detection daemon...")
        self.running = True

        try:
            while self.running:
                # Find pending requests
                request_files = list(self.ipc_dir.glob('request_*.json'))

                for request_file in request_files:
                    self.process_request(request_file)

                # Short sleep to prevent CPU spinning
                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Daemon interrupted")
        finally:
            self.running = False
            logger.info("Daemon stopped")

    def stop(self) -> None:
        """Stop daemon."""
        self.running = False


def main():
    """Main entry point for daemon."""
    import argparse

    parser = argparse.ArgumentParser(description='Anomaly Detection Daemon')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to TFLite model'
    )
    parser.add_argument(
        '--ipc-dir',
        type=str,
        default='/home/ai2/yolo_ipc',
        help='IPC directory'
    )
    parser.add_argument(
        '--threshold',
        type=str,
        help='Path to threshold configuration'
    )

    args = parser.parse_args()

    daemon = AnomalyDetectorDaemon(
        model_path=args.model,
        ipc_dir=args.ipc_dir,
        threshold_path=args.threshold
    )

    daemon.run()


if __name__ == '__main__':
    main()
