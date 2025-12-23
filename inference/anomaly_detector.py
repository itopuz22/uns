#!/usr/bin/env python3
"""
PatchCore Anomaly Detection Engine

Industrial quality control using PatchCore algorithm with WideResNet50 backbone.
Optimized for both training (Windows/GPU) and inference (Raspberry Pi/CPU).

Usage:
    from inference.anomaly_detector import AnomalyDetector

    detector = AnomalyDetector('patchcore_model.pth', 'threshold.json')
    result = detector.detect('image.jpg')
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Re-export everything from faultless module
from inference.faultless import (
    # Core classes
    PatchCore,
    KNNExtractor,
    PatchCoreDetector,

    # Dataset classes
    MVTecDataset,
    MVTecTrainDataset,
    MVTecTestDataset,

    # Functions
    create_model,
    load_model,
    train_model,
    evaluate_model,
    run_inference,
    watch_folder,
    load_backbone_weights_into_patchcore,

    # Utilities
    get_coreset_idx,
    save_scores_csv,
    append_scores_csv,
    plot_roc,

    # Constants
    IMAGE_SIZE,
    RESIZE_SIZE,
    THRESHOLD,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

from utils.logging_utils import get_logger

logger = get_logger('anomaly_detector')

# Alias for backward compatibility
AnomalyDetector = PatchCoreDetector


class AnomalyDetectorDaemon:
    """
    Daemon service for PatchCore anomaly detection.

    Monitors IPC directory for analysis requests and produces results.
    """

    def __init__(
        self,
        model_path: str,
        ipc_dir: str = '/home/ai2/yolo_ipc',
        threshold_path: str = None
    ):
        """
        Initialize daemon.

        Args:
            model_path: Path to PatchCore model (.pth)
            ipc_dir: IPC directory for request/result files
            threshold_path: Path to threshold configuration
        """
        import json
        import time

        self.ipc_dir = Path(ipc_dir)
        self.ipc_dir.mkdir(parents=True, exist_ok=True)

        self.detector = PatchCoreDetector(
            model_path=model_path,
            threshold_path=threshold_path
        )

        self.running = False
        logger.info(f"Daemon initialized, watching: {self.ipc_dir}")

    def process_request(self, request_file: Path) -> None:
        """Process a single analysis request."""
        import json
        import time

        try:
            with open(request_file, 'r') as f:
                request = json.load(f)

            request_id = request.get('request_id')
            image_path = request.get('image_path')

            if not request_id or not image_path:
                logger.warning(f"Invalid request: {request_file}")
                return

            logger.info(f"Processing request: {request_id}")

            start_time = time.perf_counter()
            detection = self.detector.detect(image_path)
            processing_time = time.perf_counter() - start_time

            result = {
                'request_id': request_id,
                'result': 'OK' if detection['is_normal'] else 'NOK',
                'defect_count': 0 if detection['is_normal'] else 1,
                'confidence': detection['confidence'],
                'anomaly_score': detection['anomaly_score'],
                'threshold': detection['threshold'],
                'processing_time': processing_time,
                'inference_time': detection['inference_time_ms'] / 1000
            }

            if not detection['is_normal']:
                result['defects'] = {
                    'anomaly': {
                        'score': detection['anomaly_score'],
                    }
                }

            result_file = self.ipc_dir / f"result_{request_id}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)

            request_file.unlink(missing_ok=True)

            logger.info(f"Request {request_id} complete: {result['result']} ({processing_time:.2f}s)")

        except Exception as e:
            logger.error(f"Error processing request {request_file}: {e}")

    def run(self) -> None:
        """Run daemon main loop."""
        import time

        logger.info("Starting PatchCore anomaly detection daemon...")
        self.running = True

        try:
            while self.running:
                request_files = list(self.ipc_dir.glob('request_*.json'))

                for request_file in request_files:
                    self.process_request(request_file)

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Daemon interrupted")
        finally:
            self.running = False
            logger.info("Daemon stopped")

    def stop(self) -> None:
        """Stop daemon."""
        self.running = False


def create_detector(
    model_path: str,
    threshold_path: str = None,
    use_gpu: bool = True
) -> PatchCoreDetector:
    """
    Create PatchCore anomaly detector.

    Args:
        model_path: Path to PatchCore model (.pth)
        threshold_path: Path to threshold configuration
        use_gpu: Use GPU if available

    Returns:
        PatchCoreDetector instance
    """
    logger.info(f"Creating PatchCore detector from {model_path}")
    return PatchCoreDetector(
        model_path=model_path,
        threshold_path=threshold_path,
        use_gpu=use_gpu
    )


__all__ = [
    # Main classes
    'AnomalyDetector',
    'AnomalyDetectorDaemon',
    'PatchCore',
    'PatchCoreDetector',
    'KNNExtractor',

    # Dataset classes
    'MVTecDataset',
    'MVTecTrainDataset',
    'MVTecTestDataset',

    # Functions
    'create_detector',
    'create_model',
    'load_model',
    'train_model',
    'evaluate_model',
    'run_inference',
    'watch_folder',
]


def main():
    """Main entry point for daemon."""
    import argparse

    parser = argparse.ArgumentParser(description='PatchCore Anomaly Detection Daemon')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to PatchCore model (.pth)'
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
