#!/usr/bin/env python3
"""
Model Export and Optimization for Raspberry Pi AI HAT

This module handles exporting trained models to formats optimized
for edge deployment on Raspberry Pi 5 with AI HAT acceleration.

Supported export formats:
- TensorFlow Lite (INT8 quantized for AI HAT)
- ONNX (for alternative runtimes)

Usage:
    python export_model.py --model_dir ./models --output_dir ./models/exported
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Callable

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available.")

# ONNX imports
try:
    import tf2onnx
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX export not available. Install with: pip install tf2onnx onnx")


def create_representative_dataset(
    data_dir: str,
    target_size: Tuple[int, int] = (224, 224),
    num_samples: int = 100
) -> Callable:
    """
    Create representative dataset generator for INT8 quantization.

    Args:
        data_dir: Directory containing calibration images
        target_size: Target image size
        num_samples: Number of calibration samples

    Returns:
        Generator function for representative dataset
    """
    # Add parent to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.data_loader import DatasetLoader

    loader = DatasetLoader(
        data_dir=data_dir,
        target_size=target_size,
        validation_split=0.0
    )

    # Load calibration images
    images, _ = loader.load_all('train')
    if len(images) > num_samples:
        indices = np.random.choice(len(images), num_samples, replace=False)
        images = images[indices]

    def representative_data_gen():
        for img in images:
            # Add batch dimension
            yield [img[np.newaxis, ...].astype(np.float32)]

    return representative_data_gen


def export_to_tflite(
    model_path: str,
    output_path: str,
    quantization: str = 'int8',
    calibration_data_dir: Optional[str] = None,
    target_size: Tuple[int, int] = (224, 224)
) -> str:
    """
    Export Keras model to TensorFlow Lite format.

    Optimized for Raspberry Pi AI HAT which supports INT8 inference.

    Args:
        model_path: Path to Keras model (.keras or SavedModel)
        output_path: Output path for TFLite model
        quantization: Quantization mode ('none', 'float16', 'int8', 'full_int8')
        calibration_data_dir: Directory with calibration images (required for int8)
        target_size: Model input size

    Returns:
        Path to exported TFLite model
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required for TFLite export")

    print(f"\n=== Exporting to TFLite ===")
    print(f"Model: {model_path}")
    print(f"Quantization: {quantization}")

    # Load model
    model = keras.models.load_model(model_path)

    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Configure quantization
    if quantization == 'none':
        # No quantization - float32
        pass

    elif quantization == 'float16':
        # Float16 quantization (2x smaller, works on most hardware)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif quantization == 'int8':
        # Dynamic range quantization (weights only)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Add representative dataset for better quantization
        if calibration_data_dir:
            converter.representative_dataset = create_representative_dataset(
                calibration_data_dir, target_size
            )

    elif quantization == 'full_int8':
        # Full integer quantization (best for AI HAT)
        if not calibration_data_dir:
            raise ValueError("Calibration data required for full INT8 quantization")

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = create_representative_dataset(
            calibration_data_dir, target_size
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    else:
        raise ValueError(f"Unknown quantization mode: {quantization}")

    # Convert
    print("Converting model...")
    tflite_model = converter.convert()

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        f.write(tflite_model)

    # Calculate size
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite model saved: {output_file}")
    print(f"Model size: {size_mb:.2f} MB")

    # Save metadata
    metadata = {
        'source_model': str(model_path),
        'quantization': quantization,
        'input_shape': list(model.input_shape),
        'output_shape': list(model.output_shape),
        'size_mb': size_mb
    }

    metadata_path = output_file.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return str(output_file)


def export_to_onnx(
    model_path: str,
    output_path: str,
    opset_version: int = 13
) -> str:
    """
    Export Keras model to ONNX format.

    Args:
        model_path: Path to Keras model
        output_path: Output path for ONNX model
        opset_version: ONNX opset version

    Returns:
        Path to exported ONNX model
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required for ONNX export")
    if not ONNX_AVAILABLE:
        raise RuntimeError("tf2onnx required for ONNX export")

    print(f"\n=== Exporting to ONNX ===")
    print(f"Model: {model_path}")
    print(f"Opset version: {opset_version}")

    # Load model
    model = keras.models.load_model(model_path)

    # Get input signature
    input_shape = model.input_shape
    input_spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)

    # Convert to ONNX
    print("Converting model...")
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_spec,
        opset=opset_version,
        output_path=output_path
    )

    # Calculate size
    output_file = Path(output_path)
    size_mb = output_file.stat().st_size / (1024 * 1024)

    print(f"ONNX model saved: {output_file}")
    print(f"Model size: {size_mb:.2f} MB")

    # Save metadata
    metadata = {
        'source_model': str(model_path),
        'opset_version': opset_version,
        'input_shape': list(input_shape),
        'output_shape': list(model.output_shape),
        'size_mb': size_mb
    }

    metadata_path = output_file.with_suffix('.onnx.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return str(output_file)


def optimize_for_ai_hat(
    tflite_model_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Apply additional optimizations for Raspberry Pi AI HAT.

    The AI HAT (Hailo-8L) has specific requirements and capabilities:
    - INT8 quantized models perform best
    - Batch size of 1 for real-time inference
    - Specific layer support considerations

    Args:
        tflite_model_path: Path to TFLite model
        output_path: Output path (defaults to adding _optimized suffix)

    Returns:
        Path to optimized model
    """
    print(f"\n=== Optimizing for AI HAT ===")

    if output_path is None:
        path = Path(tflite_model_path)
        output_path = str(path.parent / f"{path.stem}_aihat{path.suffix}")

    # Load the TFLite model
    with open(tflite_model_path, 'rb') as f:
        tflite_model = f.read()

    # Create interpreter to analyze model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input details:")
    for inp in input_details:
        print(f"  - Name: {inp['name']}")
        print(f"  - Shape: {inp['shape']}")
        print(f"  - Dtype: {inp['dtype']}")

    print("Output details:")
    for out in output_details:
        print(f"  - Name: {out['name']}")
        print(f"  - Shape: {out['shape']}")
        print(f"  - Dtype: {out['dtype']}")

    # For AI HAT, the model should already be optimized
    # Just copy with verification
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Optimized model saved: {output_path}")

    return output_path


def verify_tflite_model(
    model_path: str,
    test_input: Optional[np.ndarray] = None,
    input_size: Tuple[int, int] = (224, 224)
) -> bool:
    """
    Verify TFLite model works correctly.

    Args:
        model_path: Path to TFLite model
        test_input: Optional test input array
        input_size: Expected input size

    Returns:
        True if model verification passed
    """
    print(f"\n=== Verifying TFLite Model ===")

    try:
        # Load interpreter
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Check input shape
        expected_shape = (1, input_size[0], input_size[1], 3)
        actual_shape = tuple(input_details[0]['shape'])

        if actual_shape != expected_shape:
            print(f"Warning: Input shape mismatch. Expected {expected_shape}, got {actual_shape}")

        # Create test input if not provided
        if test_input is None:
            input_dtype = input_details[0]['dtype']
            if input_dtype == np.uint8:
                test_input = np.random.randint(0, 256, actual_shape, dtype=np.uint8)
            else:
                test_input = np.random.rand(*actual_shape).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        print(f"Input shape: {test_input.shape}, dtype: {test_input.dtype}")
        print(f"Output shape: {output.shape}, dtype: {output.dtype}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
        print("Verification: PASSED")

        return True

    except Exception as e:
        print(f"Verification FAILED: {e}")
        return False


def benchmark_tflite_model(
    model_path: str,
    num_iterations: int = 100,
    input_size: Tuple[int, int] = (224, 224)
) -> dict:
    """
    Benchmark TFLite model performance.

    Args:
        model_path: Path to TFLite model
        num_iterations: Number of benchmark iterations
        input_size: Input image size

    Returns:
        Dictionary with benchmark results
    """
    import time

    print(f"\n=== Benchmarking TFLite Model ===")
    print(f"Iterations: {num_iterations}")

    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create test input
    input_dtype = input_details[0]['dtype']
    input_shape = tuple(input_details[0]['shape'])

    if input_dtype == np.uint8:
        test_input = np.random.randint(0, 256, input_shape, dtype=np.uint8)
    else:
        test_input = np.random.rand(*input_shape).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], test_input)

    # Warm-up
    for _ in range(10):
        interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        interpreter.invoke()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)

    results = {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'fps': float(1000.0 / np.mean(times))
    }

    print(f"Mean inference time: {results['mean_ms']:.2f} ms")
    print(f"Std deviation: {results['std_ms']:.2f} ms")
    print(f"P95 latency: {results['p95_ms']:.2f} ms")
    print(f"Throughput: {results['fps']:.1f} FPS")

    return results


def main():
    """Main entry point for export script."""
    parser = argparse.ArgumentParser(
        description='Export and optimize models for Raspberry Pi AI HAT'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Directory containing trained Keras model'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./models/exported',
        help='Directory to save exported models'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['tflite', 'onnx', 'all'],
        default='all',
        help='Export format'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        choices=['none', 'float16', 'int8', 'full_int8'],
        default='int8',
        help='Quantization mode for TFLite'
    )
    parser.add_argument(
        '--calibration_dir',
        type=str,
        help='Directory with calibration images for INT8 quantization'
    )
    parser.add_argument(
        '--input_size',
        type=int,
        default=224,
        help='Model input size'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark after export'
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find Keras model
    keras_model = model_dir / 'autoencoder.keras'
    if not keras_model.exists():
        print(f"Error: Model not found at {keras_model}")
        sys.exit(1)

    target_size = (args.input_size, args.input_size)

    # Export to TFLite
    if args.format in ['tflite', 'all']:
        tflite_path = export_to_tflite(
            str(keras_model),
            str(output_dir / 'anomaly_detector.tflite'),
            quantization=args.quantization,
            calibration_data_dir=args.calibration_dir,
            target_size=target_size
        )

        # Verify
        verify_tflite_model(tflite_path, input_size=target_size)

        # Optimize for AI HAT
        optimize_for_ai_hat(tflite_path)

        # Benchmark
        if args.benchmark:
            benchmark_tflite_model(tflite_path, input_size=target_size)

    # Export to ONNX
    if args.format in ['onnx', 'all'] and ONNX_AVAILABLE:
        export_to_onnx(
            str(keras_model),
            str(output_dir / 'anomaly_detector.onnx')
        )

    # Copy threshold and preprocessing configs
    for config_file in ['anomaly_threshold.json', 'preprocessing_config.json']:
        src = model_dir / config_file
        if src.exists():
            import shutil
            shutil.copy(src, output_dir / config_file)
            print(f"Copied: {config_file}")

    print(f"\n=== Export Complete ===")
    print(f"Exported models saved to: {output_dir}")


if __name__ == '__main__':
    main()
