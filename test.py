#!/usr/bin/env python3
"""
Test Script for Anomaly Detection Model

Tests images from a specified folder and classifies them as OK or NOK
based on the trained autoencoder's reconstruction error.

Usage:
    python test.py --model_dir ./models --test_dir ./test_images
    python test.py --model_dir ./models --test_dir ./test_images --show_heatmap
    python test.py --model_dir ./models --test_dir ./test_images --threshold 0.08
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def find_images(directory: str) -> list:
    """Find all image files in directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    images = []

    path = Path(directory)
    if not path.exists():
        print(f"ERROR: Directory not found: {directory}")
        sys.exit(1)

    for ext in extensions:
        images.extend(path.glob(f'*{ext}'))
        images.extend(path.glob(f'*{ext.upper()}'))

    return sorted(images)


def main():
    parser = argparse.ArgumentParser(
        description='Test anomaly detection model on images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test
  python test.py --model_dir ./models --test_dir ./test_images

  # With custom threshold
  python test.py --model_dir ./models --test_dir ./test_images --threshold 0.05

  # Save annotated results
  python test.py --model_dir ./models --test_dir ./test_images --save_results

  # Show heatmaps (requires display)
  python test.py --model_dir ./models --test_dir ./test_images --show_heatmap
        """
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Directory containing trained model'
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        required=True,
        help='Directory containing test images'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Custom anomaly threshold (default: use model threshold)'
    )
    parser.add_argument(
        '--save_results',
        action='store_true',
        help='Save annotated result images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory for result images (default: test_dir/results)'
    )
    parser.add_argument(
        '--show_heatmap',
        action='store_true',
        help='Display reconstruction error heatmaps'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Show detailed output for each image'
    )

    args = parser.parse_args()

    # Validate model directory
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)

    # Check for model files
    keras_model = model_dir / 'autoencoder.keras'
    tflite_model = model_dir / 'anomaly_detector.tflite'
    threshold_file = model_dir / 'anomaly_threshold.json'

    if not keras_model.exists() and not tflite_model.exists():
        print(f"ERROR: No model found in {model_dir}")
        print("Expected: autoencoder.keras or anomaly_detector.tflite")
        sys.exit(1)

    # Find test images
    test_images = find_images(args.test_dir)
    if not test_images:
        print(f"ERROR: No images found in {args.test_dir}")
        sys.exit(1)

    print("\n" + "="*60)
    print("ANOMALY DETECTION TEST")
    print("="*60)
    print(f"\nModel directory: {model_dir}")
    print(f"Test directory: {args.test_dir}")
    print(f"Images found: {len(test_images)}")

    # Load threshold
    threshold = args.threshold
    if threshold is None and threshold_file.exists():
        with open(threshold_file, 'r') as f:
            threshold_config = json.load(f)
        threshold = threshold_config.get('threshold', 0.05)
        print(f"Using saved threshold: {threshold:.6f}")
    elif threshold is None:
        threshold = 0.05
        print(f"Using default threshold: {threshold:.6f}")
    else:
        print(f"Using custom threshold: {threshold:.6f}")

    # Setup output directory
    if args.save_results:
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.test_dir) / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")

    print("\n" + "-"*60)

    # Try to use TFLite model first (faster), fall back to Keras
    use_tflite = tflite_model.exists()

    if use_tflite:
        print("Using TFLite model for inference...")
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                print("TFLite runtime not available, falling back to Keras...")
                use_tflite = False

    if use_tflite:
        # TFLite inference
        interpreter = tflite.Interpreter(model_path=str(tflite_model))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        input_size = (input_shape[2], input_shape[1])  # (width, height)
        input_dtype = input_details[0]['dtype']
    else:
        # Keras inference
        print("Using Keras model for inference...")
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            print("ERROR: TensorFlow required. Install with: pip install tensorflow")
            sys.exit(1)

        model = keras.models.load_model(str(keras_model))
        input_size = (model.input_shape[2], model.input_shape[1])

    # Import utilities
    import numpy as np
    from utils.image_utils import load_image, preprocess_image, compute_reconstruction_error

    try:
        import cv2
        CV2_AVAILABLE = True
    except ImportError:
        CV2_AVAILABLE = False

    # Process each image
    results = []
    ok_count = 0
    nok_count = 0
    error_count = 0

    print(f"\nProcessing {len(test_images)} images...\n")

    for i, image_path in enumerate(test_images, 1):
        try:
            # Load and preprocess image
            img = load_image(str(image_path), color_mode='RGB')
            if img is None:
                raise ValueError("Failed to load image")

            processed = preprocess_image(
                img,
                target_size=input_size,
                normalize=True,
                normalization_mode='standard'
            )

            # Add batch dimension
            batch = np.expand_dims(processed, axis=0)

            # Run inference
            if use_tflite:
                if input_dtype == np.uint8:
                    batch_input = (batch * 255).astype(np.uint8)
                else:
                    batch_input = batch.astype(np.float32)

                interpreter.set_tensor(input_details[0]['index'], batch_input)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])

                if output.dtype == np.uint8:
                    output = output.astype(np.float32) / 255.0
            else:
                output = model.predict(batch, verbose=0)

            # Compute reconstruction error
            error, error_map = compute_reconstruction_error(batch[0], output[0], method='mse')

            # Classify
            is_ok = error < threshold
            result = 'OK' if is_ok else 'NOK'

            if is_ok:
                ok_count += 1
            else:
                nok_count += 1

            # Store result
            results.append({
                'image': image_path.name,
                'result': result,
                'error': error,
                'threshold': threshold
            })

            # Print result
            status_color = '\033[92m' if is_ok else '\033[91m'  # Green for OK, Red for NOK
            reset_color = '\033[0m'

            if args.verbose:
                print(f"[{i:3d}/{len(test_images)}] {image_path.name}")
                print(f"         Result: {status_color}{result}{reset_color}")
                print(f"         Error: {error:.6f} (threshold: {threshold:.6f})")
                print()
            else:
                margin = abs(error - threshold) / threshold * 100
                print(f"[{i:3d}/{len(test_images)}] {status_color}{result:3s}{reset_color} | "
                      f"Error: {error:.6f} | {image_path.name}")

            # Save annotated result if requested
            if args.save_results and CV2_AVAILABLE:
                from utils.image_utils import add_result_annotation, create_heatmap, overlay_heatmap

                # Load original image in BGR for OpenCV
                img_bgr = cv2.imread(str(image_path))

                # Create heatmap
                heatmap = create_heatmap(error_map, (img_bgr.shape[1], img_bgr.shape[0]))

                # Overlay heatmap on original
                overlaid = overlay_heatmap(img_bgr, heatmap, alpha=0.3)

                # Add result annotation
                annotated = add_result_annotation(
                    overlaid,
                    result=result,
                    confidence=1.0 - min(error / threshold, 1.0) if is_ok else min(error / threshold - 1.0, 1.0),
                    processing_time_ms=0
                )

                # Save
                result_path = output_dir / f"{image_path.stem}_result{image_path.suffix}"
                cv2.imwrite(str(result_path), annotated)

            # Show heatmap if requested
            if args.show_heatmap and CV2_AVAILABLE:
                from utils.image_utils import create_heatmap, overlay_heatmap

                img_bgr = cv2.imread(str(image_path))
                heatmap = create_heatmap(error_map, (img_bgr.shape[1], img_bgr.shape[0]))
                overlaid = overlay_heatmap(img_bgr, heatmap, alpha=0.4)

                # Add text
                color = (0, 255, 0) if is_ok else (0, 0, 255)
                cv2.putText(overlaid, f"{result} - Error: {error:.4f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                cv2.imshow('Anomaly Detection Result', overlaid)
                key = cv2.waitKey(0)
                if key == 27:  # ESC to quit
                    break

        except Exception as e:
            print(f"[{i:3d}/{len(test_images)}] ERROR | {image_path.name}: {e}")
            error_count += 1
            results.append({
                'image': image_path.name,
                'result': 'ERROR',
                'error': None,
                'message': str(e)
            })

    if args.show_heatmap and CV2_AVAILABLE:
        cv2.destroyAllWindows()

    # Print summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    print(f"\nTotal images tested: {len(test_images)}")
    print(f"  OK:    {ok_count:4d} ({ok_count/len(test_images)*100:5.1f}%)")
    print(f"  NOK:   {nok_count:4d} ({nok_count/len(test_images)*100:5.1f}%)")
    if error_count > 0:
        print(f"  Error: {error_count:4d} ({error_count/len(test_images)*100:5.1f}%)")
    print(f"\nThreshold used: {threshold:.6f}")

    # Calculate error statistics
    valid_errors = [r['error'] for r in results if r['error'] is not None]
    if valid_errors:
        import numpy as np
        errors = np.array(valid_errors)
        print(f"\nReconstruction Error Statistics:")
        print(f"  Min:    {errors.min():.6f}")
        print(f"  Max:    {errors.max():.6f}")
        print(f"  Mean:   {errors.mean():.6f}")
        print(f"  Median: {np.median(errors):.6f}")
        print(f"  Std:    {errors.std():.6f}")

    # Save results to JSON
    results_file = Path(args.test_dir) / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model_dir': str(model_dir),
            'test_dir': str(args.test_dir),
            'threshold': threshold,
            'summary': {
                'total': len(test_images),
                'ok': ok_count,
                'nok': nok_count,
                'errors': error_count
            },
            'results': results
        }, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")

    # List NOK images
    if nok_count > 0:
        print("\nNOK Images:")
        for r in results:
            if r['result'] == 'NOK':
                print(f"  - {r['image']} (error: {r['error']:.6f})")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
