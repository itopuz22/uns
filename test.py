#!/usr/bin/env python3
"""
Test Script for PatchCore Anomaly Detection Model

Tests images from a specified folder and classifies them as OK or NOK
based on the PatchCore anomaly score.

Usage:
    python test.py --model models/patchcore_model.pth --test_dir ./test_images
    python test.py --model models/patchcore_model.pth --test_dir ./test_images --threshold 0.5
    python test.py --model models/patchcore_model.pth --test_dir ./test_images --save_heatmaps
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np


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
        description='Test PatchCore anomaly detection model on images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test
  python test.py --model models/patchcore_model.pth --test_dir ./test_images

  # With custom threshold
  python test.py --model models/patchcore_model.pth --test_dir ./test_images --threshold 0.5

  # Save heatmap results
  python test.py --model models/patchcore_model.pth --test_dir ./test_images --save_heatmaps

  # Use CPU only
  python test.py --model models/patchcore_model.pth --test_dir ./test_images --cpu
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to PatchCore model (.pth file)'
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
        '--threshold_file',
        type=str,
        default=None,
        help='Path to threshold JSON file'
    )
    parser.add_argument(
        '--save_heatmaps',
        action='store_true',
        help='Save heatmap result images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory for result images (default: test_dir/results)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Show detailed output for each image'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU inference'
    )

    args = parser.parse_args()

    # Setup device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nUsing device: {device}")

    # Validate model file
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    # Find test images
    test_images = find_images(args.test_dir)
    if not test_images:
        print(f"ERROR: No images found in {args.test_dir}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("PATCHCORE ANOMALY DETECTION TEST")
    print("=" * 60)
    print(f"\nModel: {model_path}")
    print(f"Test directory: {args.test_dir}")
    print(f"Images found: {len(test_images)}")

    # Load threshold
    threshold = args.threshold

    if threshold is None:
        # Try to find threshold file
        threshold_file = args.threshold_file
        if threshold_file is None:
            # Look for threshold file next to model
            possible_threshold_files = [
                model_path.parent / f"{model_path.stem}_threshold.json",
                model_path.parent / "patchcore_threshold.json",
                model_path.parent / "threshold.json",
            ]
            for tf in possible_threshold_files:
                if tf.exists():
                    threshold_file = str(tf)
                    break

        if threshold_file and Path(threshold_file).exists():
            with open(threshold_file, 'r') as f:
                threshold_config = json.load(f)
            threshold = threshold_config.get('threshold', 0.0)
            print(f"Using saved threshold: {threshold:.4f}")
        else:
            threshold = 0.0
            print(f"Using default threshold: {threshold:.4f}")
    else:
        print(f"Using custom threshold: {threshold:.4f}")

    # Setup output directory
    output_dir = None
    if args.save_heatmaps:
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.test_dir) / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")

    print("\n" + "-" * 60)

    # Import PatchCore
    try:
        from inference.anomaly_detector import load_model, run_inference
    except ImportError as e:
        print(f"ERROR: Failed to import PatchCore module: {e}")
        sys.exit(1)

    # Load model
    print("\nLoading PatchCore model...")
    try:
        model = load_model(str(model_path), device)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)

    print("Model loaded successfully!")

    # Process each image
    results = []
    ok_count = 0
    nok_count = 0
    error_count = 0
    scores = []

    print(f"\nProcessing {len(test_images)} images...\n")

    for i, image_path in enumerate(test_images, 1):
        try:
            # Run inference
            score, name, label, heatmap_path = run_inference(
                model=model,
                image_path=str(image_path),
                device=device,
                save_dir=str(output_dir) if output_dir else str(Path(args.test_dir)),
                threshold=threshold,
                generate_heatmap=args.save_heatmaps
            )

            is_ok = label == "Normal"

            if is_ok:
                ok_count += 1
                result = 'OK'
            else:
                nok_count += 1
                result = 'NOK'

            scores.append(score)

            # Store result
            results.append({
                'image': image_path.name,
                'result': result,
                'score': float(score),
                'threshold': threshold,
                'heatmap': heatmap_path if args.save_heatmaps else None
            })

            # Print result
            status_color = '\033[92m' if is_ok else '\033[91m'  # Green for OK, Red for NOK
            reset_color = '\033[0m'

            if args.verbose:
                print(f"[{i:3d}/{len(test_images)}] {image_path.name}")
                print(f"         Result: {status_color}{result}{reset_color}")
                print(f"         Score: {score:.4f} (threshold: {threshold:.4f})")
                print()
            else:
                print(f"[{i:3d}/{len(test_images)}] {status_color}{result:3s}{reset_color} | "
                      f"Score: {score:.4f} | {image_path.name}")

        except Exception as e:
            print(f"[{i:3d}/{len(test_images)}] ERROR | {image_path.name}: {e}")
            error_count += 1
            results.append({
                'image': image_path.name,
                'result': 'ERROR',
                'score': None,
                'message': str(e)
            })

    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nTotal images tested: {len(test_images)}")
    print(f"  OK:    {ok_count:4d} ({ok_count / len(test_images) * 100:5.1f}%)")
    print(f"  NOK:   {nok_count:4d} ({nok_count / len(test_images) * 100:5.1f}%)")
    if error_count > 0:
        print(f"  Error: {error_count:4d} ({error_count / len(test_images) * 100:5.1f}%)")
    print(f"\nThreshold used: {threshold:.4f}")

    # Calculate score statistics
    if scores:
        scores_arr = np.array(scores)
        print(f"\nAnomaly Score Statistics:")
        print(f"  Min:    {scores_arr.min():.4f}")
        print(f"  Max:    {scores_arr.max():.4f}")
        print(f"  Mean:   {scores_arr.mean():.4f}")
        print(f"  Median: {np.median(scores_arr):.4f}")
        print(f"  Std:    {scores_arr.std():.4f}")

    # Save results to JSON
    results_file = Path(args.test_dir) / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': str(model_path),
            'test_dir': str(args.test_dir),
            'threshold': threshold,
            'device': str(device),
            'summary': {
                'total': len(test_images),
                'ok': ok_count,
                'nok': nok_count,
                'errors': error_count
            },
            'score_stats': {
                'min': float(min(scores)) if scores else None,
                'max': float(max(scores)) if scores else None,
                'mean': float(np.mean(scores)) if scores else None,
                'median': float(np.median(scores)) if scores else None,
                'std': float(np.std(scores)) if scores else None,
            },
            'results': results
        }, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")

    # List NOK images
    if nok_count > 0:
        print("\nNOK Images:")
        for r in results:
            if r['result'] == 'NOK':
                print(f"  - {r['image']} (score: {r['score']:.4f})")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
