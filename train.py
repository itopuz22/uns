#!/usr/bin/env python3
"""
Training Script for Anomaly Detection Model

This script trains an autoencoder-based anomaly detection model
for industrial quality control.

Usage:
    # Train with real data
    python train.py --data_dir ./data/good --output_dir ./models

    # Train with synthetic data (for testing)
    python train.py --synthetic --output_dir ./models

    # Quick test with minimal epochs
    python train.py --synthetic --epochs 5 --output_dir ./models
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def create_synthetic_dataset(output_dir: str, num_images: int = 100) -> str:
    """
    Create synthetic training data for testing.

    Generates simple geometric patterns that the autoencoder can learn.

    Args:
        output_dir: Directory to save synthetic images
        num_images: Number of images to generate

    Returns:
        Path to the good samples directory
    """
    import numpy as np

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("PIL required for synthetic data. Install with: pip install Pillow")
        sys.exit(1)

    good_dir = Path(output_dir) / 'good'
    good_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_images} synthetic training images...")

    for i in range(num_images):
        # Create base image with consistent background
        img = Image.new('RGB', (256, 256), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        # Add consistent geometric pattern (simulating a "good" product)
        # Central circle
        center_x, center_y = 128, 128
        radius = 60
        draw.ellipse(
            [center_x - radius, center_y - radius,
             center_x + radius, center_y + radius],
            fill=(100, 150, 200),
            outline=(50, 100, 150),
            width=3
        )

        # Add some rectangles (simulating components)
        draw.rectangle([40, 40, 80, 80], fill=(200, 100, 100), outline=(150, 50, 50))
        draw.rectangle([176, 40, 216, 80], fill=(100, 200, 100), outline=(50, 150, 50))
        draw.rectangle([40, 176, 80, 216], fill=(100, 100, 200), outline=(50, 50, 150))
        draw.rectangle([176, 176, 216, 216], fill=(200, 200, 100), outline=(150, 150, 50))

        # Add small variations (simulating normal manufacturing tolerance)
        variation = np.random.uniform(-5, 5, 2).astype(int)
        draw.ellipse(
            [108 + variation[0], 108 + variation[1],
             148 + variation[0], 148 + variation[1]],
            fill=(150, 180, 220)
        )

        # Add slight noise
        img_array = np.array(img)
        noise = np.random.normal(0, 3, img_array.shape).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)

        # Save image
        img.save(good_dir / f'sample_{i:04d}.jpg', quality=95)

    print(f"Synthetic dataset created at: {good_dir}")
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Train anomaly detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with real data
  python train.py --data_dir ./data --output_dir ./models

  # Train with synthetic data for testing
  python train.py --synthetic --output_dir ./models

  # Quick test run
  python train.py --synthetic --epochs 5 --batch_size 16
        """
    )

    # Data options
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Directory containing training data (with "good" subdirectory)'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Generate and use synthetic data for testing'
    )
    parser.add_argument(
        '--num_synthetic',
        type=int,
        default=200,
        help='Number of synthetic images to generate (default: 200)'
    )

    # Output options
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./models',
        help='Directory to save trained model (default: ./models)'
    )

    # Model architecture
    parser.add_argument(
        '--input_size',
        type=int,
        default=224,
        help='Input image size (default: 224)'
    )
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=128,
        help='Latent space dimension (default: 128)'
    )

    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--validation_split',
        type=float,
        default=0.2,
        help='Validation split ratio (default: 0.2)'
    )

    # Hardware options
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device index, -1 for CPU (default: 0)'
    )

    # Export options
    parser.add_argument(
        '--export_tflite',
        action='store_true',
        help='Export model to TFLite format after training'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        choices=['none', 'float16', 'int8'],
        default='int8',
        help='Quantization mode for TFLite export (default: int8)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.synthetic and not args.data_dir:
        parser.error("Either --data_dir or --synthetic must be specified")

    # Check TensorFlow availability
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")

        # Configure GPU
        gpus = tf.config.list_physical_devices('GPU')
        if args.gpu >= 0 and gpus:
            try:
                tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
                print(f"Using GPU: {gpus[args.gpu]}")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            tf.config.set_visible_devices([], 'GPU')
            print("Using CPU for training")

    except ImportError:
        print("ERROR: TensorFlow is required for training.")
        print("Install with: pip install tensorflow")
        sys.exit(1)

    # Prepare data directory
    if args.synthetic:
        # Create synthetic data in temp directory or output directory
        data_dir = create_synthetic_dataset(
            os.path.join(args.output_dir, 'synthetic_data'),
            args.num_synthetic
        )
    else:
        data_dir = args.data_dir

        # Verify data directory exists
        good_dir = Path(data_dir) / 'good'
        if not good_dir.exists():
            # Check if images are directly in data_dir
            image_count = len(list(Path(data_dir).glob('*.jpg'))) + \
                         len(list(Path(data_dir).glob('*.png')))
            if image_count == 0:
                print(f"ERROR: No images found in {data_dir}")
                print("Expected structure: data_dir/good/*.jpg")
                sys.exit(1)

    # Training configuration
    config = {
        'input_size': (args.input_size, args.input_size),
        'latent_dim': args.latent_dim,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'validation_split': args.validation_split,
        'early_stopping_patience': max(5, args.epochs // 10)
    }

    print("\n" + "="*60)
    print("ANOMALY DETECTION MODEL TRAINING")
    print("="*60)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Import and run training
    from training.train_autoencoder import train_model

    try:
        model, threshold_info = train_model(
            data_dir=data_dir,
            output_dir=args.output_dir,
            config=config
        )

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"\nModel saved to: {args.output_dir}")
        print(f"Anomaly threshold: {threshold_info['threshold']:.6f}")
        print(f"Mean reconstruction error: {threshold_info['mean_error']:.6f}")

        # Export to TFLite if requested
        if args.export_tflite:
            print("\n" + "="*60)
            print("EXPORTING TO TFLITE")
            print("="*60)

            from training.export_model import export_to_tflite, verify_tflite_model

            tflite_path = export_to_tflite(
                model_path=os.path.join(args.output_dir, 'autoencoder.keras'),
                output_path=os.path.join(args.output_dir, 'anomaly_detector.tflite'),
                quantization=args.quantization,
                calibration_data_dir=data_dir if args.quantization == 'int8' else None,
                target_size=config['input_size']
            )

            verify_tflite_model(tflite_path, input_size=config['input_size'])

            print(f"\nTFLite model saved to: {tflite_path}")

        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print("\nNext steps:")
        print("1. Review training_history.json for training metrics")
        print("2. Test the model with sample images")
        print("3. Adjust threshold if needed based on false positive/negative rates")
        if not args.export_tflite:
            print("4. Export to TFLite: python train.py --export_tflite ...")
        print()

    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
