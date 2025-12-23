#!/usr/bin/env python3
"""
PatchCore Model Training (FAULTLESS)

This module provides training for PatchCore anomaly detection models.
The autoencoder-based system has been replaced with PatchCore for better performance.

Usage:
    python train_autoencoder.py --data_dir ./Training --class_name product

    Or use the new script directly:
    python train_patchcore.py --class product --data ./Training
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import PatchCore training functions
from inference.faultless import (
    create_model,
    train_model,
    evaluate_model,
    MVTecDataset,
    IMAGE_SIZE,
    RESIZE_SIZE,
    BACKBONE_WEIGHTS_PATH,
)

import torch


def main():
    parser = argparse.ArgumentParser(
        description='Train PatchCore anomaly detection model (replaces autoencoder)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script has been updated to use PatchCore instead of autoencoder.
PatchCore provides better anomaly detection performance with:
- WideResNet50 backbone for feature extraction
- Memory bank with coreset sampling
- No training loop required (just feature extraction)

Examples:
    python train_autoencoder.py --data_dir ./Training --class_name bottle
    python train_autoencoder.py --data_dir ./Training --class_name product --f_coreset 0.1
        """
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='Training',
        help='Directory containing training data'
    )
    parser.add_argument(
        '--class_name',
        type=str,
        required=True,
        help='Class name (subdirectory in data_dir)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models',
        help='Output directory for trained model'
    )
    parser.add_argument(
        '--f_coreset',
        type=float,
        default=0.1,
        help='Coreset fraction (0.0-1.0, default: 0.1)'
    )
    parser.add_argument(
        '--coreset_eps',
        type=float,
        default=0.90,
        help='Random projection epsilon (default: 0.90)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=IMAGE_SIZE,
        help=f'Input image size (default: {IMAGE_SIZE})'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for training (default: 1)'
    )
    parser.add_argument(
        '--backbone_weights',
        type=str,
        default=None,
        help='Path to backbone weights (optional)'
    )
    parser.add_argument(
        '--no_eval',
        action='store_true',
        help='Skip evaluation after training'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU training'
    )

    # Legacy arguments (ignored but accepted for compatibility)
    parser.add_argument('--epochs', type=int, default=100, help='(Ignored - PatchCore does not use epochs)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='(Ignored - PatchCore does not use learning rate)')
    parser.add_argument('--latent_dim', type=int, default=128, help='(Ignored - PatchCore uses WideResNet50 features)')

    args = parser.parse_args()

    # Setup device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 60)
    print("PATCHCORE MODEL TRAINING (FAULTLESS)")
    print("=" * 60)
    print(f"\nDevice: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Class: {args.class_name}")
    print(f"Output: {args.output_dir}")
    print(f"Coreset fraction: {args.f_coreset}")
    print(f"Image size: {args.image_size}")
    print("=" * 60)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check data directory
    data_path = Path(args.data_dir)
    class_path = data_path / args.class_name
    train_path = class_path / "train"

    if not train_path.exists():
        print(f"\nERROR: Training directory not found: {train_path}")
        print("Expected structure:")
        print(f"  {args.data_dir}/")
        print(f"  └── {args.class_name}/")
        print("      ├── train/")
        print("      │   └── good/")
        print("      │       └── images...")
        print("      └── test/")
        print("          ├── good/")
        print("          └── defect/")
        sys.exit(1)

    # Find backbone weights
    backbone_weights = args.backbone_weights
    if backbone_weights is None:
        for bp in [BACKBONE_WEIGHTS_PATH, 'Weights/AI_Faultless_weights_base.pth']:
            if os.path.exists(bp):
                backbone_weights = bp
                break

    if backbone_weights and os.path.exists(backbone_weights):
        print(f"\nUsing backbone weights: {backbone_weights}")
    else:
        print("\nNo backbone weights found - using ImageNet pre-trained weights")
        backbone_weights = None

    # Create model
    print("\nCreating PatchCore model...")
    model = create_model(
        f_coreset=args.f_coreset,
        backbone_name='wideresnet50',
        backbone_weights_path=backbone_weights,
        device=device
    )

    # Load dataset
    print("\nLoading dataset...")
    try:
        dataset = MVTecDataset(
            cls=args.class_name,
            source=str(data_path),
            size=args.image_size,
            resize=args.image_size + 32  # Slightly larger for center crop
        )
        train_loader, test_loader = dataset.get_dataloaders(batch_size=args.batch_size)
        print(f"Training samples: {len(dataset.train_ds)}")
        print(f"Test samples: {len(dataset.test_ds)}")
    except Exception as e:
        print(f"\nERROR: Failed to load dataset: {e}")
        sys.exit(1)

    # Train model
    print("\n" + "-" * 60)
    print("Training PatchCore model...")
    print("-" * 60)

    from datetime import datetime
    start_time = datetime.now()

    try:
        train_model(model, train_loader)
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        sys.exit(1)

    training_time = (datetime.now() - start_time).total_seconds()
    print(f"\nTraining completed in {training_time:.1f} seconds")

    # Save model
    model_filename = f"patchcore_{args.class_name}.pth"
    model_path = output_path / model_filename

    print(f"\nSaving model to: {model_path}")
    model.save(str(model_path))

    # Evaluate if test data available
    if not args.no_eval and len(dataset.test_ds) > 0:
        print("\n" + "-" * 60)
        print("Evaluating model...")
        print("-" * 60)

        try:
            result = evaluate_model(model, test_loader, args.class_name, model_dir=str(output_path))

            if result:
                roc_auc, threshold = result
                print("\n" + "=" * 60)
                print("EVALUATION RESULTS")
                print("=" * 60)
                print(f"ROC-AUC: {roc_auc:.4f}")
                print(f"Optimal Threshold: {threshold:.4f}")
                print("=" * 60)

                # Save threshold
                import json
                threshold_path = output_path / f"patchcore_{args.class_name}_threshold.json"
                threshold_data = {
                    'model_name': f"patchcore_{args.class_name}",
                    'threshold': round(float(threshold), 4),
                    'roc_auc': round(float(roc_auc), 4),
                    'training_time_seconds': round(training_time, 2),
                    'training_samples': len(dataset.train_ds),
                    'f_coreset': args.f_coreset,
                    'image_size': args.image_size,
                }

                with open(threshold_path, 'w') as f:
                    json.dump(threshold_data, f, indent=4)

                print(f"\nThreshold saved to: {threshold_path}")
        except Exception as e:
            print(f"\nWarning: Evaluation failed: {e}")
    else:
        print("\nSkipping evaluation (no test data or --no_eval flag)")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Model saved to: {model_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
