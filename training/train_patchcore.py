#!/usr/bin/env python3
"""
PatchCore Model Training Script (FAULTLESS)

Train a PatchCore anomaly detection model using WideResNet50 backbone.

Usage:
    python train_patchcore.py --class bottle --data Training
    python train_patchcore.py --class product --data /path/to/data --output models

Directory structure expected:
    Training/
    └── <class_name>/
        ├── train/
        │   └── good/
        │       ├── image1.png
        │       └── image2.png
        └── test/
            ├── good/
            │   └── normal_images...
            └── defect/
                └── anomaly_images...
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from inference.faultless import (
    create_model,
    train_model,
    evaluate_model,
    MVTecDataset,
    load_backbone_weights_into_patchcore,
    BACKBONE_WEIGHTS_PATH,
    IMAGE_SIZE,
    RESIZE_SIZE,
)

from utils.logging_utils import get_logger

logger = get_logger('train_patchcore')


def load_config(config_path: str = None) -> dict:
    """Load training configuration from YAML file."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    # Default configuration
    return {
        'patchcore': {
            'f_coreset': 0.1,
            'coreset_eps': 0.90,
            'backbone': 'wideresnet50',
        },
        'training': {
            'batch_size': 1,
        },
        'paths': {
            'backbone_weights': BACKBONE_WEIGHTS_PATH,
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train PatchCore anomaly detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--class', '-c',
        dest='class_name',
        type=str,
        required=True,
        help='Class name (subdirectory in data folder)'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='Training',
        help='Path to training data directory (default: Training)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='models',
        help='Output directory for trained model (default: models)'
    )
    parser.add_argument(
        '--backbone-weights', '-b',
        type=str,
        default=None,
        help='Path to backbone weights (optional)'
    )
    parser.add_argument(
        '--f-coreset', '-f',
        type=float,
        default=0.1,
        help='Coreset fraction (default: 0.1)'
    )
    parser.add_argument(
        '--coreset-eps', '-e',
        type=float,
        default=0.90,
        help='Random projection epsilon (default: 0.90)'
    )
    parser.add_argument(
        '--image-size', '-s',
        type=int,
        default=IMAGE_SIZE,
        help=f'Input image size (default: {IMAGE_SIZE})'
    )
    parser.add_argument(
        '--resize-size', '-r',
        type=int,
        default=RESIZE_SIZE,
        help=f'Resize size before crop (default: {RESIZE_SIZE})'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for training (default: 1)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML config file (optional)'
    )
    parser.add_argument(
        '--no-eval',
        action='store_true',
        help='Skip evaluation after training'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU training (no GPU)'
    )

    args = parser.parse_args()

    # Setup device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Using device: {device}")

    # Load configuration
    config = load_config(args.config)

    # Override config with CLI arguments
    f_coreset = args.f_coreset
    coreset_eps = args.coreset_eps

    # Paths
    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    class_name = args.class_name

    logger.info("=" * 60)
    logger.info("PatchCore Training")
    logger.info("=" * 60)
    logger.info(f"Class: {class_name}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"f_coreset: {f_coreset}")
    logger.info(f"coreset_eps: {coreset_eps}")
    logger.info(f"Image size: {args.image_size}")
    logger.info(f"Resize size: {args.resize_size}")
    logger.info("=" * 60)

    # Verify data directory exists
    class_path = data_path / class_name
    train_path = class_path / "train"

    if not train_path.exists():
        logger.error(f"Training directory not found: {train_path}")
        logger.error("Expected structure: <data>/<class>/train/good/")
        sys.exit(1)

    # Check for training images
    train_good_path = train_path / "good"
    if not train_good_path.exists():
        # Try without 'good' subdirectory
        if not any(train_path.glob("**/*.png")) and not any(train_path.glob("**/*.jpg")):
            logger.error(f"No training images found in: {train_path}")
            sys.exit(1)

    # Determine backbone weights path
    backbone_weights = args.backbone_weights
    if backbone_weights is None:
        # Try default locations
        for bp in [BACKBONE_WEIGHTS_PATH, 'Weights/AI_Faultless_weights_base.pth']:
            if os.path.exists(bp):
                backbone_weights = bp
                break

    if backbone_weights and os.path.exists(backbone_weights):
        logger.info(f"Using backbone weights: {backbone_weights}")
    else:
        logger.info("No backbone weights found - using randomly initialized backbone")
        backbone_weights = None

    # Create model
    logger.info("Creating PatchCore model...")
    model = create_model(
        f_coreset=f_coreset,
        backbone_name='wideresnet50',
        backbone_weights_path=backbone_weights,
        device=device
    )

    # Create dataset
    logger.info("Loading dataset...")
    try:
        dataset = MVTecDataset(
            cls=class_name,
            source=str(data_path),
            size=args.image_size,
            resize=args.resize_size
        )
        train_loader, test_loader = dataset.get_dataloaders(batch_size=args.batch_size)
        logger.info(f"Training samples: {len(dataset.train_ds)}")
        logger.info(f"Test samples: {len(dataset.test_ds)}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Train model
    logger.info("Starting training...")
    start_time = datetime.now()

    try:
        train_model(model, train_loader)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Training completed in {training_time:.1f} seconds")

    # Save model
    model_filename = f"patchcore_{class_name}.pth"
    model_path = output_path / model_filename

    logger.info(f"Saving model to: {model_path}")
    model.save(str(model_path))

    # Evaluate model
    if not args.no_eval and len(dataset.test_ds) > 0:
        logger.info("Evaluating model on test set...")
        try:
            result = evaluate_model(model, test_loader, class_name, model_dir=str(output_path))

            if result:
                roc_auc, threshold = result
                logger.info("=" * 60)
                logger.info("Evaluation Results")
                logger.info("=" * 60)
                logger.info(f"ROC-AUC: {roc_auc:.4f}")
                logger.info(f"Optimal Threshold: {threshold:.4f}")
                logger.info("=" * 60)

                # Save threshold
                import json
                threshold_path = output_path / f"patchcore_{class_name}_threshold.json"
                threshold_data = {
                    'model_name': f"patchcore_{class_name}",
                    'threshold': round(float(threshold), 4),
                    'roc_auc': round(float(roc_auc), 4),
                    'training_time_seconds': round(training_time, 2),
                    'training_samples': len(dataset.train_ds),
                    'test_samples': len(dataset.test_ds),
                    'f_coreset': f_coreset,
                    'coreset_eps': coreset_eps,
                    'image_size': args.image_size,
                    'device': str(device),
                }

                with open(threshold_path, 'w') as f:
                    json.dump(threshold_data, f, indent=4)

                logger.info(f"Saved threshold to: {threshold_path}")
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
    else:
        logger.info("Skipping evaluation (no test data or --no-eval flag)")

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Model saved to: {model_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
