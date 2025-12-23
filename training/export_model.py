#!/usr/bin/env python3
"""
PatchCore Model Export (FAULTLESS)

PatchCore models are saved directly as PyTorch .pth files and don't require
conversion to TFLite or ONNX formats.

The model can be deployed directly on Raspberry Pi using CPU-based PyTorch inference.

Usage:
    # Models are already in deployable format after training
    # Just copy the .pth file to the deployment target

    # To verify model:
    python export_model.py --model models/patchcore_product.pth --verify

    # To create a lightweight model info file:
    python export_model.py --model models/patchcore_product.pth --info
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def get_model_info(model_path: str) -> dict:
    """Get information about a PatchCore model."""
    import os

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Get file size
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # Load model to get details
    device = torch.device('cpu')

    # Import PatchCore classes for unpickling
    import __main__
    from inference.faultless import PatchCore, KNNExtractor
    __main__.PatchCore = PatchCore
    __main__.KNNExtractor = KNNExtractor

    model = torch.load(model_path, map_location=device, weights_only=False)

    info = {
        'model_path': str(model_path),
        'file_size_mb': round(file_size_mb, 2),
        'format': 'PyTorch (.pth)',
        'framework': f'PyTorch {torch.__version__}',
    }

    if hasattr(model, 'memory_bank') and model.memory_bank is not None:
        if isinstance(model.memory_bank, torch.Tensor):
            info['memory_bank_shape'] = list(model.memory_bank.shape)
            info['memory_bank_dtype'] = str(model.memory_bank.dtype)

    if hasattr(model, 'f_coreset'):
        info['f_coreset'] = model.f_coreset

    if hasattr(model, 'backbone_name'):
        info['backbone'] = model.backbone_name

    if hasattr(model, 'image_size'):
        info['image_size'] = model.image_size

    return info


def verify_model(model_path: str) -> bool:
    """Verify that a PatchCore model can be loaded and used."""
    from inference.faultless import load_model
    from PIL import Image
    import numpy as np

    print(f"\nVerifying model: {model_path}")

    try:
        # Load model
        device = torch.device('cpu')
        model = load_model(model_path, device)
        print("  [OK] Model loaded successfully")

        # Check memory bank
        if hasattr(model, 'memory_bank') and model.memory_bank is not None:
            if isinstance(model.memory_bank, torch.Tensor):
                print(f"  [OK] Memory bank shape: {model.memory_bank.shape}")
            else:
                print("  [WARN] Memory bank is not a tensor")
        else:
            print("  [WARN] No memory bank found - model may not be trained")

        # Check resize layer
        if hasattr(model, 'resize') and model.resize is not None:
            print("  [OK] Resize layer present")
        else:
            print("  [WARN] No resize layer - model may not be fully initialized")

        # Check backbone
        if hasattr(model, 'featExtract_model') and model.featExtract_model is not None:
            print("  [OK] Backbone model present")
        else:
            print("  [ERROR] No backbone model found")
            return False

        print("\n  Model verification: PASSED")
        return True

    except Exception as e:
        print(f"\n  Model verification: FAILED")
        print(f"  Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='PatchCore model export and verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PatchCore models are saved as PyTorch .pth files and can be deployed directly.
No conversion to TFLite/ONNX is needed.

Examples:
    # Verify a model
    python export_model.py --model models/patchcore_product.pth --verify

    # Get model info
    python export_model.py --model models/patchcore_product.pth --info

    # Save model info to JSON
    python export_model.py --model models/patchcore_product.pth --info --output model_info.json
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to PatchCore model (.pth file)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify model can be loaded and used'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Display model information'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save info to JSON file'
    )

    # Legacy arguments (for backwards compatibility)
    parser.add_argument('--model_dir', type=str, help='(Legacy - use --model instead)')
    parser.add_argument('--output_dir', type=str, help='(Legacy - use --output instead)')
    parser.add_argument('--quantization', type=str, help='(Ignored - PatchCore uses PyTorch)')

    args = parser.parse_args()

    # Handle legacy argument
    model_path = args.model
    if args.model_dir and not args.model:
        # Try to find model in model_dir
        model_dir = Path(args.model_dir)
        pth_files = list(model_dir.glob('*.pth'))
        if pth_files:
            model_path = str(pth_files[0])
            print(f"Found model: {model_path}")
        else:
            print(f"ERROR: No .pth files found in {model_dir}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PATCHCORE MODEL EXPORT/VERIFICATION")
    print("=" * 60)

    if not Path(model_path).exists():
        print(f"\nERROR: Model not found: {model_path}")
        sys.exit(1)

    # Get model info
    if args.info or args.output:
        try:
            info = get_model_info(model_path)

            print("\nModel Information:")
            print("-" * 40)
            for key, value in info.items():
                print(f"  {key}: {value}")

            if args.output:
                output_path = Path(args.output)
                with open(output_path, 'w') as f:
                    json.dump(info, f, indent=2)
                print(f"\nInfo saved to: {output_path}")

        except Exception as e:
            print(f"\nERROR: Failed to get model info: {e}")
            sys.exit(1)

    # Verify model
    if args.verify:
        success = verify_model(model_path)
        sys.exit(0 if success else 1)

    # If no action specified, show help
    if not args.info and not args.verify and not args.output:
        print("\nPatchCore models are already in deployable format.")
        print("Use --info to view model details or --verify to test the model.")
        print("\nDeployment:")
        print("  1. Copy the .pth file to your Raspberry Pi")
        print("  2. Install PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("  3. Install timm: pip install timm")
        print("  4. Run inference using the inference module")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
