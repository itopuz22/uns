"""Data loading and augmentation for anomaly detection training.

Optimized for Raspberry Pi HQ Camera images (Sony IMX477 12.3MP sensor).
Supports various resolutions from 4056x3040 down to preview sizes.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Generator, Dict, Any

# TensorFlow imports for training on Windows
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Training functionality disabled.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_utils import preprocess_image, load_image


class DatasetLoader:
    """
    Dataset loader for anomaly detection training.

    Handles loading and preprocessing of "good" sample images
    for training unsupervised anomaly detection models.
    """

    def __init__(
        self,
        data_dir: str,
        target_size: Tuple[int, int] = (224, 224),
        validation_split: float = 0.2,
        normalization_mode: str = 'standard'
    ):
        """
        Initialize dataset loader.

        Args:
            data_dir: Directory containing 'good' subdirectory with training images
            target_size: Target image size (width, height)
            validation_split: Fraction of data for validation
            normalization_mode: Normalization mode ('standard', 'imagenet', 'centered')
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.validation_split = validation_split
        self.normalization_mode = normalization_mode

        # Supported image extensions
        self.extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        # Find all image files
        self.image_paths = self._find_images()
        print(f"Found {len(self.image_paths)} images in {data_dir}")

        # Split into train and validation
        self._split_data()

    def _find_images(self) -> List[Path]:
        """Find all image files in data directory."""
        images = []
        good_dir = self.data_dir / 'good'

        if good_dir.exists():
            search_dir = good_dir
        else:
            # Fallback to searching in data_dir directly
            search_dir = self.data_dir
            print(f"Warning: 'good' subdirectory not found. Using {search_dir}")

        for ext in self.extensions:
            images.extend(search_dir.glob(f'*{ext}'))
            images.extend(search_dir.glob(f'*{ext.upper()}'))

        return sorted(images)

    def _split_data(self) -> None:
        """Split data into training and validation sets."""
        np.random.seed(42)
        indices = np.random.permutation(len(self.image_paths))

        split_idx = int(len(indices) * (1 - self.validation_split))

        self.train_indices = indices[:split_idx]
        self.val_indices = indices[split_idx:]

        print(f"Training samples: {len(self.train_indices)}")
        print(f"Validation samples: {len(self.val_indices)}")

    def load_image(self, idx: int) -> Optional[np.ndarray]:
        """Load and preprocess a single image by index."""
        if idx >= len(self.image_paths):
            return None

        image_path = self.image_paths[idx]
        img = load_image(image_path, color_mode='RGB')

        if img is None:
            print(f"Warning: Could not load image: {image_path}")
            return None

        return preprocess_image(
            img,
            target_size=self.target_size,
            normalize=True,
            normalization_mode=self.normalization_mode
        )

    def load_all(
        self,
        subset: str = 'train'
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Load all images for a subset.

        Args:
            subset: 'train' or 'val'

        Returns:
            Tuple of (images array, list of file paths)
        """
        indices = self.train_indices if subset == 'train' else self.val_indices

        images = []
        paths = []

        for idx in indices:
            img = self.load_image(idx)
            if img is not None:
                images.append(img)
                paths.append(str(self.image_paths[idx]))

        return np.array(images), paths

    def get_generator(
        self,
        subset: str = 'train',
        batch_size: int = 32,
        augment: bool = True
    ) -> Generator:
        """
        Get a generator for batch loading.

        Args:
            subset: 'train' or 'val'
            batch_size: Batch size
            augment: Apply data augmentation

        Yields:
            Batches of (images, images) for autoencoder training
        """
        indices = self.train_indices if subset == 'train' else self.val_indices

        while True:
            # Shuffle at each epoch
            np.random.shuffle(indices)

            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]

                batch_images = []
                for idx in batch_indices:
                    img = self.load_image(idx)
                    if img is not None:
                        if augment and subset == 'train':
                            img = self._augment_image(img)
                        batch_images.append(img)

                if batch_images:
                    batch = np.array(batch_images)
                    yield batch, batch  # Input = Target for autoencoder

    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation to image."""
        if not CV2_AVAILABLE:
            return image

        augmented = image.copy()

        # Random rotation (small angle for industrial images)
        if np.random.random() < 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = augmented.shape[:2]
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            augmented = cv2.warpAffine(
                augmented, matrix, (w, h),
                borderMode=cv2.BORDER_REFLECT
            )

        # Random brightness adjustment
        if np.random.random() < 0.5:
            factor = np.random.uniform(0.8, 1.2)
            augmented = np.clip(augmented * factor, 0, 1)

        # Random contrast adjustment
        if np.random.random() < 0.3:
            factor = np.random.uniform(0.9, 1.1)
            mean = np.mean(augmented)
            augmented = np.clip((augmented - mean) * factor + mean, 0, 1)

        # Small random shift
        if np.random.random() < 0.3:
            h, w = augmented.shape[:2]
            shift_x = int(np.random.uniform(-0.1, 0.1) * w)
            shift_y = int(np.random.uniform(-0.1, 0.1) * h)
            matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            augmented = cv2.warpAffine(
                augmented, matrix, (w, h),
                borderMode=cv2.BORDER_REFLECT
            )

        return augmented


def create_data_generators(
    data_dir: str,
    target_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
    augmentation_config: Optional[Dict[str, Any]] = None
) -> Tuple[Generator, Generator, int, int]:
    """
    Create training and validation data generators.

    Args:
        data_dir: Directory containing training data
        target_size: Target image dimensions
        batch_size: Batch size
        validation_split: Validation split ratio
        augmentation_config: Augmentation configuration dict

    Returns:
        Tuple of (train_generator, val_generator, train_steps, val_steps)
    """
    loader = DatasetLoader(
        data_dir=data_dir,
        target_size=target_size,
        validation_split=validation_split
    )

    train_gen = loader.get_generator('train', batch_size, augment=True)
    val_gen = loader.get_generator('val', batch_size, augment=False)

    train_steps = len(loader.train_indices) // batch_size
    val_steps = len(loader.val_indices) // batch_size

    return train_gen, val_gen, train_steps, val_steps


def create_tf_dataset(
    data_dir: str,
    target_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2
) -> Tuple:
    """
    Create TensorFlow datasets for training.

    Args:
        data_dir: Directory containing training data
        target_size: Target image dimensions
        batch_size: Batch size
        validation_split: Validation split ratio

    Returns:
        Tuple of (train_dataset, val_dataset, train_steps, val_steps)
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required for dataset creation")

    loader = DatasetLoader(
        data_dir=data_dir,
        target_size=target_size,
        validation_split=validation_split
    )

    # Load all data
    train_images, _ = loader.load_all('train')
    val_images, _ = loader.load_all('val')

    # Create TF datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_images))
    train_dataset = train_dataset.shuffle(len(train_images))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_images))
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    train_steps = len(train_images) // batch_size
    val_steps = len(val_images) // batch_size

    return train_dataset, val_dataset, train_steps, val_steps


def save_preprocessing_config(
    output_path: str,
    target_size: Tuple[int, int],
    normalization_mode: str
) -> None:
    """
    Save preprocessing configuration for deployment.

    This ensures identical preprocessing on Raspberry Pi.
    """
    config = {
        'target_size': list(target_size),
        'normalization_mode': normalization_mode,
        'color_mode': 'RGB'
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Preprocessing config saved to: {output_path}")
