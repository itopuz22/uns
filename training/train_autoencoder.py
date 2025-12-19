#!/usr/bin/env python3
"""
Convolutional Autoencoder for Anomaly Detection

This module implements training of unsupervised anomaly detection models
optimized for deployment on Raspberry Pi 5 with AI HAT.

Training environment: Windows PC with GPU (recommended)
Deployment environment: Raspberry Pi 5 with AI HAT

Usage:
    python train_autoencoder.py --data_dir ./data --output_dir ./models
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks, optimizers
    from tensorflow.keras import backend as K
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("ERROR: TensorFlow is required for training.")
    print("Install with: pip install tensorflow")
    sys.exit(1)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.data_loader import DatasetLoader, create_tf_dataset, save_preprocessing_config


class AnomalyAutoencoder:
    """
    Convolutional Autoencoder for anomaly detection.

    Architecture optimized for:
    - Input: 224x224x3 RGB images
    - Latent space: Configurable (default 128)
    - Edge deployment: Designed for TFLite conversion and AI HAT acceleration
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        latent_dim: int = 128,
        encoder_filters: List[int] = None,
        decoder_filters: List[int] = None,
        activation: str = 'relu',
        dropout_rate: float = 0.2
    ):
        """
        Initialize autoencoder architecture.

        Args:
            input_shape: Input image shape (H, W, C)
            latent_dim: Latent space dimension
            encoder_filters: List of filter counts for encoder layers
            decoder_filters: List of filter counts for decoder layers
            activation: Activation function
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder_filters = encoder_filters or [32, 64, 128, 256]
        self.decoder_filters = decoder_filters or [256, 128, 64, 32]
        self.activation = activation
        self.dropout_rate = dropout_rate

        self.encoder = None
        self.decoder = None
        self.autoencoder = None

        self._build_model()

    def _build_encoder(self) -> keras.Model:
        """Build encoder network."""
        inputs = layers.Input(shape=self.input_shape, name='encoder_input')
        x = inputs

        # Encoder blocks
        for i, filters in enumerate(self.encoder_filters):
            x = layers.Conv2D(
                filters,
                kernel_size=3,
                strides=2,
                padding='same',
                name=f'encoder_conv_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'encoder_bn_{i}')(x)
            x = layers.Activation(self.activation, name=f'encoder_act_{i}')(x)
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate, name=f'encoder_drop_{i}')(x)

        # Flatten and dense to latent space
        x = layers.Flatten(name='encoder_flatten')(x)
        latent = layers.Dense(self.latent_dim, name='latent_space')(x)

        return keras.Model(inputs, latent, name='encoder')

    def _build_decoder(self, encoder_output_shape: Tuple) -> keras.Model:
        """Build decoder network."""
        # Calculate the shape before flattening
        h = self.input_shape[0] // (2 ** len(self.encoder_filters))
        w = self.input_shape[1] // (2 ** len(self.encoder_filters))
        reshape_dim = (h, w, self.encoder_filters[-1])

        inputs = layers.Input(shape=(self.latent_dim,), name='decoder_input')

        # Dense and reshape
        x = layers.Dense(
            h * w * self.encoder_filters[-1],
            name='decoder_dense'
        )(inputs)
        x = layers.Reshape(reshape_dim, name='decoder_reshape')(x)

        # Decoder blocks
        for i, filters in enumerate(self.decoder_filters):
            x = layers.Conv2DTranspose(
                filters,
                kernel_size=3,
                strides=2,
                padding='same',
                name=f'decoder_conv_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'decoder_bn_{i}')(x)
            x = layers.Activation(self.activation, name=f'decoder_act_{i}')(x)
            if self.dropout_rate > 0 and i < len(self.decoder_filters) - 1:
                x = layers.Dropout(self.dropout_rate, name=f'decoder_drop_{i}')(x)

        # Output layer
        outputs = layers.Conv2D(
            3,  # RGB channels
            kernel_size=3,
            padding='same',
            activation='sigmoid',  # Output in [0, 1] range
            name='decoder_output'
        )(x)

        return keras.Model(inputs, outputs, name='decoder')

    def _build_model(self) -> None:
        """Build complete autoencoder model."""
        # Build encoder
        self.encoder = self._build_encoder()

        # Build decoder
        self.decoder = self._build_decoder(self.encoder.output_shape)

        # Build autoencoder
        inputs = layers.Input(shape=self.input_shape, name='autoencoder_input')
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)

        self.autoencoder = keras.Model(inputs, decoded, name='autoencoder')

        print("\n=== Autoencoder Architecture ===")
        self.autoencoder.summary()

    def compile(
        self,
        learning_rate: float = 0.001,
        loss: str = 'mse'
    ) -> None:
        """
        Compile the autoencoder model.

        Args:
            learning_rate: Learning rate for optimizer
            loss: Loss function ('mse', 'mae', 'ssim')
        """
        optimizer = optimizers.Adam(learning_rate=learning_rate)

        if loss == 'ssim':
            # SSIM-based loss for structural similarity
            def ssim_loss(y_true, y_pred):
                return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
            loss_fn = ssim_loss
        else:
            loss_fn = loss

        self.autoencoder.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['mae']
        )

    def train(
        self,
        train_data,
        val_data,
        epochs: int = 100,
        batch_size: int = 32,
        callbacks_list: List = None,
        verbose: int = 1
    ):
        """
        Train the autoencoder.

        Args:
            train_data: Training dataset or generator
            val_data: Validation dataset or generator
            epochs: Number of training epochs
            batch_size: Batch size (if using arrays)
            callbacks_list: List of Keras callbacks
            verbose: Verbosity level

        Returns:
            Training history
        """
        if callbacks_list is None:
            callbacks_list = []

        # Check if data is a dataset or generator
        if isinstance(train_data, np.ndarray):
            history = self.autoencoder.fit(
                train_data, train_data,
                validation_data=(val_data, val_data),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks_list,
                verbose=verbose
            )
        else:
            # Assume it's a tf.data.Dataset
            history = self.autoencoder.fit(
                train_data,
                validation_data=val_data,
                epochs=epochs,
                callbacks=callbacks_list,
                verbose=verbose
            )

        return history

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Reconstruct images through autoencoder."""
        return self.autoencoder.predict(images, verbose=0)

    def compute_reconstruction_error(
        self,
        images: np.ndarray,
        method: str = 'mse'
    ) -> np.ndarray:
        """
        Compute reconstruction error for images.

        Args:
            images: Input images
            method: Error method ('mse', 'mae')

        Returns:
            Array of reconstruction errors (one per image)
        """
        reconstructed = self.predict(images)

        if method == 'mse':
            errors = np.mean((images - reconstructed) ** 2, axis=(1, 2, 3))
        elif method == 'mae':
            errors = np.mean(np.abs(images - reconstructed), axis=(1, 2, 3))
        else:
            raise ValueError(f"Unknown error method: {method}")

        return errors

    def save(self, output_dir: str) -> None:
        """Save model and its components."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save complete autoencoder
        self.autoencoder.save(output_path / 'autoencoder.keras')

        # Save encoder and decoder separately
        self.encoder.save(output_path / 'encoder.keras')
        self.decoder.save(output_path / 'decoder.keras')

        # Save model architecture config
        config = {
            'input_shape': list(self.input_shape),
            'latent_dim': self.latent_dim,
            'encoder_filters': self.encoder_filters,
            'decoder_filters': self.decoder_filters,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate
        }

        with open(output_path / 'model_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to: {output_path}")

    @classmethod
    def load(cls, model_dir: str) -> 'AnomalyAutoencoder':
        """Load model from directory."""
        model_path = Path(model_dir)

        # Load config
        with open(model_path / 'model_config.json', 'r') as f:
            config = json.load(f)

        # Create instance
        instance = cls(
            input_shape=tuple(config['input_shape']),
            latent_dim=config['latent_dim'],
            encoder_filters=config['encoder_filters'],
            decoder_filters=config['decoder_filters'],
            activation=config['activation'],
            dropout_rate=config['dropout_rate']
        )

        # Load weights
        instance.autoencoder = keras.models.load_model(model_path / 'autoencoder.keras')
        instance.encoder = keras.models.load_model(model_path / 'encoder.keras')
        instance.decoder = keras.models.load_model(model_path / 'decoder.keras')

        return instance


def determine_threshold(
    model: AnomalyAutoencoder,
    validation_images: np.ndarray,
    percentile: float = 99.0
) -> Dict[str, float]:
    """
    Determine anomaly detection threshold from validation data.

    Uses only "good" samples to determine the threshold.

    Args:
        model: Trained autoencoder
        validation_images: Validation images (all good samples)
        percentile: Percentile for threshold (99 means 1% false positive rate)

    Returns:
        Dictionary with threshold information
    """
    # Compute reconstruction errors for all validation samples
    errors = model.compute_reconstruction_error(validation_images)

    # Calculate statistics
    mean_error = float(np.mean(errors))
    std_error = float(np.std(errors))
    max_error = float(np.max(errors))
    min_error = float(np.min(errors))

    # Calculate threshold at given percentile
    threshold = float(np.percentile(errors, percentile))

    # Also calculate mean + k*std thresholds
    threshold_2std = mean_error + 2 * std_error
    threshold_3std = mean_error + 3 * std_error

    result = {
        'threshold': threshold,
        'percentile': percentile,
        'mean_error': mean_error,
        'std_error': std_error,
        'min_error': min_error,
        'max_error': max_error,
        'threshold_2std': threshold_2std,
        'threshold_3std': threshold_3std,
        'num_samples': len(errors)
    }

    print("\n=== Threshold Analysis ===")
    print(f"Samples analyzed: {len(errors)}")
    print(f"Mean reconstruction error: {mean_error:.6f}")
    print(f"Std reconstruction error: {std_error:.6f}")
    print(f"Min/Max error: {min_error:.6f} / {max_error:.6f}")
    print(f"Threshold (p{percentile}): {threshold:.6f}")
    print(f"Threshold (mean+2std): {threshold_2std:.6f}")
    print(f"Threshold (mean+3std): {threshold_3std:.6f}")

    return result


def train_model(
    data_dir: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[AnomalyAutoencoder, Dict]:
    """
    Train anomaly detection model.

    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save trained model
        config: Training configuration dictionary

    Returns:
        Tuple of (trained model, threshold info)
    """
    # Default configuration
    default_config = {
        'input_size': (224, 224),
        'latent_dim': 128,
        'encoder_filters': [32, 64, 128, 256],
        'decoder_filters': [256, 128, 64, 32],
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'validation_split': 0.2,
        'early_stopping_patience': 10
    }

    if config:
        default_config.update(config)
    config = default_config

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n=== Training Configuration ===")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Load data
    print("\n=== Loading Data ===")
    loader = DatasetLoader(
        data_dir=data_dir,
        target_size=config['input_size'],
        validation_split=config['validation_split']
    )

    train_images, _ = loader.load_all('train')
    val_images, _ = loader.load_all('val')

    print(f"Training images: {train_images.shape}")
    print(f"Validation images: {val_images.shape}")

    # Create model
    print("\n=== Creating Model ===")
    model = AnomalyAutoencoder(
        input_shape=(*config['input_size'], 3),
        latent_dim=config['latent_dim'],
        encoder_filters=config['encoder_filters'],
        decoder_filters=config['decoder_filters']
    )

    model.compile(learning_rate=config['learning_rate'])

    # Setup callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=str(output_path / 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=str(output_path / 'logs'),
            histogram_freq=1
        )
    ]

    # Train
    print("\n=== Training ===")
    history = model.train(
        train_images,
        val_images,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks_list=callbacks_list
    )

    # Determine threshold
    print("\n=== Determining Threshold ===")
    threshold_info = determine_threshold(model, val_images)

    # Save threshold info
    with open(output_path / 'anomaly_threshold.json', 'w') as f:
        json.dump(threshold_info, f, indent=2)

    # Save model
    model.save(output_dir)

    # Save preprocessing config
    save_preprocessing_config(
        str(output_path / 'preprocessing_config.json'),
        config['input_size'],
        'standard'
    )

    # Save training history
    history_dict = {key: [float(v) for v in values]
                    for key, values in history.history.items()}
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)

    print(f"\n=== Training Complete ===")
    print(f"Model saved to: {output_dir}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")

    return model, threshold_info


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description='Train anomaly detection autoencoder'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing training images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./models',
        help='Directory to save trained model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=128,
        help='Latent space dimension'
    )
    parser.add_argument(
        '--input_size',
        type=int,
        default=224,
        help='Input image size (square)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device index (-1 for CPU)'
    )

    args = parser.parse_args()

    # Configure GPU
    if args.gpu >= 0:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
                print(f"Using GPU: {gpus[args.gpu]}")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
    else:
        tf.config.set_visible_devices([], 'GPU')
        print("Using CPU for training")

    # Create configuration
    config = {
        'input_size': (args.input_size, args.input_size),
        'latent_dim': args.latent_dim,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    }

    # Train model
    train_model(args.data_dir, args.output_dir, config)


if __name__ == '__main__':
    main()
