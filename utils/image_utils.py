"""Image processing utilities for Industrial Quality Control System.

This module ensures consistent preprocessing between Windows training
and Raspberry Pi deployment environments.
"""

import io
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union

# Import OpenCV - works on both Windows and Raspberry Pi
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Image processing disabled.")

# Import PIL as fallback
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def load_image(
    source: Union[str, Path, bytes, np.ndarray],
    color_mode: str = 'RGB'
) -> Optional[np.ndarray]:
    """
    Load image from various sources.

    Args:
        source: File path, bytes, or numpy array
        color_mode: 'RGB' or 'BGR'

    Returns:
        Image as numpy array in specified color mode
    """
    if isinstance(source, np.ndarray):
        img = source
    elif isinstance(source, (str, Path)):
        if CV2_AVAILABLE:
            img = cv2.imread(str(source))
            if img is None:
                return None
        elif PIL_AVAILABLE:
            pil_img = Image.open(source)
            img = np.array(pil_img)
        else:
            raise RuntimeError("No image loading library available")
    elif isinstance(source, bytes):
        if CV2_AVAILABLE:
            nparr = np.frombuffer(source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif PIL_AVAILABLE:
            pil_img = Image.open(io.BytesIO(source))
            img = np.array(pil_img)
        else:
            raise RuntimeError("No image loading library available")
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")

    # Convert color mode if needed
    if CV2_AVAILABLE and len(img.shape) == 3:
        if color_mode == 'RGB' and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_mode == 'BGR' and img.shape[2] == 3:
            pass  # OpenCV loads in BGR by default

    return img


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    normalization_mode: str = 'standard'
) -> np.ndarray:
    """
    Preprocess image for model inference.

    IMPORTANT: This function must produce identical results on both
    Windows (training) and Raspberry Pi (deployment) to ensure
    consistent model behavior.

    Args:
        image: Input image as numpy array (HxWxC, RGB)
        target_size: Target dimensions (width, height)
        normalize: Whether to normalize pixel values
        normalization_mode: 'standard' (0-1) or 'imagenet' (mean/std)

    Returns:
        Preprocessed image ready for model input
    """
    if image is None:
        raise ValueError("Input image is None")

    # Resize to target dimensions
    if CV2_AVAILABLE:
        resized = cv2.resize(
            image,
            target_size,
            interpolation=cv2.INTER_LINEAR
        )
    elif PIL_AVAILABLE:
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize(target_size, Image.BILINEAR)
        resized = np.array(pil_img)
    else:
        raise RuntimeError("No image processing library available")

    # Convert to float32
    processed = resized.astype(np.float32)

    # Normalize
    if normalize:
        if normalization_mode == 'standard':
            # Scale to 0-1 range
            processed = processed / 255.0
        elif normalization_mode == 'imagenet':
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            processed = processed / 255.0
            processed = (processed - mean) / std
        elif normalization_mode == 'centered':
            # Center around 0 (-1 to 1)
            processed = (processed / 127.5) - 1.0

    return processed


def postprocess_image(
    image: np.ndarray,
    normalization_mode: str = 'standard'
) -> np.ndarray:
    """
    Convert normalized image back to displayable format.

    Args:
        image: Normalized image
        normalization_mode: Normalization mode used

    Returns:
        Image in uint8 format (0-255)
    """
    if normalization_mode == 'standard':
        output = image * 255.0
    elif normalization_mode == 'imagenet':
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        output = (image * std + mean) * 255.0
    elif normalization_mode == 'centered':
        output = (image + 1.0) * 127.5
    else:
        output = image * 255.0

    return np.clip(output, 0, 255).astype(np.uint8)


def compute_reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
    method: str = 'mse'
) -> Tuple[float, np.ndarray]:
    """
    Compute reconstruction error between original and reconstructed images.

    Args:
        original: Original preprocessed image
        reconstructed: Reconstructed image from autoencoder
        method: Error computation method ('mse', 'mae', 'ssim')

    Returns:
        Tuple of (scalar error value, per-pixel error map)
    """
    if method == 'mse':
        error_map = (original - reconstructed) ** 2
        error_value = float(np.mean(error_map))
    elif method == 'mae':
        error_map = np.abs(original - reconstructed)
        error_value = float(np.mean(error_map))
    elif method == 'ssim':
        # Simplified SSIM computation
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_x = np.mean(original, axis=(0, 1))
        mu_y = np.mean(reconstructed, axis=(0, 1))
        sigma_x = np.var(original, axis=(0, 1))
        sigma_y = np.var(reconstructed, axis=(0, 1))
        sigma_xy = np.mean((original - mu_x) * (reconstructed - mu_y), axis=(0, 1))

        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))

        error_value = 1.0 - float(np.mean(ssim))
        error_map = (original - reconstructed) ** 2  # Use MSE for visualization
    else:
        raise ValueError(f"Unknown error method: {method}")

    return error_value, error_map


def create_heatmap(
    error_map: np.ndarray,
    original_size: Tuple[int, int] = None,
    colormap: int = None
) -> np.ndarray:
    """
    Create a heatmap visualization from reconstruction error.

    Args:
        error_map: Per-pixel error values
        original_size: Size to resize heatmap to (width, height)
        colormap: OpenCV colormap constant

    Returns:
        Heatmap as BGR image
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV required for heatmap generation")

    if colormap is None:
        colormap = cv2.COLORMAP_JET

    # Normalize error map to 0-255
    if error_map.ndim == 3:
        error_map = np.mean(error_map, axis=2)

    error_normalized = error_map - error_map.min()
    if error_normalized.max() > 0:
        error_normalized = error_normalized / error_normalized.max()
    error_uint8 = (error_normalized * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(error_uint8, colormap)

    # Resize if needed
    if original_size is not None:
        heatmap = cv2.resize(heatmap, original_size, interpolation=cv2.INTER_LINEAR)

    return heatmap


def overlay_heatmap(
    original: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay heatmap on original image.

    Args:
        original: Original image (BGR)
        heatmap: Heatmap image (BGR)
        alpha: Blending factor (0-1)

    Returns:
        Blended image
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV required for image overlay")

    # Ensure same size
    if original.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

    # Blend images
    return cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)


def add_result_annotation(
    image: np.ndarray,
    result: str,
    confidence: float,
    processing_time_ms: float
) -> np.ndarray:
    """
    Add classification result annotation to image.

    Args:
        image: Input image (BGR)
        result: Classification result ('OK' or 'NOK')
        confidence: Confidence score (0-1)
        processing_time_ms: Processing time in milliseconds

    Returns:
        Annotated image
    """
    if not CV2_AVAILABLE:
        return image

    annotated = image.copy()

    # Colors (BGR)
    color = (0, 255, 0) if result == 'OK' else (0, 0, 255)
    bg_color = (0, 100, 0) if result == 'OK' else (0, 0, 100)

    # Draw result banner
    h, w = annotated.shape[:2]
    banner_height = 60
    cv2.rectangle(annotated, (0, 0), (w, banner_height), bg_color, -1)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        annotated,
        f"{result}",
        (10, 45),
        font,
        1.5,
        color,
        3
    )

    cv2.putText(
        annotated,
        f"Conf: {confidence:.1%}",
        (w // 3, 45),
        font,
        1.0,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        f"Time: {processing_time_ms:.0f}ms",
        (2 * w // 3, 45),
        font,
        1.0,
        (255, 255, 255),
        2
    )

    return annotated


def crop_center(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    """
    Crop the center of an image.

    Args:
        image: Input image
        crop_size: (width, height) of crop

    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    crop_w, crop_h = crop_size

    start_x = max(0, (w - crop_w) // 2)
    start_y = max(0, (h - crop_h) // 2)

    return image[start_y:start_y + crop_h, start_x:start_x + crop_w]
