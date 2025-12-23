#!/usr/bin/env python3
"""
PatchCore-based Anomaly Detection Module (FAULTLESS)

Industrial quality control using PatchCore algorithm with WideResNet50 backbone.
Supports training, inference, and folder watching modes.

Usage:
    from inference.faultless import PatchCore, create_model, load_model, run_inference
"""

import os
import sys
import glob
import csv
import warnings
import json
import threading
import time
import shutil
import signal
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFilter

# Suppress matplotlib backend warning
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

from sklearn import random_projection, metrics
from sklearn.metrics import roc_auc_score, roc_curve

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Install with: pip install timm")

# Add parent to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logging_utils import get_logger

logger = get_logger('faultless')

# =============================
# Global settings
# =============================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 512
RESIZE_SIZE = 544  # slightly larger for center crop
DATASETS_PATH = "Training"
THRESHOLD = 0  # Default threshold for classification

# Default paths
DEFAULT_MODEL_PATH = "models/patchcore_model.pth"
DEFAULT_OUTPUT_FOLDER = "results"
BACKBONE_WEIGHTS_PATH = "Weights/AI_Faultless_weights_base.pth"


def resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# =============================
# Utilities
# =============================

def print_results(results: dict, method: str):
    """Print final computation results."""
    logger.info("Final results of the computation")
    logger.info(f"Average image rocauc: {results['average image rocauc']:.2f}")


def plot_roc(y, pred, cls, save_path: Optional[str] = None):
    """Plot ROC curve."""
    logger.info("Plotting ROC curve...")
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name=str(cls))
    display.plot()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def get_coreset_idx(
    z_lib: tensor,
    n: int = 1000,
    eps: float = 0.90,
    float16: bool = True,
    force_cpu: bool = False,
) -> tensor:
    """
    Perform coreset subsampling reduction.

    Args:
        z_lib: Feature library tensor
        n: Number of coreset samples
        eps: Epsilon for random projection
        float16: Use float16 for speed/memory
        force_cpu: Force CPU computation

    Returns:
        Indices of selected coreset samples
    """
    logger.info("Beginning coreset subsampling reduction...")
    logger.info(f"Start dim = {z_lib.shape}.")

    try:
        logger.info("Fitting random projection to reduce dimensionality...")
        transformer = random_projection.SparseRandomProjection(eps=eps)
        z_lib = torch.tensor(transformer.fit_transform(z_lib))
        logger.info(f"Transformed dim = {z_lib.shape}.")
    except ValueError:
        logger.error("Could not project vectors. Please increase eps.")

    select_idx = 0
    last_item = z_lib[select_idx:select_idx + 1]
    coreset_idx = [torch.tensor(select_idx)]
    min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)

    if float16:
        logger.debug("Converting data to float16 for speed/memory...")
        last_item = last_item.half()
        z_lib = z_lib.half()
        min_distances = min_distances.half()

    if torch.cuda.is_available() and not force_cpu:
        logger.debug("Moving data to CUDA for faster computation...")
        last_item = last_item.to("cuda")
        z_lib = z_lib.to("cuda")
        min_distances = min_distances.to("cuda")

    for i in range(n - 1):
        if (i + 1) % 100 == 0:
            logger.debug(f"Coreset selection: {i + 1}/{n}...")
        distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
        min_distances = torch.minimum(distances, min_distances)
        select_idx = torch.argmax(min_distances)
        last_item = z_lib[select_idx:select_idx + 1]
        min_distances[select_idx] = 0
        coreset_idx.append(select_idx.to("cpu"))

    logger.info("End of coreset subsampling reduction")
    return torch.stack(coreset_idx)


# =============================
# KNNExtractor (Base Feature Extractor)
# =============================

class KNNExtractor(torch.nn.Module):
    """
    K-Nearest Neighbor feature extractor using WideResNet50 backbone.
    """

    def __init__(
        self,
        featExtract_model_name: str = "wideresnet50",
        output_indices: Tuple = (2, 3),
        pool_last: bool = False,
        depth: int = 20,
        width_multiplier: float = 1.0,
        num_classes: int = 15,
        dropout: float = 0.2,
        featExtract_model=None,
    ):
        super().__init__()
        self.depth = depth
        self.width_multiplier = width_multiplier
        self.num_classes = num_classes
        self.dropout = dropout
        self.output_indices = output_indices
        self.pool_last = pool_last
        self.backbone_name = featExtract_model_name
        self.preprocess = None

        if not TIMM_AVAILABLE:
            raise RuntimeError("timm library required for PatchCore. Install with: pip install timm")

        logger.info("Creating WideResNet50 backbone (features_only)...")
        self.featExtract_model = timm.create_model(
            "wide_resnet50_2",
            out_indices=output_indices,
            features_only=True,
            pretrained=False,
        )

        for param in self.featExtract_model.parameters():
            param.requires_grad = False

        self.featExtract_model.eval()
        self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
        self.featExtract_model_name = featExtract_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Moving backbone to device={self.device}...")
        self.featExtract_model = self.featExtract_model.to(self.device)

    def __call__(self, tensor: torch.Tensor) -> Tuple:
        with torch.no_grad():
            try:
                model_device = next(self.featExtract_model.parameters()).device
            except StopIteration:
                model_device = getattr(self, "device", torch.device("cpu"))

            tensor = tensor.to(model_device, non_blocking=True)
            extracted_features = self.featExtract_model(tensor)

            if self.pool:
                feat_list = [x for x in extracted_features[:-1]]
                pooled_last = self.pool(extracted_features[-1])
                feat_list.append(pooled_last)
                return feat_list
            else:
                return [x for x in extracted_features]

    def evaluate(self, test_data: DataLoader, cls: str, model_dir: str = None) -> Tuple[float, float]:
        """
        Evaluate model performance on test_data.

        Args:
            test_data: Test dataloader
            cls: Class name
            model_dir: Directory to save threshold JSON

        Returns:
            Tuple of (image-level ROC-AUC, best threshold)
        """
        logger.info("Starting model evaluation on test dataset...")
        image_predictions = []
        image_labels = []

        for idx, (sample, mask, label) in enumerate(test_data):
            logger.debug(f"Evaluating test sample {idx + 1}...")
            if sample.shape[0] == 1:
                z_score = self.predict(sample)
                score_val = float(z_score.item() if isinstance(z_score, torch.Tensor) else z_score)
                image_predictions.append(score_val)
                image_labels.append(int(label[0].numpy() if hasattr(label[0], "numpy") else label[0]))
            else:
                for i in range(sample.shape[0]):
                    z_score = self.predict(sample[i:i+1])
                    score_val = float(z_score.item() if isinstance(z_score, torch.Tensor) else z_score)
                    image_predictions.append(score_val)
                    image_labels.append(int(label[i].numpy() if hasattr(label[i], "numpy") else label[i]))

        logger.info("Computing ROC-AUC metric...")
        image_roc_auc = roc_auc_score(image_labels, image_predictions)
        fpr, tpr, thresholds = roc_curve(image_labels, image_predictions)

        # Choose threshold where TPR - FPR is maximized (Youden's J statistic)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]

        logger.info(f"Proposed threshold = {best_threshold:.4f} (TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f})")

        # Save threshold JSON
        if model_dir:
            try:
                model_name = f"patchcore_{cls.lower()}"
                threshold_json = os.path.join(model_dir, f"{model_name}_threshold.json")
                threshold_data = {
                    "model_name": model_name,
                    "threshold": round(float(best_threshold), 4),
                    "roc_auc": round(float(image_roc_auc), 4),
                    "tpr": round(float(tpr[best_idx]), 4),
                    "fpr": round(float(fpr[best_idx]), 4)
                }
                with open(threshold_json, "w") as f:
                    json.dump(threshold_data, f, indent=4)
                logger.info(f"Saved threshold ({best_threshold:.4f}) to {threshold_json}")
            except Exception as e:
                logger.warning(f"Could not save threshold JSON: {e}")

        logger.info(f"Evaluation finished. Image ROC-AUC = {image_roc_auc:.4f}")
        return image_roc_auc, best_threshold

    def get_parameters(self, extra_params: dict = None) -> dict:
        extra_params = extra_params or {}
        return {
            "backbone_name": self.backbone_name,
            "out_indices": self.output_indices,
            **extra_params,
        }


# =============================
# PatchCore Model
# =============================

class PatchCore(KNNExtractor):
    """
    PatchCore anomaly detection implementation.

    Uses coreset sampling + KNN logic for unsupervised anomaly detection.
    """

    def __init__(
        self,
        f_coreset: float = 1.0,
        backbone_name: str = "wideresnet50",
        coreset_eps: float = 0.90,
        out_indices: Tuple = None,
        pool_last: bool = False,
    ):
        super().__init__(backbone_name, output_indices=(2, 3), pool_last=pool_last)

        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps
        self.image_size = IMAGE_SIZE
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.n_reweight = 3
        self.memory_bank = []
        self.resize = None
        self.featExtract_model_name = backbone_name
        self.model_path = None

    def save(self, path: str):
        """Save model to file."""
        logger.info(f"Saving model to {path}...")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self, path)
        self.model_path = path

    @staticmethod
    def load(path: str, device: str = "cpu") -> 'PatchCore':
        """Load model from file."""
        logger.info(f"Loading model from {path}...")
        return torch.load(path, map_location=device, weights_only=False)

    def fit(self, train_dl: DataLoader):
        """
        Fit the model by building memory bank from training patches.

        Args:
            train_dl: Training dataloader
        """
        logger.info("Building memory bank from training data...")

        memory_bank = []
        largest_fmap_size = None

        for idx, sample in enumerate(train_dl):
            logger.debug(f"Processing training batch {idx + 1}...")
            imgs = sample[0]
            feature_maps = self(imgs)

            if largest_fmap_size is None:
                largest_fmap_size = feature_maps[0].shape[-2:]
                self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size).to(self.device)
                self.average = self.average.to(self.device)
                logger.info(f"Set target spatial dims to {largest_fmap_size}")

            resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
            patch = torch.cat(resized_maps, 1)
            patch = patch.reshape(patch.shape[1], -1).T
            memory_bank.append(patch)

        self.memory_bank = torch.cat(memory_bank, 0)
        logger.info(f"Memory bank shape = {self.memory_bank.shape}")

        if self.f_coreset < 1:
            logger.info(f"Applying coreset sampling with f_coreset={self.f_coreset}...")
            self.coreset_idx = get_coreset_idx(
                self.memory_bank,
                n=int(max(1, self.f_coreset * self.memory_bank.shape[0])),
                eps=self.coreset_eps,
            )
            self.memory_bank = self.memory_bank[self.coreset_idx]
            logger.info(f"Reduced memory bank shape = {self.memory_bank.shape}")

        logger.info("Fit completed. Memory bank ready.")

    def predict(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly score for a single sample.

        Args:
            sample: Input tensor of shape (1, C, H, W)

        Returns:
            Anomaly score tensor
        """
        feature_maps = self(sample)
        resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
        patch = torch.cat(resized_maps, 1)
        patch = patch.reshape(patch.shape[1], -1).T

        memory_bank = self.memory_bank
        if isinstance(memory_bank, list):
            memory_bank = torch.cat(memory_bank, 0)

        memory_bank = memory_bank.to(patch.device).float()
        dist = torch.cdist(patch, memory_bank)
        min_val, min_idx = torch.min(dist, dim=1)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        m_test = patch[s_idx].unsqueeze(0)
        m_star = memory_bank[min_idx[s_idx]].unsqueeze(0)

        w_dist = torch.cdist(m_star, memory_bank)
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)
        knn_idxs = nn_idx[0]
        if knn_idxs.shape[0] > 1:
            neighbor_idxs = knn_idxs[1:]
        else:
            neighbor_idxs = knn_idxs

        m_star_knn = torch.linalg.norm(m_test - memory_bank[neighbor_idxs], dim=1)
        s = torch.max(min_val)
        return s

    def get_parameters(self) -> dict:
        return super().get_parameters({
            "f_coreset": self.f_coreset,
            "n_reweight": self.n_reweight,
        })


# =============================
# Dataset Classes
# =============================

class MVTecTrainDataset(ImageFolder):
    """Training dataset for MVTec-style anomaly detection."""

    def __init__(
        self,
        cls: str,
        source: str = DATASETS_PATH,
        resize: int = RESIZE_SIZE,
        imagesize: int = IMAGE_SIZE
    ):
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor()
        ])
        super().__init__(root=os.path.join(source, cls, "train"), transform=transform)
        self.cls = cls


class MVTecTestDataset(ImageFolder):
    """Test dataset for MVTec-style anomaly detection with ground truth masks."""

    def __init__(
        self,
        cls: str,
        source: str = DATASETS_PATH,
        resize: int = RESIZE_SIZE,
        imagesize: int = IMAGE_SIZE
    ):
        self.cls = cls
        self.imagesize = imagesize
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor()
        ])
        super().__init__(root=os.path.join(source, cls, "test"), transform=transform)

        self.mask_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)

        if "good" in path:
            target = Image.new("RGB", (self.imagesize, self.imagesize))
            sample_class = 0
        else:
            target_path = path.replace(
                os.path.join("test"),
                os.path.join("ground_truth")
            ).replace(".png", "_mask.png")
            if os.path.exists(target_path):
                target = self.loader(target_path)
            else:
                target = Image.new("RGB", (self.imagesize, self.imagesize))
            sample_class = 1

        if self.transform is not None:
            sample = self.transform(sample)
        if self.mask_transform is not None:
            target = self.mask_transform(target)

        return sample, target[:1], sample_class


class MVTecDataset:
    """Combined train/test dataset for MVTec-style anomaly detection."""

    def __init__(
        self,
        cls: str,
        source: str = DATASETS_PATH,
        size: int = IMAGE_SIZE,
        resize: int = RESIZE_SIZE
    ):
        self.train_ds = MVTecTrainDataset(cls, source, resize=resize, imagesize=size)
        self.test_ds = MVTecTestDataset(cls, source, resize=resize, imagesize=size)

    def get_dataloaders(self, batch_size: int = 1) -> Tuple[DataLoader, DataLoader]:
        """Create training and testing dataloaders."""
        logger.info("Creating dataloaders for training and testing...")
        train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            self.test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        return train_loader, test_loader


# =============================
# Backbone Weight Loading
# =============================

def load_backbone_weights_into_patchcore(
    patchcore_model: PatchCore,
    weights_path: str,
    device: torch.device = None
):
    """
    Load local timm checkpoint into patchcore backbone.

    Args:
        patchcore_model: PatchCore model instance
        weights_path: Path to backbone weights
        device: Target device
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"{weights_path} not found")

    logger.info(f"Loading backbone weights from {weights_path}")
    ckpt = torch.load(weights_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    backbone_sd = patchcore_model.featExtract_model.state_dict()
    cleaned = {}
    skipped = 0

    for k, v in sd.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        for prefix in ("model.", "backbone."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]

        if new_k in backbone_sd and backbone_sd[new_k].shape == v.shape:
            cleaned[new_k] = v
        else:
            skipped += 1

    if not cleaned:
        logger.warning("No matching keys found in checkpoint for backbone")
    else:
        load_res = patchcore_model.featExtract_model.load_state_dict(cleaned, strict=False)
        logger.info(f"Loaded {len(cleaned)} parameters into backbone")
        if getattr(load_res, "missing_keys", None):
            logger.debug(f"Missing keys: {len(load_res.missing_keys)}")

    patchcore_model.featExtract_model = patchcore_model.featExtract_model.to(device)
    patchcore_model.featExtract_model.eval()
    patchcore_model.device = device
    logger.info("Backbone ready (moved to device and set to eval)")


# =============================
# Model Creation and Training
# =============================

def create_model(
    f_coreset: float = 0.1,
    backbone_name: str = "wideresnet50",
    pretrained_path: str = None,
    backbone_weights_path: str = None,
    device: str = "cpu"
) -> PatchCore:
    """
    Create a new PatchCore model.

    Args:
        f_coreset: Coreset fraction (0-1)
        backbone_name: Backbone architecture name
        pretrained_path: Path to pretrained PatchCore model
        backbone_weights_path: Path to backbone weights
        device: Target device

    Returns:
        PatchCore model instance
    """
    logger.info("Creating PatchCore model instance...")
    model = PatchCore(f_coreset=f_coreset, backbone_name=backbone_name)
    device_t = torch.device(device) if not isinstance(device, torch.device) else device
    model = model.to(device_t)
    model.device = device_t

    if pretrained_path and os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained model from {pretrained_path}...")
        state = torch.load(pretrained_path, map_location=device_t)
        if isinstance(state, dict) and "memory_bank" in state:
            try:
                model.load_state_dict(state)
                logger.info("Loaded model state dict")
            except Exception as e:
                logger.warning(f"Warning loading state dict: {e}")
        else:
            try:
                model = state.to(device_t)
                logger.info("Loaded full model object")
            except Exception as e:
                logger.warning(f"Failed to load full model: {e}")

    if backbone_weights_path and os.path.exists(backbone_weights_path):
        try:
            load_backbone_weights_into_patchcore(model, backbone_weights_path, device=device_t)
        except Exception as e:
            logger.warning(f"Error loading backbone weights: {e}")

    logger.info("Model creation complete")
    return model


def train_model(model: PatchCore, train_loader: DataLoader):
    """Train PatchCore model."""
    logger.info("Starting training (fit)...")
    model.fit(train_loader)
    logger.info("Training (fit) completed")


def evaluate_model(
    model: PatchCore,
    test_loader: DataLoader,
    cls: str,
    model_dir: str = None
) -> Optional[Tuple[float, float]]:
    """Evaluate PatchCore model."""
    logger.info("Running evaluation...")
    try:
        return model.evaluate(test_loader, cls=cls, model_dir=model_dir)
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return None


# =============================
# Model Loading for Inference
# =============================

def load_model(model_path: str, device: torch.device) -> PatchCore:
    """
    Load a trained PatchCore model for inference.

    Args:
        model_path: Path to model file
        device: Target device

    Returns:
        Loaded PatchCore model
    """
    import __main__
    __main__.PatchCore = PatchCore
    __main__.KNNExtractor = KNNExtractor

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} does not exist")

    logger.info(f"Loading model from {model_path} onto device {device}...")
    loaded = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(loaded, PatchCore):
        model = loaded.to(device).eval()
    elif isinstance(loaded, dict):
        model = create_model(device=device)
        try:
            model.load_state_dict(loaded, strict=False)
            model = model.to(device).eval()
        except Exception:
            if "memory_bank" in loaded:
                model.memory_bank = loaded["memory_bank"]
            model = model.to(device).eval()
    else:
        raise RuntimeError("Unrecognized model file format")

    if hasattr(model, "memory_bank") and model.memory_bank is not None:
        if isinstance(model.memory_bank, torch.Tensor):
            model.memory_bank = model.memory_bank.to(device).float()
            logger.info(f"Moved memory bank to device {device}")

    logger.info("Model loaded and ready")
    return model


# =============================
# Inference
# =============================

def run_inference(
    model: PatchCore,
    image_path: str,
    device: torch.device,
    save_dir: str,
    threshold: float = None,
    generate_heatmap: bool = True
) -> Tuple[float, str, str, Optional[str]]:
    """
    Run inference on a single image.

    Args:
        model: PatchCore model
        image_path: Path to input image
        device: Computation device
        save_dir: Directory to save results
        threshold: Classification threshold
        generate_heatmap: Whether to generate heatmap visualization

    Returns:
        Tuple of (score, image_name, label, heatmap_path)
    """
    os.makedirs(save_dir, exist_ok=True)

    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    # Preprocessing
    resized = transforms.Resize(RESIZE_SIZE)(orig_img)
    rw, rh = resized.size
    left = (rw - IMAGE_SIZE) // 2
    upper = (rh - IMAGE_SIZE) // 2
    crop_box = (left, upper, left + IMAGE_SIZE, upper + IMAGE_SIZE)
    cropped = resized.crop(crop_box)

    to_tensor = transforms.Compose([transforms.ToTensor()])
    img_tensor = to_tensor(cropped).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_score = model.predict(img_tensor)
        score = float(raw_score.item() if isinstance(raw_score, torch.Tensor) else raw_score)

        use_thresh = threshold if threshold is not None else THRESHOLD
        label = "Anomaly" if score > use_thresh else "Normal"

        if not generate_heatmap:
            logger.info(f"{os.path.basename(image_path)}: Score={score:.4f}, Label={label} (no heatmap)")
            return score, os.path.basename(image_path), label, None

        # Generate heatmap
        feature_maps = model(img_tensor)
        resized_maps = [model.resize(model.average(fmap)) for fmap in feature_maps]
        patch = torch.cat(resized_maps, 1)
        patch = patch.reshape(patch.shape[1], -1).T

        memory_bank = model.memory_bank.to(patch.device).float()
        dist = torch.cdist(patch, memory_bank)
        min_val, _ = torch.min(dist, dim=1)

        fmap_h, fmap_w = resized_maps[0].shape[-2:]
        heatmap = min_val.reshape(fmap_h, fmap_w).cpu().numpy()

        lo_clip = np.percentile(heatmap, 1.0)
        hi_clip = np.percentile(heatmap, 99.5)
        heatmap = np.clip(heatmap, lo_clip, hi_clip)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = np.power(heatmap, 1.8)

        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC
        )
        heatmap_img = heatmap_img.filter(ImageFilter.GaussianBlur(radius=3))

        heatmap_arr = np.array(heatmap_img).astype(np.float32) / 255.0
        cmap = plt.get_cmap("jet")
        colored = (cmap(heatmap_arr)[:, :, :3] * 255).astype(np.uint8)
        colored_img = Image.fromarray(colored)

        alpha = (heatmap_arr - 0.12) / (1.0 - 0.12)
        alpha = np.clip(alpha, 0.0, 1.0) ** 1.7
        alpha_img = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")

        base_rgba = resized.convert("RGBA")
        heatmap_rgba = colored_img.convert("RGBA")
        heatmap_rgba.putalpha(alpha_img)
        base_rgba.paste(heatmap_rgba, (left, upper), heatmap_rgba)

        composed_fullsize = base_rgba.resize((orig_w, orig_h), Image.BICUBIC).convert("RGB")

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img)
        plt.axis("off")
        plt.title("Input")

        plt.subplot(1, 2, 2)
        plt.imshow(composed_fullsize)
        plt.axis("off")
        plt.title(f"Score={score:.4f}\n{label}")

        out_path = os.path.join(save_dir, os.path.basename(image_path))
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

    logger.info(f"{os.path.basename(image_path)}: Score={score:.4f}, Label={label}")
    return score, os.path.basename(image_path), label, out_path


# =============================
# CSV Export
# =============================

def save_scores_csv(csv_path: str, rows: List[tuple]):
    """Save results to CSV file."""
    logger.info(f"Saving results CSV to {csv_path}...")
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "score", "label", "heatmap"])
        writer.writerows(rows)
    logger.info(f"Saved results to {csv_path}")


def append_scores_csv(csv_path: str, rows: List[tuple], threshold: float = None):
    """Append rows to CSV file."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            if threshold is not None:
                writer.writerow([f"Threshold={threshold}"])
            writer.writerow(["image", "score", "label", "heatmap"])
        writer.writerows(rows)


# =============================
# Folder Watcher
# =============================

def watch_folder(
    model_path: str,
    input_folder: str,
    output_folder: str,
    threshold: float = None,
    poll_interval: float = 2.0,
    device: torch.device = None,
    signals_dir: str = None
):
    """
    Watch input folder for new images and run inference.

    Args:
        model_path: Path to model file
        input_folder: Folder to watch for new images
        output_folder: Folder for output results
        threshold: Classification threshold
        poll_interval: Polling interval in seconds
        device: Computation device
        signals_dir: Directory for anomaly signal files
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    if signals_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        signals_dir = os.path.join(base_dir, "signals")
    os.makedirs(signals_dir, exist_ok=True)

    logger.info(f"Starting folder watcher")
    logger.info(f"Model: {model_path}")
    logger.info(f"Watching: {input_folder}")
    logger.info(f"Results: {output_folder}")
    logger.info(f"Signals: {signals_dir}")

    if threshold is None:
        threshold = THRESHOLD
    logger.info(f"Using threshold = {threshold}")

    try:
        model = load_model(model_path, device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Attempting to create model with backbone weights...")
        try:
            bw = BACKBONE_WEIGHTS_PATH if os.path.exists(BACKBONE_WEIGHTS_PATH) else None
            model = create_model(backbone_weights_path=bw, device=device)
        except Exception as e2:
            logger.error(f"Could not create model: {e2}")
            return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = os.path.join(output_folder, f"watch_results_{timestamp}.csv")
    processed = set()
    keep_running = True

    def _signal_handler(sig, frame):
        nonlocal keep_running
        logger.info("Signal received, stopping watcher...")
        keep_running = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info("Watcher active. Press Ctrl+C to stop.")

    try:
        while keep_running:
            try:
                all_images = []
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.gif"):
                    all_images.extend(glob.glob(os.path.join(input_folder, ext)))
                all_images = sorted(all_images)

                new_images = [p for p in all_images if p not in processed]

                if new_images:
                    rows_to_append = []
                    for img_path in new_images:
                        if not keep_running:
                            break
                        processed.add(img_path)
                        try:
                            score, name, label, heatmap_path = run_inference(
                                model, img_path, device, output_folder, threshold=threshold
                            )
                            rows_to_append.append((name, score, label, heatmap_path))
                            logger.info(f"Processed {name} -> {label}, score={score:.4f}")

                            if label == "Anomaly":
                                flag_path = os.path.join(signals_dir, "anomaly.txt")
                                with open(flag_path, "w") as f:
                                    f.write(f"{datetime.now().isoformat()} - {name} - score={score:.4f}\n")
                                logger.warning(f"Anomaly detected! Flag written to {flag_path}")

                        except Exception as e:
                            logger.error(f"Error processing {img_path}: {e}")
                            rows_to_append.append((os.path.basename(img_path), "ERROR", f"Error: {e}", ""))

                    if rows_to_append:
                        append_scores_csv(csv_path, rows_to_append, threshold)

                time.sleep(poll_interval)

            except Exception as loop_e:
                logger.error(f"Watcher loop error: {loop_e}")
                time.sleep(poll_interval)

    finally:
        logger.info("Watcher stopped")
        logger.info(f"Total processed = {len(processed)}")
        logger.info(f"Results CSV: {csv_path}")


# =============================
# PatchCore Anomaly Detector (Integration Class)
# =============================

class PatchCoreDetector:
    """
    PatchCore-based anomaly detector for integration with existing system.

    Compatible with the AnomalyDetector interface.
    """

    def __init__(
        self,
        model_path: str,
        threshold_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize PatchCore detector.

        Args:
            model_path: Path to PatchCore model (.pth)
            threshold_path: Path to threshold configuration JSON
            use_gpu: Whether to use GPU if available
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        # Load threshold
        self.threshold = THRESHOLD
        if threshold_path and os.path.exists(threshold_path):
            try:
                with open(threshold_path, 'r') as f:
                    config = json.load(f)
                self.threshold = config.get('threshold', THRESHOLD)
                logger.info(f"Loaded threshold: {self.threshold}")
            except Exception as e:
                logger.warning(f"Failed to load threshold config: {e}")

        # Load model
        self.model = load_model(model_path, self.device)

        # Statistics
        self.inference_count = 0
        self.total_inference_time = 0

    def detect(
        self,
        image_path: str,
        return_heatmap: bool = False
    ) -> Dict[str, Any]:
        """
        Detect anomalies in image.

        Args:
            image_path: Path to image file
            return_heatmap: Whether to generate heatmap

        Returns:
            Detection result dictionary
        """
        result = {
            'is_normal': True,
            'anomaly_score': 0.0,
            'reconstruction_error': 0.0,
            'confidence': 1.0,
            'inference_time_ms': 0,
            'threshold': self.threshold,
            'image_path': image_path
        }

        try:
            start_time = time.perf_counter()

            # Run inference
            score, name, label, heatmap_path = run_inference(
                self.model,
                image_path,
                self.device,
                save_dir=os.path.dirname(image_path) or ".",
                threshold=self.threshold,
                generate_heatmap=return_heatmap
            )

            inference_time = (time.perf_counter() - start_time) * 1000

            result['inference_time_ms'] = inference_time
            result['anomaly_score'] = score
            result['reconstruction_error'] = score  # For compatibility
            result['is_normal'] = label == "Normal"
            result['confidence'] = abs(score - self.threshold) / (self.threshold + 1e-8)
            result['confidence'] = min(1.0, result['confidence'])

            if return_heatmap and heatmap_path:
                result['heatmap_path'] = heatmap_path

            self.inference_count += 1
            self.total_inference_time += inference_time

        except Exception as e:
            logger.error(f"Detection error: {e}")
            result['error'] = str(e)
            result['is_normal'] = False
            result['confidence'] = 0.0

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        avg_time = 0
        if self.inference_count > 0:
            avg_time = self.total_inference_time / self.inference_count

        return {
            'inference_count': self.inference_count,
            'avg_inference_time_ms': avg_time,
            'total_inference_time_ms': self.total_inference_time,
            'threshold': self.threshold,
            'device': str(self.device)
        }


# =============================
# Main Entry Point
# =============================

def main():
    """Interactive main entry point."""
    print("PatchCore Anomaly Detection System")
    print("=" * 40)
    print("1. Train")
    print("2. Inference")
    print("3. Watch Folder")
    choice = input("Enter 1, 2, or 3: ").strip()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if choice == "1":
        logger.info("Training mode selected")
        cls = input("Enter class name (e.g., bottle, cable): ").strip()

        backbone_weights = BACKBONE_WEIGHTS_PATH
        if not os.path.exists(backbone_weights):
            logger.warning(f"Backbone weights not found at {backbone_weights}")
            backbone_weights = input("Enter path to backbone weights (or leave empty): ").strip() or None

        model = create_model(backbone_weights_path=backbone_weights, device=device)
        train_loader, test_loader = MVTecDataset(cls).get_dataloaders()
        train_model(model, train_loader)

        out_model_path = f"models/patchcore_{cls}.pth"
        model.save(out_model_path)
        logger.info(f"Saved model to {out_model_path}")

        result = evaluate_model(model, test_loader, cls, model_dir="models")
        if result:
            rocauc, threshold = result
            logger.info(f"ROC-AUC: {rocauc:.4f}, Threshold: {threshold:.4f}")

    elif choice == "2":
        logger.info("Inference mode selected")
        model_path = input(f"Model path [{DEFAULT_MODEL_PATH}]: ").strip() or DEFAULT_MODEL_PATH
        input_folder = input("Image folder: ").strip() or "images"
        out_folder = input(f"Output folder [{DEFAULT_OUTPUT_FOLDER}]: ").strip() or DEFAULT_OUTPUT_FOLDER

        try:
            model = load_model(model_path, device)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            bw = BACKBONE_WEIGHTS_PATH if os.path.exists(BACKBONE_WEIGHTS_PATH) else None
            model = create_model(backbone_weights_path=bw, device=device)

        image_paths = sorted(
            glob.glob(os.path.join(input_folder, "*.png")) +
            glob.glob(os.path.join(input_folder, "*.jpg")) +
            glob.glob(os.path.join(input_folder, "*.jpeg"))
        )

        if not image_paths:
            logger.error("No images found")
            return

        all_rows = []
        scores = []

        for img_path in image_paths:
            score, name, label, heatmap_path = run_inference(model, img_path, device, out_folder)
            all_rows.append((name, score, label, heatmap_path))
            scores.append(score)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_path = os.path.join(out_folder, f"results_{timestamp}.csv")
        save_scores_csv(csv_path, all_rows)

        if scores:
            scores = np.array(scores)
            logger.info(f"Summary for {len(scores)} images:")
            logger.info(f"Average score: {scores.mean():.4f}")
            logger.info(f"Min score: {scores.min():.4f}")
            logger.info(f"Max score: {scores.max():.4f}")

    elif choice == "3":
        logger.info("Watch mode selected")
        model_path = input(f"Model path [{DEFAULT_MODEL_PATH}]: ").strip() or DEFAULT_MODEL_PATH
        input_folder = input("Folder to watch: ").strip() or "watch_input"
        out_folder = input(f"Output folder [{DEFAULT_OUTPUT_FOLDER}]: ").strip() or DEFAULT_OUTPUT_FOLDER
        thresh_in = input(f"Threshold [{THRESHOLD}]: ").strip()
        poll = input("Poll interval (seconds) [2.0]: ").strip()

        try:
            threshold_val = float(thresh_in) if thresh_in else THRESHOLD
        except Exception:
            threshold_val = THRESHOLD
        try:
            poll_interval = float(poll) if poll else 2.0
        except Exception:
            poll_interval = 2.0

        watch_folder(
            model_path=model_path,
            input_folder=input_folder,
            output_folder=out_folder,
            threshold=threshold_val,
            poll_interval=poll_interval,
            device=device
        )

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
