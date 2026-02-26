"""
embed/models.py — Model loading & inference for SimCLR, PointMAE, and DINOv2.

Provides:
    infer_simclr(model, point_cloud, device) → np.ndarray (512,)
    infer_mae(model, point_cloud, device) → np.ndarray (384,)
    infer_dinov2(model, pil_images, device) → np.ndarray (1024,)

Models are cached in _model_cache so they are loaded once per process.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from pathlib import Path

from .config import (
    SIMCLR_CHECKPOINT,
    MAE_CHECKPOINT,
    DINOV2_MODEL_NAME,
    DINOV2_IMG_SIZE,
    PROJECT_ROOT,
)

# Singleton model cache
_model_cache: dict = {}

# Ensure train/ is importable
_train_dir = str(PROJECT_ROOT / "train")
if _train_dir not in sys.path:
    sys.path.insert(0, _train_dir)


# ─── SimCLR ──────────────────────────────────────────────────────────

def _load_simclr(device):
    """Load SimCLREncoder with trained weights."""
    from train_oasis import SimCLREncoder

    model = SimCLREncoder().to(device)
    state_dict = torch.load(str(SIMCLR_CHECKPOINT), map_location=device)
    new_state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def infer_simclr(model, point_cloud, device):
    """
    Run SimCLR backbone inference on a single point cloud.

    Args:
        model: SimCLREncoder instance.
        point_cloud: np.ndarray (num_points, 3).
        device: torch.device.
    Returns:
        np.ndarray of shape (512,), L2-normalized.
    """
    pts = torch.from_numpy(point_cloud).float().unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.backbone(pts)                    # (1, 512)
        emb = F.normalize(emb, p=2, dim=1)
    return emb.squeeze(0).cpu().numpy()


# ─── PointMAE ────────────────────────────────────────────────────────

def _load_mae(device):
    """Load PointMAE with trained weights."""
    from train_mae_ddp import PointMAE, config as mae_config

    # Detect in_channels from checkpoint
    state_dict = torch.load(str(MAE_CHECKPOINT), map_location=device)
    new_state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    # Infer in_channels from patch_embed.conv first layer weight shape
    first_conv_key = "patch_embed.conv.0.weight"
    in_channels = new_state_dict[first_conv_key].shape[1] if first_conv_key in new_state_dict else 6

    model = PointMAE(mae_config, in_channels=in_channels).to(device)
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def infer_mae(model, point_cloud, device):
    """
    Run PointMAE embedding inference on a single point cloud.

    Args:
        model: PointMAE instance.
        point_cloud: np.ndarray (num_points, 6) — XYZ + Normals.
        device: torch.device.
    Returns:
        np.ndarray of shape (384,).  Raw embedding (no whitening).
    """
    pts = torch.from_numpy(point_cloud).float().unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.get_embedding(pts)  # (1, 384)
    return emb.squeeze(0).cpu().numpy()


# ─── DINOv2 ──────────────────────────────────────────────────────────

def _load_dinov2(device):
    """Load DINOv2 from PyTorch Hub."""
    model = torch.hub.load("facebookresearch/dinov2", DINOV2_MODEL_NAME)
    model.to(device)
    model.eval()
    return model


def _get_dinov2_transform():
    """Standard DINOv2 preprocessing (ImageNet normalization)."""
    return T.Compose([
        T.Resize((DINOV2_IMG_SIZE, DINOV2_IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def infer_dinov2(model, pil_images, device):
    """
    Compute a DINOv2 embedding from multi-view PIL images.

    Pipeline: transform → batch forward → mean-pool → L2-normalize.

    Args:
        model: DINOv2 nn.Module.
        pil_images: list[PIL.Image] (RGB).
        device: torch.device.
    Returns:
        np.ndarray of shape (1024,), L2-normalized.
    """
    transform = _get_dinov2_transform()
    tensors = [transform(img) for img in pil_images]
    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        outputs = model(batch)                       # (N_views, emb_dim)
        embedding = torch.mean(outputs, dim=0)       # (emb_dim,)
        embedding = F.normalize(embedding, p=2, dim=0)

    return embedding.cpu().numpy()


# ─── Cached loader ───────────────────────────────────────────────────

def get_model(model_name: str, device=None):
    """
    Get or load a model (singleton).

    Args:
        model_name: one of "simclr", "mae", "dinov2".
        device: torch.device or None (auto-detect).
    Returns:
        (model, device) tuple.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_key = f"{model_name}_{device}"
    if cache_key not in _model_cache:
        loaders = {
            "simclr": _load_simclr,
            "mae": _load_mae,
            "dinov2": _load_dinov2,
        }
        if model_name not in loaders:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(loaders)}")
        _model_cache[cache_key] = loaders[model_name](device)

    return _model_cache[cache_key], device
