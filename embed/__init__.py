"""
embed — Unified embedding module for dental STL files.

Public API:
    embed_stl(stl_path, model="dinov2", device=None) → dict
    list_models() → list[str]

Example:
    >>> from embed import embed_stl, list_models
    >>> print(list_models())
    ['simclr', 'mae', 'dinov2', 'dinov3']
    >>> result = embed_stl("collecting-data/stlFiles/abc_1.stl", model="dinov2")
    >>> print(result["embedding"].shape, result["filename"])
"""

from pathlib import Path
import numpy as np

from .models import get_model, infer_simclr, infer_mae, infer_dinov2, infer_dinov3
from .preprocessing import preprocess_simclr, preprocess_mae, preprocess_dinov2
from .config import (RENDER_VIEWS_CONFIG, DINOV2_VIEWS, DINOV3_VIEWS,
                      RENDER_IMG_SIZE, RENDER_FOV_DEG)

__all__ = ["embed_stl", "list_models"]

_SUPPORTED_MODELS = ["simclr", "mae", "dinov2", "dinov3"]

# Expected output dimensions per model
_EMBED_DIMS = {
    "simclr": 512,
    "mae": 384,
    "dinov2": 1024,
    "dinov3": 1024,
}


def list_models() -> list:
    """Return the list of supported model names."""
    return list(_SUPPORTED_MODELS)


def embed_stl(stl_path, model="dinov2", device=None) -> dict:
    """
    Embed a single STL file using the specified model.

    Args:
        stl_path: path to an .stl file (str or Path).
        model: one of "simclr", "mae", "dinov2".
        device: torch.device or None (auto-detect GPU/CPU).

    Returns:
        dict with keys:
            "embedding": np.ndarray — the embedding vector.
            "filename":  str — bare filename (e.g. "abc_1.stl").
            "model":     str — model name used.
            "dim":       int — embedding dimensionality.
    """
    model_name = model.lower()
    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {_SUPPORTED_MODELS}"
        )

    stl_path = Path(stl_path)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")

    filename = stl_path.name

    # Load (or retrieve cached) model
    net, dev = get_model(model_name, device)

    # Preprocess + infer
    if model_name == "simclr":
        point_cloud = preprocess_simclr(stl_path)
        embedding = infer_simclr(net, point_cloud, dev)

    elif model_name == "mae":
        point_cloud = preprocess_mae(stl_path)
        embedding = infer_mae(net, point_cloud, dev)

    elif model_name == "dinov2":
        pil_images = preprocess_dinov2(
            stl_path,
            views_config=RENDER_VIEWS_CONFIG,
            views_order=DINOV2_VIEWS,
            render_size=RENDER_IMG_SIZE,
            fov_deg=RENDER_FOV_DEG,
        )
        embedding = infer_dinov2(net, pil_images, dev)

    elif model_name == "dinov3":
        pil_images = preprocess_dinov2(
            stl_path,
            views_config=RENDER_VIEWS_CONFIG,
            views_order=DINOV3_VIEWS,
            render_size=RENDER_IMG_SIZE,
            fov_deg=RENDER_FOV_DEG,
        )
        embedding = infer_dinov3(net, pil_images, dev)

    return {
        "embedding": embedding,
        "filename": filename,
        "model": model_name,
        "dim": int(embedding.shape[0]),
    }
