"""
embed/config.py — Centralized path constants and configuration for the embed module.
"""

from pathlib import Path

# ─── Project layout ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
STL_DIR = PROJECT_ROOT / "collecting-data" / "stlFiles"

# ─── Checkpoint paths ────────────────────────────────────────────────
SIMCLR_CHECKPOINT = PROJECT_ROOT / "train" / "simclr" / "best_dental_simclr_multi_gpu.pth"
MAE_CHECKPOINT = PROJECT_ROOT / "train" / "mae" / "best_point_mae_ddp.pth"

# ─── Normalization / cache outputs ───────────────────────────────────
NORMALIZATION_DIR = PROJECT_ROOT / "normalization"

# ─── DINOv2 settings ─────────────────────────────────────────────────
DINOV2_MODEL_NAME = "dinov2_vitl14"
DINOV2_IMG_SIZE = 224
DINOV2_EMBEDDING_DIM = 1024

# ─── View rendering settings (matches render_multiview_final.py) ─────
RENDER_IMG_SIZE = 512
RENDER_FOV_DEG = 60.0

RENDER_VIEWS_CONFIG = {
    "front":  {"base": [-90, 0, 0],  "roll": 0},
    "back":   {"base": [90, 0, 0],   "roll": 0},
    "top":    {"base": [0, 0, 0],    "roll": 0},
    "bottom": {"base": [0, 180, 0],  "roll": 0},
    "right":  {"base": [0, 90, 0],   "roll": 90},
    "left":   {"base": [0, -90, 0],  "roll": -90},
}

# Views used for DINOv2 embedding (no "back")
DINOV2_VIEWS = ["front", "left", "right", "top", "bottom"]

# ─── Point cloud settings ────────────────────────────────────────────
NUM_POINTS = 4096
SIMCLR_INITIAL_SAMPLE = 100_000
MAE_INITIAL_SAMPLE = 50_000

# ─── First-scans JSON (used by DINOv2 batch) ────────────────────────
FIRST_SCANS_JSON = PROJECT_ROOT / "collecting-data" / "first_scans.json"
RENDERED_IMAGES_DIR = PROJECT_ROOT / "collecting-data" / "rendered_images"
