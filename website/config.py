import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'oasis-dev-secret-key-change-in-production')
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{BASE_DIR / 'database' / 'oasis.db'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Upload settings
    UPLOAD_FOLDER = str(BASE_DIR / 'static' / 'uploads')
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200MB max upload
    ALLOWED_EXTENSIONS = {'stl'}

    # Model settings
    STL_DATA_DIR = str(PROJECT_ROOT / 'collecting-data' / 'stlFiles')
    RENDERED_IMAGES_DIR = str(PROJECT_ROOT / 'collecting-data' / 'rendered_images')

    # DINOv2 settings
    DINOV2_MODEL_NAME = 'dinov2_vitl14'
    DINOV2_IMG_SIZE = 224
    DINOV2_VIEWS = ['front', 'left', 'right', 'top', 'bottom']
    EMBEDDING_DIM = 1024  # dinov2_vitl14 output dimension

    # Pre-computed DINOv2 embeddings (database)
    EMBEDDINGS_CACHE = str(PROJECT_ROOT / 'train' / 'dinov2_embeddings.npy')
    FILENAMES_CACHE = str(PROJECT_ROOT / 'train' / 'dinov2_filenames.json')

    # Rendering settings (must match render_multiview_final.py)
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

    TOP_K = 5
