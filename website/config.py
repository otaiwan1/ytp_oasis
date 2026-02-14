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
    MODEL_PATH = str(PROJECT_ROOT / 'train' / 'best_dental_simclr_multi_gpu.pth')
    STL_DATA_DIR = str(PROJECT_ROOT / 'collecting-data' / 'stlFiles')
    RENDERED_IMAGES_DIR = str(PROJECT_ROOT / 'collecting-data' / 'rendered_images')
    EMBEDDINGS_CACHE = str(BASE_DIR / 'database' / 'embeddings_cache.npy')
    FILENAMES_CACHE = str(BASE_DIR / 'database' / 'filenames_cache.json')

    # Model hyperparameters (must match training)
    NUM_POINTS = 4096
    EMBEDDING_DIM = 512
    K_NEIGHBORS = 20
    TOP_K = 5
