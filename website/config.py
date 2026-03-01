import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent

class Config:
    PROJECT_ROOT_DIR = str(PROJECT_ROOT)
    SECRET_KEY = os.environ.get('SECRET_KEY', 'oasis-dev-secret-key-change-in-production')
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{BASE_DIR / 'database' / 'oasis.db'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Upload settings
    UPLOAD_FOLDER = str(BASE_DIR / 'static' / 'uploads')
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200MB max upload
    ALLOWED_EXTENSIONS = {'zip'}

    # Data directories
    STL_DATA_DIR = str(PROJECT_ROOT / 'collecting-data' / 'stlFiles')
    GLB_DATA_DIR = str(PROJECT_ROOT / 'collecting-data' / 'glbFiles')
    SCANFILES_DIR = str(PROJECT_ROOT / 'collecting-data' / 'scanfiles')
    RENDERED_IMAGES_DIR = str(PROJECT_ROOT / 'collecting-data' / 'rendered_images')

    # Search model: DINOv3 Gallery (hybrid pooling on iTero photos)
    SEARCH_MODEL = 'dinov3_gallery'
    EMBEDDING_DIM = 2048

    # Pre-computed embeddings (database)
    EMBEDDINGS_CACHE = str(PROJECT_ROOT / 'train' / 'dinov3_gallery' / 'dinov3_gallery_embeddings.npy')
    FILENAMES_CACHE = str(PROJECT_ROOT / 'train' / 'dinov3_gallery' / 'dinov3_gallery_filenames.json')

    TOP_K = 5
