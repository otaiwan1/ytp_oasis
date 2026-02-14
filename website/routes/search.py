import os
import json
import copy
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from pathlib import Path

search_bp = Blueprint('search', __name__, url_prefix='/search')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


# ─── DINOv2 model loading ───────────────────────────────────────────

def load_dinov2_model(model_name, device):
    """Load DINOv2 model from PyTorch Hub."""
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model.to(device)
    model.eval()
    return model


def get_dinov2_transforms(img_size=224):
    """Standard DINOv2 preprocessing (ImageNet normalization)."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])


# ─── STL → multi-view rendering ─────────────────────────────────────

def _get_combined_rotation_matrix(base_euler_deg, roll_deg):
    """Compute combined rotation matrix from Euler angles + roll."""
    rx, ry, rz = np.deg2rad(base_euler_deg)
    import open3d as o3d
    R_base = o3d.geometry.get_rotation_matrix_from_xyz((rx, ry, rz))
    roll_rad = np.deg2rad(roll_deg)
    R_roll = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad),  np.cos(roll_rad), 0],
        [0,                 0,                1]
    ])
    return np.matmul(R_roll, R_base)


def _render_view(renderer, mesh, view_cfg, fov_deg):
    """Render a single view of a mesh using offscreen renderer."""
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    mesh_copy = copy.deepcopy(mesh)
    R = _get_combined_rotation_matrix(view_cfg["base"], view_cfg["roll"])
    mesh_copy.rotate(R, center=(0, 0, 0))

    renderer.scene.clear_geometry()
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("tooth", mesh_copy, mat)

    bounds = mesh_copy.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_max_bound() - bounds.get_min_bound()
    max_len = np.max(extent)
    dist = (max_len / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0)) * 1.2
    eye = center + np.array([0, 0, dist])
    up = np.array([0, 1, 0])

    renderer.setup_camera(fov_deg,
                          center.astype(np.float32),
                          eye.astype(np.float32),
                          up.astype(np.float32))
    img = renderer.render_to_image()
    return np.asarray(img)


def render_stl_multiview(stl_path, views_config, views_order, render_size=512, fov_deg=60.0):
    """Render an STL file from multiple viewpoints.

    Returns a list of PIL Images (RGB) in the order given by *views_order*.
    """
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Empty mesh: {stl_path}")

    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())
    normals = np.asarray(mesh.vertex_normals)
    colors = (normals + 1) / 2.0
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    renderer = rendering.OffscreenRenderer(render_size, render_size)
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])

    images = []
    for view_name in views_order:
        cfg = views_config[view_name]
        img_np = _render_view(renderer, mesh, cfg, fov_deg)
        images.append(Image.fromarray(img_np).convert('RGB'))

    return images


# ─── Embedding computation ───────────────────────────────────────────

def get_dinov2_embedding(model, pil_images, transform, device):
    """Compute a single DINOv2 embedding from a list of view images.

    Mirrors the approach in extract_dinov2.py:
      1. Transform each view image
      2. Forward through DINOv2 as a batch
      3. Mean-pool across views
      4. L2-normalize
    """
    tensors = [transform(img) for img in pil_images]
    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        outputs = model(batch)                       # (N_views, emb_dim)
        embedding = torch.mean(outputs, dim=0)       # (emb_dim,)
        embedding = F.normalize(embedding, p=2, dim=0)

    return embedding.cpu().numpy()


# ─── Database embeddings ─────────────────────────────────────────────

def load_embeddings_db():
    """Load pre-computed DINOv2 embeddings and filenames."""
    cache_path = current_app.config['EMBEDDINGS_CACHE']
    filenames_path = current_app.config['FILENAMES_CACHE']

    if not os.path.exists(cache_path) or not os.path.exists(filenames_path):
        return np.array([]), []

    embeddings = np.load(cache_path)
    with open(filenames_path, 'r') as f:
        filenames = json.load(f)

    return embeddings, filenames


# ─── Similarity search ───────────────────────────────────────────────

def search_similar(query_embedding, db_embeddings, db_filenames, top_k=5):
    """Find top-k most similar scans using cosine similarity."""
    if len(db_embeddings) == 0:
        return []

    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    db_norms = db_embeddings / (np.linalg.norm(db_embeddings, axis=1, keepdims=True) + 1e-8)

    similarities = db_norms @ query_norm
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        filename = db_filenames[idx]
        parts = filename.replace('.stl', '').split('_', 1)
        patient_uid = parts[0] if parts else filename
        serial_number = parts[1] if len(parts) > 1 else ''

        results.append({
            'filename': filename,
            'patient_uid': patient_uid,
            'serial_number': serial_number,
            'similarity': float(similarities[idx]),
            'rank': len(results) + 1
        })

    return results


# ─── Global model cache (singleton) ──────────────────────────────────

_model_cache = {}


def get_cached_model():
    """Get or load the DINOv2 model (singleton)."""
    if 'model' not in _model_cache:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = current_app.config['DINOV2_MODEL_NAME']
        model = load_dinov2_model(model_name, device)
        img_size = current_app.config['DINOV2_IMG_SIZE']
        transform = get_dinov2_transforms(img_size)
        _model_cache['model'] = model
        _model_cache['device'] = device
        _model_cache['transform'] = transform
    return _model_cache['model'], _model_cache['device'], _model_cache['transform']


# ─── Flask routes ────────────────────────────────────────────────────

@search_bp.route('/')
@login_required
def search_page():
    return render_template('search/search.html')


@search_bp.route('/query', methods=['POST'])
@login_required
def query():
    """Handle similarity search query using DINOv2."""
    if 'stl_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['stl_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only .stl files are allowed'}), 400

    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'],
                                   f'query_{current_user.id}_{filename}')
        file.save(upload_path)

        # Load DINOv2 model
        model, device, transform = get_cached_model()

        # Render uploaded STL → multi-view images
        views_config = current_app.config['RENDER_VIEWS_CONFIG']
        views_order = current_app.config['DINOV2_VIEWS']
        render_size = current_app.config['RENDER_IMG_SIZE']
        fov_deg = current_app.config['RENDER_FOV_DEG']

        pil_images = render_stl_multiview(upload_path, views_config,
                                          views_order, render_size, fov_deg)

        # Compute query embedding
        query_embedding = get_dinov2_embedding(model, pil_images, transform, device)

        # Load pre-computed database embeddings
        db_embeddings, db_filenames = load_embeddings_db()

        if len(db_embeddings) == 0:
            return jsonify({
                'error': 'No DINOv2 embeddings available. '
                         'Please run extract_dinov2.py first.'
            }), 404

        # Search
        top_k = current_app.config['TOP_K']
        results = search_similar(query_embedding, db_embeddings,
                                 db_filenames, top_k)

        # Clean up temp file
        os.remove(upload_path)

        return jsonify({'results': results, 'query_file': filename})

    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'}), 500


@search_bp.route('/patient/<patient_uid>')
@login_required
def patient_scans(patient_uid):
    """Show all scans for a given patient."""
    stl_dir = Path(current_app.config['STL_DATA_DIR'])
    rendered_dir = Path(current_app.config['RENDERED_IMAGES_DIR'])

    scans = []
    if stl_dir.exists():
        for f in sorted(stl_dir.glob(f'{patient_uid}_*.stl')):
            fname = f.name
            parts = fname.replace('.stl', '').split('_', 1)
            serial = parts[1] if len(parts) > 1 else ''

            # Check for rendered images
            patient_render_dir = rendered_dir / patient_uid
            rendered_views = {}
            if patient_render_dir.exists():
                for view in ['front', 'back', 'top', 'bottom', 'left', 'right']:
                    img_path = patient_render_dir / f'{view}.png'
                    if img_path.exists():
                        rendered_views[view] = str(img_path)

            scans.append({
                'filename': fname,
                'serial_number': serial,
                'has_renders': bool(rendered_views),
                'render_views': list(rendered_views.keys())
            })

    return render_template('collection/patient.html',
                           patient_uid=patient_uid, scans=scans)
