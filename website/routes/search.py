import os
import json
import numpy as np
import torch
from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from pathlib import Path

search_bp = Blueprint('search', __name__, url_prefix='/search')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


def load_model(model_path, device):
    """Load the SimCLR encoder model."""
    from models.encoder import SimCLREncoder

    model = SimCLREncoder(
        k=current_app.config['K_NEIGHBORS'],
        emb_dim=current_app.config['EMBEDDING_DIM']
    )

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    # Handle DataParallel saved models
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)
    return model


def stl_to_point_cloud(stl_path, num_points=4096):
    """Convert an STL file to a normalized point cloud."""
    import trimesh
    mesh = trimesh.load(stl_path, force='mesh')
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    points = np.array(points, dtype=np.float32)

    # Normalize: center and scale
    centroid = points.mean(axis=0)
    points -= centroid
    max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if max_dist > 0:
        points /= max_dist

    return points


def get_embedding(model, point_cloud, device):
    """Get embedding vector from a point cloud."""
    pc_tensor = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.backbone(pc_tensor)
    return embedding.cpu().numpy().flatten()


def build_embeddings_cache(model, device):
    """Build embedding cache for all STL files in the data directory."""
    stl_dir = Path(current_app.config['STL_DATA_DIR'])
    cache_path = current_app.config['EMBEDDINGS_CACHE']
    filenames_path = current_app.config['FILENAMES_CACHE']
    num_points = current_app.config['NUM_POINTS']

    if not stl_dir.exists():
        return np.array([]), []

    stl_files = sorted([f.name for f in stl_dir.glob('*.stl')])
    if not stl_files:
        return np.array([]), []

    # Check if cache exists and is up to date
    if os.path.exists(cache_path) and os.path.exists(filenames_path):
        with open(filenames_path, 'r') as f:
            cached_filenames = json.load(f)
        if cached_filenames == stl_files:
            embeddings = np.load(cache_path)
            return embeddings, cached_filenames

    # Build new cache
    print(f"Building embeddings cache for {len(stl_files)} files...")
    embeddings = []
    valid_filenames = []

    for fname in stl_files:
        try:
            pc = stl_to_point_cloud(str(stl_dir / fname), num_points)
            emb = get_embedding(model, pc, device)
            embeddings.append(emb)
            valid_filenames.append(fname)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    if embeddings:
        embeddings = np.stack(embeddings)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings)
        with open(filenames_path, 'w') as f:
            json.dump(valid_filenames, f)
    else:
        embeddings = np.array([])

    return embeddings, valid_filenames


def search_similar(query_embedding, db_embeddings, db_filenames, top_k=5):
    """Find top-k most similar scans using cosine similarity."""
    if len(db_embeddings) == 0:
        return []

    # Normalize
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    db_norms = db_embeddings / (np.linalg.norm(db_embeddings, axis=1, keepdims=True) + 1e-8)

    similarities = db_norms @ query_norm
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        filename = db_filenames[idx]
        # Parse patientUID-serialNumber format
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


# Global model cache
_model_cache = {}


def get_cached_model():
    """Get or load the model (singleton pattern)."""
    if 'model' not in _model_cache:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model(current_app.config['MODEL_PATH'], device)
        _model_cache['model'] = model
        _model_cache['device'] = device
    return _model_cache['model'], _model_cache['device']


@search_bp.route('/')
@login_required
def search_page():
    return render_template('search/search.html')


@search_bp.route('/query', methods=['POST'])
@login_required
def query():
    """Handle similarity search query."""
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
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f'query_{current_user.id}_{filename}')
        file.save(upload_path)

        # Load model and compute embedding
        model, device = get_cached_model()
        num_points = current_app.config['NUM_POINTS']

        query_pc = stl_to_point_cloud(upload_path, num_points)
        query_embedding = get_embedding(model, query_pc, device)

        # Build/load embeddings cache
        db_embeddings, db_filenames = build_embeddings_cache(model, device)

        if len(db_embeddings) == 0:
            return jsonify({
                'error': 'No scan data available in the database. Please add STL files to the data directory first.'
            }), 404

        # Search
        top_k = current_app.config['TOP_K']
        results = search_similar(query_embedding, db_embeddings, db_filenames, top_k)

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
