import os
import sys
import json
import numpy as np
from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from pathlib import Path

# Ensure the project root is importable so we can use the embed module
_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from embed import embed_stl  # noqa: E402

search_bp = Blueprint('search', __name__, url_prefix='/search')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


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

        # Compute query embedding via embed module
        result = embed_stl(upload_path, model="dinov2")
        query_embedding = result["embedding"]

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
