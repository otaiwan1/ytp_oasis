import os
import sys
import json
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from pathlib import Path
import torch.multiprocessing as mp

# Ensure the project root is importable
_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

search_bp = Blueprint('search', __name__, url_prefix='/search')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


# ─── Subprocess embedding (GPU-free web server) ─────────────────────

def _embed_worker(stl_path, model_name, result_dict):
    """Run in a child process — loads model, computes embedding, exits."""
    from embed import embed_stl
    result = embed_stl(stl_path, model=model_name)
    result_dict["embedding"] = result["embedding"].tolist()
    result_dict["filename"] = result["filename"]


def embed_in_subprocess(stl_path, model_name="dinov3_gallery"):
    """Spawn a child process for inference so the web server stays GPU-free."""
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    result_dict = manager.dict()

    p = ctx.Process(target=_embed_worker, args=(str(stl_path), model_name, result_dict))
    p.start()
    p.join()

    if p.exitcode != 0:
        raise RuntimeError(f"Embedding subprocess failed with exit code {p.exitcode}")

    return {
        "embedding": np.array(result_dict["embedding"], dtype=np.float32),
        "filename": result_dict["filename"],
    }


# ─── Zip processing ─────────────────────────────────────────────────

def _parse_zip_info(zip_path):
    """Extract patient_uuid and scan_id from an iTero zip file.

    Reads the XML report inside the zip to get UniquePatientIdentifier.
    Gets scan_id from the zip filename: OrthoCAD_Export_{scan_id}.zip
    """
    zip_path = Path(zip_path)

    # Get scan_id from filename (may have prefix from secure_filename)
    m = re.search(r'OrthoCAD_Export_(\d+)\.zip', zip_path.name)
    if not m:
        raise ValueError(f"Zip filename doesn't contain OrthoCAD_Export_{{scan_id}}.zip: {zip_path.name}")
    scan_id = m.group(1)

    # Get patient UUID from XML inside zip
    patient_uuid = None
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        xml_members = [n for n in zf.namelist() if n.lower().endswith('.xml')]
        for xml_name in xml_members:
            try:
                with zf.open(xml_name) as xf:
                    tree = ET.parse(xf)
                    root = tree.getroot()
                    for elem in root.iter('UniquePatientIdentifier'):
                        if elem.text and elem.text.strip():
                            patient_uuid = elem.text.strip()
                            break
                if patient_uuid:
                    break
            except ET.ParseError:
                continue

    if not patient_uuid:
        raise ValueError("Could not find UniquePatientIdentifier in the zip's XML report")

    return patient_uuid, scan_id


def _process_upload(zip_path, app_config):
    """Process an uploaded iTero zip:
    1. Parse patient_uuid and scan_id from zip (XML + filename)
    2. Save zip to scanfiles/{patient_uuid}/
    3. Extract STL to stlFiles/
    4. Convert STL to gzipped GLB in glbFiles/
    5. Compute DINOv3 Gallery embedding
    6. Append embedding to database

    Returns:
        dict with 'embedding', 'stl_filename'
    """
    from embed.convert import stl_to_glb_gzipped

    patient_uuid, scan_id = _parse_zip_info(zip_path)
    stl_filename = f"{patient_uuid}_{scan_id}.stl"

    scanfiles_dir = Path(app_config['SCANFILES_DIR'])
    stl_dir = Path(app_config['STL_DATA_DIR'])
    glb_dir = Path(app_config['GLB_DATA_DIR'])

    # 1. Copy zip to scanfiles/{patient_uuid}/
    patient_scan_dir = scanfiles_dir / patient_uuid
    patient_scan_dir.mkdir(parents=True, exist_ok=True)
    dest_zip = patient_scan_dir / Path(zip_path).name
    if not dest_zip.exists():
        shutil.copy2(str(zip_path), str(dest_zip))

    # 2. Extract STL from zip
    stl_dir.mkdir(parents=True, exist_ok=True)
    stl_output = stl_dir / stl_filename

    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        # Find the STL file inside the zip
        stl_members = [n for n in zf.namelist() if n.lower().endswith('.stl')]
        if not stl_members:
            raise FileNotFoundError("No .stl file found inside the zip")
        # Extract the first STL
        with zf.open(stl_members[0]) as src, open(str(stl_output), 'wb') as dst:
            dst.write(src.read())

    # 3. Convert STL to gzipped GLB
    glb_dir.mkdir(parents=True, exist_ok=True)
    glb_output = glb_dir / f"{patient_uuid}_{scan_id}.glb"
    stl_to_glb_gzipped(stl_output, glb_output)

    # 4. Compute embedding (subprocess, GPU-free)
    model_name = app_config.get('SEARCH_MODEL', 'dinov3_gallery')
    result = embed_in_subprocess(str(stl_output), model_name=model_name)
    embedding = result["embedding"]

    # 5. Append to database
    emb_path = Path(app_config['EMBEDDINGS_CACHE'])
    fn_path = Path(app_config['FILENAMES_CACHE'])
    emb_path.parent.mkdir(parents=True, exist_ok=True)

    if emb_path.exists() and fn_path.exists():
        all_embs = np.load(str(emb_path))
        with open(str(fn_path)) as f:
            all_fnames = json.load(f)

        # Skip if already exists
        if stl_filename not in all_fnames:
            all_embs = np.vstack([all_embs, embedding.reshape(1, -1)])
            all_fnames.append(stl_filename)
    else:
        all_embs = embedding.reshape(1, -1)
        all_fnames = [stl_filename]

    np.save(str(emb_path), all_embs)
    with open(str(fn_path), 'w') as f:
        json.dump(all_fnames, f)

    return {"embedding": embedding, "stl_filename": stl_filename}


# ─── Database embeddings ─────────────────────────────────────────────

def load_embeddings_db(app_config):
    """Load pre-computed embeddings and filenames."""
    cache_path = app_config['EMBEDDINGS_CACHE']
    filenames_path = app_config['FILENAMES_CACHE']

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
            'rank': len(results) + 1,
            'glb_url': f"/validation/glb/{filename.replace('.stl', '.glb')}",
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
    """Handle similarity search: upload zip → extract → embed → search."""
    if 'zip_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['zip_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only .zip files are allowed'}), 400

    try:
        # Save uploaded zip temporarily
        filename = secure_filename(file.filename)
        upload_dir = Path(current_app.config['UPLOAD_FOLDER'])
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / f'query_{current_user.id}_{filename}'
        file.save(str(upload_path))

        # Process: parse UUID from XML, extract STL, convert GLB, embed
        result = _process_upload(str(upload_path), current_app.config)
        query_embedding = result["embedding"]
        query_stl = result["stl_filename"]

        # Load full database (now includes the new scan)
        db_embeddings, db_filenames = load_embeddings_db(current_app.config)

        if len(db_embeddings) == 0:
            return jsonify({'error': 'No embeddings available.'}), 404

        # Search (exclude the query itself from results)
        top_k = int(request.form.get('top_k', current_app.config['TOP_K']))
        all_results = search_similar(query_embedding, db_embeddings,
                                     db_filenames, top_k + 1)
        # Filter out self
        results = [r for r in all_results if r['filename'] != query_stl][:top_k]
        # Re-rank
        for i, r in enumerate(results):
            r['rank'] = i + 1

        # Clean up temp file
        os.remove(str(upload_path))

        return jsonify({
            'results': results,
            'query_file': filename,
            'query_stl': query_stl,
            'query_glb_url': f"/validation/glb/{query_stl.replace('.stl', '.glb')}",
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
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
