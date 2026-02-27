"""
validation.py — Validation routes for the OASIS website.

/validation/          — Model selection page (pick model to validate)
/validation/judge     — Top-k validation with Pass/Fail verdicts

Supports multiple models: dinov2, simclr, mae.
Test/base scans are loaded from pre-existing JSON files:
  validation/validation_test_scans.json  (60 test cases)
  validation/validation_base_scans.json  (base cases)

Per-model files:
  validation/progress/validation_progress_{model}.json
  validation/reports/validation_report_{model}.json
"""

import json
import numpy as np
from pathlib import Path
from flask import (Blueprint, render_template, request, jsonify,
                   send_from_directory, current_app, abort, redirect,
                   url_for)

validation_bp = Blueprint('validation', __name__, url_prefix='/validation')

# ─── Supported models ────────────────────────────────────────────────
SUPPORTED_MODELS = {
    'dinov2': {'label': 'DINOv2',    'dim': 1024, 'desc': 'Vision Transformer (multi-view images)'},
    'simclr': {'label': 'SimCLR',    'dim': 512,  'desc': 'Contrastive learning (3D point cloud)'},
    'mae':    {'label': 'Point-MAE', 'dim': 384,  'desc': 'Masked autoencoder (3D point cloud + normals)'},
    'dinov3': {'label': 'DINOv3',    'dim': 1024, 'desc': 'DINOv3 ViT-L/16 (multi-view, Gram Anchoring)'},
    'dinov3_gallery': {'label': 'DINOv3 Gallery', 'dim': 2048, 'desc': 'DINOv3 hybrid pooling (iTero gallery photos)'},
}

# ─── Paths (relative to project root) ────────────────────────────────

def _project_root():
    return Path(current_app.config.get('PROJECT_ROOT_DIR',
                str(Path(__file__).parent.parent.parent.resolve())))


def _stl_dir():
    return _project_root() / 'collecting-data' / 'stlFiles'


def _glb_dir():
    return _project_root() / 'collecting-data' / 'glbFiles'


def _validation_dir():
    return _project_root() / 'validation'


def _embeddings_path(model_name):
    return _project_root() / 'train' / model_name / f'{model_name}_embeddings.npy'


def _filenames_path(model_name):
    return _project_root() / 'train' / model_name / f'{model_name}_filenames.json'


def _progress_path(model_name):
    return _validation_dir() / 'progress' / f'validation_progress_{model_name}.json'


def _report_path(model_name):
    return _validation_dir() / 'reports' / f'validation_report_{model_name}.json'


NUM_TEST = 60
TOP_K_VALUES = [1, 3, 5, 10]


def _validate_model(model_name):
    """Validate model name, return None if invalid."""
    if model_name and model_name in SUPPORTED_MODELS:
        return model_name
    return None


# ─────────────────────────────────────────────────────────────────────
#  Model selection page
# ─────────────────────────────────────────────────────────────────────

@validation_bp.route('/')
def model_select_page():
    """Render the model selection page."""
    models_info = []
    for name, meta in SUPPORTED_MODELS.items():
        emb_exists = _embeddings_path(name).exists()
        fn_exists = _filenames_path(name).exists()
        available = emb_exists and fn_exists

        # Load progress summary if exists
        prog_path = _progress_path(name)
        progress_summary = None
        if prog_path.exists():
            try:
                with open(prog_path) as f:
                    prog = json.load(f)
                # Count unique queries judged (check top_1 key)
                progress_summary = len(prog.get('top_1', {}))
            except Exception:
                pass

        # Load report summary if exists
        report_path = _report_path(name)
        report_summary = None
        if report_path.exists():
            try:
                with open(report_path) as f:
                    rep = json.load(f)
                report_summary = rep.get('top_k_results', {})
            except Exception:
                pass

        models_info.append({
            'name': name,
            'label': meta['label'],
            'desc': meta['desc'],
            'dim': meta['dim'],
            'available': available,
            'progress_count': progress_summary,
            'report': report_summary,
        })

    return render_template('validation/select_model.html', models=models_info)


# ─────────────────────────────────────────────────────────────────────
#  Legacy redirect — skip Step 1 (pick scans)
# ─────────────────────────────────────────────────────────────────────

@validation_bp.route('/pick')
def pick_page():
    """Redirect to model selection page."""
    return redirect(url_for('validation.model_select_page'))


# ─────────────────────────────────────────────────────────────────────
#  Top-k validation (judge)
# ─────────────────────────────────────────────────────────────────────

@validation_bp.route('/judge')
def judge_page():
    """Render the validation judge page for a specific model."""
    model_name = request.args.get('model', 'dinov2')
    if not _validate_model(model_name):
        return redirect(url_for('validation.model_select_page'))

    meta = SUPPORTED_MODELS[model_name]
    return render_template('validation/judge.html',
                           model_name=model_name,
                           model_label=meta['label'])


@validation_bp.route('/api/judge/state')
def api_judge_state():
    """Return current validation state: progress, test/base splits,
    similarity rankings for the current round."""
    model_name = request.args.get('model', 'dinov2')
    if not _validate_model(model_name):
        return jsonify({'error': f'Unknown model: {model_name}'}), 400

    vdir = _validation_dir()

    # Load test/base splits
    test_path = vdir / 'validation_test_scans.json'
    base_path = vdir / 'validation_base_scans.json'
    if not test_path.exists() or not base_path.exists():
        return jsonify({'error': 'Run Step 1 (pick scans) first'}), 400

    with open(test_path) as f:
        test_fnames = json.load(f)
    with open(base_path) as f:
        base_fnames = json.load(f)

    # Load embeddings for the selected model
    emb_path = _embeddings_path(model_name)
    ids_path = _filenames_path(model_name)
    if not emb_path.exists() or not ids_path.exists():
        return jsonify({'error': f'{SUPPORTED_MODELS[model_name]["label"]} embeddings not found'}), 404

    all_embeddings = np.load(str(emb_path))
    with open(ids_path) as f:
        all_filenames = json.load(f)
    fname_to_idx = {fn: i for i, fn in enumerate(all_filenames)}

    # Build index arrays
    test_indices = [fname_to_idx[fn] for fn in test_fnames if fn in fname_to_idx]
    base_indices = [fname_to_idx[fn] for fn in base_fnames if fn in fname_to_idx]

    test_embs = all_embeddings[test_indices]
    base_embs = all_embeddings[base_indices]

    test_fnames_valid = [all_filenames[i] for i in test_indices]
    base_fnames_valid = [all_filenames[i] for i in base_indices]

    # Compute full similarity matrix: test × base
    test_norms = test_embs / (np.linalg.norm(test_embs, axis=1, keepdims=True) + 1e-8)
    base_norms = base_embs / (np.linalg.norm(base_embs, axis=1, keepdims=True) + 1e-8)
    sim_matrix = test_norms @ base_norms.T  # (num_test, num_base)

    # For each test scan, get top-10 base results (covers all k values)
    max_k = max(TOP_K_VALUES)
    rankings = {}
    for i, qname in enumerate(test_fnames_valid):
        sims = sim_matrix[i]
        top_idx = np.argsort(sims)[::-1][:max_k]
        results = []
        for j in top_idx:
            results.append({
                'filename': base_fnames_valid[j],
                'similarity': round(float(sims[j]), 6)
            })
        rankings[qname] = results

    # Load per-model progress
    progress_path = _progress_path(model_name)
    progress = {}
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)

    return jsonify({
        'model': model_name,
        'model_label': SUPPORTED_MODELS[model_name]['label'],
        'test_scans': test_fnames_valid,
        'top_k_values': TOP_K_VALUES,
        'rankings': rankings,
        'progress': progress
    })


@validation_bp.route('/api/judge/verdict', methods=['POST'])
def api_judge_verdict():
    """Save verdicts for a query based on the chosen rank.

    Accepts chosen_rank (1-10) or 0 for fail-all.  Rules:
      rank 1       → pass top-1,3,5,10
      rank 2-3     → pass top-3,5,10; fail top-1
      rank 4-5     → pass top-5,10; fail top-1,3
      rank 6-10    → pass top-10; fail top-1,3,5
      rank 0 (fail)→ fail all
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data'}), 400

    model_name = data.get('model', 'dinov2')
    if not _validate_model(model_name):
        return jsonify({'error': f'Unknown model: {model_name}'}), 400

    query_file = data.get('query_file')
    chosen_rank = data.get('chosen_rank')  # 1-10 or 0
    results = data.get('results', [])      # full top-10 results

    if not query_file or chosen_rank is None:
        return jsonify({'error': 'Missing fields'}), 400

    chosen_rank = int(chosen_rank)

    # Determine verdict for each top-k
    verdicts = {}
    for k in TOP_K_VALUES:
        if chosen_rank == 0:
            verdicts[f'top_{k}'] = 'fail'
        elif chosen_rank <= k:
            verdicts[f'top_{k}'] = 'pass'
        else:
            verdicts[f'top_{k}'] = 'fail'

    progress_path = _progress_path(model_name)
    progress = {}
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)

    for key, verdict in verdicts.items():
        if key not in progress:
            progress[key] = {}
        k_val = int(key.split('_')[1])
        progress[key][query_file] = {
            'verdict': verdict,
            'chosen_rank': chosen_rank,
            'results': results[:k_val]
        }

    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)

    return jsonify({'ok': True, 'verdicts': verdicts})


@validation_bp.route('/review')
def review_page():
    """Render the validation review page for a specific model."""
    model_name = request.args.get('model', 'dinov2')
    if not _validate_model(model_name):
        return redirect(url_for('validation.model_select_page'))

    meta = SUPPORTED_MODELS[model_name]
    return render_template('validation/review.html',
                           model_name=model_name,
                           model_label=meta['label'])


@validation_bp.route('/api/judge/report', methods=['POST'])
def api_generate_report():
    """Generate final validation report for a specific model."""
    data = request.get_json() or {}
    model_name = data.get('model', 'dinov2')
    if not _validate_model(model_name):
        return jsonify({'error': f'Unknown model: {model_name}'}), 400

    progress_path = _progress_path(model_name)

    if not progress_path.exists():
        return jsonify({'error': 'No progress data'}), 400

    with open(progress_path) as f:
        progress = json.load(f)

    report = {
        'model': model_name,
        'model_label': SUPPORTED_MODELS[model_name]['label'],
        'top_k_results': {},
        'details': {}
    }

    for k in TOP_K_VALUES:
        key = f'top_{k}'
        if key not in progress:
            continue
        verdicts = progress[key]
        total = len(verdicts)
        passes = sum(1 for v in verdicts.values() if v['verdict'] == 'pass')
        fails = total - passes
        rate = (passes / total * 100) if total > 0 else 0.0

        report['top_k_results'][key] = {
            'total_queries': total,
            'passes': passes,
            'fails': fails,
            'pass_rate_percent': round(rate, 2)
        }
        report['details'][key] = verdicts

    report_path = _report_path(model_name)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    return jsonify({'ok': True, 'report': report})


@validation_bp.route('/api/judge/reset', methods=['POST'])
def api_reset_progress():
    """Reset validation progress for a specific model."""
    data = request.get_json() or {}
    model_name = data.get('model', 'dinov2')
    if not _validate_model(model_name):
        return jsonify({'error': f'Unknown model: {model_name}'}), 400

    progress_path = _progress_path(model_name)
    if progress_path.exists():
        with open(progress_path, 'w') as f:
            json.dump({}, f)

    return jsonify({'ok': True})


# ─────────────────────────────────────────────────────────────────────
#  Serve 3D model files for viewing
# ─────────────────────────────────────────────────────────────────────

@validation_bp.route('/glb/<filename>')
def serve_glb(filename):
    """Serve a pre-compressed GLB file (gzip-compressed on disk).

    The files are stored as gzip-compressed GLB.  We set the appropriate
    Content-Encoding header so the browser decompresses transparently.
    """
    from flask import make_response
    glb_dir = _glb_dir()
    # Filename coming in is like "scan_name.glb", stored as "scan_name.glb" (gzipped)
    stem = filename.replace('.glb', '')
    disk_name = stem + '.glb'
    fpath = glb_dir / disk_name
    if not fpath.exists():
        abort(404)
    data = fpath.read_bytes()
    resp = make_response(data)
    resp.headers['Content-Type'] = 'model/gltf-binary'
    resp.headers['Content-Encoding'] = 'gzip'
    resp.headers['Cache-Control'] = 'public, max-age=604800'  # 7 days
    return resp


@validation_bp.route('/stl/<filename>')
def serve_stl(filename):
    """Serve an STL file from the data directory."""
    stl_dir = _stl_dir()
    fpath = stl_dir / filename
    if not fpath.exists():
        abort(404)
    return send_from_directory(str(stl_dir), filename,
                               mimetype='application/octet-stream')
