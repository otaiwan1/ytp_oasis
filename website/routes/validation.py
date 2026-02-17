"""
validation.py — Temporary validation routes for the OASIS website.

Step 1: /validation/pick   — Pick 60 test scans with 3D preview
Step 2: /validation/judge  — Top-k validation with Pass/Fail verdicts
"""

import json
import random
import numpy as np
from pathlib import Path
from flask import (Blueprint, render_template, request, jsonify,
                   send_from_directory, current_app, abort)

validation_bp = Blueprint('validation', __name__, url_prefix='/validation')

# ─── Paths (relative to project root) ────────────────────────────────

def _project_root():
    return Path(current_app.config.get('PROJECT_ROOT_DIR',
                str(Path(__file__).parent.parent.parent.resolve())))


def _stl_dir():
    return _project_root() / 'collecting-data' / 'stlFiles'


def _validation_dir():
    return _project_root() / 'validation'


def _dinov2_filenames_path():
    return _project_root() / 'train' / 'dinov2_filenames.json'


def _dinov2_embeddings_path():
    return _project_root() / 'train' / 'dinov2_embeddings.npy'


NUM_TEST = 60
TOP_K_VALUES = [1, 3, 5, 10]


# ─────────────────────────────────────────────────────────────────────
#  Step 1 — Pick test scans
# ─────────────────────────────────────────────────────────────────────

@validation_bp.route('/pick')
def pick_page():
    """Render the scan picker page."""
    return render_template('validation/pick.html')


@validation_bp.route('/api/filenames')
def api_filenames():
    """Return all DINOv2-eligible filenames."""
    path = _dinov2_filenames_path()
    if not path.exists():
        return jsonify({'error': 'dinov2_filenames.json not found'}), 404
    with open(path) as f:
        fnames = json.load(f)
    return jsonify({'filenames': fnames, 'num_test': NUM_TEST})


@validation_bp.route('/api/pick/save', methods=['POST'])
def api_pick_save():
    """Save the test/base split."""
    data = request.get_json()
    if not data or 'test' not in data or 'base' not in data:
        return jsonify({'error': 'Missing test/base arrays'}), 400

    test_scans = sorted(data['test'])
    base_scans = sorted(data['base'])

    vdir = _validation_dir()
    vdir.mkdir(parents=True, exist_ok=True)

    with open(vdir / 'validation_test_scans.json', 'w') as f:
        json.dump(test_scans, f, indent=2)
    with open(vdir / 'validation_base_scans.json', 'w') as f:
        json.dump(base_scans, f, indent=2)

    return jsonify({
        'ok': True,
        'test_count': len(test_scans),
        'base_count': len(base_scans)
    })


# ─────────────────────────────────────────────────────────────────────
#  Step 2 — Top-k validation (judge)
# ─────────────────────────────────────────────────────────────────────

@validation_bp.route('/judge')
def judge_page():
    """Render the validation judge page."""
    return render_template('validation/judge.html')


@validation_bp.route('/api/judge/state')
def api_judge_state():
    """Return current validation state: progress, test/base splits,
    similarity rankings for the current round."""
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

    # Load embeddings
    emb_path = _dinov2_embeddings_path()
    ids_path = _dinov2_filenames_path()
    if not emb_path.exists() or not ids_path.exists():
        return jsonify({'error': 'DINOv2 embeddings not found'}), 404

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
    # Normalize
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

    # Load progress
    progress_path = vdir / 'validation_progress.json'
    progress = {}
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)

    return jsonify({
        'test_scans': test_fnames_valid,
        'top_k_values': TOP_K_VALUES,
        'rankings': rankings,
        'progress': progress
    })


@validation_bp.route('/api/judge/verdict', methods=['POST'])
def api_judge_verdict():
    """Save a single verdict."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data'}), 400

    top_k_key = data.get('top_k_key')      # e.g. "top_1"
    query_file = data.get('query_file')
    verdict = data.get('verdict')            # "pass" or "fail"
    results = data.get('results', [])

    if not all([top_k_key, query_file, verdict]):
        return jsonify({'error': 'Missing fields'}), 400

    vdir = _validation_dir()
    progress_path = vdir / 'validation_progress.json'

    progress = {}
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)

    if top_k_key not in progress:
        progress[top_k_key] = {}

    progress[top_k_key][query_file] = {
        'verdict': verdict,
        'results': results
    }

    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)

    return jsonify({'ok': True})


@validation_bp.route('/api/judge/report', methods=['POST'])
def api_generate_report():
    """Generate final validation report."""
    vdir = _validation_dir()
    progress_path = vdir / 'validation_progress.json'

    if not progress_path.exists():
        return jsonify({'error': 'No progress data'}), 400

    with open(progress_path) as f:
        progress = json.load(f)

    report = {'top_k_results': {}, 'details': {}}

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

    report_path = vdir / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    return jsonify({'ok': True, 'report': report})


# ─────────────────────────────────────────────────────────────────────
#  Serve STL files for 3D viewing
# ─────────────────────────────────────────────────────────────────────

@validation_bp.route('/stl/<filename>')
def serve_stl(filename):
    """Serve an STL file from the data directory."""
    stl_dir = _stl_dir()
    fpath = stl_dir / filename
    if not fpath.exists():
        abort(404)
    return send_from_directory(str(stl_dir), filename,
                               mimetype='application/octet-stream')
