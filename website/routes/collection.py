import os
from flask import Blueprint, render_template, send_from_directory, current_app, abort
from flask_login import login_required
from pathlib import Path

collection_bp = Blueprint('collection', __name__, url_prefix='/collection')


@collection_bp.route('/')
@login_required
def browse():
    """Browse all patients and their scans."""
    stl_dir = Path(current_app.config['STL_DATA_DIR'])

    patients = {}
    if stl_dir.exists():
        for f in sorted(stl_dir.glob('*.stl')):
            fname = f.name
            parts = fname.replace('.stl', '').split('_', 1)
            patient_uid = parts[0]
            serial = parts[1] if len(parts) > 1 else ''

            if patient_uid not in patients:
                patients[patient_uid] = []
            patients[patient_uid].append({
                'filename': fname,
                'serial_number': serial
            })

    # Sort patients by UID
    sorted_patients = dict(sorted(patients.items()))

    return render_template('collection/browse.html', patients=sorted_patients)


@collection_bp.route('/patient/<patient_uid>')
@login_required
def patient_detail(patient_uid):
    """Show all scans for a specific patient."""
    stl_dir = Path(current_app.config['STL_DATA_DIR'])
    rendered_dir = Path(current_app.config['RENDERED_IMAGES_DIR'])

    scans = []
    if stl_dir.exists():
        for f in sorted(stl_dir.glob(f'{patient_uid}_*.stl')):
            fname = f.name
            parts = fname.replace('.stl', '').split('_', 1)
            serial = parts[1] if len(parts) > 1 else ''

            # Check for rendered views
            views = []
            patient_render_dir = rendered_dir / patient_uid
            if patient_render_dir.exists():
                for view in ['front', 'back', 'top', 'bottom', 'left', 'right']:
                    img_path = patient_render_dir / f'{view}.png'
                    if img_path.exists():
                        views.append(view)

            scans.append({
                'filename': fname,
                'serial_number': serial,
                'views': views
            })

    if not scans:
        abort(404)

    return render_template('collection/patient.html',
                           patient_uid=patient_uid, scans=scans)


@collection_bp.route('/render/<patient_uid>/<view>')
@login_required
def serve_render(patient_uid, view):
    """Serve rendered images."""
    rendered_dir = Path(current_app.config['RENDERED_IMAGES_DIR'])
    patient_dir = rendered_dir / patient_uid

    if not patient_dir.exists():
        abort(404)

    filename = f'{view}.png'
    filepath = patient_dir / filename
    if not filepath.exists():
        abort(404)

    return send_from_directory(str(patient_dir), filename)
