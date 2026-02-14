import os
from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from pathlib import Path
import shutil

upload_bp = Blueprint('upload', __name__, url_prefix='/upload')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@upload_bp.route('/', methods=['GET', 'POST'])
@login_required
def upload_page():
    if request.method == 'POST':
        patient_uid = request.form.get('patient_uid', '').strip()

        if not patient_uid:
            flash('Please provide a Patient UID.', 'danger')
            return redirect(url_for('upload.upload_page'))

        files = request.files.getlist('stl_files')
        if not files or all(f.filename == '' for f in files):
            flash('No files selected.', 'danger')
            return redirect(url_for('upload.upload_page'))

        stl_dir = Path(current_app.config['STL_DATA_DIR'])
        stl_dir.mkdir(parents=True, exist_ok=True)

        uploaded_count = 0
        for idx, file in enumerate(files):
            if file and file.filename and allowed_file(file.filename):
                # Name format: patientUID_serialNumber.stl
                serial = idx + 1
                # Check existing serial numbers for this patient
                existing = list(stl_dir.glob(f'{patient_uid}_*.stl'))
                if existing:
                    existing_serials = []
                    for ef in existing:
                        parts = ef.stem.split('_', 1)
                        if len(parts) > 1:
                            try:
                                existing_serials.append(int(parts[1]))
                            except ValueError:
                                existing_serials.append(0)
                    serial = max(existing_serials, default=0) + idx + 1

                output_name = f'{patient_uid}_{serial}.stl'
                output_path = stl_dir / output_name
                file.save(str(output_path))
                uploaded_count += 1

        if uploaded_count > 0:
            # Invalidate embeddings cache since new data was added
            cache_path = current_app.config['EMBEDDINGS_CACHE']
            filenames_path = current_app.config['FILENAMES_CACHE']
            for p in [cache_path, filenames_path]:
                if os.path.exists(p):
                    os.remove(p)

            flash(f'Successfully uploaded {uploaded_count} scan(s) for patient {patient_uid}.', 'success')
        else:
            flash('No valid .stl files were uploaded.', 'warning')

        return redirect(url_for('upload.upload_page'))

    return render_template('upload/upload.html')
