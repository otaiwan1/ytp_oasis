#!/usr/bin/env python3
"""
build_database.py — Full database build tool.

Scans the scanfiles/ directory for all iTero zip exports and ensures:
  1. STL files are extracted to stlFiles/
  2. Gzip-compressed GLB files are generated in glbFiles/
  3. First-scan embeddings are computed and saved to the embeddings database

Usage:
    python build_database.py                   # Dry-run: show what would be done
    python build_database.py --run             # Execute all pending tasks
    python build_database.py --run --workers 8 # Use 8 CPU workers (default: 12)
    python build_database.py --model dinov3_gallery  # Specify embedding model
"""

import argparse
import gzip
import json
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ── Directories ──────────────────────────────────────────────────────
SCANFILES_DIR = PROJECT_ROOT / 'collecting-data' / 'scanfiles'
STL_DIR       = PROJECT_ROOT / 'collecting-data' / 'stlFiles'
GLB_DIR       = PROJECT_ROOT / 'collecting-data' / 'glbFiles'


def get_embedding_paths(model_name):
    """Return (embeddings.npy path, filenames.json path) for a model."""
    base = PROJECT_ROOT / 'train' / model_name
    return (
        base / f'{model_name}_embeddings.npy',
        base / f'{model_name}_filenames.json',
    )


# ── Parsing ──────────────────────────────────────────────────────────

def parse_zip_info(zip_path):
    """Extract patient_uuid and scan_id from an iTero zip.

    Returns (patient_uuid, scan_id) or raises ValueError.
    """
    zip_path = Path(zip_path)

    m = re.search(r'OrthoCAD_Export_(\d+)\.zip', zip_path.name)
    if not m:
        raise ValueError(f"Bad filename: {zip_path.name}")
    scan_id = m.group(1)

    patient_uuid = None
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        for name in zf.namelist():
            if not name.lower().endswith('.xml'):
                continue
            try:
                with zf.open(name) as xf:
                    tree = ET.parse(xf)
                    for elem in tree.getroot().iter('UniquePatientIdentifier'):
                        if elem.text and elem.text.strip():
                            patient_uuid = elem.text.strip()
                            break
                if patient_uuid:
                    break
            except ET.ParseError:
                continue

    if not patient_uuid:
        # Fallback: use parent directory name as patient UUID
        patient_uuid = zip_path.parent.name

    return patient_uuid, scan_id


# ── STL extraction ───────────────────────────────────────────────────

def extract_stl(zip_path, patient_uuid, scan_id):
    """Extract STL from zip. Returns output path."""
    stl_filename = f"{patient_uuid}_{scan_id}.stl"
    stl_output = STL_DIR / stl_filename

    if stl_output.exists():
        return stl_output

    STL_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        stl_members = [n for n in zf.namelist() if n.lower().endswith('.stl')]
        if not stl_members:
            raise FileNotFoundError(f"No .stl in {zip_path.name}")
        with zf.open(stl_members[0]) as src, open(str(stl_output), 'wb') as dst:
            dst.write(src.read())

    return stl_output


# ── GLB conversion ───────────────────────────────────────────────────

def convert_to_glb(stl_path, patient_uuid, scan_id):
    """Convert STL to gzip-compressed GLB. Returns output path."""
    glb_filename = f"{patient_uuid}_{scan_id}.glb"
    glb_output = GLB_DIR / glb_filename

    if glb_output.exists():
        return glb_output

    GLB_DIR.mkdir(parents=True, exist_ok=True)
    import trimesh
    mesh = trimesh.load(str(stl_path), force='mesh')
    glb_data = mesh.export(file_type='glb')
    with open(str(glb_output), 'wb') as f:
        f.write(gzip.compress(glb_data))

    return glb_output


# ── CPU worker for STL + GLB ─────────────────────────────────────────

def process_zip_cpu(zip_path_str):
    """Worker: parse, extract STL, convert GLB. Returns dict with status."""
    zip_path = Path(zip_path_str)
    try:
        patient_uuid, scan_id = parse_zip_info(zip_path)
        stl_filename = f"{patient_uuid}_{scan_id}.stl"
        stl_existed = (STL_DIR / stl_filename).exists()
        glb_existed = (GLB_DIR / f"{patient_uuid}_{scan_id}.glb").exists()

        stl_path = extract_stl(zip_path, patient_uuid, scan_id)
        glb_path = convert_to_glb(stl_path, patient_uuid, scan_id)

        return {
            'ok': True,
            'zip': zip_path.name,
            'patient_uuid': patient_uuid,
            'scan_id': scan_id,
            'stl_filename': stl_filename,
            'stl_new': not stl_existed,
            'glb_new': not glb_existed,
        }
    except Exception as e:
        return {
            'ok': False,
            'zip': zip_path.name,
            'error': str(e),
        }


# ── Embedding computation (GPU) ─────────────────────────────────────

def compute_embeddings_batch(stl_filenames, model_name):
    """Compute embeddings for a list of STL files using GPU.

    This loads the model once, processes all files, then releases GPU.
    """
    if not stl_filenames:
        return {}

    from embed import embed_stl

    results = {}
    for i, stl_fn in enumerate(stl_filenames):
        stl_path = STL_DIR / stl_fn
        if not stl_path.exists():
            print(f"  [WARN] STL not found: {stl_fn}")
            continue
        try:
            result = embed_stl(str(stl_path), model=model_name)
            results[stl_fn] = result['embedding']
            if (i + 1) % 10 == 0 or i == len(stl_filenames) - 1:
                print(f"  Embedded {i+1}/{len(stl_filenames)}: {stl_fn}")
        except Exception as e:
            print(f"  [ERROR] Failed to embed {stl_fn}: {e}")

    # Release GPU
    try:
        from embed import release_models
        release_models()
    except ImportError:
        pass

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    return results


# ── Identify first scans ────────────────────────────────────────────

def find_first_scans(all_info):
    """Given list of {patient_uuid, scan_id, stl_filename, ...},
    return set of stl_filenames that are the first scan per patient.

    First scan = the one with the lowest scan_id for each patient.
    """
    patients = {}
    for info in all_info:
        uid = info['patient_uuid']
        sid = int(info['scan_id'])
        if uid not in patients or sid < patients[uid][0]:
            patients[uid] = (sid, info['stl_filename'])
    return set(v[1] for v in patients.values())


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Build database from scanfiles')
    parser.add_argument('--run', action='store_true', help='Execute (default is dry-run)')
    parser.add_argument('--workers', type=int, default=12, help='CPU workers for STL/GLB (default: 12)')
    parser.add_argument('--model', default='dinov3_gallery', help='Embedding model (default: dinov3_gallery)')
    args = parser.parse_args()

    emb_path, fn_path = get_embedding_paths(args.model)

    # 1. Discover all zip files
    print("=" * 60)
    print("SCAN DATABASE BUILD TOOL")
    print("=" * 60)

    all_zips = sorted(SCANFILES_DIR.rglob('OrthoCAD_Export_*.zip'))
    print(f"\nFound {len(all_zips)} zip files in scanfiles/")

    # 2. Parse all zips to get info
    print("\nParsing zip files...")
    all_info = []
    errors = []
    for zp in all_zips:
        try:
            patient_uuid, scan_id = parse_zip_info(zp)
            stl_fn = f"{patient_uuid}_{scan_id}.stl"
            all_info.append({
                'zip_path': str(zp),
                'patient_uuid': patient_uuid,
                'scan_id': scan_id,
                'stl_filename': stl_fn,
            })
        except Exception as e:
            errors.append(f"{zp.name}: {e}")

    if errors:
        print(f"\n  {len(errors)} zip files had parse errors:")
        for err in errors[:5]:
            print(f"    {err}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")

    # 3. Check what needs to be done
    stl_needed = []
    glb_needed = []
    for info in all_info:
        stl_fn = info['stl_filename']
        glb_fn = stl_fn.replace('.stl', '.glb')
        if not (STL_DIR / stl_fn).exists():
            stl_needed.append(info)
        if not (GLB_DIR / glb_fn).exists():
            glb_needed.append(info)

    # Identify first scans
    first_scan_fns = find_first_scans(all_info)

    # Check which first scans need embeddings
    existing_fns = set()
    if fn_path.exists():
        with open(fn_path) as f:
            existing_fns = set(json.load(f))

    emb_needed = [fn for fn in first_scan_fns if fn not in existing_fns]

    # Union: zips that need either STL or GLB work
    zips_needing_cpu = set()
    for info in stl_needed + glb_needed:
        zips_needing_cpu.add(info['zip_path'])

    print(f"\n── Summary ────────────────────────────────────────")
    print(f"  Total zips:           {len(all_zips)}")
    print(f"  Total patients:       {len(set(i['patient_uuid'] for i in all_info))}")
    print(f"  First scans:          {len(first_scan_fns)}")
    print(f"  STL to extract:       {len(stl_needed)}")
    print(f"  GLB to convert:       {len(glb_needed)}")
    print(f"  Embeddings to add:    {len(emb_needed)}")
    print(f"  Existing embeddings:  {len(existing_fns)}")
    print(f"  Zips needing work:    {len(zips_needing_cpu)}")
    print(f"──────────────────────────────────────────────────")

    if not args.run:
        print("\n  [DRY RUN] Use --run to execute.")
        return

    # 4. Process STL + GLB (multi-CPU)
    if zips_needing_cpu:
        print(f"\n[STEP 1] Extracting STL & converting GLB ({args.workers} workers)...")
        zip_paths = list(zips_needing_cpu)
        done = 0
        new_stl = 0
        new_glb = 0
        err_count = 0

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_zip_cpu, zp): zp for zp in zip_paths}
            for future in as_completed(futures):
                result = future.result()
                done += 1
                if result['ok']:
                    if result['stl_new']:
                        new_stl += 1
                    if result['glb_new']:
                        new_glb += 1
                else:
                    err_count += 1
                    print(f"  [ERROR] {result['zip']}: {result['error']}")

                if done % 50 == 0 or done == len(zip_paths):
                    print(f"  Progress: {done}/{len(zip_paths)} "
                          f"(+{new_stl} STL, +{new_glb} GLB, {err_count} errors)")

        print(f"  Done: +{new_stl} new STL, +{new_glb} new GLB, {err_count} errors")
    else:
        print("\n[STEP 1] All STL and GLB files up to date. ✓")

    # 5. Compute embeddings (GPU)
    if emb_needed:
        print(f"\n[STEP 2] Computing {len(emb_needed)} embeddings ({args.model})...")
        new_embeddings = compute_embeddings_batch(sorted(emb_needed), args.model)

        # Load existing and merge
        if emb_path.exists() and fn_path.exists():
            all_embs = np.load(str(emb_path))
            with open(str(fn_path)) as f:
                all_fnames = json.load(f)
        else:
            all_embs = None
            all_fnames = []

        added = 0
        for fn, emb in sorted(new_embeddings.items()):
            if fn not in set(all_fnames):
                if all_embs is not None:
                    all_embs = np.vstack([all_embs, emb.reshape(1, -1)])
                else:
                    all_embs = emb.reshape(1, -1)
                all_fnames.append(fn)
                added += 1

        # Save
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(emb_path), all_embs)
        with open(str(fn_path), 'w') as f:
            json.dump(all_fnames, f)

        print(f"  Added {added} embeddings. Total: {len(all_fnames)}")
    else:
        print("\n[STEP 2] All first-scan embeddings up to date. ✓")

    print(f"\n{'=' * 60}")
    print("BUILD COMPLETE")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
