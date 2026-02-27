#!/usr/bin/env python3
"""
Batch-convert STL files to compressed GLB for fast web viewing.

Usage:
    python tools/convert_stl_to_glb.py [--workers N] [--keep-frac 0.15]

Output:  collecting-data/glbFiles/<original_name>.glb   (gzip-compressed)

Compression pipeline:
  1. Load STL
  2. Simplify to `keep_frac` of original face count (quadric decimation)
  3. Export as GLB (glTF Binary)
  4. gzip-compress and write to disk

Typical results (keep_frac=0.15):
  34 MB STL → ~1 MB .glb.gz   (97% reduction)
"""

import argparse
import gzip
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import trimesh

# ── Defaults ─────────────────────────────────────────────────────────
DEFAULT_KEEP_FRAC = 0.15      # keep 15% of faces
DEFAULT_WORKERS   = 4
STL_DIR           = Path(__file__).resolve().parent.parent / "collecting-data" / "stlFiles"
GLB_DIR           = Path(__file__).resolve().parent.parent / "collecting-data" / "glbFiles"


def convert_one(stl_path: Path, glb_dir: Path, keep_frac: float) -> dict:
    """Convert a single STL → compressed GLB.  Returns stats dict."""
    out_path = glb_dir / (stl_path.stem + ".glb")
    if out_path.exists():
        return {"file": stl_path.name, "skipped": True}

    t0 = time.time()
    mesh = trimesh.load(stl_path)
    orig_faces = len(mesh.faces)

    # Simplify
    target = max(int(orig_faces * keep_frac), 1000)
    if orig_faces > target:
        mesh = mesh.simplify_quadric_decimation(face_count=target)

    # Export GLB then gzip
    glb_bytes = mesh.export(file_type="glb")
    gz_bytes  = gzip.compress(glb_bytes, compresslevel=6)

    out_path.write_bytes(gz_bytes)
    elapsed = time.time() - t0

    return {
        "file": stl_path.name,
        "skipped": False,
        "orig_mb": stl_path.stat().st_size / 1024 / 1024,
        "glb_mb": len(gz_bytes) / 1024 / 1024,
        "orig_faces": orig_faces,
        "new_faces": len(mesh.faces),
        "time": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch STL → GLB converter")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--keep-frac", type=float, default=DEFAULT_KEEP_FRAC,
                        help="Fraction of faces to keep (0.0-1.0)")
    args = parser.parse_args()

    stl_files = sorted(STL_DIR.glob("*.stl"))
    if not stl_files:
        print(f"No STL files found in {STL_DIR}")
        sys.exit(1)

    GLB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Converting {len(stl_files)} STL files → GLB  (keep {args.keep_frac*100:.0f}% faces, {args.workers} workers)")
    print(f"  Source: {STL_DIR}")
    print(f"  Output: {GLB_DIR}")
    print()

    done = 0
    skipped = 0
    total_orig = 0
    total_glb  = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(convert_one, f, GLB_DIR, args.keep_frac): f
            for f in stl_files
        }
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if result["skipped"]:
                skipped += 1
                continue
            total_orig += result["orig_mb"]
            total_glb  += result["glb_mb"]
            if done % 20 == 0 or done == len(stl_files):
                elapsed = time.time() - t_start
                pct = done / len(stl_files) * 100
                print(f"  [{done:4d}/{len(stl_files)}] {pct:5.1f}%  "
                      f"last: {result['orig_mb']:.1f}→{result['glb_mb']:.1f}MB  "
                      f"elapsed: {elapsed:.0f}s")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Converted: {done - skipped}, Skipped (existing): {skipped}")
    if total_orig > 0:
        print(f"  Total: {total_orig:.0f} MB → {total_glb:.0f} MB "
              f"({total_glb/total_orig*100:.1f}% of original)")


if __name__ == "__main__":
    main()
