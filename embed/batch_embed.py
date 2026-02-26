#!/usr/bin/env python3
"""
embed/batch_embed.py — CLI script to batch-embed all STL files.

Usage:
    python -m embed.batch_embed --model dinov2 [--stl-dir PATH] [--output-dir PATH]
    python -m embed.batch_embed --model simclr
    python -m embed.batch_embed --model mae

Outputs:
    {output_dir}/{model}_embeddings.npy   — (N, D) float32 array
    {output_dir}/{model}_filenames.json   — list of bare filenames
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from embed import embed_stl
from embed.config import STL_DIR, FIRST_SCANS_JSON


def main():
    parser = argparse.ArgumentParser(description="Batch embed STL files.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["simclr", "mae", "dinov2"],
        help="Embedding model to use.",
    )
    parser.add_argument(
        "--stl-dir",
        type=str,
        default=str(STL_DIR),
        help=f"Directory containing STL files (default: {STL_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_PROJECT_ROOT / "train"),
        help="Output directory for embeddings and filenames.",
    )
    parser.add_argument(
        "--whiten",
        action="store_true",
        help="Apply whitening post-processing (useful for MAE).",
    )
    args = parser.parse_args()

    stl_dir = Path(args.stl_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model

    # ─── Collect file list ───────────────────────────────────────────
    # All models use first_scans.json (one scan per patient)
    if FIRST_SCANS_JSON.exists():
        with open(FIRST_SCANS_JSON, "r") as f:
            file_list = json.load(f)
        # Filter to files that actually exist
        file_list = [fn for fn in file_list if (stl_dir / fn).exists()]
        print(f"Using {len(file_list)} first-scan files from first_scans.json")
    else:
        print("Warning: first_scans.json not found. Embedding all STL files.")
        file_list = sorted(f.name for f in stl_dir.glob("*.stl"))

    if not file_list:
        print("No STL files found. Exiting.")
        return

    # ─── Batch embed ─────────────────────────────────────────────────
    embeddings = []
    filenames = []
    failed = 0

    for fname in tqdm(file_list, desc=f"Embedding ({model_name})"):
        stl_path = stl_dir / fname
        try:
            result = embed_stl(stl_path, model=model_name)
            embeddings.append(result["embedding"])
            filenames.append(result["filename"])
        except Exception as e:
            print(f"\n  Failed: {fname} — {e}")
            failed += 1

    if not embeddings:
        print("No successful embeddings. Exiting.")
        return

    emb_array = np.stack(embeddings).astype(np.float32)

    # ─── Optional whitening (MAE) ────────────────────────────────────
    if args.whiten:
        print("Applying whitening (mean-subtract + re-normalize)...")
        mean_vec = np.mean(emb_array, axis=0)
        emb_array = emb_array - mean_vec
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        emb_array = emb_array / (norms + 1e-8)

    # ─── Save ────────────────────────────────────────────────────────
    emb_path = output_dir / f"{model_name}_embeddings.npy"
    ids_path = output_dir / f"{model_name}_filenames.json"

    np.save(str(emb_path), emb_array)
    with open(ids_path, "w") as f:
        json.dump(filenames, f)

    print(f"\nDone!")
    print(f"  Embeddings: {emb_path}  shape={emb_array.shape}")
    print(f"  Filenames:  {ids_path}  count={len(filenames)}")
    if failed:
        print(f"  Failed:     {failed}")


if __name__ == "__main__":
    main()
