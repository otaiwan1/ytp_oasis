#!/usr/bin/env python3
"""
embed/batch_embed.py — CLI script to batch-embed all STL files.

Usage:
    # Single GPU, auto-detect
    python -m embed.batch_embed --model dinov3

    # Specify GPU + parallel workers (recommended)
    python -m embed.batch_embed --model dinov3 --gpu 1 --workers 4

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

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from embed.config import STL_DIR, FIRST_SCANS_JSON


# ─── Worker for parallel rendering + inference ──────────────────────

def _worker(args):
    """Process a single STL file. Runs in a spawned subprocess."""
    stl_path, model_name, gpu_id = args
    import torch
    from embed import embed_stl

    device = torch.device(f"cuda:{gpu_id}")
    try:
        result = embed_stl(stl_path, model=model_name, device=device)
        return (result["filename"], result["embedding"], None)
    except Exception as e:
        return (Path(stl_path).name, None, str(e))


def _worker_init(gpu_id):
    """
    Set EGL_DEVICE_ID BEFORE importing open3d so that
    Filament/EGL renders on the specified GPU.
    """
    os.environ["EGL_DEVICE_ID"] = str(gpu_id)
    os.environ["OPEN3D_VERBOSITY_LEVEL"] = "Warning"
    try:
        import open3d as o3d
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Batch embed STL files.")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["simclr", "mae", "dinov2", "dinov3"],
        help="Embedding model to use.",
    )
    parser.add_argument(
        "--stl-dir", type=str, default=str(STL_DIR),
        help=f"Directory containing STL files (default: {STL_DIR}).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: <project>/train/<model>/).",
    )
    parser.add_argument(
        "--whiten", action="store_true",
        help="Apply whitening post-processing (useful for MAE).",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU ID for both CUDA inference and EGL rendering "
             "(default: auto-detect).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1). "
             "Each worker renders + infers independently.",
    )
    # Keep --gpus for backward compat but use first one only
    parser.add_argument("--gpus", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Handle --gpus backward compat
    gpu_id = args.gpu
    if gpu_id is None and args.gpus:
        gpu_id = int(args.gpus.split(",")[0].strip())

    stl_dir = Path(args.stl_dir)
    model_name = args.model

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _PROJECT_ROOT / "train" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Collect file list ───────────────────────────────────────────
    if FIRST_SCANS_JSON.exists():
        with open(FIRST_SCANS_JSON, "r") as f:
            file_list = json.load(f)
        file_list = [fn for fn in file_list if (stl_dir / fn).exists()]
        print(f"Using {len(file_list)} first-scan files from first_scans.json")
    else:
        print("Warning: first_scans.json not found. Embedding all STL files.")
        file_list = sorted(f.name for f in stl_dir.glob("*.stl"))

    if not file_list:
        print("No STL files found. Exiting.")
        return

    stl_paths = [str(stl_dir / fn) for fn in file_list]
    effective_gpu = gpu_id if gpu_id is not None else 0

    # ─── Single worker (sequential) ──────────────────────────────────
    if args.workers <= 1:
        # Set EGL GPU in main process
        os.environ["EGL_DEVICE_ID"] = str(effective_gpu)
        os.environ["OPEN3D_VERBOSITY_LEVEL"] = "Warning"
        try:
            import open3d as o3d
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
        except Exception:
            pass

        from tqdm import tqdm
        from embed import embed_stl
        import torch

        device = torch.device(f"cuda:{effective_gpu}")
        print(f"GPU {effective_gpu} (CUDA + EGL rendering)")

        embeddings = []
        filenames = []
        failed = 0

        for stl_path in tqdm(stl_paths, desc=f"Embedding ({model_name})"):
            try:
                result = embed_stl(stl_path, model=model_name, device=device)
                embeddings.append(result["embedding"])
                filenames.append(result["filename"])
            except Exception as e:
                tqdm.write(f"  Failed: {Path(stl_path).name} — {e}")
                failed += 1

    # ─── Multi-worker (parallel rendering + inference) ───────────────
    else:
        import torch.multiprocessing as mp
        from tqdm import tqdm

        num_workers = args.workers
        print(f"GPU {effective_gpu} (CUDA + EGL) × {num_workers} workers")

        mp.set_start_method("spawn", force=True)

        task_args = [(p, model_name, effective_gpu) for p in stl_paths]

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(effective_gpu,),
        ) as pool:
            embeddings = []
            filenames = []
            failed = 0

            results_iter = pool.imap_unordered(_worker, task_args)
            for fname, emb, err in tqdm(results_iter, total=len(task_args),
                                        desc=f"Embedding ({model_name})"):
                if err is not None:
                    tqdm.write(f"  Failed: {fname} — {err}")
                    failed += 1
                else:
                    embeddings.append(emb)
                    filenames.append(fname)

    if not embeddings:
        print("No successful embeddings. Exiting.")
        return

    emb_array = np.stack(embeddings).astype(np.float32)

    # ─── Sort by filename for consistent ordering ────────────────────
    sorted_pairs = sorted(zip(filenames, range(len(filenames))))
    sorted_indices = [p[1] for p in sorted_pairs]
    filenames = [filenames[i] for i in sorted_indices]
    emb_array = emb_array[sorted_indices]

    # ─── Optional whitening ──────────────────────────────────────────
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
