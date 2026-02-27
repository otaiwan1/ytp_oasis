#!/usr/bin/env python3
"""
embed/batch_embed.py — CLI script to batch-embed all STL files.

Usage:
    # Single GPU (default: auto-detect)
    python -m embed.batch_embed --model dinov3

    # Multi-GPU parallel (e.g. GPU 1, 2, 3)
    python -m embed.batch_embed --model dinov3 --gpus 1,2,3

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

# Suppress Open3D INFO before any imports
os.environ["OPEN3D_VERBOSITY_LEVEL"] = "Warning"

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from embed.config import STL_DIR, FIRST_SCANS_JSON


# ─── Standalone worker function for multi-GPU (called via spawn) ─────

def _gpu_worker(gpu_id, file_chunk, stl_dir, model_name, result_dir):
    """
    Process a chunk of files on one GPU.
    Writes results to a temp file to avoid pickling large arrays.
    """
    os.environ["OPEN3D_VERBOSITY_LEVEL"] = "Warning"
    try:
        import open3d as o3d
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    except Exception:
        pass

    import torch
    from embed import embed_stl

    device = torch.device(f"cuda:{gpu_id}")
    embeddings = []
    filenames = []
    failed = []

    for fname in file_chunk:
        stl_path = str(Path(stl_dir) / fname)
        try:
            result = embed_stl(stl_path, model=model_name, device=device)
            embeddings.append(result["embedding"])
            filenames.append(result["filename"])
        except Exception as e:
            failed.append((fname, str(e)))

    # Save to temp files
    out_prefix = Path(result_dir) / f"gpu_{gpu_id}"
    if embeddings:
        np.save(str(out_prefix) + "_emb.npy",
                np.stack(embeddings).astype(np.float32))
        with open(str(out_prefix) + "_fnames.json", "w") as f:
            json.dump(filenames, f)
    with open(str(out_prefix) + "_failed.json", "w") as f:
        json.dump(failed, f)

    print(f"  GPU {gpu_id}: {len(embeddings)} done, {len(failed)} failed",
          flush=True)


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
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs (e.g. '1,2,3').",
    )
    args = parser.parse_args()

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

    # ─── Parse GPU IDs ───────────────────────────────────────────────
    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]

    # ─── Suppress Open3D in main process ─────────────────────────────
    try:
        import open3d as o3d
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    except Exception:
        pass

    # ─── Single GPU mode ─────────────────────────────────────────────
    if gpu_ids is None or len(gpu_ids) <= 1:
        from tqdm import tqdm
        from embed import embed_stl
        import torch

        device = None
        if gpu_ids and len(gpu_ids) == 1:
            device = torch.device(f"cuda:{gpu_ids[0]}")

        embeddings = []
        filenames = []
        failed = 0

        for fname in tqdm(file_list, desc=f"Embedding ({model_name})"):
            stl_path = stl_dir / fname
            try:
                result = embed_stl(str(stl_path), model=model_name, device=device)
                embeddings.append(result["embedding"])
                filenames.append(result["filename"])
            except Exception as e:
                tqdm.write(f"  Failed: {fname} — {e}")
                failed += 1

    # ─── Multi-GPU mode ──────────────────────────────────────────────
    else:
        import torch.multiprocessing as mp
        import tempfile

        num_gpus = len(gpu_ids)
        print(f"Multi-GPU mode: {num_gpus} GPUs {gpu_ids}")

        # Split files round-robin
        chunks = [[] for _ in range(num_gpus)]
        for i, fname in enumerate(file_list):
            chunks[i % num_gpus].append(fname)

        for i, gpu_id in enumerate(gpu_ids):
            print(f"  GPU {gpu_id}: {len(chunks[i])} files")

        # Use a temp dir for inter-process communication
        with tempfile.TemporaryDirectory() as tmpdir:
            # Launch one process per GPU using spawn
            mp.set_start_method("spawn", force=True)
            processes = []
            for i, gpu_id in enumerate(gpu_ids):
                if not chunks[i]:
                    continue
                p = mp.Process(
                    target=_gpu_worker,
                    args=(gpu_id, chunks[i], str(stl_dir), model_name, tmpdir),
                )
                p.start()
                processes.append((p, gpu_id))

            # Wait for all to finish
            for p, gpu_id in processes:
                p.join()
                if p.exitcode != 0:
                    print(f"  ⚠ GPU {gpu_id} worker exited with code {p.exitcode}")

            # Collect results from temp files
            embeddings = []
            filenames = []
            failed = 0

            for _, gpu_id in processes:
                prefix = Path(tmpdir) / f"gpu_{gpu_id}"
                emb_f = str(prefix) + "_emb.npy"
                fn_f = str(prefix) + "_fnames.json"
                fail_f = str(prefix) + "_failed.json"

                if os.path.exists(emb_f):
                    emb = np.load(emb_f)
                    with open(fn_f) as f:
                        fns = json.load(f)
                    embeddings.append(emb)
                    filenames.extend(fns)

                if os.path.exists(fail_f):
                    with open(fail_f) as f:
                        fails = json.load(f)
                    for fname, err in fails:
                        print(f"  Failed (GPU {gpu_id}): {fname} — {err}")
                    failed += len(fails)

        if embeddings:
            embeddings = [np.concatenate(embeddings, axis=0)]

    if not embeddings and not filenames:
        print("No successful embeddings. Exiting.")
        return

    if isinstance(embeddings, list) and len(embeddings) > 0:
        if isinstance(embeddings[0], np.ndarray) and embeddings[0].ndim == 2:
            emb_array = embeddings[0]
        else:
            emb_array = np.stack(embeddings).astype(np.float32)
    else:
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
