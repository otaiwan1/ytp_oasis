#!/usr/bin/env python3
"""
extract_simclr.py — Fast SimCLR embedding extraction using multiprocessing + batched GPU.

Pipeline:
  1. CPU pool (N workers): trimesh load → center → PCA-align → sample 100k points
  2. GPU: FPS 100k → 4096 per file
  3. GPU: Batched SimCLR backbone inference → L2-normalize

Outputs:
    train/simclr_embeddings.npy   (N, 512) float32
    train/simclr_filenames.json   list of bare filenames

Usage:
    python train/extract_simclr.py [--workers 12] [--batch-size 32]
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ─── Environment ─────────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

current_folder = Path(__file__).parent.resolve()
project_root = current_folder.parent.parent
train_dir = project_root / "train"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(train_dir) not in sys.path:
    sys.path.insert(0, str(train_dir))

from embed.config import STL_DIR, FIRST_SCANS_JSON, SIMCLR_INITIAL_SAMPLE, NUM_POINTS

output_emb_path = current_folder / "simclr_embeddings.npy"
output_ids_path = current_folder / "simclr_filenames.json"


# ─── CPU worker (runs in subprocess) ─────────────────────────────────

def _preprocess_cpu(file_path_str):
    """
    Pure-CPU: load STL → center → PCA-align → uniform sample 100k → (100000, 3) float32.
    Returns None on failure.
    """
    import trimesh

    try:
        mesh = trimesh.load(file_path_str, force="mesh")
        mesh.vertices -= mesh.center_mass

        # PCA alignment
        inertia = mesh.moment_inertia
        _eigenvalues, eigenvectors = np.linalg.eigh(inertia)
        T = np.eye(4)
        T[:3, :3] = eigenvectors
        mesh.apply_transform(np.linalg.inv(T))

        points = mesh.sample(SIMCLR_INITIAL_SAMPLE).astype(np.float32)
        return points
    except Exception:
        return None


# ─── GPU FPS ─────────────────────────────────────────────────────────

def _fps_gpu(xyz, npoint):
    """Farthest-point sampling on GPU. xyz: (N, 3) tensor → (npoint, 3) tensor."""
    N = xyz.shape[0]
    device = xyz.device
    centroids = torch.zeros(npoint, dtype=torch.long, device=device)
    distance = torch.ones(N, device=device) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device).item()
    for i in range(npoint):
        centroids[i] = farthest
        c = xyz[farthest, :].unsqueeze(0)
        dist = torch.sum((xyz - c) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance).item()
    return xyz[centroids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12, help="CPU workers for preprocessing")
    parser.add_argument("--batch-size", type=int, default=32, help="GPU inference batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}  |  CPU workers: {args.workers}")

    # ─── 1. File list ────────────────────────────────────────────────
    if not FIRST_SCANS_JSON.exists():
        print(f"❌ first_scans.json not found: {FIRST_SCANS_JSON}")
        return
    with open(FIRST_SCANS_JSON) as f:
        raw_filenames = json.load(f)
    file_list = [fn for fn in raw_filenames if (STL_DIR / fn).exists()]
    print(f"📂 {len(file_list)} first-scan STL files.")

    # ─── 2. Parallel CPU preprocessing ───────────────────────────────
    print(f"⚙️  CPU preprocessing ({args.workers} workers)...")
    cpu_results = {}  # index → (filename, points_100k)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_to_idx = {
            pool.submit(_preprocess_cpu, str(STL_DIR / fn)): (i, fn)
            for i, fn in enumerate(file_list)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(file_list), desc="CPU preprocess"):
            idx, fname = future_to_idx[future]
            pts = future.result()
            if pts is not None:
                cpu_results[idx] = (fname, pts)

    # Sort to maintain deterministic order
    sorted_items = sorted(cpu_results.items())
    print(f"   ✅ {len(sorted_items)}/{len(file_list)} preprocessed successfully.")

    # ─── 3. GPU FPS (100k → 4096) ───────────────────────────────────
    print("⚙️  GPU FPS downsampling...")
    fps_results = []  # list of (filename, (4096, 3) ndarray)
    for _idx, (fname, pts_100k) in tqdm(sorted_items, desc="GPU FPS"):
        pts_tensor = torch.from_numpy(pts_100k).float().to(device)
        pts_4096 = _fps_gpu(pts_tensor, NUM_POINTS)
        fps_results.append((fname, pts_4096.cpu().numpy()))

    # ─── 4. Load model ───────────────────────────────────────────────
    print("🧠 Loading SimCLR model...")
    from train_oasis import SimCLREncoder
    from embed.config import SIMCLR_CHECKPOINT

    model = SimCLREncoder().to(device)
    sd = torch.load(str(SIMCLR_CHECKPOINT), map_location=device)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    # ─── 5. Batched GPU inference ────────────────────────────────────
    print("⚙️  Batched SimCLR inference...")
    all_embeddings = []
    all_filenames = []
    batch_size = args.batch_size
    n = len(fps_results)

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Inference"):
            batch_items = fps_results[start : start + batch_size]
            fnames = [item[0] for item in batch_items]
            pts_batch = np.stack([item[1] for item in batch_items])  # (B, 4096, 3)
            pts_tensor = torch.from_numpy(pts_batch).float().to(device)

            emb = model.backbone(pts_tensor)       # (B, 512)
            emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu().numpy())
            all_filenames.extend(fnames)

    final_embs = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    # ─── 6. Save ─────────────────────────────────────────────────────
    np.save(str(output_emb_path), final_embs)
    with open(output_ids_path, "w") as f:
        json.dump(all_filenames, f)

    print("-" * 40)
    print(f"✅ Done!  Shape: {final_embs.shape}")
    print(f"   Embeddings: {output_emb_path}")
    print(f"   Filenames:  {output_ids_path}")


if __name__ == "__main__":
    main()
