#!/usr/bin/env python3
"""
extract_mae.py — Fast Point-MAE embedding extraction using multiprocessing + batched GPU.

Pipeline:
  1. CPU pool (N workers): trimesh load → center → PCA-align → sample 50k →
     unit-sphere normalize → Open3D normal estimation → concat XYZ+Normal (N,6)
  2. GPU: FPS (N,6) → (4096, 6) per file  (distance on XYZ only)
  3. GPU: Batched PointMAE get_embedding → (384,)

Outputs:
    train/mae_embeddings.npy   (N, 384) float32
    train/mae_filenames.json   list of bare filenames

Usage:
    python train/extract_mae.py [--workers 2] [--batch-size 32]
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
project_root = current_folder.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(current_folder) not in sys.path:
    sys.path.insert(0, str(current_folder))

from embed.config import STL_DIR, FIRST_SCANS_JSON, MAE_INITIAL_SAMPLE, NUM_POINTS

output_emb_path = current_folder / "mae_embeddings.npy"
output_ids_path = current_folder / "mae_filenames.json"


# ─── CPU worker (runs in subprocess) ─────────────────────────────────

def _preprocess_cpu(file_path_str):
    """
    Pure-CPU: load STL → center → PCA-align → sample 50k →
    unit-sphere normalize → estimate normals → concat (50000, 6) float32.
    Returns None on failure.
    """
    import trimesh
    import open3d as o3d

    try:
        mesh = trimesh.load(file_path_str, force="mesh")
        if mesh.is_empty:
            return None

        mesh.vertices -= mesh.center_mass

        # PCA alignment (with safe fallback)
        try:
            inertia = mesh.moment_inertia
            _eigenvalues, eigenvectors = np.linalg.eigh(inertia)
            T = np.eye(4)
            T[:3, :3] = eigenvectors
            if np.all(np.isfinite(T)):
                mesh.apply_transform(np.linalg.inv(T))
        except Exception:
            pass

        points = mesh.sample(MAE_INITIAL_SAMPLE).astype(np.float32)
        if np.isnan(points).any():
            return None

        # Unit-sphere normalization
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points /= max_dist

        # Normal estimation via Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
        normals = np.asarray(pcd.normals, dtype=np.float32)
        normals = np.nan_to_num(normals)

        features = np.hstack([points, normals]).astype(np.float32)  # (50000, 6)
        return features
    except Exception:
        return None


# ─── GPU FPS (6-channel, distance on XYZ only) ──────────────────────

def _fps_gpu(xyz_feat, npoint):
    """
    Farthest-point sampling on GPU.
    xyz_feat: (N, 6) tensor → (npoint, 6) tensor.
    Distance computed on first 3 columns (XYZ) only.
    """
    N, C = xyz_feat.shape
    device = xyz_feat.device
    xyz = xyz_feat[:, :3]

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

    return xyz_feat[centroids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2, help="CPU workers for preprocessing")
    parser.add_argument("--batch-size", type=int, default=32, help="GPU inference batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  CPU workers: {args.workers}")

    # ─── 1. File list ────────────────────────────────────────────────
    if not FIRST_SCANS_JSON.exists():
        print(f"first_scans.json not found: {FIRST_SCANS_JSON}")
        return
    with open(FIRST_SCANS_JSON) as f:
        raw_filenames = json.load(f)
    file_list = [fn for fn in raw_filenames if (STL_DIR / fn).exists()]
    print(f"{len(file_list)} first-scan STL files.")

    # ─── 2. Parallel CPU preprocessing ───────────────────────────────
    print(f"CPU preprocessing ({args.workers} workers)...")
    cpu_results = {}  # index → (filename, features_50k)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_to_idx = {
            pool.submit(_preprocess_cpu, str(STL_DIR / fn)): (i, fn)
            for i, fn in enumerate(file_list)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(file_list), desc="CPU preprocess"):
            idx, fname = future_to_idx[future]
            feat = future.result()
            if feat is not None:
                cpu_results[idx] = (fname, feat)

    # Sort to maintain deterministic order
    sorted_items = sorted(cpu_results.items())
    print(f"  {len(sorted_items)}/{len(file_list)} preprocessed successfully.")

    # ─── 3. GPU FPS (50k → 4096, 6-channel) ─────────────────────────
    print("GPU FPS downsampling...")
    fps_results = []  # list of (filename, (4096, 6) ndarray)
    for _idx, (fname, feat_50k) in tqdm(sorted_items, desc="GPU FPS"):
        feat_tensor = torch.from_numpy(feat_50k).float().to(device)
        feat_4096 = _fps_gpu(feat_tensor, NUM_POINTS)
        fps_results.append((fname, feat_4096.cpu().numpy()))

    # ─── 4. Load model ───────────────────────────────────────────────
    print("Loading Point-MAE model...")
    from train_mae_ddp import PointMAE, config as mae_config
    from embed.config import MAE_CHECKPOINT

    # Detect in_channels from checkpoint
    state_dict = torch.load(str(MAE_CHECKPOINT), map_location=device)
    new_state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    first_conv_key = "patch_embed.conv.0.weight"
    in_channels = new_state_dict[first_conv_key].shape[1] if first_conv_key in new_state_dict else 6

    model = PointMAE(mae_config, in_channels=in_channels).to(device)
    model.load_state_dict(new_state_dict)
    model.eval()

    # ─── 5. Batched GPU inference ────────────────────────────────────
    print("Batched Point-MAE inference...")
    all_embeddings = []
    all_filenames = []
    batch_size = args.batch_size
    n = len(fps_results)

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Inference"):
            batch_items = fps_results[start : start + batch_size]
            fnames = [item[0] for item in batch_items]
            pts_batch = np.stack([item[1] for item in batch_items])  # (B, 4096, 6)
            pts_tensor = torch.from_numpy(pts_batch).float().to(device)

            emb = model.get_embedding(pts_tensor)  # (B, 384)
            all_embeddings.append(emb.cpu().numpy())
            all_filenames.extend(fnames)

    final_embs = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    # ─── 6. Save ─────────────────────────────────────────────────────
    np.save(str(output_emb_path), final_embs)
    with open(output_ids_path, "w") as f:
        json.dump(all_filenames, f)

    print("-" * 40)
    print(f"Done!  Shape: {final_embs.shape}")
    print(f"   Embeddings: {output_emb_path}")
    print(f"   Filenames:  {output_ids_path}")


if __name__ == "__main__":
    main()
