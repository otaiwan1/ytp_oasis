"""
embed/preprocessing.py — Shared preprocessing pipelines for all 3 embedding models.

Functions:
    preprocess_simclr(stl_path, num_points=4096) → np.ndarray (num_points, 3)
    preprocess_mae(stl_path, num_points=4096) → np.ndarray (num_points, 6)
    preprocess_dinov2(stl_path, views_config, views_order, render_size=512, fov_deg=60.0) → list[PIL.Image]
"""

import copy
import numpy as np
import trimesh
import torch
from PIL import Image

from .config import (
    NUM_POINTS,
    SIMCLR_INITIAL_SAMPLE,
    MAE_INITIAL_SAMPLE,
    RENDER_VIEWS_CONFIG,
    DINOV2_VIEWS,
    RENDER_IMG_SIZE,
    RENDER_FOV_DEG,
)


# ─── GPU FPS (shared) ───────────────────────────────────────────────

def _farthest_point_sample_gpu(xyz_tensor, npoint):
    """
    Farthest-point sampling on GPU.

    Args:
        xyz_tensor: (N, C) float tensor on GPU.  FPS distances are computed
                    on the first 3 columns (XYZ).
        npoint: number of points to sample.
    Returns:
        (npoint, C) float tensor on GPU.
    """
    N, C = xyz_tensor.shape
    device = xyz_tensor.device
    xyz = xyz_tensor[:, :3]  # distances based on XYZ only

    centroids = torch.zeros((npoint, C), device=device)
    distance = torch.ones(N, device=device) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device).item()

    for i in range(npoint):
        centroids[i] = xyz_tensor[farthest]
        c_xyz = xyz[farthest, :].view(1, 3)
        dist = torch.sum((xyz - c_xyz) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance).item()

    return centroids


# ─── SimCLR preprocessing ───────────────────────────────────────────

def preprocess_simclr(stl_path, num_points=NUM_POINTS):
    """
    Load an STL, center, PCA-align, uniform sample, GPU-FPS → (num_points, 3).

    Pipeline mirrors normalization/make_npy.py.
    """
    mesh = trimesh.load(str(stl_path), force="mesh")
    mesh.vertices -= mesh.center_mass

    # PCA alignment
    inertia = mesh.moment_inertia
    _eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    T = np.eye(4)
    T[:3, :3] = eigenvectors
    mesh.apply_transform(np.linalg.inv(T))

    # Uniform sample
    points = mesh.sample(SIMCLR_INITIAL_SAMPLE).astype(np.float32)

    # GPU FPS
    if torch.cuda.is_available():
        pts = torch.from_numpy(points).float().cuda()
        pts = _farthest_point_sample_gpu(pts, num_points)
        return pts.cpu().numpy()
    else:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.farthest_point_down_sample(num_points)
        return np.asarray(pcd.points, dtype=np.float32)


# ─── MAE preprocessing ──────────────────────────────────────────────

def preprocess_mae(stl_path, num_points=NUM_POINTS):
    """
    Load an STL → center, PCA-align, sample, unit-sphere normalize,
    estimate normals, concat XYZ+Normal → GPU-FPS → (num_points, 6).

    Pipeline mirrors normalization/make_mae_npy.py.
    """
    import open3d as o3d

    mesh = trimesh.load(str(stl_path), force="mesh")
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {stl_path}")

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
        raise ValueError(f"NaN in sampled points: {stl_path}")

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

    features = np.hstack([points, normals]).astype(np.float32)  # (N, 6)

    # GPU FPS on 6-channel (distance on XYZ only)
    if torch.cuda.is_available():
        pts = torch.from_numpy(features).float().cuda()
        pts = _farthest_point_sample_gpu(pts, num_points)
        return pts.cpu().numpy()
    else:
        # Fallback: random sample
        idx = np.random.choice(features.shape[0], num_points, replace=False)
        return features[idx]


# ─── DINOv2 preprocessing (multi-view rendering) ────────────────────

def _get_combined_rotation_matrix(base_euler_deg, roll_deg):
    """Rotation matrix from Euler angles + roll (matches render_multiview_final.py)."""
    import open3d as o3d

    rx, ry, rz = np.deg2rad(base_euler_deg)
    R_base = o3d.geometry.get_rotation_matrix_from_xyz((rx, ry, rz))
    roll_rad = np.deg2rad(roll_deg)
    R_roll = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad),  np.cos(roll_rad), 0],
        [0,                 0,                1],
    ])
    return np.matmul(R_roll, R_base)


def _render_view(renderer, mesh, view_cfg, fov_deg):
    """Render a single view using an offscreen renderer."""
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    mesh_copy = copy.deepcopy(mesh)
    R = _get_combined_rotation_matrix(view_cfg["base"], view_cfg["roll"])
    mesh_copy.rotate(R, center=(0, 0, 0))

    renderer.scene.clear_geometry()
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("tooth", mesh_copy, mat)

    bounds = mesh_copy.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_max_bound() - bounds.get_min_bound()
    max_len = np.max(extent)
    dist = (max_len / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0)) * 1.2
    eye = center + np.array([0, 0, dist])
    up = np.array([0, 1, 0])

    renderer.setup_camera(
        fov_deg,
        center.astype(np.float32),
        eye.astype(np.float32),
        up.astype(np.float32),
    )
    img = renderer.render_to_image()
    return np.asarray(img)


def preprocess_dinov2(
    stl_path,
    views_config=None,
    views_order=None,
    render_size=RENDER_IMG_SIZE,
    fov_deg=RENDER_FOV_DEG,
):
    """
    Render an STL from multiple viewpoints → list[PIL.Image].

    Pipeline mirrors render_multiview_final.py + website/routes/search.py.
    """
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    if views_config is None:
        views_config = RENDER_VIEWS_CONFIG
    if views_order is None:
        views_order = DINOV2_VIEWS

    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Empty mesh: {stl_path}")

    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())

    # Colour by normal direction (same as render_multiview_final.py)
    normals = np.asarray(mesh.vertex_normals)
    colors = (normals + 1) / 2.0
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    renderer = rendering.OffscreenRenderer(render_size, render_size)
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])

    images = []
    for view_name in views_order:
        cfg = views_config[view_name]
        img_np = _render_view(renderer, mesh, cfg, fov_deg)
        images.append(Image.fromarray(img_np).convert("RGB"))

    return images
