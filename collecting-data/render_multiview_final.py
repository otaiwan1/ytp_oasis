import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import copy
import argparse

# =================設定=================
IMG_SIZE = 512 
FOV_DEG = 60.0 

current_folder = Path(__file__).parent.resolve()
stl_dir = current_folder / "stlFiles"
output_dir = current_folder / "rendered_images"

# 視角設定
VIEWS_CONFIG = {
    "front":  {"base": [-90, 0, 0],  "roll": 0}, 
    "back":   {"base": [90, 0, 0],   "roll": 0},
    "top":    {"base": [0, 0, 0],    "roll": 0},
    "bottom": {"base": [0, 180, 0],  "roll": 0},
    "right":  {"base": [0, 90, 0],   "roll": 90},
    "left":   {"base": [0, -90, 0],  "roll": -90}   
}

# ... (中間的 get_combined_rotation_matrix 和 render_scene_auto_fit 函式完全不動，照舊) ...
# 為了節省篇幅，這裡省略這兩個函式的定義，請保留原本的內容
# ...

def get_combined_rotation_matrix(base_euler_deg, roll_deg):
    # ... (請保留原本內容) ...
    rx, ry, rz = np.deg2rad(base_euler_deg)
    R_base = o3d.geometry.get_rotation_matrix_from_xyz((rx, ry, rz))
    roll_rad = np.deg2rad(roll_deg)
    R_roll = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad),  np.cos(roll_rad), 0],
        [0,                 0,                1]
    ])
    return np.matmul(R_roll, R_base)

def render_scene_auto_fit(renderer, mesh, view_cfg):
    # ... (請保留原本內容) ...
    # 這裡複製您原本 render_scene_auto_fit 的完整程式碼
    mesh_copy = copy.deepcopy(mesh)
    R = get_combined_rotation_matrix(view_cfg["base"], view_cfg["roll"])
    mesh_copy.rotate(R, center=(0, 0, 0))
    renderer.scene.clear_geometry()
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit" 
    renderer.scene.add_geometry("tooth", mesh_copy, mat)
    bounds = mesh_copy.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_max_bound() - bounds.get_min_bound()
    max_len = np.max(extent)
    dist = (max_len / 2.0) / np.tan(np.deg2rad(FOV_DEG / 2.0)) * 1.2
    eye = center + np.array([0, 0, dist])
    up = np.array([0, 1, 0])
    renderer.setup_camera(FOV_DEG, center.astype(np.float32), eye.astype(np.float32), up.astype(np.float32))
    img = renderer.render_to_image()
    return np.asarray(img)


# ==========================================
# [核心修改] 將處理單一檔案的邏輯封裝
# ==========================================
def process_one_file(filename):
    """
    輸入檔名，執行渲染，回傳成功與否 (True/False)
    """
    # 每個 Process 第一次執行時初始化 Renderer
    # 這是一個全域變數技巧，讓每個 Process 只初始化一次 EGL Context
    global renderer
    if 'renderer' not in globals():
        renderer = rendering.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
        renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])

    stl_path = stl_dir / filename
    uuid = filename.split('_')[0]
    save_folder = output_dir / uuid
    save_folder.mkdir(exist_ok=True)
    
    try:
        mesh = o3d.io.read_triangle_mesh(str(stl_path))
        if len(mesh.vertices) == 0: return False

        mesh.compute_vertex_normals()
        mesh.translate(-mesh.get_center())
        normals = np.asarray(mesh.vertex_normals)
        colors = (normals + 1) / 2.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        for view_name, cfg in VIEWS_CONFIG.items():
            img_np = render_scene_auto_fit(renderer, mesh, cfg)
            save_path = save_folder / f"{view_name}.png"
            plt.imsave(str(save_path), img_np)
            
        return True # 成功
    except Exception as e:
        # 在多行程中，print 可能會沒看到，可以考慮寫 log
        return False

# 舊的 Main 保留著，方便單獨測試用
if __name__ == "__main__":
    os.environ["EGL_PLATFORM"] = "surfaceless"
    
    # 這裡只是為了單檔測試，平常不會走到這
    json_path = current_folder / "first_scans.json"
    with open(json_path, 'r') as f:
        file_list = json.load(f)
    
    for f in tqdm(file_list):
        process_one_file(f)