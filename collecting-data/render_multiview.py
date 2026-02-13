import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

# ==========================================
# 1. 設定
# ==========================================
IMG_SIZE = 512 
FOV_DEG = 60.0 # 視野角度

# 路徑
current_folder = Path(__file__).parent.resolve()
stl_dir = current_folder / "stlFiles"
json_path = current_folder / "first_scans.json"
output_dir = current_folder / "rendered_images"

# [關鍵修正] 視角旋轉設定
# 根據您的回饋：原本的 Bottom 才是 Front
# 我們依此邏輯推導出新的六個視角
ROTATIONS = {
    # 原始正面設定 (Bottom -> Front)
    "front":  [-np.pi/2, 0, 0], 
    "back":   [np.pi/2, 0, 0],
    "top":    [0, 0, 0],
    "bottom": [0, np.pi, 0],

    # [本次修正] 左右視角加入 Z 軸旋轉
    # Right: 原本 [0, np.pi/2, 0] -> 加入 Z+90 (逆時針) -> [0, np.pi/2, np.pi/2]
    "right":  [0, np.pi/2, np.pi/2],

    # Left:  原本 [0, -np.pi/2, 0] -> 加入 Z-90 (順時針) -> [0, -np.pi/2, -np.pi/2]
    "left":   [0, -np.pi/2, -np.pi/2]   
}

def render_scene_auto_fit(renderer, mesh, rotation_euler):
    """
    使用 OffscreenRenderer + 手動計算相機距離 (Math Auto-Fit)
    """
    # 1. 複製並旋轉物體
    mesh_copy = copy.deepcopy(mesh)
    R = mesh_copy.get_rotation_matrix_from_xyz(rotation_euler)
    mesh_copy.rotate(R, center=(0, 0, 0))
    
    # 2. 清除舊場景
    renderer.scene.clear_geometry()
    
    # 3. 設定材質
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit" 
    
    # 4. 加入 Mesh
    renderer.scene.add_geometry("tooth", mesh_copy, mat)
    
    # ==========================================
    # 手動計算完美相機位置
    # ==========================================
    # A. 取得物體中心與大小
    bounds = mesh_copy.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_max_bound() - bounds.get_min_bound()
    max_len = np.max(extent) # 取最長的一邊作為直徑
    
    # B. 計算相機距離
    # 距離 = (半徑) / tan(視角一半) * 1.2 (留邊)
    dist = (max_len / 2.0) / np.tan(np.deg2rad(FOV_DEG / 2.0)) * 1.2
    
    # C. 設定相機參數
    eye = center + np.array([0, 0, dist])
    up = np.array([0, 1, 0])
    
    # D. 套用設定
    renderer.setup_camera(FOV_DEG, 
                          center.astype(np.float32), 
                          eye.astype(np.float32), 
                          up.astype(np.float32))
    
    # 6. 渲染
    img = renderer.render_to_image()
    
    return np.asarray(img)

def main():
    if not json_path.exists():
        print("❌ 請先執行 get_first_scan.py")
        return

    with open(json_path, 'r') as f:
        file_list = json.load(f)

    print(f"🚀 [Corrected Views] 開始渲染 {len(file_list)} 個檔案 (方向已修正)")
    output_dir.mkdir(exist_ok=True)

    # 初始化渲染器
    render = rendering.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
    render.scene.set_background([0.0, 0.0, 0.0, 1.0])

    for filename in tqdm(file_list):
        stl_path = stl_dir / filename
        uuid = filename.split('_')[0]
        
        save_folder = output_dir / uuid
        save_folder.mkdir(exist_ok=True)
        
        # 強制覆蓋舊檔案
        try:
            # 1. 讀取 STL
            mesh = o3d.io.read_triangle_mesh(str(stl_path))
            
            if len(mesh.vertices) == 0:
                print(f"⚠️ Empty: {filename}")
                continue

            # 2. 幾何處理
            mesh.compute_vertex_normals()
            
            # 中心化
            center = mesh.get_center()
            mesh.translate(-center)
            
            # 法向量著色
            normals = np.asarray(mesh.vertex_normals)
            colors = (normals + 1) / 2.0
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            
            # 3. 迴圈渲染 6 視角
            for view_name, rot_euler in ROTATIONS.items():
                img_np = render_scene_auto_fit(render, mesh, rot_euler)
                
                save_path = save_folder / f"{view_name}.png"
                plt.imsave(str(save_path), img_np)
                
        except Exception as e:
            print(f"❌ Error {filename}: {e}")

    print("\n✅ 渲染完成！請再次檢查 front.png 是否為正確的正面。")

if __name__ == "__main__":
    os.environ["EGL_PLATFORM"] = "surfaceless"
    main()