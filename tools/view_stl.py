import open3d as o3d
import json
import os
import numpy as np
from pathlib import Path

# =================設定=================
# 請確保 Windows 上也有同樣的資料夾結構，或是修改這裡的路徑
current_folder = Path(__file__).parent.resolve()
project_root = current_folder.parent
stl_dir = project_root / "collecting-data" / "stlFiles"
json_path = project_root / "collecting-data" / "first_scans.json"
# ======================================

def view_mesh(filename):
    stl_path = stl_dir / filename
    if not stl_path.exists():
        print(f"❌ 找不到檔案: {stl_path}")
        return

    print(f"👀 開啟: {filename}")
    
    # 1. 讀取模型
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    mesh.compute_vertex_normals()
    
    # 2. 中心化 (方便旋轉觀察)
    mesh.translate(-mesh.get_center())

    # 3. 上色 (法向量顏色)
    normals = np.asarray(mesh.vertex_normals)
    colors = (normals + 1) / 2.0
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # 4. 加入座標軸 (Size=20)
    # 紅=X (Right), 綠=Y (Up), 藍=Z (Front)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0])

    print("------------------------------------------------")
    print("操作說明:")
    print("   [滑鼠左鍵] 旋轉")
    print("   [滑鼠滾輪] 縮放")
    print("   [Shift + 左鍵] 平移")
    print("   [座標軸] 紅=X, 綠=Y, 藍=Z")
    print("------------------------------------------------")

    # 5. 開啟視窗
    o3d.visualization.draw_geometries([mesh, axes], 
                                      window_name=f"Checking: {filename}",
                                      width=800, height=800)

def main():
    if not json_path.exists():
        print("❌ 找不到 first_scans.json，請確認路徑")
        return

    with open(json_path, 'r') as f:
        file_list = json.load(f)

    print(f"📂 共有 {len(file_list)} 個檔案")
    
    while True:
        try:
            user_input = input("\n👉 輸入 Index (0 ~ 1283) 查看模型，或 'q' 離開: ")
            if user_input.lower() == 'q':
                break
            
            idx = int(user_input)
            if 0 <= idx < len(file_list):
                view_mesh(file_list[idx])
            else:
                print("❌ Index 超出範圍")
        except ValueError:
            print("❌ 請輸入數字")

if __name__ == "__main__":
    main()