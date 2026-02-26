import os
os.environ['LP_NUM_THREADS'] = '8'
import open3d as o3d
import numpy as np
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# =================設定=================
# 檔案路徑
current_folder = Path(__file__).parent.resolve()
project_root = current_folder.parent
stl_dir = project_root / "collecting-data" / "stlFiles"

# DINOv2 提取出的特徵與檔名
emb_path = current_folder / "dinov2_embeddings.npy"
ids_path = current_folder / "dinov2_filenames.json"

# 視覺化設定
OFFSET_STEP = 30.0  # 模型之間的垂直間距
COLOR_QUERY = [0.8, 0.8, 0.8]  # Query 顏色 (灰白)
COLOR_MATCH = [0.6, 0.7, 0.9]  # Match 顏色 (淡藍)
# ======================================

def load_stl(filename, color):
    """讀取 STL 並做基本的中心化與上色"""
    stl_path = stl_dir / filename
    if not stl_path.exists():
        print(f"⚠️ 找不到檔案: {filename}")
        return None

    try:
        mesh = o3d.io.read_triangle_mesh(str(stl_path))
        mesh.compute_vertex_normals()
        
        # 中心化
        mesh.translate(-mesh.get_center())
        
        # 統一顏色 (方便觀察幾何形狀，不被法向量顏色干擾)
        mesh.paint_uniform_color(color)
        
        return mesh
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None

def main():
    if not emb_path.exists() or not ids_path.exists():
        print("❌ 找不到 embedding 檔案，請先執行 extract_dinov2.py")
        return

    # 1. 載入特徵資料庫
    print("🚀 Loading DINOv2 embeddings...")
    embeddings = np.load(emb_path)
    with open(ids_path, 'r') as f:
        filenames = json.load(f)
        
    print(f"📚 Database ready: {len(filenames)} patients.")

    while True:
        try:
            inp = input("\n👉 Enter Index (or 'q' to quit, 'r' for random): ")
            if inp.lower() == 'q': break
            
            # 2. 決定 Query Index
            if inp.lower() == 'r':
                query_idx = np.random.randint(len(filenames))
            else:
                query_idx = int(inp)
                if query_idx < 0 or query_idx >= len(filenames):
                    print("Index out of range.")
                    continue

            print(f"Search index: {query_idx}")
            # 3. 搜尋 (Cosine Similarity)
            query_vec = embeddings[query_idx].reshape(1, -1)
            sims = cosine_similarity(query_vec, embeddings)[0]
            
            # 排序 (Top 4, 含自己)
            top_indices = np.argsort(sims)[::-1][:4]

            # 4. 準備 3D 場景
            geometries = []
            print(f"\n🔍 Query Case: {filenames[query_idx]}")
            
            # 為了讓牙齒「立起來」比較好比較，我們繞 X 軸轉 -90 度
            # (這取決於您的原始坐標系，您可以依需求拿掉或修改)
            R_stand_up = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))

            for i, idx in enumerate(top_indices):
                fname = filenames[idx]
                score = sims[idx]
                
                # 判斷是 Query 還是 Match
                is_query = (i == 0)
                color = COLOR_QUERY if is_query else COLOR_MATCH
                
                mesh = load_stl(fname, color)
                if mesh:
                    # 旋轉讓它立起來 (選用)
                    mesh.rotate(R_stand_up, center=(0,0,0))
                    
                    # 往下排列 (Rank 1 在最上面，依序往下)
                    # 這裡 i=0 是 Query (放在最上面)
                    mesh.translate((0, -i * OFFSET_STEP, 0))
                    
                    geometries.append(mesh)
                    
                    rank_str = "Query" if is_query else f"Rank {i}"
                    print(f"   [{rank_str}] Sim: {score:.4f} | {fname}")

            # 5. 開啟視窗
            if geometries:
                print("✨ Opening 3D Window...")
                o3d.visualization.draw_geometries(
                    geometries, 
                    window_name=f"DINOv2 Search: {filenames[query_idx]}",
                    width=800, 
                    height=800,
                    left=50, top=50
                )
            else:
                print("❌ No meshes loaded.")

        except ValueError:
            print("Invalid input.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()