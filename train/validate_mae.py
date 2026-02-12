import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ==========================================
# 1. 設定與路徑
# ==========================================
# 驗證時只需要一張顯卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_folder = Path(__file__).parent.resolve()
project_root = current_folder.parent

# [路徑設定] 必須對應您訓練時的檔案
# 模型權重
MODEL_PATH = current_folder / "best_point_mae_ddp.pth"
# 預處理好的數據 (必須是 6-channel 的那份)
NORMALIZATION_DIR = project_root / "normalization"
DATASET_PATH = NORMALIZATION_DIR / "teeth3ds_mae_dataset.npy"     
FILENAMES_PATH = NORMALIZATION_DIR / "teeth3ds_mae_filenames.json"
# 原始 STL 資料夾 (用來讀取並顯示 3D 模型)
STL_ROOT_DIR = project_root / "collecting-data" / "stlFiles"      

# 特徵快取 (第一次跑完會存起來，第二次秒讀)
CACHE_PATH = current_folder / "mae_embeddings_cache.npy"

# ==========================================
# 2. 載入模型與數據
# ==========================================
# 嘗試從訓練腳本 import 模型定義
try:
    # 確保 python path 包含當前目錄
    sys.path.append(str(current_folder))
    from train_mae_ddp import PointMAE, config
except ImportError:
    print("❌ Error: Cannot import PointMAE from train_mae_ddp.py")
    print("   Please make sure validate_mae.py is in the same folder as train_mae_ddp.py")
    sys.exit(1)

def load_resources():
    # A. 載入檔名表
    if not FILENAMES_PATH.exists():
        print(f"❌ Error: {FILENAMES_PATH} not found.")
        sys.exit(1)
    with open(FILENAMES_PATH, 'r') as f:
        filenames = json.load(f)
    print(f"📂 Loaded {len(filenames)} filenames.")

    # B. 檢查快取 (如果有就直接用，省去 inference 時間)
    if CACHE_PATH.exists():
        print(f"🚀 Loading cached embeddings from {CACHE_PATH.name}...")
        embeddings = np.load(CACHE_PATH)
        return embeddings, filenames

    # C. 無快取 -> 載入模型與數據重新計算
    print("⚠️ No cache found. Computing embeddings from scratch...")
    
    # 載入數據 (N, 4096, 6)
    if not DATASET_PATH.exists():
        print(f"❌ Error: Dataset not found at {DATASET_PATH}")
        sys.exit(1)

    print(f"   Loading dataset: {DATASET_PATH}")
    data = np.load(DATASET_PATH, allow_pickle=True)
    if isinstance(data, list): data = np.stack(data)
    
    # 檢查維度
    in_channels = data.shape[2]
    print(f"   Data Shape: {data.shape}")
    print(f"   Model Input Channels: {in_channels} (Expected 6 for Normals)")
    
    # 初始化模型
    model = PointMAE(config, in_channels=in_channels).to(device)
    
    # [關鍵] 處理 DDP 權重
    # DDP 儲存的權重 key 會變成 "module.encoder...", 我們要移除 "module."
    if not MODEL_PATH.exists():
        print(f"❌ Error: Model checkpoint not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"   Loading weights from {MODEL_PATH}...")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v # 移除 'module.'
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # D. 推論 (Inference)
    embeddings = []
    batch_size = 64 # 推論時 Batch 可以大一點
    
    print("⚙️ Extracting features...")
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i : i + batch_size]
            # 轉 Tensor
            batch_tensor = torch.tensor(batch).float().to(device)
            
            # [關鍵] 使用 get_embedding 獲取特徵 (Encoder Output)
            # 這會跳過 Decoder，直接拿 Encoder 的特徵
            emb = model.get_embedding(batch_tensor)
            
            # Normalize (Cosine Similarity 需要向量長度為 1)
            emb = F.normalize(emb, p=2, dim=1)
            
            embeddings.append(emb.cpu().numpy())
            
    final_embeddings = np.concatenate(embeddings, axis=0)
    
    # 存快取
    np.save(CACHE_PATH, final_embeddings)
    print(f"💾 Saved embeddings to {CACHE_PATH}")
    
    return final_embeddings, filenames

# ==========================================
# 3. 視覺化 (Open3D) - 保持您喜歡的石膏風格
# ==========================================
def load_stl_mesh(idx, filenames):
    if idx >= len(filenames): return None
    
    # 組合路徑
    rel_path = filenames[idx]
    stl_path = STL_ROOT_DIR / rel_path
    
    # 容錯處理：有些檔名可能少了 .stl
    if not stl_path.exists():
        if not stl_path.suffix: stl_path = stl_path.with_suffix('.stl')
    
    if not stl_path.exists():
        print(f"⚠️ STL not found: {stl_path}")
        return None

    try:
        mesh = o3d.io.read_triangle_mesh(str(stl_path))
        # 重新計算法向量以供渲染
        mesh.compute_vertex_normals()
        return mesh
    except Exception as e:
        print(f"Error loading STL: {e}")
        return None

def visualize_interactive(query_idx, top_indices, filenames, similarities):
    print(f"\n👀 Visualizing Case {query_idx}...")
    
    # 旋轉矩陣 (繞 X 軸轉 -90 度，讓牙齒立起來)
    R_x = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    
    geometries = []
    # [設定] 垂直間距與顏色
    offset_step = 40.0 
    model_color = [0.7, 0.7, 0.7] # 石膏灰
    
    # 1. Query Mesh (最上面)
    query_mesh = load_stl_mesh(query_idx, filenames)
    if query_mesh:
        query_mesh.rotate(R_x, center=(0,0,0))
        query_mesh.paint_uniform_color(model_color)
        geometries.append(query_mesh)
        print(f"   [Query]: {filenames[query_idx]}")

    # 2. Top-K Matches (往下排)
    for i, idx in enumerate(top_indices):
        mesh = load_stl_mesh(idx, filenames)
        if mesh:
            mesh.rotate(R_x, center=(0,0,0))
            # 垂直向下平移
            mesh.translate((0, -(i + 1) * offset_step, 0))
            
            mesh.paint_uniform_color(model_color)
            geometries.append(mesh)
            print(f"   [Rank {i+1}]: {filenames[idx]} (Sim: {similarities[i]:.4f})")

    # 3. Open Window
    print("✨ Opening Window... (Press 'q' to close)")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point-MAE Search: Case {query_idx}", width=600, height=800, left=50, top=50)
    
    for geom in geometries:
        vis.add_geometry(geom)
        
    # 設定視角
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 1])   # 正視
    ctr.set_up([0, 1, 0])      # Y軸朝上
    ctr.set_lookat([0, -offset_step, 0]) # 看向中間
    ctr.set_zoom(0.6)          # 拉遠一點以便看到全部
    
    vis.run()
    vis.destroy_window()

# ==========================================
# 4. 主程式迴圈
# ==========================================
if __name__ == "__main__":
    embeddings, filenames = load_resources()
    
    print("-" * 40)
    print(f"✅ Validation Ready! Total Cases: {len(filenames)}")
    print("-" * 40)
    
    while True:
        inp = input("\n👉 Enter Index (0-1283), 'r' for random, or 'q' to quit: ")
        
        if inp.lower() == 'q': 
            break
        
        query_idx = -1
        try:
            if inp.lower() == 'r':
                query_idx = np.random.randint(len(filenames))
            else:
                query_idx = int(inp)
                if query_idx < 0 or query_idx >= len(filenames):
                    print("Index out of range.")
                    continue
            
            # --- 搜尋邏輯 ---
            # 取出 query 向量
            query_vec = embeddings[query_idx].reshape(1, -1)
            
            # 計算與所有庫存向量的 Cosine Similarity
            sims = cosine_similarity(query_vec, embeddings)[0]
            
            # 排序：由大到小
            # [::-1] 反轉
            # [1:4] 取第 2~4 名 (第 1 名通常是自己，Sim=1.0，跳過)
            sorted_indices = np.argsort(sims)[::-1]
            top_indices = sorted_indices[1:4] 
            
            # 視覺化
            visualize_interactive(query_idx, top_indices, filenames, sims[top_indices])
            
        except ValueError:
            print("Invalid input.")
        except Exception as e:
            print(f"Error: {e}")