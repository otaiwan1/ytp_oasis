import os
import sys
import json
import torch
import numpy as np
import open3d as o3d
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # [新增] 引入進度條

# --- 設定環境變數 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# 1. 配置路徑
# ==========================================
current_folder = Path(__file__).parent.resolve()
project_root = current_folder.parent

# [設定] STL 檔案的根目錄 (請修改為您的實際路徑)
# 既然 filenames.json 已經是相對路徑，這裡只要設定最外層的資料夾即可
STL_ROOT_DIR = project_root / "raw_data" 

# 其他路徑
MODEL_PATH = current_folder / "best_dental_simclr_multi_gpu.pth"
NORMALIZATION_DIR = project_root / "normalization"
DATASET_PATH = NORMALIZATION_DIR / "teeth3ds_dataset.npy"
FILENAMES_PATH = NORMALIZATION_DIR / "teeth3ds_filenames.json"

# [快取] 檔案路徑
CACHE_PATH = current_folder / "embeddings_cache.npy"

# 加入 import 路徑
sys.path.append(str(current_folder))

# ==========================================
# 2. 核心邏輯
# ==========================================

def load_resources_and_embeddings(device):
    """
    智慧載入：
    1. 如果有快取 (npy)，直接讀取，跳過模型載入 (秒開)。
    2. 如果沒快取，才載入模型跟數據集進行計算，並存檔。
    """
    
    # 1. 載入檔名表 (一定要有，不然找不到 STL)
    if not FILENAMES_PATH.exists():
        print(f"❌ Filenames json not found: {FILENAMES_PATH}")
        sys.exit()
        
    with open(FILENAMES_PATH, 'r') as f:
        filenames = json.load(f)
    print(f"📂 Loaded {len(filenames)} filenames.")

    # 2. 檢查是否有快取
    if CACHE_PATH.exists():
        print(f"🚀 Found cached embeddings at {CACHE_PATH.name}")
        print("   Loading directly (Skipping model inference)...")
        embeddings = np.load(CACHE_PATH)
        print(f"✅ Embeddings loaded. Shape: {embeddings.shape}")
        return embeddings, filenames

    # 3. 無快取：必須載入模型與數據集重新計算
    print("⚠️ No cache found. Starting computation process...")
    
    # A. 匯入模型 Class (只有需要計算時才 import)
    try:
        from train_oasis import SimCLREncoder
    except ImportError as e:
        print(f"❌ Error importing model: {e}")
        sys.exit()

    # B. 載入點雲數據
    print(f"📂 Loading point cloud dataset from {DATASET_PATH}...")
    data = np.load(DATASET_PATH, allow_pickle=True)
    if isinstance(data, list) or data.dtype == 'object':
        data = np.stack(data).astype(np.float32)

    # C. 載入模型權重
    print(f"🧠 Loading model from {MODEL_PATH}...")
    model = SimCLREncoder().to(device)
    
    state_dict = torch.load(MODEL_PATH, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # D. 計算特徵 (加入 tqdm)
    print("⚙️ Computing embeddings...")
    embeddings = []
    batch_size = 32
    
    with torch.no_grad():
        # [修改] 使用 tqdm 包裝 range
        for i in tqdm(range(0, len(data), batch_size), desc="Inference", unit="batch"):
            batch = data[i : i + batch_size]
            batch_tensor = torch.tensor(batch).float().to(device)
            
            # 獲取特徵
            if hasattr(model, 'get_embedding'):
                emb = model.get_embedding(batch_tensor)
            else:
                emb = model.backbone(batch_tensor)
            
            # 正規化
            emb = torch.nn.functional.normalize(emb, dim=1)
            embeddings.append(emb.cpu().numpy())

    final_embeddings = np.concatenate(embeddings, axis=0)
    
    # E. 儲存快取
    print(f"💾 Saving embeddings to {CACHE_PATH}...")
    np.save(CACHE_PATH, final_embeddings)
    
    return final_embeddings, filenames

def load_stl_mesh(idx, filenames):
    """讀取 STL"""
    if idx >= len(filenames):
        return None
        
    filename = filenames[idx]
    
    # 直接使用拼接路徑
    stl_path = STL_ROOT_DIR / filename
            
    if not stl_path.exists():
        # 容錯：嘗試加上 .stl
        if not stl_path.suffix:
            stl_path = stl_path.with_suffix('.stl')
            
    if not stl_path.exists():
        print(f"⚠️ STL file not found: {stl_path}")
        return None

    try:
        # Open3D 讀取
        mesh = o3d.io.read_triangle_mesh(str(stl_path))
        mesh.compute_vertex_normals()
        return mesh
    except Exception as e:
        print(f"⚠️ Error reading STL: {e}")
        return None

def visualize_interactive(query_idx, top_indices, filenames, similarities):
    """Open3D 視覺化 (緊密排列 + 石膏灰色調)"""
    print(f"\n👀 Loading meshes for visualization...")
    
    # 1. 定義旋轉 (讓模型立起來面對鏡頭)
    # 繞 X 軸轉 -90 度
    R_x = np.array([
        [1, 0, 0],
        [0, 0, 1],   # cos(-90) = 0, -sin(-90) = 1
        [0, -1, 0]   # sin(-90) = -1, cos(-90) = 0
    ])
    
    geometries = []
    
    # 2. 設定參數
    # [修改點] 垂直間距：從 80 改為 40 (視模型實際大小而定，可再微調)
    offset_step = 40.0 
    # [修改點] 顏色：統一使用淺灰色 (0.7, 0.7, 0.7)，看起來像石膏模型
    model_color = [0.7, 0.7, 0.7] 
    
    # --- 處理 Query (最上方) ---
    query_mesh = load_stl_mesh(query_idx, filenames)
    if query_mesh is None: 
        print("Query mesh failed to load.")
        return
    
    # 旋轉 & 上色
    query_mesh.rotate(R_x, center=(0,0,0))
    query_mesh.paint_uniform_color(model_color)
    
    # 為了讓細節更明顯，計算 vertex normals
    query_mesh.compute_vertex_normals()
    
    geometries.append(query_mesh)
    print(f"   [Query]: {filenames[query_idx]}")
    
    # --- 處理 Matches (往下緊密排列) ---
    for i, idx in enumerate(top_indices):
        mesh = load_stl_mesh(idx, filenames)
        if mesh:
            mesh.rotate(R_x, center=(0,0,0))
            
            # [修改點] 緊密向下平移
            mesh.translate((0, -(i + 1) * offset_step, 0))
            
            mesh.paint_uniform_color(model_color)
            mesh.compute_vertex_normals()
            
            geometries.append(mesh)
            print(f"   [Rank {i+1}]: {filenames[idx]} (Sim: {similarities[i]:.4f})")
    
    # --- 顯示視窗 ---
    print("✨ Opening Window... (Press 'q' to close)")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Query {query_idx} vs Matches", width=800, height=1000, left=50, top=50)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # 設定視角 (正視圖)
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 1])  # Z 軸朝外
    ctr.set_up([0, 1, 0])     # Y 軸朝上
    # 視角中心往下移一點，因為模型是往下排的
    ctr.set_lookat([0, -offset_step * 1.5, 0]) 
    ctr.set_zoom(0.7)         # 拉近一點
    
    vis.run()
    vis.destroy_window()

# ==========================================
# 3. 主程式
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # 1. 載入資源 (自動處理快取)
    embeddings, filenames = load_resources_and_embeddings(device)
    
    # 2. 互動迴圈
    while True:
        try:
            user_input = input("\n👉 Enter Index (0-1283), 'r' random, or 'q' quit: ")
            
            if user_input.lower() == 'q':
                break
                
            if user_input.lower() == 'r':
                query_idx = np.random.randint(len(embeddings))
                print(f"Random index: {query_idx}")
            elif user_input.isdigit():
                query_idx = int(user_input)
                if query_idx >= len(embeddings):
                    print("Index out of range!")
                    continue
            else:
                continue
                
            # 計算相似度
            query_vec = embeddings[query_idx].reshape(1, -1)
            sims = cosine_similarity(query_vec, embeddings)[0]
            
            # 取 Top 3 (排除自己)
            top_indices = np.argsort(sims)[::-1][1:4]
            top_sims = sims[top_indices]
            
            visualize_interactive(query_idx, top_indices, filenames, top_sims)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break