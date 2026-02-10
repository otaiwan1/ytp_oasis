import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# --- FIX FOR OMP ERROR (畫圖時常見錯誤) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# 1. SETUP PATHS (參考您提供的架構)
# ==========================================
current_folder = Path(__file__).parent.resolve()
project_root = current_folder.parent

# 設定資料夾路徑
TRAIN_DIR = current_folder  # 假設此 script 與 train_oasis.py 在同一層
DATA_DIR = project_root / "normalization"
MODEL_PATH = current_folder / "best_dental_simclr_multi_gpu.pth"
DATASET_PATH = DATA_DIR / "teeth3ds_dataset.npy"

# 將 Train 資料夾加入系統路徑，以便 import 模型
sys.path.append(str(TRAIN_DIR))

# ==========================================
# 2. IMPORT MODEL
# ==========================================
try:
    from train_oasis import SimCLREncoder
    print("✅ Successfully imported SimCLREncoder from train_oasis.py")
except ImportError as e:
    print(f"❌ Error importing model: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def load_model_and_data(device):
    """載入模型權重與數據集"""
    
    # 1. 載入數據
    if not DATASET_PATH.exists():
        print(f"❌ Data file not found: {DATASET_PATH}")
        sys.exit()
        
    print(f"📂 Loading dataset from {DATASET_PATH}...")
    data = np.load(DATASET_PATH, allow_pickle=True)
    
    # 處理 Ragged Array
    if isinstance(data, list) or data.dtype == 'object':
        data = np.stack(data).astype(np.float32)
    print(f"   Shape: {data.shape}")

    # 2. 載入模型
    if not MODEL_PATH.exists():
        print(f"❌ Model file not found: {MODEL_PATH}")
        sys.exit()
        
    print(f"🧠 Loading model from {MODEL_PATH}...")
    model = SimCLREncoder().to(device)
    
    # [關鍵] 處理多卡訓練的權重 (DataParallel)
    # 如果權重檔包含 'module.' 前綴，需要移除才能載入單卡模型
    state_dict = torch.load(MODEL_PATH, map_location=device)
    new_state_dict = {}
    
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # 移除 'module.'
        else:
            new_state_dict[k] = v
            
    try:
        model.load_state_dict(new_state_dict)
        print("✅ Weights loaded successfully.")
    except RuntimeError as e:
        print(f"❌ Weight mismatch: {e}")
        sys.exit()
        
    model.eval()
    return model, data

def get_all_embeddings(model, data, device, batch_size=32):
    """計算所有資料的特徵向量"""
    print("⚙️ Computing embeddings for all scans...")
    embeddings = []
    total_samples = len(data)
    
    # 批次處理以節省顯存
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch = data[i : i + batch_size]
            # 確保維度正確 (Batch, Points, 3)
            batch_tensor = torch.tensor(batch).float().to(device)
            
            # 使用 get_embedding 獲取 512維向量 (不經過 Projection Head)
            # 注意：如果您的 SimCLREncoder 只有 forward，請確認 forward 是否回傳 512 維
            # 根據我給您的修正版代碼，應該使用 model.get_embedding()
            if hasattr(model, 'get_embedding'):
                emb = model.get_embedding(batch_tensor)
            else:
                # Fallback: 如果沒有 get_embedding，假設 forward 回傳的是 embedding
                # 但通常 forward 回傳的是 projection (128維)，這會影響效果
                print("⚠️ Warning: .get_embedding() not found, using .backbone() directly.")
                emb = model.backbone(batch_tensor)
            
            # [SimCLR 核心] Cosine Similarity 需要正規化向量
            emb = torch.nn.functional.normalize(emb, dim=1)
            embeddings.append(emb.cpu().numpy())
            
            # 進度條
            if (i // batch_size) % 10 == 0:
                sys.stdout.write(f"\r   Processed {min(i + batch_size, total_samples)}/{total_samples}")
                sys.stdout.flush()
                
    print("\n✅ Embeddings computed.")
    return np.concatenate(embeddings, axis=0)

def plot_point_cloud(ax, pc, title, color='blue'):
    """使用 Matplotlib 繪製簡易 3D 點雲"""
    # 降採樣以便快速繪圖 (只畫 2000 點)
    if len(pc) > 2000:
        idx = np.random.choice(len(pc), 2000, replace=False)
        pc = pc[idx]
    
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c=color, alpha=0.5)
    ax.set_title(title, fontsize=10)
    # 隱藏座標軸刻度，專注於形狀
    ax.set_axis_off()
    # 讓比例尺一致 (避免形狀扁掉)
    max_range = np.array([pc[:,0].max()-pc[:,0].min(), pc[:,1].max()-pc[:,1].min(), pc[:,2].max()-pc[:,2].min()]).max() / 2.0
    mid_x = (pc[:,0].max()+pc[:,0].min()) * 0.5
    mid_y = (pc[:,1].max()+pc[:,1].min()) * 0.5
    mid_z = (pc[:,2].max()+pc[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def visualize_top_k(data, embeddings, query_idx, k=3):
    """搜尋並視覺化"""
    query_vec = embeddings[query_idx].reshape(1, -1)
    
    # 計算 Cosine Similarity
    sims = cosine_similarity(query_vec, embeddings)[0]
    
    # 找出 Top K (排除掉自己，雖然 sim=1.0)
    top_indices = np.argsort(sims)[::-1][1:k+1] 
    
    print(f"\n🔍 Query Index: {query_idx}")
    print(f"   Top {k} Matches: {top_indices}")
    print(f"   Similarities: {sims[top_indices]}")

    # --- 繪圖設定 ---
    fig = plt.figure(figsize=(16, 5))
    
    # 1. 畫 Query (左邊)
    ax = fig.add_subplot(1, k+1, 1, projection='3d')
    plot_point_cloud(ax, data[query_idx], "QUERY (Input)", color='red')
    
    # 2. 畫 Neighbors (右邊)
    for i, idx in enumerate(top_indices):
        ax = fig.add_subplot(1, k+1, i+2, projection='3d')
        score = sims[idx]
        plot_point_cloud(ax, data[idx], f"Rank {i+1}\nSim: {score:.4f}", color='green')
        
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 設定裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # 1. 載入模型與數據
    model, data = load_model_and_data(device)
    
    # 2. 計算所有特徵向量
    embeddings = get_all_embeddings(model, data, device)
    
    # 3. 互動式測試 loop
    print("\n✨ Validation Ready! Close the plot window to see the next case.")
    try:
        while True:
            user_input = input("\n👉 Press Enter to see a random case, or type an Index (0-1283), or 'q' to quit: ")
            
            if user_input.lower() == 'q':
                break
                
            if user_input.isdigit():
                query_idx = int(user_input)
                if query_idx >= len(data):
                    print("Index out of range!")
                    continue
            else:
                # 隨機挑選
                query_idx = np.random.randint(len(data))
                
            visualize_top_k(data, embeddings, query_idx, k=4)
            
    except KeyboardInterrupt:
        print("\nExiting...")