import numpy as np
import torch
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# 路徑設定
current_folder = Path(__file__).parent.resolve()
CACHE_PATH = current_folder / "mae_embeddings_cache.npy"

def diagnose():
    if not CACHE_PATH.exists():
        print("❌ 找不到 embedding cache，請先跑一次 validate_mae.py")
        return

    print(f"🔍 讀取: {CACHE_PATH}")
    emb = np.load(CACHE_PATH)
    print(f"   Shape: {emb.shape}") # (1284, 384)
    
    # 1. 檢查是否有 NaN 或 Inf
    if np.isnan(emb).any():
        print("🚨 嚴重警告：Embedding 含有 NaN！(數值爆炸)")
    if np.isinf(emb).any():
        print("🚨 嚴重警告：Embedding 含有 Inf！")

    # 2. 隨機抽兩個向量比對
    v1 = emb[0]
    v2 = emb[100] # 找遠一點的
    
    print("\n--- 向量樣本 (前 10 維) ---")
    print(f"V1: {v1[:10]}")
    print(f"V2: {v2[:10]}")
    
    # 3. 計算差異
    diff = np.linalg.norm(v1 - v2)
    print(f"\n📏 歐式距離差異 (L2 Distance): {diff:.6f}")
    
    sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]
    print(f"🔗 Cosine Similarity: {sim:.10f}")
    
    # 4. 檢查統計分佈
    print("\n--- 統計分佈 ---")
    print(f"Max Value: {np.max(emb):.4f}")
    print(f"Min Value: {np.min(emb):.4f}")
    print(f"Mean: {np.mean(emb):.4f}")
    print(f"Std Dev: {np.std(emb):.4f}")

    # 5. 診斷
    if diff < 1e-6:
        print("\n💀 結論：【完全崩塌】所有向量數值完全一模一樣。")
        print("   原因推測：模型權重可能沒載入成功，或是 BatchNorm 的 running stats 全是 0。")
    elif sim > 0.999:
        print("\n⚠️ 結論：【高度崩塌】向量有微小差異，但方向幾乎重疊。")
        print("   原因推測：某個巨大的 Bias 值主導了向量 (例如 [1000, 1000, ...] + 微小訊號)。")
        print("   這通常是 LayerNorm 或 BatchNorm 在訓練與驗證模式切換時的問題。")
    else:
        print("\n✅ 結論：向量看起來正常，問題可能出在 validate 程式碼的 query 邏輯。")

if __name__ == "__main__":
    diagnose()