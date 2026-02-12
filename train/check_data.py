import numpy as np
from pathlib import Path

# 設定路徑
current_folder = Path(__file__).parent.resolve()
project_root = current_folder.parent
DATA_PATH = project_root / "normalization" / "teeth3ds_mae_dataset.npy"

def check():
    if not DATA_PATH.exists():
        print(f"❌ 找不到檔案: {DATA_PATH}")
        return

    print(f"📂 載入: {DATA_PATH}")
    data = np.load(DATA_PATH)
    
    print("-" * 30)
    print(f"資料形狀: {data.shape}")
    print(f"資料型態: {data.dtype}")
    
    # 檢查數值範圍
    # 我們只關心前 3 個 channel (x, y, z)
    xyz = data[:, :, :3]
    
    global_max = np.max(xyz)
    global_min = np.min(xyz)
    global_mean = np.mean(xyz)
    
    print(f"XYZ 最大值: {global_max:.4f} (預期接近 1.0)")
    print(f"XYZ 最小值: {global_min:.4f} (預期接近 -1.0)")
    print(f"XYZ 平均值: {global_mean:.4f} (預期接近 0.0)")
    
    # 檢查是否有全 0 的死資料
    zeros_count = 0
    for i in range(len(data)):
        if np.all(data[i] == 0):
            zeros_count += 1
            
    print(f"全零樣本數: {zeros_count} (預期為 0)")
    
    if global_max > 20.0:
        print("\n⚠️ 警告: 數值似乎沒有正規化 (大於 20)！這會導致 Loss 很難下降。")
    elif zeros_count > 0:
        print("\n⚠️ 警告: 資料集中含有全 0 的無效樣本！")
    else:
        print("\n✅ 資料看起來正常 (已正規化)。")

if __name__ == "__main__":
    check()