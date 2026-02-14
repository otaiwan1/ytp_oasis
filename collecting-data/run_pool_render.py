import json
import os
import time
from pathlib import Path
from multiprocessing import Pool, set_start_method
from tqdm import tqdm

# =================設定=================
NUM_WORKERS = 8       # 同時跑 8 個
TARGET_GPU = "1"      # 指定 GPU 1
# ======================================

current_folder = Path(__file__).parent.resolve()
source_json = current_folder / "first_scans.json"

# [重要] 從剛剛改好的模組 import 處理函式
from render_multiview_final import process_one_file

def main():
    # 1. 設定 GPU (這會影響所有子行程)
    os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU
    os.environ["EGL_PLATFORM"] = "surfaceless"

    if not source_json.exists():
        print("❌ 找不到 first_scans.json")
        return

    # 2. 讀取清單
    with open(source_json, 'r') as f:
        all_files = json.load(f)
    
    total_files = len(all_files)
    print(f"🚀 準備處理 {total_files} 個檔案 (GPU {TARGET_GPU}, Workers {NUM_WORKERS})...")

    # 3. 使用 Pool 進行平行處理
    # imap_unordered: 誰先做完誰先回傳，最適合搭配 tqdm
    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_one_file, all_files),
            total=total_files,
            unit="file",
            desc="Rendering"
        ))

    # 4. 統計結果
    success_count = sum(results)
    print("-" * 40)
    print(f"✅ 全部完成！")
    print(f"   成功: {success_count}")
    print(f"   失敗: {total_files - success_count}")

if __name__ == "__main__":
    # 設定啟動模式為 spawn (比較穩定，尤其是涉及 CUDA/OpenGL 時)
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    main()