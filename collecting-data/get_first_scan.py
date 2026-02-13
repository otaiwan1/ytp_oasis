import os
import json
from pathlib import Path
from collections import defaultdict

# ==========================================
# 1. 路徑設定
# ==========================================
# 取得目前檔案所在的目錄 -> collecting-data/
current_folder = Path(__file__).parent.resolve()
stl_dir = current_folder / "stlFiles"
output_json = current_folder / "first_scans.json"
output_txt = current_folder / "first_scans.txt"

def get_first_scans():
    if not stl_dir.exists():
        print(f"❌ Error: Directory not found: {stl_dir}")
        return

    print(f"📂 Scanning directory: {stl_dir} ...")
    
    # 用來儲存每個 UUID 對應到的所有掃描檔案
    # 結構: { "uuid_string": [ (scan_id_int, filename_string), ... ] }
    patient_scans = defaultdict(list)
    
    files = [f for f in os.listdir(stl_dir) if f.lower().endswith('.stl')]
    total_files = len(files)
    
    print(f"   Found {total_files} STL files.")

    # ==========================================
    # 2. 解析檔名與分組
    # ==========================================
    for filename in files:
        # 移除副檔名 .stl
        stem = Path(filename).stem
        
        # 解析 UUID 和 scanID
        # 假設格式為: UUID_scanID
        # 我們用 rsplit('_', 1) 從右邊切一次，確保 UUID 裡如果有底線也不會切錯
        try:
            parts = stem.rsplit('_', 1)
            if len(parts) != 2:
                print(f"⚠️ Warning: Skipping file with unexpected format: {filename}")
                continue
                
            uuid = parts[0]
            scan_id_str = parts[1]
            
            # 將 scanID 轉為整數以便比較大小
            scan_id = int(scan_id_str)
            
            patient_scans[uuid].append((scan_id, filename))
            
        except ValueError:
            print(f"⚠️ Warning: Skipping file with non-integer scanID: {filename}")
            continue

    print(f"   Identified {len(patient_scans)} unique patients.")

    # ==========================================
    # 3. 找出每個病人的最早掃描 (Minimum scanID)
    # ==========================================
    first_scan_files = []
    
    for uuid, scan_list in patient_scans.items():
        # 根據 scan_id (tuple 的第 0 個元素) 進行排序 (由小到大)
        scan_list.sort(key=lambda x: x[0])
        
        # 取第一個 (最小的 scanID)
        earliest_scan = scan_list[0]
        filename = earliest_scan[1]
        
        # 加入列表 (只存檔名)
        first_scan_files.append(filename)

    # 排序一下檔名，讓輸出好看一點
    first_scan_files.sort()

    # ==========================================
    # 4. 存檔
    # ==========================================
    # A. 存成 JSON (給之後的程式用)
    with open(output_json, 'w') as f:
        json.dump(first_scan_files, f, indent=4)
        
    # B. 存成 TXT (方便肉眼檢查)
    with open(output_txt, 'w') as f:
        for fname in first_scan_files:
            f.write(f"{fname}\n")

    print("-" * 40)
    print(f"✅ Processing Complete!")
    print(f"   Total unique 'First Scans' found: {len(first_scan_files)}")
    print(f"   JSON saved to: {output_json}")
    print(f"   TXT saved to:  {output_txt}")
    print("-" * 40)

if __name__ == "__main__":
    get_first_scans()