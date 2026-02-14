import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm

# =================設定=================
# 使用的模型: dinov2_vitb14 (Base model, 768 dim)
# 如果顯卡記憶體夠大，也可以換 dinov2_vitl14 (Large, 1024 dim)
MODEL_NAME = 'dinov2_vitl14' 

# 圖片大小 (DINOv2 預設偏好 14 的倍數)
IMG_SIZE = 224 

# 路徑
current_folder = Path(__file__).parent.resolve()
project_root = current_folder.parent
rendered_dir = project_root / "collecting-data" / "rendered_images"
json_path = project_root / "collecting-data" / "first_scans.json"
output_emb_path = current_folder / "dinov2_embeddings.npy"
output_ids_path = current_folder / "dinov2_filenames.json"

# 視角順序 (確保每次讀取的順序一致)
VIEWS = ["front", "left", "right", "top", "bottom"]
# ======================================

def get_transforms():
    # DINOv2 的標準前處理
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet mean
                    std=[0.229, 0.224, 0.225])  # ImageNet std
    ])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Loading {MODEL_NAME} on {device}...")
    
    # 1. 載入模型 (從 PyTorch Hub)
    model = torch.hub.load('facebookresearch/dinov2', MODEL_NAME)
    model.to(device)
    model.eval()
    
    # 2. 準備資料列表
    if not json_path.exists():
        print("❌ 找不到 first_scans.json")
        return

    with open(json_path, 'r') as f:
        # 這裡的 filenames 格式是 "UUID_scanID.stl"
        # 我們需要轉成 UUID 來找資料夾
        raw_filenames = json.load(f)

    # 轉成 (UUID, folder_path)
    patient_list = []
    for fname in raw_filenames:
        uuid = fname.split('_')[0]
        folder = rendered_dir / uuid
        if folder.exists() and len(list(folder.glob("*.png"))) == 6:
            patient_list.append((fname, folder))
    
    print(f"📂 Found {len(patient_list)} patients with complete views.")
    
    transform = get_transforms()
    all_embeddings = []
    valid_filenames = []

    print("⚙️ Extracting features...")
    
    # 3. 迴圈提取特徵
    with torch.no_grad():
        for filename, folder in tqdm(patient_list):
            view_imgs = []
            
            # 讀取 6 張圖
            try:
                for view in VIEWS:
                    img_path = folder / f"{view}.png"
                    img = Image.open(img_path).convert('RGB')
                    img_t = transform(img)
                    view_imgs.append(img_t)
                
                # Stack 成一個 Batch: (6, 3, 224, 224)
                batch = torch.stack(view_imgs).to(device)
                
                # Inference
                # 輸出 shape: (6, 768)
                outputs = model(batch)
                
                # [關鍵步驟] 特徵融合 (Mean Pooling)
                # 將 6 個視角的特徵取平均，變成一個向量 (1, 768)
                patient_emb = torch.mean(outputs, dim=0)
                
                # 正規化 (L2 Normalize)，方便算 Cosine Similarity
                patient_emb = torch.nn.functional.normalize(patient_emb, p=2, dim=0)
                
                all_embeddings.append(patient_emb.cpu().numpy())
                valid_filenames.append(filename)
                
            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")
                continue

    # 4. 存檔
    final_embs = np.stack(all_embeddings)
    np.save(output_emb_path, final_embs)
    with open(output_ids_path, 'w') as f:
        json.dump(valid_filenames, f)
        
    print("-" * 40)
    print(f"✅ Extraction Complete!")
    print(f"   Shape: {final_embs.shape}") # (N_patients, 768)
    print(f"   Saved to: {output_emb_path}")

if __name__ == "__main__":
    main()