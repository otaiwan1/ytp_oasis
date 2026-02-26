import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from pathlib import Path

# ==========================================
# 1. 極致配置 (Configuration)
# ==========================================
config = {
    # [關鍵] 因為有 4x4090，我們可以開大 Batch
    # Global Batch Size = BATCH_SIZE_PER_GPU * GPU_COUNT
    # 如果用 4 張卡，總 Batch = 32 * 4 = 128 (非常理想)
    "BATCH_SIZE_PER_GPU": 64,  
    
    "EPOCHS": 8000,             # MAE 收斂慢，多卡訓練快，建議跑久一點 (800-1600)
    "LR": 5e-4,                # 基礎學習率 (會隨 Batch Size 自動縮放)
    "WEIGHT_DECAY": 0.05,      
    
    "NUM_POINTS": 4096,        
    "NUM_PATCHES": 128,        # [升級] 切更細，捕捉更多細節 (原本 64)
    "PATCH_POINTS": 32,        
    "MASK_RATIO": 0.75,        # [升級] 提高遮蔽率，防止過擬合，強迫學習結構
    
    "EMBED_DIM": 384,          # Transformer Base 規格
    "DEPTH": 12,               
    "NUM_HEADS": 6,            
    "DECODER_DEPTH": 4,        
    "DECODER_EMBED_DIM": 384,
    
    # [自動偵測] 如果輸入數據是 (N, 4096, 6) 則使用 Normals，否則只用 XYZ
    "USE_NORMALS": True        
}

# 路徑設定
current_folder = Path(__file__).parent.resolve()
project_root = current_folder.parent
DATA_PATH = project_root / "normalization" / "mae_dataset.npy"
MODEL_SAVE_PATH = current_folder / "mae" / "best_point_mae_ddp.pth"

# ==========================================
# 2. DDP 初始化與工具
# ==========================================
def ddp_setup():
    # TorchRun 會自動設定這些環境變數
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    destroy_process_group()

def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0

# ==========================================
# 3. 資料集 (Dataset)
# ==========================================
class PointCloudDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True)
        if isinstance(self.data, list) or self.data.dtype == 'object':
            self.data = np.stack(self.data).astype(np.float32)
        
        # 自動適應輸入維度 (XYZ vs XYZ+Normal)
        self.input_dim = self.data.shape[2] 
        if is_main_process():
            print(f"📂 Dataset loaded. Shape: {self.data.shape}")
            print(f"ℹ️ Input Feature Dimension: {self.input_dim} (3=XYZ, 6=XYZ+Normal)")

    def __len__(self):
        return len(self.data)

    def augment(self, pc):
        # pc: (N, 3) or (N, 6)
        xyz = pc[:, :3]
        normals = pc[:, 3:] if pc.shape[1] > 3 else None

        # Rotation
        theta = np.random.uniform(0, 2 * np.pi)
        scale = np.random.uniform(0.8, 1.2)
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        xyz = np.dot(xyz, Ry) * scale
        
        # 如果有 Normals，也要跟著旋轉 (但不縮放)
        if normals is not None:
            normals = np.dot(normals, Ry)
        
        # Jitter
        noise = np.random.normal(0, 0.002, xyz.shape)
        xyz += noise
        
        if normals is not None:
            return torch.tensor(np.hstack([xyz, normals]).astype(np.float32))
        return torch.tensor(xyz.astype(np.float32))

    def __getitem__(self, idx):
        return self.augment(self.data[idx])

# ==========================================
# 4. Patch Embedding (FPS + KNN) - 支援 6 Channel
# ==========================================
def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    # FPS 只看座標 (前3維)
    xyz = xyz[:, :, :3] 
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    # KNN 只看座標
    sqrdists = square_distance(new_xyz[:, :, :3], xyz[:, :, :3])
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

class PointPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, num_patches=64, patch_points=32, embed_dim=384):
        super().__init__()
        self.num_patches = num_patches
        self.patch_points = patch_points
        self.in_channels = in_channels
        
        # Mini-PointNet: 輸入維度根據是否有 Normals 自動調整
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, embed_dim, 1)
        )

    def forward(self, x):
        # x: (B, N, C_in)
        B, N, C = x.shape
        xyz = x[:, :, :3]
        
        # FPS & KNN
        fps_idx = farthest_point_sample(x, self.num_patches)
        center_pos = index_points(xyz, fps_idx) # Center 只存座標 (B, NP, 3)
        idx = query_ball_point(0, self.patch_points, x, index_points(x, fps_idx))
        patches = index_points(x, idx) # Patches 包含所有特徵 (B, NP, K, C_in)
        
        # Normalize: 每個 Patch 的 XYZ 減去中心點
        patches[:, :, :, :3] = patches[:, :, :, :3] - center_pos.unsqueeze(2)
        # 如果有 Normals (index 3-5)，不需要減中心點，保持原樣
        
        # Feature Extraction
        patches_flat = patches.view(B * self.num_patches, self.patch_points, C)
        patches_flat = patches_flat.transpose(2, 1) 
        embeddings = self.conv(patches_flat) 
        embeddings = torch.max(embeddings, 2)[0] 
        embeddings = embeddings.view(B, self.num_patches, -1)
        
        return embeddings, center_pos

# ==========================================
# 5. Point-MAE 模型 (支援 DDP & Normals)
# ==========================================
class PointMAE(nn.Module):
    def __init__(self, config, in_channels=3):
        super().__init__()
        self.config = config
        self.mask_ratio = config["MASK_RATIO"]
        
        self.patch_embed = PointPatchEmbed(
            in_channels=in_channels,
            num_patches=config["NUM_PATCHES"], 
            patch_points=config["PATCH_POINTS"], 
            embed_dim=config["EMBED_DIM"]
        )
        
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, config["EMBED_DIM"])
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["EMBED_DIM"], nhead=config["NUM_HEADS"], 
            dim_feedforward=config["EMBED_DIM"]*4, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["DEPTH"])
        
        self.decoder_embed = nn.Linear(config["EMBED_DIM"], config["DECODER_EMBED_DIM"])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config["DECODER_EMBED_DIM"]))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, config["DECODER_EMBED_DIM"])
        )
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config["DECODER_EMBED_DIM"], nhead=config["NUM_HEADS"],
            dim_feedforward=config["DECODER_EMBED_DIM"]*4, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=config["DECODER_DEPTH"])
        
        # 預測頭：我們預測的是 XYZ 座標 (3 channels)，不管輸入有沒有 Normals
        self.head = nn.Linear(config["DECODER_EMBED_DIM"], config["PATCH_POINTS"] * 3)

    def forward(self, x):
        B, N, C = x.shape
        patches, centers = self.patch_embed(x)
        pos_emb = self.pos_embed(centers)
        
        # Masking
        num_patches = self.config["NUM_PATCHES"]
        num_mask = int(self.mask_ratio * num_patches)
        rand_idx = torch.rand(B, num_patches, device=x.device).argsort(dim=1)
        mask_idx = rand_idx[:, :num_mask]
        visible_idx = rand_idx[:, num_mask:]
        
        batch_range = torch.arange(B, device=x.device)[:, None]
        visible_patches = patches[batch_range, visible_idx]
        visible_pos_emb = pos_emb[batch_range, visible_idx]
        
        # Encoder
        x_vis = visible_patches + visible_pos_emb
        encoded_vis = self.encoder(x_vis)
        
        # Decoder
        encoded_vis = self.decoder_embed(encoded_vis)
        mask_tokens = self.mask_token.expand(B, num_mask, -1)
        mask_pos_emb = self.decoder_pos_embed(centers[batch_range, mask_idx])
        mask_tokens = mask_tokens + mask_pos_emb
        visible_pos_emb_dec = self.decoder_pos_embed(centers[batch_range, visible_idx])
        encoded_vis = encoded_vis + visible_pos_emb_dec
        x_full = torch.cat([encoded_vis, mask_tokens], dim=1)
        
        decoded = self.decoder(x_full)
        
        # Predict
        mask_decoded = decoded[:, -num_mask:]
        predicted_points = self.head(mask_decoded)
        predicted_points = predicted_points.view(B, num_mask, self.config["PATCH_POINTS"], 3)
        
        # Ground Truth (只取 XYZ 來算 Loss)
        mask_centers = centers[batch_range, mask_idx]
        gt_knn_idx = query_ball_point(0, self.config["PATCH_POINTS"], x, mask_centers)
        gt_points = index_points(x[:, :, :3], gt_knn_idx) # 只取 XYZ
        gt_points_normalized = gt_points - mask_centers.unsqueeze(2)
        
        return predicted_points, gt_points_normalized

    def get_embedding(self, x):
        patches, centers = self.patch_embed(x)
        pos_emb = self.pos_embed(centers)
        x_full = patches + pos_emb
        encoded = self.encoder(x_full)
        global_max = torch.max(encoded, dim=1)[0]
        global_avg = torch.mean(encoded, dim=1)
        return global_max + global_avg

# ==========================================
# 6. DDP 訓練流程
# ==========================================
def chamfer_distance(p1, p2):
    dist1 = square_distance(p1, p2)
    min_dist1, _ = torch.min(dist1, dim=-1)
    loss1 = torch.mean(min_dist1, dim=-1)
    min_dist2, _ = torch.min(dist1, dim=1)
    loss2 = torch.mean(min_dist2, dim=-1)
    return torch.mean(loss1 + loss2)

def main():
    ddp_setup()
    
    # 自動獲取 GPU ID
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    
    # 資料集 (自動偵測是否包含 Normals)
    dataset = PointCloudDataset(DATA_PATH)
    in_channels = dataset.input_dim
    
    # DDP Sampler
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, 
        batch_size=config["BATCH_SIZE_PER_GPU"], 
        shuffle=False, # DDP 下必須設為 False，由 sampler 負責 shuffle
        sampler=sampler,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 模型
    model = PointMAE(config, in_channels=in_channels).to(device)
    # SyncBatchNorm 確保多卡之間的 BatchNorm 同步 (如果有的話)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Wrap DDP
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = optim.AdamW(model.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])
    # 學習率依照總 Batch Size 自動調整
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["EPOCHS"], eta_min=1e-6)
    
    best_loss = float('inf')
    
    if is_main_process():
        print(f"🚀 Starting DDP Training on {torch.cuda.device_count()} GPUs")
        print(f"   Input Channels: {in_channels}")
        print(f"   Mask Ratio: {config['MASK_RATIO']}")
        print(f"   Epochs: {config['EPOCHS']}")
    
    for epoch in range(config["EPOCHS"]):
        sampler.set_epoch(epoch) # 確保每個 epoch 的隨機性不同
        model.train()
        total_loss = 0
        
        for batch_idx, points in enumerate(dataloader):
            points = points.to(device)
            optimizer.zero_grad()
            
            pred, target = model(points)
            
            B, M, K, C = pred.shape
            loss = chamfer_distance(pred.view(B, M*K, C), target.view(B, M*K, C))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        # 彙整所有 GPU 的 Loss 進行打印
        avg_loss = torch.tensor(total_loss / len(dataloader)).to(device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss.item() / dist.get_world_size()
        
        if is_main_process():
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{config['EPOCHS']} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                # 儲存時要存 model.module
                torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
                print(f"  >>> Best Model Saved: {best_loss:.6f}")
                
    cleanup()

if __name__ == "__main__":
    main()