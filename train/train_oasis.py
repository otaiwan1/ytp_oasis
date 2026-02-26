import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# [修正] 改回最穩定的引用路徑
from torch.cuda.amp import GradScaler, autocast 
from pathlib import Path

# ==========================================
# 0. GPU SETTING
# ==========================================
# 指定使用 GPU 1, 2, 3
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# Fix for OMP & Memory Fragmentation
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# [修正] 使用新版環境變數名稱，消除警告
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# CONFIGURATION
# ==========================================
config = {
    "BATCH_SIZE": 36,          # 3張卡，總共36，每張卡分12
    "EPOCHS": 300,             
    "LR": 1e-3,                
    "WEIGHT_DECAY": 1e-4,      
    "TEMPERATURE": 0.1,        
    "EMBEDDING_DIM": 512,      
    "PROJECTION_DIM": 128,     
    "NUM_POINTS": 4096,        
    "K_NEIGHBORS": 20,         
    "DROP_RATE": 0.7           
}

# Paths
current_folder = Path(__file__).parent.resolve()
project_root = current_folder.parent
DATA_PATH = project_root / "normalization" / "simclr_dataset.npy"
MODEL_SAVE_PATH = current_folder / "simclr" / "best_dental_simclr_multi_gpu.pth"

# ==========================================
# 1. DATASET
# ==========================================
class SimCLRDataset(Dataset):
    def __init__(self, data_path):
        print(f"Loading dataset from {data_path}...")
        self.data = np.load(data_path, allow_pickle=True)
        if self.data.dtype == 'object' or isinstance(self.data, list):
            self.data = np.stack(self.data).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def random_cuboid_cutout(self, pc):
        center_idx = np.random.randint(0, len(pc))
        center = pc[center_idx]
        dists = np.sum((pc - center)**2, axis=1)
        num_drop = int(len(pc) * np.random.uniform(0.1, 0.2))
        drop_indices = np.argsort(dists)[:num_drop]
        keep_indices = list(set(range(len(pc))) - set(drop_indices))
        if len(keep_indices) == 0: return pc 
        fill_indices = np.random.choice(keep_indices, num_drop, replace=True)
        pc[drop_indices] = pc[fill_indices]
        return pc

    def augment(self, point_cloud):
        pc = point_cloud.copy()
        
        # Rotation
        theta = np.random.uniform(0, 2 * np.pi)
        tilt_x = np.random.uniform(-0.1, 0.1) 
        tilt_z = np.random.uniform(-0.1, 0.1)
        Rx = np.array([[1, 0, 0], [0, np.cos(tilt_x), -np.sin(tilt_x)], [0, np.sin(tilt_x), np.cos(tilt_x)]])
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        Rz = np.array([[np.cos(tilt_z), -np.sin(tilt_z), 0], [np.sin(tilt_z), np.cos(tilt_z), 0], [0, 0, 1]])
        R = np.dot(Ry, np.dot(Rx, Rz))
        pc = np.dot(pc, R)

        # Scaling
        scale = np.random.uniform(0.8, 1.2)
        pc = pc * scale

        # Jitter
        noise = np.random.normal(0, 0.01, pc.shape)
        pc += noise
        
        # Cutout
        if np.random.random() < config["DROP_RATE"]:
            pc = self.random_cuboid_cutout(pc)

        return torch.tensor(pc.astype(np.float32))

    def __getitem__(self, idx):
        original_pc = self.data[idx]
        view1 = self.augment(original_pc)
        view2 = self.augment(original_pc)
        return view1, view2

# ==========================================
# 2. MODEL (EdgeConv)
# ==========================================
class SimCLREncoder(nn.Module):
    def __init__(self, k=config["K_NEIGHBORS"], emb_dim=config["EMBEDDING_DIM"]):
        super(SimCLREncoder, self).__init__()
        self.k = k
        self.emb_dim = emb_dim
        
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), 
                                   nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False), 
                                   nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False), 
                                   nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False), 
                                   nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dim, kernel_size=1, bias=False), 
                                   nn.BatchNorm1d(emb_dim), nn.LeakyReLU(0.2))

        self.projection_head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, config["PROJECTION_DIM"])
        )

    def knn(self, x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx

    def get_graph_feature(self, x, k=20, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = self.knn(x, k=k)
        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous() 
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims) 
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    def backbone(self, x):
        if x.shape[2] == 3:
            x = x.permute(0, 2, 1)
        
        x_f = self.get_graph_feature(x, k=self.k) 
        x = self.conv1(x_f)
        x1 = x.max(dim=-1, keepdim=False)[0] 

        x_f = self.get_graph_feature(x1, k=self.k)
        x = self.conv2(x_f)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x_f = self.get_graph_feature(x2, k=self.k)
        x = self.conv3(x_f)
        x3 = x.max(dim=-1, keepdim=False)[0]
        
        x_f = self.get_graph_feature(x3, k=self.k)
        x = self.conv4(x_f)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x = torch.max(x, 2)[0]
        return x

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        return z

# ==========================================
# 3. LOSS FUNCTION (Fixed for FP16 Overflow)
# ==========================================
class NTXentLoss(nn.Module):
    def __init__(self, temperature=config["TEMPERATURE"]):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        
        # Normalize 避免數值過大
        z_i = torch.nn.functional.normalize(z_i, dim=1)
        z_j = torch.nn.functional.normalize(z_j, dim=1)
        
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        labels = torch.cat([
            torch.arange(batch_size, device=z.device) + batch_size,
            torch.arange(batch_size, device=z.device)
        ], dim=0)
        
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        
        # [關鍵修正] 使用 -1e4 (負一萬) 而不是 -9e15
        # 因為 FP16 的最小值約為 -65504，超過會溢出報錯
        # 對於 exp 函數，exp(-10000) 已經是 0 了，足夠達到屏蔽效果
        sim_matrix = sim_matrix.masked_fill(mask, -1e4)
        
        loss = self.criterion(sim_matrix, labels)
        return loss

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train():
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    dataset = SimCLRDataset(DATA_PATH)
    dataloader = DataLoader(
        dataset, 
        batch_size=config["BATCH_SIZE"], 
        shuffle=True, 
        drop_last=True, 
        num_workers=4 * num_gpus, 
        pin_memory=True
    )
    
    model = SimCLREncoder()
    
    if num_gpus > 1:
        print(f"Enabling DataParallel on {num_gpus} GPUs...")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])
    criterion = NTXentLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # [修正] 恢復使用標準的 GradScaler
    scaler = GradScaler()

    best_loss = float('inf')

    print("Starting training loop...")
    model.train()
    
    for epoch in range(config["EPOCHS"]):
        total_loss = 0
        valid_batches = 0
        
        for batch_idx, (view1, view2) in enumerate(dataloader):
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # [修正] 恢復使用標準的 autocast
            with autocast():
                z1 = model(view1)
                z2 = model(view2)
                loss = criterion(z1, z2)
            
            if not torch.isfinite(loss):
                print(f"!! WARNING: Loss is {loss.item()} at epoch {epoch+1}, batch {batch_idx}. Skipping step.")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            valid_batches += 1
            
        scheduler.step()
        avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{config['EPOCHS']} | Loss: {avg_loss:.4f} | LR: {current_lr:.8f}")
        
        if avg_loss < best_loss and valid_batches > 0:
            best_loss = avg_loss
            if num_gpus > 1:
                torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
            else:
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  >>> New Best Model Saved (Loss: {best_loss:.4f})")

    print(f"✅ Training Complete. Best model at {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()