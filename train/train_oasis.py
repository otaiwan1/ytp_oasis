import os
# --- FIX FOR OMP ERROR #179 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
from pathlib import Path

# --- CONFIGURATION ---
BATCH_SIZE = 32        # Increased from 16
EPOCHS = 200           # Increased from 20 to allow convergence
LR = 1e-3              # Starting Learning Rate
EMBEDDING_DIM = 512
PROJECTION_DIM = 128   # Increased from 32 (Bottleneck Fix)
NUM_POINTS = 4096

# Define Paths
current_folder = Path(__file__).parent.resolve()
project_root = current_folder.parent
DATA_PATH = project_root / "normalization" / "teeth3ds_dataset.npy"
MODEL_SAVE_PATH = current_folder / "oasis_simclr_edgeconv.pth"

# ==========================================
# 1. THE DATASET (With Stronger Augmentation)
# ==========================================
class SimCLRDataset(Dataset):
    def __init__(self, data_path):
        # Load data with pickle allowed (Fix for Ragged Array)
        self.data = np.load(data_path, allow_pickle=True)
        
        # If data is a list of arrays (Ragged), stack it into a Tensor
        if self.data.dtype == 'object' or isinstance(self.data, list):
            print("Optimizing dataset structure for training...")
            self.data = np.stack(self.data).astype(np.float32)
            
    def __len__(self):
        return len(self.data)

    def augment(self, point_cloud):
        """
        Applies random transformations to a single point cloud.
        1. Rotation (Global)
        2. Jitter (Local noise)
        3. Scaling (Size variation) - NEW
        4. Random Drop (Occlusion) - NEW
        """
        pc = point_cloud.copy()

        # A. Random Rotation (Y-axis for dental arches)
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        pc = np.dot(pc, rotation_matrix)

        # B. Random Scaling (NEW: 80% to 120% size)
        scale = np.random.uniform(0.8, 1.2)
        pc = pc * scale

        # C. Random Jitter (Noise)
        noise = np.random.normal(0, 0.02, pc.shape)
        pc += noise
        
        # D. Random Point Drop (NEW: Simulate missing data/different density)
        # Drop 10% of points occasionally
        if np.random.random() > 0.5:
            drop_indices = np.random.choice(len(pc), int(len(pc)*0.1), replace=False)
            pc[drop_indices] = 0 # specific to implementation, or just leave as is since PointNet handles it

        return pc.astype(np.float32)

    def __getitem__(self, idx):
        original_pc = self.data[idx]
        
        # Generate two different views of the SAME patient
        view1 = self.augment(original_pc)
        view2 = self.augment(original_pc)
        
        return torch.tensor(view1), torch.tensor(view2)

# ==========================================
# 2. THE MODEL (EdgeConv + Larger Head)
# ==========================================
class SimCLREncoder(nn.Module):
    def __init__(self, k=20, emb_dim=EMBEDDING_DIM):
        super(SimCLREncoder, self).__init__()
        self.k = k
        
        # --- Feature Extractor (EdgeConv / DGCNN-like) ---
        self.conv1 = nn.Sequential(nn.Conv1d(6, 64, kernel_size=1), nn.BatchNorm1d(64), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1), nn.BatchNorm1d(128), nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1), nn.BatchNorm1d(256), nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(512, emb_dim, kernel_size=1), nn.BatchNorm1d(emb_dim), nn.LeakyReLU())
        
        self.fc_backbone = nn.Sequential(
            nn.Linear(emb_dim * 2, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, emb_dim)
        )

        # --- Projection Head (The Bottleneck Fix) ---
        # We increase the output dim to 128 to capture more geometry info
        self.projection_head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, PROJECTION_DIM) # Changed 32 -> 128
        )

    def knn(self, x, k):
        # (Simple KNN implementation for EdgeConv)
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    def forward(self, x):
        # Expect input: (Batch, Points, 3) -> Convert to (Batch, 3, Points)
        if x.shape[2] == 3:
            x = x.permute(0, 2, 1)
            
        # EdgeConv Layers
        x_f = self.get_graph_feature(x, k=self.k)
        x = self.conv1(x_f)
        x = x.max(dim=-1, keepdim=False)[0]

        x_f = self.get_graph_feature(x, k=self.k)
        x = self.conv2(x_f)
        x = x.max(dim=-1, keepdim=False)[0]
        
        x_f = self.get_graph_feature(x, k=self.k)
        x = self.conv3(x_f)
        x = x.max(dim=-1, keepdim=False)[0]
        
        x_f = self.get_graph_feature(x, k=self.k)
        x = self.conv4(x_f)
        x = x.max(dim=-1, keepdim=False)[0]
        
        # Global Pooling
        x_avg = torch.mean(x, dim=2)
        x_max = torch.max(x, dim=2)[0]
        x = torch.cat([x_avg, x_max], dim=1)
        
        # Embedding
        embedding = self.fc_backbone(x)
        
        # Start of Projection (Only used during training)
        projection = self.projection_head(embedding)
        return projection # Returns 128-dim vector during training

    def get_embedding(self, x):
        # Helper to get the 512-dim vector for Search (Skip projection)
        if x.shape[2] == 3:
            x = x.permute(0, 2, 1)
        # ... (Duplicate forward pass logic without projection) ...
        # For simplicity in this script, you can just call forward() 
        # But correctly, we should separate them. 
        # Hack for now: use forward output for inference if needed, 
        # or copy the layers logic here.
        # Ideally: Split forward into `encoder(x)` and `projector(x)`.
        pass 

# ==========================================
# 3. LOSS FUNCTION (NT-Xent)
# ==========================================
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        
        # Similarity matrix
        z = torch.nn.functional.normalize(z, dim=1)
        sim_matrix = torch.exp(torch.mm(z, z.t()) / self.temperature)
        
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, 0)

        positives = torch.exp(torch.sum(z_i * z_j, dim=-1) / self.temperature)
        positives = torch.cat([positives, positives], dim=0)

        loss = -torch.log(positives / torch.sum(sim_matrix, dim=-1))
        return torch.sum(loss) / (2 * batch_size)

# ==========================================
# 4. TRAIN FUNCTION (Complete & Upgraded)
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Training on {device}...")
    print(f"Config: Epochs={EPOCHS}, Batch={BATCH_SIZE}, Head={PROJECTION_DIM}")

    # 1. Prepare Data
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    dataset = SimCLRDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 2. Prepare Model
    model = SimCLREncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = NTXentLoss()
    
    # --- SCHEDULER (NEW) ---
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 3. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch_idx, (view1, view2) in enumerate(dataloader):
            view1, view2 = view1.to(device), view2.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (get 128-dim projections)
            z1 = model(view1)
            z2 = model(view2)
            
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Step the LR Scheduler
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        # Save occasionally
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved checkpoint to {MODEL_SAVE_PATH}")

    # Final Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Training Complete. Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()