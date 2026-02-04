import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- CONFIGURATION ---
DATASET_FILE = "../normalization/teeth3ds_dataset.npy"  # The file you created in Step 1
VALIDATION_VIEW_FILE = "../validation/validation_views.npy"  # Third view for validation
BATCH_SIZE = 16    # Lower this to 8 or 4 if you run out of GPU memory
EPOCHS = 20
LR = 0.001
TEMPERATURE = 0.1  # The "temperature" for NT-Xent loss
K_NEIGHBORS = 20   # How many neighbors EdgeConv looks at

# --- 1. DATASET WITH AUGMENTATIONS ---
class SimCLRDataset(Dataset):
    def __init__(self, npy_file):
        try:
            self.data = np.load(npy_file)
            print(f"Loaded dataset: {len(self.data)} scans.")
        except FileNotFoundError:
            print(f"ERROR: Could not find {npy_file}. Did you run Step 1?")
            exit()

    def __len__(self):
        return len(self.data)

    def augment(self, point_cloud):
        """
        Create a distorted view of the teeth using Jitter, Flip, Shear, and Rotation.
        """
        pc = point_cloud.copy()
        
        # A. Random Flip (e.g., flip X axis)
        if np.random.random() > 0.5:
            pc[:, 0] = -pc[:, 0]
            
        # B. Random Shear (Stretch)
        shear_matrix = np.eye(3)
        shear_matrix[0, 1] = np.random.uniform(-0.2, 0.2)
        pc = np.dot(pc, shear_matrix.T)

        # C. Random Rotation (Z-axis)
        theta = np.random.uniform(0, 2 * np.pi)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        pc = np.dot(pc, rot_matrix.T)
        
        # D. Jitter (Noise)
        noise = np.random.normal(0, 0.02, pc.shape)
        pc += noise
        
        return pc.astype(np.float32)

    def __getitem__(self, idx):
        # SimCLR needs TWO different views of the SAME patient
        # Third view is reserved for validation
        original = self.data[idx]
        view1 = self.augment(original)
        view2 = self.augment(original)
        view3 = self.augment(original)  # Validation view
        return torch.tensor(view1), torch.tensor(view2), torch.tensor(view3), idx

# --- 2. EDGECONV LAYERS (Manual Implementation) ---
def knn(x, k):
    """Finds k-nearest neighbors using Euclidean distance"""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (Batch, N, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    """Constructs the local graph for EdgeConv"""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
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

# --- 3. THE MODEL (Encoder + Projection Head) ---
class SimCLREncoder(nn.Module):
    def __init__(self, k=K_NEIGHBORS, emb_dim=512):
        super(SimCLREncoder, self).__init__()
        self.k = k
        
        # EdgeConv Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # EdgeConv Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Global Features
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Output Head (The Search Vector)
        self.fc_backbone = nn.Linear(512, emb_dim)
        
        # Projection Head (Only for Training!)
        self.projection_head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32) 
        )

    def forward(self, x):
        # x: [Batch, 3, 4096]
        
        # Layer 1
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0] 
        
        # Layer 2
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        
        # Concatenate & Global Pool
        x_combined = torch.cat((x1, x2), dim=1)
        x_out = self.conv3(x_combined)
        x_out = x_out.max(dim=2)[0]
        
        # Get Vectors
        representation = self.fc_backbone(x_out) # Save this later
        projection = self.projection_head(representation) # Use this now
        
        return representation, projection

# --- 4. NT-Xent LOSS FUNCTION ---
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        features = torch.cat((z_i, z_j), dim=0)
        features = nn.functional.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z_i.device),
            torch.arange(0, batch_size, device=z_i.device)
        ], dim=0)
        
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z_i.device)
        similarity_matrix.masked_fill_(mask, -9e15)
        
        loss = self.criterion(similarity_matrix, labels)
        return loss / (2 * batch_size)

# --- 5. TRAINING LOOP ---
def generate_validation_views(dataset):
    """
    Generate and save the third augmented views for validation.
    These views are generated once before training and saved to disk.
    """
    import os
    os.makedirs(os.path.dirname(VALIDATION_VIEW_FILE), exist_ok=True)
    
    print("Generating validation views (third augmented view for each sample)...")
    validation_views = []
    indices = []
    
    for idx in range(len(dataset.data)):
        view3 = dataset.augment(dataset.data[idx])
        validation_views.append(view3)
        indices.append(idx)
    
    validation_views = np.array(validation_views, dtype=np.float32)
    indices = np.array(indices, dtype=np.int32)
    
    # Save validation views and their original indices
    np.save(VALIDATION_VIEW_FILE, validation_views)
    np.save(VALIDATION_VIEW_FILE.replace('.npy', '_indices.npy'), indices)
    
    print(f"Saved {len(validation_views)} validation views to {VALIDATION_VIEW_FILE}")
    return validation_views

def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Starting Training on device: {device}")
    
    # Setup
    dataset = SimCLRDataset(DATASET_FILE)
    
    # Generate validation views BEFORE training (only once)
    generate_validation_views(dataset)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = SimCLREncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_func = NTXentLoss(temperature=TEMPERATURE)

    model.train()
    print("------------------------------------------------")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (view1, view2, view3, idx) in enumerate(dataloader):
            # Move to GPU and fix shape [Batch, 3, N]
            # Only use view1 and view2 for training, view3 is for validation only
            view1 = view1.transpose(2, 1).to(device)
            view2 = view2.transpose(2, 1).to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass (Get the 32-dim projections)
            _, proj1 = model(view1)
            _, proj2 = model(view2)
            
            # Calculate Loss
            loss = loss_func(proj1, proj2)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"  Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete! Average Loss: {avg_loss:.4f}")
        print("------------------------------------------------")

    # Save the model
    torch.save(model.state_dict(), "oasis_simclr_edgeconv.pth")
    print("DONE! Model saved to 'oasis_simclr_edgeconv.pth'")

if __name__ == "__main__":
    train()