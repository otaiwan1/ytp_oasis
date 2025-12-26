import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- CONFIGURATION ---
DATASET_PATH = "./../normalization/teeth3ds_dataset.npy" # Run Step 1 script on Teeth3DS to get this!
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
EMBEDDING_DIM = 512 # Size of the "fingerprint" vector

# --- 1. THE MODEL (PointNet Encoder) ---
# This acts as the "Eyes" of the system. It takes 4096 points and outputs a 512D vector.
class PointNetEncoder(nn.Module):
    def __init__(self):
        super(PointNetEncoder, self).__init__()
        # Input: (Batch, 3, 4096) -> Output: (Batch, 64, 4096)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Mapping to the final embedding vector (The "Fingerprint")
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, EMBEDDING_DIM)
        self.bn4 = nn.BatchNorm1d(512)

    def forward(self, x):
        # x shape: [Batch, 4096, 3] -> Permute to [Batch, 3, 4096] for Conv1d
        x = x.transpose(2, 1)
        
        # Layer 1-3: Learn features for each point
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        
        # Max Pooling: The "Symmetric Function" discussed in the report 
        # This collapses 4096 point features into 1 global feature
        x = torch.max(x, 2, keepdim=False)[0]
        
        # Fully Connected Layers: Refine the vector
        x = torch.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x) # No activation on final layer (this is the embedding)
        
        # Normalize the vector to length 1 (Critical for Cosine Similarity!)
        return torch.nn.functional.normalize(x, p=2, dim=1)

# --- 2. THE DATASET (SimCLR / Triplet Logic) ---
class DentalDataset(Dataset):
    def __init__(self, npy_file):
        self.data = np.load(npy_file) # Shape (N, 4096, 3)
        print(f"Loaded dataset with {len(self.data)} scans.")

    def __len__(self):
        return len(self.data)

    def augment(self, point_cloud):
        """Randomly rotate and jitter the scan to create a 'Positive' pair."""
        # Jitter (Add noise)
        noise = np.random.normal(0, 0.02, point_cloud.shape)
        aug_pc = point_cloud + noise
        
        # Rotation (Random Z-axis rotation)
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        aug_pc = np.dot(aug_pc, rotation_matrix.T)
        return aug_pc.astype(np.float32)

    def __getitem__(self, idx):
        anchor_pc = self.data[idx]
        
        # The "Positive" is an augmented version of the Anchor (Self-Supervised) [cite: 234]
        positive_pc = self.augment(anchor_pc)
        
        # The "Negative" is a random DIFFERENT patient
        neg_idx = np.random.randint(0, len(self.data))
        while neg_idx == idx: # Ensure it's not the same patient
            neg_idx = np.random.randint(0, len(self.data))
        negative_pc = self.data[neg_idx]

        return torch.tensor(anchor_pc, dtype=torch.float32), \
               torch.tensor(positive_pc, dtype=torch.float32), \
               torch.tensor(negative_pc, dtype=torch.float32)

# --- 3. TRAINING LOOP ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Initialize components
    dataset = DentalDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = PointNetEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Triplet Loss: Pushes Positive closer, Negative further away 
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for anchors, positives, negatives in dataloader:
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: Get vectors for all 3 versions
            anchor_vec = model(anchors)
            pos_vec = model(positives)
            neg_vec = model(negatives)
            
            # Compute Loss
            loss = criterion(anchor_vec, pos_vec, neg_vec)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")

    # Save the trained brain
    torch.save(model.state_dict(), "oasis_model_v1.pth")
    print("Training Complete! Model saved as 'oasis_model_v1.pth'")

if __name__ == "__main__":
    train()