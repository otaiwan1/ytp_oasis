import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import sys
from pathlib import Path

# --- 1. SETUP PATHS ---
current_folder = Path(__file__).parent.resolve()
project_root = current_folder.parent

# Define the neighbor folders
TRAIN_DIR = project_root / "train"
VALIDATION_DIR = project_root / "validation"
DATA_DIR = project_root / "normalization"

# Add 'train' to path to import the model class
sys.path.append(str(TRAIN_DIR))

try:
    from train_oasis import SimCLREncoder
except ImportError:
    print(f"Error importing SimCLREncoder from {TRAIN_DIR}")
    sys.exit()

# --- 2. CONFIGURATION ---
# Base model from 'train'
MODEL_PATH = TRAIN_DIR / "oasis_simclr_edgeconv.pth"
# CSV Report from 'validation'
FEEDBACK_CSV = VALIDATION_DIR / "oasis_validation_report.csv"
# Data from 'normalization'
DATASET_PATH = DATA_DIR / "teeth3ds_dataset.npy"
# Save new model in CURRENT folder ('fine-tuning')
NEW_MODEL_PATH = current_folder / "oasis_finetuned_feedback.pth"

LR = 0.0001
EPOCHS = 10

class FeedbackDataset(Dataset):
    def __init__(self):
        if not FEEDBACK_CSV.exists():
            print(f"Error: Report not found at {FEEDBACK_CSV}")
            print("Please run 'validate_interactive.py' in the validation folder first.")
            sys.exit()
            
        self.feedback = pd.read_csv(FEEDBACK_CSV)
        self.data = np.load(DATASET_PATH)
        
        self.pairs = []
        for _, row in self.feedback.iterrows():
            # Convert grades to Binary Labels (1=Similar, 0=Different)
            # Grade 2 or 3 is "Similar", Grade 0 or 1 is "Different"
            label = 1.0 if row['Dentist_Grade'] >= 2 else 0.0
            self.pairs.append((
                int(row['Query_ID']), 
                int(row['Match_ID']), 
                label
            ))
        print(f"Loaded {len(self.pairs)} graded pairs.")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        q, m, lbl = self.pairs[idx]
        return (
            torch.tensor(self.data[q], dtype=torch.float32),
            torch.tensor(self.data[m], dtype=torch.float32),
            torch.tensor(lbl, dtype=torch.float32)
        )

class FeedbackLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.margin = 1.0
        
    def forward(self, v1, v2, label):
        dist = torch.nn.functional.pairwise_distance(v1, v2)
        # Contrastive Loss Logic for Feedback
        # If label=1 (Similar): Pull together
        loss_similar = label * torch.pow(dist, 2)
        # If label=0 (Different): Push apart (up to margin)
        loss_dissimilar = (1-label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        return torch.mean(loss_similar + loss_dissimilar)

def run_finetuning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = FeedbackDataset()
    if len(dataset) == 0: return

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"Loading Base Model: {MODEL_PATH}")
    model = SimCLREncoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = FeedbackLoss()
    
    print("Starting Fine-Tuning...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for q_pc, m_pc, label in dataloader:
            q_pc = q_pc.transpose(2, 1).to(device)
            m_pc = m_pc.transpose(2, 1).to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            # We use the backbone output (v1, v2), ignoring the projection head
            v1, _ = model(q_pc) 
            v2, _ = model(m_pc)
            
            loss = criterion(v1, v2, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), NEW_MODEL_PATH)
    print(f"Done! New model saved to: {NEW_MODEL_PATH}")

if __name__ == "__main__":
    run_finetuning()