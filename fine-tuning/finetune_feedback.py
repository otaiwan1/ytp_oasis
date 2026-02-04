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
DATA_DIR = project_root / "normalization"

# Add 'train' to path to import the model class
sys.path.append(str(TRAIN_DIR))

try:
    from train_oasis import SimCLREncoder
except ImportError:
    print(f"Error importing SimCLREncoder from {TRAIN_DIR}")
    sys.exit()

# --- 2. CONFIGURATION ---
# Base model from 'train' (original, never modified)
BASE_MODEL_PATH = TRAIN_DIR / "oasis_simclr_edgeconv.pth"
# CSV Report from 'fine-tuning' folder
FEEDBACK_CSV = current_folder / "feedback_report.csv"
# Directory for feedback history
FEEDBACK_HISTORY_DIR = current_folder / "feedback_history"
# Data from 'normalization'
DATASET_PATH = DATA_DIR / "teeth3ds_dataset.npy"
# Directory for storing fine-tuned model history
MODEL_HISTORY_DIR = current_folder / "model_history"
# File to track the latest model
LATEST_MODEL_INFO = MODEL_HISTORY_DIR / "latest.txt"

LR = 0.0001
EPOCHS = 10


def get_latest_model_path():
    """
    Get the path to the latest fine-tuned model.
    If no fine-tuned model exists, return the base model path.
    """
    if LATEST_MODEL_INFO.exists():
        with open(LATEST_MODEL_INFO, 'r') as f:
            latest_path = Path(f.read().strip())
            if latest_path.exists():
                return latest_path
    return BASE_MODEL_PATH


def get_next_version():
    """
    Get the next version number for fine-tuning.
    """
    MODEL_HISTORY_DIR.mkdir(exist_ok=True)
    
    existing_models = list(MODEL_HISTORY_DIR.glob("v*_*.pth"))
    if not existing_models:
        return 1
    
    versions = []
    for model_path in existing_models:
        try:
            # Extract version number from filename like "v001_20260204_153000.pth"
            version_str = model_path.stem.split('_')[0][1:]  # Remove 'v' prefix
            versions.append(int(version_str))
        except (ValueError, IndexError):
            continue
    
    return max(versions) + 1 if versions else 1


def save_model_with_history(model):
    """
    Save model with version number and timestamp.
    Returns the path where the model was saved.
    """
    from datetime import datetime
    
    MODEL_HISTORY_DIR.mkdir(exist_ok=True)
    
    version = get_next_version()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"v{version:03d}_{timestamp}.pth"
    save_path = MODEL_HISTORY_DIR / filename
    
    # Save the model
    torch.save(model.state_dict(), save_path)
    
    # Update the latest.txt pointer
    with open(LATEST_MODEL_INFO, 'w') as f:
        f.write(str(save_path))
    
    return save_path, version

class FeedbackDataset(Dataset):
    def __init__(self, csv_path=None):
        """
        Load feedback data from CSV file(s).
        
        Args:
            csv_path: Path to specific CSV file, or None to use all feedback history
        """
        self.data = np.load(DATASET_PATH)
        self.pairs = []
        
        if csv_path is not None:
            # Use specific CSV file
            csv_files = [Path(csv_path)]
        else:
            # Use all feedback history
            csv_files = []
            if FEEDBACK_CSV.exists():
                csv_files.append(FEEDBACK_CSV)
            if FEEDBACK_HISTORY_DIR.exists():
                csv_files.extend(sorted(FEEDBACK_HISTORY_DIR.glob("*.csv")))
        
        if not csv_files:
            print(f"Error: No feedback reports found.")
            print("Please run 'collect_feedback.py' first.")
            sys.exit()
        
        print(f"Loading feedback from {len(csv_files)} file(s):")
        for csv_file in csv_files:
            if not csv_file.exists():
                print(f"  ⚠️  Skipping (not found): {csv_file}")
                continue
            
            feedback = pd.read_csv(csv_file)
            count = 0
            for _, row in feedback.iterrows():
                # Convert grades to Binary Labels
                # Grade >= 7 is "Similar", Grade <= 4 is "Different"
                # Grades 5-6 are ambiguous and skipped
                grade = row['Dentist_Grade']
                if grade >= 7:
                    label = 1.0
                elif grade <= 4:
                    label = 0.0
                else:
                    continue

                self.pairs.append((
                    int(row['Query_ID']), 
                    int(row['Match_ID']), 
                    label
                ))
                count += 1
            print(f"  ✓ {csv_file.name}: {count} pairs")
        
        print(f"Total: {len(self.pairs)} graded pairs.")

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

def run_finetuning(base_model=None, report_path=None):
    """
    Run fine-tuning with specified base model and feedback report.
    
    Args:
        base_model: Path to base model (None = latest fine-tuned or base model)
        report_path: Path to feedback CSV (None = use all feedback history)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = FeedbackDataset(csv_path=report_path)
    if len(dataset) == 0: return

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Determine which model to use as base
    if base_model is not None:
        model_path = Path(base_model)
        if not model_path.exists():
            # Try to find in model_history
            possible_path = MODEL_HISTORY_DIR / base_model
            if possible_path.exists():
                model_path = possible_path
            else:
                print(f"Error: Model not found: {base_model}")
                return
        print("="*60)
        print(f"Loading SPECIFIED model: {model_path.name}")
    else:
        model_path = get_latest_model_path()
        is_base = (model_path == BASE_MODEL_PATH)
        print("="*60)
        if is_base:
            print(f"Loading BASE model: {model_path}")
            print("(This is the first fine-tuning run)")
        else:
            print(f"Loading LATEST fine-tuned model: {model_path.name}")
    print("="*60)
    
    model = SimCLREncoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = FeedbackLoss()
    
    print("\nStarting Fine-Tuning...")
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
    
    # Save model with version and timestamp
    save_path, version = save_model_with_history(model)
    
    print("\n" + "="*60)
    print(f"✅ Fine-tuning complete!")
    print(f"   Version: v{version:03d}")
    print(f"   Saved to: {save_path}")
    print(f"   This model will be used for next feedback collection.")
    print("="*60)


def list_model_history():
    """List all fine-tuned model versions."""
    if not MODEL_HISTORY_DIR.exists():
        print("No fine-tuning history found.")
        return
    
    models = sorted(MODEL_HISTORY_DIR.glob("v*_*.pth"))
    if not models:
        print("No fine-tuned models found.")
        return
    
    print("\n" + "="*60)
    print("  FINE-TUNING HISTORY")
    print("="*60)
    
    print(f"\n📦 Base Model: {BASE_MODEL_PATH.name}")
    print(f"\n📦 Fine-tuned Models:")
    
    latest = get_latest_model_path()
    
    for model_path in models:
        # Parse filename: v001_20260204_153000.pth
        parts = model_path.stem.split('_')
        version = parts[0]
        date = parts[1] if len(parts) > 1 else "unknown"
        time = parts[2] if len(parts) > 2 else ""
        
        # Format date/time nicely
        if len(date) == 8:
            date_str = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        else:
            date_str = date
        if len(time) == 6:
            time_str = f"{time[:2]}:{time[2:4]}:{time[4:]}"
        else:
            time_str = time
        
        is_latest = "← LATEST" if model_path == latest else ""
        print(f"  {version}  |  {date_str} {time_str}  |  {model_path.name}  {is_latest}")
    
    print("="*60)


def list_feedback_history():
    """List all feedback report files."""
    print("\n" + "="*60)
    print("  FEEDBACK HISTORY")
    print("="*60)
    
    reports = []
    
    # Current feedback_report.csv
    if FEEDBACK_CSV.exists():
        reports.append(FEEDBACK_CSV)
    
    # Historical reports
    if FEEDBACK_HISTORY_DIR.exists():
        reports.extend(sorted(FEEDBACK_HISTORY_DIR.glob("*.csv")))
    
    if not reports:
        print("No feedback reports found.")
        print("="*60)
        return
    
    total_pairs = 0
    for report in reports:
        try:
            df = pd.read_csv(report)
            count = len(df)
            total_pairs += count
            print(f"  📋 {report.name}: {count} entries")
        except Exception as e:
            print(f"  ⚠️  {report.name}: Error reading ({e})")
    
    print(f"\n  Total: {total_pairs} feedback entries across {len(reports)} files")
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune OASIS model with feedback")
    parser.add_argument("--list", action="store_true", help="List all fine-tuned model versions")
    parser.add_argument("--list-feedback", action="store_true", help="List all feedback reports")
    parser.add_argument("--base", type=str, default=None, 
                        help="Base model to fine-tune from (default: latest)")
    parser.add_argument("--report", type=str, default=None,
                        help="Specific feedback report to use (default: all)")
    args = parser.parse_args()
    
    if args.list:
        list_model_history()
    elif args.list_feedback:
        list_feedback_history()
    else:
        run_finetuning(base_model=args.base, report_path=args.report)